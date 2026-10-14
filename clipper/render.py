"""Шаг 5. Рендер финального клипа через ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path
from .config import Config
from .face_crop import CropPlan, crop_filter_expr



# --- выбор кодировщика ------------------------------------------------------
# NVENC — отдельный аппаратный блок видеокарты. Он не отнимает вычислительные
# ядра, поэтому рендер одного клипа может идти параллельно с распознаванием
# речи другого видео, не замедляя его. На libx264 рендер грузит процессор и
# оказывается самым долгим этапом всего пайплайна.
#
# Наличие NVENC проверяется один раз при первом обращении: если карты нет
# (например, локальный запуск на ноутбуке), молча используется libx264.

_ENCODER_CACHE: dict | None = None


def _detect_encoder() -> dict:
    """-> {'name': 'h264_nvenc'|'libx264', 'args': [...] }"""
    global _ENCODER_CACHE
    if _ENCODER_CACHE is not None:
        return _ENCODER_CACHE

    import os
    forced = (os.environ.get("CLIPPER_ENCODER") or "").strip().lower()

    def _fallback(reason: str) -> dict:
        print(f"[render] кодировщик: libx264 (процессор) — {reason}")
        return {"name": "libx264", "hw": False}

    if forced == "libx264":
        _ENCODER_CACHE = _fallback("задано CLIPPER_ENCODER=libx264")
        return _ENCODER_CACHE

    try:
        probe = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                               capture_output=True, text=True, timeout=30)
        if "h264_nvenc" not in (probe.stdout or ""):
            _ENCODER_CACHE = _fallback("ffmpeg собран без h264_nvenc")
            return _ENCODER_CACHE

        # Наличие в списке не значит работоспособность: карты может не быть
        # или все сессии NVENC заняты. Проверяем настоящим кодированием.
        test = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
        if test.returncode != 0:
            _ENCODER_CACHE = _fallback("видеокарта недоступна для NVENC")
            return _ENCODER_CACHE
    except Exception as e:  # noqa: BLE001
        _ENCODER_CACHE = _fallback(f"проверка не удалась ({type(e).__name__})")
        return _ENCODER_CACHE

    print("[render] кодировщик: h264_nvenc (видеокарта)")
    _ENCODER_CACHE = {"name": "h264_nvenc", "hw": True}
    return _ENCODER_CACHE


def video_codec_args(cfg) -> list[str]:
    """Аргументы кодировщика для ffmpeg.

    У NVENC нет CRF: качество задаётся через -cq в режиме constqp/vbr.
    Значения подобраны так, чтобы визуально соответствовать crf18 на x264.
    """
    enc = _detect_encoder()
    if enc["hw"]:
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p5",          # p1 быстрее / p7 качественнее
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", str(max(1, min(51, cfg.video_crf + 5))),
            "-b:v", "0",              # чистый режим по качеству, без битрейта
            "-profile:v", "high",
            "-bf", "3",
        ]
    return ["-c:v", "libx264", "-preset", cfg.video_preset,
            "-crf", str(cfg.video_crf)]
def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg упал:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def concat_clips(parts: list[Path], out_path: Path, workdir: Path) -> Path:
    """Склеивает сегменты (закодированные одинаковыми параметрами) без перекодирования."""
    if len(parts) == 1:
        parts[0].replace(out_path)
        return out_path
    lst = workdir / f"concat_{out_path.stem}.txt"
    lst.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in parts),
        encoding="utf-8",
    )
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst),
                "-c", "copy", "-movflags", "+faststart", str(out_path)])
    lst.unlink(missing_ok=True)
    for p in parts:
        p.unlink(missing_ok=True)
    return out_path


def _subtitles_filter(ass_file: Path) -> str:
    ass_escaped = str(ass_file).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    sub = f"subtitles='{ass_escaped}'"
    fonts_dir = Path(__file__).resolve().parent.parent / "assets"
    if any(fonts_dir.glob("*.ttf")) or any(fonts_dir.glob("*.otf")):
        fd = str(fonts_dir).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        sub += f":fontsdir='{fd}'"
    return sub


def _audio_filter(start: float, end: float,
                  fade_in: bool, fade_out: bool) -> str:
    af = ["loudnorm=I=-14:TP=-1.5:LRA=11"]
    if fade_in:
        af.append("afade=t=in:st=0:d=0.05")
    if fade_out:
        af.append(f"afade=t=out:st={max(0.0, end - start - 0.05):.3f}:d=0.05")
    return ",".join(af)


def _render_split(video: Path, start: float, end: float, plan: CropPlan,
                  ass_file: Path | None, out_path: Path, cfg: Config,
                  fade_in: bool, fade_out: bool) -> Path:
    """Сплит-скрин диалога: два человека стеком (каждый занимает пол-экрана)."""
    half_h = cfg.out_height // 2                          # 960
    ww = min(plan.src_w, plan.src_h * cfg.out_width / half_h)  # аспект 9:8
    hh = ww * half_h / cfg.out_width
    ww, hh = int(round(ww)), int(round(hh))
    y = max(0, (plan.src_h - hh) // 2)

    def x_for(center: float) -> int:
        return int(round(max(0, min(plan.src_w - ww, center - ww / 2))))

    xa, xb = x_for(plan.split_xa), x_for(plan.split_xb)
    chain = (
        f"[0:v]split=2[a][b];"
        f"[a]crop={ww}:{hh}:{xa}:{y},scale={cfg.out_width}:{half_h}:flags=lanczos[top];"
        f"[b]crop={ww}:{hh}:{xb}:{y},scale={cfg.out_width}:{half_h}:flags=lanczos[bot];"
        f"[top][bot]vstack=inputs=2[stacked]"
    )
    if ass_file is not None:
        chain += f";[stacked]{_subtitles_filter(ass_file)}[v]"
    else:
        chain += ";[stacked]null[v]"

    run_ffmpeg([
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}",
        "-filter_complex", chain, "-map", "[v]", "-map", "0:a?",
        "-af", _audio_filter(start, end, fade_in, fade_out),
        *video_codec_args(cfg),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", cfg.audio_bitrate,
        "-movflags", "+faststart", str(out_path),
    ])
    return out_path


def _render_wide(video: Path, start: float, end: float, plan: CropPlan,
                 ass_file: Path | None, out_path: Path, cfg: Config,
                 fade_in: bool, fade_out: bool) -> Path:
    """Общий план летербоксом: весь кадр целиком по центру (вписан по ширине),
    сверху/снизу — тот же кадр во весь экран, размытый (blurred pillarbox).

    Используется на дальних/массовых планах (толпа, студия), где выделять
    одного бессмысленно — показываем сцену целиком, но кадр остаётся 9:16."""
    W, H = cfg.out_width, cfg.out_height          # 1080 x 1920
    # опциональная лёгкая обрезка по бокам ПЕРЕД вписыванием: убираем по
    # crop_frac с каждого края, оставшееся крупнее. crop_frac=0 -> весь кадр.
    crop_frac = float(cfg.extra.get("wide_crop_frac", 0.0))
    crop_frac = max(0.0, min(0.35, crop_frac))
    if crop_frac > 0.001:
        kept_w = int(round(plan.src_w * (1 - 2 * crop_frac)))
        kept_w -= kept_w % 2
        x_off = (plan.src_w - kept_w) // 2
        pre_crop = f"crop={kept_w}:{plan.src_h}:{x_off}:0,"
        eff_w = kept_w
    else:
        pre_crop = ""
        eff_w = plan.src_w
    # чёткий слой: (обрезанный) кадр, вписанный по ширине -> высота fg_h
    fg_h = int(round(W * plan.src_h / eff_w))
    fg_h = min(fg_h, H)                           # не выше экрана
    fg_h -= fg_h % 2
    chain = (
        f"[0:v]split=2[bg][fg];"
        # фон: заполняем весь экран, обрезаем излишки, сильно размываем и гасим
        f"[bg]{pre_crop}scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},boxblur=luma_radius=40:luma_power=2,"
        f"eq=brightness=-0.12[bgb];"
        # передний план: (обрезанный) кадр, вписан по ширине
        f"[fg]{pre_crop}scale={W}:{fg_h}:flags=lanczos[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2[comp]"
    )
    if ass_file is not None:
        chain += f";[comp]{_subtitles_filter(ass_file)}[v]"
    else:
        chain += ";[comp]null[v]"

    run_ffmpeg([
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}",
        "-filter_complex", chain, "-map", "[v]", "-map", "0:a?",
        "-af", _audio_filter(start, end, fade_in, fade_out),
        *video_codec_args(cfg),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", cfg.audio_bitrate,
        "-movflags", "+faststart", str(out_path),
    ])
    return out_path


def render_clip(
    video: Path,
    start: float,
    end: float,
    plan: CropPlan,
    ass_file: Path | None,
    out_path: Path,
    cfg: Config,
    fade_in: bool = False,
    fade_out: bool = False,
) -> Path:
    if plan.mode == "split":
        return _render_split(video, start, end, plan, ass_file, out_path,
                             cfg, fade_in, fade_out)
    if plan.mode == "wide":
        return _render_wide(video, start, end, plan, ass_file, out_path,
                            cfg, fade_in, fade_out)
    filters = [
        crop_filter_expr(plan),
        f"scale={cfg.out_width}:{cfg.out_height}:flags=lanczos",
    ]
    if ass_file is not None:
        # экранируем путь для фильтра subtitles
        ass_escaped = str(ass_file).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        sub_filter = f"subtitles='{ass_escaped}'"
        # шрифты из assets/ проекта — не требуем установки Montserrat в систему
        fonts_dir = Path(__file__).resolve().parent.parent / "assets"
        if any(fonts_dir.glob("*.ttf")) or any(fonts_dir.glob("*.otf")):
            fd_escaped = str(fonts_dir).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            sub_filter += f":fontsdir='{fd_escaped}'"
        filters.append(sub_filter)

    args = [
        # -ss ДО -i: ffmpeg прыгает к ближайшему ключевому кадру и декодирует
        # только короткий хвост до нужного кадра — покадрово точно при
        # перекодировании и в десятки раз быстрее, чем декодировать весь
        # префикс часового видео. Тайм-коды сбрасываются в 0 — как раз то,
        # что нужно выражению кропа и субтитрам.
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}",
        "-vf", ",".join(filters),
    ]
    args += ["-af", _audio_filter(start, end, fade_in, fade_out)]
    args += [
        *video_codec_args(cfg),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", cfg.audio_bitrate,
        "-movflags", "+faststart",
        str(out_path),
    ]
    run_ffmpeg(args)
    return out_path
