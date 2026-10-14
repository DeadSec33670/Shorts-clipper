"""Оркестратор: видео → транскрипт → моменты → клипы.

Два режима:
  run(source, cfg)               — одно видео, все моменты рендерятся.
  run_films(sources, cfg, ...)   — НЕСКОЛЬКО видео: по каждому находим
                                    моменты (виральность считается ДО
                                    рендера), затем рендерим только
                                    глобально лучшие total_clips штук —
                                    остальные кандидаты не тратят время
                                    на рендер.
"""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

from .config import Config
from .decorate import decorate_clips
from .download import obtain_video
from .face_crop import plan_shots
from .moments import find_moments, Moment
from .render import concat_clips, render_clip
from .subtitles import build_ass, clip_words
from .transcribe import transcribe


def _source_key(source: str) -> str:
    """Стабильный ключ видео: ID ролика YouTube или хеш пути/ссылки."""
    import hashlib
    m = re.search(r"(?:[?&]v=|youtu\.be/|/shorts/|/live/)([\w-]{6,})", source)
    if m:
        return m.group(1)
    if not source.startswith(("http://", "https://")):
        stem = re.sub(r"[^\w-]+", "_", Path(source).stem)[:24]
        return f"{stem}_{hashlib.sha1(source.encode()).hexdigest()[:8]}"
    return hashlib.sha1(source.encode()).hexdigest()[:12]


def _slug(text: str, limit: int = 40) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:limit].rstrip("-")


def _speech_intervals(segments, a: float, b: float) -> list:
    """Слитые интервалы, когда в [a, b] реально звучит речь (по Whisper)."""
    out = []
    for s in segments:
        for w in s.words:
            if w.end < a or w.start > b:
                continue
            if out and w.start - out[-1][1] < 0.3:
                out[-1][1] = w.end
            else:
                out.append([w.start, w.end])
    return [(x, y) for x, y in out]


def _render_moment(video: Path, m: Moment, segments: list, index: int,
                   cfg: Config, asd_scorer, decor, out_path: Path) -> dict:
    """Рендерит ОДИН уже отобранный момент (со всеми его сегментами/склейками)
    в out_path. Возвращает запись для manifest.json. Используется и обычным
    run(), и run_films() — рендер не дублируется между режимами."""
    parts_desc = " + ".join(f"{s:.0f}-{e:.0f}с" for s, e in m.segments)
    print(f"  [{index}] {m.title} "
          f"({m.total_sec:.0f}с из {len(m.segments)} сегм.: {parts_desc}, "
          f"виральность {m.virality:.1f})")

    part_files: list = []
    for j, (seg_start, seg_end) in enumerate(m.segments, 1):
        speech = _speech_intervals(segments, seg_start, seg_end)
        shots = plan_shots(video, seg_start, seg_end, cfg,
                           speech_intervals=speech, asd_scorer=asd_scorer)

        for k, (a_rel, b_rel, plan) in enumerate(shots, 1):
            shot_start = seg_start + a_rel
            shot_end = seg_start + b_rel

            ass_file = None
            if cfg.burn_subtitles:
                words = clip_words(segments, shot_start, shot_end)
                if words:
                    show_hook = (j == 1 and k == 1) and not cfg.extra.get("no_hook")
                    ass_file = build_ass(
                        words, cfg,
                        cfg.workdir / f"sub_{index:02d}_{j}_{k}.ass",
                        title=m.title if show_hook else None,
                        position_frac=0.47 if plan.mode == "split" else 0.26,
                        decor=decor)

            part = cfg.workdir / f"part_{index:02d}_{j}_{k}.mp4"
            render_clip(video, shot_start, shot_end, plan, ass_file,
                        part, cfg,
                        fade_in=(len(m.segments) > 1 and j > 1 and k == 1),
                        fade_out=(len(m.segments) > 1 and j < len(m.segments)
                                  and k == len(shots)))
            part_files.append(part)

    concat_clips(part_files, out_path, cfg.workdir)
    return {
        "file": out_path.name, "title": m.title, "hook": m.hook,
        "theme": m.theme, "reason": m.reason, "virality": m.virality,
        "segments": [{"start": s, "end": e} for s, e in m.segments],
    }


def _build_asd_scorer(cfg: Config):
    if not cfg.extra.get("asd"):
        return None
    try:
        from .asd import ASDScorer, ensure_ready
        device = ensure_ready()
        print(f"[asd] Режим точного определения говорящего включён (устройство: {device})")
        return ASDScorer(device)
    except Exception as e:  # noqa: BLE001
        print(f"[asd] {e}")
        print("[asd] Продолжаю в обычном режиме (без ASD)")
        return None


def run(source: str, cfg: Config) -> list[str]:
    _t0 = time.time()
    # у каждого видео — своя папка кеша и своя папка клипов:
    # смена ссылки не подхватит чужое видео/транскрипт, а возврат
    # к прежнему ролику пройдёт мгновенно по его кешу
    key = _source_key(source)
    cfg.workdir = cfg.workdir / key
    cfg.outdir = cfg.outdir / key
    print(f"[cache] Видео '{key}': кеш в {cfg.workdir}, клипы в {cfg.outdir}")
    cfg.ensure_dirs()

    print("=" * 60)
    print("ШАГ 1/5: получение видео")
    video = obtain_video(source, cfg)

    print("=" * 60)
    print("ШАГ 2/5: транскрипция")
    segments = transcribe(video, cfg)
    if not segments:
        raise RuntimeError("Пустой транскрипт — в видео нет распознаваемой речи")

    print("=" * 60)
    print("ШАГ 3/5: поиск виральных моментов")
    moments = find_moments(segments, cfg)
    if not moments:
        raise RuntimeError("LLM не нашла подходящих фрагментов")

    decor_map = decorate_clips(moments, segments, cfg)
    asd_scorer = _build_asd_scorer(cfg)

    print("=" * 60)
    print(f"ШАГ 4-5/5: рендер {len(moments)} клипов")
    results = []
    manifest = []
    for i, m in enumerate(moments, 1):
        # Маркер прогресса для веб-интерфейса: рендер — самый долгий этап,
        # и без отчёта по каждому клипу индикатор надолго замирал бы.
        print(f"[прогресс] клип {i}/{len(moments)}", flush=True)
        slug = _slug(m.title) or f"clip-{i}"
        out_path = cfg.outdir / f"{i:02d}_{slug}.mp4"
        entry = _render_moment(video, m, segments, i, cfg, asd_scorer,
                               decor_map.get(i - 1), out_path)
        results.append(str(out_path))
        manifest.append(entry)

    (cfg.outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        from .usage import record_run
        record_run(source, len(results), time.time() - _t0, cfg.llm_provider)
    except Exception:  # noqa: BLE001
        pass
    print("=" * 60)
    print(f"Готово! {len(results)} клипов в {cfg.outdir}/ (+ manifest.json с описаниями)")
    return results


def run_films(sources: list[str], cfg: Config, total_clips: int,
             candidates_per_film: int = 10) -> list[str]:
    """Несколько видео -> ОДИН отфильтрованный набор лучших клипов.

    По каждому видео находим до candidates_per_film кандидатов (виральность
    считается сразу, до рендера). Затем из ВСЕХ кандидатов со всех видео
    берём total_clips лучших по виральности и рендерим ТОЛЬКО их — остальные
    кандидаты не рендерятся вовсе, это и экономит время по сравнению с
    «отрендерить всё, потом выбросить лишнее»."""
    _t0_films = time.time()
    base_outdir = Path(cfg.outdir)
    base_workdir = Path(cfg.workdir)
    best_dir = base_outdir / "_best_of_films"
    best_dir.mkdir(parents=True, exist_ok=True)

    asd_scorer = _build_asd_scorer(cfg)

    print("=" * 60)
    print(f"РЕЖИМ --films: {len(sources)} видео, кандидатов с каждого — "
          f"{candidates_per_film}, в финале — {total_clips} лучших")

    pool: list[dict] = []
    for fi, source in enumerate(sources, 1):
        print("=" * 60)
        print(f"[films] Видео {fi}/{len(sources)}: {source}")
        cfg_i = copy.deepcopy(cfg)
        key = _source_key(source)
        cfg_i.workdir = base_workdir / key
        cfg_i.outdir = base_outdir / key
        cfg_i.clips_count = candidates_per_film
        cfg_i.ensure_dirs()

        print("  ШАГ 1/3: получение видео")
        try:
            video = obtain_video(source, cfg_i)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ не удалось получить видео '{source}': {e}. Пропускаю.")
            continue

        print("  ШАГ 2/3: транскрипция")
        try:
            segments = transcribe(video, cfg_i)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ транскрипция не удалась: {e}. Пропускаю это видео.")
            continue
        if not segments:
            print("  ⚠ пустой транскрипт (нет распознаваемой речи). Пропускаю.")
            continue

        print("  ШАГ 3/3: поиск виральных моментов")
        try:
            moments = find_moments(segments, cfg_i)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ поиск моментов не удался: {e}. Пропускаю это видео.")
            continue
        if not moments:
            print("  ⚠ подходящих моментов не найдено. Пропускаю.")
            continue

        decor_map = decorate_clips(moments, segments, cfg_i)
        film_label = key if source.startswith(("http://", "https://")) else Path(source).stem

        for mi, m in enumerate(moments):
            pool.append({
                "virality": m.virality, "moment": m, "video": video,
                "cfg": cfg_i, "segments": segments,
                "decor": decor_map.get(mi),
                "film": film_label,
            })
        print(f"  [films] Кандидатов из этого видео: {len(moments)} "
              f"(виральность {min(m.virality for m in moments):.1f}"
              f"–{max(m.virality for m in moments):.1f})")

    if not pool:
        raise RuntimeError("Ни одно из видео не дало подходящих моментов")

    pool.sort(key=lambda p: -p["virality"])
    selected = pool[:total_clips]
    if len(selected) < total_clips:
        print(f"[films] ⚠ кандидатов всего {len(pool)}, меньше запрошенных "
              f"{total_clips} — беру все {len(selected)}")

    print("=" * 60)
    print(f"[films] Отобрано {len(selected)} лучших из {len(pool)} кандидатов. "
          f"Рендерю только их…")
    print("=" * 60)

    manifest = []
    results = []
    for rank, item in enumerate(selected, 1):
        m = item["moment"]
        slug = _slug(m.title) or f"clip-{rank}"
        out_path = best_dir / f"{rank:02d}_{item['virality']:.1f}_{slug}.mp4"
        entry = _render_moment(item["video"], m, item["segments"], rank,
                               item["cfg"], asd_scorer, item["decor"], out_path)
        entry["rank"] = rank
        entry["source_film"] = item["film"]
        results.append(str(out_path))
        manifest.append(entry)

    (best_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        from .usage import record_run
        record_run(f"--films ({len(sources)} видео)", len(results),
                   time.time() - _t0_films, cfg.llm_provider)
    except Exception:  # noqa: BLE001
        pass
    print("=" * 60)
    print(f"Готово! {len(results)} лучших клипов из {len(sources)} видео "
          f"лежат в {best_dir}/ (+ manifest.json)")
    return results
