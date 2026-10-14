"""Шаг 2. Транскрипция с тайм-кодами на уровне слов (faster-whisper)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config
from .render import run_ffmpeg


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]


def transcribe(video: Path, cfg: Config) -> list[Segment]:
    """Возвращает список сегментов; результат кешируется в workdir/transcript.json."""
    cache = cfg.workdir / "transcript.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return [
            Segment(s["start"], s["end"], s["text"],
                    [Word(**w) for w in s["words"]])
            for s in data
        ]

    audio = cfg.workdir / "audio.wav"
    if not audio.exists():
        # 16 кГц моно — оптимально для Whisper и в разы меньше по размеру
        run_ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(audio)])

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError("Установите faster-whisper: pip install faster-whisper") from e

    device = cfg.whisper_device
    compute = "float16" if device == "cuda" else "int8"
    if device == "auto":
        try:
            import torch  # noqa: F401
            import torch.cuda
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
        except ImportError:
            device, compute = "cpu", "int8"

    # Метка запуска на видеокарте. Если процесс упадёт внутри библиотеки
    # CUDA (это происходит на уровне C, и перехватить в Python нельзя),
    # метка останется — и следующий запуск сам перейдёт на процессор,
    # вместо того чтобы падать снова и снова.
    gpu_flag = cfg.workdir / ".gpu_attempt"
    if device == "cuda" and gpu_flag.exists():
        print("[transcribe] Прошлый запуск на видеокарте завершился аварийно "
              "(несовместимые библиотеки CUDA/cuDNN).")
        print("[transcribe] Перехожу на процессор. Будет дольше, но надёжно.")
        print("[transcribe] Починить видеокарту: "
              "pip install -U ctranslate2 nvidia-cudnn-cu12")
        device, compute = "cpu", "int8"
        gpu_flag.unlink(missing_ok=True)
    elif device == "cuda":
        gpu_flag.write_text("1", encoding="utf-8")

    print(f"[transcribe] Whisper {cfg.whisper_model} на {device} ({compute})…")
    try:
        model = WhisperModel(cfg.whisper_model, device=device, compute_type=compute)
    except Exception as e:  # noqa: BLE001
        if device != "cuda":
            raise
        # Библиотеки CUDA/cuDNN несовместимы — на процессоре медленнее,
        # но лучше медленно, чем никак.
        print(f"[transcribe] Видеокарта недоступна ({type(e).__name__}), "
              f"перехожу на процессор. Это будет заметно дольше.")
        print("[transcribe] Починить: pip install -U ctranslate2 "
              "или запуск с --whisper-device cpu")
        device, compute = "cpu", "int8"
        model = WhisperModel(cfg.whisper_model, device=device, compute_type=compute)

    raw_segments, info = model.transcribe(
        str(audio),
        language=cfg.language,
        word_timestamps=True,
        vad_filter=True,  # отрезает тишину — быстрее и чище
    )

    segments: list[Segment] = []
    # Пишем каждый сегмент на диск СРАЗУ, по одной строке JSON.
    # Обычный try/finally здесь не спасает: сбой в библиотеке CUDA убивает
    # процесс на уровне C, и Python до обработки не доходит — всё
    # распознанное пропадает. Построчный файл переживает даже такое.
    partial = cfg.workdir / "transcript.partial.jsonl"
    done = 0
    if partial.exists():
        # Продолжаем с места обрыва предыдущего запуска
        for line in partial.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001 — последняя строка могла оборваться
                continue
            segments.append(Segment(d["start"], d["end"], d["text"],
                                    [Word(**w) for w in d["words"]]))
        done = len(segments)
        if done:
            print(f"[transcribe] Найдено {done} сегментов от прошлого запуска, "
                  f"продолжаю с них")

    with open(partial, "a", encoding="utf-8") as pf:
        for i, s in enumerate(raw_segments):
            if i < done:
                continue                      # эти уже распознаны
            words = [Word(w.start, w.end, w.word.strip())
                     for w in (s.words or []) if w.word.strip()]
            seg = Segment(s.start, s.end, s.text.strip(), words)
            segments.append(seg)
            pf.write(json.dumps(asdict(seg), ensure_ascii=False) + "\n")
            pf.flush()                        # на диск немедленно
            print(f"  [{_ts(s.start)} → {_ts(s.end)}] {s.text.strip()[:80]}")

    if not segments:
        cache.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        raise RuntimeError("В видео не нашлось распознаваемой речи.")

    cache.write_text(
        json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    partial.unlink(missing_ok=True)
    gpu_flag.unlink(missing_ok=True)      # дошли до конца — метка не нужна

    print(f"[transcribe] {len(segments)} сегментов, язык: {info.language}")

    # Освобождаем видеокарту явно. Без этого выгрузка модели происходит
    # позже и непредсказуемо — на несовместимых сборках cuDNN это роняет
    # весь процесс уже ПОСЛЕ успешного распознавания.
    try:
        del model
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass

    return segments


def _ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
