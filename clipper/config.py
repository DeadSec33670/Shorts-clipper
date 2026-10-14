"""Настройки пайплайна. Все параметры можно переопределить через CLI-аргументы."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- Вход/выход ---
    workdir: Path = Path("work")          # временные файлы (скачанное видео, аудио, транскрипт)
    outdir: Path = Path("clips")          # готовые клипы

    # --- Транскрипция (faster-whisper) ---
    whisper_model: str = "large-v3-turbo"  # качество почти как large-v3, но в разы быстрее
    whisper_device: str = "auto"          # auto / cpu / cuda
    language: str | None = None           # None = автоопределение

    # --- Поиск моментов (LLM) ---
    # provider: "gemini" (бесплатно, по умолчанию) / "groq" (бесплатно) /
    #           "openrouter" (бесплатно) / "claude" (платно)
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    clips_count: int = 15                 # сколько клипов хотим получить
    min_clip_sec: float = 20.0
    max_clip_sec: float = 60.0

    # --- Кадрирование ---
    out_width: int = 1080
    out_height: int = 1920                # 9:16
    face_sample_fps: float = 3.0            # выше частота — точнее ловим смену говорящего          # как часто сэмплируем кадры для детекции лица
    crop_smoothing: float = 0.85          # 0..1, сглаживание движения окна кадрирования

    # --- Субтитры ---
    burn_subtitles: bool = True
    sub_font: str = "Montserrat"
    sub_font_size: int = 88
    # ВНИМАНИЕ: ASS хранит цвет как &H00BBGGRR (BGR, не RGB!)
    sub_highlight_color: str = "&H0047D4FF"  # тёплый жёлтый #FFD447
    sub_base_color: str = "&H00FFFFFF"       # обычные слова — белый
    sub_max_chars_per_line: int = 16         # бюджет символов на строку

    # --- Рендер ---
    video_crf: int = 18
    video_preset: str = "fast"   # fast+crf18 ~ medium+crf19 по качеству, но ~35% быстрее
    audio_bitrate: str = "192k"

    extra: dict = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.outdir.mkdir(parents=True, exist_ok=True)
