"""Превращает технический вывод обработки в понятный человеку прогресс.

ПОЧЕМУ ЭТО ОТДЕЛЬНЫЙ МОДУЛЬ
Ожидание переносится плохо не из-за длительности, а из-за неизвестности.
Но индикатор обязан быть честным: полоска, ползущая по таймеру и застревающая
на 90%, разрушает доверие ко всем последующим индикаторам. Поэтому здесь
проценты привязаны к реальным событиям в логе, а не ко времени.

Доли этапов — из типового прогона часового видео. Они приблизительны, поэтому
внутри этапа прогресс не выдумывается: он стоит на границе этапа и двигается,
только когда пайплайн действительно сообщил о переходе. Исключение — рендер,
там пайплайн отчитывается по каждому клипу, и движение настоящее.
"""
from __future__ import annotations

import re

# (доля от общего времени, что показать пользователю)
STAGES: list[tuple[str, float, str]] = [
    ("queued",     0.00, "Ждём свободную машину"),
    ("starting",   0.03, "Запускаем обработку"),
    ("download",   0.08, "Скачиваем видео"),
    ("transcribe", 0.30, "Распознаём речь"),
    ("moments",    0.45, "Ищем самые интересные фрагменты"),
    ("render",     0.55, "Собираем клипы"),
    ("done",       1.00, "Готово"),
]
_FRACTION = {name: frac for name, frac, _ in STAGES}
_LABEL = {name: label for name, _, label in STAGES}

_RENDER_RE = re.compile(r"\[прогресс\]\s*клип\s*(\d+)\s*/\s*(\d+)")
_FILMS_RE = re.compile(r"\[films\]\s*Видео\s*(\d+)\s*/\s*(\d+)")


class ProgressTracker:
    """Хранит состояние прогресса одной задачи и обновляет его по строкам лога."""

    def __init__(self) -> None:
        self.stage = "queued"
        self.percent = 0.0
        self.detail = ""
        self.clips_done = 0
        self.clips_total = 0
        self.film = 0
        self.films_total = 0

    # -- разбор строки лога -------------------------------------------------
    def feed(self, line: str) -> bool:
        """-> True, если состояние изменилось (значит, стоит обновить экран)."""
        before = (self.stage, round(self.percent, 1), self.detail)
        low = line.lower()

        if "шаг 1" in low and "получение видео" in low:
            self._set("download")
        elif "шаг 2" in low and "транскрипц" in low:
            self._set("transcribe")
        elif "шаг 3" in low and "виральных" in low:
            self._set("moments")
        elif "шаг 4" in low or "рендер" in low and "клип" in low:
            self._set("render")
        elif low.startswith("готово!") or "=== готово ===" in low:
            self._set("done")
            self.percent = 100.0

        m = _RENDER_RE.search(line)
        if m:
            self.clips_done, self.clips_total = int(m.group(1)), int(m.group(2))
            self._set("render")
            if self.clips_total:
                # Внутри рендера прогресс настоящий: пайплайн отчитывается
                # по каждому готовому клипу.
                base = _FRACTION["render"]
                span = 1.0 - base
                done = (self.clips_done - 1) / self.clips_total
                self.percent = round((base + span * done) * 100, 1)
            self.detail = f"клип {self.clips_done} из {self.clips_total}"

        m = _FILMS_RE.search(line)
        if m:
            self.film, self.films_total = int(m.group(1)), int(m.group(2))
            self.detail = f"видео {self.film} из {self.films_total}"

        if "загружаю модель" in low or "whisper" in low and "cuda" in low:
            self.detail = "готовим модель распознавания"
        elif "[asd]" in low and "включён" in low:
            self.detail = "разбираем, кто говорит"

        return before != (self.stage, round(self.percent, 1), self.detail)

    def _set(self, stage: str) -> None:
        if stage not in _FRACTION:
            return
        # Прогресс не откатывается назад: скачок вниз выглядит как сбой.
        if _FRACTION[stage] * 100 >= self.percent:
            self.stage = stage
            self.percent = round(_FRACTION[stage] * 100, 1)
            if stage != "render":
                self.detail = ""

    # -- вывод для интерфейса ----------------------------------------------
    def snapshot(self, queue_position: int | None = None,
                 eta_seconds: float | None = None) -> dict:
        label = _LABEL.get(self.stage, "Обрабатываем")
        if self.stage == "queued" and queue_position is not None:
            label = ("Ждём свободную машину — вы следующий"
                     if queue_position == 0
                     else f"В очереди: перед вами {queue_position}")
        return {
            "stage": self.stage,
            "percent": round(self.percent, 1),
            "label": label,
            "detail": self.detail,
            "eta": _human_eta(eta_seconds),
        }


def _human_eta(seconds: float | None) -> str:
    """Осторожные формулировки: точное «осталось 4:31» обещает точность,
    которой нет, и каждая ошибка бьёт по доверию."""
    if not seconds or seconds <= 0:
        return ""
    mins = seconds / 60
    if mins < 1:
        return "меньше минуты"
    if mins < 2.5:
        return "около 2 минут"
    if mins < 60:
        return f"около {int(round(mins / 5) * 5) or 5} минут"
    return "больше часа"


def estimate_seconds_left(percent: float, elapsed: float) -> float | None:
    """Оценка по фактической скорости этого прогона, а не по среднему.

    Пока сделано меньше 5%, оценка не показывается: на таких данных она
    ошибается в разы, а увидев «осталось 2 часа», человек просто уйдёт.
    """
    if percent < 5 or elapsed < 20:
        return None
    total = elapsed / (percent / 100.0)
    return max(0.0, total - elapsed)
