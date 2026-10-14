"""Раскладка файлов на диске — в одном месте, чтобы изоляция пользователей
не зависела от того, кто и как склеил путь в очередном обработчике.

Структура:
    data/clips/<user_id>/<job_id>/*.mp4      готовые клипы
    data/work/<user_id>/<job_id>/            временные файлы (чистятся)
    data/logs/<user_id>/<job_id>.log         лог обработки
    data/uploads/<user_id>/<случайное_имя>   загруженные исходники

Идентификатор пользователя — случайный UUID, а не почта: почта не должна
попадать в имена папок, иначе она утечёт через любое сообщение об ошибке,
случайный листинг или ссылку.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("CLIPPER_DATA_DIR") or (ROOT / "data")).resolve()

CLIPS = DATA / "clips"
WORK = DATA / "work"
LOGS = DATA / "logs"
UPLOADS = DATA / "uploads"

_SAFE_ID = re.compile(r"^[0-9a-fA-F-]{8,64}$")


def _safe(part: str, what: str) -> str:
    """Пропускает только идентификаторы нашего формата.

    Все идентификаторы — шестнадцатеричные UUID, поэтому проверка строгая:
    ничего похожего на '..', '/' или имя файла сюда не пройдёт в принципе.
    """
    if not _SAFE_ID.match(part or ""):
        raise ValueError(f"Недопустимый {what}")
    return part


def ensure_dirs() -> None:
    for d in (CLIPS, WORK, LOGS, UPLOADS):
        d.mkdir(parents=True, exist_ok=True)


def user_clips(user_id: str) -> Path:
    return CLIPS / _safe(user_id, "идентификатор пользователя")


def user_uploads(user_id: str) -> Path:
    return UPLOADS / _safe(user_id, "идентификатор пользователя")


def job_dirs(user_id: str, job_id: str) -> tuple[Path, Path]:
    """-> (папка клипов, рабочая папка) для задачи."""
    u = _safe(user_id, "идентификатор пользователя")
    j = _safe(job_id, "идентификатор задачи")
    return CLIPS / u / j, WORK / u / j


def job_log_path(user_id: str, job_id: str) -> Path:
    u = _safe(user_id, "идентификатор пользователя")
    j = _safe(job_id, "идентификатор задачи")
    return LOGS / u / f"{j}.log"


def resolve_clip(user_id: str, relpath: str) -> Path:
    """Путь к клипу пользователя с защитой от выхода за его папку.

    Проверка через is_relative_to, а не через startswith: строковое сравнение
    пропускало бы соседнюю папку с похожим именем.
    """
    base = user_clips(user_id).resolve()
    target = (base / relpath).resolve()
    if not target.is_relative_to(base):
        raise ValueError("Недопустимый путь")
    return target
