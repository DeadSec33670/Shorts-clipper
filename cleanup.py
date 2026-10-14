"""Автоочистка диска: сроки хранения и уборка мусора.

ЗАЧЕМ
Клипы, логи и загруженные файлы копятся бесконечно. На арендованном сервере
диск кончается за считанные дни, и когда он кончится, встанет всё — включая
базу. Плюс это требование к персональным данным: чужие лица и голоса не должны
храниться дольше, чем нужно.

Запуск:
    python cleanup.py            # один проход
    python cleanup.py --loop     # postоянно, раз в час (для systemd)
    python cleanup.py --dry-run  # показать, что удалилось бы

Сроки задаются переменными окружения, значения по умолчанию — в RETENTION.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import paths  # noqa: E402
import storage  # noqa: E402
import store  # noqa: E402

DRY = "--dry-run" in sys.argv
LOOP = "--loop" in sys.argv
INTERVAL = int(os.environ.get("CLIPPER_CLEANUP_INTERVAL", "3600"))


def _days(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


RETENTION = {
    # Готовые клипы. Главный расход диска. 30 дней — разумный компромисс;
    # пользователю стоит показывать дату удаления, чтобы он успел скачать.
    "clips": _days("CLIPPER_KEEP_CLIPS_DAYS", 30),
    # Рабочие файлы. Воркер чистит их сам, здесь подбираем хвосты после сбоев.
    "work": _days("CLIPPER_KEEP_WORK_DAYS", 1),
    # Загруженные исходники. Нужны только до конца обработки.
    "uploads": _days("CLIPPER_KEEP_UPLOADS_DAYS", 2),
    # Логи обработки. Нужны для разбора жалоб, но содержат ссылки и имена
    # файлов, то есть персональные данные.
    "logs": _days("CLIPPER_KEEP_LOGS_DAYS", 14),
    # Завершённые задачи в базе. Статистика расхода при этом сохраняется.
    "jobs": _days("CLIPPER_KEEP_JOBS_DAYS", 90),
    # Журнал действий. Держим долго: он нужен при разборе инцидентов.
    "audit": _days("CLIPPER_KEEP_AUDIT_DAYS", 365),
}

MIN_FREE_GB = float(os.environ.get("CLIPPER_MIN_FREE_GB", "20"))


def _older_than(path: Path, days: int) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
        return age > days * 86400
    except OSError:
        return False


def _rm(path: Path, stats: dict, reason: str) -> None:
    try:
        size = _size_of(path)
        if DRY:
            print(f"  [пробно] {reason}: {path.name} ({size / 1e6:.1f} МБ)")
        else:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        stats["files"] += 1
        stats["bytes"] += size
    except Exception as e:  # noqa: BLE001
        print(f"  ! не удалось удалить {path}: {type(e).__name__}: {e}")


def _size_of(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


def _free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return 999.0


# ---------------------------------------------------------------------------

def clean_files(stats: dict) -> None:
    """Удаляет файлы старше срока хранения."""
    # Клипы: удаляем папками задач, а не отдельными файлами — иначе остаются
    # пустые каталоги и осиротевшие manifest.json.
    if paths.CLIPS.exists():
        for user_dir in paths.CLIPS.iterdir():
            if not user_dir.is_dir():
                continue
            for job_dir in user_dir.iterdir():
                if job_dir.is_dir() and _older_than(job_dir, RETENTION["clips"]):
                    _rm(job_dir, stats, "клипы старше срока")
            _rm_if_empty(user_dir)

    for root, key, what in ((paths.WORK, "work", "рабочие файлы"),
                            (paths.UPLOADS, "uploads", "загруженный файл")):
        if not root.exists():
            continue
        for user_dir in root.iterdir():
            if not user_dir.is_dir():
                continue
            for item in user_dir.iterdir():
                if _older_than(item, RETENTION[key]):
                    _rm(item, stats, what)
            _rm_if_empty(user_dir)

    if paths.LOGS.exists():
        for user_dir in paths.LOGS.iterdir():
            if not user_dir.is_dir():
                continue
            for f in user_dir.glob("*.log"):
                if _older_than(f, RETENTION["logs"]):
                    _rm(f, stats, "лог старше срока")
            _rm_if_empty(user_dir)


def _rm_if_empty(d: Path) -> None:
    try:
        if d.is_dir() and not any(d.iterdir()) and not DRY:
            d.rmdir()
    except OSError:
        pass


def clean_orphans(stats: dict) -> None:
    """Удаляет файлы пользователей и задач, которых уже нет в базе.

    Такое остаётся после сбоев или ручного вмешательства в базу. Для чужих
    видео это прямая утечка: файлы есть, а владельца нет.
    """
    known_users = {r["id"] for r in store.query("SELECT id FROM users")}
    for root in (paths.CLIPS, paths.WORK, paths.UPLOADS, paths.LOGS):
        if not root.exists():
            continue
        for user_dir in root.iterdir():
            if user_dir.is_dir() and user_dir.name not in known_users:
                _rm(user_dir, stats, "файлы удалённого пользователя")


def clean_db() -> dict:
    """Убирает старые записи из базы. Записи расхода не трогаем: на них
    держится статистика себестоимости."""
    res = {}
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=RETENTION["jobs"])).isoformat()
    rows = store.query("""SELECT COUNT(*) n FROM jobs WHERE created_at < ?
                          AND status IN ('done','error','canceled')""", (cutoff,))
    res["jobs"] = int(rows[0]["n"]) if rows else 0
    if not DRY and res["jobs"]:
        store.execute("""DELETE FROM jobs WHERE created_at < ?
                         AND status IN ('done','error','canceled')""", (cutoff,))
    res["audit"] = 0 if DRY else store.purge_audit(RETENTION["audit"])
    return res


def emergency_cleanup(stats: dict) -> None:
    """Если места почти нет, удаляем самые старые клипы, не дожидаясь срока.

    Кончившийся диск останавливает сервис целиком, поэтому здесь мы жертвуем
    самыми старыми результатами ради работоспособности.
    """
    free = _free_gb(paths.DATA)
    if free >= MIN_FREE_GB:
        return
    print(f"  ! свободно всего {free:.1f} ГБ (порог {MIN_FREE_GB}) — "
          f"удаляю самые старые клипы")
    job_dirs = []
    if paths.CLIPS.exists():
        for user_dir in paths.CLIPS.iterdir():
            if user_dir.is_dir():
                job_dirs += [d for d in user_dir.iterdir() if d.is_dir()]
    job_dirs.sort(key=lambda d: d.stat().st_mtime if d.exists() else 0)
    for d in job_dirs:
        if _free_gb(paths.DATA) >= MIN_FREE_GB:
            break
        _rm(d, stats, "аварийная очистка")


def run_once() -> None:
    started = time.time()
    print("=" * 60)
    print(f"Очистка {datetime.now().strftime('%Y-%m-%d %H:%M')}"
          + (" (пробный прогон)" if DRY else ""))
    print(f"Свободно на диске: {_free_gb(paths.DATA):.1f} ГБ")

    stats = {"files": 0, "bytes": 0}
    clean_files(stats)

    # В объектном хранилище сроки нужно соблюдать отдельно: локальных копий
    # там нет, и файл, забытый в бакете, будет оплачиваться вечно.
    if storage.is_s3() and not DRY:
        try:
            n = storage.delete_older_than(RETENTION["clips"])
            if n:
                print(f"  из хранилища удалено объектов: {n}")
        except Exception as e:  # noqa: BLE001
            print(f"  ! очистка хранилища не удалась: {type(e).__name__}: {e}")
    clean_orphans(stats)
    emergency_cleanup(stats)
    db = clean_db()

    # Задачи умерших воркеров возвращаем в очередь — здесь же, чтобы не
    # заводить ещё одну фоновую службу.
    stale = store.requeue_stale_jobs()

    print(f"Удалено объектов: {stats['files']}, освобождено "
          f"{stats['bytes'] / 1e9:.2f} ГБ")
    print(f"Записей задач удалено: {db['jobs']}, журнала: {db['audit']}")
    if stale:
        print(f"Возвращено в очередь зависших задач: {stale}")
    print(f"Свободно стало: {_free_gb(paths.DATA):.1f} ГБ "
          f"(за {time.time() - started:.1f} с)")


def main() -> None:
    store.init_db()
    paths.ensure_dirs()
    if not LOOP:
        run_once()
        return
    print(f"Служба очистки запущена, интервал {INTERVAL} с")
    while True:
        try:
            run_once()
        except Exception as e:  # noqa: BLE001 — служба не должна падать
            print(f"! ошибка очистки: {type(e).__name__}: {e}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
