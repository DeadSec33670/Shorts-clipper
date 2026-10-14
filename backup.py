"""Резервное копирование базы.

ЗАЧЕМ
В базе аккаунты, квоты, история и статистика. Потеря базы означает, что все
пользователи лишились доступа, а вы — понимания, кто за что платил. Файлы
клипов при этом ценны меньше: они всё равно удаляются через 30 дней.

Запуск:
    python backup.py                 # сделать копию
    python backup.py --loop          # раз в сутки (для systemd)
    python backup.py --list          # показать имеющиеся копии
    python backup.py --restore ФАЙЛ  # восстановить

ВАЖНО ПРО ХРАНЕНИЕ
Копия содержит персональные данные: адреса почты, хеши паролей, историю.
Держите её на отдельном носителе и зашифрованной — копия, лежащая рядом
с базой на том же диске, не спасает ни от отказа диска, ни от взлома сервера.
"""
from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import store  # noqa: E402

ROOT = Path(__file__).resolve().parent
BACKUP_DIR = Path(os.environ.get("CLIPPER_BACKUP_DIR") or (ROOT / "backups"))
KEEP = int(os.environ.get("CLIPPER_BACKUP_KEEP", "14"))
INTERVAL = int(os.environ.get("CLIPPER_BACKUP_INTERVAL", str(24 * 3600)))


def make_backup() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if store.DIALECT.is_pg:
        out = BACKUP_DIR / f"clipper-{stamp}.sql.gz"
        # pg_dump читает параметры подключения из URL
        proc = subprocess.run(["pg_dump", "--no-owner", store.DIALECT.url],
                              capture_output=True, timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(f"pg_dump: {proc.stderr.decode('utf-8', 'replace')[:500]}")
        with gzip.open(out, "wb") as fh:
            fh.write(proc.stdout)
    else:
        src = store.DIALECT.url.replace("sqlite:///", "", 1)
        tmp = BACKUP_DIR / f".tmp-{stamp}.db"
        out = BACKUP_DIR / f"clipper-{stamp}.db.gz"
        # Штатный механизм копирования SQLite: он согласован даже во время
        # записи. Простое копирование файла дало бы битую копию.
        conn = sqlite3.connect(src)
        dst = sqlite3.connect(tmp)
        with dst:
            conn.backup(dst)
        dst.close()
        conn.close()
        with open(tmp, "rb") as fi, gzip.open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        tmp.unlink(missing_ok=True)

    try:
        out.chmod(0o600)          # копия с персональными данными
    except Exception:  # noqa: BLE001
        pass
    return out


def verify(path: Path) -> bool:
    """Проверяет, что копия читается и содержит пользователей.

    Непроверенная копия — это не резервная копия: о её негодности узнают
    в тот момент, когда она единственная и уже нужна.
    """
    if store.DIALECT.is_pg:
        try:
            with gzip.open(path, "rb") as fh:
                head = fh.read(4096).decode("utf-8", "replace")
            return "CREATE TABLE" in head or "PostgreSQL database dump" in head
        except Exception:  # noqa: BLE001
            return False

    tmp = path.with_suffix(".check")
    try:
        with gzip.open(path, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        conn = sqlite3.connect(tmp)
        n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()
        print(f"  проверка: пользователей {n}, целостность: {ok}")
        return ok == "ok"
    except Exception as e:  # noqa: BLE001
        print(f"  проверка не пройдена: {type(e).__name__}: {e}")
        return False
    finally:
        tmp.unlink(missing_ok=True)


def rotate() -> int:
    files = sorted(BACKUP_DIR.glob("clipper-*.gz"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for f in files[KEEP:]:
        f.unlink(missing_ok=True)
        removed += 1
    return removed


def list_backups() -> None:
    files = sorted(BACKUP_DIR.glob("clipper-*.gz"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("Копий нет.")
        return
    print(f"Копии в {BACKUP_DIR}:")
    for f in files:
        when = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {f.name:<32} {f.stat().st_size / 1e6:>7.1f} МБ  {when}")


def restore(path: str) -> None:
    src = Path(path)
    if not src.exists():
        src = BACKUP_DIR / path
    if not src.exists():
        print(f"Файл не найден: {path}")
        sys.exit(1)

    if store.DIALECT.is_pg:
        print("Для PostgreSQL восстанавливайте вручную:")
        print(f"  gunzip -c {src} | psql {store.DIALECT.url}")
        return

    target = Path(store.DIALECT.url.replace("sqlite:///", "", 1))
    print(f"ВНИМАНИЕ: текущая база {target} будет заменена.")
    print("Остановите сайт и воркеры перед восстановлением.")
    if input("Введите «да» для продолжения: ").strip().lower() != "да":
        print("Отменено.")
        return

    if target.exists():
        # Прежнюю базу не удаляем, а отодвигаем: если копия окажется негодной,
        # откатиться будет уже нечем.
        keep = target.with_suffix(f".before-restore-{int(time.time())}")
        shutil.copy2(target, keep)
        print(f"Прежняя база сохранена как {keep.name}")

    with gzip.open(src, "rb") as fi, open(target, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    print(f"Восстановлено из {src.name}. Запускайте службы.")


def run_once() -> None:
    started = time.time()
    print(f"Копия {datetime.now().strftime('%Y-%m-%d %H:%M')}…")
    path = make_backup()
    ok = verify(path)
    if not ok:
        print(f"  ! КОПИЯ НЕГОДНА: {path.name} — разберитесь немедленно")
        return
    removed = rotate()
    print(f"  готово: {path.name} ({path.stat().st_size / 1e6:.1f} МБ, "
          f"{time.time() - started:.1f} с), старых удалено: {removed}")


def main() -> None:
    if "--list" in sys.argv:
        list_backups()
        return
    if "--restore" in sys.argv:
        i = sys.argv.index("--restore")
        if i + 1 >= len(sys.argv):
            print("Укажите файл: python backup.py --restore ФАЙЛ")
            sys.exit(1)
        restore(sys.argv[i + 1])
        return

    store.init_db()
    if "--loop" not in sys.argv:
        run_once()
        return
    print(f"Служба копирования запущена, интервал {INTERVAL} с")
    while True:
        try:
            run_once()
        except Exception as e:  # noqa: BLE001
            print(f"! ошибка копирования: {type(e).__name__}: {e}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
