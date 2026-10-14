"""Разовый перенос данных из users.json / usage.json в базу.

Запуск:
    python migrate_to_db.py            # перенести
    python migrate_to_db.py --dry-run  # показать, что будет сделано

Пароли переносятся как есть: старый формат salt$hash понимается при входе, а
при первом успешном входе хеш тихо пересчитывается в новый формат. То есть
никому не придётся заново задавать пароль.

Прежние JSON-файлы не удаляются — переименовываются в *.migrated, чтобы
осталась возможность откатиться.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import store

ROOT = Path(__file__).resolve().parent
DRY = "--dry-run" in sys.argv


def _iso(ts: str | None, fallback: str) -> str:
    if not ts:
        return fallback
    try:
        d = datetime.fromisoformat(ts)
        return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).isoformat()
    except Exception:  # noqa: BLE001
        return fallback


def migrate_users() -> int:
    src = ROOT / "users.json"
    if not src.exists():
        print("users.json не найден — пропускаю")
        return 0
    data = json.loads(src.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for email, u in (data.get("users") or {}).items():
        email = email.strip().lower()
        if store.get_user_by_email(email):
            print(f"  = {email} уже в базе, пропускаю")
            continue
        if DRY:
            print(f"  + {email} (админ: {bool(u.get('is_admin'))})")
            n += 1
            continue
        uid = store._uid()          # noqa: SLF001 — разовый скрипт
        store.execute(
            """INSERT INTO users (id, email, password_hash, is_admin, disabled,
               created_at, expires_at, password_changed_at, note, quota_minutes,
               max_concurrent, max_video_minutes)
               VALUES (?,?,?,?,0,?,?,?,?,?,?,?)""",
            (uid, email, u.get("password", ""), 1 if u.get("is_admin") else 0,
             _iso(u.get("created"), now), _iso(u.get("expires"), now),
             _iso(u.get("created"), now), u.get("note", ""),
             10 ** 6 if u.get("is_admin") else 600,
             4 if u.get("is_admin") else 1,
             100000 if u.get("is_admin") else 180))
        print(f"  + {email}")
        n += 1
    return n


def migrate_usage() -> int:
    src = ROOT / "usage.json"
    if not src.exists():
        print("usage.json не найден — пропускаю")
        return 0
    data = json.loads(src.read_text(encoding="utf-8"))
    calls = data.get("calls") or []
    if DRY:
        print(f"  + {len(calls)} записей расхода (без привязки к пользователю)")
        return len(calls)
    for c in calls:
        # user_id и job_id неизвестны: прежний учёт их не хранил.
        # Записи остаются в общей статистике как «до перехода».
        store.record_usage(None, None, c.get("provider", ""), c.get("model", ""),
                           c.get("stage", ""), int(c.get("tok_in", 0)),
                           int(c.get("tok_out", 0)), float(c.get("cost", 0)))
    print(f"  + {len(calls)} записей расхода")
    return len(calls)


def main() -> None:
    store.init_db()
    print("=" * 60)
    print("Перенос данных в базу" + (" (пробный прогон)" if DRY else ""))
    print("=" * 60)

    print("\nПользователи:")
    nu = migrate_users()
    print("\nСтатистика расхода:")
    nc = migrate_usage()

    if not DRY:
        for name in ("users.json", "usage.json"):
            f = ROOT / name
            if f.exists():
                f.rename(f.with_suffix(f.suffix + ".migrated"))
        print("\nСтарые файлы переименованы в *.migrated (не удалены).")

    print(f"\nИтог: пользователей {nu}, записей расхода {nc}")
    if not DRY:
        admins = store.count_admins()
        if admins == 0:
            created = store.ensure_admin_exists()
            if created:
                print(f"\nАдминистраторов не было — создан: {created[0]} / {created[1]}")
        print("\nГотово. Проверьте вход, затем можно удалять *.migrated.")


if __name__ == "__main__":
    main()
