"""Хранилище данных: пользователи, задачи, расход, журнал действий.

ЗАЧЕМ ЭТОТ МОДУЛЬ
Раньше пользователи и статистика лежали в users.json / usage.json, которые
переписывались целиком без блокировок. При двух одновременных запросах записи
терялись, а оборванная запись превращала файл в мусор — и все аккаунты
исчезали. Здесь всё в транзакционной БД, такой сценарий невозможен.

ВЫБОР БАЗЫ
По умолчанию SQLite в режиме WAL: ноль установки, транзакции, сотни
одновременных читателей. Этого достаточно, потому что узкое место сервиса —
видеокарта, а не база: сколько бы пользователей ни было, одновременно
обрабатывается одна-две задачи, остальные ждут в очереди.

Когда упрётесь в запись (сотни задач в минуту) — переезд на PostgreSQL это
смена CLIPPER_DB_URL на postgresql://... и `pip install psycopg[binary]`.
Схема и запросы уже совместимы, диалектные различия собраны в _Dialect.

ОЧЕРЕДЬ ЗАДАЧ
Отдельный брокер (Redis/RabbitMQ) не нужен: очередь — это таблица jobs.
Воркер атомарно забирает задачу (см. claim_job), поэтому две копии воркера
никогда не возьмут одну и ту же. Меньше движущихся частей — меньше того, что
нужно отдельно настраивать, бэкапить и защищать.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "clipper.db"

# PBKDF2 — не лучший вариант, но единственный в стандартной библиотеке.
# Argon2id заметно устойчивее к перебору на видеокартах: если поставите
# argon2-cffi, модуль подхватит его автоматически (см. hash_password).
_PBKDF2_ROUNDS = 480_000        # ориентир OWASP для PBKDF2-SHA256

try:
    from argon2 import PasswordHasher            # type: ignore
    from argon2.exceptions import VerifyMismatchError  # type: ignore
    _ARGON = PasswordHasher()
except Exception:  # noqa: BLE001 — библиотека необязательна
    _ARGON = None


# ---------------------------------------------------------------------------
# Подключение
# ---------------------------------------------------------------------------

class _Dialect:
    """Различия между SQLite и PostgreSQL, собранные в одном месте."""

    def __init__(self, url: str):
        self.url = url
        self.is_pg = url.startswith(("postgres://", "postgresql://"))
        self.ph = "%s" if self.is_pg else "?"      # плейсхолдер параметра

    def q(self, sql: str) -> str:
        """Переводит '?' в стиль текущей БД."""
        return sql.replace("?", "%s") if self.is_pg else sql


_url = os.environ.get("CLIPPER_DB_URL") or f"sqlite:///{DEFAULT_DB}"
DIALECT = _Dialect(_url)
_local = threading.local()


def _connect():
    if DIALECT.is_pg:
        import psycopg                              # noqa: PLC0415
        conn = psycopg.connect(DIALECT.url, autocommit=False)
        return conn
    path = DIALECT.url.replace("sqlite:///", "", 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL: читатели не блокируют писателя и наоборот — без этого веб-запросы
    # висели бы на каждой записи воркера.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _conn():
    c = getattr(_local, "conn", None)
    if c is None:
        c = _local.conn = _connect()
    return c


@contextmanager
def tx():
    """Транзакция. При исключении — полный откат, частичных записей не будет."""
    c = _conn()
    if DIALECT.is_pg:
        try:
            yield c.cursor()
            c.commit()
        except Exception:
            c.rollback()
            raise
        return
    cur = c.cursor()
    cur.execute("BEGIN IMMEDIATE")     # сразу берём блокировку записи
    try:
        yield cur
        cur.execute("COMMIT")
    except Exception:
        cur.execute("ROLLBACK")
        raise


def query(sql: str, params: tuple = ()) -> list[dict]:
    cur = _conn().cursor()
    cur.execute(DIALECT.q(sql), params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple = ()) -> None:
    with tx() as cur:
        cur.execute(DIALECT.q(sql), params)


# ---------------------------------------------------------------------------
# Схема
# ---------------------------------------------------------------------------

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id                TEXT PRIMARY KEY,
        email             TEXT NOT NULL UNIQUE,
        password_hash     TEXT NOT NULL,
        is_admin          INTEGER NOT NULL DEFAULT 0,
        disabled          INTEGER NOT NULL DEFAULT 0,
        created_at        TEXT NOT NULL,
        expires_at        TEXT,
        password_changed_at TEXT NOT NULL,
        note              TEXT NOT NULL DEFAULT '',
        quota_minutes     INTEGER NOT NULL DEFAULT 600,
        max_concurrent    INTEGER NOT NULL DEFAULT 1,
        max_video_minutes INTEGER NOT NULL DEFAULT 180,
        notify INTEGER NOT NULL DEFAULT 1
    )""",
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id            TEXT PRIMARY KEY,
        user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        status        TEXT NOT NULL,
        params        TEXT NOT NULL DEFAULT '{}',
        sources       TEXT NOT NULL DEFAULT '[]',
        title         TEXT NOT NULL DEFAULT '',
        created_at    TEXT NOT NULL,
        started_at    TEXT,
        finished_at   TEXT,
        heartbeat_at  TEXT,
        worker_id     TEXT,
        error         TEXT,
        clips_count   INTEGER NOT NULL DEFAULT 0,
        video_minutes REAL NOT NULL DEFAULT 0,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        progress      REAL NOT NULL DEFAULT 0,
        stage         TEXT NOT NULL DEFAULT 'queued',
        stage_detail  TEXT NOT NULL DEFAULT '',
        stage_times   TEXT NOT NULL DEFAULT '{}',
        wait_seconds  REAL NOT NULL DEFAULT 0,
        attempts      INTEGER NOT NULL DEFAULT 0,
        next_attempt_at TEXT,
        last_error    TEXT NOT NULL DEFAULT '',
        notified      INTEGER NOT NULL DEFAULT 0
    )""",
    """
    CREATE TABLE IF NOT EXISTS usage_records (
        id         TEXT PRIMARY KEY,
        user_id    TEXT,
        job_id     TEXT,
        provider   TEXT NOT NULL DEFAULT '',
        model      TEXT NOT NULL DEFAULT '',
        stage      TEXT NOT NULL DEFAULT '',
        tok_in     INTEGER NOT NULL DEFAULT 0,
        tok_out    INTEGER NOT NULL DEFAULT 0,
        cost       REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id          TEXT PRIMARY KEY,
        ts          TEXT NOT NULL,
        actor       TEXT NOT NULL DEFAULT '',
        action      TEXT NOT NULL,
        target      TEXT NOT NULL DEFAULT '',
        ip          TEXT NOT NULL DEFAULT '',
        detail      TEXT NOT NULL DEFAULT ''
    )""",
    "CREATE INDEX IF NOT EXISTS ix_jobs_user   ON jobs(user_id, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs(status, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_usage_user  ON usage_records(user_id, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_audit_ts    ON audit_log(ts)",
]

STATUSES = ("queued", "running", "done", "error", "canceled")


def init_db() -> None:
    with tx() as cur:
        for stmt in _SCHEMA:
            cur.execute(stmt)


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(ts)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def _uid() -> str:
    return uuid.uuid4().hex


def _norm_email(email: str) -> str:
    return (email or "").strip().lower()


# ---------------------------------------------------------------------------
# Пароли
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    if _ARGON is not None:
        return _ARGON.hash(password)
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             salt.encode("utf-8"), _PBKDF2_ROUNDS)
    return f"pbkdf2${_PBKDF2_ROUNDS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Понимает три формата: argon2, новый pbkdf2$… и старый salt$hash из
    users.json — иначе после переезда никто бы не вошёл."""
    if not stored:
        return False
    try:
        if stored.startswith("$argon2"):
            if _ARGON is None:
                return False
            try:
                _ARGON.verify(stored, password)
                return True
            except Exception:  # noqa: BLE001
                return False
        parts = stored.split("$")
        if stored.startswith("pbkdf2$") and len(parts) == 4:
            rounds, salt, digest = int(parts[1]), parts[2], parts[3]
        elif len(parts) == 2:                     # старый формат salt$hash
            rounds, salt, digest = 200_000, parts[0], parts[1]
        else:
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 salt.encode("utf-8"), rounds)
        # Сравниваем байты: compare_digest на не-ASCII строках бросает TypeError
        return secrets.compare_digest(dk.hex().encode(), digest.encode())
    except Exception:  # noqa: BLE001
        return False


def needs_rehash(stored: str) -> bool:
    """Пароль сохранён старым способом — стоит пересчитать при следующем входе."""
    if _ARGON is not None:
        return not stored.startswith("$argon2")
    return not stored.startswith("pbkdf2$")


# ---------------------------------------------------------------------------
# Пользователи
# ---------------------------------------------------------------------------

MIN_PASSWORD_LEN = 10


def _check_password(password: str) -> None:
    if len(password or "") < MIN_PASSWORD_LEN:
        raise ValueError(f"Пароль должен быть не короче {MIN_PASSWORD_LEN} символов")


def create_user(email: str, password: str, days: int = 30, is_admin: bool = False,
                note: str = "", quota_minutes: int = 600,
                max_concurrent: int = 1, max_video_minutes: int = 180) -> dict:
    email = _norm_email(email)
    if not email or "@" not in email:
        raise ValueError("Нужен корректный адрес почты")
    _check_password(password)
    if get_user_by_email(email):
        raise ValueError(f"Пользователь {email} уже существует")
    uid, now = _uid(), _now()
    expires = (datetime.now(timezone.utc) + timedelta(days=int(days))).isoformat()
    execute("""INSERT INTO users (id, email, password_hash, is_admin, disabled,
               created_at, expires_at, password_changed_at, note, quota_minutes,
               max_concurrent, max_video_minutes)
               VALUES (?,?,?,?,0,?,?,?,?,?,?,?)""",
            (uid, email, hash_password(password), 1 if is_admin else 0, now,
             expires, now, note, quota_minutes, max_concurrent, max_video_minutes))
    return {"id": uid, "email": email}


def get_user_by_email(email: str) -> dict | None:
    return query_one("SELECT * FROM users WHERE email = ?", (_norm_email(email),))


def get_user(user_id: str) -> dict | None:
    return query_one("SELECT * FROM users WHERE id = ?", (user_id,))


def set_password(email: str, password: str) -> None:
    _check_password(password)
    u = get_user_by_email(email)
    if not u:
        raise ValueError("Пользователь не найден")
    # password_changed_at обновляется, чтобы разлогинить чужие сессии:
    # web.py сверяет эту метку с сохранённой в cookie.
    execute("UPDATE users SET password_hash = ?, password_changed_at = ? WHERE id = ?",
            (hash_password(password), _now(), u["id"]))


def refresh_hash(user_id: str, password: str) -> None:
    """Тихо пересчитывает хеш в новый формат — сессии при этом не рвутся."""
    execute("UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(password), user_id))


def set_limits(email: str, **fields) -> None:
    allowed = {"quota_minutes", "max_concurrent", "max_video_minutes",
               "note", "disabled", "notify"}
    upd = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not upd:
        return
    u = get_user_by_email(email)
    if not u:
        raise ValueError("Пользователь не найден")
    sets = ", ".join(f"{k} = ?" for k in upd)
    execute(f"UPDATE users SET {sets} WHERE id = ?", (*upd.values(), u["id"]))


def extend_access(email: str, days: int) -> str:
    u = get_user_by_email(email)
    if not u:
        raise ValueError("Пользователь не найден")
    cur = _parse(u.get("expires_at"))
    now = datetime.now(timezone.utc)
    base = cur if (cur and cur > now) else now
    new = (base + timedelta(days=int(days))).isoformat()
    execute("UPDATE users SET expires_at = ? WHERE id = ?", (new, u["id"]))
    return new


def delete_user(email: str) -> None:
    u = get_user_by_email(email)
    if not u:
        raise ValueError("Пользователь не найден")
    if u["is_admin"] and count_admins() <= 1:
        raise ValueError("Нельзя удалить единственного администратора")
    execute("DELETE FROM users WHERE id = ?", (u["id"],))


def count_admins() -> int:
    r = query_one("SELECT COUNT(*) AS n FROM users WHERE is_admin = 1")
    return int(r["n"]) if r else 0


def is_active(user: dict | None) -> bool:
    if not user or user.get("disabled"):
        return False
    if user.get("is_admin"):
        return True
    exp = _parse(user.get("expires_at"))
    return bool(exp and exp > datetime.now(timezone.utc))


def days_left(user: dict | None) -> int | None:
    if not user:
        return None
    exp = _parse(user.get("expires_at"))
    if not exp:
        return None
    secs = (exp - datetime.now(timezone.utc)).total_seconds()
    if secs <= 0:
        return 0
    return max(1, -(-int(secs) // 86400))       # округление вверх


def hours_left(user: dict | None) -> int | None:
    if not user:
        return None
    exp = _parse(user.get("expires_at"))
    if not exp:
        return None
    return max(0, int((exp - datetime.now(timezone.utc)).total_seconds() // 3600))


def check_login(email: str, password: str) -> tuple[dict | None, str]:
    """-> (пользователь, текст ошибки). Пароль проверяется всегда, даже для
    несуществующего аккаунта: иначе по времени ответа можно было бы узнать,
    какие адреса зарегистрированы."""
    u = get_user_by_email(email)
    stored = u["password_hash"] if u else hash_password("нет такого пользователя")
    ok = verify_password(password or "", stored)
    if not u or not ok:
        return None, "Неверная почта или пароль"
    if u["disabled"]:
        return None, "Доступ приостановлен. Обратитесь к администратору."
    if not is_active(u):
        return None, "Срок доступа истёк. Обратитесь к администратору."
    if needs_rehash(stored):
        refresh_hash(u["id"], password)
    return u, ""


def list_users() -> list[dict]:
    rows = query("SELECT * FROM users ORDER BY email")
    out = []
    for u in rows:
        out.append({
            "email": u["email"],
            "is_admin": bool(u["is_admin"]),
            "disabled": bool(u["disabled"]),
            "created": u["created_at"],
            "expires": u["expires_at"],
            "days_left": days_left(u),
            "active": is_active(u),
            "note": u["note"],
            "quota_minutes": u["quota_minutes"],
            "used_minutes": round(used_minutes(u["id"]), 1),
            "max_concurrent": u["max_concurrent"],
            "max_video_minutes": u["max_video_minutes"],
        })
    return out


def ensure_admin_exists() -> tuple[str, str] | None:
    if count_admins() > 0:
        return None
    email = _norm_email(os.environ.get("CLIPPER_ADMIN_EMAIL", "admin@local"))
    password = os.environ.get("CLIPPER_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
    create_user(email, password, days=3650, is_admin=True, note="владелец",
                quota_minutes=10 ** 6, max_concurrent=4, max_video_minutes=100000)
    return email, password


# ---------------------------------------------------------------------------
# Квоты
# ---------------------------------------------------------------------------

def used_minutes(user_id: str, days: int = 30) -> float:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    r = query_one("""SELECT COALESCE(SUM(video_minutes), 0) AS m FROM jobs
                     WHERE user_id = ? AND created_at >= ?
                       AND status IN ('running','done')""", (user_id, since))
    return float(r["m"]) if r else 0.0


def active_jobs_count(user_id: str) -> int:
    r = query_one("""SELECT COUNT(*) AS n FROM jobs
                     WHERE user_id = ? AND status IN ('queued','running')""",
                  (user_id,))
    return int(r["n"]) if r else 0


def check_quota(user: dict) -> str:
    """-> текст ошибки или '' если можно запускать."""
    if active_jobs_count(user["id"]) >= int(user["max_concurrent"]):
        return (f"У вас уже {user['max_concurrent']} задач(и) в работе. "
                f"Дождитесь завершения.")
    used, limit = used_minutes(user["id"]), int(user["quota_minutes"])
    if used >= limit:
        return (f"Исчерпан месячный лимит: {int(used)} из {limit} минут видео. "
                f"Напишите администратору.")
    return ""


# ---------------------------------------------------------------------------
# Задачи
# ---------------------------------------------------------------------------

def create_job(user_id: str, sources: list[str], params: dict,
               title: str = "") -> str:
    job_id = _uid()
    execute("""INSERT INTO jobs (id, user_id, status, params, sources, title,
               created_at) VALUES (?,?,'queued',?,?,?,?)""",
            (job_id, user_id, json.dumps(params, ensure_ascii=False),
             json.dumps(sources, ensure_ascii=False), title[:200], _now()))
    return job_id


def get_job(job_id: str) -> dict | None:
    j = query_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if j:
        j["params"] = json.loads(j["params"] or "{}")
        j["sources"] = json.loads(j["sources"] or "[]")
    return j


def list_jobs(user_id: str | None = None, limit: int = 50) -> list[dict]:
    """Без user_id — все задачи (только для админа)."""
    if user_id:
        rows = query("""SELECT j.*, u.email FROM jobs j JOIN users u ON u.id = j.user_id
                        WHERE j.user_id = ? ORDER BY j.created_at DESC LIMIT ?""",
                     (user_id, limit))
    else:
        rows = query("""SELECT j.*, u.email FROM jobs j JOIN users u ON u.id = j.user_id
                        ORDER BY j.created_at DESC LIMIT ?""", (limit,))
    for r in rows:
        r.pop("params", None)          # в списке не нужны, могут быть объёмными
        r["sources"] = json.loads(r["sources"] or "[]")
    return rows


# Окно справедливости: за какой срок учитываются уже обслуженные задачи.
# Час — разумный компромисс: пачка не блокирует других прямо сейчас, но
# вчерашняя активность не наказывает пользователя сегодня.
FAIRNESS_WINDOW_HOURS = 1

# «Вес» задачи = сколько задач этого пользователя уже обслужено за последний
# час (load) + её номер в его собственной очереди (rn). Берётся задача с
# наименьшим весом, при равенстве — пришедшая раньше.
#
# Без слагаемого load нумерация пересчитывалась бы после каждой завершённой
# задачи, и пользователь с пачкой из 20 видео снова оказывался бы первым —
# то есть очередь только выглядела бы честной. Это ровно та ошибка, которую
# поймал тест: сначала обслуживались 8 задач подряд одного пользователя.
_FAIR_SQL = """
    SELECT q.id FROM (
        SELECT j.id, j.created_at,
               ROW_NUMBER() OVER (PARTITION BY j.user_id
                                  ORDER BY j.created_at) AS rn,
               (SELECT COUNT(*) FROM jobs r
                 WHERE r.user_id = j.user_id
                   AND (r.status = 'running'
                        OR (r.started_at IS NOT NULL AND r.started_at > ?))
               ) AS load
        FROM jobs j
        WHERE j.status = 'queued'
          -- задачи, отложенные до повтора, пока не берём
          AND (j.next_attempt_at IS NULL OR j.next_attempt_at <= ?)
    ) q
    ORDER BY (q.load + q.rn), q.created_at
    LIMIT 1
"""


def queue_position(job_id: str) -> int:
    """Сколько задач будет обработано раньше этой. 0 — следующая на очереди.

    Считается по тому же правилу, что и claim_job: иначе человек видел бы
    «перед вами 19», а его задача взялась бы третьей.
    """
    cutoff = (datetime.now(timezone.utc)
              - timedelta(hours=FAIRNESS_WINDOW_HOURS)).isoformat()
    rows = query(_FAIR_SQL.replace("LIMIT 1", ""), (cutoff, _now()))
    for i, r in enumerate(rows):
        if r["id"] == job_id:
            return i
    return 0


def claim_job(worker_id: str) -> dict | None:
    """Атомарно забирает одну задачу из очереди — по-честному, по кругу.

    ПОЧЕМУ НЕ ПРОСТО «КТО ПЕРВЫЙ ВСТАЛ»
    При строгом порядке по времени пользователь, отправивший пачку из 20 видео,
    занимал бы очередь на несколько часов, и все остальные ждали бы за ним.
    Поэтому задачи нумеруются внутри каждого пользователя (первая, вторая,
    третья…), и сначала берутся все первые задачи, потом все вторые и так
    далее. Один человек с пачкой получает свою долю мощности, но не всю.

    Ключевое место всей очереди: две копии воркера не должны взять одну
    задачу. В SQLite это обеспечивает BEGIN IMMEDIATE внутри tx(), в
    PostgreSQL — SELECT ... FOR UPDATE SKIP LOCKED.
    """
    now = _now()
    cutoff = (datetime.now(timezone.utc)
              - timedelta(hours=FAIRNESS_WINDOW_HOURS)).isoformat()
    with tx() as cur:
        sql = _FAIR_SQL + (" FOR UPDATE SKIP LOCKED" if DIALECT.is_pg else "")
        cur.execute(DIALECT.q(sql), (cutoff, now))
        row = cur.fetchone()
        if not row:
            return None
        job_id = row[0] if not isinstance(row, dict) else row["id"]
        cur.execute(DIALECT.q("""UPDATE jobs SET status = 'running', worker_id = ?,
                       started_at = ?, heartbeat_at = ?, attempts = attempts + 1
                       WHERE id = ?"""),
                    (worker_id, now, now, job_id))
    return get_job(job_id)


def heartbeat(job_id: str) -> bool:
    """Отметка «воркер жив». Возвращает False, если задачу попросили отменить."""
    execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ?", (_now(), job_id))
    j = query_one("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,))
    return not (j and j["cancel_requested"])


def request_cancel(job_id: str) -> None:
    execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))


# Максимум попыток. Больше трёх смысла нет: если не вышло трижды с паузами,
# причина устойчивая, и дальнейшие попытки только жгут время видеокарты.
MAX_ATTEMPTS = 3

# Паузы перед повторами. Растут, чтобы не долбить внешний сервис, который
# уже ответил «слишком много запросов».
RETRY_DELAYS_SEC = (60, 300, 900)


def retry_job(job_id: str, error: str) -> bool:
    """Возвращает задачу в очередь с задержкой. -> False, если попытки кончились.

    Временные сбои (YouTube не отдал видео, LLM упёрлась в лимит, сеть моргнула)
    не должны хоронить задачу: пользователь ни при чём и повторять вручную
    ему неоткуда.
    """
    j = query_one("SELECT attempts FROM jobs WHERE id = ?", (job_id,))
    attempts = int(j["attempts"]) if j else MAX_ATTEMPTS
    if attempts >= MAX_ATTEMPTS:
        return False
    delay = RETRY_DELAYS_SEC[min(attempts - 1, len(RETRY_DELAYS_SEC) - 1)]
    nxt = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
    execute("""UPDATE jobs SET status = 'queued', worker_id = NULL,
               started_at = NULL, heartbeat_at = NULL, next_attempt_at = ?,
               last_error = ?, progress = 0, stage = 'queued'
               WHERE id = ?""", (nxt, (error or "")[:1000], job_id))
    return True


def finish_job(job_id: str, status: str, error: str = "",
               clips_count: int = 0, video_minutes: float = 0.0) -> None:
    if status not in STATUSES:
        raise ValueError(f"Неизвестный статус: {status}")
    execute("""UPDATE jobs SET status = ?, finished_at = ?, error = ?,
               clips_count = ?, video_minutes = ? WHERE id = ?""",
            (status, _now(), (error or "")[:4000], clips_count,
             video_minutes, job_id))


def set_progress(job_id: str, percent: float, stage: str, detail: str) -> None:
    """Сохраняет прогресс в базу, чтобы он пережил перезагрузку страницы:
    иначе обновивший вкладку пользователь увидел бы пустой экран."""
    execute("UPDATE jobs SET progress = ?, stage = ?, stage_detail = ? WHERE id = ?",
            (float(percent), stage[:32], (detail or "")[:200], job_id))


def mark_notified(job_id: str) -> bool:
    """Помечает задачу как «уведомление отправлено».

    -> False, если уже было отправлено. Защищает от повторных писем при
    перезапуске воркера: получить три одинаковых письма о готовности
    неприятнее, чем не получить ни одного.
    """
    with tx() as cur:
        cur.execute(DIALECT.q("SELECT notified FROM jobs WHERE id = ?"), (job_id,))
        row = cur.fetchone()
        if not row:
            return False
        already = row[0] if not isinstance(row, dict) else row["notified"]
        if already:
            return False
        cur.execute(DIALECT.q("UPDATE jobs SET notified = 1 WHERE id = ?"), (job_id,))
    return True


def set_stage_times(job_id: str, times: dict) -> None:
    """Сохраняет длительность каждого этапа в секундах.

    Это основа для планирования мощностей: без реальных замеров на конкретной
    видеокарте автомасштабирование настраивается наугад — неверные пороги
    означают либо очередь, либо оплату простаивающих карт.
    """
    execute("UPDATE jobs SET stage_times = ? WHERE id = ?",
            (json.dumps(times, ensure_ascii=False), job_id))


def set_wait_seconds(job_id: str, seconds: float) -> None:
    """Сколько задача пролежала в очереди до старта — главный показатель
    качества сервиса для пользователя."""
    execute("UPDATE jobs SET wait_seconds = ? WHERE id = ?", (float(seconds), job_id))


def metrics(days: int = 7) -> dict:
    """Сводка для планирования мощностей и наблюдения за сервисом."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    jobs = query("""SELECT stage_times, video_minutes, wait_seconds, status,
                           started_at, finished_at, created_at
                    FROM jobs WHERE created_at >= ?""", (since,))
    done = [j for j in jobs if j["status"] == "done"]

    # длительности этапов
    stage_tot: dict[str, list[float]] = {}
    for j in done:
        try:
            for st, sec in json.loads(j["stage_times"] or "{}").items():
                stage_tot.setdefault(st, []).append(float(sec))
        except Exception:  # noqa: BLE001
            continue

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "avg": 0, "p50": 0, "p95": 0}
        v = sorted(vals)
        return {"n": len(v), "avg": round(sum(v) / len(v), 1),
                "p50": round(v[len(v) // 2], 1),
                "p95": round(v[min(len(v) - 1, int(len(v) * 0.95))], 1)}

    totals = []
    for j in done:
        a, b = _parse(j["started_at"]), _parse(j["finished_at"])
        if a and b:
            totals.append((b - a).total_seconds())
    waits = [float(j["wait_seconds"] or 0) for j in jobs
             if j["status"] in ("done", "error", "running")]

    # Этапы, реально занимающие видеокарту — по ним считается потребность в GPU.
    gpu_stages = ("transcribe", "render")
    gpu_sec = sum(sum(stage_tot.get(s, [])) for s in gpu_stages)

    hours = max(1.0, days * 24)
    return {
        "period_days": days,
        "jobs_total": len(jobs),
        "jobs_done": len(done),
        "jobs_error": sum(1 for j in jobs if j["status"] == "error"),
        "error_rate": round(100 * sum(1 for j in jobs if j["status"] == "error")
                            / max(1, len(jobs)), 1),
        "by_stage": {k: _stats(v) for k, v in sorted(stage_tot.items())},
        "total_seconds": _stats(totals),
        "wait_seconds": _stats(waits),
        "video_minutes": round(sum(float(j["video_minutes"] or 0) for j in done), 1),
        "gpu_hours": round(gpu_sec / 3600, 2),
        "gpu_hours_per_day": round(gpu_sec / 3600 / max(1, days), 2),
        # Сколько карт нужно в среднем; пик обычно в 2-3 раза выше.
        "gpus_avg": round(gpu_sec / 3600 / hours, 2),
        "queue_now": queue_depth(),
    }


def queue_depth() -> int:
    r = query_one("SELECT COUNT(*) n FROM jobs WHERE status = 'queued'")
    return int(r["n"]) if r else 0


def running_count() -> int:
    r = query_one("SELECT COUNT(*) n FROM jobs WHERE status = 'running'")
    return int(r["n"]) if r else 0


def active_workers(max_silence_sec: int = 120) -> int:
    """Сколько воркеров подавали признаки жизни за последние минуты."""
    cutoff = (datetime.now(timezone.utc)
              - timedelta(seconds=max_silence_sec)).isoformat()
    r = query_one("""SELECT COUNT(DISTINCT worker_id) n FROM jobs
                     WHERE status = 'running' AND heartbeat_at >= ?""", (cutoff,))
    return int(r["n"]) if r else 0


def set_video_minutes(job_id: str, minutes: float) -> None:
    execute("UPDATE jobs SET video_minutes = ? WHERE id = ?", (minutes, job_id))


def requeue_stale_jobs(max_silence_sec: int = 600) -> int:
    """Возвращает в очередь задачи, чей воркер умер (нет отметок heartbeat).

    Без этого падение воркера навсегда оставляло бы задачу в статусе
    «выполняется», а пользователя — с крутящимся индикатором.
    """
    cutoff = (datetime.now(timezone.utc)
              - timedelta(seconds=max_silence_sec)).isoformat()
    rows = query("""SELECT id FROM jobs WHERE status = 'running'
                    AND (heartbeat_at IS NULL OR heartbeat_at < ?)""", (cutoff,))
    for r in rows:
        execute("""UPDATE jobs SET status = 'queued', worker_id = NULL,
                   started_at = NULL, heartbeat_at = NULL WHERE id = ?""", (r["id"],))
    return len(rows)


# ---------------------------------------------------------------------------
# Расход и журнал
# ---------------------------------------------------------------------------

def record_usage(user_id: str | None, job_id: str | None, provider: str,
                 model: str, stage: str, tok_in: int, tok_out: int,
                 cost: float) -> None:
    execute("""INSERT INTO usage_records (id, user_id, job_id, provider, model,
               stage, tok_in, tok_out, cost, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (_uid(), user_id, job_id, provider, model, stage,
             tok_in, tok_out, cost, _now()))


def audit(action: str, actor: str = "", target: str = "", ip: str = "",
          detail: str = "") -> None:
    """Журнал значимых действий: входы, изменения аккаунтов, удаления."""
    try:
        execute("""INSERT INTO audit_log (id, ts, actor, action, target, ip, detail)
                   VALUES (?,?,?,?,?,?,?)""",
                (_uid(), _now(), actor[:200], action[:64], target[:200],
                 ip[:64], detail[:1000]))
    except Exception:  # noqa: BLE001 — журнал не должен ломать основную работу
        pass


def list_audit(limit: int = 200) -> list[dict]:
    return query("SELECT * FROM audit_log ORDER BY ts DESC LIMIT ?", (limit,))


def purge_audit(days: int = 365) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = query("SELECT COUNT(*) AS n FROM audit_log WHERE ts < ?", (cutoff,))
    execute("DELETE FROM audit_log WHERE ts < ?", (cutoff,))
    return int(rows[0]["n"]) if rows else 0
