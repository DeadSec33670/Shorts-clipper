"""Локальный веб-интерфейс для shorts-clipper.

Запуск (из корня проекта, в активном venv):
    .\\venv312\\Scripts\\python.exe web.py
затем открыть в браузере http://127.0.0.1:5000

Сервер работает ТОЛЬКО на твоём компьютере: он запускает тот же main.py как
подпроцесс, показывает его лог в реальном времени и даёт скачать готовые
клипы. Никуда в интернет ничего не отправляется (кроме обычных запросов
самого скрипта — к YouTube и LLM-провайдеру).
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
from datetime import timedelta
from functools import wraps
from pathlib import Path

# Windows-консоль по умолчанию в cp1251/cp866 — падает на → … и эмодзи.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

try:
    from flask import (Flask, Response, g, jsonify, redirect, render_template,
                       request, send_from_directory, session)
except ImportError:
    print("Не установлен Flask. Выполни в активном venv:\n"
          "    pip install flask")
    sys.exit(1)

import notify
import paths
import storage
import store

ROOT = Path(__file__).resolve().parent

# Куда писать администратору за доступом. Оплаты на сайте нет — только контакт.
CONTACT = os.environ.get("CLIPPER_CONTACT", "").strip()

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------------
# РЕЖИМ РАБОТЫ
#
# По умолчанию — ЛОКАЛЬНЫЙ: сервер слушает 127.0.0.1, за экраном хозяин машины,
# доступны удобные, но небезопасные возможности (пути к файлам на диске, куки
# браузера для yt-dlp).
#
# CLIPPER_PUBLIC=1 — ПУБЛИЧНЫЙ режим для сервера в интернете. Отключает всё,
# что позволяет пользователю дотянуться до файловой системы сервера, и включает
# строгие настройки cookie. Ставьте его ВСЕГДА при деплое наружу.
# ---------------------------------------------------------------------------
def _envflag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


PUBLIC_MODE = _envflag("CLIPPER_PUBLIC")

# ключ подписи сессий: постоянный между перезапусками, чтобы не разлогинивало.
# В публичном режиме берём его ТОЛЬКО из окружения — файл в корне проекта
# слишком легко утекает с бэкапом или в git, а утечка ключа = подделка любой
# сессии, включая админскую.
_env_secret = (os.environ.get("CLIPPER_SECRET_KEY") or "").strip()
if _env_secret:
    app.secret_key = _env_secret
elif PUBLIC_MODE:
    print("ОШИБКА: в публичном режиме задайте CLIPPER_SECRET_KEY "
          "(случайная строка, например: python -c \"import secrets;"
          "print(secrets.token_hex(32))\")")
    sys.exit(1)
else:
    _SECRET_FILE = ROOT / ".flask_secret"
    if not _SECRET_FILE.exists():
        _SECRET_FILE.write_text(secrets.token_hex(32), encoding="utf-8")
        try:                      # не даём читать ключ другим пользователям ОС
            _SECRET_FILE.chmod(0o600)
        except Exception:  # noqa: BLE001 — на Windows chmod почти не работает
            pass
    app.secret_key = _SECRET_FILE.read_text(encoding="utf-8").strip()

# Максимальный размер запроса. Без него один запрос может забить диск целиком.
_MAX_UPLOAD_GB = float(os.environ.get("CLIPPER_MAX_UPLOAD_GB", "2"))

app.config.update(
    MAX_CONTENT_LENGTH=int(_MAX_UPLOAD_GB * 1024 ** 3),
    SESSION_COOKIE_HTTPONLY=True,      # cookie недоступна из JavaScript
    SESSION_COOKIE_SAMESITE="Lax",     # не уходит на сторонние POST-запросы
    SESSION_COOKIE_SECURE=PUBLIC_MODE,  # только по HTTPS (локально ломало бы вход)
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
)

# Разрешённые расширения загружаемых файлов. Проверка по расширению — не
# гарантия (содержимое всё равно проверит ffmpeg), но отсекает мусор сразу.
_ALLOWED_UPLOAD_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
                       ".mpg", ".mpeg", ".ts", ".flv", ".wmv"}


# ---------------------------------------------------------------------------
# Защита от CSRF
#
# Все изменяющие запросы идут POST-ом с cookie-сессией. Без токена сторонний
# сайт мог бы отправить форму от имени залогиненного администратора и, скажем,
# удалить всех пользователей. Проверяем два независимых признака: источник
# запроса (Origin/Referer) и одноразовый токен из сессии.
# ---------------------------------------------------------------------------

def csrf_token() -> str:
    """Токен текущей сессии; создаётся при первом обращении."""
    tok = session.get("_csrf")
    if not tok:
        tok = secrets.token_urlsafe(32)
        session["_csrf"] = tok
    return tok


def _same_origin(req) -> bool:
    """Пришёл ли запрос с нашей же страницы."""
    origin = req.headers.get("Origin") or ""
    referer = req.headers.get("Referer") or ""
    source = origin or referer
    if not source:
        # Старые клиенты и curl не шлют заголовок. Токен всё равно проверяется
        # отдельно, поэтому здесь не блокируем.
        return True
    from urllib.parse import urlparse
    host = urlparse(source).netloc
    return bool(host) and host == req.host


@app.before_request
def _csrf_protect():
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    if not _same_origin(request):
        return jsonify({"ok": False, "error": "Запрос с чужого источника"}), 403
    sent = (request.headers.get("X-CSRF-Token")
            or (request.form.get("csrf_token") if request.form else None)
            or ((request.get_json(silent=True) or {}).get("csrf_token")))
    expected = session.get("_csrf")
    # Сравниваем именно байты: compare_digest на строках с не-ASCII символами
    # бросает TypeError, и подделанный токен с кириллицей давал бы 500 вместо 403.
    if (not expected or not sent
            or not secrets.compare_digest(str(sent).encode("utf-8", "replace"),
                                          str(expected).encode("utf-8", "replace"))):
        return jsonify({"ok": False, "error": "Недействительный CSRF-токен. "
                                              "Обновите страницу и повторите."}), 403
    return None


@app.after_request
def _security_headers(resp):
    """Заголовки, снижающие ущерб от возможной XSS и от встраивания сайта."""
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault(
        "Content-Security-Policy",
        # script-src без 'unsafe-inline': скрипты вынесены в static/, а
        # встроенные onclick= заменены делегированием событий. Это главная
        # защита от внедрения чужого кода через содержимое страницы.
        # style-src пока с 'unsafe-inline' — в разметке остаются style="…".
        "default-src 'self'; img-src 'self' data:; "
        "media-src 'self' blob: https:; "
        "script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
        "form-action 'self'")
    if PUBLIC_MODE:
        resp.headers.setdefault("Strict-Transport-Security",
                                "max-age=31536000; includeSubDomains")
    return resp


# ---------------------------------------------------------------------------
# Ограничение попыток входа
#
# Счётчики в памяти процесса: их достаточно для одного сервера. При нескольких
# воркерах счётчик нужно перенести в Redis, иначе лимит делится между ними.
# ---------------------------------------------------------------------------
_LOGIN_MAX_ATTEMPTS = 7
_LOGIN_WINDOW_SEC = 15 * 60
_login_attempts: dict[str, list[float]] = {}
_login_lock = threading.Lock()


def _login_key(email: str) -> tuple[str, str]:
    ip = (request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
          or request.remote_addr or "?")
    return f"ip:{ip}", f"em:{(email or '').strip().lower()}"


def _login_blocked(email: str) -> int:
    """-> сколько секунд ждать (0 — можно пробовать)."""
    now = time.time()
    wait = 0
    with _login_lock:
        for key in _login_key(email):
            tries = [t for t in _login_attempts.get(key, [])
                     if now - t < _LOGIN_WINDOW_SEC]
            _login_attempts[key] = tries
            if len(tries) >= _LOGIN_MAX_ATTEMPTS:
                wait = max(wait, int(_LOGIN_WINDOW_SEC - (now - tries[0])) + 1)
    return wait


def _login_failed(email: str) -> None:
    now = time.time()
    with _login_lock:
        for key in _login_key(email):
            _login_attempts.setdefault(key, []).append(now)
        if len(_login_attempts) > 10000:        # защита от роста словаря
            for k in list(_login_attempts):
                if not [t for t in _login_attempts[k]
                        if now - t < _LOGIN_WINDOW_SEC]:
                    del _login_attempts[k]


def _login_ok(email: str) -> None:
    with _login_lock:
        for key in _login_key(email):
            _login_attempts.pop(key, None)


def _client_ip() -> str:
    """IP клиента. За nginx настоящий адрес приходит в X-Forwarded-For;
    берём первый элемент — остальные подставляются самим клиентом."""
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() or request.remote_addr or "?")[:64]


@app.before_request
def _load_user():
    """Подставляет текущего пользователя в g.user до каждого запроса.

    Проверяется не только наличие сессии, но и метка смены пароля: если пароль
    изменили (сам пользователь или админ), все прежние сессии становятся
    недействительными — иначе укравший cookie сохранял бы доступ.
    """
    g.user = None
    uid = session.get("uid")
    if not uid:
        return
    u = store.get_user(uid)
    if not u or not store.is_active(u):
        session.clear()
        return
    if session.get("pwd") != u["password_changed_at"]:
        session.clear()
        return
    g.user = u


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not g.get("user"):
            if request.path.startswith("/api/"):
                return jsonify({"ok": False, "auth": False,
                                "error": "Требуется вход"}), 401
            return redirect("/login")
        return fn(*a, **kw)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not g.get("user") or not g.user["is_admin"]:
            return jsonify({"ok": False,
                            "error": "Только для администратора"}), 403
        return fn(*a, **kw)
    return wrapper


# ===========================================================================
# ЗАДАЧИ
#
# Веб-сервер больше не запускает обработку сам: он кладёт задачу в очередь
# (таблица jobs), а её забирает worker.py. Благодаря этому несколько
# пользователей работают одновременно, а перезапуск сайта не убивает
# начатую обработку.
# ===========================================================================

# Значения, которые разрешено передавать в обработку. Всё, чего нет в списках,
# отвергается: поля формы попадают в аргументы командной строки, и без белых
# списков через них можно было бы дописать в команду посторонние флаги.
_ALLOWED = {
    "llm_provider": {"gemini", "groq", "openrouter", "claude"},
    "length": {"30s", "1m", "2m"},
    "model": {"tiny", "base", "small", "medium",
              "large-v2", "large-v3", "large-v3-turbo"},
    "camera_mode": {"smart", "smart_asd", "static"},
    "whisper_device": {"auto", "cpu", "cuda"},
}
_NUM = {           # поле: (тип, минимум, максимум)
    "clips": (int, 1, 50),
    "candidates_per_film": (int, 1, 30),
    "min_sec": (float, 3, 600),
    "max_sec": (float, 5, 900),
    "static_crop": (float, 0.0, 0.5),
}


def _clean_params(f: dict) -> dict:
    """Проверяет поля формы и возвращает параметры задачи."""
    p: dict = {}

    for key, allowed in _ALLOWED.items():
        v = (f.get(key) or "").strip()
        if v:
            if v not in allowed:
                raise ValueError(f"Недопустимое значение поля «{key}»")
            p[key] = v

    for key, (cast, lo, hi) in _NUM.items():
        v = (f.get(key) or "").strip()
        if v:
            try:
                num = cast(v)
            except ValueError:
                raise ValueError(f"Поле «{key}»: нужно число") from None
            if not (lo <= num <= hi):
                raise ValueError(f"Поле «{key}»: допустимо от {lo} до {hi}")
            p[key] = num

    for key in ("llm_model", "language", "font"):
        v = (f.get(key) or "").strip()
        if v:
            if len(v) > 64 or v.startswith("-"):
                raise ValueError(f"Недопустимое значение поля «{key}»")
            p[key] = v

    for flag, key, default in (("sub_color", "sub_color", "#ffffff"),
                               ("sub_highlight", "sub_highlight", "#ffd447")):
        v = (f.get(key) or "").strip()
        if v and v.lower() != default:
            if not re.fullmatch(r"#?[0-9a-fA-F]{6}", v):
                raise ValueError("Цвет должен быть в формате #RRGGBB")
            p[flag] = v.lstrip("#").upper()

    # Куки читаются с диска СЕРВЕРА и дают доступ к чужому Google-аккаунту.
    # В публичном режиме недопустимы ни в каком виде, локально — нужны для
    # обхода проверки «подтвердите, что вы не робот».
    if not PUBLIC_MODE:
        cb = (f.get("cookies_browser") or "").strip()
        if cb:
            if cb not in {"chrome", "firefox", "edge", "brave", "opera",
                          "vivaldi", "chromium"}:
                raise ValueError("Недопустимый браузер для куков")
            p["cookies_browser"] = cb
        px = (f.get("proxy") or "").strip()
        if px:
            # socks5h/socks4a — разрешение имён НА СТОРОНЕ ПРОКСИ.
            # Это важно там, где провайдер подменяет DNS: с обычным socks5
            # имя youtube.com разрешается локально и может вернуться
            # неверный адрес, хотя сам прокси исправен.
            if not px.startswith(("http://", "https://", "socks5://",
                                  "socks5h://", "socks4://", "socks4a://")):
                raise ValueError("Прокси должен начинаться с http://, https://, "
                                 "socks5:// или socks5h://")
            if len(px) > 200:
                raise ValueError("Слишком длинный адрес прокси")
            p["proxy"] = px

        cf = (f.get("cookies_file") or "").strip()
        if cf:
            if cf.startswith("-"):
                raise ValueError("Недопустимый путь к файлу куков")
            p["cookies_file"] = cf

    mode = p.pop("camera_mode", "smart")
    if mode == "static":
        p["static_fit"] = True
    elif mode == "smart_asd":
        p["asd"] = True

    for key in ("no_subs", "no_hook", "debug_camera", "batch"):
        if f.get(key) == "on":
            p[key] = True

    if p.get("min_sec") and p.get("max_sec") and p["min_sec"] > p["max_sec"]:
        raise ValueError("Минимальная длительность больше максимальной")
    return p


def _collect_sources(form: dict, files) -> list[str]:
    """Ссылки и загруженные файлы. Пути к файлам на диске сервера в публичном
    режиме не принимаются — это было бы чтением произвольного файла."""
    sources: list[str] = []

    urls = (form.get("urls") or "").strip()
    if urls:
        for u in urls.replace(",", "\n").split("\n"):
            u = u.strip()
            if not u:
                continue
            if not u.startswith(("http://", "https://")):
                raise ValueError(f"Не похоже на ссылку: {u[:60]}")
            if len(u) > 500:
                raise ValueError("Слишком длинная ссылка")
            sources.append(u)

    user_dir = paths.user_uploads(g.user["id"])
    user_dir.mkdir(parents=True, exist_ok=True)
    for fs in files:
        if not fs.filename:
            continue
        ext = Path(fs.filename).suffix.lower()
        if ext not in _ALLOWED_UPLOAD_EXT:
            raise ValueError(f"Формат {ext or '?'} не поддерживается")
        dest = user_dir / f"{secrets.token_hex(8)}{ext}"
        fs.save(dest)
        sources.append(str(dest))

    manual = (form.get("paths") or "").strip()
    if manual:
        if PUBLIC_MODE:
            raise ValueError("Указание путей к файлам на сервере отключено. "
                             "Загрузите файл или дайте ссылку.")
        sources += [p.strip().strip('"') for p in manual.split("\n") if p.strip()]

    if not sources:
        raise ValueError("Не указано ни одного видео: добавьте ссылку или файл.")
    if len(sources) > 20:
        raise ValueError("За один раз можно отправить не больше 20 видео")
    return sources


@app.post("/api/start")
@login_required
def api_start():
    err = store.check_quota(g.user)
    if err:
        return jsonify({"ok": False, "error": err}), 429
    try:
        params = _clean_params(request.form.to_dict())
        sources = _collect_sources(request.form.to_dict(),
                                   request.files.getlist("files"))
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    title = sources[0][:120] if sources else ""
    job_id = store.create_job(g.user["id"], sources, params, title)
    store.audit("job_create", g.user["email"], job_id, _client_ip(),
                f"источников: {len(sources)}")
    return jsonify({"ok": True, "job_id": job_id,
                    "position": store.queue_position(job_id)})


def _owned_job(job_id: str) -> dict | None:
    """Задача, если она принадлежит текущему пользователю (админу — любая)."""
    job = store.get_job(job_id)
    if not job:
        return None
    if job["user_id"] != g.user["id"] and not g.user["is_admin"]:
        return None
    return job


@app.get("/api/jobs")
@login_required
def api_jobs():
    """Свои задачи. Админ через ?all=1 видит все — для поддержки."""
    if request.args.get("all") and g.user["is_admin"]:
        rows = store.list_jobs(None, limit=100)
    else:
        rows = store.list_jobs(g.user["id"], limit=50)
    for r in rows:
        if r["status"] == "queued":
            r["position"] = store.queue_position(r["id"])
        r["progress_info"] = _progress_info(r)
        if not g.user["is_admin"]:
            r.pop("email", None)
            r.pop("worker_id", None)
    return jsonify(rows)


@app.get("/api/jobs/<job_id>")
@login_required
def api_job(job_id: str):
    job = _owned_job(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Задача не найдена"}), 404
    job["position"] = store.queue_position(job_id)
    job["progress_info"] = _progress_info(job)
    if not g.user["is_admin"]:
        job.pop("worker_id", None)
    return jsonify(job)


def _progress_info(job: dict) -> dict:
    """Прогресс задачи в понятном пользователю виде."""
    from clipper.progress import ProgressTracker, estimate_seconds_left

    t = ProgressTracker()
    t.stage = job.get("stage") or "queued"
    t.percent = float(job.get("progress") or 0)
    t.detail = job.get("stage_detail") or ""

    if job["status"] == "done":
        t.stage, t.percent = "done", 100.0
    elif job["status"] in ("error", "canceled"):
        return {"stage": job["status"], "percent": t.percent,
                "label": "Отменено" if job["status"] == "canceled"
                         else "Завершилось ошибкой",
                "detail": job.get("error") or "", "eta": ""}

    eta = None
    started = job.get("started_at")
    if started and job["status"] == "running":
        from datetime import datetime, timezone
        d = store._parse(started)          # noqa: SLF001
        if d:
            elapsed = (datetime.now(timezone.utc) - d).total_seconds()
            eta = estimate_seconds_left(t.percent, elapsed)

    pos = job.get("position") if job["status"] == "queued" else None
    return t.snapshot(queue_position=pos, eta_seconds=eta)


@app.post("/api/jobs/<job_id>/cancel")
@login_required
def api_job_cancel(job_id: str):
    job = _owned_job(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Задача не найдена"}), 404
    if job["status"] not in ("queued", "running"):
        return jsonify({"ok": False, "error": "Задача уже завершена"}), 400
    if job["status"] == "queued":
        store.finish_job(job_id, "canceled")
    else:
        store.request_cancel(job_id)      # воркер увидит и остановит обработку
    store.audit("job_cancel", g.user["email"], job_id, _client_ip())
    return jsonify({"ok": True})


@app.get("/api/jobs/<job_id>/log")
@login_required
def api_job_log(job_id: str):
    """Лог задачи потоком. У каждой задачи он свой — чужой лог недоступен,
    в отличие от прежнего общего окна вывода."""
    job = _owned_job(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Задача не найдена"}), 404

    log_path = paths.job_log_path(job["user_id"], job_id)
    is_admin = bool(g.user["is_admin"])

    def gen():
        pos = 0
        idle = 0
        while True:
            try:
                if log_path.exists():
                    with open(log_path, encoding="utf-8", errors="replace") as fh:
                        fh.seek(pos)
                        chunk = fh.read()
                        pos = fh.tell()
                    if chunk:
                        idle = 0
                        for line in chunk.splitlines(keepends=True):
                            yield f"data: {json.dumps(_filter_log(line, is_admin))}\n\n"
            except Exception:  # noqa: BLE001
                pass

            fresh = store.get_job(job_id)
            if not fresh or fresh["status"] not in ("queued", "running"):
                status = fresh["status"] if fresh else "error"
                msg = {"done": "\n=== Готово ===\n",
                       "error": f"\n=== Ошибка: {(fresh or {}).get('error','')} ===\n",
                       "canceled": "\n=== Отменено ===\n"}.get(status, "")
                if msg:
                    yield f"data: {json.dumps(msg)}\n\n"
                yield f"event: end\ndata: {json.dumps(status)}\n\n"
                return

            idle += 1
            if idle > 3600:            # час без единой строки — обрываем поток
                yield "event: end\ndata: \"stalled\"\n\n"
                return
            time.sleep(1)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/)[^\s'\"]+")


def _filter_log(line: str, is_admin: bool) -> str:
    """Убирает из лога пути на сервере и трейсбеки.

    Пользователю они бесполезны, а нам подсказывать устройство сервера незачем.
    Администратор видит лог как есть.
    """
    if is_admin:
        return line
    if line.lstrip().startswith(("File \"", "Traceback", "  File")):
        return ""
    return _PATH_RE.sub("[путь]", line)


@app.get("/api/results")
@login_required
def api_results():
    """Клипы ТОЛЬКО текущего пользователя — из диска или из S3."""
    uid = g.user["id"]
    files = storage.list_user_files(uid)

    # Описания клипов лежат в manifest.json рядом с ними.
    manifests: dict[str, dict] = {}
    for folder in {f["folder"] for f in files}:
        raw = storage.read_text(uid, f"{folder}/manifest.json"
                                     if folder != "." else "manifest.json")
        if not raw:
            continue
        try:
            data = json.loads(raw)
            manifests[folder] = {e.get("file"): e for e in data if e.get("file")}
        except Exception:  # noqa: BLE001
            continue

    items = []
    for f in files:
        meta = manifests.get(f["folder"], {}).get(f["name"], {})
        segs = meta.get("segments") or []
        dur = sum(max(0.0, s.get("end", 0) - s.get("start", 0)) for s in segs)
        items.append({
            "name": f["name"], "rel": f["rel"], "folder": f["folder"],
            "size_mb": round(f["size"] / 1024 / 1024, 1),
            "mtime": f["mtime"],
            "virality": meta.get("virality"),
            "title": meta.get("title") or "",
            "hook": meta.get("hook") or "",
            "reason": meta.get("reason") or "",
            "duration": round(dur) if dur else None,
            "parts": len(segs) or None,
        })
    items.sort(key=lambda i: -i["mtime"])
    return jsonify(items[:300])


@app.get("/clips/<path:relpath>")
@login_required
def serve_clip(relpath: str):
    """Отдаёт клип ТЕКУЩЕГО пользователя.

    Чужой файл получить нельзя: и путь на диске, и ключ в хранилище строятся
    от идентификатора вошедшего пользователя, а не из запроса.
    """
    if not relpath.lower().endswith(".mp4") or ".." in relpath:
        return jsonify({"ok": False, "error": "Недопустимый путь"}), 400

    if storage.is_s3():
        # Файл отдаёт хранилище по недолгой подписанной ссылке: приложение
        # не тратит на раздачу ни трафик, ни время.
        parts = relpath.split("/")
        if len(parts) != 2:
            return jsonify({"ok": False, "error": "Недопустимый путь"}), 400
        url = storage.stream_url(g.user["id"], parts[0], parts[1])
        if not url:
            return jsonify({"ok": False, "error": "Файл не найден"}), 404
        return redirect(url)

    try:
        target = paths.resolve_clip(g.user["id"], relpath)
    except ValueError:
        return jsonify({"ok": False, "error": "Недопустимый путь"}), 400
    if not target.is_file() or target.suffix.lower() != ".mp4":
        return jsonify({"ok": False, "error": "Файл не найден"}), 404
    return send_from_directory(target.parent, target.name, conditional=True)


@app.post("/api/delete")
@login_required
def api_delete():
    body = request.get_json(silent=True) or {}
    rels = body.get("rels") or ([body["rel"]] if body.get("rel") else [])
    if not rels:
        return jsonify({"ok": False, "error": "Не указан файл"}), 400

    deleted, errors = [], []
    for rel in rels[:200]:
        if not rel.lower().endswith(".mp4") or ".." in rel:
            errors.append(f"{rel}: недопустимый путь")
            continue
        if storage.delete(g.user["id"], rel):
            deleted.append(rel)
        else:
            errors.append(f"{rel}: файл не найден")
    if deleted:
        store.audit("clips_delete", g.user["email"], "", _client_ip(),
                    f"удалено: {len(deleted)}")
    return jsonify({"ok": not errors, "deleted": deleted, "errors": errors})


@app.post("/api/notify")
@login_required
def api_notify():
    """Включение/отключение писем о готовности."""
    d = request.get_json(silent=True) or {}
    enabled = bool(d.get("enabled"))
    store.execute("UPDATE users SET notify = ? WHERE id = ?",
                  (1 if enabled else 0, g.user["id"]))
    return jsonify({"ok": True, "enabled": enabled})


@app.get("/api/me")
@login_required
def api_me():
    u = g.user
    return jsonify({
        "email": u["email"],
        "is_admin": bool(u["is_admin"]),
        "days_left": store.days_left(u),
        "hours_left": store.hours_left(u),
        "quota_minutes": u["quota_minutes"],
        "used_minutes": round(store.used_minutes(u["id"]), 1),
        "max_concurrent": u["max_concurrent"],
        "contact": CONTACT,
        "notify": bool(u["notify"]),
        "notify_available": notify.is_configured(),
    })


# ===========================================================================
# ВХОД / ВЫХОД
# ===========================================================================

def _contact_href() -> str:
    """Ссылка для связи из CLIPPER_CONTACT.

    Понимает три записи: @имя — телеграм, адрес@почта — письмо, готовый
    https-адрес — как есть. Так владельцу не нужно помнить формат.
    """
    c = CONTACT
    if not c:
        return ""
    if c.startswith("@"):
        return f"https://t.me/{c[1:]}"
    if c.startswith(("http://", "https://")):
        return c
    if "@" in c:
        return f"mailto:{c}"
    return f"https://t.me/{c}"


def _render(template: str) -> Response:
    """Отдаёт страницу.

    В публичном режиме вырезаются блоки <!--LOCAL_ONLY-->: поля для путей
    к файлам на сервере и настройки куков не должны попадать в разметку даже
    скрытыми — скрытое поле по-прежнему можно отправить.
    """
    html = render_template(template, csrf_token=csrf_token(), contact=CONTACT,
                           contact_href=_contact_href(), public_mode=PUBLIC_MODE)
    if PUBLIC_MODE:
        html = re.sub(r"<!--LOCAL_ONLY-->.*?<!--/LOCAL_ONLY-->", "", html, flags=re.S)
    return Response(html, mimetype="text/html")


@app.get("/login")
def login_page():
    if g.get("user"):
        return redirect("/")
    return _render("login.html")


@app.post("/api/login")
def api_login():
    data = request.get_json(silent=True) or request.form
    email = (data.get("email", "") or "").strip().lower()

    wait = _login_blocked(email)
    if wait:
        store.audit("login_throttled", email, "", _client_ip())
        return jsonify({"ok": False, "error": f"Слишком много попыток входа. "
                                              f"Повторите через {wait // 60 + 1} мин."}), 429

    user, err = store.check_login(email, data.get("password", ""))
    if not user:
        _login_failed(email)
        store.audit("login_failed", email, "", _client_ip(), err)
        return jsonify({"ok": False, "error": err}), 401

    _login_ok(email)
    # Новая сессия после входа — защита от фиксации сессии.
    session.clear()
    session.permanent = True
    session["uid"] = user["id"]
    # Метка смены пароля: если пароль поменяют, все прежние сессии перестанут
    # действовать (см. login_required).
    session["pwd"] = user["password_changed_at"]
    csrf_token()
    store.audit("login", email, "", _client_ip())
    return jsonify({"ok": True})


@app.post("/api/logout")
def api_logout():
    if g.get("user"):
        store.audit("logout", g.user["email"], "", _client_ip())
    session.clear()
    return jsonify({"ok": True})


@app.post("/api/password")
@login_required
def api_password():
    """Смена собственного пароля."""
    d = request.get_json(silent=True) or {}
    old, new = d.get("old", ""), d.get("new", "")
    if not store.verify_password(old, g.user["password_hash"]):
        return jsonify({"ok": False, "error": "Текущий пароль неверен"}), 400
    try:
        store.set_password(g.user["email"], new)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    store.audit("password_change", g.user["email"], "", _client_ip())
    session["pwd"] = store.get_user(g.user["id"])["password_changed_at"]
    return jsonify({"ok": True})


# ===========================================================================
# АДМИНКА
# ===========================================================================

@app.get("/api/admin/users")
@login_required
@admin_required
def api_admin_users():
    return jsonify(store.list_users())


@app.post("/api/admin/users")
@login_required
@admin_required
def api_admin_create():
    d = request.get_json(silent=True) or {}
    try:
        store.create_user(
            d.get("email", ""), d.get("password", ""),
            days=int(d.get("days", 30)), note=d.get("note", ""),
            quota_minutes=int(d.get("quota_minutes", 600)),
            max_concurrent=int(d.get("max_concurrent", 1)),
            max_video_minutes=int(d.get("max_video_minutes", 180)))
        store.audit("user_create", g.user["email"], d.get("email", ""), _client_ip())
        return jsonify({"ok": True})
    except (ValueError, TypeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.post("/api/admin/limits")
@login_required
@admin_required
def api_admin_limits():
    """Квоты пользователя: минуты в месяц, одновременных задач, длина видео."""
    d = request.get_json(silent=True) or {}
    try:
        store.set_limits(d.get("email", ""),
                         quota_minutes=d.get("quota_minutes"),
                         max_concurrent=d.get("max_concurrent"),
                         max_video_minutes=d.get("max_video_minutes"),
                         note=d.get("note"),
                         disabled=(1 if d.get("disabled") else 0)
                         if "disabled" in d else None)
        store.audit("user_limits", g.user["email"], d.get("email", ""), _client_ip())
        return jsonify({"ok": True})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.post("/api/admin/extend")
@login_required
@admin_required
def api_admin_extend():
    d = request.get_json(silent=True) or {}
    emails = d.get("emails") or ([d["email"]] if d.get("email") else [])
    if not emails:
        return jsonify({"ok": False, "error": "Не выбран ни один аккаунт"}), 400
    try:
        days = int(d.get("days", 30))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Некорректное число дней"}), 400
    done, errors = [], []
    for em in emails:
        try:
            store.extend_access(em, days)
            done.append(em)
        except ValueError as e:
            errors.append(f"{em}: {e}")
    store.audit("user_extend", g.user["email"], ", ".join(done)[:200],
                _client_ip(), f"дней: {days}")
    return jsonify({"ok": not errors, "done": done, "errors": errors,
                    "count": len(done)})


@app.post("/api/admin/password")
@login_required
@admin_required
def api_admin_password():
    d = request.get_json(silent=True) or {}
    try:
        store.set_password(d.get("email", ""), d.get("password", ""))
        store.audit("user_password_reset", g.user["email"], d.get("email", ""),
                    _client_ip())
        return jsonify({"ok": True})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.post("/api/admin/delete")
@login_required
@admin_required
def api_admin_delete():
    d = request.get_json(silent=True) or {}
    return _delete_users([d.get("email", "")])


@app.post("/api/admin/bulk-delete")
@login_required
@admin_required
def api_admin_bulk_delete():
    d = request.get_json(silent=True) or {}
    return _delete_users(d.get("emails") or [])


def _delete_users(emails: list[str]):
    """Удаляет аккаунты вместе со ВСЕМИ их файлами.

    Удаление должно быть настоящим: аккаунт без файлов оставлял бы на диске
    чужие видео и лица — то есть персональные данные, которые пользователь
    просил удалить.
    """
    if not emails or not any(e.strip() for e in emails):
        return jsonify({"ok": False, "error": "Не выбран ни один аккаунт"}), 400
    done, errors = [], []
    for em in emails:
        em = (em or "").strip().lower()
        if em == g.user["email"]:
            errors.append(f"{em}: нельзя удалить самого себя")
            continue
        try:
            u = store.get_user_by_email(em)
            if not u:
                errors.append(f"{em}: пользователь не найден")
                continue
            uid = u["id"]
            store.delete_user(em)
            for d in (paths.CLIPS / uid, paths.WORK / uid,
                      paths.UPLOADS / uid, paths.LOGS / uid):
                shutil.rmtree(d, ignore_errors=True)
            if storage.is_s3():
                # Иначе в хранилище остались бы чужие видео с лицами и
                # голосами — то есть персональные данные удалённого человека.
                storage.delete_prefix(f"clips/{uid}/")
            done.append(em)
        except ValueError as e:
            errors.append(f"{em}: {e}")
    store.audit("user_delete", g.user["email"], ", ".join(done)[:200], _client_ip())
    return jsonify({"ok": not errors, "done": done, "errors": errors,
                    "count": len(done)})


@app.get("/api/admin/usage")
@login_required
@admin_required
def api_admin_usage():
    try:
        from clipper.usage import summary_db
        return jsonify(summary_db())
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.get("/api/admin/metrics")
@login_required
@admin_required
def api_admin_metrics():
    """Замеры для планирования мощностей: сколько времени занимает каждый
    этап, сколько нужно видеокарт, сколько люди ждут в очереди."""
    try:
        days = max(1, min(90, int(request.args.get("days", 7))))
    except (TypeError, ValueError):
        days = 7
    m = store.metrics(days)
    m["workers_active"] = store.active_workers()
    m["running_now"] = store.running_count()
    return jsonify(m)


@app.get("/healthz")
def healthz():
    """Проверка живости для мониторинга и балансировщика.

    Без авторизации: система мониторинга не логинится. Поэтому наружу отдаём
    только техническое состояние — ни одного имени пользователя или ссылки.
    Отвечает 503, когда очередь есть, а живых воркеров нет: это ровно тот
    сбой, который иначе замечают по жалобам пользователей.
    """
    try:
        depth = store.queue_depth()
        workers = store.active_workers()
        running = store.running_count()
        ok = not (depth > 0 and workers == 0)
        return jsonify({
            "ok": ok,
            "queue": depth,
            "running": running,
            "workers": workers,
            "problem": None if ok else "есть задачи в очереди, но нет воркеров",
        }), (200 if ok else 503)
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "problem": type(e).__name__}), 503


@app.post("/api/admin/usage/reset")
@login_required
@admin_required
def api_admin_usage_reset():
    """Обнуляет статистику расходов.

    Удаляются только записи о запросах к ИИ. Задачи и их история остаются:
    иначе вместе со статистикой пропала бы информация о том, кто и сколько
    обрабатывал, а она нужна для квот.
    """
    r = store.query_one("SELECT COUNT(*) n FROM usage_records")
    n = int(r["n"]) if r else 0
    store.execute("DELETE FROM usage_records")
    store.audit("usage_reset", g.user["email"], "", _client_ip(),
                f"удалено записей: {n}")
    return jsonify({"ok": True, "deleted": n})


@app.get("/api/admin/audit")
@login_required
@admin_required
def api_admin_audit():
    return jsonify(store.list_audit(300))


@app.get("/")
@login_required
def index():
    return _render("index.html")




if __name__ == "__main__":
    paths.ensure_dirs()
    store.init_db()

    if PUBLIC_MODE:
        print("=" * 60)
        print("  ОСТАНОВЛЕНО: публичный режим требует боевого сервера.")
        print()
        print("  Запускайте так (см. DEPLOY.md):")
        print("      gunicorn -c gunicorn.conf.py web:app     # сайт")
        print("      python worker.py                          # обработка")
        print("=" * 60)
        sys.exit(1)

    created = store.ensure_admin_exists()
    print("=" * 60)
    print("  shorts-clipper — веб-панель")
    if created:
        email, password = created
        print()
        print("  СОЗДАН АККАУНТ АДМИНИСТРАТОРА (показывается один раз!):")
        print(f"     почта:  {email}")
        print(f"     пароль: {password}")
        print("  Запишите их. Сменить пароль можно в разделе «Пользователи».")
        print()
    print("  Открой в браузере:  http://127.0.0.1:5000")
    print()
    print("  ВАЖНО: обработку выполняет отдельный процесс. В соседнем окне:")
    print("      python worker.py")
    print()
    print("  Остановить сервер:  Ctrl+C в этом окне")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
