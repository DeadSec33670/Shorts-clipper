"""Проверка исправлений Этапа 1. Запуск: python3 test_security.py"""
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

WORK = Path(tempfile.mkdtemp())
SRC = Path(__file__).resolve().parent
shutil.copytree(SRC, WORK / "app", dirs_exist_ok=True)
os.chdir(WORK / "app")
sys.path.insert(0, str(WORK / "app"))

os.environ["CLIPPER_PUBLIC"] = "1"
os.environ["CLIPPER_SECRET_KEY"] = "test-secret-key-for-checks-only"
os.environ["CLIPPER_ADMIN_EMAIL"] = "admin@test.local"
os.environ["CLIPPER_ADMIN_PASSWORD"] = "SuperSecret123"
os.environ["CLIPPER_DB_URL"] = f"sqlite:///{WORK}/app/sec.db"
os.environ["CLIPPER_DATA_DIR"] = str(WORK / "app" / "data")

import web  # noqa: E402
import store as user_db  # noqa: E402

user_db.init_db()
user_db.ensure_admin_exists()
web.app.config["TESTING"] = True

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name + (f"  <- {detail}" if detail and not cond else ""))


def get_csrf(c, path="/login"):
    html = c.get(path).get_data(as_text=True)
    m = re.search(r'name="csrf-token" content="([^"]+)"' , html)
    return m.group(1) if m else None


print("\n--- CSRF ---")
with web.app.test_client() as c:
    tok = get_csrf(c)
    check("токен выдаётся на странице входа", bool(tok) and tok != "__CSRF_TOKEN__", tok)
    r = c.post("/api/login", json={"email": "admin@test.local", "password": "SuperSecret123"})
    check("вход БЕЗ токена отклоняется", r.status_code == 403, r.status_code)
    r = c.post("/api/login", json={"email": "admin@test.local", "password": "SuperSecret123"},
               headers={"X-CSRF-Token": "подделка"})
    check("вход с ЧУЖИМ токеном отклоняется", r.status_code == 403, r.status_code)
    r = c.post("/api/login", json={"email": "admin@test.local", "password": "SuperSecret123"},
               headers={"X-CSRF-Token": tok})
    check("вход с верным токеном проходит", r.status_code == 200, r.get_data(as_text=True)[:120])

print("\n--- Origin / межсайтовый запрос ---")
with web.app.test_client() as c:
    tok = get_csrf(c)
    c.post("/api/login", json={"email": "admin@test.local", "password": "SuperSecret123"},
           headers={"X-CSRF-Token": tok})
    tok2 = get_csrf(c, "/")
    r = c.post("/api/delete", json={"rel": "x.mp4"},
               headers={"X-CSRF-Token": tok2, "Origin": "https://evil.example.com"})
    check("запрос с чужого Origin отклоняется", r.status_code == 403, r.status_code)

print("\n--- Произвольные пути к файлам (публичный режим) ---")
with web.app.test_client() as c:
    tok = get_csrf(c)
    c.post("/api/login", json={"email": "admin@test.local", "password": "SuperSecret123"},
           headers={"X-CSRF-Token": tok})
    tok2 = get_csrf(c, "/")
    r = c.post("/api/start", data={"paths": "/etc/passwd"}, headers={"X-CSRF-Token": tok2})
    check("поле paths отклоняется",
          r.status_code == 400 and "отключено" in (r.get_json() or {}).get("error", ""),
          f"{r.status_code} {(r.get_json() or {}).get('error','')[:80]}")
    html = c.get("/").get_data(as_text=True)
    check("поле paths отсутствует в разметке", 'name="paths"' not in html)
    check("блок куков отсутствует в разметке", 'name="cookies_file"' not in html)

print("\n--- Валидация аргументов командной строки ---")
with web.app.test_client() as c:
    tok = get_csrf(c)
    c.post("/api/login", json={"email": "admin@test.local", "password": "SuperSecret123"},
           headers={"X-CSRF-Token": tok})
    tok2 = get_csrf(c, "/")
    r = c.post("/api/start", data={"urls": "https://x.test/v", "llm_provider": "злой"},
               headers={"X-CSRF-Token": tok2})
    check("неизвестный провайдер отклоняется", r.status_code == 400, r.status_code)
    r = c.post("/api/start", data={"urls": "https://x.test/v", "font": "--debug-camera"},
               headers={"X-CSRF-Token": tok2})
    check("значение-флаг в поле font отклоняется", r.status_code == 400, r.status_code)
    # cookies в публичном режиме не должны попадать в параметры задачи
    p = web._clean_params({"urls": "https://x.test/v", "cookies_file": "/etc/shadow",
                           "cookies_browser": "chrome"})
    check("cookies не попадают в параметры задачи",
          "cookies_file" not in p and "cookies_browser" not in p, p)

print("\n--- Загрузка файлов ---")
import io  # noqa: E402
with web.app.test_client() as c:
    tok = get_csrf(c)
    c.post("/api/login", json={"email": "admin@test.local", "password": "SuperSecret123"},
           headers={"X-CSRF-Token": tok})
    tok2 = get_csrf(c, "/")
    r = c.post("/api/start", headers={"X-CSRF-Token": tok2},
               data={"files": (io.BytesIO(b"MZ..."), "virus.exe")},
               content_type="multipart/form-data")
    check("недопустимое расширение отклоняется", r.status_code == 400, r.get_data(as_text=True)[:120])
    check("лимит размера запроса задан",
          web.app.config["MAX_CONTENT_LENGTH"] == 2 * 1024 ** 3,
          web.app.config["MAX_CONTENT_LENGTH"])

print("\n--- Ограничение попыток входа ---")
with web.app.test_client() as c:
    tok = get_csrf(c)
    codes = [c.post("/api/login", json={"email": "admin@test.local", "password": "неверный"},
                    headers={"X-CSRF-Token": tok}).status_code for _ in range(9)]
    check("после серии неудач приходит 429", 429 in codes, codes)
    r = c.post("/api/login", json={"email": "admin@test.local", "password": "SuperSecret123"},
               headers={"X-CSRF-Token": tok})
    check("верный пароль тоже блокируется во время лимита", r.status_code == 429, r.status_code)
web._login_attempts.clear()

print("\n--- Обход проверки пути при удалении ---")
import paths as _p
_p.ensure_dirs()
clips = _p.CLIPS.resolve()
clips.mkdir(exist_ok=True)
sibling = clips.parent / "clips_backup"
sibling.mkdir(exist_ok=True)
(sibling / "secret.mp4").write_bytes(b"x")
with web.app.test_client() as c:
    tok = get_csrf(c)
    c.post("/api/login", json={"email": "admin@test.local", "password": "SuperSecret123"},
           headers={"X-CSRF-Token": tok})
    tok2 = get_csrf(c, "/")
    r = c.post("/api/delete", json={"rel": "../clips_backup/secret.mp4"},
               headers={"X-CSRF-Token": tok2})
    check("соседняя папка clips_backup защищена",
          (sibling / "secret.mp4").exists(), r.get_data(as_text=True)[:160])

print("\n--- Заголовки и cookie ---")
with web.app.test_client() as c:
    r = c.get("/login")
    h = r.headers
    check("X-Frame-Options", h.get("X-Frame-Options") == "DENY")
    check("X-Content-Type-Options", h.get("X-Content-Type-Options") == "nosniff")
    check("Content-Security-Policy", "frame-ancestors 'none'" in h.get("Content-Security-Policy", ""))
    check("HSTS в публичном режиме", "max-age" in h.get("Strict-Transport-Security", ""))
    check("cookie Secure", web.app.config["SESSION_COOKIE_SECURE"] is True)
    check("cookie HttpOnly", web.app.config["SESSION_COOKIE_HTTPONLY"] is True)
    check("cookie SameSite", web.app.config["SESSION_COOKIE_SAMESITE"] == "Lax")

print(f"\n===== Пройдено: {len(PASS)}, провалено: {len(FAIL)} =====")
if FAIL:
    print("Провалены:", *FAIL, sep="\n  - ")
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(1 if FAIL else 0)
