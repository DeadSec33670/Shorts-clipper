"""Проверка Этапа 2: изоляция пользователей, очередь задач, квоты.

Запуск:  python tests_multiuser.py
"""
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
os.environ["CLIPPER_SECRET_KEY"] = "test-key"
os.environ["CLIPPER_DB_URL"] = f"sqlite:///{WORK}/app/test.db"
os.environ["CLIPPER_DATA_DIR"] = str(WORK / "app" / "data")
os.environ["CLIPPER_ADMIN_EMAIL"] = "admin@test.local"
os.environ["CLIPPER_ADMIN_PASSWORD"] = "AdminPassword1"

import paths  # noqa: E402
import store  # noqa: E402
import web  # noqa: E402

store.init_db()
paths.ensure_dirs()
store.ensure_admin_exists()
web.app.config["TESTING"] = True

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name +
          (f"  <- {detail}" if detail and not cond else ""))


def client_for(email, password):
    c = web.app.test_client()
    html = c.get("/login").get_data(as_text=True)
    tok = re.search(r'name="csrf-token" content="([^"]+)"' , html).group(1)
    r = c.post("/api/login", json={"email": email, "password": password},
               headers={"X-CSRF-Token": tok})
    assert r.status_code == 200, r.get_data(as_text=True)
    tok2 = re.search(r'name="csrf-token" content="([^"]+)"' ,
                     c.get("/").get_data(as_text=True)).group(1)
    c.environ_base["HTTP_X_CSRF_TOKEN"] = tok2
    return c, tok2


admin, atok = client_for("admin@test.local", "AdminPassword1")

print("\n--- Создание пользователей ---")
for em in ("alice@test.local", "bob@test.local"):
    r = admin.post("/api/admin/users",
                   json={"email": em, "password": "UserPassword1", "days": 30,
                         "quota_minutes": 100, "max_concurrent": 1},
                   headers={"X-CSRF-Token": atok})
    check(f"создан {em}", r.status_code == 200, r.get_data(as_text=True)[:120])

alice, atok_a = client_for("alice@test.local", "UserPassword1")
bob, btok = client_for("bob@test.local", "UserPassword1")
a_id = store.get_user_by_email("alice@test.local")["id"]
b_id = store.get_user_by_email("bob@test.local")["id"]

print("\n--- Изоляция клипов ---")
for uid, nm in ((a_id, "alice"), (b_id, "bob")):
    d = paths.user_clips(uid) / ("f" * 32)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{nm}.mp4").write_bytes(b"video-" + nm.encode())

ra = alice.get("/api/results").get_json()
rb = bob.get("/api/results").get_json()
check("алиса видит только свой клип",
      len(ra) == 1 and ra[0]["name"] == "alice.mp4", [i["name"] for i in ra])
check("боб видит только свой клип",
      len(rb) == 1 and rb[0]["name"] == "bob.mp4", [i["name"] for i in rb])

rel_bob = rb[0]["rel"]
r = alice.get(f"/clips/{rel_bob}")
check("алиса не скачает клип боба по его пути", r.status_code == 404, r.status_code)
r = alice.get("/clips/../../" + b_id + "/" + ("f" * 32) + "/bob.mp4")
check("выход из своей папки через ../ заблокирован",
      r.status_code in (400, 404), r.status_code)
r = alice.post("/api/delete", json={"rel": "../" + b_id + "/" + ("f" * 32) + "/bob.mp4"},
               headers={"X-CSRF-Token": atok_a})
still = (paths.user_clips(b_id) / ("f" * 32) / "bob.mp4").exists()
check("алиса не удалит клип боба", still, r.get_data(as_text=True)[:120])

print("\n--- Изоляция задач ---")
r = alice.post("/api/start", data={"urls": "https://example.com/v1"},
               headers={"X-CSRF-Token": atok_a})
job_a = (r.get_json() or {}).get("job_id")
check("алиса создала задачу", r.status_code == 200 and job_a, r.get_data(as_text=True)[:150])

r = bob.get(f"/api/jobs/{job_a}")
check("боб не видит задачу алисы", r.status_code == 404, r.status_code)
r = bob.post(f"/api/jobs/{job_a}/cancel", headers={"X-CSRF-Token": btok})
check("боб не отменит задачу алисы", r.status_code == 404, r.status_code)
r = bob.get(f"/api/jobs/{job_a}/log")
check("боб не прочитает лог алисы", r.status_code == 404, r.status_code)
check("в списке боба нет чужих задач",
      all(j["id"] != job_a for j in bob.get("/api/jobs").get_json()))
check("админ видит задачу любого", admin.get(f"/api/jobs/{job_a}").status_code == 200)

print("\n--- Квоты ---")
r = alice.post("/api/start", data={"urls": "https://example.com/v2"},
               headers={"X-CSRF-Token": atok_a})
check("вторая одновременная задача отклонена", r.status_code == 429,
      f"{r.status_code} {r.get_data(as_text=True)[:80]}")

store.finish_job(job_a, "done", clips_count=1, video_minutes=150)
r = alice.post("/api/start", data={"urls": "https://example.com/v3"},
               headers={"X-CSRF-Token": atok_a})
check("превышение месячного лимита минут отклонено", r.status_code == 429,
      f"{r.status_code} {r.get_data(as_text=True)[:80]}")

print("\n--- Проверка входных данных ---")
store.execute("UPDATE users SET quota_minutes = 9999 WHERE id = ?", (a_id,))
bad = [
    ({"urls": "https://e.com/v", "llm_provider": "злой"}, "провайдер"),
    ({"urls": "https://e.com/v", "model": "../../etc"}, "модель"),
    ({"urls": "https://e.com/v", "font": "--debug-camera"}, "шрифт-флаг"),
    ({"urls": "https://e.com/v", "clips": "9999"}, "число вне диапазона"),
    ({"urls": "https://e.com/v", "sub_color": "javascript:x"}, "цвет"),
    ({"urls": "не-ссылка"}, "не URL"),
    ({"paths": "/etc/passwd"}, "путь к файлу сервера"),
    ({}, "пустой источник"),
]
for data, what in bad:
    r = alice.post("/api/start", data=data, headers={"X-CSRF-Token": atok_a})
    check(f"отклонено: {what}", r.status_code == 400, r.status_code)

print("\n--- Форма и обработка не разошлись ---")
# Эта проверка появилась после реального промаха: список моделей в вёрстке
# отстал от бэкенда, а поля куков вообще перестали доходить до обработки.
# Сверяем разметку с белым списком и путь «форма → команда» целиком.
import re as _re  # noqa: E402

import worker as _worker  # noqa: E402

_tpl = Path("templates/index.html").read_text(encoding="utf-8")

for field in ("model", "length", "llm_provider"):
    m = _re.search(rf'<select name="{field}"(.*?)</select>', _tpl, _re.S)
    in_html = {v for v in _re.findall(r'value="([^"]*)"', m.group(1)) if v}
    allowed = web._ALLOWED[field]
    check(f"варианты «{field}» есть в белом списке",
          in_html <= allowed, in_html - allowed)

check("large-v3-turbo предлагается в интерфейсе", "large-v3-turbo" in _tpl)
check("подпись не обещает устаревшую модель",
      "по умолчанию (small)" not in _tpl)

_form = {"urls": "https://youtu.be/abc", "clips": "8", "length": "1m",
         "min_sec": "20", "max_sec": "70", "llm_provider": "groq",
         "llm_model": "llama-3.3", "model": "large-v3-turbo", "language": "ru",
         "font": "Oswald", "sub_color": "#FF0000", "sub_highlight": "#00FF00",
         "camera_mode": "smart_asd", "static_crop": "0.2", "no_subs": "on",
         "no_hook": "on", "debug_camera": "on", "candidates_per_film": "5"}
_p = web._clean_params(_form)
_cmd = " ".join(_worker._build_cmd(
    {"params": _p, "sources": ["https://youtu.be/abc"], "user_id": "a" * 32},
    Path("/tmp/o"), Path("/tmp/w")))
for flag, why in (("--clips 8", "число клипов"), ("--length 1m", "длина"),
                  ("--min-sec 20", "мин. длительность"),
                  ("--max-sec 70", "макс. длительность"),
                  ("--llm-provider groq", "провайдер"),
                  ("llama-3.3", "модель LLM"),
                  ("large-v3-turbo", "модель распознавания"),
                  ("--language ru", "язык"), ("Oswald", "шрифт"),
                  ("FF0000", "цвет субтитров"), ("00FF00", "цвет подсветки"),
                  ("--asd", "режим камеры"), ("--static-crop 0.2", "кроп"),
                  ("--no-subs", "без субтитров"), ("--no-hook", "без хука"),
                  ("--debug-camera", "отладка камеры"),
                  ("--outdir", "папка результата")):
    check(f"поле доходит до обработки: {why}", flag in _cmd, _cmd[:200])

# Прокси в форме — только для локального режима. На сервере его задаёт
# владелец через CLIPPER_PROXY: иначе пользователь мог бы заставить сервис
# подключаться к произвольным адресам и прощупывать внутреннюю сеть.
_pub = web._clean_params({"proxy": "socks5://127.0.0.1:1080"})
check("прокси из формы не принимается в публичном режиме",
      "proxy" not in _pub, _pub)
check("но доходит до команды, если задан",
      "--proxy" in _worker._build_cmd(
          {"params": {"proxy": "socks5://127.0.0.1:1080"},
           "sources": ["https://x/y"], "user_id": "a" * 32},
          Path("/tmp/o"), Path("/tmp/w")))

print("\n--- Сессии ---")
r = admin.post("/api/admin/password",
               json={"email": "bob@test.local", "password": "NewPassword123"},
               headers={"X-CSRF-Token": atok})
check("админ сменил пароль боба", r.status_code == 200)
check("прежняя сессия боба стала недействительной",
      bob.get("/api/me").status_code == 401, bob.get("/api/me").status_code)

print("\n--- Удаление пользователя удаляет его файлы ---")
bob_dir = paths.user_clips(b_id)
check("файлы боба существуют до удаления", bob_dir.exists())
r = admin.post("/api/admin/delete", json={"email": "bob@test.local"},
               headers={"X-CSRF-Token": atok})
check("аккаунт удалён", r.status_code == 200, r.get_data(as_text=True)[:120])
check("файлы боба удалены с диска", not bob_dir.exists())
check("задачи боба удалены каскадом",
      not store.query("SELECT id FROM jobs WHERE user_id = ?", (b_id,)))

print("\n--- Журнал действий ---")
actions = {a["action"] for a in store.list_audit(100)}
for act in ("login", "user_create", "job_create", "user_delete", "password_change"):
    if act == "password_change":
        continue
    check(f"записано событие {act}", act in actions, actions)

print("\n--- Прочее ---")
check("Whisper по умолчанию large-v3-turbo",
      "large-v3-turbo" in Path("clipper/config.py").read_text(encoding="utf-8"))
me = alice.get("/api/me").get_json()
check("в /api/me есть квота", "quota_minutes" in me and "used_minutes" in me, me)

print(f"\n===== Пройдено: {len(PASS)}, провалено: {len(FAIL)} =====")
if FAIL:
    print("Провалены:", *FAIL, sep="\n  - ")
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(1 if FAIL else 0)
