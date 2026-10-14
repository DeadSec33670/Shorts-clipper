"""Проверка, что каждая кнопка интерфейса действительно работает.

ЗАЧЕМ ОТДЕЛЬНЫЙ НАБОР
Кнопка может выглядеть исправной и ничего не делать: обработчик есть, запрос
уходит, а маршрута на сервере нет — в консоли 404, на экране тишина. Именно
так сломалась «Сбросить статистику». Здесь каждый адрес, который дёргает
интерфейс, проверяется настоящим запросом.

Запуск: python tests_ui.py
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

os.environ["CLIPPER_DB_URL"] = f"sqlite:///{WORK}/app/ui.db"
os.environ["CLIPPER_DATA_DIR"] = str(WORK / "app" / "data")
os.environ["CLIPPER_ADMIN_EMAIL"] = "admin@ui.local"
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


def client(email, password):
    c = web.app.test_client()
    tok = re.search(r'name="csrf-token" content="([^"]+)"',
                    c.get("/login").get_data(as_text=True)).group(1)
    c.post("/api/login", json={"email": email, "password": password},
           headers={"X-CSRF-Token": tok})
    tok2 = re.search(r'name="csrf-token" content="([^"]+)"',
                     c.get("/").get_data(as_text=True)).group(1)
    c.environ_base["HTTP_X_CSRF_TOKEN"] = tok2
    return c, tok2


admin, atok = client("admin@ui.local", "AdminPassword1")

# ===========================================================================
print("\n--- Каждый адрес из интерфейса существует на сервере ---")

js = (Path("static/js/app.js").read_text(encoding="utf-8")
      + Path("static/js/login.js").read_text(encoding="utf-8"))
called = {p for p in re.findall(r"""fetch\(\s*['"](/[^'"]+)['"]""", js)
          if not p.endswith("/")}      # обрывки склеиваемых адресов — ниже
# адреса, собираемые из кусков: '/api/jobs/' + id + '/log'
# Адреса, склеиваемые из кусков ('/api/jobs/' + id + '/log'), проверяем
# по существующему шаблону маршрута, а не по обрывку строки.
called |= {"/api/jobs/JOBID", "/api/jobs/JOBID/log", "/api/jobs/JOBID/cancel"}

rules = {str(r) for r in web.app.url_map.iter_rules()}


def route_exists(path: str) -> bool:
    if path in rules:
        return True
    # маршруты с параметром: /api/jobs/<job_id>/log и т.п.
    for r in rules:
        pattern = re.sub(r"<[^>]+>", "[^/]+", r)
        if re.fullmatch(pattern, path) or re.fullmatch(pattern, path + "/x"):
            return True
    return False


for path in sorted(called):
    check(f"маршрут существует: {path}", route_exists(path), sorted(rules))

# ===========================================================================
print("\n--- Кнопки обычного пользователя ---")

admin.post("/api/admin/users",
           json={"email": "user@ui.local", "password": "UserPassword1",
                 "days": 30, "quota_minutes": 500},
           headers={"X-CSRF-Token": atok})
user, utok = client("user@ui.local", "UserPassword1")
uid = store.get_user_by_email("user@ui.local")["id"]

check("«Мои данные» загружаются", user.get("/api/me").status_code == 200)
check("список задач загружается", user.get("/api/jobs").status_code == 200)
check("список клипов загружается", user.get("/api/results").status_code == 200)

r = user.post("/api/start", data={"urls": "https://youtu.be/test"},
              headers={"X-CSRF-Token": utok})
jid = (r.get_json() or {}).get("job_id")
check("кнопка «Запустить» создаёт задачу", r.status_code == 200 and jid,
      r.get_data(as_text=True)[:120])

check("состояние задачи читается",
      user.get(f"/api/jobs/{jid}").status_code == 200)
check("кнопка «Отменить» работает",
      user.post(f"/api/jobs/{jid}/cancel",
                headers={"X-CSRF-Token": utok}).status_code == 200)
check("после отмены статус изменился",
      store.get_job(jid)["status"] == "canceled", store.get_job(jid)["status"])

for enabled in (True, False):
    r = user.post("/api/notify", json={"enabled": enabled},
                  headers={"X-CSRF-Token": utok})
    check(f"переключатель писем ({'вкл' if enabled else 'выкл'})",
          r.status_code == 200
          and bool(store.get_user(uid)["notify"]) == enabled)

d = paths.user_clips(uid) / ("c" * 32)
d.mkdir(parents=True, exist_ok=True)
(d / "clip.mp4").write_bytes(b"video")
items = user.get("/api/results").get_json()
check("клип виден в результатах", len(items) == 1, items)
check("клип скачивается",
      user.get(f"/clips/{items[0]['rel']}").status_code == 200)
r = user.post("/api/delete", json={"rel": items[0]["rel"]},
              headers={"X-CSRF-Token": utok})
check("кнопка удаления клипа работает",
      r.status_code == 200 and r.get_json()["ok"], r.get_data(as_text=True)[:120])
check("файл действительно удалён", not (d / "clip.mp4").exists())

r = user.post("/api/password", json={"old": "UserPassword1",
                                     "new": "BrandNewPass1"},
              headers={"X-CSRF-Token": utok})
check("смена своего пароля работает", r.status_code == 200,
      r.get_data(as_text=True)[:120])

# ===========================================================================
print("\n--- Кнопки администратора ---")

check("список пользователей", admin.get("/api/admin/users").status_code == 200)
check("вкладка «Расходы»", admin.get("/api/admin/usage").status_code == 200)
check("метрики", admin.get("/api/admin/metrics").status_code == 200)
check("журнал действий", admin.get("/api/admin/audit").status_code == 200)

store.record_usage(None, None, "gemini", "g", "поиск моментов", 100, 50, 0.001)
before = store.query_one("SELECT COUNT(*) n FROM usage_records")["n"]
r = admin.post("/api/admin/usage/reset", headers={"X-CSRF-Token": atok})
after = store.query_one("SELECT COUNT(*) n FROM usage_records")["n"]
check("кнопка «Сбросить статистику» работает",
      r.status_code == 200 and before > 0 and after == 0, f"{before} -> {after}")

r = admin.post("/api/admin/extend", json={"email": "user@ui.local", "days": 30},
               headers={"X-CSRF-Token": atok})
check("кнопка «+30 дн» работает", r.status_code == 200 and r.get_json()["ok"],
      r.get_data(as_text=True)[:120])

r = admin.post("/api/admin/extend",
               json={"emails": ["user@ui.local"], "days": 7},
               headers={"X-CSRF-Token": atok})
check("массовое продление работает", r.status_code == 200, r.get_data(as_text=True)[:120])

r = admin.post("/api/admin/password",
               json={"email": "user@ui.local", "password": "ResetByAdmin1"},
               headers={"X-CSRF-Token": atok})
check("кнопка «пароль» (сброс) работает", r.status_code == 200)
check("новый пароль действует",
      store.check_login("user@ui.local", "ResetByAdmin1")[0] is not None)

r = admin.post("/api/admin/limits",
               json={"email": "user@ui.local", "quota_minutes": 999},
               headers={"X-CSRF-Token": atok})
check("изменение квоты работает",
      r.status_code == 200
      and store.get_user_by_email("user@ui.local")["quota_minutes"] == 999)

r = admin.post("/api/admin/users",
               json={"email": "temp@ui.local", "password": "TempPassword1",
                     "days": 7},
               headers={"X-CSRF-Token": atok})
check("кнопка «Создать» работает", r.status_code == 200, r.get_data(as_text=True)[:120])

r = admin.post("/api/admin/delete", json={"email": "temp@ui.local"},
               headers={"X-CSRF-Token": atok})
check("кнопка «удалить» работает",
      r.status_code == 200 and store.get_user_by_email("temp@ui.local") is None)

admin.post("/api/admin/users",
           json={"email": "b1@ui.local", "password": "TempPassword1", "days": 7},
           headers={"X-CSRF-Token": atok})
r = admin.post("/api/admin/bulk-delete", json={"emails": ["b1@ui.local"]},
               headers={"X-CSRF-Token": atok})
check("массовое удаление работает",
      r.status_code == 200 and store.get_user_by_email("b1@ui.local") is None)

# ===========================================================================
print("\n--- Обычный пользователь не может в админку ---")
# Пароль дважды менялся выше, и прежняя сессия перестала действовать — это
# сама по себе верная защита. Входим заново, чтобы проверять именно права.
user, utok = client("user@ui.local", "ResetByAdmin1")

for path in ("/api/admin/users", "/api/admin/usage", "/api/admin/metrics",
             "/api/admin/audit"):
    check(f"закрыто: {path}", user.get(path).status_code == 403)
for path in ("/api/admin/usage/reset", "/api/admin/extend",
             "/api/admin/delete", "/api/admin/limits"):
    check(f"закрыто: {path}",
          user.post(path, json={}, headers={"X-CSRF-Token": utok}).status_code == 403)

# ===========================================================================
print("\n--- Элементы на странице ---")

html = admin.get("/").get_data(as_text=True)
for el in ("go", "stopBtn", "log", "progFill", "jobsList", "res",
           "adminBtn", "logoutBtn", "costReset", "usersList", "notifyCb"):
    check(f"элемент #{el} есть в разметке", f'id="{el}"' in html)

# Каждый обработчик из скрипта должен находить свой элемент.
ids_in_html = set(re.findall(r'id="([\w-]+)"', html))
missing = set()
for el_id in re.findall(r"""\$\(['"]#([\w-]+)['"]\)""",
                        Path("static/js/app.js").read_text(encoding="utf-8")):
    if el_id not in ids_in_html:
        missing.add(el_id)
check("скрипт не обращается к несуществующим элементам",
      not missing, sorted(missing))

check("выход работает",
      admin.post("/api/logout", headers={"X-CSRF-Token": atok}).status_code == 200)

print(f"\n===== Пройдено: {len(PASS)}, провалено: {len(FAIL)} =====")
if FAIL:
    print("Провалены:", *FAIL, sep="\n  - ")
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(1 if FAIL else 0)
