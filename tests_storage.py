"""Проверка режима S3 без настоящего бакета.

boto3 подменяется заглушкой: проверяется, что формируются правильные ключи,
подписанные ссылки и что удаление доходит до хранилища. Работу с живым
бакетом это не заменяет — команды для проверки см. в DEPLOY.md.

Запуск: python tests_storage.py
"""
import os
import shutil
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(tempfile.mkdtemp())
SRC = Path(__file__).resolve().parent
shutil.copytree(SRC, WORK / "app", dirs_exist_ok=True)
os.chdir(WORK / "app")
sys.path.insert(0, str(WORK / "app"))

os.environ["CLIPPER_DB_URL"] = f"sqlite:///{WORK}/app/s.db"
os.environ["CLIPPER_DATA_DIR"] = str(WORK / "app" / "data")
os.environ["CLIPPER_STORAGE"] = "s3"
os.environ["CLIPPER_S3_BUCKET"] = "clips-test"
os.environ["CLIPPER_S3_ENDPOINT"] = "https://s3.test"

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name +
          (f"  <- {detail}" if detail and not cond else ""))


# --- заглушка S3 ------------------------------------------------------------
class FakeS3:
    def __init__(self):
        self.objects = {}      # key -> (bytes, LastModified)
        self.calls = []

    def upload_file(self, local, bucket, key, ExtraArgs=None):  # noqa: N803
        self.calls.append(("upload", key, (ExtraArgs or {}).get("ContentType")))
        self.objects[key] = (Path(local).read_bytes(),
                             datetime.now(timezone.utc))

    def generate_presigned_url(self, op, Params, ExpiresIn):  # noqa: N803
        self.calls.append(("sign", Params["Key"], ExpiresIn))
        disp = Params.get("ResponseContentDisposition", "")
        return (f"https://s3.test/{Params['Bucket']}/{Params['Key']}"
                f"?X-Amz-Expires={ExpiresIn}"
                + ("&disposition=attachment" if "attachment" in disp else ""))

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.calls.append(("delete", Key))
        self.objects.pop(Key, None)

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        for o in Delete["Objects"]:
            self.objects.pop(o["Key"], None)
        self.calls.append(("delete_many", len(Delete["Objects"])))

    def head_bucket(self, Bucket):  # noqa: N803
        return {}

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        data = self.objects[Key][0]
        return {"Body": types.SimpleNamespace(read=lambda: data)}

    def get_paginator(self, _op):
        outer = self

        class P:
            def paginate(self, Bucket, Prefix):  # noqa: N803
                items = [{"Key": k, "Size": len(v[0]), "LastModified": v[1]}
                         for k, v in outer.objects.items() if k.startswith(Prefix)]
                yield {"Contents": items}
        return P()


fake = FakeS3()
import storage  # noqa: E402
storage._client = fake  # noqa: SLF001

import paths  # noqa: E402
import store  # noqa: E402

store.init_db()
paths.ensure_dirs()
store.create_user("s3@test.local", "LongPassword1")
uid = store.get_user_by_email("s3@test.local")["id"]
jid = "a" * 32

print("\n--- Режим ---")
check("режим s3 определён", storage.is_s3(), storage.mode())
ok, msg = storage.check()
check("проверка доступности проходит", ok, msg)

print("\n--- Выгрузка ---")
out = paths.CLIPS / uid / jid
out.mkdir(parents=True, exist_ok=True)
(out / "01_klip.mp4").write_bytes(b"video-data")
(out / "02_klip.mp4").write_bytes(b"video-data-2")
(out / "manifest.json").write_text('[{"file":"01_klip.mp4","title":"Момент"}]',
                                   encoding="utf-8")
n = storage.upload_job(uid, jid, out)
check("выгружены все файлы", n == 3, n)
check("ключ построен верно",
      f"clips/{uid}/{jid}/01_klip.mp4" in fake.objects,
      list(fake.objects)[:2])
ctypes = {c[1]: c[2] for c in fake.calls if c[0] == "upload"}
check("тип содержимого у видео",
      ctypes.get(f"clips/{uid}/{jid}/01_klip.mp4") == "video/mp4")
check("тип содержимого у манифеста",
      ctypes.get(f"clips/{uid}/{jid}/manifest.json") == "application/json")
check("локальная копия удалена после выгрузки", not out.exists())

print("\n--- Ссылки ---")
url = storage.stream_url(uid, jid, "01_klip.mp4")
check("ссылка на просмотр подписана", url and "X-Amz-Expires" in url, url)
check("срок жизни ссылки ограничен", f"X-Amz-Expires={storage.URL_TTL}" in url, url)
dl = storage.download_url(uid, jid, "01_klip.mp4")
check("ссылка на скачивание помечена attachment", "attachment" in dl, dl)

print("\n--- Список файлов ---")
files = storage.list_user_files(uid)
names = sorted(f["name"] for f in files)
check("видны только mp4", names == ["01_klip.mp4", "02_klip.mp4"], names)
check("путь относительный", all(f["rel"].startswith(jid) for f in files), files[:1])
raw = storage.read_text(uid, f"{jid}/manifest.json")
check("манифест читается", raw and "Момент" in raw, raw)

print("\n--- Изоляция ---")
other = "b" * 32
fake.objects[f"clips/{other}/x/secret.mp4"] = (b"other-user-data", datetime.now(timezone.utc))
mine = [f["name"] for f in storage.list_user_files(uid)]
check("чужие файлы не попадают в список", "secret.mp4" not in mine, mine)

print("\n--- Удаление ---")
check("файл удалён", storage.delete(uid, f"{jid}/02_klip.mp4"))
check("его больше нет", f"clips/{uid}/{jid}/02_klip.mp4" not in fake.objects)
check("выход за свою папку отклонён",
      not storage.delete(uid, "../" + other + "/x/secret.mp4")
      or f"clips/{other}/x/secret.mp4" in fake.objects)
removed = storage.delete_prefix(f"clips/{uid}/")
check("удаление пользователя чистит хранилище", removed >= 1, removed)
check("чужие файлы при этом целы", f"clips/{other}/x/secret.mp4" in fake.objects)

print("\n--- Отдача через веб ---")
import re  # noqa: E402

import web  # noqa: E402

web.app.config["TESTING"] = True
fake.objects[f"clips/{uid}/{jid}/01_klip.mp4"] = (b"v", datetime.now(timezone.utc))
c = web.app.test_client()
tok = re.search(r'name="csrf-token" content="([^"]+)"' ,
                c.get("/login").get_data(as_text=True)).group(1)
c.post("/api/login", json={"email": "s3@test.local", "password": "LongPassword1"},
       headers={"X-CSRF-Token": tok})
r = c.get(f"/clips/{jid}/01_klip.mp4")
check("отдаётся переадресацией на хранилище", r.status_code in (302, 303),
      r.status_code)
check("переадресация ведёт на подписанную ссылку",
      "X-Amz-Expires" in (r.headers.get("Location") or ""),
      r.headers.get("Location"))
r = c.get("/clips/../../etc/passwd")
check("выход за пределы отклонён", r.status_code in (400, 404), r.status_code)

print(f"\n===== Пройдено: {len(PASS)}, провалено: {len(FAIL)} =====")
if FAIL:
    print("Провалены:", *FAIL, sep="\n  - ")
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(1 if FAIL else 0)
