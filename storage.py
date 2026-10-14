"""Хранилище готовых клипов: локальный диск или S3-совместимое хранилище.

ЗАЧЕМ ЭТО НУЖНО
Пока сайт и обработка живут на одной машине, локального диска достаточно.
Но как только воркеры начнут подниматься и гаснуть на арендованных машинах
(Vast.ai, RunPod), локальный диск перестаёт работать в принципе: инстанс
исчезает вместе с клипами. Поэтому для автомасштабирования нужен S3.

Ещё два довода: диск арендованной машины дорог и мал, а раздачу файлов лучше
снять с приложения — пусть её несёт хранилище.

ВЫБОР
CLIPPER_STORAGE=local (по умолчанию) или s3.
Для s3 нужен пакет boto3 и переменные CLIPPER_S3_* (см. env.example).
Работает с любым S3-совместимым хранилищем: AWS, Cloudflare R2, Selectel,
Yandex Object Storage, MinIO.

БЕЗОПАСНОСТЬ
Бакет должен быть ЗАКРЫТЫМ. Ссылки выдаются подписанные и недолгие: открытый
бакет означает, что чужие видео с лицами и голосами доступны любому, кто
угадает или подсмотрит адрес.
"""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

import paths

_MODE = (os.environ.get("CLIPPER_STORAGE") or "local").strip().lower()
# Сколько живёт подписанная ссылка. Час: достаточно, чтобы скачать большой
# файл, и мало, чтобы утёкшая ссылка долго оставалась рабочей.
URL_TTL = int(os.environ.get("CLIPPER_S3_URL_TTL", "3600"))

_client = None
_lock = threading.Lock()


def mode() -> str:
    return _MODE


def is_s3() -> bool:
    return _MODE == "s3"


def _s3():
    """Клиент S3. Создаётся один раз и переиспользуется."""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        import boto3                                   # noqa: PLC0415
        from botocore.config import Config             # noqa: PLC0415

        endpoint = (os.environ.get("CLIPPER_S3_ENDPOINT") or "").strip() or None
        _client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=os.environ.get("CLIPPER_S3_REGION") or "auto",
            aws_access_key_id=os.environ.get("CLIPPER_S3_KEY"),
            aws_secret_access_key=os.environ.get("CLIPPER_S3_SECRET"),
            # s3v4 нужен большинству совместимых хранилищ; path-style —
            # для MinIO и части провайдеров, где виртуальные хосты не заведены.
            config=Config(signature_version="s3v4",
                          s3={"addressing_style": os.environ.get(
                              "CLIPPER_S3_ADDRESSING", "path")},
                          retries={"max_attempts": 3, "mode": "standard"}),
        )
        return _client


def _bucket() -> str:
    b = (os.environ.get("CLIPPER_S3_BUCKET") or "").strip()
    if not b:
        raise RuntimeError("Не задан CLIPPER_S3_BUCKET")
    return b


def key_for(user_id: str, job_id: str, filename: str) -> str:
    """Ключ объекта. Совпадает с раскладкой на диске, чтобы одно можно было
    заменить другим без миграции путей."""
    return f"clips/{user_id}/{job_id}/{filename}"


# ---------------------------------------------------------------------------

def upload_job(user_id: str, job_id: str, local_dir: Path,
               remove_local: bool = True) -> int:
    """Переносит готовые клипы задачи в хранилище. -> сколько файлов перенесено.

    При локальном режиме не делает ничего: файлы уже там, где нужно.
    """
    if not is_s3():
        return 0
    if not local_dir.exists():
        return 0

    n = 0
    for f in sorted(local_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(local_dir).as_posix()
        key = key_for(user_id, job_id, rel)
        ctype = ("video/mp4" if f.suffix.lower() == ".mp4"
                 else "application/json" if f.suffix.lower() == ".json"
                 else "application/octet-stream")
        _s3().upload_file(str(f), _bucket(), key,
                          ExtraArgs={"ContentType": ctype})
        n += 1

    if remove_local and n:
        # Локальную копию убираем только после успешной загрузки всех файлов:
        # исключение выше прервёт функцию и файлы останутся на месте.
        shutil.rmtree(local_dir, ignore_errors=True)
    return n


def download_url(user_id: str, job_id: str, filename: str) -> str | None:
    """Подписанная ссылка на скачивание. None — если режим локальный."""
    if not is_s3():
        return None
    key = key_for(user_id, job_id, filename)
    return _s3().generate_presigned_url(
        "get_object",
        Params={"Bucket": _bucket(), "Key": key,
                # Заставляем браузер скачивать, а не проигрывать в новой
                # вкладке, и подставляем осмысленное имя файла.
                "ResponseContentDisposition":
                    f'attachment; filename="{Path(filename).name}"'},
        ExpiresIn=URL_TTL)


def stream_url(user_id: str, job_id: str, filename: str) -> str | None:
    """Ссылка для просмотра в браузере (без принудительного скачивания)."""
    if not is_s3():
        return None
    return _s3().generate_presigned_url(
        "get_object",
        Params={"Bucket": _bucket(),
                "Key": key_for(user_id, job_id, filename)},
        ExpiresIn=URL_TTL)


def list_job_files(user_id: str, job_id: str) -> list[dict]:
    """-> [{'name','size','mtime'}] для одной задачи."""
    if not is_s3():
        out = []
        d = paths.CLIPS / user_id / job_id
        if d.exists():
            for f in d.rglob("*.mp4"):
                st = f.stat()
                out.append({"name": f.name, "size": st.st_size,
                            "mtime": st.st_mtime})
        return out

    prefix = f"clips/{user_id}/{job_id}/"
    out = []
    for page in _s3().get_paginator("list_objects_v2").paginate(
            Bucket=_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(prefix):]
            if name.endswith(".mp4"):
                out.append({"name": name, "size": obj["Size"],
                            "mtime": obj["LastModified"].timestamp()})
    return out


def list_user_files(user_id: str) -> list[dict]:
    """Все клипы пользователя: [{'rel','name','folder','size','mtime'}]."""
    if not is_s3():
        out = []
        base = paths.CLIPS / user_id
        if base.exists():
            for f in base.rglob("*.mp4"):
                st = f.stat()
                out.append({"rel": f.relative_to(base).as_posix(),
                            "name": f.name,
                            "folder": f.parent.relative_to(base).as_posix() or ".",
                            "size": st.st_size, "mtime": st.st_mtime})
        return out

    prefix = f"clips/{user_id}/"
    out = []
    for page in _s3().get_paginator("list_objects_v2").paginate(
            Bucket=_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel.endswith(".mp4"):
                continue
            out.append({"rel": rel, "name": rel.rsplit("/", 1)[-1],
                        "folder": rel.rsplit("/", 1)[0] if "/" in rel else ".",
                        "size": obj["Size"],
                        "mtime": obj["LastModified"].timestamp()})
    return out


def read_text(user_id: str, relkey: str) -> str | None:
    """Читает небольшой текстовый объект (manifest.json)."""
    if not is_s3():
        p = paths.CLIPS / user_id / relkey
        try:
            return p.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return None
    try:
        obj = _s3().get_object(Bucket=_bucket(), Key=f"clips/{user_id}/{relkey}")
        return obj["Body"].read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None


def delete(user_id: str, rel: str) -> bool:
    """Удаляет один файл (rel — путь относительно папки пользователя)."""
    if not is_s3():
        try:
            target = paths.resolve_clip(user_id, rel)
            if target.is_file():
                target.unlink()
                return True
        except Exception:  # noqa: BLE001
            pass
        return False
    try:
        _s3().delete_object(Bucket=_bucket(), Key=f"clips/{user_id}/{rel}")
        return True
    except Exception:  # noqa: BLE001
        return False


def delete_prefix(prefix: str) -> int:
    """Удаляет всё по префиксу — например, при удалении пользователя.

    Удаление должно быть настоящим: аккаунт без файлов оставлял бы в
    хранилище чужие видео, то есть персональные данные.
    """
    if not is_s3():
        p = paths.DATA / prefix
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            return 1
        return 0

    n = 0
    paginator = _s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if not objs:
            continue
        # Удаляем пачками по 1000 — ограничение S3 API.
        for i in range(0, len(objs), 1000):
            _s3().delete_objects(Bucket=_bucket(),
                                 Delete={"Objects": objs[i:i + 1000]})
            n += len(objs[i:i + 1000])
    return n


def delete_older_than(days: int) -> int:
    """Удаляет клипы старше срока хранения (для cleanup.py)."""
    if not is_s3():
        return 0
    import datetime as _dt
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=days)
    old = []
    for page in _s3().get_paginator("list_objects_v2").paginate(
            Bucket=_bucket(), Prefix="clips/"):
        for obj in page.get("Contents", []):
            if obj["LastModified"] < cutoff:
                old.append({"Key": obj["Key"]})
    for i in range(0, len(old), 1000):
        _s3().delete_objects(Bucket=_bucket(),
                             Delete={"Objects": old[i:i + 1000]})
    return len(old)


def check() -> tuple[bool, str]:
    """Проверка настроек при запуске: лучше узнать о проблеме сразу, чем
    после первой обработанной задачи."""
    if not is_s3():
        return True, "локальный диск"
    if _client is None:
        try:
            import boto3  # noqa: F401,PLC0415
        except ImportError:
            return False, "не установлен boto3 (pip install boto3)"
    try:
        _s3().head_bucket(Bucket=_bucket())
        return True, f"S3: бакет {_bucket()} доступен"
    except Exception as e:  # noqa: BLE001
        return False, f"S3 недоступен: {type(e).__name__}: {e}"
