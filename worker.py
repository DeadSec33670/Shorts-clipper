"""Воркер: берёт задачи из очереди и обрабатывает видео.

Запуск (по одному процессу на видеокарту):
    python worker.py

Веб-сервер больше НЕ запускает обработку сам. Он только кладёт задачу в
таблицу jobs, а воркер её забирает. Это и снимает главное ограничение прежней
версии — «обработка уже идёт», — и позволяет держать веб и обработку на разных
машинах: достаточно общей базы и общего хранилища файлов.

Изоляция: каждой задаче выдаются свои --workdir и --outdir внутри папки
пользователя, поэтому пользователи физически не могут увидеть чужие файлы.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import notify  # noqa: E402
import storage  # noqa: E402
import store  # noqa: E402
from clipper.progress import ProgressTracker, estimate_seconds_left  # noqa: E402
from paths import job_dirs, job_log_path  # noqa: E402

ROOT = Path(__file__).resolve().parent
POLL_SEC = float(os.environ.get("CLIPPER_POLL_SEC", "3"))
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"

_stop = threading.Event()


def _log(msg: str) -> None:
    print(f"[worker {WORKER_ID}] {msg}", flush=True)


def _build_cmd(job: dict, out_dir: Path, work_dir: Path) -> list[str]:
    """Собирает команду main.py. Значения уже проверены в web.py при создании
    задачи, здесь только подстановка."""
    p = job["params"]
    cmd = [sys.executable, "main.py"]

    sources = job["sources"]
    if len(sources) > 1 or p.get("batch"):
        cmd += ["--films", *sources]
        if p.get("candidates_per_film"):
            cmd += ["--candidates-per-film", str(int(p["candidates_per_film"]))]
    else:
        cmd += [sources[0]]

    for flag, key in (("--clips", "clips"), ("--length", "length"),
                      ("--min-sec", "min_sec"), ("--max-sec", "max_sec"),
                      ("--llm-provider", "llm_provider"), ("--llm-model", "llm_model"),
                      ("--model", "model"), ("--language", "language"),
                      ("--font", "font"), ("--sub-color", "sub_color"),
                      ("--sub-highlight", "sub_highlight"),
                      ("--static-crop", "static_crop"),
                      # Куки принимаются только в локальном режиме (см. web.py),
                      # поэтому здесь достаточно просто их передать.
                      ("--cookies-browser", "cookies_browser"),
                      ("--cookies-file", "cookies_file"),
                      ("--proxy", "proxy"),
                      ("--whisper-device", "whisper_device")):
        v = p.get(key)
        if v not in (None, ""):
            cmd += [flag, str(v)]

    for flag, key in (("--asd", "asd"), ("--static-fit", "static_fit"),
                      ("--no-subs", "no_subs"), ("--no-hook", "no_hook"),
                      ("--debug-camera", "debug_camera")):
        if p.get(key):
            cmd += [flag]

    cmd += ["--outdir", str(out_dir), "--workdir", str(work_dir)]
    return cmd


def _run_job(job: dict) -> None:
    job_id = job["id"]
    out_dir, work_dir = job_dirs(job["user_id"], job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_log_path(job["user_id"], job_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_cmd(job, out_dir, work_dir)
    _log(f"задача {job_id[:8]} → {' '.join(cmd[:4])}…")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["CLIPPER_JOB_ID"] = job_id
    env["CLIPPER_USER_ID"] = job["user_id"]

    proc = None
    try:
        with open(log_path, "w", encoding="utf-8", buffering=1) as lf:
            lf.write(f"Задача принята в работу {time.strftime('%H:%M:%S')}\n")
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                encoding="utf-8", errors="replace", env=env,
                # Своя группа процессов: иначе при отмене останутся висеть
                # дочерние ffmpeg и yt-dlp.
                start_new_session=(os.name != "nt"),
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                               if os.name == "nt" else 0),
            )

            last_beat = 0.0
            last_prog = 0.0
            tracker = ProgressTracker()
            tracker._set("starting")
            started = time.time()
            store.set_progress(job_id, 3, "starting", "")

            # Замер длительности этапов: на нём строится расчёт мощностей.
            stage_times: dict[str, float] = {}
            cur_stage, cur_since = tracker.stage, started

            # Сколько задача пролежала в очереди — главный показатель качества
            # для пользователя.
            waited = 0.0
            created = store._parse(job.get("created_at"))  # noqa: SLF001
            if created:
                from datetime import datetime as _dt, timezone as _tz
                waited = max(0.0, (_dt.now(_tz.utc) - created).total_seconds())
            store.set_wait_seconds(job_id, waited)

            for line in proc.stdout:
                lf.write(line)
                now = time.time()

                # Прогресс пишем в базу не чаще раза в 2 секунды: обновление
                # на каждой строке лога — это сотни лишних записей.
                changed = tracker.feed(line)
                if changed and tracker.stage != cur_stage:
                    stage_times[cur_stage] = round(
                        stage_times.get(cur_stage, 0) + now - cur_since, 1)
                    cur_stage, cur_since = tracker.stage, now
                if changed and now - last_prog > 2:
                    last_prog = now
                    store.set_progress(job_id, tracker.percent,
                                       tracker.stage, tracker.detail)
                if now - last_beat > 10:          # отметка «жив» + проверка отмены
                    last_beat = now
                    if not store.heartbeat(job_id):
                        lf.write("\n=== Отменено пользователем ===\n")
                        _terminate(proc)
                        store.finish_job(job_id, "canceled")
                        return
                if _stop.is_set():
                    lf.write("\n=== Воркер останавливается, задача вернётся в очередь ===\n")
                    _terminate(proc)
                    store.execute("""UPDATE jobs SET status='queued', worker_id=NULL,
                                     started_at=NULL, heartbeat_at=NULL
                                     WHERE id = ?""", (job_id,))
                    return

            code = proc.wait()
            stage_times[cur_stage] = round(
                stage_times.get(cur_stage, 0) + time.time() - cur_since, 1)
            store.set_stage_times(job_id, stage_times)
            lf.write("\n[время этапов] "
                     + ", ".join(f"{k}: {v:.0f}с" for k, v in stage_times.items())
                     + "\n")
            store.set_progress(job_id, 100 if code == 0 else tracker.percent,
                               "done" if code == 0 else tracker.stage, "")
            clips = len(list(out_dir.rglob("*.mp4")))
            if code == 0:
                # Выгрузка в объектное хранилище. Делается до отметки «готово»:
                # иначе пользователь увидел бы готовую задачу, а файлов ещё нет.
                if storage.is_s3() and clips:
                    try:
                        n = storage.upload_job(job["user_id"], job_id, out_dir)
                        lf.write(f"[хранилище] выгружено файлов: {n}\n")
                    except Exception as e:  # noqa: BLE001
                        lf.write(f"[хранилище] ошибка выгрузки: {type(e).__name__}\n")
                        store.finish_job(job_id, "error",
                                         error="Не удалось сохранить клипы. "
                                               "Попробуйте ещё раз.")
                        return
                lf.write(f"\n=== Готово. Клипов: {clips} ===\n")
                store.finish_job(job_id, "done", clips_count=clips,
                                 video_minutes=_source_minutes(work_dir))
            else:
                tail = _tail(log_path, 4000)
                msg = _user_error(tail)
                if _is_transient(tail) and store.retry_job(job_id, msg):
                    attempt = store.get_job(job_id)
                    n = (attempt or {}).get("attempts", 1)
                    lf.write(f"\n=== Временный сбой (попытка {n} из "
                             f"{store.MAX_ATTEMPTS}). Повторим автоматически ===\n")
                    _log(f"задача {job_id[:8]}: временный сбой, поставлена на повтор")
                    return
                lf.write(f"\n=== Ошибка (код {code}) ===\n")
                store.finish_job(job_id, "error",
                                 error=msg, clips_count=clips,
                                 video_minutes=_source_minutes(work_dir))
    except Exception as e:  # noqa: BLE001
        _log(f"сбой задачи {job_id[:8]}: {type(e).__name__}: {e}")
        store.finish_job(job_id, "error", error=f"Внутренняя ошибка: {type(e).__name__}")
    finally:
        if proc and proc.poll() is None:
            _terminate(proc)
        _cleanup_work(work_dir)
        _notify_user(job_id)


def _notify_user(job_id: str) -> None:
    """Письмо о завершении — если почта настроена и пользователь не отказался.

    Отправляется в самом конце и только один раз: mark_notified защищает от
    повторов при перезапуске воркера.
    """
    if not notify.is_configured():
        return
    try:
        job = store.get_job(job_id)
        if not job or job["status"] not in ("done", "error"):
            return          # отменённые и повторяемые не тревожим
        user = store.get_user(job["user_id"])
        if not user or not user.get("notify"):
            return
        if not store.mark_notified(job_id):
            return
        notify.job_finished(user["email"], job.get("title", ""), job["status"],
                            int(job.get("clips_count") or 0),
                            job.get("error") or "")
    except Exception as e:  # noqa: BLE001 — письмо не важнее клипов
        _log(f"уведомление не отправлено: {type(e).__name__}: {e}")


def _terminate(proc) -> None:
    """Гасит процесс вместе с потомками (ffmpeg, yt-dlp)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _source_minutes(work_dir: Path) -> float:
    """Длительность исходников — для учёта квоты. Считаем через ffprobe;
    если не вышло, квота просто не пополнится (в пользу пользователя)."""
    total = 0.0
    for src in work_dir.rglob("source.mp4"):
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(src)],
                capture_output=True, text=True, timeout=60)
            total += float((r.stdout or "0").strip() or 0)
        except Exception:  # noqa: BLE001
            pass
    return round(total / 60.0, 2)


def _tail(path: Path, chars: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-chars:]
    except Exception:  # noqa: BLE001
        return ""


# Признаки временного сбоя: причина снаружи и, скорее всего, пройдёт сама.
# Повторять имеет смысл только их — задача без распознаваемой речи и на третий
# раз останется без речи, а время видеокарты будет потрачено зря.
_TRANSIENT = (
    "sign in to confirm",        # YouTube просит подтверждение
    "http error 429", "too many requests", "rate limit", "quota",
    "temporary failure", "timed out", "timeout", "connection reset",
    "connection aborted", "connection error", "network is unreachable",
    "unable to download", "fragment not found", "giving up after",
    "service unavailable", "502 bad gateway", "503",
    "cuda out of memory", "out of memory",   # карту мог занять сосед
    "no space left on device",
)

# Это устойчивые причины: повтор ничего не изменит.
_PERMANENT = (
    "нет распознаваемой речи", "пустой транскрипт",
    "подходящих моментов не найдено", "llm не нашла",
    "unsupported", "invalid data found", "файл не найден",
    "is not a valid url", "unsupported url",
)


def _is_transient(tail: str) -> bool:
    t = tail.lower()
    for marker in _PERMANENT:
        if marker in t:
            return False
    return any(marker in t for marker in _TRANSIENT)


def _user_error(tail: str) -> str:
    """Превращает трейсбек в понятную фразу.

    Пользователю нельзя показывать пути на сервере и внутренности кода — это
    и непонятно, и подсказывает устройство системы. Технические подробности
    остаются в файле лога, доступном администратору.
    """
    t = tail.lower()
    known = [
        ("sign in to confirm", "YouTube требует подтверждения и не отдаёт видео. "
                               "Попробуйте позже или загрузите файл вручную."),
        ("нет распознаваемой речи", "В видео не нашлось распознаваемой речи."),
        ("пустой транскрипт", "В видео не нашлось распознаваемой речи."),
        ("llm не нашла", "Не удалось выделить подходящие фрагменты в этом видео."),
        ("не задан", "Сервис временно не настроен. Сообщите администратору."),
        ("quota", "Исчерпан лимит внешнего сервиса. Попробуйте позже."),
        ("rate limit", "Исчерпан лимит внешнего сервиса. Попробуйте позже."),
        ("out of memory", "Не хватило памяти видеокарты. Попробуйте видео короче."),
        ("файл не найден", "Исходный файл не найден."),
        ("unsupported", "Формат файла не поддерживается."),
    ]
    for marker, msg in known:
        if marker in t:
            return msg
    return ("Обработка завершилась ошибкой. Технические подробности сохранены "
            "для администратора.")


def _cleanup_work(work_dir: Path) -> None:
    """Удаляет исходное видео сразу после обработки.

    Исходники — самые тяжёлые файлы и самые чувствительные данные (чужие лица
    и голоса). Хранить их незачем: результат уже получен. Побочный эффект —
    повторный запуск того же видео пойдёт без кеша, это осознанный размен.
    """
    try:
        for src in work_dir.rglob("source.mp4"):
            src.unlink(missing_ok=True)
        for pat in ("*.wav", "*.ass", "part_*.mp4", "*.npy"):
            for f in work_dir.rglob(pat):
                f.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    store.init_db()
    ok, msg = storage.check()
    _log(f"хранилище — {msg}")
    if not ok:
        _log("остановлен: хранилище не настроено")
        sys.exit(1)
    _log("запущен, жду задачи")

    def _sig(_signum, _frame):
        _log("получен сигнал остановки — завершаю текущую задачу")
        _stop.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    last_sweep = 0.0
    while not _stop.is_set():
        try:
            # Раз в минуту возвращаем в очередь задачи умерших воркеров
            if time.time() - last_sweep > 60:
                last_sweep = time.time()
                n = store.requeue_stale_jobs()
                if n:
                    _log(f"вернул в очередь зависших задач: {n}")

            job = store.claim_job(WORKER_ID)
            if not job:
                _stop.wait(POLL_SEC)
                continue
            _run_job(job)
        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001 — воркер не должен падать насовсем
            _log(f"ошибка цикла: {type(e).__name__}: {e}")
            _stop.wait(5)

    _log("остановлен")


if __name__ == "__main__":
    main()
