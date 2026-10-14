"""Шаг 1. Получение исходного видео: YouTube-ссылка или локальный файл."""
from __future__ import annotations

import shutil
import re
from pathlib import Path

from .config import Config


def obtain_video(source: str, cfg: Config) -> Path:
    """Возвращает путь к mp4-файлу с исходным видео.

    source — либо URL (http/https), либо путь к локальному файлу.
    """
    cfg.ensure_dirs()
    target = cfg.workdir / "source.mp4"

    if source.startswith(("http://", "https://")):
        return _download(source, target, cfg)

    src = Path(source).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Файл не найден: {src}")
    if src.suffix.lower() == ".mp4":
        shutil.copy2(src, target)
    else:
        from .render import run_ffmpeg
        # Сначала пробуем БЫСТРУЮ переупаковку без перекодирования (-c copy):
        # большинство рипов (в т.ч. типичные WEB-DL .mkv) уже используют
        # h264/h265 + aac/ac3 — эти дорожки можно просто переложить в mp4
        # за секунды, вместо полного перекодирования всего файла (может
        # занимать десятки минут на CPU и НЕ зависит от длительности видео,
        # а зависит от битрейта/качества исходника — отсюда ощущение, что
        # «локальный файл обрабатывается дольше, чем более длинный с YouTube»).
        print(f"[download] Локальный файл не mp4 ({src.suffix}) — пробую "
              f"быструю переупаковку без перекодирования…")
        try:
            run_ffmpeg(["-i", str(src), "-c", "copy",
                       "-movflags", "+faststart", str(target)])
            print("[download] Переупаковка без перекодирования удалась (быстро).")
        except RuntimeError as e:
            print(f"[download] Быстрая переупаковка не подошла для этого файла "
                  f"({type(e).__name__}) — перекодирую полностью (медленнее, "
                  f"но надёжно для любого кодека)…")
            run_ffmpeg(["-i", str(src), "-c:v", "libx264",
                        "-preset", cfg.video_preset, "-crf", str(cfg.video_crf),
                        "-c:a", "aac", "-movflags", "+faststart", str(target)])
    return target


def _download(url: str, target: Path, cfg: Config | None = None) -> Path:
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("Установите yt-dlp: pip install yt-dlp") from e

    import os

    opts = {
        # до 1080p достаточно: исходник всё равно кропается до 1080x1920
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(target.with_suffix("")) + ".%(ext)s",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        # Таймаут по умолчанию у yt-dlp — 20 секунд. Там, где доступ к YouTube
        # замедляют, страница не успевает прийти, и скачивание падает с
        # Read timed out ещё до начала загрузки видео.
        "socket_timeout": float(os.environ.get("CLIPPER_SOCKET_TIMEOUT", "60")),
        "retries": int(os.environ.get("CLIPPER_DL_RETRIES", "10")),
        "fragment_retries": int(os.environ.get("CLIPPER_DL_RETRIES", "10")),
        "extractor_retries": 5,
        # Клиенты YouTube API. Ссылки на видео у веб-клиента привязаны к
        # адресу, который их запросил: через VPN или прокси адрес может
        # смениться, и раздача обрывается с 403 Forbidden на первых процентах.
        # Клиенты android/ios таких проверок не делают, поэтому пробуем их.
        "extractor_args": {
            "youtube": {
                "player_client": (os.environ.get("CLIPPER_YT_CLIENTS")
                                  or "default,android,ios,web_safari").split(",")
            }
        },
        # Скачивание одним потоком. При многопоточном скачивании каждый кусок
        # запрашивается отдельно, и вероятность поймать 403 в середине выше.
        "concurrent_fragment_downloads": 1,
        # Паузы между попытками: без них десять попыток подряд упрутся
        # в ту же самую блокировку за пару секунд.
        "retry_sleep_functions": {
            "http": lambda n: min(30, 2 ** n),
            "fragment": lambda n: min(15, 2 ** n),
            "extractor": lambda n: min(30, 5 * n),
        },
    }

    # Прокси. Основное лекарство, когда провайдер режет YouTube: сам сервис
    # работает, но соединение не устанавливается или рвётся по таймауту.
    proxy = (cfg.extra.get("proxy") if cfg else None) or os.environ.get("CLIPPER_PROXY")
    if proxy:
        opts["proxy"] = proxy
        # В логах прокси печатаем без пароля: логи читают и пересылают.
        safe = re.sub(r"//[^@/]*@", "//***@", proxy)
        print(f"[download] Через прокси: {safe}")

    # Обход блокировки "Sign in to confirm you're not a bot" (HTTP 429):
    # либо куки напрямую из поддерживаемого браузера, либо из файла
    # cookies.txt — второй способ работает с ЛЮБЫМ браузером, включая
    # Яндекс.Браузер, который yt-dlp не умеет читать напрямую.
    browser = cfg.extra.get("cookies_browser") if cfg else None
    cookies_file = cfg.extra.get("cookies_file") if cfg else None
    if cookies_file:
        opts["cookiefile"] = cookies_file
        print(f"[download] Использую куки из файла: {cookies_file}")
    elif browser:
        opts["cookiesfrombrowser"] = (browser,)
        print(f"[download] Использую куки из браузера: {browser}")

    # При 403 перебираем клиентов YouTube ПООЧЕРЁДНО, каждого своей попыткой.
    # Список в одном запросе не помогает: yt-dlp берёт первого подходящего для
    # получения сведений о видео и при отказе на скачивании к остальным уже
    # не возвращается. Ссылки, выданные разным клиентам, проверяются
    # по-разному, поэтому смена клиента часто и решает проблему.
    clients = [c.strip() for c in
               (os.environ.get("CLIPPER_YT_CLIENTS")
                or "default,android_vr,ios,android,tv,web_safari").split(",")
               if c.strip()]

    last_error: Exception | None = None
    for i, client in enumerate(clients, 1):
        attempt = dict(opts)
        attempt["extractor_args"] = {"youtube": {"player_client": [client]}}
        if i > 1:
            print(f"[download] Пробую клиент YouTube: {client} "
                  f"({i} из {len(clients)})…")
        try:
            with yt_dlp.YoutubeDL(attempt) as ydl:
                ydl.download([url])
            last_error = None
            break
        except yt_dlp.utils.DownloadError as e:
            low_e = str(e).lower()
            last_error = e
            # Меняем клиента только при отказе в доступе. Остальные причины
            # (нет сети, видео удалено) от смены клиента не изменятся.
            if "403" in str(e) or "forbidden" in low_e or "unable to download video data" in low_e:
                # Частично скачанные куски мешают следующей попытке
                for junk in target.parent.glob(target.stem + "*.part"):
                    junk.unlink(missing_ok=True)
                continue
            break

    try:
        if last_error is not None:
            raise last_error
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        low = msg.lower()
        if ("timed out" in low or "timeout" in low
                or "connection reset" in low or "network is unreachable" in low
                or "failed to establish" in low):
            raise RuntimeError(
                "Не удалось связаться с YouTube: соединение обрывается "
                "по таймауту.\n"
                "Это происходит вне программы — сам YouTube недоступен "
                "с этого подключения.\n"
                "ЧТО ДЕЛАТЬ:\n"
                "  1. Проверьте, открывается ли youtube.com в браузере "
                "на этой же машине.\n"
                "  2. Если открывается только через VPN — укажите прокси:\n"
                "     переменная CLIPPER_PROXY=socks5://127.0.0.1:1080\n"
                "     (или http://логин:пароль@хост:порт)\n"
                "  3. Увеличьте таймаут: CLIPPER_SOCKET_TIMEOUT=120\n"
                "  4. Либо скачайте видео вручную и загрузите файлом."
            ) from e
        if "403" in msg or "forbidden" in low:
            raise RuntimeError(
                "YouTube оборвал скачивание (403 Forbidden).\n"
                "Обычно это значит, что исходящий адрес сменился в середине "
                "загрузки: ссылка на видео у YouTube привязана к тому адресу, "
                "который её запросил.\n"
                "ЧТО ДЕЛАТЬ:\n"
                "  1. В клиенте VPN отключите смену серверов и балансировку — "
                "нужен один постоянный сервер на всё скачивание.\n"
                "  2. Попробуйте режим ПРОКСИ вместо ТУННЕЛЯ (или наоборот).\n"
                "  3. Задайте другой клиент YouTube:\n"
                "     CLIPPER_YT_CLIENTS=android,ios\n"
                "  4. Обновите yt-dlp: pip install -U yt-dlp\n"
                "Задача будет повторена автоматически."
            ) from e
        if "Sign in to confirm" in msg or "429" in msg or "not a bot" in msg:
            raise RuntimeError(
                "YouTube временно блокирует скачивание с вашего IP "
                "(слишком много запросов / антибот-проверка).\n"
                "РЕШЕНИЕ 1 — если браузер Chrome/Firefox/Edge/Brave/Opera:\n"
                "    --cookies-browser chrome\n"
                "РЕШЕНИЕ 2 — для ЛЮБОГО браузера (в т.ч. Яндекс.Браузер):\n"
                "    экспортируйте куки youtube.com в файл cookies.txt "
                "расширением вида 'Get cookies.txt LOCALLY', затем:\n"
                "    --cookies-file cookies.txt\n"
                "Если не поможет сразу — подождите 10-30 минут (лимит IP "
                "сбрасывается) и повторите. Также помогает включить VPN."
            ) from e
        raise

    if not target.exists():
        # yt-dlp мог сохранить с другим расширением до merge — ищем результат
        candidates = sorted(target.parent.glob("source.*"))
        if not candidates:
            raise RuntimeError("Скачивание не удалось: файл не найден")
        candidates[0].rename(target)
    return target
