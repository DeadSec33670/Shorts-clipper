#!/usr/bin/env python3
"""AI-нарезка шортсов из длинных видео.

Примеры:
  python main.py "https://www.youtube.com/watch?v=XXXX"
  python main.py lecture.mp4 --clips 10 --model medium --no-subs
"""
import argparse
import os
import sys
from pathlib import Path

# На Windows консоль по умолчанию открывается в cp1251/cp866 и падает
# UnicodeEncodeError на любом непривычном символе (→, …, эмодзи и т.п.),
# которые есть в логах пайплайна. Принудительно переключаем stdout/stderr
# на UTF-8 (Python 3.7+); если это почему-то не поддерживается — не падаем.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from clipper.config import Config
from clipper.pipeline import run


VERSION = "5.5 (учёт расхода токенов и себестоимости — раздел в админке)"


def main() -> None:
    print(f"shorts-clipper v{VERSION}")
    p = argparse.ArgumentParser(description="AI-нарезка шортсов из длинных видео")
    p.add_argument("source", nargs="?", default=None,
                   help="YouTube-ссылка или путь к видеофайлу (не нужен при --films)")
    p.add_argument("--films", nargs="+", default=None, metavar="ВИДЕО",
                   help="НЕСКОЛЬКО видео сразу (ссылки и/или пути через пробел). "
                        "С каждого видео берётся --candidates-per-film кандидатов "
                        "(виральность считается до рендера), затем со ВСЕХ видео "
                        "вместе рендерятся только --clips лучших по виральности — "
                        "остальные кандидаты не рендерятся, это экономит время.")
    p.add_argument("--candidates-per-film", type=int, default=10, metavar="N",
                   help="только для --films: сколько кандидатов запрашивать "
                        "с КАЖДОГО видео до отбора лучших (по умолчанию 10)")
    p.add_argument("--clips", type=int, default=15, help="сколько клипов (по умолчанию 15)")
    p.add_argument("--length", choices=["30s", "1m", "2m"], default="1m",
                   help="длина клипов: 30s (15-30с, динамичные), "
                        "1m (30-60с, стандарт), 2m (60-120с, глубокие истории)")
    p.add_argument("--min-sec", type=float, default=None,
                   help="мин. длительность клипа (переопределяет --length)")
    p.add_argument("--max-sec", type=float, default=None,
                   help="макс. длительность клипа (переопределяет --length)")
    p.add_argument("--llm-provider", default="gemini",
                   choices=["gemini", "groq", "openrouter", "claude"],
                   help="провайдер для поиска моментов: gemini/groq/openrouter (бесплатно) или claude (платно)")
    p.add_argument("--llm-model", default=None,
                   help="конкретная модель провайдера (по умолчанию — разумный дефолт для каждого)")
    p.add_argument("--model", default="large-v3-turbo",
                   help="модель Whisper: tiny/base/small/medium/large-v3/"
                        "large-v3-turbo (по умолчанию turbo: качество почти "
                        "как large-v3 при кратно меньшем времени на GPU)")
    p.add_argument("--whisper-device", default=None,
                   choices=["auto", "cpu", "cuda"],
                   help="где распознавать речь: auto (по умолчанию), cpu "
                        "(медленнее, но работает всегда) или cuda")
    p.add_argument("--language", default=None, help="язык (ru/en/…), по умолчанию авто")
    p.add_argument("--font", default="Montserrat", help="шрифт субтитров")
    p.add_argument("--cookies-browser", default=None,
                   choices=["chrome", "firefox", "edge", "brave", "opera", "vivaldi", "chromium"],
                   help="брать куки из этого браузера для обхода антибот-блокировки YouTube")
    p.add_argument("--proxy", default=None,
                   help="прокси для скачивания, напр. socks5://127.0.0.1:1080 "
                        "или http://логин:пароль@хост:порт (нужен, если "
                        "провайдер режет YouTube)")
    p.add_argument("--cookies-file", default=None,
                   help="путь к cookies.txt (экспорт из ЛЮБОГО браузера, "
                        "включая Яндекс.Браузер) — приоритетнее --cookies-browser")
    p.add_argument("--asd", action="store_true",
                   help="ТОЧНЫЙ режим: нейросеть Light-ASD определяет говорящего "
                        "по совпадению губ и звука (нужен PyTorch, медленнее, но качественнее)")
    p.add_argument("--no-subs", action="store_true", help="не прожигать субтитры")
    p.add_argument("--no-hook", action="store_true",
                   help="не показывать заголовок-хук в первые 3.5 секунды клипа")
    p.add_argument("--sub-color", metavar="HEX",
                   help="цвет обычных слов субтитров в HEX, напр. FFFFFF (белый)")
    p.add_argument("--sub-highlight", metavar="HEX",
                   help="цвет подсветки активного слова в HEX, напр. FFD447 (жёлтый)")
    p.add_argument("--debug-camera", action="store_true",
                   help="ОТЛАДКА: рядом с каждым клипом сохранить JSON с тем, что "
                        "нашёл детектор и как встала камера (для диагностики кадрирования)")
    p.add_argument("--static-fit", action="store_true",
                   help="Альтернативный режим БЕЗ умной камеры: весь исходный кадр "
                        "вписывается в 9:16 (letterbox), с лёгкой обрезкой краёв "
                        "(--static-crop), без слежения за людьми. Люди крупнее, "
                        "чем при полном вписывании, но мельче умного кропа.")
    p.add_argument("--static-crop", type=float, default=0.15, metavar="ДОЛЯ",
                   help="Для --static-fit: доля обрезки с КАЖДОГО края (0..0.35). "
                        "0.15 = обрезать 15%% слева и справа (по умолчанию). "
                        "Больше = крупнее люди и тоньше полосы. 0 = весь кадр целиком.")
    p.add_argument("--outdir", default="clips", help="папка для готовых клипов")
    p.add_argument("--workdir", default="work", help="папка для временных файлов")
    args = p.parse_args()

    from clipper.llm_providers import PROVIDER_DEFAULT_MODEL

    LENGTH_PRESETS = {"30s": (15, 30), "1m": (30, 60), "2m": (60, 120)}
    preset_min, preset_max = LENGTH_PRESETS[args.length]

    cfg = Config(
        workdir=Path(args.workdir),
        outdir=Path(args.outdir),
        whisper_model=args.model,
        language=args.language,
        clips_count=args.clips,
        min_clip_sec=args.min_sec if args.min_sec is not None else preset_min,
        max_clip_sec=args.max_sec if args.max_sec is not None else preset_max,
        sub_font=args.font,
        burn_subtitles=not args.no_subs,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model or PROVIDER_DEFAULT_MODEL[args.llm_provider],
    )
    if args.cookies_browser:
        cfg.extra["cookies_browser"] = args.cookies_browser
    if args.cookies_file:
        cfg.extra["cookies_file"] = args.cookies_file
    if args.proxy:
        cfg.extra["proxy"] = args.proxy
    if args.whisper_device:
        cfg.whisper_device = args.whisper_device

    # Прокси должен действовать НЕ ТОЛЬКО на скачивание видео.
    # Запросы к LLM идут по HTTPS напрямую, и там, где сервис не работает
    # с вашим адресом (Gemini отвечает «User location is not supported»),
    # они падают, хотя видео уже скачалось. Переменные HTTP(S)_PROXY
    # понимают все используемые библиотеки: google-genai, openai, requests.
    _proxy = cfg.extra.get("proxy") or os.environ.get("CLIPPER_PROXY")

    # Если прокси не задан вручную, но есть ключ API VPN — получаем адрес
    # автоматически. Так не нужно переписывать пароль при каждой смене.
    if not _proxy:
        try:
            import vpn_api
            if vpn_api.is_configured():
                _proxy = vpn_api.get_proxy()
                if _proxy:
                    cfg.extra["proxy"] = _proxy
                    print(f"[vpn] Прокси получен через API: "
                          f"{vpn_api.describe(_proxy)}")
        except Exception as e:  # noqa: BLE001 — без прокси попробуем напрямую
            print(f"[vpn] Не удалось получить прокси через API: {e}")
    if _proxy:
        for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.setdefault(_var, _proxy)
        # Локальные адреса — мимо прокси
        os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")

    # цвета субтитров: пользователь задаёт привычный HEX (RRGGBB), а ASS хочет
    # &HBBGGRR — порядок байтов обратный, поэтому переворачиваем
    def _hex_to_ass(h: str) -> str:
        h = h.strip().lstrip("#").upper()
        if len(h) != 6 or any(c not in "0123456789ABCDEF" for c in h):
            p.error(f"Цвет должен быть в формате HEX RRGGBB, напр. FFD447 (получено: {h})")
        return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}"

    if args.sub_color:
        cfg.sub_base_color = _hex_to_ass(args.sub_color)
    if args.sub_highlight:
        cfg.sub_highlight_color = _hex_to_ass(args.sub_highlight)
    if args.no_hook:
        cfg.extra["no_hook"] = True

    if args.asd:
        cfg.extra["asd"] = True
    if args.debug_camera:
        cfg.extra["debug_camera"] = True
    if args.static_fit:
        cfg.extra["static_fit"] = True
        cfg.extra["wide_crop_frac"] = max(0.0, min(0.35, args.static_crop))

    if args.films and args.source:
        p.error("Укажи либо одно видео (source), либо --films с несколькими — не то и другое сразу.")
    if not args.films and not args.source:
        p.error("Нужно указать видео: либо source (одна ссылка/файл), либо --films (несколько).")

    if args.films:
        from clipper.pipeline import run_films
        run_films(args.films, cfg, total_clips=args.clips,
                 candidates_per_film=args.candidates_per_film)
    else:
        run(args.source, cfg)


if __name__ == "__main__":
    main()
