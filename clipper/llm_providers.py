"""Единый интерфейс к разным LLM-провайдерам.

Поддерживаются:
  gemini      — Google AI Studio, бесплатно, без карты, до 1500 запр/день,
                контекст 1M токенов. https://aistudio.google.com/apikey
  groq        — бесплатно, без карты, открытые модели на LPU-железе.
                https://console.groq.com/keys
  openrouter  — бесплатно (модели с суффиксом :free), один ключ на много
                моделей. https://openrouter.ai/keys
  claude      — платный Anthropic API (исходный вариант).
                https://console.anthropic.com

Groq и OpenRouter — OpenAI-совместимые endpoint'ы, поэтому оба идут через
один и тот же клиент `openai`, отличаясь только base_url и моделью.
"""
from __future__ import annotations

import os

PROVIDER_ENV = {
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}

PROVIDER_DEFAULT_MODEL = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "claude": "claude-sonnet-4-6",
}

PROVIDER_SIGNUP_URL = {
    "gemini": "https://aistudio.google.com/apikey",
    "groq": "https://console.groq.com/keys",
    "openrouter": "https://openrouter.ai/keys",
    "claude": "https://console.anthropic.com",
}


def get_api_key(provider: str) -> str:
    env_var = PROVIDER_ENV[provider]
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"Не задан {env_var} для провайдера '{provider}'.\n"
            f"Получите бесплатный ключ здесь: {PROVIDER_SIGNUP_URL[provider]}\n"
            f"Затем: export {env_var}=...  (и перезапустите)"
        )
    return key


def get_api_keys(provider: str) -> list[str]:
    """Собирает ВСЕ доступные ключи провайдера для ротации.

    Источники (в порядке приоритета, дубликаты убираются):
      1) {ENV}          — основной (напр. GOOGLE_API_KEY)
      2) {ENV}S         — несколько через запятую (GOOGLE_API_KEYS=k1,k2,k3)
      3) {ENV}_2, _3…   — пронумерованные (GOOGLE_API_KEY_2, _3, …)

    Наличие нескольких ключей позволяет обходить дневные/минутные лимиты:
    при исчерпании квоты одного ключа запрос повторяется следующим.
    """
    env_var = PROVIDER_ENV[provider]
    keys: list[str] = []

    def _add(v: str | None):
        if v:
            for part in v.split(","):
                part = part.strip()
                if part and part not in keys:
                    keys.append(part)

    _add(os.environ.get(env_var))
    _add(os.environ.get(env_var + "S"))       # GOOGLE_API_KEYS=k1,k2
    i = 2
    while True:
        v = os.environ.get(f"{env_var}_{i}")
        if not v:
            break
        _add(v)
        i += 1

    if not keys:
        raise RuntimeError(
            f"Не задан {env_var} для провайдера '{provider}'.\n"
            f"Получите бесплатный ключ здесь: {PROVIDER_SIGNUP_URL[provider]}\n"
            f"Затем: export {env_var}=...  (и перезапустите).\n"
            f"Можно задать несколько ключей для ротации: "
            f"{env_var}_2=..., {env_var}_3=...  или  {env_var}S=key1,key2,key3"
        )
    return keys


_TRANSIENT_MARKERS = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                      "overloaded", "high demand", "rate limit", "quota")
_GEMINI_MODEL_OVERRIDE: str | None = None   # найденная рабочая модель на этот запуск

# Имена моделей у Google меняются часто, старые отключают. Поэтому вместо
# одного жёстко зашитого имени пробуем список: от новых к заведомо рабочим.
# Список нужен и потому, что запрос ListModels может быть недоступен —
# например, когда сервис не обслуживает регион.
_GEMINI_FALLBACKS = (
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
)


def call_llm(provider: str, model: str, system: str, user: str,
             max_tokens: int = 8000, stage: str = "") -> str:
    """Отправляет запрос выбранному провайдеру; временные сбои (перегрузка,
    лимиты) переживает автоповтором с нарастающей паузой.

    stage — метка этапа («моменты», «стыки», «декор») для учёта расходов."""
    import time
    last_err: Exception | None = None
    for attempt in range(1, 5):
        try:
            answer = _dispatch(provider, model, system, user, max_tokens)
            try:
                from .usage import record_call
                record_call(provider, model, system, user, answer, stage)
            except Exception:  # noqa: BLE001 — учёт не должен ломать работу
                pass
            return answer
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if not any(m in msg for m in _TRANSIENT_MARKERS):
                raise
            last_err = e
            if attempt < 4:
                wait = 15 * attempt
                print(f"[llm] Провайдер перегружен (попытка {attempt}/4) — "
                      f"жду {wait} с и повторяю…")
                time.sleep(wait)
    raise RuntimeError(
        f"Провайдер {provider} перегружен даже после 4 попыток ({last_err}). "
        f"Подождите несколько минут или переключитесь: --llm-provider groq"
    ) from last_err


def _dispatch(provider: str, model: str, system: str, user: str, max_tokens: int) -> str:
    if provider == "gemini":
        return _call_gemini(model, system, user, max_tokens)
    if provider in ("groq", "openrouter"):
        return _call_openai_compatible(provider, model, system, user, max_tokens)
    if provider == "claude":
        return _call_claude(model, system, user, max_tokens)
    raise ValueError(f"Неизвестный провайдер: {provider}. "
                      f"Доступны: {', '.join(PROVIDER_ENV)}")


_GEMINI_KEY_IDX = 0                          # индекс текущего рабочего ключа


def _call_gemini(model: str, system: str, user: str, max_tokens: int) -> str:
    try:
        from google import genai
        from google.genai import types
        from google.genai import errors as genai_errors
    except ImportError as e:
        raise RuntimeError("Установите SDK: pip install google-genai") from e

    keys = get_api_keys("gemini")
    global _GEMINI_MODEL_OVERRIDE, _GEMINI_KEY_IDX

    def _make_generate(client):
        def _generate(model_name: str) -> str:
            resp = client.models.generate_content(
                model=model_name,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )
            return resp.text
        return _generate

    # начинаем с последнего рабочего ключа и по кругу пробуем остальные
    order = list(range(len(keys)))
    order = order[_GEMINI_KEY_IDX:] + order[:_GEMINI_KEY_IDX]
    quota_err: Exception | None = None

    # Таймаут запроса. Через медленный или загруженный прокси рукопожатие
    # и ответ не укладываются в значение по умолчанию, и запрос падает
    # с ReadTimeout, хотя прокси исправен.
    _timeout_ms = int(float(os.environ.get("CLIPPER_LLM_TIMEOUT", "180")) * 1000)

    for ki in order:
        client = genai.Client(
            api_key=keys[ki],
            http_options=types.HttpOptions(timeout=_timeout_ms),
        )
        _generate = _make_generate(client)
        model_to_use = _GEMINI_MODEL_OVERRIDE or model
        try:
            result = _generate(model_to_use)
            _GEMINI_KEY_IDX = ki          # запомним рабочий ключ на процесс
            return result
        except Exception as e:
            # Сетевые сбои (в том числе таймаут прокси) — не ошибки API.
            # Разбираем их отдельно, иначе пользователь получает простыню
            # трассировки вместо понятной причины.
            if not isinstance(e, genai_errors.ClientError):
                low = f"{type(e).__name__} {e}".lower()
                if ("timeout" in low or "10060" in low or "timed out" in low
                        or "socks" in low or "proxy" in low):
                    raise RuntimeError(
                        "Не удалось достучаться до Gemini через прокси: "
                        "соединение не отвечает.\n"
                        "Видео при этом скачалось, значит прокси в целом "
                        "работает — но на запросы к ИИ он не отвечает.\n"
                        "ЧТО ДЕЛАТЬ:\n"
                        "  1. Проверьте прокси на нужном адресе:\n"
                        "     curl.exe --socks5-hostname ЛОГИН:ПАРОЛЬ@АДРЕС:ПОРТ "
                        "-s -o NUL -w \"%{http_code}\" "
                        "https://generativelanguage.googleapis.com\n"
                        "  2. Увеличьте ожидание: CLIPPER_LLM_TIMEOUT=300\n"
                        "  3. Проще всего — смените провайдера, Groq работает "
                        "без прокси в большинстве регионов:\n"
                        "     ключ на console.groq.com, затем "
                        "--llm-provider groq"
                    ) from e
                raise
            # квота исчерпана на этом ключе — пробуем следующий
            if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                quota_err = e
                if len(keys) > 1:
                    print(f"[gemini] Ключ #{ki + 1}/{len(keys)} исчерпал квоту — "
                          f"переключаюсь на следующий…")
                continue
            if "location is not supported" in str(e).lower():
                raise RuntimeError(
                    "Google не обслуживает ваш регион (User location is not "
                    "supported).\n"
                    "Запросы к Gemini идут напрямую и не зависят от того, "
                    "открывается ли YouTube.\n"
                    "ЧТО ДЕЛАТЬ:\n"
                    "  1. Укажите прокси — он теперь применяется и к запросам "
                    "к ИИ:\n"
                    "     CLIPPER_PROXY=socks5h://логин:пароль@адрес:порт\n"
                    "     Для SOCKS нужен пакет: pip install \"httpx[socks]\"\n"
                    "  2. Или смените провайдера на другой, он работает шире:\n"
                    "     --llm-provider groq     (ключ: console.groq.com)\n"
                    "     --llm-provider openrouter"
                ) from e
            if e.code != 404:
                raise
            # модель недоступна аккаунту — ищем доступную и повторяем этим же ключом
            print(f"[gemini] Модель '{model_to_use}' недоступна, пробую другие…")
            fallback = None
            for cand in _GEMINI_FALLBACKS:
                if cand == model_to_use:
                    continue
                try:
                    result = _generate(cand)
                except genai_errors.ClientError as e2:
                    if e2.code == 404:
                        continue        # и этой модели нет — следующая
                    raise
                _GEMINI_MODEL_OVERRIDE = cand
                _GEMINI_KEY_IDX = ki
                print(f"[gemini] Использую '{cand}'. Чтобы не искать каждый раз, "
                      f"запускайте с флагом: --llm-model {cand}")
                return result
            # ни одна из известных не подошла — спрашиваем список у сервиса
            fallback = _pick_gemini_model(client)
            _GEMINI_MODEL_OVERRIDE = fallback
            print(f"[gemini] Использую '{fallback}'. Чтобы не искать каждый раз, "
                  f"запускайте с флагом: --llm-model {fallback}")
            result = _generate(fallback)
            _GEMINI_KEY_IDX = ki
            return result

    # все ключи исчерпали квоту — пусть внешний ретрай подождёт и попробует снова
    raise RuntimeError(
        f"Все {len(keys)} ключ(ей) Gemini исчерпали квоту (RESOURCE_EXHAUSTED). "
        f"Добавьте ещё ключ (GOOGLE_API_KEY_2=…) или подождите сброса лимита."
    ) from quota_err


def _pick_gemini_model(client) -> str:
    """Выбирает лучшую доступную модель: flash > flash-lite, новее > старее."""
    candidates = []
    for m in client.models.list():
        name = m.name.removeprefix("models/")
        actions = getattr(m, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        low = name.lower()
        if "flash" not in low:
            continue
        if any(x in low for x in ("image", "live", "audio", "tts", "embedding")):
            continue
        # скоринг: обычный flash приоритетнее lite, более новое имя — выше
        score = (0 if "lite" in low else 100) + (10 if "preview" not in low else 0)
        candidates.append((score, name))
    if not candidates:
        raise RuntimeError(
            "Не нашёл ни одной доступной flash-модели Gemini. "
            "Откройте https://aistudio.google.com и посмотрите список моделей, "
            "затем укажите её явно: --llm-model <имя>"
        )
    # по убыванию score; при равном score — лексикографически последнее имя
    # (у Gemini более новые версии сортируются позже: 2.5 < 3 < 3.1)
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return candidates[0][1]


def _call_openai_compatible(provider: str, model: str, system: str, user: str, max_tokens: int) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Установите SDK: pip install openai") from e

    base_url = {
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }[provider]
    client = OpenAI(api_key=get_api_key(provider), base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def _call_claude(model: str, system: str, user: str, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("Установите SDK: pip install anthropic") from e

    client = anthropic.Anthropic(api_key=get_api_key("claude"))
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")
