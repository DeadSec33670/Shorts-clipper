"""Получение прокси через API StealthSurf.

ЗАЧЕМ
Программе нужен обычный SOCKS5-прокси, а VPN работает по протоколу VLESS,
который она использовать не умеет. Но у сервиса есть «дополнительный
конфиг» — бесплатный HTTP/SOCKS5-прокси, привязанный к основной подписке.
Этот модуль получает его через API и подставляет автоматически, чтобы не
приходилось вручную переписывать адрес и пароль при каждой смене.

НАСТРОЙКА
    CLIPPER_VPN_API_KEY=stlth_xxxxxxxx_yyyyyyyyyyyyyyyyyyyyyyyy
Ключ создаётся в консоли: https://console.stealthsurf.net

Полученный прокси кешируется на диск, чтобы не дёргать API на каждой задаче:
у него ограничение 60 запросов в минуту, а прокси меняется редко.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("CLIPPER_VPN_API_BASE", "https://api.stealthsurf.net")
KEY_RE = re.compile(r"^stlth_[a-f0-9]{8}_[a-f0-9]{24}$")

CACHE_FILE = Path(__file__).resolve().parent / "data" / ".vpn_proxy_cache.json"
CACHE_TTL = int(os.environ.get("CLIPPER_VPN_CACHE_TTL", str(6 * 3600)))

_TIMEOUT = 30


class VpnApiError(RuntimeError):
    """Ошибка обращения к API VPN — с текстом, понятным человеку."""


def api_key() -> str | None:
    key = (os.environ.get("CLIPPER_VPN_API_KEY") or "").strip()
    return key or None


def is_configured() -> bool:
    return api_key() is not None


def _request(method: str, path: str, body: dict | None = None) -> dict:
    key = api_key()
    if not key:
        raise VpnApiError("Не задан CLIPPER_VPN_API_KEY")
    if not KEY_RE.match(key):
        raise VpnApiError(
            "Ключ API имеет неверный формат. Должен быть вида "
            "stlth_0123abcd_0123456789abcdef01234567 (создаётся в консоли "
            "https://console.stealthsurf.net)")

    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{API_BASE}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
            msg = payload.get("message") or payload.get("error") or raw[:200]
        except Exception:  # noqa: BLE001
            msg = raw[:200]
        if e.code == 401:
            raise VpnApiError("Ключ API отклонён (401). Проверьте "
                              "CLIPPER_VPN_API_KEY.") from e
        if e.code == 403:
            raise VpnApiError(f"Доступ запрещён (403): {msg}") from e
        if e.code == 429:
            raise VpnApiError("Слишком много запросов к API (429). "
                              "Подождите минуту.") from e
        raise VpnApiError(f"API ответил {e.code}: {msg}") from e
    except urllib.error.URLError as e:
        raise VpnApiError(
            f"Не удалось связаться с API ({e.reason}). Обычно это значит, "
            f"что нет доступа в интернет.") from e


def _data(resp: dict):
    """Достаёт полезную часть ответа: {status, statusCode, data}."""
    if isinstance(resp, dict) and "data" in resp:
        return resp["data"]
    return resp


def list_configs() -> list[dict]:
    d = _data(_request("GET", "/configs"))
    if isinstance(d, dict):
        # На случай, если список завёрнут в объект с пагинацией
        for field in ("configs", "items", "list", "results"):
            if isinstance(d.get(field), list):
                return d[field]
        return []
    return d if isinstance(d, list) else []


def _config_id(cfg: dict):
    for field in ("id", "_id", "configId", "uuid"):
        if cfg.get(field) not in (None, ""):
            return cfg[field]
    return None


def _find_proxy(obj) -> dict | None:
    """Ищет в ответе данные прокси: адрес, порт, логин, пароль.

    Поля у разных методов называются по-разному, поэтому вместо жёсткой
    привязки к одному имени просматриваем структуру целиком. Так интеграция
    переживёт мелкие изменения в API.
    """
    if isinstance(obj, list):
        for item in obj:
            found = _find_proxy(item)
            if found:
                return found
        return None
    if not isinstance(obj, dict):
        return None

    host = _pick(obj, ("host", "server", "address", "ip", "domain", "hostname"))
    port = _pick(obj, ("port", "listen_port", "listenPort"))
    if host and port:
        return {
            "host": str(host),
            "port": int(port),
            "user": _pick(obj, ("login", "username", "user", "uuid")) or "",
            "password": _pick(obj, ("password", "pass", "secret")) or "",
            "protocol": str(_pick(obj, ("protocol", "type", "scheme"))
                            or "socks5").lower(),
        }
    for value in obj.values():
        found = _find_proxy(value)
        if found:
            return found
    return None


def _pick(d: dict, names: tuple[str, ...]):
    for n in names:
        if d.get(n) not in (None, ""):
            return d[n]
    return None


def _to_url(p: dict) -> str:
    from urllib.parse import quote

    scheme = "http" if p["protocol"].startswith("http") else "socks5h"
    auth = ""
    if p["user"] or p["password"]:
        auth = f"{quote(str(p['user']), safe='')}:{quote(str(p['password']), safe='')}@"
    return f"{scheme}://{auth}{p['host']}:{p['port']}"


def _cache_read() -> str | None:
    try:
        d = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if time.time() - float(d.get("ts", 0)) < CACHE_TTL and d.get("proxy"):
            return d["proxy"]
    except Exception:  # noqa: BLE001
        pass
    return None


def _cache_write(proxy: str) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps({"ts": time.time(), "proxy": proxy}),
                              encoding="utf-8")
        CACHE_FILE.chmod(0o600)      # внутри пароль от прокси
    except Exception:  # noqa: BLE001
        pass


def get_proxy(force: bool = False) -> str | None:
    """Возвращает адрес прокси вида socks5h://логин:пароль@хост:порт.

    Сначала ищет уже созданный «дополнительный конфиг», и только если его
    нет — создаёт новый. Создавать лишние незачем: он привязан к основному
    конфигу и живёт вместе с ним.
    """
    if not is_configured():
        return None
    if not force:
        cached = _cache_read()
        if cached:
            return cached

    configs = list_configs()
    if not configs:
        raise VpnApiError(
            "В аккаунте нет ни одной конфигурации VPN. Создайте её в консоли: "
            "https://console.stealthsurf.net")

    last_error: Exception | None = None
    for cfg in configs:
        cid = _config_id(cfg)
        if cid is None:
            continue

        # 1. Прокси уже может быть в самой конфигурации
        found = _find_proxy(cfg.get("subconfig") or cfg.get("subConfig") or {})
        if found:
            url = _to_url(found)
            _cache_write(url)
            return url

        # 2. Спрашиваем отдельно
        try:
            resp = _request("GET", f"/configs/{cid}/subconfig/protocols")
            found = _find_proxy(_data(resp))
            if found:
                url = _to_url(found)
                _cache_write(url)
                return url
        except VpnApiError as e:
            last_error = e

        # 3. Создаём новый дополнительный конфиг с SOCKS5
        for payload in ({"protocol": "socks5"}, {"type": "socks5"},
                        {"protocol": "socks"}):
            try:
                resp = _request("POST", f"/configs/{cid}/subconfig", payload)
                found = _find_proxy(_data(resp))
                if found:
                    url = _to_url(found)
                    _cache_write(url)
                    return url
            except VpnApiError as e:
                last_error = e
                continue

    hint = f" Последняя ошибка: {last_error}" if last_error else ""
    raise VpnApiError(
        "Не удалось получить прокси через API. Создайте его вручную: консоль "
        "→ ваш конфиг → «Дополнительный конфиг» → SOCKS5, затем впишите "
        "адрес в CLIPPER_PROXY." + hint)


def describe(proxy: str) -> str:
    """Адрес без пароля — для показа в логах."""
    return re.sub(r"//[^@/]*@", "//***@", proxy or "")
