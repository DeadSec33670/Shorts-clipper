"""Учёт расхода LLM-токенов и себестоимости прогонов.

Пишет в usage.json рядом с проектом. Нужен, чтобы ПОСЧИТАТЬ реальную
себестоимость одного видео и на её основе назначать цену подписки —
информация внутренняя, показывается только в админке.

Токены считаются приблизительно: точное число возвращают не все провайдеры,
поэтому оцениваем по длине текста. Для прикидки себестоимости этого хватает
(ошибка порядка 10-20%), а порядок цифр виден сразу.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USAGE_FILE = ROOT / "usage.json"
_lock = threading.Lock()

# Цены за 1 млн токенов (USD), ввод/вывод. Ориентировочные — провайдеры
# меняют тарифы, поэтому это КАЛЬКУЛЯТОР, а не источник истины.
# Обнови при необходимости, цифры видны в админке.
PRICES = {
    "gemini":     {"in": 0.075, "out": 0.30,  "note": "Gemini Flash (платный тариф)"},
    "groq":       {"in": 0.05,  "out": 0.08,  "note": "Groq Llama (платный тариф)"},
    "openrouter": {"in": 0.10,  "out": 0.30,  "note": "зависит от модели"},
    "claude":     {"in": 0.80,  "out": 4.00,  "note": "Claude Haiku"},
}

# сколько символов текста приходится на 1 токен (для русского ~2.5-3)
_CHARS_PER_TOKEN = 2.8


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text or "") / _CHARS_PER_TOKEN))


def _load() -> dict:
    if not USAGE_FILE.exists():
        return {"calls": [], "runs": []}
    try:
        d = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        d.setdefault("calls", [])
        d.setdefault("runs", [])
        return d
    except Exception:  # noqa: BLE001
        return {"calls": [], "runs": []}


def _save(d: dict) -> None:
    try:
        USAGE_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass          # учёт не должен ронять обработку


def cost_usd(provider: str, tok_in: int, tok_out: int) -> float:
    p = PRICES.get(provider, PRICES["gemini"])
    return tok_in / 1e6 * p["in"] + tok_out / 1e6 * p["out"]


def record_call(provider: str, model: str, system: str, user: str,
                answer: str, stage: str = "") -> None:
    """Фиксирует один запрос к LLM. Вызывается из call_llm.

    Пишет в базу с привязкой к пользователю и задаче: воркер передаёт их
    через переменные окружения. Прежний вариант привязывал запросы к прогону
    по времени, что при параллельной работе давало неверные цифры.
    """
    tok_in = estimate_tokens(system) + estimate_tokens(user)
    tok_out = estimate_tokens(answer)
    try:
        import os
        import sys
        sys.path.insert(0, str(ROOT))
        import store
        store.record_usage(os.environ.get("CLIPPER_USER_ID"),
                           os.environ.get("CLIPPER_JOB_ID"),
                           provider, model, stage, tok_in, tok_out,
                           cost_usd(provider, tok_in, tok_out))
        return
    except Exception:  # noqa: BLE001 — при недоступной базе пишем в файл
        pass
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": provider, "model": model, "stage": stage,
        "tok_in": tok_in, "tok_out": tok_out,
        "cost": round(cost_usd(provider, tok_in, tok_out), 6),
    }
    with _lock:
        d = _load()
        d["calls"].append(entry)
        d["calls"] = d["calls"][-5000:]      # не растим файл бесконечно
        _save(d)


def record_run(source: str, clips: int, seconds: float, provider: str) -> None:
    """Фиксирует завершённый прогон (одно видео -> N клипов) и привязывает к
    нему LLM-запросы, сделанные с момента прошлого прогона."""
    with _lock:
        d = _load()
        last_ts = d["runs"][-1]["ts"] if d["runs"] else ""
        mine = [c for c in d["calls"] if c["ts"] > last_ts]
        d["runs"].append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": str(source)[:200], "clips": clips,
            "seconds": round(seconds, 1), "provider": provider,
            "llm_calls": len(mine),
            "tok_in": sum(c["tok_in"] for c in mine),
            "tok_out": sum(c["tok_out"] for c in mine),
            "cost": round(sum(c["cost"] for c in mine), 6),
        })
        d["runs"] = d["runs"][-1000:]
        _save(d)


def summary() -> dict:
    """Сводка для админки: сколько потрачено и сколько стоит одно видео."""
    d = _load()
    calls, runs = d["calls"], d["runs"]
    total_cost = sum(c["cost"] for c in calls)
    total_in = sum(c["tok_in"] for c in calls)
    total_out = sum(c["tok_out"] for c in calls)
    n_clips = sum(r["clips"] for r in runs) or 0

    per_run = [r for r in runs if r["cost"] > 0]
    avg_run = (sum(r["cost"] for r in per_run) / len(per_run)) if per_run else 0.0
    avg_clip = (sum(r["cost"] for r in per_run) /
                max(1, sum(r["clips"] for r in per_run))) if per_run else 0.0
    avg_sec = (sum(r["seconds"] for r in runs) / len(runs)) if runs else 0.0

    by_stage: dict[str, dict] = {}
    for c in calls:
        s = by_stage.setdefault(c.get("stage") or "прочее",
                                {"calls": 0, "cost": 0.0, "tok_in": 0, "tok_out": 0})
        s["calls"] += 1
        s["cost"] += c["cost"]
        s["tok_in"] += c["tok_in"]
        s["tok_out"] += c["tok_out"]
    for s in by_stage.values():
        s["cost"] = round(s["cost"], 6)

    return {
        "total_cost": round(total_cost, 4),
        "total_tok_in": total_in, "total_tok_out": total_out,
        "runs": len(runs), "clips": n_clips, "calls": len(calls),
        "avg_cost_per_run": round(avg_run, 5),
        "avg_cost_per_clip": round(avg_clip, 5),
        "avg_seconds_per_run": round(avg_sec, 1),
        "by_stage": by_stage,
        "prices": PRICES,
        "recent": list(reversed(runs[-20:])),
    }


def reset() -> None:
    with _lock:
        _save({"calls": [], "runs": []})


def summary_db() -> dict:
    """Сводка себестоимости из базы — для админки.

    ВАЖНО: здесь только расходы на LLM. Главная статья затрат — время
    видеокарты (Whisper, ASD, рендер), и её нужно считать отдельно, исходя
    из стоимости аренды сервера и суммарных минут обработки.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    import store

    tot = store.query_one(
        "SELECT COUNT(*) n, COALESCE(SUM(cost),0) c, "
        "COALESCE(SUM(tok_in),0) ti, COALESCE(SUM(tok_out),0) tob "
        "FROM usage_records") or {}
    jobs = store.query_one(
        "SELECT COUNT(*) n, COALESCE(SUM(clips_count),0) clips, "
        "COALESCE(SUM(video_minutes),0) mins FROM jobs WHERE status='done'") or {}

    by_stage = {}
    for r in store.query("SELECT stage, COUNT(*) calls, SUM(cost) cost, "
                         "SUM(tok_in) tok_in, SUM(tok_out) tok_out "
                         "FROM usage_records GROUP BY stage"):
        by_stage[r["stage"] or "прочее"] = {
            "calls": r["calls"], "cost": round(r["cost"] or 0, 6),
            "tok_in": r["tok_in"] or 0, "tok_out": r["tok_out"] or 0}

    recent = store.query(
        "SELECT j.id, j.title source, j.clips_count clips, j.video_minutes, "
        "j.created_at ts, j.status, u.email, "
        "(SELECT COALESCE(SUM(cost),0) FROM usage_records r WHERE r.job_id=j.id) cost "
        "FROM jobs j LEFT JOIN users u ON u.id=j.user_id "
        "ORDER BY j.created_at DESC LIMIT 20")

    n_jobs = int(jobs.get("n") or 0)
    n_clips = int(jobs.get("clips") or 0)
    total = float(tot.get("c") or 0)
    return {
        "total_cost": round(total, 4),
        "total_tok_in": int(tot.get("ti") or 0),
        "total_tok_out": int(tot.get("tob") or 0),
        "calls": int(tot.get("n") or 0),
        "runs": n_jobs, "clips": n_clips,
        "total_minutes": round(float(jobs.get("mins") or 0), 1),
        "avg_cost_per_run": round(total / n_jobs, 5) if n_jobs else 0,
        "avg_cost_per_clip": round(total / n_clips, 5) if n_clips else 0,
        "avg_seconds_per_run": 0,
        "by_stage": by_stage, "prices": PRICES,
        "recent": recent,
    }
