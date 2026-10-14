"""Декорации субтитров: LLM выбирает эмоциональные слова каждого клипа
и назначает им цвет и/или эмодзи (стиль Vizard).

Один запрос на все клипы сразу; результат кешируется в workdir/decor.json.
Сбой декораций не роняет пайплайн — клипы просто выходят без украшений.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .llm_providers import call_llm
from .moments import Moment, _parse_json, _segment_text
from .transcribe import Segment

# палитра ASS (BGR!): жёлтый = штатная подсветка, красный/зелёный — акценты
PALETTE = {
    "yellow": "&H0047D4FF",   # #FFD447
    "red": "&H004B4BFF",      # #FF4B4B
    "green": "&H0081DC3E",    # #3EDC81
}

DECOR_SYSTEM = """Ты — моушн-дизайнер субтитров для вирусных шортсов. Тебе дают
тексты клипов. Для каждого клипа выдели 6-12 самых ЦЕПЛЯЮЩИХ слов — не только
про деньги и шок, а любые слова, на которых зритель мысленно «спотыкается».
Категории слов, которые стоит выделять:
- деньги и числа: суммы, проценты, «миллион», «бесплатно», «дорого», «даром»;
- ставки и риск: спор, пари, вызов, «на кону», дедлайн, «слабо»;
- сильные эмоции: шок, восторг, злость, страх, стыд, «ужас», «кошмар»;
- оценки и превосходство: «лучший», «худший», «главный», «единственный»,
  «впервые», «никогда», «всегда», «самый», «нереально», «легендарный»;
- конфликт и обвинения: «врёшь», «обман», «скандал», «предал», «провал»;
- победа/поражение: «выиграл», «проиграл», «уничтожил», «порвал», «сдался»;
- усилители и сленг: «жёстко», «капец», «реально», «серьёзно», «внезапно»,
  «офигеть», «дичь», «топ», «имба», «кринж», «вау»;
- имена людей, брендов, мест, если они в фокусе;
- слова-крючки вопроса/интриги: «почему», «секрет», «правда», «оказалось».
Цвет (color): "red" — шок/опасность/провал/обвинение; "green" — деньги/победа/
успех; "yellow" — любой другой смысловой акцент (по умолчанию для выделения);
или "none".
Выделяй ЩЕДРО, но не подряд: не два выделенных слова вплотную, чтобы строка не
пестрила. emoji: ОДИН подходящий или "" (эмодзи по-прежнему МАКСИМУМ 2-4 на
клип, только где реально усиливает: 💰 деньги, 🔥 мощно, 😂 смешно, 🤯 шок,
💪 сила, 🏆 победа, 😱 страх). Большинство выделенных слов идут БЕЗ эмодзи,
только с цветом.
Слова бери ТОЧНО в той форме, как они написаны в тексте.
Отвечай ТОЛЬКО валидным JSON без пояснений."""

DECOR_TEMPLATE = """Тексты клипов:

{clips_text}

Ответ строго в формате:
{{"clips": [{{"clip": 0, "words": [{{"word": "миллион", "color": "green", "emoji": "💰"}}]}}]}}"""


def decorate_clips(moments: list[Moment], segments: list[Segment],
                   cfg: Config) -> dict[int, dict[str, tuple[str, str]]]:
    """-> {индекс клипа: {слово: (ass_color|"", emoji|"")}}. Терпит любые сбои."""
    cache = cfg.workdir / "decor.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return {int(k): {w: tuple(v) for w, v in d.items()} for k, d in raw.items()}

    blocks = []
    for i, m in enumerate(moments):
        text = " ".join(_segment_text(a, b, segments) for a, b in m.segments)
        blocks.append(f"Клип {i}: «{text[:1500]}»")

    print(f"[decor] Подбираю цвета и эмодзи для {len(moments)} клипов…")
    try:
        raw = call_llm(
            provider=cfg.llm_provider, model=cfg.llm_model,
            system=DECOR_SYSTEM,
            user=DECOR_TEMPLATE.format(clips_text="\n\n".join(blocks)),
            max_tokens=9000,
            stage="декор субтитров",
        )
        data = _parse_json(raw)
    except Exception as e:  # noqa: BLE001
        print(f"[decor] Не удалось ({type(e).__name__}) — клипы без украшений")
        return {}

    out: dict[int, dict[str, tuple[str, str]]] = {}
    for c in data.get("clips", []):
        try:
            idx = int(c["clip"])
        except (KeyError, TypeError, ValueError):
            continue
        words = {}
        for w in c.get("words", []):
            token = str(w.get("word", "")).strip()
            if not token:
                continue
            color = PALETTE.get(str(w.get("color", "")).lower(), "")
            emoji = str(w.get("emoji", "")).strip()[:2]
            if color or emoji:
                words[token.lower()] = (color, emoji)
        if words:
            out[idx] = words

    cache.write_text(
        json.dumps({str(k): {w: list(v) for w, v in d.items()}
                    for k, d in out.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    n_words = sum(len(d) for d in out.values())
    print(f"[decor] Украшено слов: {n_words} в {len(out)} клипах")
    return out
