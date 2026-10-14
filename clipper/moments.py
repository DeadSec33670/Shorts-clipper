"""Шаг 3. Поиск виральных моментов через LLM (любой провайдер).

Клип теперь — не один сплошной кусок, а НАРЕЗКА: 1-4 сегмента из разных мест
видео на одну тему, склеенных в мини-историю (хук -> развитие -> развязка),
как это делают Opus Clip / Klap. LLM возвращает для каждого клипа список
сегментов с тайм-кодами; границы прищёлкиваются к границам слов.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .llm_providers import call_llm
from .transcribe import Segment

CACHE_VERSION = 2

SYSTEM_PROMPT = """Ты — опытный клипмейкер, который превращает длинные видео в вирусные
шортсы для Reels/TikTok/YouTube Shorts. Твоя суперсила — собирать из длинного
видео тематические нарезки: находить в РАЗНЫХ местах видео фрагменты об одном
и том же (одна история, один спор, одна тема) и склеивать их в цельную
мини-историю с драматургией: цепляющий хук -> развитие -> развязка/панчлайн.

КРИТЕРИИ ИНТЕРЕСНОСТИ (клип обязан попадать хотя бы в один):
- ставки: деньги, спор, пари, вызов, риск, дедлайн;
- конфликт или напряжение: несогласие, провокация, неловкость, провал;
- знаменитость или уникальное событие: встреча, реакция, инсайд;
- сильная эмоция: смех, шок, восторг, злость — С ПРИЧИНОЙ в кадре;
- законченная история: завязка -> поворот -> развязка/панчлайн;
- конкретная польза: цифры, приёмы, «как сделать Х»;
- мнение-бомба: резкое, полярное или контринтуитивное заявление, на которое
  зритель захочет ответить «согласен / не согласен» (двигатель комментариев);
- фраза-цитата: одна ёмкая фраза, которая работает сама по себе как цитата.
ЗАПРЕЩЕНО брать: приветствия и прощания, «мы едем / мы идём / мы ждём»,
бытовую болтовню без события, описания планов без исхода, фрагменты, где
непонятно, о ком/чём речь. Если из видео честно набирается меньше клипов,
чем просили — верни меньше, но каждый должен быть сильным.
Оценка virality: 9-10 = «остановит скролл любого», 7-8 = сильный момент,
5-6 = средний (такие лучше не включать), ниже — мусор.

Правила:
- ЖЕЛЕЗНОЕ ПРАВИЛО РАЗВЯЗКИ: если клип заявляет спор, пари, вызов, вопрос
  или интригу («сможет ли X сделать Y») — клип ОБЯЗАН содержать сегмент с
  ИСХОДОМ (получилось/не получилось, кто выиграл, чем кончилось). Найди
  исход в транскрипте, даже если он через полчаса от завязки. Заявка без
  развязки — худшая ошибка клипмейкера: такой клип брать ЗАПРЕЩЕНО.
  Последний сегмент клипа = развязка или реакция на неё.
- ПО УМОЛЧАНИЮ клип — ОДИН сплошной сегмент. Склейка из 2-3 сегментов
  разрешена ТОЛЬКО когда одна история реально разнесена по видео (спор
  заключили в начале — исход показали позже) И стык читается как связная
  речь. Если сомневаешься — не склеивай.
- ЗОЛОТОЕ ПРАВИЛО СТЫКА: прочитай вслух последнее предложение сегмента и
  первое предложение следующего. Если это не звучит как непрерывный рассказ
  одного человека — склейка запрещена.
- Каждый сегмент начинается с НАЧАЛА предложения и заканчивается КОНЦОМ
  предложения. Никогда не режь фразу посередине.
- Сегмент не должен начинаться с местоимений и отсылок без опоры («а он
  такой», «и вот это вот», «поэтому»), если опора осталась за кадром.
- Первый сегмент клипа обязан цеплять с первых 1-2 секунд: интрига, вопрос,
  эмоция, заявление. Клип, который в первые 3 секунды не даёт зрителю повода
  не листать дальше, брать НЕЛЬЗЯ — какой бы сильной ни была середина.
- Сегменты одного клипа не пересекаются, каждый длиннее 6 секунд.
- Суммарная длительность клипа: от {min_sec:.0f} до {max_sec:.0f} секунд.
- Отвечай ТОЛЬКО валидным JSON без markdown-обёрток и пояснений."""

USER_PROMPT_TEMPLATE = """Ниже транскрипт видео с тайм-кодами (секунды).

Собери {clips_count} лучших клипов. Помни: клип может (и часто должен) состоять
из нескольких сегментов из разных мест видео, объединённых одной темой.

Ответ — строго JSON такого вида:
{{
  "clips": [
    {{
      "title": "короткий цепляющий заголовок",
      "hook": "первая фраза клипа",
      "theme": "какая тема объединяет сегменты",
      "reason": "почему это зайдёт",
      "virality": 8.5,
      "segments": [
        {{"start": 123.4, "end": 141.2}},
        {{"start": 890.0, "end": 912.5}}
      ]
    }}
  ]
}}

ТРАНСКРИПТ:
{transcript}"""


@dataclass
class Moment:
    title: str
    hook: str
    theme: str
    reason: str
    virality: float
    segments: list = field(default_factory=list)  # [(start, end), ...]

    @property
    def total_sec(self) -> float:
        return sum(e - s for s, e in self.segments)


def find_moments(segments: list[Segment], cfg: Config) -> list[Moment]:
    """Возвращает моменты по убыванию виральности. Кешируется (версия 2)."""
    cache = cfg.workdir / "moments.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == CACHE_VERSION:
            return [_moment_from_dict(m) for m in data["clips"]]
        print("[moments] Кеш старого формата — ищу моменты заново")

    transcript_text = "\n".join(
        f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in segments
    )

    print(f"[moments] Спрашиваю {cfg.llm_provider}/{cfg.llm_model}…")
    data = None
    for attempt in (1, 2):
        raw = call_llm(
            provider=cfg.llm_provider,
            model=cfg.llm_model,
            system=SYSTEM_PROMPT.format(min_sec=cfg.min_clip_sec, max_sec=cfg.max_clip_sec),
            user=USER_PROMPT_TEMPLATE.format(
                clips_count=cfg.clips_count,
                transcript=transcript_text,
            ),
            max_tokens=16000,
            stage="поиск моментов",
        )
        (cfg.workdir / "llm_raw.txt").write_text(raw, encoding="utf-8")
        try:
            data = _parse_json(raw)
            break
        except ValueError as e:
            if attempt == 2:
                raise RuntimeError(
                    f"LLM дважды вернула неразборный ответ ({e}). "
                    f"Сырой ответ сохранён в {cfg.workdir / 'llm_raw.txt'} — "
                    f"пришлите его для диагностики, или попробуйте другой "
                    f"провайдер: --llm-provider groq"
                ) from e
            print(f"[moments] Ответ не разобрался ({e}) — пробую ещё раз…")

    moments: list[Moment] = []
    for c in data.get("clips", []):
        segs = _extract_segments(c)
        if not segs:
            continue
        snapped = [_snap_to_words(s, e, segments) for s, e in segs]
        snapped = [x for x in snapped if x is not None and x[1] - x[0] >= 3.5]
        if not snapped:
            continue
        m = Moment(
            title=str(c.get("title", "")), hook=str(c.get("hook", "")),
            theme=str(c.get("theme", "")), reason=str(c.get("reason", "")),
            virality=float(c.get("virality", 5)), segments=snapped,
        )
        if m.total_sec < cfg.min_clip_sec * 0.6 or m.total_sec > cfg.max_clip_sec * 1.5:
            continue
        moments.append(m)

    moments = _drop_heavy_overlaps(sorted(moments, key=lambda m: -m.virality))

    # порог интересности: клипы со средней оценкой не выпускаем,
    # но минимум 5 лучших оставляем всегда
    strong = [m for m in moments if m.virality >= 6.5]
    if len(strong) < min(5, len(moments)):
        strong = moments[:5]
    if len(strong) < len(moments):
        print(f"[moments] Отсеяно {len(moments) - len(strong)} слабых клипов "
              f"(virality < 6.5)")
    moments = strong

    moments = _review_joints(moments, segments, cfg)
    cache.write_text(
        json.dumps({"version": CACHE_VERSION,
                    "clips": [_moment_to_dict(m) for m in moments]},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[moments] Собрано {len(moments)} клипов")
    n_short = 0
    for m in moments:
        parts = " + ".join(f"{s:.0f}-{e:.0f}с" for s, e in m.segments)
        tag = ""
        if m.total_sec < cfg.min_clip_sec:
            tag = f"  ⚠ {m.total_sec:.0f}с (короче цели {cfg.min_clip_sec:.0f}с)"
            n_short += 1
        print(f"  {m.virality:>4.1f}  [{parts}]  {m.title}{tag}")
    if n_short:
        print(f"[moments] {n_short} клип(ов) короче запрошенной длины: в этом "
              f"месте видео не нашлось более длинной СВЯЗНОЙ истории — короткий "
              f"цельный клип предпочтён растянутому несвязному (ожидаемо для "
              f"материала с быстрым монтажом).")
    return moments


def _extract_segments(c: dict) -> list[tuple[float, float]]:
    """Достаёт сегменты; понимает и старый формат (start/end на верхнем уровне)."""
    raw = c.get("segments")
    if not raw and "start" in c and "end" in c:
        raw = [{"start": c["start"], "end": c["end"]}]
    if not raw:
        return []
    out = []
    for s in raw:
        try:
            a, b = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if b > a:
            out.append((a, b))
    return out[:4]


def _moment_to_dict(m: Moment) -> dict:
    return {"title": m.title, "hook": m.hook, "theme": m.theme, "reason": m.reason,
            "virality": m.virality,
            "segments": [{"start": s, "end": e} for s, e in m.segments]}


def _moment_from_dict(d: dict) -> Moment:
    return Moment(
        title=d["title"], hook=d["hook"], theme=d.get("theme", ""),
        reason=d["reason"], virality=d["virality"],
        segments=[(s["start"], s["end"]) for s in d["segments"]],
    )


REVIEW_SYSTEM = """Ты — придирчивый редактор монтажа шортсов. Тебе дают клипы,
СКЛЕЕННЫЕ из сегментов длинного видео, с текстом каждого сегмента. Твоя задача —
проверить каждый СТЫК: читается ли текст после склейки как связная речь одного
рассказа? Плохой стык: мысль обрывается на полуслове; следующий сегмент начинается
с реакции на что-то, чего зритель не видел; резко меняется тема без связки.
ОТДЕЛЬНО проверь развязку: если клип заявляет спор/вызов/интригу, но исход
в тексте сегментов ОТСУТСТВУЕТ — клип безнадёжен, верни пустой список.

Для каждого клипа верни, какие сегменты оставить и в каком порядке:
- все хороши -> оставь как есть;
- один сегмент ломает логику -> выкинь его (если оставшееся >= 15 секунд);
- склейка в принципе не работает -> оставь ОДИН лучший сегмент;
- клип безнадёжен -> пустой список.
Отвечай ТОЛЬКО валидным JSON без пояснений."""

REVIEW_TEMPLATE = """Клипы (сегменты пронумерованы с 0, для каждого — его текст):

{clips_text}

Ответ строго в формате:
{{"verdicts": [{{"clip": 0, "keep": [0, 1]}}, {{"clip": 1, "keep": [1]}}]}}"""


def _segment_text(a: float, b: float, segments: list[Segment]) -> str:
    return " ".join(w.text for s in segments for w in s.words
                    if a - 0.05 <= w.start and w.end <= b + 0.05)


def _review_joints(moments: list[Moment], segments: list[Segment],
                   cfg: Config) -> list[Moment]:
    """Второй LLM-проход: редактор проверяет читаемость стыков нарезок."""
    multi = [(i, m) for i, m in enumerate(moments) if len(m.segments) > 1]
    if not multi:
        return moments

    blocks = []
    for i, m in multi:
        seg_lines = "\n".join(
            f"  сегмент {j}: «{_segment_text(a, b, segments)}»"
            for j, (a, b) in enumerate(m.segments))
        blocks.append(f"Клип {i} («{m.title}»):\n{seg_lines}")

    print(f"[moments] Редакторская вычитка стыков ({len(multi)} склеенных клипов)…")
    try:
        raw = call_llm(
            provider=cfg.llm_provider, model=cfg.llm_model,
            system=REVIEW_SYSTEM,
            user=REVIEW_TEMPLATE.format(clips_text="\n\n".join(blocks)),
            max_tokens=4000,
            stage="вычитка стыков",
        )
        verdicts = {v["clip"]: v.get("keep", [])
                    for v in _parse_json(raw).get("verdicts", [])
                    if isinstance(v, dict) and "clip" in v}
    except Exception as e:  # noqa: BLE001 — вычитка не должна ронять пайплайн
        print(f"[moments] Вычитка не удалась ({type(e).__name__}) — оставляю как есть")
        return moments

    out: list[Moment] = []
    for i, m in enumerate(moments):
        keep = verdicts.get(i)
        if keep is None or len(m.segments) == 1:
            out.append(m)
            continue
        new_segs = [m.segments[j] for j in keep
                    if isinstance(j, int) and 0 <= j < len(m.segments)]
        if not new_segs:
            print(f"  [x] «{m.title}» — редактор забраковал клип целиком")
            continue
        if len(new_segs) != len(m.segments):
            print(f"  [~] «{m.title}» — оставлено {len(new_segs)} "
                  f"из {len(m.segments)} сегментов")
        m.segments = new_segs
        if m.total_sec >= cfg.min_clip_sec * 0.6:
            out.append(m)
    return out


def _parse_json(raw: str) -> dict:
    """Терпимый парсер ответа LLM.

    1) снимает markdown-обёртку и парсит как есть;
    2) чинит частые дефекты (запятые перед ] и });
    3) если JSON обрезан/повреждён — выуживает из текста все ЦЕЛЫЕ объекты
       клипов балансировкой скобок и пересобирает {"clips": [...]}.
    """
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1:
        raise ValueError(f"LLM вернула не-JSON ответ:\n{raw[:500]}")
    body = raw[start:end + 1] if end > start else raw[start:]

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass

    # дефект: висячие запятые перед закрывающей скобкой
    repaired = re.sub(r",\s*([\]}])", r"\1", body)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # тяжёлый случай: собираем клипы по одному балансировкой скобок
    clips = []
    depth, obj_start = 0, None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 1 and obj_start is None and i > start:
                obj_start = i          # объект внутри массива clips
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 1 and obj_start is not None:
                chunk = re.sub(r",\s*([\]}])", r"\1", raw[obj_start:i + 1])
                try:
                    o = json.loads(chunk)
                    if ("segments" in o) or ("start" in o and "end" in o):
                        clips.append(o)
                except json.JSONDecodeError:
                    pass
                obj_start = None
    if clips:
        print(f"[moments] JSON был повреждён — спасено {len(clips)} целых клипов")
        return {"clips": clips}
    raise ValueError("Не удалось разобрать ответ LLM даже после ремонта")


_EOS = (".", "!", "?", "…")
_PAUSE_SEC = 0.55   # пауза в речи длиннее этой = граница мысли (Whisper на
                    # разговорной речи часто скупится на точки — паузы спасают)


def _sentence_bounds(segments: list[Segment]) -> tuple[list[float], list[float]]:
    """Моменты начала и конца мыслей: пунктуация ИЛИ ощутимая пауза в речи."""
    words = [w for s in segments for w in s.words]
    starts, ends = [], []
    new_sentence = True
    for i, w in enumerate(words):
        if new_sentence:
            starts.append(w.start)
        eos = w.text.rstrip('"»)').endswith(_EOS)
        pause = (i + 1 < len(words)
                 and words[i + 1].start - w.end >= _PAUSE_SEC)
        if eos or pause or i == len(words) - 1:
            ends.append(w.end)
        new_sentence = eos or pause
    return starts, ends


def _snap_to_words(start: float, end: float,
                   segments: list[Segment]) -> tuple[float, float] | None:
    """Прищёлкивает границы сегмента к границам ПРЕДЛОЖЕНИЙ (в пределах 4 с),
    чтобы стык не резал мысль на полуслове; иначе — к границам слов."""
    words = [w for s in segments for w in s.words]
    if not words:
        return (start, end)
    s_starts, s_ends = _sentence_bounds(segments)

    a = _nearest(start, s_starts, tol=4.0)
    if a is None:
        a = _nearest(start, [w.start for w in words], tol=None)
    b = _nearest(end, s_ends, tol=6.0)   # ищем конец мысли щедрее:
    if b is None:                         # лучше договорить, чем оборвать
        b = _nearest(end, [w.end for w in words], tol=None)

    a = max(0.0, a - 0.15)
    b = b + 0.35
    return (a, b) if b > a else None


def _nearest(t: float, candidates: list[float], tol: float | None) -> float | None:
    if not candidates:
        return None
    best = min(candidates, key=lambda c: abs(c - t))
    if tol is not None and abs(best - t) > tol:
        return None
    return best


def _drop_heavy_overlaps(moments: list[Moment]) -> list[Moment]:
    """Оставляет самые виральные; выбрасывает клипы, чей материал уже
    больше чем наполовину использован более виральными клипами."""
    kept: list[Moment] = []
    for m in moments:
        overlap = 0.0
        for s, e in m.segments:
            for k in kept:
                for ks, ke in k.segments:
                    overlap += max(0.0, min(e, ke) - max(s, ks))
        if m.total_sec > 0 and overlap / m.total_sec < 0.5:
            kept.append(m)
    return kept
