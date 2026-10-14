"""Шаг 4б. Анимированные субтитры в формате ASS.

Стиль «караоке для шортсов»: короткие строки по 3-5 слов, активное слово
подсвечивается цветом и слегка увеличено. Тайм-коды берутся из Whisper
на уровне слов, поэтому подсветка идёт точно в такт речи.
"""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .transcribe import Segment, Word

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,{font},{size},{base_color},&H000000FF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,60,60,{margin_v},1
Style: Hook,{font},{hook_size},&H00FFFFFF,&H000000FF,&H00101010,&HA0000000,-1,0,0,0,100,100,0,0,3,6,0,8,70,70,{hook_margin},1
Style: Emoji,Noto Emoji,{emoji_size},&H0047D4FF,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,{emoji_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

_SENTENCE_END = (".", "!", "?", "…")


def clip_words(segments: list[Segment], start: float, end: float) -> list[Word]:
    """Слова, попадающие в интервал клипа, со временем относительно его начала."""
    out: list[Word] = []
    for s in segments:
        if s.end < start or s.start > end:
            continue
        for w in s.words:
            if w.start >= start - 0.05 and w.end <= end + 0.05:
                out.append(Word(max(0.0, w.start - start), max(0.0, w.end - start), w.text))
    return out


def _group_words(words: list[Word], max_chars: int) -> list[list[Word]]:
    """Разбивает слова на строки по бюджету символов.

    Правила:
      - строка не длиннее max_chars (считая пробелы между словами);
      - после слова с завершающей пунктуацией (. ! ? …) строка закрывается,
        чтобы конец одной фразы не смешивался с началом следующей;
      - одиночное слово длиннее бюджета остаётся один в строке
        (WrapStyle 0 в заголовке страхует от вылета за край).
    """
    groups: list[list[Word]] = []
    line: list[Word] = []
    length = 0
    for w in words:
        add = len(w.text) + (1 if line else 0)
        if line and (length + add > max_chars or len(line) >= 2):
            groups.append(line)
            line, length = [], 0
            add = len(w.text)
        line.append(w)
        length += add
        if w.text.rstrip('"»)').endswith(_SENTENCE_END):
            groups.append(line)
            line, length = [], 0
    if line:
        groups.append(line)
    return groups


def _decor_for(word_text: str, decor: dict | None) -> tuple[str, str]:
    """(ass_color|"", emoji|"") для слова; пунктуация по краям не мешает."""
    if not decor:
        return "", ""
    key = word_text.strip('.,!?…:;«»"()').lower()
    return decor.get(key, ("", ""))


def build_ass(words: list[Word], cfg: Config, path: Path,
              title: str | None = None,
              position_frac: float = 0.26,
              decor: dict | None = None) -> Path:
    """Пишет .ass: 1-2 слова на строку (стиль Vizard) + заголовок-хук.

    position_frac — высота субтитров от низа кадра (0.26 = нижняя треть,
    0.47 = на шве сплит-скрина)."""
    header = ASS_HEADER.format(
        w=cfg.out_width, h=cfg.out_height,
        font=cfg.sub_font, size=cfg.sub_font_size,
        base_color=cfg.sub_base_color,
        margin_v=int(cfg.out_height * position_frac),
        hook_size=int(cfg.sub_font_size * 0.82),
        hook_margin=int(cfg.out_height * 0.10),
        emoji_size=int(cfg.sub_font_size * 1.35),
        emoji_margin=int(cfg.out_height * position_frac) + int(cfg.sub_font_size * 1.5),
    )
    lines: list[str] = []

    if title:
        # заголовок-хук: плашка сверху первые 3.5 секунды, как в проф. клипах
        lines.append(
            f"Dialogue: 1,{_ass_ts(0.0)},{_ass_ts(3.5)},Hook,,0,0,0,,"
            f"{{\\fad(150,250)}}{_esc(title.upper())}"
        )

    groups = _group_words(words, cfg.sub_max_chars_per_line)

    # максимум, сколько строка «висит» после того, как отзвучало её последнее
    # слово. Раньше строка держалась до НАЧАЛА следующей группы — если дальше
    # шла длинная пауза/бролл (речи нет), субтитр застывал на экране на секунды.
    _HOLD_AFTER_SPEECH = 0.6

    for gi, group in enumerate(groups):
        line_start = group[0].start
        speech_end = group[-1].end          # когда реально отзвучало последнее слово
        next_start = (groups[gi + 1][0].start if gi + 1 < len(groups)
                      else speech_end + _HOLD_AFTER_SPEECH)
        # строка гаснет либо перед следующей фразой, либо вскоре после конца
        # речи — смотря что раньше. Так в паузах/бролле субтитр не застывает.
        line_end = min(next_start, speech_end + _HOLD_AFTER_SPEECH)
        if line_end <= line_start:
            line_end = line_start + 0.1

        # эмодзи строки (первое среди слов) — отдельным событием НАД текстом,
        # с лёгким появлением; так монохромный глиф выглядит элементом дизайна
        line_emoji = next((e for w in group
                           for _, e in [_decor_for(w.text, decor)] if e), "")
        if line_emoji:
            lines.append(
                f"Dialogue: 1,{_ass_ts(line_start)},{_ass_ts(line_end)},Emoji,,"
                f"0,0,0,,{{\\fad(120,80)}}{line_emoji}")

        # одно событие на каждое активное слово — так делается «подсветка»
        for wi, w in enumerate(group):
            ev_start = w.start if wi > 0 else line_start
            ev_end = group[wi + 1].start if wi + 1 < len(group) else line_end
            if ev_end <= ev_start:
                continue
            parts = []
            for xi, x in enumerate(group):
                d_color, _ = _decor_for(x.text, decor)
                shown = _esc(x.text)
                if xi == wi:   # активное слово: поп + цвет (свой или жёлтый)
                    col = d_color or cfg.sub_highlight_color
                    parts.append(f"{{\\c{col}\\fscx108\\fscy108}}{shown}"
                                 f"{{\\c{cfg.sub_base_color}\\fscx100\\fscy100}}")
                elif d_color:  # декорированное слово окрашено всегда
                    parts.append(f"{{\\c{d_color}}}{shown}"
                                 f"{{\\c{cfg.sub_base_color}}}")
                else:
                    parts.append(shown)
            text = " ".join(parts)
            lines.append(
                f"Dialogue: 0,{_ass_ts(ev_start)},{_ass_ts(ev_end)},Base,,0,0,0,,{text}"
            )

    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def _esc(text: str) -> str:
    """Обезвреживает текст для формата ASS.

    Фигурные скобки — управляющие последовательности, обратный слэш их
    открывает. А перевод строки обязательно заменяется на \\N: в ASS каждая
    реплика должна занимать ровно одну строку файла, и живой перенос сломал
    бы разбор всех последующих субтитров.
    """
    return (text.replace("\\", "")
                .replace("{", "(").replace("}", ")")
                .replace("\r\n", "\\N").replace("\n", "\\N").replace("\r", "\\N"))


def _ass_ts(sec: float) -> str:
    cs = int(round(sec * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
