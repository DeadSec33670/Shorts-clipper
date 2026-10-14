"""Проверка ядра продукта: разбор ответа LLM, субтитры, тайминги.

До сих пор тестами была покрыта инфраструктура (доступ, очередь, хранилище),
а самое хрупкое место оставалось без проверки: LLM возвращает текст, и он
регулярно приходит повреждённым — обрезанным по лимиту токенов, в markdown-
обёртке, с висячими запятыми. Каждый такой случай без ремонта означает
потерянный прогон и потраченное время видеокарты.

Запуск: python tests_pipeline.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clipper.config import Config  # noqa: E402
from clipper.moments import (_drop_heavy_overlaps, _extract_segments,  # noqa: E402
                             _moment_from_dict, _moment_to_dict, _nearest,
                             _parse_json, _snap_to_words, Moment)
from clipper.subtitles import _ass_ts, _esc, _group_words, clip_words  # noqa: E402
from clipper.transcribe import Segment, Word  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name +
          (f"  <- {detail}" if detail and not cond else ""))


# ===========================================================================
print("\n--- Разбор ответа LLM ---")

check("чистый JSON",
      _parse_json('{"clips":[{"start":1,"end":5}]}')["clips"][0]["end"] == 5)

check("markdown-обёртка снимается",
      _parse_json('```json\n{"clips":[{"start":1,"end":5}]}\n```')["clips"])

check("обёртка без указания языка",
      _parse_json('```\n{"clips":[{"start":1,"end":5}]}\n```')["clips"])

check("болтовня вокруг JSON игнорируется",
      _parse_json('Конечно! Вот результат:\n{"clips":[{"start":1,"end":5}]}\n'
                  'Надеюсь, подойдёт.')["clips"])

check("висячая запятая перед ]",
      _parse_json('{"clips":[{"start":1,"end":5},]}')["clips"])

check("висячая запятая перед }",
      _parse_json('{"clips":[{"start":1,"end":5,}]}')["clips"])

# Самый частый реальный сбой: ответ обрезан по лимиту токенов на середине.
truncated = ('{"clips":[{"title":"Первый","hook":"h","reason":"r","virality":9,'
             '"segments":[{"start":10,"end":40}]},'
             '{"title":"Второй","hook":"h","reason":"r","virality":8,'
             '"segments":[{"start":50,"end":80}]},'
             '{"title":"Третий","hook":"обрыв на середин')
try:
    got = _parse_json(truncated)
    check("обрезанный ответ: целые клипы спасены",
          len(got["clips"]) == 2, len(got["clips"]))
    check("спасённые клипы не испорчены",
          got["clips"][0]["title"] == "Первый"
          and got["clips"][1]["segments"][0]["end"] == 80)
except Exception as e:  # noqa: BLE001
    check("обрезанный ответ: целые клипы спасены", False, repr(e))
    check("спасённые клипы не испорчены", False, "")

try:
    _parse_json("Извините, я не могу выполнить этот запрос.")
    check("ответ без JSON вызывает понятную ошибку", False, "исключения не было")
except ValueError:
    check("ответ без JSON вызывает понятную ошибку", True)
except Exception as e:  # noqa: BLE001
    check("ответ без JSON вызывает понятную ошибку", False, repr(e))

try:
    _parse_json("{совсем не json")
    check("мусор вместо JSON вызывает ошибку", False, "исключения не было")
except ValueError:
    check("мусор вместо JSON вызывает ошибку", True)

# ===========================================================================
print("\n--- Разбор сегментов клипа ---")

check("обычные сегменты",
      _extract_segments({"segments": [{"start": 1, "end": 5},
                                      {"start": 8, "end": 12}]})
      == [(1.0, 5.0), (8.0, 12.0)])

check("старый формат start/end поддержан",
      _extract_segments({"start": 3, "end": 9}) == [(3.0, 9.0)])

check("строковые числа приводятся",
      _extract_segments({"segments": [{"start": "1.5", "end": "4"}]})
      == [(1.5, 4.0)])

check("сегмент нулевой длины отброшен",
      _extract_segments({"segments": [{"start": 5, "end": 5}]}) == [])

check("перевёрнутый сегмент отброшен",
      _extract_segments({"segments": [{"start": 9, "end": 3}]}) == [])

check("битый сегмент не роняет разбор",
      _extract_segments({"segments": [{"start": "нет", "end": 5},
                                      {"start": 1, "end": 4}]}) == [(1.0, 4.0)])

check("больше четырёх сегментов обрезается",
      len(_extract_segments({"segments": [{"start": i, "end": i + 1}
                                          for i in range(10)]})) == 4)

check("пустой ввод не роняет", _extract_segments({}) == [])

# ===========================================================================
print("\n--- Сохранение и чтение момента ---")

m = Moment(title="Заголовок", hook="Крючок", theme="тема", reason="причина",
           virality=8, segments=[(1.0, 5.0), (9.0, 12.5)])
back = _moment_from_dict(_moment_to_dict(m))
check("момент переживает сохранение без потерь",
      (back.title, back.virality, back.segments) == (m.title, m.virality, m.segments),
      back)

# ===========================================================================
print("\n--- Отбрасывание пересечений ---")

a = Moment("A", "h", "t", "r", 9, [(0.0, 30.0)])
b = Moment("B", "h", "t", "r", 7, [(2.0, 28.0)])       # почти внутри A
c = Moment("C", "h", "t", "r", 8, [(60.0, 90.0)])      # отдельный
kept = _drop_heavy_overlaps([a, b, c])
titles = [x.title for x in kept]
check("сильно пересекающийся клип убран", "B" not in titles, titles)
check("непересекающиеся сохранены", {"A", "C"} <= set(titles), titles)

# ===========================================================================
print("\n--- Подгонка границ ---")

segs = [Segment(0.0, 6.0, "Привет это тест речи здесь.",
                [Word(0.0, 0.6, "Привет"), Word(0.7, 1.1, "это"),
                 Word(1.2, 1.7, "тест"), Word(1.8, 2.3, "речи"),
                 Word(2.4, 3.0, "здесь.")])]

res = _snap_to_words(0.65, 2.35, segs)
check("границы возвращены", res is not None)
if res:
    a, b = res
    # Границы намеренно раздвигаются: 0.15 с до начала и 0.35 с после конца,
    # чтобы речь не обрывалась вплотную к первому и последнему слову.
    # Границы предпочтительно тянутся к предложениям, а не к словам: клип
    # не должен начинаться и заканчиваться на полумысли.
    check("начало притянуто к началу предложения", abs(a - 0.0) < 0.02, a)
    check("конец притянут к концу предложения с запасом", b > 3.0, b)
    check("начало не уходит в минус", a >= 0.0, a)
    check("конец позже начала", b > a, (a, b))

res0 = _snap_to_words(0.0, 1.0, segs)
check("клип от нуля не даёт отрицательного начала",
      res0 and res0[0] >= 0.0, res0)

check("без слов возвращаются исходные границы",
      _snap_to_words(1.0, 5.0, []) == (1.0, 5.0),
      _snap_to_words(1.0, 5.0, []))

check("ближайшее значение в пределах допуска",
      _nearest(1.0, [0.9, 5.0], tol=0.5) == 0.9)
check("вне допуска — None", _nearest(1.0, [5.0], tol=0.5) is None)
check("без допуска берётся ближайшее", _nearest(1.0, [5.0], tol=None) == 5.0)
check("пустой список кандидатов", _nearest(1.0, [], tol=None) is None)

print("\n--- Слова клипа ---")

words = clip_words(segs, 1.2, 3.0)
check("взяты только слова внутри клипа",
      [w.text for w in words] == ["тест", "речи", "здесь."],
      [w.text for w in words])
check("время пересчитано от начала клипа",
      abs(words[0].start - 0.0) < 0.01, words[0].start)
check("отрицательного времени не бывает",
      all(w.start >= 0 and w.end >= 0 for w in words))

# ===========================================================================
print("\n--- Группировка строк субтитров ---")

def W(t):
    return Word(0.0, 0.5, t)

groups = _group_words([W("Привет"), W("мир"), W("это"), W("тест")], max_chars=40)
check("строки не длиннее двух слов",
      all(len(g) <= 2 for g in groups), [[w.text for w in g] for g in groups])

groups = _group_words([W("Конец."), W("Начало"), W("новой")], max_chars=40)
check("после точки строка закрывается",
      [w.text for w in groups[0]] == ["Конец."],
      [[w.text for w in g] for g in groups])

long_word = "Ставосьмидесятивосьмимиллиметровый"
groups = _group_words([W(long_word), W("и")], max_chars=10)
check("слишком длинное слово не теряется",
      any(long_word in [w.text for w in g] for g in groups),
      [[w.text for w in g] for g in groups])

check("пустой список слов не роняет", _group_words([], 40) == [])

all_words = [W("раз"), W("два"), W("три"), W("четыре"), W("пять")]
groups = _group_words(all_words, 40)
flat = [w.text for g in groups for w in g]
check("ни одно слово не потеряно при группировке",
      flat == [w.text for w in all_words], flat)

# ===========================================================================
print("\n--- Формат ASS ---")

check("время 0", _ass_ts(0) == "0:00:00.00", _ass_ts(0))
check("время 1 час", _ass_ts(3600) == "1:00:00.00", _ass_ts(3600))
check("сотые доли", _ass_ts(65.43).startswith("0:01:05.4"), _ass_ts(65.43))
check("отрицательное время не ломает формат",
      ":" in _ass_ts(-1), _ass_ts(-1))

check("перевод строки заменён на \\N",
      _esc("две\nстроки") == "две\\Nстроки", repr(_esc("две\nстроки")))
check("реплика не разрывается переносом",
      "\n" not in _esc("а\r\nб"), repr(_esc("а\r\nб")))
check("фигурные скобки обезврежены",
      "{" not in _esc("{\\an8}вредное"), _esc("{\\an8}вредное"))

# ===========================================================================
print("\n--- Сборка файла субтитров ---")

from clipper.subtitles import build_ass  # noqa: E402

cfg = Config()
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "s.ass"
    build_ass([Word(0.0, 0.5, "Привет"), Word(0.6, 1.2, "мир!")], cfg, p)
    text = p.read_text(encoding="utf-8")
    check("файл создан", p.exists() and len(text) > 100)
    check("есть заголовок Script Info", "[Script Info]" in text)
    check("есть секция событий", "[Events]" in text)
    check("текст попал в файл", "Привет" in text)
    # Подсветка активного слова сделана не тегами \k, а отдельной репликой
    # на каждую строку со сменой цвета и размера — так надёжнее в ffmpeg.
    check("подсветка активного слова присутствует",
          "\\c&H" in text and "\\fscx" in text, text[-400:])
    check("каждая реплика на одной строке",
          all(l.count("Dialogue:") <= 1 for l in text.split("\n")))

    p2 = Path(td) / "empty.ass"
    build_ass([], cfg, p2)
    check("пустые субтитры не роняют сборку", p2.exists())

print(f"\n===== Пройдено: {len(PASS)}, провалено: {len(FAIL)} =====")
if FAIL:
    print("Провалены:", *FAIL, sep="\n  - ")
sys.exit(1 if FAIL else 0)
