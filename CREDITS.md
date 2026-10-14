# Сторонние идеи и алгоритмы

- **One Euro Filter** (clipper/one_euro.py) — адаптивное сглаживание.
  Алгоритм: Casiez, Roussel, Vogel, "1€ Filter", CHI 2012 (публичный).
  Реализация написана с нуля. Идею подсмотрели в gauravzazz/smart-reframe.
- **Выбор главного лица по важности** (primary+buddy) — идея из
  gauravzazz/smart-reframe, переписана под нашу архитектуру.
- **Сигналы виральности в промпте отбора** — вдохновлено промптом из
  SamurAIGPT/AI-Youtube-Shorts-Generator (идеи, не код).

Изучены, но НЕ использованы: ClipsAI/clipsai (MIT), modelscope/FunClip (MIT),
KazKozDev/auto-vertical-reframe (MIT) — другая архитектура/тяжёлые зависимости.

## Обновление (v4.3)
- **Debug-кадры с разметкой** — идея из KazKozDev/auto-vertical-reframe (draw_debug).
- **Continuity-бонус выбора лица** (lock_match) — идея из auto-vertical-reframe.
- **Re-ID лиц (заготовка)**: модели SFace + YuNet из OpenCV Zoo (Apache-2.0),
  идея Face Re-ID из gauravzazz/smart-reframe. Полная интеграция «имён по
  фото» — в отдельном заходе.
- Дедупликация клипов (overlap suppression, идея из AI-Youtube-Shorts-Generator)
  уже была реализована ранее в _drop_heavy_overlaps — не дублировали.
