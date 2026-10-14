"""Шаг 4а. Автокадрирование 9:16 с переключением на активного говорящего.

Как это работает:
  1. Детектор (ONNX Ultra-Light RFB-320 из assets/) находит ВСЕ лица в кадре.
  2. Для каждого лица измеряется "речевая активность" — движение в нижней
     половине рамки лица (рот/челюсть) между двумя кадрами с шагом ~120 мс.
  3. Камера держится на текущем говорящем; переключается ЖЁСТКОЙ СКЛЕЙКОЙ
     (мгновенный прыжок, как в проф. сервисах), только если другой человек
     стабильно активнее два замера подряд — защита от ложных переключений.
  4. Внутри одного говорящего движение окна плавно сглаживается.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config

_ASSETS = Path(__file__).resolve().parent.parent / "assets"

# порог: скачок центра больше этой доли ширины кадра = склейка, а не панорама
_CUT_JUMP_FRAC = 0.18


@dataclass
class CropPlan:
    """План кадрирования: размер окна, вертикальный сдвиг и траектория центра."""
    crop_w: int
    crop_h: int
    src_w: int
    src_h: int
    keyframes: list  # [(t_rel, x_center, is_cut), ...]
    y0: int = 0      # верх окна кропа (умный zoom центрирует лицо по вертикали)
    mode: str = "single"       # "single" | "split" (диалог) | "wide" (общий план)
    split_xa: float = 0.0      # центры окон для сплит-скрина
    split_xb: float = 0.0


# ----------------------------------------------------------------- детекторы

def _nms(boxes, scores, iou_thr=0.4):
    """Простой NMS: подавляет дубли рамок одного лица."""
    import numpy as np
    idx = scores.argsort()[::-1]
    keep = []
    while len(idx) > 0:
        i = idx[0]
        keep.append(i)
        if len(idx) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idx[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idx[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idx[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idx[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = ((boxes[idx[1:], 2] - boxes[idx[1:], 0])
                  * (boxes[idx[1:], 3] - boxes[idx[1:], 1]))
        iou = inter / np.maximum(area_i + area_r - inter, 1e-6)
        idx = idx[1:][iou < iou_thr]
    return keep


def _make_detector(cv2, src_w: int, src_h: int):
    """Возвращает detect(frame_bgr) -> список лиц [(x1,y1,x2,y2), ...] в пикселях.

    Приоритет: ONNX (не зависит от версии OpenCV) -> cv2.dnn -> Haar -> None.
    """
    for model_name, in_w, in_h, tag in [
            ("rfb640.onnx", 640, 480, "RFB-640 (усиленный)"),
            ("face_rfb_320.onnx", 320, 240, "RFB-320")]:
        onnx_model = _ASSETS / model_name
        if not onnx_model.exists():
            continue
        try:
            import numpy as np
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            sess = ort.InferenceSession(str(onnx_model), sess_options=opts,
                                        providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            iw, ih = in_w, in_h

            def detect_onnx(frame, _sess=sess, _in=input_name, _iw=iw, _ih=ih):
                rgb = frame[:, :, ::-1]
                im = cv2.resize(rgb, (_iw, _ih)).astype(np.float32)
                x = ((im - 127.0) / 128.0).transpose(2, 0, 1)[None]
                confidences, boxes = _sess.run(None, {_in: x})
                conf = confidences[0][:, 1]
                mask = conf > 0.75            # выше порог: меньше ложных «лиц»
                if not mask.any():
                    return []
                b, s = boxes[0][mask], conf[mask]
                keep = _nms(b, s)
                scale = np.array([src_w, src_h, src_w, src_h])
                return [tuple((b[i] * scale).astype(float)) for i in keep]

            print(f"[face_crop] Детектор лиц: ONNX Ultra-Light {tag} (из assets/)")
            return detect_onnx
        except Exception as e:  # noqa: BLE001
            print(f"[face_crop] {model_name} недоступен ({type(e).__name__}: {e})")

    proto = _ASSETS / "deploy.prototxt"
    weights = _ASSETS / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    if proto.exists() and weights.exists():
        try:
            import numpy as np
            net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))

            def detect_dnn(frame):
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)
                net.setInput(blob)
                det = net.forward()
                out = []
                for i in range(det.shape[2]):
                    if float(det[0, 0, i, 2]) > 0.5:
                        x1, y1, x2, y2 = det[0, 0, i, 3:7]
                        out.append((float(x1) * src_w, float(y1) * src_h,
                                    float(x2) * src_w, float(y2) * src_h))
                return out

            print("[face_crop] Детектор лиц: DNN res10 SSD (из assets/)")
            return detect_dnn
        except Exception as e:  # noqa: BLE001
            print(f"[face_crop] DNN-детектор недоступен ({type(e).__name__}: {e})")

    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not cascade.empty():
            def detect_haar(frame):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=6,
                    minSize=(src_h // 12, src_h // 12))
                return [(float(x), float(y), float(x + w), float(y + h))
                        for x, y, w, h in faces]

            print("[face_crop] Детектор лиц: каскад Хаара")
            return detect_haar
    except Exception as e:  # noqa: BLE001
        print(f"[face_crop] Каскад Хаара недоступен ({type(e).__name__}: {e})")

    return None


# ------------------------------------------------- выбор активного говорящего

def _plausible_face(box, src_w: int, src_h: int) -> bool:
    """Анти-мусор: отсекает рамки, не похожие на лицо человека в кадре.

    Машины, номера, логотипы обычно дают широкие или крошечные рамки."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if h < src_h * 0.05 or w <= 0:          # слишком мелкое — фон/мусор
        return False
    aspect = w / h
    # лицо — вертикальный овал (0.6-1.0). Плечи, торс, шея в тёмной одежде
    # дают более широкие/квадратные рамки — их отсекаем (было 0.5-1.15)
    return 0.6 <= aspect <= 1.0


def _persistent(faces: list, prev_faces: list, src_w: int) -> list:
    """Лицо учитывается, только если оно было и в предыдущем замере рядом —
    одиночные вспышки детектора (блики, машины) отсеиваются."""
    if not prev_faces:
        return []
    out = []
    for b in faces:
        cx = (b[0] + b[2]) / 2
        if any(abs(cx - (p[0] + p[2]) / 2) < src_w * 0.08 for p in prev_faces):
            out.append(b)
    return out


def _reject_static(faces: list, gray_a, gray_prev, src_w, src_h) -> list:
    """Отсеивает НЕПОДВИЖНЫЕ рамки — маски на стене, портреты, манекены.

    Живое лицо между двумя кадрами всегда чуть меняется (моргание, микро-
    движение, блик), нарисованная маска — нет. Порог мягкий, чтобы не
    выкинуть человека, застывшего на секунду."""
    if gray_prev is None:
        return faces
    import numpy as np
    out = []
    for b in faces:
        x1, y1, x2, y2 = (int(max(0, b[0])), int(max(0, b[1])),
                          int(min(src_w, b[2])), int(min(src_h, b[3])))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        a = gray_a[y1:y2, x1:x2].astype(np.float32)
        p = gray_prev[y1:y2, x1:x2].astype(np.float32)
        if a.shape != p.shape:
            out.append(b)
            continue
        if float(np.abs(a - p).mean()) >= 0.8:   # хоть немного меняется -> живое
            out.append(b)
    return out


def _speech_score(gray_a, gray_b, box, src_w, src_h) -> float:
    """Насколько похоже, что человек ГОВОРИТ.

    Сравниваем движение зоны рта (нижние 45% рамки) с движением зоны
    глаз/лба (верхние 40%). Тряска камеры и повороты головы двигают ОБЕ
    зоны одинаково и взаимно вычитаются; речь двигает только нижнюю."""
    import numpy as np
    x1, y1, x2, y2 = box
    h = y2 - y1
    def region_motion(ry1, ry2):
        mx1, mx2 = int(max(0, x1)), int(min(src_w, x2))
        my1, my2 = int(max(0, ry1)), int(min(src_h, ry2))
        if mx2 - mx1 < 4 or my2 - my1 < 4:
            return 0.0
        a = gray_a[my1:my2, mx1:mx2].astype(np.float32)
        b = gray_b[my1:my2, mx1:mx2].astype(np.float32)
        return float(np.abs(a - b).mean())
    mouth = region_motion(y2 - 0.45 * h, y2)
    ref = region_motion(y1, y1 + 0.40 * h)
    return max(0.0, mouth - 0.7 * ref)


_SPEECH_MIN = 1.1        # ниже — считаем, что явного говорящего нет
_DEADBAND_FRAC = 0.08    # мёртвая зона: цель ближе 8% ширины — камера стоит
_VOTES_TO_CUT = 2        # столько замеров подряд нужно новой цели для склейки


# ----------------------------------------------------- ASD-переоценка речи

def _rescore_with_asd(samples, asd_frames, video, clip_start, clip_end,
                      scorer, src_w, src_h):
    """Заменяет визуальную оценку речи (poле [1] в scored) на оценку Light-ASD.

    Для каждой позиции лица собираем его кроп во всех замерах (трек),
    прогоняем трек + звук клипа через ASD и раскладываем оценки обратно."""
    import cv2
    import numpy as np
    from .asd import extract_audio_mfcc

    audio_mfcc = extract_audio_mfcc(video, clip_start, clip_end)

    # группируем лица в треки по близости центра (грубый трекинг по X)
    tracks: list = []   # каждый: {"cx": x, "items": [(sample_idx, face_idx, crop)]}
    for si, (t_rel, gray, boxes) in enumerate(asd_frames):
        for fi, b in enumerate(boxes):
            cx = (b[0] + b[2]) / 2
            crop = _face_crop_112(gray, b, src_w, src_h, cv2)
            track = next((t for t in tracks
                          if abs(t["cx"] - cx) < src_w * 0.08), None)
            if track is None:
                track = {"cx": cx, "items": []}
                tracks.append(track)
            track["cx"] = 0.7 * track["cx"] + 0.3 * cx
            track["items"].append((si, fi, crop))

    # оценка речи по каждому треку -> обратно в samples
    new_scores = {}   # (sample_idx, face_idx) -> asd_raw
    for track in tracks:
        crops = [it[2] for it in track["items"]]
        need = len(crops) * 4
        af = audio_mfcc[:need] if len(audio_mfcc) >= need else np.pad(
            audio_mfcc, ((0, need - len(audio_mfcc)), (0, 0)), mode="edge")
        scores = scorer.score_track(crops, af)
        for (si, fi, _), sc in zip(track["items"], scores):
            new_scores[(si, fi)] = sc

    # калибровка: ASD выдаёт узкий диапазон (~0.3-0.6), абсолютный порог не
    # подходит. Нормализуем по всему клипу в шкалу [0..~15], сохраняя
    # относительный порядок — говорит тот, чья оценка заметно выше фоновой.
    raw = np.array(list(new_scores.values())) if new_scores else np.array([0.0])
    lo = float(np.percentile(raw, 40))       # «молчит» — типичный низкий уровень
    hi = float(np.percentile(raw, 95)) if len(raw) > 2 else lo + 0.1
    span = max(hi - lo, 0.05)

    def to_speech_scale(v: float) -> float:
        # lo -> 0, hi -> ~2*_SPEECH_MIN; ниже lo зануляем
        return max(0.0, (v - lo) / span) * (_SPEECH_MIN * 2.2)

    # пересобираем samples с новыми оценками речи
    out = []
    for si, (t_rel, scored) in enumerate(samples):
        new_scored = []
        for fi, (cx, _old, b) in enumerate(scored):
            sc = to_speech_scale(new_scores.get((si, fi), lo))
            new_scored.append((cx, sc, b))
        out.append((t_rel, new_scored))
    n_speaking = sum(1 for v in new_scores.values()
                     if to_speech_scale(v) >= _SPEECH_MIN)
    print(f"[asd] Переоценено {len(new_scores)} лиц в {len(tracks)} треках "
          f"(порог речи: raw>{lo + span*_SPEECH_MIN/(_SPEECH_MIN*2.2):.2f}, "
          f"{n_speaking} говорящих кадров)")
    return out


def _face_crop_112(gray, box, src_w, src_h, cv2):
    """Серый кроп лица 112x112 для Light-ASD."""
    import numpy as np
    x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])),
                      int(min(src_w, box[2])), int(min(src_h, box[3])))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((112, 112), dtype=np.float32)
    face = gray[y1:y2, x1:x2]
    return cv2.resize(face, (112, 112)).astype(np.float32)


# ----------------------------------------------------- проход 2: режиссура

_PREROLL_SAMPLES = 4      # склейка на говорящего за N замеров ДО начала речи (J-cut)
                          # с точным ASD упреждаем смелее — камера ждёт говорящего
_MIN_HOLD_SAMPLES = 6     # владение кадром короче N замеров = мигание, поглощается
                          # (2с при 3 замерах/с — камера не скачет по толпе)


def _direct_offline(samples: list, src_w: int, crop_w: int, cfg: Config,
                    src_h: int = 1080) -> list:
    """Проход 2. Зная всю таблицу замеров, строит траекторию камеры:
    целевая позиция на каждый замер -> устойчивые отрезки владения ->
    упреждение склеек (J-cut) -> сглаживание внутри отрезков.

    Возвращает keyframes [(t_rel, x_center, is_cut), ...].
    """
    if not samples:
        return [(0.0, src_w / 2, False)]

    # 1) сырая цель на каждый замер (кто в фокусе), без решений о склейке.
    #    Передаём предыдущую позицию: при равнозначных кандидатах камера
    #    предпочитает ОСТАТЬСЯ на том, кого уже показывает (continuity-бонус,
    #    идея lock_match из auto-vertical-reframe) — меньше нервных перескоков.
    targets: list = []   # (t_rel, x|None, kind)
    prev_target_x = None
    for t_rel, scored in samples:
        tx, kind = _frame_target(scored, crop_w, src_h, prev_x=prev_target_x)
        targets.append((t_rel, tx, kind))
        if tx is not None:
            prev_target_x = tx

    # 2) заполнить "дыры" (object/None). Ключевой момент: ОДИНОЧНЫЙ пропуск
    #    детекции посреди сцены с лицом НЕ должен кидать камеру в центр —
    #    иначе рывок туда-обратно. Короткие дыры (<= _HOLD_GAP замеров)
    #    держат последнюю позицию лица; только длинная пустота (реально нет
    #    людей в кадре) уводит камеру в центр.
    _HOLD_GAP = 3
    last_x = next((x for _, x, k in targets if x is not None), src_w / 2)
    filled: list = []
    i = 0
    while i < len(targets):
        t_rel, x, kind = targets[i]
        if x is not None:
            last_x = x
            filled.append((t_rel, x, kind))
            i += 1
            continue
        # начало дыры — измеряем её длину
        j = i
        while j < len(targets) and targets[j][1] is None:
            j += 1
        gap_len = j - i
        # есть ли лицо ПОСЛЕ дыры (чтобы держать позицию между двумя лицами)?
        has_after = j < len(targets)
        for k in range(i, j):
            if gap_len <= _HOLD_GAP and has_after:
                filled.append((targets[k][0], last_x, "hold"))
            else:
                filled.append((targets[k][0], src_w / 2, "object"))
        i = j

    # 3) отрезки владения: подряд идущие замеры с близкой целью — один "план"
    runs = _own_runs(filled, src_w)

    # 4) выкинуть слишком короткие отрезки (мигание) — поглощаются соседями
    runs = _absorb_flicker(runs)

    # 4.5) ситуативная ПАНОРАМА: длинный статичный отрезок, где в кадре
    #      несколько разнесённых людей и нет явного говорящего — заменяем
    #      медленным проездом вдоль ряда (вместо замирания на одном лице)
    pan = _pan_keyframes(filled, runs, samples, src_w, crop_w)
    if pan is not None:
        return pan

    # 5) построить keyframes: внутри отрезка позиция сглажена, на границе —
    #    склейка, СДВИНУТАЯ на упреждение к началу речи говорящего
    return _runs_to_keyframes(filled, runs, src_w, cfg)


_PAN_MIN_SEC = 3.5        # проезд включается только для статики длиннее ~3.5с
_PAN_SPREAD = 0.55        # люди должны занимать хотя бы 55% ширины кадра


def _pan_keyframes(filled, runs, samples, src_w, crop_w):
    """Плавный проезд вдоль ряда людей — если сцена одна, длинная и статичная.

    Условия (все обязательны):
      - сегмент — один сплошной отрезок владения (без склеек между людьми);
      - длится дольше _PAN_MIN_SEC;
      - в кадре стабильно несколько лиц, разнесённых по ширине;
      - нет одного явного доминирующего говорящего (иначе нужен он, не проезд).
    Возвращает keyframes проезда слева-направо или None (тогда обычная логика)."""
    if len(runs) != 1:
        return None
    if not filled:
        return None
    dur = filled[-1][0] - filled[0][0]
    if dur < _PAN_MIN_SEC:
        return None

    # сколько лиц и как разнесены, усреднённо по сегменту
    spreads, counts, speak_frac = [], [], 0
    for _, scored in samples:
        if len(scored) >= 2:
            cs = [f[0] for f in scored]
            spreads.append(max(cs) - min(cs))
            counts.append(len(scored))
        if any(s >= _SPEECH_MIN for _, s, _ in scored):
            speak_frac += 1
    if len(spreads) < len(samples) * 0.5:      # чаще всего в кадре 0-1 лицо
        return None
    import statistics
    med_spread = statistics.median(spreads)
    if med_spread < crop_w * _PAN_SPREAD:      # люди сидят кучно — проезд не нужен
        return None
    # если кто-то явно говорит бОльшую часть времени — это его сцена, не проезд
    if speak_frac > len(samples) * 0.35:
        return None

    # крайние позиции ряда: левый и правый человек
    left_x = min(min(f[0] for f in sc) for _, sc in samples if sc)
    right_x = max(max(f[0] for f in sc) for _, sc in samples if sc)
    half = crop_w / 2
    x_start = max(half, min(src_w - half, left_x))
    x_end = max(half, min(src_w - half, right_x))

    # медленный линейный проезд с мягким входом/выходом (ease-in-out)
    kf = []
    t0, t1 = filled[0][0], filled[-1][0]
    for t_rel, _, _ in filled:
        p = (t_rel - t0) / max(t1 - t0, 1e-6)
        p_ease = p * p * (3 - 2 * p)           # smoothstep — плавные концы
        x = x_start + (x_end - x_start) * p_ease
        kf.append((t_rel, x, False))
    print("[face_crop] Ситуативная панорама: плавный проезд вдоль ряда людей")
    return kf


def _box_h(box) -> float:
    return box[3] - box[1]


def _dominant_faces(scored: list, src_h: int) -> list:
    """Отсекает мелкие лица-врезки/фон в присутствии крупного лица.

    В смонтированном интервью часто в кадре одновременно: крупный план
    героя (лицо 40-70% высоты) И мелкие лица врезки-коллажа или толпы в
    углу (5-18% высоты). Наводить камеру/сплит на мелкие — ошибка: это
    фон, а не объект сцены. Оставляем лица, чей размер сопоставим с
    крупнейшим в этом кадре, и не микроскопические.

    Правило: лицо остаётся, если его высота >= 45% высоты крупнейшего лица
    в кадре И >= 9% высоты всего кадра. Если так отфильтровали всё (лица
    все мелкие — дальний план целиком) — возвращаем исходный список."""
    if len(scored) <= 1:
        return scored
    heights = [_box_h(f[2]) for f in scored]
    max_h = max(heights)
    if max_h <= 0:
        return scored
    kept = [f for f, h in zip(scored, heights)
            if h >= 0.45 * max_h and h >= src_h * 0.09]
    return kept if kept else scored


_WIDE_FACE_FRAC = 0.14   # крупнейшее лицо мельче этой доли высоты -> общий план


def _frame_target(scored: list, crop_w: int, src_h: int = 1080, prev_x=None):
    """Куда СЛЕДУЕТ смотреть в этом замере. -> (x|None, kind).

    kind: "speak"/"group"/"object"/"wide". "wide" = общий/дальний план (в
    кадре только мелкие лица, толпа/студия) — наводиться не на кого, кадр
    показываем целиком летербоксом, а не зумим в мыло.

    prev_x — где камера была в прошлом замере. При выборе среди нескольких
    равнозначных лиц даём бонус тому, кто ближе к prev_x (continuity): камера
    не перескакивает на чуть более «важного» соседа из-за мелкой разницы.

    Важно: НИКОГДА не ставим камеру в среднее между двумя разнесёнными
    людьми (это давало «двух носов по краям»). Мелкие лица-врезки/фон
    игнорируются (_dominant_faces)."""
    if not scored:
        return None, "object"
    # общий план: даже самое крупное лицо мелкое -> наводиться бессмысленно
    # (получится зум в толпу/мыло). Показываем весь кадр летербоксом.
    max_h = max(_box_h(f[2]) for f in scored)
    if max_h < src_h * _WIDE_FACE_FRAC:
        cx = sum(f[0] for f in scored) / len(scored)
        return cx, "wide"
    scored = _dominant_faces(scored, src_h)
    speaking = [f for f in scored if f[1] >= _SPEECH_MIN]
    if speaking:
        best = max(speaking, key=lambda f: f[1])
        # явный одиночный говорящий?
        others = [f[1] for f in speaking if f is not best]
        if not others or best[1] >= 1.4 * max(others):
            return best[0], "speak"
        # двое говорят одновременно и близко — держим группу (влезают вместе)
    centers = [f[0] for f in scored]
    spread = max(centers) - min(centers)
    if spread <= crop_w * 0.6:
        # люди помещаются в один кадр без «носов» — можно держать группу
        return sum(centers) / len(centers), "group"
    # разнесены широко и никто явно не говорит: НЕ усредняем (иначе носы).
    # Держим ОДНО лицо — самое «важное». Важность = размер лица (крупное = ближе
    # к камере = главный герой) + бонус за речь + continuity-бонус за близость к
    # текущей позиции камеры (идея primary/lock_match из smart-reframe и
    # auto-vertical-reframe): не перескакиваем на соседа из-за мелкой разницы.
    def _importance(f):
        area = (f[2][2] - f[2][0]) * (f[2][3] - f[2][1])
        speech_bonus = 1.0 + 0.6 * min(f[1] / max(_SPEECH_MIN, 1e-6), 2.0)
        lock_bonus = 1.0
        if prev_x is not None and abs(f[0] - prev_x) < crop_w * 0.5:
            lock_bonus = 1.25          # уже смотрим примерно сюда — держимся
        return area * speech_bonus * lock_bonus
    pick = max(scored, key=_importance)
    return pick[0], "group"


def _own_runs(filled: list, src_w: int) -> list:
    """Сливает соседние замеры с близкой целью в отрезки [i0, i1, x_median, kind]."""
    import statistics
    runs = []
    i = 0
    while i < len(filled):
        j = i
        xs = [filled[i][1]]
        while (j + 1 < len(filled)
               and abs(filled[j + 1][1] - statistics.median(xs)) < src_w * 0.10):
            j += 1
            xs.append(filled[j][1])
        # тип отрезка — самый "сильный" среди входящих (speak > group > object).
        # "hold" — удержание позиции лица в короткой дыре — считаем как group.
        kinds = [filled[k][2] for k in range(i, j + 1)]
        kind = ("speak" if "speak" in kinds
                else "group" if ("group" in kinds or "hold" in kinds)
                else "object")
        runs.append([i, j, statistics.median(xs), kind])
        i = j + 1
    return runs


def _absorb_flicker(runs: list) -> list:
    """Короткие отрезки (< _MIN_HOLD) вливаются в соседний, чтобы камера не
    прыгала на человека, который говорит меньше секунды."""
    if len(runs) <= 1:
        return runs
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for idx, r in enumerate(runs):
            length = r[1] - r[0] + 1
            if length >= _MIN_HOLD_SAMPLES:
                continue
            # выбираем соседа, к которому приклеиться (более длинного)
            left = runs[idx - 1] if idx > 0 else None
            right = runs[idx + 1] if idx + 1 < len(runs) else None
            pick = left if (right is None or
                            (left is not None and
                             (left[1] - left[0]) >= (right[1] - right[0]))) else right
            pick[0] = min(pick[0], r[0])
            pick[1] = max(pick[1], r[1])
            runs.pop(idx)
            changed = True
            break
    return runs


def _runs_to_keyframes(filled: list, runs: list, src_w: int, cfg: Config) -> list:
    """Отрезки владения -> keyframes с упреждением склейки и сглаживанием."""
    half_dead = src_w * _DEADBAND_FRAC
    keyframes: list = []
    prev_x = None
    for ri, (i0, i1, x_run, kind) in enumerate(runs):
        # упреждение: если отрезок начинается с речи, сдвигаем начало склейки
        # на _PREROLL раньше — камера встаёт на говорящего заранее (J-cut)
        cut_i = i0
        if kind == "speak" and ri > 0:
            cut_i = max(runs[ri - 1][0] + 1, i0 - _PREROLL_SAMPLES)

        is_cut = prev_x is not None and abs(x_run - prev_x) > half_dead
        for k in range(i0, i1 + 1):
            t_rel = filled[k][0]
            if k == i0 and is_cut:
                # склейку ставим на упреждённый кадр
                t_cut = filled[cut_i][0]
                keyframes.append((t_cut, x_run, True))
                if t_rel > t_cut:
                    keyframes.append((t_rel, x_run, False))
            elif kind == "speak":
                # говорящий не движется — камера стоит НАМЕРТВО на его позиции,
                # никакого сглаживания (оно только добавляло вялости по горизонтали)
                keyframes.append((t_rel, x_run, False))
            else:
                # только для группы/объекта — лёгкое сглаживание дрейфа
                if keyframes and not is_cut:
                    px = keyframes[-1][1]
                    x_sm = cfg.crop_smoothing * px + (1 - cfg.crop_smoothing) * x_run
                else:
                    x_sm = x_run
                keyframes.append((t_rel, x_sm, False))
            is_cut = False
        prev_x = x_run
    keyframes.sort(key=lambda kf: kf[0])
    keyframes = _smooth_between_cuts(keyframes)
    return keyframes


def _smooth_between_cuts(keyframes: list) -> list:
    """One Euro-сглаживание позиции ВНУТРИ непрерывных участков; на каждой
    склейке (is_cut) фильтр сбрасывается, чтобы склейка осталась резкой.

    Убирает микро-дрожь дрейфа на группе/панораме, не внося задержки: при
    резком движении цели фильтр реагирует мгновенно (см. one_euro)."""
    from .one_euro import OneEuroFilter
    if len(keyframes) < 3:
        return keyframes
    out = []
    f = OneEuroFilter()
    for t, x, is_cut in keyframes:
        if is_cut:
            f = OneEuroFilter()          # новый участок — сброс истории
            out.append((t, x, True))
            f(x, t)                      # инициализируем фильтр позицией склейки
        else:
            out.append((t, f(x, t), False))
    return out


# ----------------------------------------------------- (старый онлайн-режиссёр)

class ShotDirector:
    """Камера-штатив: стоит на месте и переключается СКЛЕЙКАМИ, не ездит.

    Логика кадра на каждый замер:
      - есть явный говорящий -> кадр на нём;
      - никто явно не говорит, лица помещаются в кроп -> кадр на ГРУППЕ
        (важно: главный герой сцены не выпадает из кадра);
      - иначе -> самое крупное лицо;
      - лиц нет -> камера стоит где стояла.
    Смена кадра — только если новая цель устойчива _VOTES_TO_CUT замера
    подряд и дальше мёртвой зоны. Тогда жёсткая склейка."""

    def __init__(self, src_w: int, crop_w: int):
        self.src_w = src_w
        self.crop_w = crop_w
        self.x: float | None = None
        self.pend_x: float | None = None
        self.votes = 0
        self.empty_streak = 0   # подряд замеров без лиц (план объекта?)
        self.cooldown = 0       # перезарядка: после склейки новая запрещена
                                 # ~2.7с (8 замеров) — никаких секундных кадров

    def _desired(self, scored: list) -> float | None:
        if not scored:
            return None
        best = max(scored, key=lambda f: f[1])
        if best[1] >= _SPEECH_MIN:
            others = [f for f in scored if f is not best]
            if not others or best[1] >= 1.5 * max(f[1] for f in others):
                return best[0]                       # явный говорящий
        # никто явно не говорит: если текущий кадр УЖЕ держит кого-то из
        # присутствующих — стоим (не мечемся "говорящий <-> группа");
        # групповой/крупный кадр — только когда мы были на плане объекта
        if self.x is not None and any(
                abs(f[0] - self.x) < self.src_w * _DEADBAND_FRAC * 1.6
                for f in scored):
            return None
        centers = [f[0] for f in scored]
        if max(centers) - min(centers) <= self.crop_w * 0.82:
            return sum(centers) / len(centers)       # группа влезает — показываем всех
        big = max(scored, key=lambda f: (f[2][2] - f[2][0]) * (f[2][3] - f[2][1]))
        return big[0]                                # иначе — самый крупный

    _COOLDOWN_SAMPLES = 8

    def _fire_cut(self, x: float) -> tuple[float, bool]:
        self.x = x
        self.pend_x, self.votes = None, 0
        self.cooldown = self._COOLDOWN_SAMPLES
        return self.x, True

    def update(self, scored: list) -> tuple[float | None, bool]:
        if self.cooldown > 0:
            self.cooldown -= 1
        desired = self._desired(scored)
        if desired is None and scored:
            # лица есть, но менять кадр незачем — стоим, счётчики сбрасываем
            self.votes = 0
            self.empty_streak = 0
            return self.x, False
        if desired is None:
            # лиц нет: оператор показывает ОБЪЕКТ (дом, машину, штангу).
            # Такие планы скомпонованы по центру — после 2 пустых замеров
            # склейкой уходим в центр кадра, а не висим на месте лица.
            self.votes = 0
            self.empty_streak += 1
            center = self.src_w / 2
            if (self.empty_streak >= 2 and self.cooldown == 0
                    and self.x is not None
                    and abs(self.x - center) > self.src_w * _DEADBAND_FRAC):
                return self._fire_cut(center)
            return self.x, False
        self.empty_streak = 0
        if self.x is None:
            self.x = desired
            return self.x, False
        if abs(desired - self.x) < self.src_w * _DEADBAND_FRAC:
            self.votes = 0                            # цель рядом — стоим
            return self.x, False
        if self.pend_x is not None and abs(desired - self.pend_x) < self.src_w * 0.08:
            self.votes += 1
        else:
            self.pend_x, self.votes = desired, 1
        if self.votes >= _VOTES_TO_CUT and self.cooldown == 0:
            return self._fire_cut(self.pend_x)        # склейка
        return self.x, False


# ----------------------------------------------------------------- план кропа

def _smart_zoom(boxes: list, crop_w: int, crop_h: int,
                src_w: int, src_h: int, cfg) -> tuple[int, int, int]:
    """Наезд камеры, если люди мелкие в широком плане.

    Целевой размер лица ~26% высоты кадра, лицо в верхней трети (как в проф.
    клипах). Zoom ограничен, чтобы апскейл не разрушил картинку."""
    if len(boxes) < 3:
        return crop_w, crop_h, 0
    import statistics
    # опорное лицо — САМОЕ ВЕРХНЕЕ (голова). Медиана центра ненадёжна: если
    # детектор принял плечо/шею за «лицо» (рамка ниже настоящего лица),
    # медиана съезжает вниз и окно тянет на плечо. Верх кадра — всегда голова.
    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    top_third = boxes_sorted[:max(1, len(boxes_sorted) // 2)]
    face_h = statistics.median(b[3] - b[1] for b in top_third)
    face_cy = statistics.median((b[1] + b[3]) / 2 for b in top_third)
    if face_h <= 0:
        return crop_w, crop_h, 0
    if face_h < src_h * 0.07:
        # лицо мелкое = дальний план или экшн (жим штанги, вид из окна) —
        # зрителю нужен ОБЩИЙ план происходящего, наезд запрещён
        return crop_w, crop_h, 0

    desired_h = face_h / 0.42
    # супер-минимальный наезд: окно не меньше 95% высоты кадра (макс. зум 5%).
    # Наезд едва заметен и служит лишь монтажным акцентом, не «душит» кадр
    floor_h = int(src_h * 0.95)
    new_h = int(round(min(crop_h, max(desired_h, floor_h))))
    if new_h >= crop_h - 4:                       # наезд незначителен — не трогаем
        return crop_w, crop_h, 0
    new_w = int(round(new_h * crop_w / crop_h))
    y0 = int(round(face_cy - 0.36 * new_h))       # лицо в верхней трети
    # ЖЕЛЕЗНАЯ ГАРАНТИЯ: верх самого высокого лица — всегда в кадре
    # (иначе окно уезжает на торс/штаны при движении камеры)
    face_top = min(b[1] for b in boxes)
    y0 = min(y0, int(face_top - 0.06 * new_h))
    y0 = max(0, min(src_h - new_h, y0))
    return new_w, new_h, y0


def plan_shots(video: Path, clip_start: float, clip_end: float, cfg: Config,
              speech_intervals: list | None = None, asd_scorer=None) -> CropPlan:
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("Установите OpenCV: pip install opencv-python-headless") from e

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_h = src_h
    crop_w = int(round(crop_h * cfg.out_width / cfg.out_height))
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(round(crop_w * cfg.out_height / cfg.out_width))

    # режим "статичное вписывание": весь кадр целиком (видно 100% исходника),
    # без обрезки по бокам и без слежения камеры за людьми. Просим только по
    # явному флагу --static-fit — это НЕ замена умной камере, а альтернативный
    # режим для сравнения (нет "не того выделило", нет резких движений камеры,
    # но люди мельче и по бокам/сверху-снизу — размытые полосы).
    if cfg.extra.get("static_fit"):
        cap.release()
        return [(0.0, clip_end - clip_start,
                CropPlan(crop_w, crop_h, src_w, src_h,
                        [(0.0, src_w / 2, False)], 0, "wide"))]

    detect = _make_detector(cv2, src_w, src_h)
    if detect is None:
        print("[face_crop] Ни один детектор не доступен — статичный центральный кроп. "
              "Проверьте папку assets/ и pip install onnxruntime")
        cap.release()
        return [(0.0, clip_end - clip_start,
                CropPlan(crop_w, crop_h, src_w, src_h, [(0.0, src_w / 2, False)]))]

    half = crop_w / 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 240:
        fps = 25.0
    stride = max(1, round(fps / cfg.face_sample_fps))
    gap = max(1, round(fps * 0.12))

    cap.set(cv2.CAP_PROP_POS_MSEC, clip_start * 1000)
    pending = None
    pending_n = -10**9
    prev_faces: list = []
    prev_detect_gray = None       # серый кадр прошлой детекции (для отсева статики)
    speaker_boxes: list = []
    speech_positions: list = []   # где стояли говорящие — для детекции диалога
    empty_samples = 0             # замеры без лиц (планы объектов)
    total_samples = 0
    samples: list = []            # (t_rel, scored) — полная таблица сегмента
    asd_frames: list = []         # (t_rel, gray, boxes) — для переоценки речи ASD
    n = 0
    while True:
        t_abs = clip_start + n / fps
        if t_abs >= clip_end:
            break
        ok, frame = cap.read()
        if not ok:
            break

        if pending is not None and n >= pending_n + gap:
            t_rel, faces, gray_a = pending
            gray_b = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scored = [(((b[0] + b[2]) / 2),
                       _speech_score(gray_a, gray_b, b, src_w, src_h), b)
                      for b in faces]
            # сильный сигнал из транскрипта: если в эту секунду речи НЕТ,
            # никакое движение ртом не считается говорением
            if speech_intervals is not None and not _in_speech(
                    clip_start + t_rel, speech_intervals):
                scored = [(cx, 0.0, b) for cx, _, b in scored]
            for cx, sc, _ in scored:
                if sc >= _SPEECH_MIN:
                    speech_positions.append(cx)
            samples.append((t_rel, scored))
            if not scored:
                empty_samples += 1
            total_samples += 1
            speaker_boxes.extend(b for _, _, b in scored)
            # для ASD: серый кадр каждого замера храним, лица вырежем потом
            if asd_scorer is not None:
                asd_frames.append((t_rel, gray_a, [b for _, _, b in scored]))
            pending = None

        if n % stride == 0 and pending is None:
            raw_faces = [b for b in detect(frame)
                         if _plausible_face(b, src_w, src_h)]
            gray_a = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_faces = _reject_static(raw_faces, gray_a, prev_detect_gray,
                                       src_w, src_h)
            faces = _persistent(raw_faces, prev_faces, src_w)
            prev_faces = raw_faces
            prev_detect_gray = gray_a
            pending = (t_abs - clip_start, faces, gray_a)
            pending_n = n
        n += 1

    cap.release()

    # ASD-переоценка: заменяем визуальную оценку речи на нейросетевую
    # (совпадение губ со звуком) — точнее отвечает, КТО реально говорит
    if asd_scorer is not None and asd_frames:
        try:
            samples = _rescore_with_asd(samples, asd_frames, video,
                                        clip_start, clip_end, asd_scorer, src_w, src_h)
        except Exception as e:  # noqa: BLE001 — сбой ASD не должен ронять клип
            print(f"[asd] Переоценка не удалась ({type(e).__name__}: {e}) — "
                  f"использую визуальную оценку")

    # ПРОХОД 2 — режиссура задним числом: зная всю таблицу, ставим склейки
    # с упреждением (J-cut) и без мигающих переключений
    keyframes = _direct_offline(samples, src_w, crop_w, cfg, src_h)
    if not keyframes:
        keyframes = [(0.0, src_w / 2, False)]

    empty_ratio = empty_samples / total_samples if total_samples else 1.0
    if empty_ratio > 0.30:
        # треть сегмента — планы объектов: наезд отключаем, контекст важнее
        y0, zoom_w, zoom_h = 0, crop_w, crop_h
    else:
        zoom_w, zoom_h, y0 = _smart_zoom(speaker_boxes, crop_w, crop_h,
                                         src_w, src_h, cfg)

    base = CropPlan(zoom_w, zoom_h, src_w, src_h, keyframes, y0)
    plans = _storyboard(base, samples, keyframes, crop_w, crop_h,
                        src_w, src_h, clip_end - clip_start,
                        punch_ok=empty_ratio < 0.15, cfg=cfg)

    # --- режим отладки камеры: дамп того, что «видел» и «решил» движок ---
    if cfg.extra.get("debug_camera"):
        try:
            _dump_camera_debug(cfg, video, clip_start, clip_end, samples,
                               keyframes, plans, src_w, src_h, crop_w, crop_h)
        except Exception as e:  # noqa: BLE001 — отладка не должна ронять рендер
            print(f"[debug-camera] дамп не удался ({type(e).__name__}: {e})")
        try:
            _render_debug_frames(cfg, video, clip_start, clip_end, samples,
                                 keyframes, plans, src_w, src_h, crop_w, crop_h)
        except Exception as e:  # noqa: BLE001
            print(f"[debug-camera] превью-кадры не удались ({type(e).__name__}: {e})")

    return plans


def _render_debug_frames(cfg, video, clip_start, clip_end, samples, keyframes,
                         plans, src_w, src_h, crop_w, crop_h, n_frames: int = 6):
    """Сохраняет несколько кадров клипа с разметкой: рамки всех лиц (серым),
    ВЫБРАННАЯ камерой цель (зелёным), окно кропа/режим (жёлтым). По ним видно
    ГЛАЗАМИ, что решил движок — не нужно читать JSON или писать скрипты.

    Идея из auto-vertical-reframe (draw_debug). Кадры кладём в
    work/<ID>/debug_frames_<start>-<end>/."""
    import subprocess
    import cv2
    import numpy as np

    if not samples:
        return
    out_dir = cfg.workdir / f"debug_frames_{clip_start:.0f}-{clip_end:.0f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # выбираем n_frames равномерно по клипу
    idxs = [int(round(k)) for k in
            np.linspace(0, len(samples) - 1, min(n_frames, len(samples)))]
    kf_by_t = {round(t, 2): (x, c) for t, x, c in keyframes}

    def _plan_at(t_rel):
        for a, b, p in plans:
            if a <= t_rel < b:
                return p
        return plans[-1][2] if plans else None

    for oi, si in enumerate(idxs):
        t_rel, scored = samples[si]
        # извлекаем кадр из исходника через ffmpeg
        tmp = out_dir / f"_raw_{oi}.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{clip_start + t_rel:.3f}",
             "-i", str(video), "-frames:v", "1", "-y", str(tmp)],
            check=False)
        if not tmp.exists():
            continue
        img = cv2.imread(str(tmp))
        if img is None:
            continue

        # цель камеры в этот момент
        kx = kf_by_t.get(round(t_rel, 2), (src_w / 2, False))[0]
        plan = _plan_at(t_rel)
        mode = plan.mode if plan else "single"

        # рамки всех лиц
        for cx, sc, box in scored:
            x1, y1, x2, y2 = [int(v) for v in box]
            is_target = abs(cx - kx) < crop_w * 0.15
            color = (0, 220, 0) if is_target else (150, 150, 150)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if is_target else 1)
            cv2.putText(img, f"s{sc:.1f} h{(y2-y1)/src_h:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # окно кропа (жёлтым) — для single/group; для split/wide подпись
        if mode in ("single",):
            half = crop_w / 2
            cxw = int(max(half, min(src_w - half, kx)))
            cv2.rectangle(img, (int(cxw - half), 0),
                          (int(cxw + half), src_h - 1), (0, 235, 235), 3)
        cv2.putText(img, f"t={t_rel:.1f}s  mode={mode}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 235, 235), 2)

        cv2.imwrite(str(out_dir / f"frame_{oi:02d}_t{t_rel:04.1f}.jpg"), img)
        tmp.unlink(missing_ok=True)

    print(f"[debug-camera] превью-кадры с разметкой: {out_dir}")


def _dump_camera_debug(cfg, video, clip_start, clip_end, samples, keyframes,
                       plans, src_w, src_h, crop_w, crop_h):
    """Пишет рядом с клипом JSON: какие лица найдены в каждый замер, куда
    встала камера (keyframes) и как разложились планы. Плюс печатает сводку.

    Это ГЛАВНЫЙ инструмент диагностики: по нему видно, ПОЧЕМУ камера встала
    не туда — лицо не нашлось, выбрано не то, склейка не там."""
    import json
    dbg = {
        "video": str(video),
        "clip_start": round(clip_start, 2),
        "clip_end": round(clip_end, 2),
        "src_w": src_w, "src_h": src_h, "crop_w": crop_w, "crop_h": crop_h,
        "samples": [
            {"t": round(t, 2),
             "faces": [{"cx": round(cx, 1),
                        "speech": round(sc, 2),
                        "box": [round(v, 1) for v in box],
                        "aspect": round((box[2] - box[0]) / max(box[3] - box[1], 1), 2),
                        "h_frac": round((box[3] - box[1]) / src_h, 2)}
                       for cx, sc, box in scored]}
            for t, scored in samples
        ],
        "keyframes": [{"t": round(t, 2), "x": round(x, 1), "cut": bool(c)}
                      for t, x, c in keyframes],
        "plans": [{"a": round(a, 2), "b": round(b, 2), "mode": p.mode,
                   "crop_w": p.crop_w, "crop_h": p.crop_h, "y0": p.y0,
                   "split_xa": round(p.split_xa, 1), "split_xb": round(p.split_xb, 1),
                   "n_keyframes": len(p.keyframes)}
                  for a, b, p in plans],
    }
    # имя файла — по времени сегмента, чтобы не перетереть соседние
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    name = f"debug_cam_{clip_start:.0f}-{clip_end:.0f}.json"
    out = cfg.workdir / name
    out.write_text(json.dumps(dbg, ensure_ascii=False, indent=1), encoding="utf-8")

    # краткая сводка в консоль — сразу видно проблемные места
    n_empty = sum(1 for _, s in samples if not s)
    n_cuts = sum(1 for _, _, c in keyframes if c)
    n_split = sum(1 for _, _, p in plans if p.mode == "split")
    max_faces = max((len(s) for _, s in samples), default=0)
    print(f"[debug-camera] {name}: замеров={len(samples)} "
          f"(пустых={n_empty}), склеек={n_cuts}, планов={len(plans)} "
          f"(сплит={n_split}), макс.лиц в кадре={max_faces}")
    # предупреждаем о подозрительных местах
    for t, scored in samples:
        for cx, sc, box in scored:
            asp = (box[2] - box[0]) / max(box[3] - box[1], 1)
            if asp < 0.5 or asp > 1.15:
                print(f"[debug-camera]   ⚠ t={t:.1f}s странная рамка "
                      f"aspect={asp:.2f} cx={cx:.0f} — возможно обрезок/мусор")
                break


def _boxes_in(samples: list, a: float, b: float) -> list:
    """Все подтверждённые рамки лиц из замеров интервала [a, b)."""
    return [box for t, scored in samples if a <= t < b
            for _, _, box in scored]


def _wide_intervals(samples: list, src_h: int, dur: float,
                    min_dur: float = 1.5) -> list:
    """Интервалы «общего плана»: в кадре только мелкие лица (толпа/студия/
    дальний план), наводиться не на кого. -> [(a, b), ...].

    Кадр считается общим, если крупнейшее лицо мельче _WIDE_FACE_FRAC высоты.
    Пустые кадры (лиц нет вовсе) не считаем общим планом — там работает
    обычная логика (камера держит последнюю позицию/центр). Терпим короткие
    разрывы, чтобы одиночный крупный кадр не рвал длинный общий план."""
    flags = []
    for t, scored in samples:
        if not scored:
            flags.append((t, None))          # нет данных — не мешаем
            continue
        max_h = max(_box_h(f[2]) for f in scored)
        flags.append((t, max_h < src_h * _WIDE_FACE_FRAC))

    out, i, n = [], 0, len(flags)
    while i < n:
        if flags[i][1] is not True:
            i += 1
            continue
        j = i
        gap = 0
        last_wide = i
        while j + 1 < n:
            nxt = flags[j + 1][1]
            if nxt is True:
                j += 1
                last_wide = j
                gap = 0
            elif nxt is None:               # пустой кадр — терпим внутри
                j += 1
                gap = 0
            else:                            # крупное лицо — считаем разрыв
                gap += 1
                if gap > 2:
                    break
                j += 1
        a = flags[i][0]
        b = min(dur, flags[last_wide][0] + 0.34)
        if b - a >= min_dur:
            out.append((a, b))
        i = j + 1
    return out


def _apply_wide(plans: list, wide: list, crop_w: int, crop_h: int,
                src_w: int, src_h: int) -> list:
    """Заменяет части планов, попадающие в общий план, на mode='wide'
    (летербокс — весь кадр целиком с размытыми полосами)."""
    if not wide:
        return plans
    out = []
    for a, b, p in plans:
        # вырезаем из [a,b] все wide-интервалы, между ними оставляем исходный план
        cuts = sorted((max(a, wa), min(b, wb)) for wa, wb in wide
                      if wb > a and wa < b)
        cur = a
        for wa, wb in cuts:
            if wa > cur + 0.05:
                out.append((cur, wa, p))
            out.append((wa, wb, CropPlan(crop_w, crop_h, src_w, src_h,
                                         [(0.0, src_w / 2, False)], 0, "wide")))
            cur = wb
        if b > cur + 0.05:
            out.append((cur, b, p))
    return out


def _storyboard(base: CropPlan, samples: list, keyframes: list,
                crop_w: int, crop_h: int, src_w: int, src_h: int,
                dur: float, punch_ok: bool = True, cfg=None) -> list:
    """Раскадровка сегмента: [(a_rel, b_rel, CropPlan), ...].

    Сплит-скрин СИТУАТИВЕН: включается только на интервалах, где оба
    собеседника одновременно в кадре и слишком далеко для одного кропа,
    и оба на этом интервале говорили. Показал оператор одного крупно —
    возвращаемся к обычному кадру. Общий/дальний план (толпа) идёт
    летербоксом (mode='wide')."""
    wide = _wide_intervals(samples, src_h, dur)
    runs = _dialog_runs(samples, crop_w, src_h=src_h)
    if not runs:
        plans = [(0.0, dur, base)]
        plans = _punch_in_rhythm(plans, src_h, samples) if punch_ok else plans
        return _apply_wide(plans, wide, crop_w, crop_h, src_w, src_h)

    events: list = []
    t0 = 0.0
    for a, b, xa, xb in runs:
        a, b = max(0.0, a), min(dur, b)
        if a - t0 >= 2.5:
            events.append([t0, a, "single", None])
        # крошечный single перед сплитом поглощается сплитом
        events.append([min(a, max(t0, a)), b, "split", (xa, xb)])
        events[-1][0] = t0 if a - t0 < 2.5 else a
        t0 = b
    if dur - t0 >= 2.5:
        events.append([t0, dur, "single", None])
    elif events:
        events[-1][1] = dur          # хвост короче 2с — доигрывает последний режим

    plans: list = []
    for a, b, mode, extra in events:
        if b - a <= 0.05:
            continue
        if mode == "split":
            xa, xb = extra
            plans.append((a, b, CropPlan(crop_w, crop_h, src_w, src_h,
                                         [(0.0, src_w / 2, False)], 0,
                                         "split", xa, xb)))
        else:
            kf = [(t - a, x, c) for t, x, c in keyframes if a <= t < b]
            if not kf:
                kf = [(0.0, src_w / 2, False)]
            elif kf[0][0] > 0.01:
                kf.insert(0, (0.0, kf[0][1], False))
            # вертикаль окна пересчитываем по лицам ИМЕННО этого плана:
            # при склейке на другого человека окно должно встать на ЕГО лицо,
            # иначе остаётся привязка от прошлого кадра («камера на плече»)
            local = _boxes_in(samples, a, b)
            if len(local) >= 2 and cfg is not None:
                _, _, y_local = _smart_zoom(local, base.crop_w, base.crop_h,
                                            src_w, src_h, cfg)
            else:
                y_local = base.y0
            plans.append((a, b, CropPlan(base.crop_w, base.crop_h, src_w, src_h,
                                         kf, y_local)))
    if punch_ok:
        # панч-наезды — только на сегментах, уверенно ведущихся по лицам;
        # на планах объектов зум попадал бы в случайные куски дома/машины
        plans = _punch_in_rhythm(plans, src_h, samples)
    plans = _apply_wide(plans, wide, crop_w, crop_h, src_w, src_h)
    if len(plans) > 1:
        n_split = sum(1 for _, _, p in plans if p.mode == "split")
        n_wide = sum(1 for _, _, p in plans if p.mode == "wide")
        print(f"[face_crop] Раскадровка: {len(plans)} планов, "
              f"сплит-скрин: {n_split}, общий план (летербокс): {n_wide}")
    return plans


_PUNCH_CHUNK = 5.2      # спокойный план длиннее ~6.5с режем на куски по ~5с
_PUNCH_FACTOR = 1.13    # чередуемый наезд 13% — заметно, но не наляписто


def _punch_in_rhythm(plans: list, src_h: int, samples: list | None = None) -> list:
    """Монтажный ритм как у проф. сервисов: склейка каждые ~5 секунд.

    Длинные однокадровые планы разбиваются на куски с чередованием
    масштаба (обычный / +13%). Смена масштаба = визуальная склейка,
    контент и звук непрерывны."""
    out: list = []
    for a, b, plan in plans:
        dur = b - a
        has_cuts = any(cut for _, _, cut in plan.keyframes)
        # планы, где режиссёр уже клеил, не трогаем — иначе склейка на склейке
        if plan.mode != "single" or has_cuts or dur < _PUNCH_CHUNK * 1.25:
            out.append((a, b, plan))
            continue
        n_chunks = max(2, round(dur / _PUNCH_CHUNK))
        step = dur / n_chunks
        for c in range(n_chunks):
            ca, cb = a + c * step, a + (c + 1) * step
            kf = [(t - (ca - a), x, cut) for t, x, cut in plan.keyframes
                  if ca - a <= t < cb - a]
            if not kf:
                kf = [(0.0, plan.src_w / 2, False)]
            elif kf[0][0] > 0.01:
                kf.insert(0, (0.0, kf[0][1], False))
            # чётность считаем С КОНЦА: последний кусок всегда без наезда,
            # чтобы переход к следующему плану (сплиту) не был "зум -> сплит"
            if (n_chunks - 1 - c) % 2 == 0:
                out.append((ca, cb, CropPlan(plan.crop_w, plan.crop_h,
                                             plan.src_w, plan.src_h, kf, plan.y0)))
            else:
                zw = int(round(plan.crop_w / _PUNCH_FACTOR))
                zh = int(round(plan.crop_h / _PUNCH_FACTOR))
                chunk_boxes = (_boxes_in(samples, ca, cb)
                               if samples is not None else [])
                if zh < src_h * 0.42 or len(chunk_boxes) < 2:
                    # нет живых лиц в куске / апскейл-каша -> без наезда
                    out.append((ca, cb, CropPlan(plan.crop_w, plan.crop_h,
                                                 plan.src_w, plan.src_h,
                                                 kf, plan.y0)))
                    continue
                import statistics
                face_top = statistics.median(b[1] for b in chunk_boxes)
                zy = int(face_top - 0.10 * zh)     # верх лица чуть ниже верха окна
                zy = max(0, min(plan.src_h - zh, zy))
                out.append((ca, cb, CropPlan(zw, zh, plan.src_w, plan.src_h,
                                             kf, zy)))
    return out


_SPLIT_SPREAD = 0.52     # разнос центров > этой доли ширины окна = «крупно не влезут»
_SPLIT_MIN_DUR = 1.5     # сплит включаем уже от 1.5с (было 3.0 — короткие переглядки терялись)


def _dialog_runs(samples: list, crop_w: int, min_dur: float = _SPLIT_MIN_DUR,
                 src_h: int = 1080) -> list:
    """Интервалы, где ДВА лица одновременно в кадре и не влезают комфортно
    в один вертикальный кроп 9:16. -> [(a, b, x_лев, x_прав)]

    Правило пользователя: как только двое стоят рядом и крупно оба не влезают —
    показываем СПЛИТ-СКРИН (стек), а НЕ ставим камеру между ними («носами»).
    Поэтому речь с обеих сторон больше НЕ требуется: достаточно стабильного
    присутствия двух разнесённых лиц. Если один из них — явный доминирующий
    говорящий почти весь интервал, сплит не нужен (это его крупный план).
    Мелкие лица-врезки/фон не участвуют в решении (_dominant_faces)."""
    import statistics
    raw, cur, gap = [], None, 0
    for t, scored in samples:
        # мелкие лица-врезки/фон не участвуют в решении о сплите
        scored = _dominant_faces(scored, src_h) if scored else scored
        far = False
        if len(scored) >= 2:
            cs = [f[0] for f in scored]
            far = max(cs) - min(cs) > crop_w * _SPLIT_SPREAD
        if far:
            if cur is None:
                cur = [t, t, [], 0, 0, 0]     # a, b, data, n_frames, dom_left, dom_right
            cur[1] = t
            cur[2].extend((f[0], f[1]) for f in scored)
            cur[3] += 1
            # доминирующий говорящий на этом замере? считаем, с КАКОЙ стороны,
            # чтобы отличить «один всё время говорит» (его план) от диалога
            cs = [f[0] for f in scored]
            mid = (min(cs) + max(cs)) / 2
            spk = sorted(scored, key=lambda f: f[1], reverse=True)
            top = spk[0]
            second = spk[1][1] if len(spk) > 1 else 0.0
            if top[1] >= _SPEECH_MIN and top[1] >= 1.6 * second:
                if top[0] < mid:
                    cur[4] += 1
                else:
                    cur[5] += 1
            gap = 0
        elif cur is not None:
            gap += 1
            if gap > 1:                      # терпим 1 пропуск замера
                raw.append(cur)
                cur = None
    if cur is not None:
        raw.append(cur)

    out = []
    for a, b, data, n_frames, dom_l, dom_r in raw:
        if b - a < min_dur:
            continue
        # если ОДНА сторона доминирует почти весь интервал — это её крупный
        # план (собеседник лишь фоном), сплит не нужен. Если доминируют ПО
        # ОЧЕРЕДИ обе — это диалог, сплит нужен.
        if n_frames and max(dom_l, dom_r) >= n_frames * 0.75:
            continue
        xs = sorted(c for c, _ in data)
        mid = (xs[0] + xs[-1]) / 2
        left = [c for c, _ in data if c < mid]
        right = [c for c, _ in data if c >= mid]
        # обе стороны должны РЕАЛЬНО присутствовать (по паре замеров с каждой),
        # иначе это одиночное лицо с редким выбросом на другой стороне
        if len(left) >= 2 and len(right) >= 2:
            out.append((a, b,
                        statistics.median(left),
                        statistics.median(right)))
    return out


def _in_speech(t: float, intervals: list) -> bool:
    return any(a - 0.25 <= t <= b + 0.25 for a, b in intervals)


def _detect_dialog(speech_positions: list, crop_w: int):
    """Диалог = говорили из ДВУХ мест, слишком далёких для одного кадра.

    Возвращает (x_левого, x_правого) или None."""
    if len(speech_positions) < 6:
        return None
    import statistics
    xs = sorted(speech_positions)
    if xs[-1] - xs[0] <= crop_w * 0.55:     # влезают комфортно в один кадр
        return None
    mid = (xs[0] + xs[-1]) / 2
    left = [x for x in xs if x < mid]
    right = [x for x in xs if x >= mid]
    # диалог, а не случайный выброс: с обеих сторон говорили по-настоящему
    if len(left) >= 3 and len(right) >= 3:
        return statistics.median(left), statistics.median(right)
    return None


# ------------------------------------------------------------ ffmpeg-выражение

_MAX_KEYFRAMES = 40   # лимит парсера выражений ffmpeg: больше ~сотни if() он не ест


def _rdp(points: list, tol: float) -> list:
    """Дуглас-Пекер: убирает точки, лежащие на прямой между соседями (± tol px)."""
    if len(points) <= 2:
        return points
    (t0, x0), (t1, x1) = points[0], points[-1]
    best_d, best_i = -1.0, 0
    for i in range(1, len(points) - 1):
        t, x = points[i]
        xi = x0 + (x1 - x0) * (t - t0) / max(t1 - t0, 1e-9)
        d = abs(x - xi)
        if d > best_d:
            best_d, best_i = d, i
    if best_d <= tol:
        return [points[0], points[-1]]
    left = _rdp(points[:best_i + 1], tol)
    right = _rdp(points[best_i:], tol)
    return left[:-1] + right


def _simplify_keyframes(kfs: list, tol: float) -> list:
    """Упрощает траекторию, не трогая моменты склеек (границы прыжков)."""
    runs, cur = [], [kfs[0]]
    for k in kfs[1:]:
        if k[2]:                       # склейка: закрываем плавный участок
            runs.append(cur)
            cur = [k]
        else:
            cur.append(k)
    runs.append(cur)

    out = []
    for run in runs:
        pts = _rdp([(t, x) for t, x, _ in run], tol)
        first_cut = run[0][2]
        out.extend((t, x, (i == 0 and first_cut)) for i, (t, x) in enumerate(pts))
    return out


def crop_filter_expr(plan: CropPlan) -> str:
    """Кроп с кусочно-линейной интерполяцией и мгновенными склейками.

    Траектория сперва упрощается (Дуглас-Пекер), чтобы выражение влезло
    в лимиты парсера ffmpeg; склейки при упрощении сохраняются точно.
    """
    kfs = plan.keyframes
    xs = [x for _, x, _ in kfs]
    half = plan.crop_w / 2
    max_x = plan.src_w - plan.crop_w

    if max(xs) - min(xs) < plan.src_w * 0.02:
        x0 = int(round(sum(xs) / len(xs) - half))
        return f"crop={plan.crop_w}:{plan.crop_h}:{max(0, min(max_x, x0))}:{plan.y0}"

    tol = 8.0
    while len(kfs) > _MAX_KEYFRAMES:
        kfs = _simplify_keyframes(plan.keyframes, tol)
        tol *= 1.6

    expr = f"{kfs[-1][1] - half:.1f}"
    for (t0, x0, _), (t1, x1, cut1) in reversed(list(zip(kfs, kfs[1:]))):
        jump = abs(x1 - x0) > plan.src_w * _CUT_JUMP_FRAC
        if cut1 or jump:
            seg = f"{x0 - half:.1f}"              # держим до t1, потом прыжок
        else:
            seg = (f"({x0 - half:.1f}+({x1 - x0:.1f})*"
                   f"(t-{t0:.2f})/{max(t1 - t0, 1e-3):.2f})")
        expr = f"if(lt(t\\,{t1:.2f})\\,{seg}\\,{expr})"
    expr = f"if(lt(t\\,{kfs[0][0]:.2f})\\,{kfs[0][1] - half:.1f}\\,{expr})"

    return (f"crop={plan.crop_w}:{plan.crop_h}:"
            f"x='max(0\\,min({max_x}\\,{expr}))':y={plan.y0}")
