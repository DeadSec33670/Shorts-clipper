"""Re-ID лиц: узнавание «это тот же человек» через эмбеддинги SFace.

Использует модель SFace из OpenCV Zoo (assets/face_sface.onnx, ~38 МБ,
лицензия Apache-2.0). Работает через встроенный в OpenCV cv2.FaceRecognizerSF
— не требует PyTorch. Идею подсмотрели в gauravzazz/smart-reframe.

Назначение:
  1. Стабильный трек человека при склейках и пропусках детекции (сопоставляем
     лица по внешности, а не только по позиции X).
  2. Фундамент под «имена по фото»: эталонные фото -> эмбеддинги -> камера
     знает, кто есть кто по имени.

Отказоустойчивость: если модель недоступна или что-то падает — Re-ID тихо
отключается, пайплайн работает по-старому (сопоставление по позиции).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_SFACE_PATH = _ASSETS / "face_sface.onnx"
_YUNET_PATH = _ASSETS / "face_yunet.onnx"

# порог косинусного сходства: выше -> «тот же человек» (рекоменд. SFace ~0.363)
SAME_PERSON_COSINE = 0.363


class ReIDScorer:
    """Извлекает эмбеддинги лиц и сравнивает их. Один экземпляр на прогон.

    Для ТОЧНОГО узнавания лицо выравнивается по 5 ключевым точкам (глаза,
    нос, углы рта) детектором YuNet перед подачей в SFace. Без выравнивания
    (грубый ресайз bbox) точность заметно ниже — годится лишь для «явно
    разные лица». Оба пути отказоустойчивы."""

    def __init__(self):
        self._rec = None
        self._det = None            # YuNet — для выравнивания
        self._ok = False
        try:
            import cv2
            if not _SFACE_PATH.exists():
                raise FileNotFoundError(_SFACE_PATH)
            self._rec = cv2.FaceRecognizerSF.create(str(_SFACE_PATH), "")
            self._cv2 = cv2
            self._ok = True
            if _YUNET_PATH.exists() and hasattr(cv2, "FaceDetectorYN"):
                self._det = cv2.FaceDetectorYN.create(
                    str(_YUNET_PATH), "", (320, 320), 0.7, 0.3, 5000)
        except Exception as e:  # noqa: BLE001
            print(f"[reid] Re-ID выключен ({type(e).__name__}: {e}). "
                  f"Трек лиц — по позиции.")

    @property
    def available(self) -> bool:
        return self._ok

    @property
    def has_alignment(self) -> bool:
        return self._det is not None

    def embed_aligned(self, face_112: "np.ndarray"):
        """Эмбеддинг уже выровненного лица 112x112 BGR. -> (128,) или None."""
        if not self._ok:
            return None
        try:
            feat = self._rec.feature(face_112)
            return feat.flatten().astype(np.float32)
        except Exception:  # noqa: BLE001
            return None

    def embed_region(self, frame_bgr: "np.ndarray", box=None):
        """Точный эмбеддинг: YuNet находит лицо (в области box, если задан),
        выравнивает по ключевым точкам, SFace даёт эмбеддинг. Если YuNet
        недоступен/не нашёл — грубый путь (ресайз bbox). -> (128,) | None."""
        if not self._ok:
            return None
        try:
            if self._det is not None:
                region = frame_bgr
                ox, oy = 0, 0
                if box is not None:
                    h, w = frame_bgr.shape[:2]
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    # расширяем рамку — YuNet любит контекст вокруг лица
                    pad = int(0.3 * max(x2 - x1, y2 - y1))
                    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                    if x2 - x1 >= 16 and y2 - y1 >= 16:
                        region = frame_bgr[y1:y2, x1:x2]
                        ox, oy = x1, y1
                rh, rw = region.shape[:2]
                self._det.setInputSize((rw, rh))
                _, faces = self._det.detect(region)
                if faces is not None and len(faces):
                    if box is not None:
                        # лицо, чей центр ближе к центру запрошенной рамки
                        # (в координатах region), а не просто крупнейшее —
                        # иначе при двух лицах рядом берём не того
                        bcx = (box[0] + box[2]) / 2 - ox
                        bcy = (box[1] + box[3]) / 2 - oy
                        f = min(faces, key=lambda ff:
                                (ff[0] + ff[2] / 2 - bcx) ** 2
                                + (ff[1] + ff[3] / 2 - bcy) ** 2)
                    else:
                        f = max(faces, key=lambda ff: ff[2] * ff[3])
                    aligned = self._rec.alignCrop(region, f)
                    return self.embed_aligned(aligned)
            # fallback — грубый ресайз
            if box is not None:
                return self.embed_crop(frame_bgr, box)
        except Exception:  # noqa: BLE001
            pass
        return None

    def embed_crop(self, frame_bgr: "np.ndarray", box) -> "np.ndarray | None":
        """Грубый эмбеддинг по рамке (без выравнивания) — fallback, когда
        YuNet недоступен. Годится лишь отличать явно разных людей."""
        if not self._ok:
            return None
        try:
            h, w = frame_bgr.shape[:2]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                return None
            crop = frame_bgr[y1:y2, x1:x2]
            face = self._cv2.resize(crop, (112, 112),
                                    interpolation=self._cv2.INTER_LINEAR)
            return self.embed_aligned(face)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def cosine(a, b) -> float:
        """Косинусное сходство двух эмбеддингов (SFace выдаёт ненормированные,
        поэтому нормируем сами). Диапазон [-1, 1]."""
        if a is None or b is None:
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def same_person(self, a, b, thr: float = SAME_PERSON_COSINE) -> bool:
        return self.cosine(a, b) >= thr


class IdentityBank:
    """Хранилище эталонных лиц с именами (для «имён по фото»).

    Загружает фото из папки: каждый файл `Имя.jpg` (или `Имя_1.jpg`,
    `Имя_2.jpg` для нескольких ракурсов) -> эмбеддинг. Потом по эмбеддингу
    лица из видео находит ближайшее имя (если сходство выше порога)."""

    def __init__(self, scorer: ReIDScorer):
        self.scorer = scorer
        self.people: dict[str, list] = {}     # имя -> [эмбеддинги]

    def load_dir(self, folder: Path) -> int:
        """Грузит эталоны из папки. Имя = имя файла без расширения и без
        суффикса _N. Возвращает число загруженных людей."""
        if not self.scorer.available:
            return 0
        import cv2
        folder = Path(folder)
        if not folder.is_dir():
            return 0
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            name = img_path.stem
            if "_" in name and name.rsplit("_", 1)[1].isdigit():
                name = name.rsplit("_", 1)[0]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            emb = self.scorer.embed_region(img)   # точный путь с выравниванием
            if emb is not None:
                self.people.setdefault(name, []).append(emb)
        if self.people:
            print(f"[reid] Загружено эталонов: "
                  f"{', '.join(f'{n}×{len(v)}' for n, v in self.people.items())}")
        return len(self.people)

    def identify(self, emb, thr: float = SAME_PERSON_COSINE):
        """Имя ближайшего эталона или None. -> (name, score) | (None, best)."""
        if emb is None or not self.people:
            return None, 0.0
        best_name, best_score = None, -1.0
        for name, embs in self.people.items():
            for ref in embs:
                s = self.scorer.cosine(emb, ref)
                if s > best_score:
                    best_name, best_score = name, s
        if best_score >= thr:
            return best_name, best_score
        return None, best_score
