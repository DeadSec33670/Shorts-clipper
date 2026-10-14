"""Active Speaker Detection (ASD) — определение говорящего по совпадению
движения губ со звуковой дорожкой.

Опциональный режим (флаг --asd). Заменяет визуальный _speech_score, который
угадывает речь только по движению рта, на нейросеть, которая сопоставляет
лицо со звуком и отвечает точнее — кто из видимых людей РЕАЛЬНО говорит.

Модель: Light-ASD (CVPR 2023, MIT). PyTorch подтягивается при первом запуске.
Всё работает локально и бесплатно. При любом сбое (нет PyTorch, не скачались
веса) модуль сообщает об этом и пайплайн откатывается на обычный режим.
"""
from __future__ import annotations

from pathlib import Path

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_WEIGHTS = _ASSETS / "light_asd_weights.pth"
_WEIGHTS_URL = (
    "https://github.com/Junhua-Liao/Light-ASD/raw/main/weight/finetuning_TalkSet.model"
)


class ASDUnavailable(Exception):
    """ASD нельзя использовать (нет зависимостей/весов) — откат на обычный режим."""


def ensure_ready() -> str:
    """Проверяет доступность PyTorch и весов. Возвращает устройство cpu/cuda
    или бросает ASDUnavailable с понятным сообщением."""
    try:
        import torch
    except ImportError as e:
        raise ASDUnavailable(
            "Для режима --asd нужен PyTorch. Установите:\n"
            "  CPU:  pip install torch torchaudio --index-url "
            "https://download.pytorch.org/whl/cpu\n"
            "  GPU:  pip install torch torchaudio --index-url "
            "https://download.pytorch.org/whl/cu121\n"
            "  и: pip install python_speech_features"
        ) from e

    try:
        import python_speech_features  # noqa: F401
    except ImportError as e:
        raise ASDUnavailable(
            "Для --asd нужен python_speech_features: pip install python_speech_features"
        ) from e

    if not _WEIGHTS.exists():
        _download_weights()

    return "cuda" if torch.cuda.is_available() else "cpu"


def _download_weights() -> None:
    import urllib.request
    print(f"[asd] Скачиваю веса Light-ASD (~1 МБ) в {_WEIGHTS}…")
    _ASSETS.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_WEIGHTS_URL, _WEIGHTS)
    except Exception as e:  # noqa: BLE001
        raise ASDUnavailable(
            f"Не удалось скачать веса Light-ASD ({e}).\n"
            f"Скачайте вручную с {_WEIGHTS_URL}\n"
            f"и положите в {_WEIGHTS}"
        ) from e


class ASDScorer:
    """Оборачивает Light-ASD: по трекам лиц и аудио клипа даёт для каждого
    лица в каждый момент оценку «говорит / молчит».

    Использование:
        scorer = ASDScorer(device)
        scores = scorer.score_segment(video, start, end, face_tracks)
    """

    def __init__(self, device: str):
        import torch
        from .asd_model import ASD_Net   # локальная копия архитектуры

        self.device = device
        self.net = ASD_Net().to(device)
        state = torch.load(str(_WEIGHTS), map_location=device, weights_only=False)
        # веса Light-ASD хранят префикс 'model.' — снимаем при необходимости
        clean = {}
        for k, v in state.items():
            clean[k.replace("model.", "", 1) if k.startswith("model.") else k] = v
        try:
            self.net.load_state_dict(clean, strict=False)
        except Exception:  # noqa: BLE001
            self.net.load_state_dict(state, strict=False)
        self.net.eval()

    def score_track(self, face_crops, audio_feat) -> list[float]:
        """face_crops: список серых лиц 112x112 (по кадрам трека);
        audio_feat: MFCC аудио того же интервала.
        Возвращает оценку речи на каждый кадр трека (0..1 после сигмоиды)."""
        import numpy as np
        import torch

        if not face_crops:
            return []
        with torch.no_grad():
            v = torch.FloatTensor(np.array(face_crops)).unsqueeze(0).to(self.device)
            a = torch.FloatTensor(audio_feat).unsqueeze(0).to(self.device)
            afe = self.net.model.forward_audio_frontend(a)
            vfe = self.net.model.forward_visual_frontend(v)
            out = self.net.model.forward_audio_visual_backend(afe, vfe)
            scores = torch.sigmoid(self.net.lossAV.FC(out)[:, 1]).cpu().numpy()
        return [float(s) for s in scores]


def extract_audio_mfcc(video, start: float, end: float):
    """Достаёт MFCC аудио интервала [start, end] для Light-ASD (100 фич/с)."""
    import subprocess
    import tempfile
    import wave

    import numpy as np
    from python_speech_features import mfcc

    tmp = Path(tempfile.gettempdir()) / f"asd_audio_{int(start*1000)}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
         "-i", str(video), "-t", f"{end - start:.3f}",
         "-ac", "1", "-ar", "16000", str(tmp)],
        check=True,
    )
    with wave.open(str(tmp), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    tmp.unlink(missing_ok=True)
    # winstep 0.010 -> 100 аудио-фич в секунду (4 на видеокадр при 25 fps)
    return mfcc(pcm, sr, numcep=13, winlen=0.025, winstep=0.010)

