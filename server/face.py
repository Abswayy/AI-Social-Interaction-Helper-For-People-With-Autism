"""Face detection + embedding with OpenCV's YuNet and SFace.

Both models are tiny ONNX files (~0.3 MB + ~37 MB), run on CPU in a few ms,
and need nothing beyond opencv-python. The same files can later run on a phone
or the glasses' SoC via ONNX Runtime, so there is no lock-in.
"""
from __future__ import annotations

import logging
import threading
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from . import config

log = logging.getLogger("face")

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "yunet.onnx": f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def ensure_models() -> None:
    for name, url in MODELS.items():
        path = config.MODELS_DIR / name
        if path.exists() and path.stat().st_size > 10_000:
            continue
        log.info("Downloading %s ...", name)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)


@dataclass
class Face:
    bbox: tuple[int, int, int, int]   # x, y, w, h in frame pixels
    score: float
    embedding: np.ndarray             # L2-normalised 128-d float32
    crop: np.ndarray | None = None    # aligned 112x112 face, used as the profile photo

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


class FaceEngine:
    """Thread-safe wrapper. One instance is shared by every connected client."""

    def __init__(self):
        ensure_models()
        self._lock = threading.Lock()
        self._det = cv2.FaceDetectorYN.create(
            str(config.MODELS_DIR / "yunet.onnx"), "", (320, 320),
            score_threshold=config.FACE_DET_SCORE, nms_threshold=0.3, top_k=20)
        self._rec = cv2.FaceRecognizerSF.create(str(config.MODELS_DIR / "sface.onnx"), "")

    def analyze(self, frame_bgr: np.ndarray, max_faces: int = 3) -> list[Face]:
        h, w = frame_bgr.shape[:2]
        with self._lock:
            self._det.setInputSize((w, h))
            _, dets = self._det.detect(frame_bgr)
            if dets is None:
                return []
            faces = []
            # Largest faces first: the person you're talking to is closest.
            for d in sorted(dets, key=lambda d: d[2] * d[3], reverse=True)[:max_faces]:
                x, y, fw, fh = (int(v) for v in d[:4])
                if min(fw, fh) < config.FACE_MIN_SIZE_PX:
                    continue
                aligned = self._rec.alignCrop(frame_bgr, d)
                if _blurry(aligned):
                    continue
                emb = self._rec.feature(aligned).flatten().astype(np.float32)
                emb /= (np.linalg.norm(emb) + 1e-9)
                faces.append(Face((x, y, fw, fh), float(d[-1]), emb, aligned))
            return faces


def _blurry(img: np.ndarray, threshold: float = 40.0) -> bool:
    """Motion-blurred crops give bad embeddings (head turns, walking)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def decode_jpeg(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""
