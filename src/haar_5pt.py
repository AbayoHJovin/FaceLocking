# src/haar_five_points.py
"""
Haar-based face detection + lightweight 5-point landmarks via MediaPipe FaceMesh.

Rationale:
- Haar cascade provides fast CPU-based face proposals.
- FaceMesh validates real faces and supplies stable geometry.
- Only 5 landmarks are used: eyes, nose, mouth corners.
- Bounding box is rebuilt from landmarks to avoid side offsets.
- Haar detections without FaceMesh confirmation are discarded.

Run:
python -m src.haar_five_points
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as err:
    mp = None
    _MP_ERR = err


# =========================
# Structures
# =========================

@dataclass
class FaceResult:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    landmarks: np.ndarray  # (5,2) float32


# =========================
# Geometry utilities
# =========================

def _compute_affine_5pt(
    pts: np.ndarray,
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Estimate affine transform mapping 5 facial points to ArcFace-style template.
    Order: [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    src = pts.astype(np.float32)

    template = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    out_w, out_h = output_size
    if (out_w, out_h) != (112, 112):
        scale = np.array([out_w / 112.0, out_h / 112.0], dtype=np.float32)
        template *= scale

    M, _ = cv2.estimateAffinePartial2D(src, template, method=cv2.LMEDS)

    if M is None:
        M = cv2.getAffineTransform(
            np.array([src[0], src[1], src[2]], dtype=np.float32),
            np.array([template[0], template[1], template[2]], dtype=np.float32),
        )

    return M.astype(np.float32)


def warp_face_5pt(
    image: np.ndarray,
    pts: np.ndarray,
    output_size: Tuple[int, int] = (112, 112),
) -> Tuple[np.ndarray, np.ndarray]:
    M = _compute_affine_5pt(pts, output_size)
    w, h = output_size
    aligned = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, M


def _limit_bbox(box: np.ndarray, w: int, h: int) -> np.ndarray:
    clipped = box.astype(np.float32).copy()
    clipped[0] = np.clip(clipped[0], 0, w - 1)
    clipped[1] = np.clip(clipped[1], 0, h - 1)
    clipped[2] = np.clip(clipped[2], 0, w - 1)
    clipped[3] = np.clip(clipped[3], 0, h - 1)
    return clipped


def _bbox_from_landmarks(
    pts: np.ndarray,
    pad_x: float = 0.55,
    pad_top: float = 0.85,
    pad_bottom: float = 1.15,
) -> np.ndarray:
    p = pts.astype(np.float32)

    x0, x1 = np.min(p[:, 0]), np.max(p[:, 0])
    y0, y1 = np.min(p[:, 1]), np.max(p[:, 1])

    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)

    return np.array(
        [
            x0 - pad_x * w,
            y0 - pad_top * h,
            x1 + pad_x * w,
            y1 + pad_bottom * h,
        ],
        dtype=np.float32,
    )


def _smooth(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return cur.astype(np.float32)
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)


def _landmark_geometry_ok(pts: np.ndarray, min_eye_gap: float = 12.0) -> bool:
    le, re, nose, ml, mr = pts.astype(np.float32)

    if np.linalg.norm(re - le) < min_eye_gap:
        return False

    if not (ml[1] > nose[1] and mr[1] > nose[1]):
        return False

    return True


# =========================
# Detector
# =========================

class HaarFivePointDetector:
    def __init__(
        self,
        cascade_path: Optional[str] = None,
        min_face_size: Tuple[int, int] = (60, 60),
        smoothing: float = 0.80,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.min_face_size = min_face_size
        self.smoothing = smoothing

        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Unable to load Haar cascade: {cascade_path}")

        if mp is None:
            raise RuntimeError(
                f"MediaPipe failed to import: {_MP_ERR}\n"
                f"Install with: pip install mediapipe==0.10.21"
            )

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.LM_IDS = [33, 263, 1, 61, 291]
        self._last_box: Optional[np.ndarray] = None
        self._last_pts: Optional[np.ndarray] = None

    def _detect_haar(self, gray: np.ndarray) -> np.ndarray:
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_face_size,
        )

        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)

        return faces.astype(np.int32)

    def _extract_5pt(self, frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)

        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0].landmark
        pts = [[lm[i].x * w, lm[i].y * h] for i in self.LM_IDS]
        pts = np.array(pts, dtype=np.float32)

        if pts[0, 0] > pts[1, 0]:
            pts[[0, 1]] = pts[[1, 0]]
        if pts[3, 0] > pts[4, 0]:
            pts[[3, 4]] = pts[[4, 3]]

        return pts

    def detect(self, frame: np.ndarray, max_faces: int = 1) -> List[FaceResult]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        haar_faces = self._detect_haar(gray)
        if haar_faces.shape[0] == 0:
            return []

        areas = haar_faces[:, 2] * haar_faces[:, 3]
        fx, fy, fw, fh = haar_faces[np.argmax(areas)]

        pts = self._extract_5pt(frame)
        if pts is None:
            if self.verbose:
                print("[haar_5pt] Haar hit but FaceMesh returned nothing → dropped")
            return []

        margin = 0.35
        x0, y0 = fx - margin * fw, fy - margin * fh
        x1, y1 = fx + (1 + margin) * fw, fy + (1 + margin) * fh

        valid = (
            (pts[:, 0] >= x0)
            & (pts[:, 0] <= x1)
            & (pts[:, 1] >= y0)
            & (pts[:, 1] <= y1)
        )

        if valid.mean() < 0.60:
            if self.verbose:
                print("[haar_5pt] Landmark / Haar mismatch → dropped")
            return []

        if not _landmark_geometry_ok(pts, min_eye_gap=max(10.0, 0.18 * fw)):
            if self.verbose:
                print("[haar_5pt] Invalid 5-point geometry → dropped")
            return []

        box = _bbox_from_landmarks(pts)
        box = _limit_bbox(box, w, h)

        box_s = _smooth(self._last_box, box, self.smoothing)
        pts_s = _smooth(self._last_pts, pts, self.smoothing)

        self._last_box = box_s.copy()
        self._last_pts = pts_s.copy()

        x_min, y_min, x_max, y_max = box_s.tolist()

        return [
            FaceResult(
                x_min=int(round(x_min)),
                y_min=int(round(y_min)),
                x_max=int(round(x_max)),
                y_max=int(round(y_max)),
                confidence=1.0,
                landmarks=pts_s.astype(np.float32),
            )
        ][:max_faces]


# =========================
# Demo
# =========================

def main():
    cam = cv2.VideoCapture(0)
    detector = HaarFivePointDetector(min_face_size=(70, 70), smoothing=0.80, verbose=True)

    print("Haar + 5-point FaceMesh demo. Press 'q' to exit.")

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        display = frame.copy()
        results = detector.detect(frame)

        if results:
            face = results[0]
            cv2.rectangle(
                display,
                (face.x_min, face.y_min),
                (face.x_max, face.y_max),
                (0, 255, 0),
                2,
            )
            for (x, y) in face.landmarks.astype(int):
                cv2.circle(display, (x, y), 3, (0, 255, 0), -1)
        else:
            cv2.putText(
                display,
                "no face",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        cv2.imshow("haar_five_points", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
