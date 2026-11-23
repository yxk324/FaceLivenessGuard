import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# 自动挂载同级 pyVHR 仓库路径
_ROOT_DIR = Path(__file__).resolve().parents[2]
_PYVHR_REPO = _ROOT_DIR / "pyVHR"
if _PYVHR_REPO.exists() and str(_PYVHR_REPO) not in sys.path:
    sys.path.insert(0, str(_PYVHR_REPO))

from pyVHR.BVP.filters import BPfilter, apply_filter
from pyVHR.BVP.methods import cpu_POS


class InMemoryPyVHRAdapter:
    """
    基于 MediaPipe 解剖学多边形皮肤分区 + pyVHR POS 投影的纯内存实时提取器。
    """

    def __init__(self, patch_size: int = 28):
        self.patch_size = patch_size
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )

        # MediaPipe 468 关键点中的三个纯皮肤多边形区域（天然避开眉毛、眼睛、鼻孔与嘴唇）
        self.roi_polygons = [
            [109, 10, 338, 337, 151, 108],   # 额头中央安全区
            [118, 101, 205, 187, 123],       # 左脸颊高灌注区
            [347, 330, 425, 411, 352],       # 右脸颊高灌注区
        ]

    def process_frame(self, frame_bgr: np.ndarray):
        """
        输入单帧 BGR 图像，提取 3 个皮肤多边形区域的 RGB 均值与背景对照均值。
        返回: (skin_patches_rgb, bg_rgb, debug_boxes)
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None, None, None

        face_lms = results.multi_face_landmarks[0].landmark
        pts_all = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in face_lms], dtype=np.int32
        )

        patch_means = []
        debug_boxes = []

        for poly_idx in self.roi_polygons:
            poly_pts = pts_all[poly_idx]
            x, y, rw, rh = cv2.boundingRect(poly_pts)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + rw), min(h, y + rh)

            if (x2 - x1) < 4 or (y2 - y1) < 4:
                return None, None, None

            crop = frame_rgb[y1:y2, x1:x2]
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            shifted_pts = poly_pts - np.array([x1, y1], dtype=np.int32)
            cv2.fillConvexPoly(mask, shifted_pts, 255)

            roi_pixels = crop[mask == 255]
            if len(roi_pixels) < 8:
                return None, None, None

            patch_means.append(roi_pixels.mean(axis=0))
            debug_boxes.append((x1, y1, x2, y2))

        # 采样画面左上、右上角作为环境光背景对照块
        bh, bw = max(20, h // 8), max(20, w // 8)
        tl = frame_rgb[10 : 10 + bh, 10 : 10 + bw]
        tr = frame_rgb[10 : 10 + bh, w - 10 - bw : w - 10]
        bg_rgb = 0.5 * (tl.mean(axis=(0, 1)) + tr.mean(axis=(0, 1)))

        return (
            np.array(patch_means, dtype=np.float32),
            bg_rgb.astype(np.float32),
            debug_boxes,
        )

    def extract_bvp(self, rgb_seq: np.ndarray, fps: float) -> np.ndarray:
        """调用 pyVHR 的 cpu_POS 与带通滤波器提取 BVP 脉搏波。"""
        pyvhr_tensor = np.transpose(rgb_seq, (1, 2, 0)).astype(np.float32)
        bvps = cpu_POS(pyvhr_tensor, fps=fps)
        bvps_filt = apply_filter(
            [bvps],
            BPfilter,
            fps=fps,
            params={"order": 4, "minHz": 0.7, "maxHz": 3.0, "fps": fps},
        )[0]
        return bvps_filt

    def close(self):
        self.face_mesh.close()