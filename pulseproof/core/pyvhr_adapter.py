import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# 自动挂载同级或原路径下的 pyVHR 仓库
_ROOT_DIR = Path(__file__).resolve().parents[2]  # 自动指向 E:\FaceLivenessGuard
_PYVHR_REPO = _ROOT_DIR / "pyVHR"
if _PYVHR_REPO.exists() and str(_PYVHR_REPO) not in sys.path:
    sys.path.insert(0, str(_PYVHR_REPO))

from pyVHR.BVP.filters import BPfilter, apply_filter
from pyVHR.BVP.methods import cpu_POS

try:
    from pyVHR.BVP.methods import cpu_CHROM
except ImportError:
    cpu_CHROM = None


def _fallback_chrom(signal: np.ndarray, fps: float) -> np.ndarray:
    """当本地 pyVHR 版本未导出 cpu_CHROM 时的标准 CHROM 投影实现，输入 (P, 3, T)。"""
    P, _, T = signal.shape
    out = np.zeros((P, T), dtype=np.float32)
    for p in range(P):
        rgb = signal[p]  # (3, T)
        mean_rgb = np.mean(rgb, axis=1, keepdims=True) + 1e-6
        rn, gn, bn = rgb[0] / mean_rgb[0], rgb[1] / mean_rgb[1], rgb[2] / mean_rgb[2]
        xs = 3.0 * rn - 2.0 * gn
        ys = 1.5 * rn + gn - 1.5 * bn
        alpha = np.std(xs) / (np.std(ys) + 1e-6)
        out[p] = xs - alpha * ys
    return out


class InMemoryPyVHRAdapter:
    """
    基于 MediaPipe 解剖学多边形分区 + pyVHR POS/CHROM 双投影的纯内存实时提取器。
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
        self.roi_polygons = [
            [109, 10, 338, 337, 151, 108],   # 额头中央安全区
            [118, 101, 205, 187, 123],       # 左脸颊高灌注区
            [347, 330, 425, 411, 352],       # 右脸颊高灌注区
        ]

    def process_frame_detailed(self, frame_bgr: np.ndarray):
        """
        提取单帧多区域 RGB 均值、背景对照均值、采样框坐标以及过曝/位置元数据。
        返回: (patch_means, bg_rgb, debug_boxes, meta)
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None, None, None, None

        face_lms = results.multi_face_landmarks[0].landmark
        pts_all = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in face_lms], dtype=np.int32
        )

        patch_means = []
        debug_boxes = []
        glare_ratios = []

        for poly_idx in self.roi_polygons:
            poly_pts = pts_all[poly_idx]
            x, y, rw, rh = cv2.boundingRect(poly_pts)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + rw), min(h, y + rh)

            if (x2 - x1) < 4 or (y2 - y1) < 4:
                return None, None, None, None

            crop = frame_rgb[y1:y2, x1:x2]
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            shifted_pts = poly_pts - np.array([x1, y1], dtype=np.int32)
            cv2.fillConvexPoly(mask, shifted_pts, 255)

            roi_pixels = crop[mask == 255]
            if len(roi_pixels) < 8:
                return None, None, None, None

            # 统计单通道超过 248 的过曝高光像素占比
            overexposed = np.any(roi_pixels >= 248, axis=-1)
            glare_ratios.append(float(np.mean(overexposed)))

            patch_means.append(roi_pixels.mean(axis=0))
            debug_boxes.append((x1, y1, x2, y2))

        bh, bw = max(20, h // 8), max(20, w // 8)
        tl = frame_rgb[10 : 10 + bh, 10 : 10 + bw]
        tr = frame_rgb[10 : 10 + bh, w - 10 - bw : w - 10]
        bg_rgb = 0.5 * (tl.mean(axis=(0, 1)) + tr.mean(axis=(0, 1)))

        # 归一化鼻尖关键点 (index 1) 作为人脸运动跟踪锚点
        nose = face_lms[1]
        meta = {
            "glare_ratio": float(np.mean(glare_ratios)),
            "face_center": (float(nose.x), float(nose.y)),
        }

        return (
            np.array(patch_means, dtype=np.float32),
            bg_rgb.astype(np.float32),
            debug_boxes,
            meta,
        )

    def process_frame(self, frame_bgr: np.ndarray):
        """保持与旧版脚本兼容的 3 元组接口。"""
        patches, bg, boxes, _ = self.process_frame_detailed(frame_bgr)
        return patches, bg, boxes

    @staticmethod
    def _compute_cardiac_snr(bvps: np.ndarray, fps: float) -> float:
        """计算多区域平均 BVP 在生理心率带 (0.75-2.8 Hz) 内的功率谱集中度 (SNR)。"""
        sig = bvps[:, 12:].mean(axis=0) if bvps.shape[1] > 20 else bvps.mean(axis=0)
        sig = sig - np.mean(sig)
        if np.std(sig) < 1e-6:
            return 0.0
        n_fft = max(256, len(sig) * 4)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)
        psd = np.abs(np.fft.rfft(sig * np.hanning(len(sig)), n=n_fft)) ** 2

        total_band = (freqs >= 0.5) & (freqs <= 4.0)
        cardiac_band = (freqs >= 0.75) & (freqs <= 2.8)
        total_power = float(np.sum(psd[total_band])) + 1e-8

        # 取主峰周围 ±0.15 Hz 能量占总生理频带能量的比例
        peak_idx = int(np.argmax(psd[cardiac_band]))
        peak_freq = float(freqs[cardiac_band][peak_idx])
        peak_band = (freqs >= peak_freq - 0.15) & (freqs <= peak_freq + 0.15)
        peak_power = float(np.sum(psd[peak_band]))
        return peak_power / total_power

    def extract_bvp_dual(self, rgb_seq: np.ndarray, fps: float) -> tuple[np.ndarray, str, float]:
        """
        并行运行 POS 与 CHROM 两种投影算法，选择生理频段信噪比更高的脉搏波形。
        返回: (best_bvps, method_name, snr_score)
        """
        pyvhr_tensor = np.transpose(rgb_seq, (1, 2, 0)).astype(np.float32)
        filter_params = {"order": 4, "minHz": 0.7, "maxHz": 3.0, "fps": fps}

        pos_raw = np.nan_to_num(cpu_POS(pyvhr_tensor, fps=fps))
        pos_filt = apply_filter([pos_raw], BPfilter, fps=fps, params=filter_params)[0]
        pos_snr = self._compute_cardiac_snr(pos_filt, fps)

        # pyVHR 原生 cpu_CHROM(signal) 仅接受单参数，不可传入 fps
        if cpu_CHROM is not None:
            chrom_raw = np.nan_to_num(cpu_CHROM(pyvhr_tensor))
        else:
            chrom_raw = _fallback_chrom(pyvhr_tensor, fps=fps)

        chrom_filt = apply_filter([chrom_raw], BPfilter, fps=fps, params=filter_params)[0]
        chrom_snr = self._compute_cardiac_snr(chrom_filt, fps)

        if chrom_snr > pos_snr * 1.05:
            return chrom_filt, "CHROM", chrom_snr
        return pos_filt, "POS", pos_snr

    def extract_bvp(self, rgb_seq: np.ndarray, fps: float) -> np.ndarray:
        """保持与旧版脚本兼容的单一波形返回接口。"""
        bvps, _, _ = self.extract_bvp_dual(rgb_seq, fps)
        return bvps

    def close(self):
        self.face_mesh.close()