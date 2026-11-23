from typing import Optional
import numpy as np
from scipy.signal import butter, filtfilt
from core.pyvhr_adapter import InMemoryPyVHRAdapter

_HIGH_LUMA_PALETTE = np.array(
    [
        [255, 145, 145],  # 浅珊瑚红
        [145, 255, 155],  # 浅薄荷绿
        [145, 175, 255],  # 浅天蓝
        [255, 245, 145],  # 浅琥珀黄
        [145, 245, 255],  # 浅青色
        [255, 145, 245],  # 浅品红
        [255, 250, 242],  # 亮暖白
    ],
    dtype=np.uint8,
)


def generate_challenge_sequence(
    num_frames: int = 75, fps: float = 25.0, min_color_dist: float = 135.0
) -> np.ndarray:
    """
    生成高亮度浅彩色块阶跃挑战序列 C(t)，强制相邻色块欧氏距离 >= min_color_dist，
    防止连续抽到色相接近的颜色导致阶跃沿过弱。
    """
    rng = np.random.default_rng()
    num_blocks = 8
    block_edges = np.linspace(0, num_frames, num_blocks + 1, dtype=int)

    seq = np.zeros((num_frames, 3), dtype=np.uint8)
    prev_idx = -1

    for b in range(num_blocks):
        if prev_idx < 0:
            idx = int(rng.integers(0, len(_HIGH_LUMA_PALETTE)))
        else:
            prev_color = _HIGH_LUMA_PALETTE[prev_idx].astype(np.float32)
            dists = [
                float(np.linalg.norm(_HIGH_LUMA_PALETTE[i].astype(np.float32) - prev_color))
                for i in range(len(_HIGH_LUMA_PALETTE))
            ]
            candidates = [i for i, d in enumerate(dists) if i != prev_idx and d >= min_color_dist]
            if not candidates:
                candidates = [int(np.argmax(dists))]
            idx = int(rng.choice(candidates))

        prev_idx = idx
        seq[block_edges[b] : block_edges[b + 1]] = _HIGH_LUMA_PALETTE[idx]

    return seq


def resample_to_uniform_grid(
    timestamps_ms: np.ndarray,
    sent_rgb: np.ndarray,
    skin_rgb: np.ndarray,
    bg_rgb: np.ndarray,
    target_fps: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    将浏览器端存在轻微抖动的采集时间戳重采样到严格等间隔时间网格，
    保证滤波器与延迟互相关计算的相位精准对齐。
    """
    t_sec = (timestamps_ms - timestamps_ms[0]) / 1000.0
    valid_mask = np.concatenate([[True], np.diff(t_sec) > 1e-4])
    t_sec = t_sec[valid_mask]
    sent_rgb = sent_rgb[valid_mask]
    skin_rgb = skin_rgb[valid_mask]
    bg_rgb = bg_rgb[valid_mask]

    duration = float(t_sec[-1])
    if duration < 1.5 or len(t_sec) < 35:
        return sent_rgb, skin_rgb, bg_rgb, target_fps

    n_uniform = max(45, int(round(duration * target_fps)))
    uniform_t = np.linspace(0.0, duration, n_uniform)
    actual_fps = (n_uniform - 1) / duration

    sent_u = np.column_stack([np.interp(uniform_t, t_sec, sent_rgb[:, c]) for c in range(3)])
    bg_u = np.column_stack([np.interp(uniform_t, t_sec, bg_rgb[:, c]) for c in range(3)])

    T_u, P, C = n_uniform, skin_rgb.shape[1], 3
    skin_u = np.zeros((T_u, P, C), dtype=np.float32)
    for p in range(P):
        for c in range(C):
            skin_u[:, p, c] = np.interp(uniform_t, t_sec, skin_rgb[:, p, c])

    return sent_u.astype(np.float32), skin_u, bg_u.astype(np.float32), actual_fps


class LivenessVerifier:
    """PulseProof 多模态物理-生理联合活体校验器。"""

    def __init__(
        self,
        adapter: InMemoryPyVHRAdapter,
        replay_floor: float = 0.45,
        sync_floor: float = 0.15,
        joint_score_thresh: float = 0.56,
        max_lag: int = 10,
    ):
        self.adapter = adapter
        self.replay_floor = replay_floor
        self.sync_floor = sync_floor
        self.joint_score_thresh = joint_score_thresh
        self.max_lag = max_lag

    @staticmethod
    def _bandpass_edges(sig: np.ndarray, fps: float) -> np.ndarray:
        nyq = 0.5 * fps
        low = 0.8 / nyq
        high = min(5.5, nyq * 0.88) / nyq
        if low >= high or len(sig) < 20:
            return sig - np.mean(sig, axis=0, keepdims=True)
        b, a = butter(3, [low, high], btype="band")
        padlen = min(len(sig) - 1, 3 * max(len(a), len(b)))
        return filtfilt(b, a, sig, axis=0, padlen=padlen)

    def _check_replay_correlation(
        self, sent_rgb: np.ndarray, skin_rgb: np.ndarray, fps: float
    ) -> tuple[float, int, np.ndarray, np.ndarray]:
        """计算色块跳变边缘的延迟对齐互相关，并返回归一化对比曲线用于前端可视化。"""
        mean_skin = skin_rgb.mean(axis=1)
        sent_linear = np.power(np.clip(sent_rgb, 0, 255) / 255.0, 2.2)

        sent_edge = self._bandpass_edges(sent_linear, fps)
        skin_edge = self._bandpass_edges(mean_skin, fps)
        ch_weights = np.array([0.42, 0.42, 0.16], dtype=np.float32)

        best_score = -1.0
        best_lag = 0

        for lag in range(self.max_lag + 1):
            s = sent_edge if lag == 0 else sent_edge[:-lag]
            r = skin_edge if lag == 0 else skin_edge[lag:]
            if len(s) < 25:
                break

            corrs = np.zeros(3, dtype=np.float32)
            for c in range(3):
                sc = s[:, c] - s[:, c].mean()
                rc = r[:, c] - r[:, c].mean()
                corrs[c] = np.dot(sc, rc) / (np.linalg.norm(sc) * np.linalg.norm(rc) + 1e-8)

            score = float(np.sum(corrs * ch_weights))
            if score > best_score:
                best_score = score
                best_lag = lag

        s_wave = sent_edge[:, :2].mean(axis=1)
        r_wave = skin_edge[:, :2].mean(axis=1)
        s_wave = (s_wave - np.mean(s_wave)) / (np.std(s_wave) + 1e-6)
        r_wave = (r_wave - np.mean(r_wave)) / (np.std(r_wave) + 1e-6)
        return best_score, best_lag, s_wave, r_wave

    @staticmethod
    def _check_specular_diffuse_ratio(
        skin_rgb: np.ndarray, sent_rgb: np.ndarray, lag: int
    ) -> float:
        """SpecDiff 高低光差分增益比：区分平面玻璃均匀反光与三维人脸曲面漫反射。"""
        aligned_sent = np.roll(sent_rgb, shift=lag, axis=0)
        luma = aligned_sent.mean(axis=1)
        k = min(12, len(luma) // 3)
        if k < 3:
            return 1.0
        high_idx = np.argsort(luma)[-k:]
        low_idx = np.argsort(luma)[:k]

        diff_all = np.mean(skin_rgb[high_idx, :, :2] - skin_rgb[low_idx, :, :2], axis=(0, 2))
        forehead_gain = float(diff_all[0])
        cheeks_gain = float(np.mean(diff_all[1:]))
        if abs(cheeks_gain) < 1e-3:
            return 1.0
        return forehead_gain / cheeks_gain

    @staticmethod
    def _strip_screen_light_nlms(
        skin_rgb: np.ndarray, sent_rgb: np.ndarray, lag: int, mu: float = 0.25, taps: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        T, P, C = skin_rgb.shape
        sent_linear = np.power(np.clip(sent_rgb, 0, 255) / 255.0, 2.2) * 255.0
        aligned_sent = np.roll(sent_linear, shift=lag, axis=0)
        if lag > 0:
            aligned_sent[:lag] = aligned_sent[lag]

        residual = np.zeros_like(skin_rgb, dtype=np.float32)
        for p in range(P):
            for c in range(C):
                d = skin_rgb[:, p, c].astype(np.float32)
                x = aligned_sent[:, c].astype(np.float32)
                w = np.zeros(taps, dtype=np.float32)
                e = np.zeros(T, dtype=np.float32)

                for n in range(taps, T):
                    x_vec = x[n - taps : n][::-1]
                    err = d[n] - float(np.dot(w, x_vec))
                    e[n] = err
                    w += (mu / (float(np.dot(x_vec, x_vec)) + 1e-6)) * err * x_vec

                residual[:, p, c] = e

        return residual, aligned_sent

    def verify(
        self,
        sent_rgb: np.ndarray,
        skin_rgb: np.ndarray,
        bg_rgb: np.ndarray,
        fps: float,
        timestamps_ms: Optional[np.ndarray] = None,
        glare_ratios: Optional[list[float]] = None,
        face_centers: Optional[list[tuple[float, float]]] = None,
        is_virtual_cam: bool = False,
    ) -> dict:
        """执行完整质量门控与多特征联合活体判定。"""
        if is_virtual_cam:
            return {"live": False, "reason": "REJECT_VIRTUAL_CAMERA"}

        if len(skin_rgb) < 40:
            return {"live": False, "reason": "INSUFFICIENT_FACE_FRAMES"}

        # 1. 信号质量门控：过曝高光与剧烈头部晃动检测
        if glare_ratios and float(np.mean(glare_ratios)) > 0.25:
            return {
                "live": False,
                "reason": "RETRY_EXCESSIVE_GLARE",
                "glare_ratio": round(float(np.mean(glare_ratios)), 3),
            }

        if face_centers and len(face_centers) > 10:
            centers = np.array(face_centers, dtype=np.float32)
            step_move = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            if float(np.mean(step_move)) > 0.025:
                return {
                    "live": False,
                    "reason": "RETRY_EXCESSIVE_MOTION",
                    "motion_score": round(float(np.mean(step_move)), 4),
                }

        sent_f = sent_rgb.astype(np.float32)
        skin_f = skin_rgb.astype(np.float32)
        bg_f = bg_rgb.astype(np.float32)

        # 2. 若前端提供了真实时间戳，重采样至严格等间隔时间轴
        if timestamps_ms is not None and len(timestamps_ms) == len(skin_f):
            sent_f, skin_f, bg_f, fps = resample_to_uniform_grid(
                np.asarray(timestamps_ms, dtype=np.float64),
                sent_f,
                skin_f,
                bg_f,
                target_fps=25.0,
            )

        # 3. 第一关：屏幕光挑战校验（拦截高清视频回放）
        challenge_corr, best_lag, s_wave, r_wave = self._check_replay_correlation(
            sent_f, skin_f, fps=fps
        )
        spec_ratio = self._check_specular_diffuse_ratio(skin_f, sent_f, lag=best_lag)

        if challenge_corr < self.replay_floor:
            return {
                "live": False,
                "reason": "REJECT_SCREEN_REPLAY",
                "challenge_corr": round(challenge_corr, 3),
                "spec_ratio": round(spec_ratio, 3),
                "lag": best_lag,
            }

        # 4. 第二关：NLMS + 双投影(POS/CHROM) + 三通道 SVD 正交剥离屏幕跳变残留
        residual_rgb, aligned_sent = self._strip_screen_light_nlms(skin_f, sent_f, lag=best_lag)
        bvps, bvp_method, bvp_snr = self.adapter.extract_bvp_dual(residual_rgb, fps=fps)

        sent_edges_3ch = self._bandpass_edges(aligned_sent, fps)[12:]
        S = sent_edges_3ch - np.mean(sent_edges_3ch, axis=0, keepdims=True)
        U, s_vals, _ = np.linalg.svd(S, full_matrices=False)
        U_valid = U[:, s_vals > 1e-4]

        bvp_trim = bvps[:, 12:].copy()
        for p in range(bvp_trim.shape[0]):
            b = bvp_trim[p] - np.mean(bvp_trim[p])
            if U_valid.shape[1] > 0:
                b = b - U_valid @ (U_valid.T @ b)
            bvp_trim[p] = b

        # 遮挡容错空间同步性
        corr_mat = np.corrcoef(bvp_trim)
        off_diag = corr_mat[~np.eye(corr_mat.shape[0], dtype=bool)]
        mean_sync = float(np.nanmean(off_diag))
        cheeks_sync = float(corr_mat[1, 2]) if not np.isnan(corr_mat[1, 2]) else mean_sync
        spatial_sync = max(mean_sync, cheeks_sync)

        # 5. 计算残留脉搏 PSD 频谱与心率主峰
        mean_bvp = bvp_trim.mean(axis=0)
        n_fft = max(256, len(mean_bvp) * 4)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)
        psd = np.abs(np.fft.rfft(mean_bvp * np.hanning(len(mean_bvp)), n=n_fft)) ** 2
        valid_band = (freqs >= 0.7) & (freqs <= 3.0)
        est_bpm = float(freqs[valid_band][np.argmax(psd[valid_band])] * 60.0)

        if spatial_sync < self.sync_floor:
            return {
                "live": False,
                "reason": "REJECT_MASK_OR_NO_PULSE",
                "challenge_corr": round(challenge_corr, 3),
                "spatial_sync": round(spatial_sync, 3),
                "spec_ratio": round(spec_ratio, 3),
            }

        # 6. 多特征联合置信度评分
        c_norm = float(np.clip((challenge_corr - 0.35) / 0.45, 0.0, 1.0))
        s_norm = float(np.clip((spatial_sync - 0.10) / 0.70, 0.0, 1.0))
        snr_norm = float(np.clip(bvp_snr / 0.45, 0.0, 1.0))
        spec_bonus = float(np.clip(abs(spec_ratio - 1.0) / 0.35, 0.0, 1.0))

        liveness_score = (
            0.48 * c_norm + 0.30 * s_norm + 0.14 * snr_norm + 0.08 * spec_bonus
        )

        plot_band = (freqs >= 0.5) & (freqs <= 3.5)
        psd_sub = psd[plot_band]
        psd_norm = (psd_sub / (np.max(psd_sub) + 1e-8)).tolist()

        diagnostics = {
            "bvp_method": bvp_method,
            "sent_wave": np.round(s_wave, 3).tolist(),
            "skin_wave": np.round(r_wave, 3).tolist(),
            "bvp_rois": [np.round(bvp_trim[p] / (np.std(bvp_trim[p]) + 1e-6), 3).tolist() for p in range(3)],
            "fft_freqs": np.round(freqs[plot_band], 2).tolist(),
            "fft_psd": [round(float(v), 3) for v in psd_norm],
        }

        passed = liveness_score >= self.joint_score_thresh
        return {
            "live": passed,
            "reason": "PASS_LIVE_PERSON" if passed else "REJECT_LOW_CONFIDENCE",
            "liveness_score": round(liveness_score, 3),
            "challenge_corr": round(challenge_corr, 3),
            "spatial_sync": round(spatial_sync, 3),
            "spec_ratio": round(spec_ratio, 3),
            "bvp_snr": round(bvp_snr, 3),
            "residual_bpm": round(est_bpm, 1),
            "diagnostics": diagnostics,
        }