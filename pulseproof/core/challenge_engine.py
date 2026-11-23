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

def check_surface_reflection_ratio(skin_rgb: np.ndarray, sent_rgb: np.ndarray, lag: int) -> float:
    """
    借鉴 SpecDiff 原理：计算屏幕最高亮度帧与最低亮度帧下，
    额头区(ROI 0)与双颊区(ROI 1, 2)的反射增益比，过滤平面玻璃均匀镜面反射。
    """
    aligned_sent = np.roll(sent_rgb, shift=lag, axis=0)
    luma = aligned_sent.mean(axis=1)
    high_idx = np.argsort(luma)[-10:]  # 屏幕最亮的 10 帧
    low_idx = np.argsort(luma)[:10]    # 屏幕最暗的 10 帧

    delta_forehead = np.mean(skin_rgb[high_idx, 0] - skin_rgb[low_idx, 0])
    delta_cheeks = np.mean(skin_rgb[high_idx, 1:] - skin_rgb[low_idx, 1:])
    # 平面玻璃或屏幕翻拍时各区域增益高度一致（比值趋近 1.0），立体人脸存在法线光照衰减差异
    return float(delta_forehead / (delta_cheeks + 1e-6))

def generate_challenge_sequence(num_frames: int = 75, fps: float = 25.0) -> np.ndarray:
    """生成高亮度浅彩色块阶跃挑战序列 C(t)，形状 (T, 3)。"""
    rng = np.random.default_rng()
    num_blocks = 8
    block_edges = np.linspace(0, num_frames, num_blocks + 1, dtype=int)

    seq = np.zeros((num_frames, 3), dtype=np.uint8)
    prev_idx = -1

    for b in range(num_blocks):
        candidates = [i for i in range(len(_HIGH_LUMA_PALETTE)) if i != prev_idx]
        idx = int(rng.choice(candidates))
        prev_idx = idx
        seq[block_edges[b] : block_edges[b + 1]] = _HIGH_LUMA_PALETTE[idx]

    return seq


class LivenessVerifier:
    """PulseProof 主动光挑战-应答与残差 rPPG 活体校验器。"""

    def __init__(
        self,
        adapter: InMemoryPyVHRAdapter,
        replay_thresh: float = 0.55,
        pulse_sync_thresh: float = 0.35,
        max_lag: int = 10,
    ):
        self.adapter = adapter
        self.replay_thresh = replay_thresh
        self.pulse_sync_thresh = pulse_sync_thresh
        self.max_lag = max_lag

    @staticmethod
    def _bandpass_edges(sig: np.ndarray, fps: float) -> np.ndarray:
        """0.8 - 5.5 Hz 带通滤波：剥离缓慢 AE/AWB 漂移，保留色块跳变边缘。"""
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
    ) -> tuple[float, int]:
        """第一道防线（防高清屏幕回放）：比对色块跳变边缘的延迟对齐互相关。"""
        mean_skin = skin_rgb.mean(axis=1)
        sent_linear = np.power(sent_rgb / 255.0, 2.2)

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

        return best_score, best_lag

    @staticmethod
    def _strip_screen_light_nlms(
        skin_rgb: np.ndarray, sent_rgb: np.ndarray, lag: int, mu: float = 0.25, taps: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        """NLMS 自适应滤波剥离屏幕打光分量，并返回对齐后的发送序列用于正交投影。"""
        T, P, C = skin_rgb.shape
        sent_linear = np.power(sent_rgb / 255.0, 2.2) * 255.0
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
        self, sent_rgb: np.ndarray, skin_rgb: np.ndarray, bg_rgb: np.ndarray, fps: float
    ) -> dict:
        """执行完整三层活体判定。"""
        if len(skin_rgb) < 40:
            return {"live": False, "reason": "INSUFFICIENT_FACE_FRAMES"}

        sent_f = sent_rgb.astype(np.float32)
        skin_f = skin_rgb.astype(np.float32)

        # 第一关：色块阶跃挑战校验（拦截高清视频回放）
        challenge_corr, best_lag = self._check_replay_correlation(sent_f, skin_f, fps=fps)
        if challenge_corr < self.replay_thresh:
            return {
                "live": False,
                "reason": "REJECT_SCREEN_REPLAY",
                "challenge_corr": round(challenge_corr, 3),
                "lag": best_lag,
            }

        # 第二关：NLMS 剥离屏幕打光 + 调用 pyVHR POS 提取残差脉搏波
        residual_rgb, aligned_sent = self._strip_screen_light_nlms(skin_f, sent_f, lag=best_lag)
        bvps = self.adapter.extract_bvp(residual_rgb, fps=fps)

        # 正交投影剥离残余的屏幕跳变沿振铃，防止反光面具产生假同步
        sent_edges_3ch = self._bandpass_edges(aligned_sent, fps)[12:]  # shape: (T', 3)
        S = sent_edges_3ch - np.mean(sent_edges_3ch, axis=0, keepdims=True)
        U, s_vals, _ = np.linalg.svd(S, full_matrices=False)
        U_valid = U[:, s_vals > 1e-4]  # 提取 R/G/B 屏幕跳变的正交子空间基

        bvp_trim = bvps[:, 12:].copy()
        for p in range(bvp_trim.shape[0]):
            b = bvp_trim[p] - np.mean(bvp_trim[p])
            if U_valid.shape[1] > 0:
                b = b - U_valid @ (U_valid.T @ b)
            bvp_trim[p] = b

        corr_mat = np.corrcoef(bvp_trim)
        off_diag = corr_mat[~np.eye(corr_mat.shape[0], dtype=bool)]
        spatial_sync = float(np.nanmean(off_diag))

        mean_bvp = bvp_trim.mean(axis=0)
        freqs = np.fft.rfftfreq(len(mean_bvp) * 4, d=1.0 / fps)
        spec = np.abs(np.fft.rfft(mean_bvp * np.hanning(len(mean_bvp)), n=len(mean_bvp) * 4))
        valid_band = (freqs >= 0.7) & (freqs <= 3.0)
        est_bpm = float(freqs[valid_band][np.argmax(spec[valid_band])] * 60.0)

        # 第三关：拦截 3D 面具或静态照片反光
        if spatial_sync < self.pulse_sync_thresh:
            return {
                "live": False,
                "reason": "REJECT_MASK_OR_NO_PULSE",
                "challenge_corr": round(challenge_corr, 3),
                "spatial_sync": round(spatial_sync, 3),
            }

        return {
            "live": True,
            "reason": "PASS_LIVE_PERSON",
            "challenge_corr": round(challenge_corr, 3),
            "spatial_sync": round(spatial_sync, 3),
            "residual_bpm": round(est_bpm, 1),
        }