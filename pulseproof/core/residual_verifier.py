# -*- coding: utf-8 -*-
"""
pulseproof/core/residual_verifier.py
NLMS 自适应滤波剥离屏幕光 + 多子块脉搏同步校验（防 3D 面具）

[待补全] 本文件对应规范后续步骤, 详细设计尚未收到 (消息截断)。当前仅建骨架。
"""
import numpy as np


class ResidualVerifier:
    """残差脉搏校验器 (骨架)"""

    def __init__(self, mu=0.01, filter_order=16):
        self.mu = float(mu)
        self.filter_order = int(filter_order)

    def strip_challenge_light(self, face_rgb_sig, challenge_seq):
        """TODO(规范待补全): NLMS 自适应滤波, 从人脸 RGB 中剥离屏幕光成分"""
        raise NotImplementedError('等待规范后续步骤设计')

    def verify_pulse_sync(self, block_bvps):
        """TODO(规范待补全): 多子块脉搏同步性校验, 判定 3D 面具 / 真脸"""
        raise NotImplementedError('等待规范后续步骤设计')
