# -*- coding: utf-8 -*-
"""
pulseproof/eval/benchmark_replay.py
录制真人与手机翻拍测试集，统计 APCER / BPCER 与阈值标定

[待补全] 规范后续步骤设计尚未收到 (消息截断)。当前仅建骨架。
"""
import numpy as np


class ReplayBenchmark:
    """回放攻击基准评测 (骨架)"""

    def __init__(self):
        self.samples = []   # [(label, score), ...]  label: 'live' / 'replay'

    def add_sample(self, label, score):
        """记录一次判定分数; label ∈ {'live','replay'}"""
        self.samples.append((label, float(score)))

    def compute_apcer_bpcer(self, threshold):
        """TODO(规范待补全): 按阈值统计 APCER(攻击误通过率) / BPCER(真人误拒率)"""
        raise NotImplementedError('等待规范后续步骤设计')

    def calibrate_threshold(self, target_acer=None, target_bcer=None):
        """TODO(规范待补全): 阈值标定 (EER / 给定误报率下的最优阈值)"""
        raise NotImplementedError('等待规范后续步骤设计')
