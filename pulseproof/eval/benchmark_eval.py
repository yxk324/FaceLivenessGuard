import json
from pathlib import Path
import numpy as np

LOG_FILE = Path(__file__).parent / "trials_log.jsonl"


def log_trial(ground_truth: str, result: dict):
    """记录带真值标签的单次测试结果（'live' | 'replay' | 'static'）。"""
    record = {"label": ground_truth, **result}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_metrics(replay_thresh: float = 0.52, sync_thresh: float = 0.35):
    if not LOG_FILE.exists():
        print("暂无评测日志，请先按 [1]/[2]/[3] 采集样本。")
        return

    records = [json.loads(line) for line in open(LOG_FILE, encoding="utf-8") if line.strip()]
    live_recs = [r for r in records if r["label"] == "live"]
    spoof_recs = [r for r in records if r["label"] in ("replay", "static")]

    def predict(r):
        return (
            r.get("challenge_corr", -1.0) >= replay_thresh
            and r.get("spatial_sync", -1.0) >= sync_thresh
        )

    bpcer = sum(1 for r in live_recs if not predict(r)) / max(len(live_recs), 1)
    apcer = sum(1 for r in spoof_recs if predict(r)) / max(len(spoof_recs), 1)
    acer = 0.5 * (apcer + bpcer)

    print(f"累计样本: 真人 {len(live_recs)} 条 | 攻击 {len(spoof_recs)} 条")
    print(f"当前阈值 (replay={replay_thresh}, sync={sync_thresh}):")
    print(f"  BPCER (真人误拒率): {bpcer * 100:.2f}%")
    print(f"  APCER (攻击通过率): {apcer * 100:.2f}%")
    print(f"  ACER  (平均错误率): {acer * 100:.2f}%")