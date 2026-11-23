print("[0/3] 正在加载依赖库...", flush=True)

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import cv2
import numpy as np

from core.challenge_engine import LivenessVerifier, generate_challenge_sequence
from core.pyvhr_adapter import InMemoryPyVHRAdapter

try:
    from eval.benchmark_eval import evaluate_metrics, log_trial
except ImportError as e:
    print(f"[警告] eval/benchmark_eval.py 导入不完整 ({e})，将仅运行实时检测不记录日志。", flush=True)
    log_trial = None
    evaluate_metrics = None

KEY_TO_LABEL = {
    ord("1"): "live",
    ord("2"): "replay",
    ord("3"): "static",
    ord(" "): None,
}


def main():
    print("[1/3] 正在初始化 pyVHR 内存提取器...", flush=True)
    adapter = InMemoryPyVHRAdapter(patch_size=28)
    verifier = LivenessVerifier(
        adapter=adapter,
        replay_thresh=0.55,
        pulse_sync_thresh=0.35,
        max_lag=10,
    )

    print("[2/3] 正在打开摄像头...", flush=True)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头，请检查是否被占用。", flush=True)
        adapter.close()
        return

    win_name = "PulseProof Liveness Verification"
    target_fps = 25.0
    duration_sec = 3.0
    num_frames = int(target_fps * duration_sec)
    last_banner = "[1]Live [2]Replay [3]Static [SPACE]Test [E]Eval [Q]Quit"
    banner_color = (0, 220, 120)

    print("[3/3] 就绪！按 [1]测真人 | [2]测视频回放 | [3]测静态/反光 | [E]看评测指标 | [Q]退出", flush=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(10)
            continue

        _, _, boxes = adapter.process_frame(frame)
        preview = frame.copy()

        if boxes:
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 120), 2)
            status_txt, status_col = "Face: LOCKED (3 ROIs)", (0, 220, 120)
        else:
            status_txt, status_col = "Face: LOST", (0, 60, 255)

        cv2.putText(preview, last_banner, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, banner_color, 2)
        cv2.putText(preview, status_txt, (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_col, 2)
        cv2.imshow(win_name, preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("e"):
            if evaluate_metrics is not None:
                print("\n" + "=" * 50, flush=True)
                evaluate_metrics(
                    replay_thresh=verifier.replay_thresh,
                    sync_thresh=verifier.pulse_sync_thresh,
                )
                print("=" * 50 + "\n", flush=True)
        elif key in KEY_TO_LABEL:
            gt_label = KEY_TO_LABEL[key]
            tag_str = f"记录标签={gt_label}" if gt_label else "普通测试(不记录)"
            print(f">>> 开始 3 秒挑战 [{tag_str}]...", flush=True)

            challenge_rgb = generate_challenge_sequence(num_frames=num_frames, fps=target_fps)
            sent_log, skin_log, bg_log = [], [], []

            canvas_h, canvas_w = 900, 1440
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            pip_w, pip_h = 360, 270
            px1, py1 = (canvas_w - pip_w) // 2, (canvas_h - pip_h) // 2

            t_start = time.time()
            for i in range(num_frames):
                ok, cam_frame = cap.read()
                if not ok:
                    cv2.waitKey(1)
                    continue

                rgb = challenge_rgb[i]
                canvas[:, :] = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

                skin_patches, bg_rgb, boxes = adapter.process_frame(cam_frame)
                if skin_patches is not None:
                    sent_log.append(rgb)
                    skin_log.append(skin_patches)
                    bg_log.append(bg_rgb)

                cam_h, cam_w, _ = cam_frame.shape
                pip = cv2.resize(cam_frame, (pip_w, pip_h))
                canvas[py1 : py1 + pip_h, px1 : px1 + pip_w] = pip
                cv2.rectangle(canvas, (px1, py1), (px1 + pip_w, py1 + pip_h), (255, 255, 255), 2)

                if boxes:
                    sx, sy = pip_w / cam_w, pip_h / cam_h
                    for (x1, y1, x2, y2) in boxes:
                        cv2.rectangle(
                            canvas,
                            (px1 + int(x1 * sx), py1 + int(y1 * sy)),
                            (px1 + int(x2 * sx), py1 + int(y2 * sy)),
                            (0, 220, 120),
                            2,
                        )

                cv2.putText(
                    canvas,
                    f"Verifying {i + 1}/{num_frames} (Valid: {len(skin_log)})",
                    (px1, py1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(win_name, canvas)
                cv2.waitKey(1)

            elapsed = max(time.time() - t_start, 0.5)
            actual_fps = len(skin_log) / elapsed

            result = verifier.verify(
                sent_rgb=np.array(sent_log),
                skin_rgb=np.array(skin_log),
                bg_rgb=np.array(bg_log),
                fps=actual_fps,
            )
            print(f"[{actual_fps:.1f} FPS | {len(skin_log)}/{num_frames}] => {result}", flush=True)

            if gt_label and log_trial is not None and result.get("reason") != "INSUFFICIENT_FACE_FRAMES":
                log_trial(gt_label, result)
                print(f"    已写入评测日志: label='{gt_label}'", flush=True)

            if result.get("live"):
                last_banner = (
                    f"PASS | Corr:{result['challenge_corr']} | "
                    f"Sync:{result['spatial_sync']} | HR:{result['residual_bpm']}"
                )
                banner_color = (0, 220, 120)
            else:
                last_banner = f"REJECT: {result['reason']} (Corr:{result.get('challenge_corr', 'N/A')})"
                banner_color = (0, 60, 255)

    cap.release()
    cv2.destroyAllWindows()
    adapter.close()


if __name__ == "__main__":
    main()