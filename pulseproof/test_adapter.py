import time
import cv2
import numpy as np
from core.pyvhr_adapter import InMemoryPyVHRAdapter


def main():
    adapter = InMemoryPyVHRAdapter(patch_size=28)
    cap = cv2.VideoCapture(0)

    skin_buffer = []
    bg_buffer = []
    t_start = time.time()

    print("正在测试内存适配器：将采集 100 帧并计算三区域 BVP 波形（按 q 提前退出）...")

    while cap.isOpened() and len(skin_buffer) < 100:
        ret, frame = cap.read()
        if not ret:
            break

        skin_rgb, bg_rgb, boxes = adapter.process_frame(frame)

        if skin_rgb is not None:
            skin_buffer.append(skin_rgb)
            bg_buffer.append(bg_rgb)

            # 在画面上画出额头、左颊、右颊 3 个皮肤采样框
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 120), 2)

        cv2.putText(
            frame,
            f"Captured valid frames: {len(skin_buffer)}/100",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 120),
            2,
        )
        cv2.imshow("InMemory pyVHR Adapter Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    adapter.close()

    if len(skin_buffer) >= 40:
        elapsed = max(time.time() - t_start, 0.5)
        fps = len(skin_buffer) / elapsed
        skin_arr = np.array(skin_buffer)  # (T, 3, 3)
        bvps = adapter.extract_bvp(skin_arr, fps=fps)
        print(f"成功！采集帧率: {fps:.1f} FPS")
        print(f"多子块皮肤张量形状: {skin_arr.shape}")
        print(f"输出 BVP 波形形状: {bvps.shape} (3 个区域各自独立的脉搏波)")
    else:
        print("有效人脸帧不足，请确保人脸正对摄像头且光线充足。")


if __name__ == "__main__":
    main()