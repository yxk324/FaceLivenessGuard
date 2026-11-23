# -*- coding: utf-8 -*-
"""
用摄像头录制一段测试视频, 保存到项目 Temp 文件夹 (不写 C 盘)
用法: python Temp/record_face.py [录制秒数, 默认20]
录制建议: 正对摄像头、光照稳定、头部尽量少动、表情自然 —— rPPG 需要皮肤颜色的细微变化
"""
import sys
import time

import cv2

def main():
    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = r'E:\rppG\pyVHR\Temp\test_face.mp4'

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[FAIL] 摄像头打不开 (index 0)')
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f'录制 {secs}s @ {fps:.0f}fps {w}x{h} -> {out}   (按 q 提前结束)')
    t0 = time.time()
    n = 0
    while time.time() - t0 < secs:
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(frame)
        n += 1
        cv2.imshow('recording... (q to stop)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    vw.release()
    cv2.destroyAllWindows()
    print(f'完成: {n} 帧, 约 {n/fps:.1f} 秒 -> {out}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
