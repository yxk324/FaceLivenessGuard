# -*- coding: utf-8 -*-
"""
人脸关键点检测测试 (最终版) —— 与 pyVHR 相同的 mediapipe FaceMesh API
绘图用纯 cv2 (绕开 mediapipe draw_landmarks 的原生崩溃, pyVHR 主流程也不使用它)
用法:
  python Temp/face_test.py          # 自动探测 0~3 号摄像头, 选画面最亮的
  python Temp/face_test.py 1        # 指定 1 号摄像头
  python Temp/face_test.py xx.mp4   # 视频文件模式 (处理120帧)
"""
import sys
import time

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


def probe_cameras(user_index=None):
    if user_index is not None:
        return user_index
    best, best_score = None, -1
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        ok, frame = cap.read()
        if ok and frame is not None:
            bright = float(frame.mean())
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            print(f'  摄像头 #{idx}: {w}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}, '
                  f'亮度={bright:.1f}{"  <-- 全黑!" if bright < 10 else ""}')
            score = bright + (w * 10) if bright > 10 else -1
            if score > best_score:
                best, best_score = idx, score
        else:
            print(f'  摄像头 #{idx}: 打不开')
        cap.release()
    return best


def main():
    user_index = None
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            user_index = int(sys.argv[1])
            is_file = False
            video_source = user_index
            print(f'使用摄像头 #{user_index}')
        else:
            is_file = True
            video_source = sys.argv[1]
            print(f'使用视频文件: {video_source}')
    else:
        is_file = False
        print('探测摄像头 (0~3):')
        video_source = probe_cameras()
        if video_source is None:
            print('[FAIL] 没有可用摄像头')
            return 1
        print(f'自动选择摄像头 #{video_source}')

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print('[FAIL] 打不开视频源:', video_source)
        return 1

    # ---- 与 pyVHR sig_processing.py 相同的 FaceMesh 配置 ----
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    n_frames = n_detected = 0
    t0 = time.time()
    saved = False
    max_frames = 120 if is_file else 10**9

    while n_frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            print('读取帧失败, 结束')
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            n_detected += 1
            h, w = frame.shape[:2]
            # ---- 纯 cv2 绘制关键点 (mediapipe draw_landmarks 在本机有原生崩溃, 不用) ----
            for lms in results.multi_face_landmarks:
                for pt in lms.landmark:
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, (0, 255, 0), -1)
            if n_frames % 30 == 0:
                print(f'frame {n_frames:4d}: [检出] 关键点={len(lms.landmark)} '
                      f'亮度={float(frame.mean()):.1f}')
            if not saved:
                cv2.imwrite(r'E:\rppG\pyVHR\Temp\face_test_output.png', frame)
                print('已保存标注截图 -> Temp/face_test_output.png')
                saved = True
        elif n_frames % 30 == 0:
            print(f'frame {n_frames:4d}: [未检出] 亮度={float(frame.mean()):.1f}')

        n_frames += 1
        cv2.imshow('face test (ESC/q to quit)', frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    dt = time.time() - t0
    rate = n_detected / n_frames * 100 if n_frames else 0
    fps = n_frames / dt if dt > 0 else 0
    print('-' * 50)
    print(f'处理帧数: {n_frames} | 检出人脸帧: {n_detected} ({rate:.1f}%) | 平均 FPS: {fps:.1f}')
    if n_detected > 0:
        print('[PASS] mediapipe 人脸检测工作正常')
        return 0
    print('[WARN] 通道正常但没检测到人脸 - 换摄像头: python Temp/face_test.py 1, 或改善光照')
    return 2

if __name__ == '__main__':
    sys.exit(main())
