import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import base64
from contextlib import asynccontextmanager
import sys
import time
from typing import Optional
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.challenge_engine import LivenessVerifier, generate_challenge_sequence
from core.pyvhr_adapter import InMemoryPyVHRAdapter

adapter: Optional[InMemoryPyVHRAdapter] = None
verifier: Optional[LivenessVerifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global adapter, verifier
    adapter = InMemoryPyVHRAdapter(patch_size=28)
    verifier = LivenessVerifier(
        adapter=adapter,
        replay_floor=0.45,
        sync_floor=0.15,
        joint_score_thresh=0.56,
        max_lag=10,
    )
    yield
    if adapter is not None:
        adapter.close()


app = FastAPI(title="PulseProof Liveness API", lifespan=lifespan)

ACTIVE_SESSIONS: dict[str, dict] = {}
VIRTUAL_CAM_KEYWORDS = ("obs", "manycam", "splitcam", "youcam", "snap camera", "virtual", "vcam", "droidcam")


class VerifyPayload(BaseModel):
    session_id: str
    fps: float
    frames_b64: list[str]
    timestamps_ms: Optional[list[float]] = None
    rendered_colors: Optional[list[list[int]]] = None
    camera_label: str = ""


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = ROOT_DIR / "frontend" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/api/challenge")
def create_challenge(num_frames: int = 75, fps: float = 25.0):
    now = time.time()
    expired = [k for k, v in ACTIVE_SESSIONS.items() if now - v["created_at"] > 60]
    for k in expired:
        del ACTIVE_SESSIONS[k]

    session_id = uuid.uuid4().hex
    seq = generate_challenge_sequence(num_frames=num_frames, fps=fps, min_color_dist=135.0)
    ACTIVE_SESSIONS[session_id] = {"seq": seq, "created_at": now}

    return {
        "session_id": session_id,
        "fps": fps,
        "num_frames": num_frames,
        "colors": seq.tolist(),
    }


@app.post("/api/verify")
def verify_liveness(payload: VerifyPayload):
    if adapter is None or verifier is None:
        raise HTTPException(status_code=503, detail="活体检测引擎尚未初始化完成")

    session = ACTIVE_SESSIONS.pop(payload.session_id, None)
    if session is None:
        raise HTTPException(status_code=400, detail="挑战会话不存在或已过期")

    # 1. 软件虚拟摄像头拦截（防御 OBS / DeepFaceLive 软注入）
    cam_lower = payload.camera_label.lower()
    is_virtual = any(kw in cam_lower for kw in VIRTUAL_CAM_KEYWORDS)
    if is_virtual:
        return {"live": False, "reason": "REJECT_VIRTUAL_CAMERA", "camera_label": payload.camera_label}

    # 2. 服务端采集时间跨度防作弊校验
    if payload.timestamps_ms and len(payload.timestamps_ms) >= 2:
        span_sec = (payload.timestamps_ms[-1] - payload.timestamps_ms[0]) / 1000.0
        if span_sec < 1.8 or span_sec > 7.0:
            return {"live": False, "reason": "REJECT_TIMING_TAMPERED", "span_sec": round(span_sec, 2)}

    sent_seq = session["seq"]
    sent_log, skin_log, bg_log = [], [], []
    ts_log, glare_log, center_log = [], [], []

    limit = min(len(sent_seq), len(payload.frames_b64))
    for i in range(limit):
        b64_str = payload.frames_b64[i].split(",")[-1]
        img_bytes = base64.b64decode(b64_str)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        skin_patches, bg_rgb, _, meta = adapter.process_frame_detailed(frame_bgr)
        if skin_patches is not None:
            color_i = sent_seq[i]
            if payload.rendered_colors and i < len(payload.rendered_colors):
                client_c = np.array(payload.rendered_colors[i], dtype=np.float32)
                if np.linalg.norm(client_c - color_i.astype(np.float32)) > 25.0:
                    return {"live": False, "reason": "REJECT_COLOR_SPOOFED"}

            sent_log.append(color_i)
            skin_log.append(skin_patches)
            bg_log.append(bg_rgb)
            glare_log.append(meta["glare_ratio"])
            center_log.append(meta["face_center"])
            if payload.timestamps_ms and i < len(payload.timestamps_ms):
                ts_log.append(payload.timestamps_ms[i])

    if len(skin_log) < 40:
        return {"live": False, "reason": "INSUFFICIENT_FACE_FRAMES", "valid_frames": len(skin_log)}

    result = verifier.verify(
        sent_rgb=np.array(sent_log),
        skin_rgb=np.array(skin_log),
        bg_rgb=np.array(bg_log),
        fps=payload.fps,
        timestamps_ms=np.array(ts_log) if len(ts_log) == len(skin_log) else None,
        glare_ratios=glare_log,
        face_centers=center_log,
        is_virtual_cam=is_virtual,
    )
    result["valid_frames"] = len(skin_log)
    return result