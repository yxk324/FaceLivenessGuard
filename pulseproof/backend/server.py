import base64
import sys
import time
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

app = FastAPI(title="PulseProof Liveness API")
adapter = InMemoryPyVHRAdapter(patch_size=28)
verifier = LivenessVerifier(adapter=adapter, replay_thresh=0.52, pulse_sync_thresh=0.35, max_lag=10)

# 存储活跃挑战会话：{session_id: {"seq": np.ndarray, "created_at": float}}
ACTIVE_SESSIONS: dict[str, dict] = {}


class VerifyPayload(BaseModel):
    session_id: str
    fps: float
    frames_b64: list[str]


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = ROOT_DIR / "frontend" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/api/challenge")
def create_challenge(num_frames: int = 75, fps: float = 25.0):
    # 清理超过 60 秒的过期会话
    now = time.time()
    expired = [k for k, v in ACTIVE_SESSIONS.items() if now - v["created_at"] > 60]
    for k in expired:
        del ACTIVE_SESSIONS[k]

    session_id = uuid.uuid4().hex
    seq = generate_challenge_sequence(num_frames=num_frames, fps=fps)
    ACTIVE_SESSIONS[session_id] = {"seq": seq, "created_at": now}

    return {
        "session_id": session_id,
        "fps": fps,
        "num_frames": num_frames,
        "colors": seq.tolist(),
    }


@app.post("/api/verify")
def verify_liveness(payload: VerifyPayload):
    session = ACTIVE_SESSIONS.pop(payload.session_id, None)
    if session is None:
        raise HTTPException(status_code=400, detail="挑战会话不存在或已过期")

    sent_seq = session["seq"]
    sent_log, skin_log, bg_log = [], [], []

    limit = min(len(sent_seq), len(payload.frames_b64))
    for i in range(limit):
        b64_str = payload.frames_b64[i].split(",")[-1]
        img_bytes = base64.b64decode(b64_str)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        skin_patches, bg_rgb, _ = adapter.process_frame(frame_bgr)
        if skin_patches is not None:
            sent_log.append(sent_seq[i])
            skin_log.append(skin_patches)
            bg_log.append(bg_rgb)

    if len(skin_log) < 40:
        return {"live": False, "reason": "INSUFFICIENT_FACE_FRAMES", "valid_frames": len(skin_log)}

    result = verifier.verify(
        sent_rgb=np.array(sent_log),
        skin_rgb=np.array(skin_log),
        bg_rgb=np.array(bg_log),
        fps=payload.fps,
    )
    result["valid_frames"] = len(skin_log)
    return result