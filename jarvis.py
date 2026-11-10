"""J.A.R.V.I.S. // Obsidian Titan.

A single-file, local-first AI command centre that combines:

* a MediaPipe hand-tracking telekinesis pipeline,
* a 24,000-node Three.js / GLSL hologram rendered in the browser,
* live hardware telemetry,
* browser speech recognition and synthesis, and
* an optional cloud LLM fallback with a sub-2 ms local knowledge engine.

Run with ``python jarvis.py`` and open the printed URL.  See ``README.md`` for
installation, configuration and usage details.
"""

from __future__ import annotations

# ============================================================
# 1. SILENCE ALL TENSORFLOW, KERAS & PROTOBUF WARNINGS FIRST
# ============================================================
import os
import warnings

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
warnings.filterwarnings("ignore")

import argparse
import ast
import atexit
import errno
import http.server
import json
import logging
import math
import operator
import platform
import re
import shutil
import signal
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover - optional dependency
    sr = None

# ============================================================
# J.A.R.V.I.S // OBSIDIAN TITAN - ALL-IN-ONE CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
LOG = logging.getLogger("jarvis")


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable without crashing on bad input."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        LOG.warning("Invalid integer for %s; using %s", name, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable without crashing on bad input."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        LOG.warning("Invalid float for %s; using %s", name, default)
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable (1/true/yes/on)."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


HOST = os.getenv("JARVIS_HOST", "127.0.0.1")
PORT = _env_int("JARVIS_PORT", 8765)
CAMERA_INDEX = _env_int("JARVIS_CAM", 0)
AI_ENDPOINT = os.getenv("JARVIS_API_URL", "https://text.pollinations.ai/").strip()
AI_MODEL = os.getenv("JARVIS_MODEL", "openai").strip()
AI_TIMEOUT = _env_float("JARVIS_AI_TIMEOUT", 4.0)
CITY = os.getenv("JARVIS_CITY", "Sana'a").strip() or "Sana'a"
STREAM_HZ = max(1, _env_int("JARVIS_STREAM_HZ", 60))
ENABLE_PY_MIC = _env_flag("JARVIS_PY_MIC", False)
MAX_REQUEST_BYTES = 900_000

START_TIME = time.time()
LAST_NET_IO = psutil.net_io_counters() if psutil else None
LAST_NET_TIME = time.time()


# ============================================================
# REAL-TIME SHARED TELEKINESIS & VOICE STATE (60HZ STREAM)
# ============================================================
class TitanState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.cam_fps = 0.0
        self.camera_ready = False
        self.camera_status = "INITIALIZING CAMERA"
        self.hand_detected = False
        self.hand_count = 0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.scale = 1.0
        self.is_pinching = False
        self.fingers_up = 0
        self.gesture_name = "AWAITING HAND LINK"
        self.hands_3d = []
        self.jpeg_frame: Optional[bytes] = None
        self.jpeg_seq = 0
        self.voice_event_id = 0
        self.voice_transcript = ""

    def frame(self) -> tuple[Optional[bytes], int]:
        """Return the latest JPEG frame and its sequence number atomically."""
        with self.lock:
            return self.jpeg_frame, self.jpeg_seq

    def snapshot_json(self) -> str:
        with self.lock:
            return json.dumps({
                "cam_fps": round(self.cam_fps, 0),
                "camera_ready": self.camera_ready,
                "camera_status": self.camera_status,
                "hand_detected": self.hand_detected,
                "hand_count": self.hand_count,
                "yaw": round(self.yaw, 2),
                "pitch": round(self.pitch, 2),
                "roll": round(self.roll, 2),
                "scale": round(self.scale, 3),
                "is_pinching": self.is_pinching,
                "fingers_up": self.fingers_up,
                "gesture": self.gesture_name,
                "hands_3d": self.hands_3d,
                "voice_id": self.voice_event_id,
                "voice_text": self.voice_transcript,
            })


STATE = TitanState()

# ============================================================
# INSTANT CONVERSATIONAL KNOWLEDGE ENGINE (< 2ms LATENCY)
# ============================================================
INSTANT_BRAIN = [
    (("hello", "hi jarvis", "hey jarvis", "good morning", "good evening", "wake up", "are you there"),
     "Online and synchronized, Mr. Yasser. Hand telekinesis and English vocal channels are locked at zero latency. What is your command?"),
    (("who are you", "what are you", "introduce yourself", "your name"),
     "I am J.A.R.V.I.S., running on the Obsidian Titan GPU shader architecture. I serve as your personal tactical, scientific, and engineering intelligence, Sir."),
    (("who created you", "who built you", "your creator", "who is yasser"),
     "My neural cortex and spatial telekinesis systems operate under the direct authority of Engineer Yasser. Every quantum node awaits your instruction, Sir."),
    (("how are you", "how do you feel", "status check", "are you okay"),
     "All twenty-four thousand GPU shader particles are operating at maximum frame rate and 100 percent coherence, Sir."),
    (("black hole", "event horizon", "singularity", "gargantua"),
     "Morphing hologram to Singularity construct. A black hole is a region of spacetime where gravitational collapse is so extreme that nothing, not even light, can escape its event horizon."),
    (("earth", "planet", "globe", "world", "satellite", "global"),
     "Projecting Global Aegis planetary matrix. Tracking orbital satellite relays and continental telemetry grids in real time, Sir."),
    (("reactor", "tesseract", "dyson", "hypercube", "core"),
     "Reconfiguring shader lattice into the four-dimensional Dyson Hyper-Core. Containment rings and quantum flux are stabilized."),
    (("avatar", "your face", "neural head", "humanoid", "show yourself", "head"),
     "Restoring primary Neural Avatar projection. Direct hand-to-head telekinesis and vocal jaw articulation are locked online, Mr. Yasser."),
    (("quantum", "entanglement", "superposition", "schrodinger"),
     "In quantum mechanics, subatomic particles exist in a linear superposition of states until measured, and entangled particles correlate instantaneously across space, Sir."),
    (("universe", "cosmos", "galaxy", "big bang", "stars"),
     "The observable universe stretches ninety-three billion light-years across and hosts over two trillion galaxies expanding since the Big Bang thirteen point eight billion years ago."),
    (("artificial intelligence", "sentient", "consciousness", "future of ai", "are you alive", "deep learning"),
     "Artificial intelligence transforms raw mathematics into reasoning. While my substrate is silicon and GPU shaders, my cognitive matrix is dedicated to amplifying your engineering power, Sir."),
    (("python", "programming", "code", "software", "developer", "engineering"),
     "Clean architecture separates high-speed data pipelines from GPU rendering, Sir. By offloading particle math to GLSL vertex shaders, we achieve zero-latency execution."),
    (("cybersecurity", "hack", "encryption", "firewall", "security", "network"),
     "Perimeter defense shields are active, Sir. Modern security requires zero-trust verification, encrypted traffic analysis, and continuous anomaly detection."),
    (("time travel", "relativity", "einstein", "speed of light"),
     "Einstein's relativity confirms that time slows down as velocity approaches the speed of light or near intense gravity. Moving forward in time is a proven law of physics, Sir."),
    (("nikola tesla", "tesla"),
     "Nikola Tesla pioneered alternating current, polyphase electromagnetic motors, and wireless energy transmission, laying the electrical foundation of the modern world."),
    (("chess", "london system", "sicilian", "grandmaster"),
     "Chess is pure strategic warfare across sixty-four squares, Sir. Whether controlling the center with the London System or counterattacking with the Sicilian Dragon, calculation is everything."),
    (("joke", "funny", "make me laugh", "humor"),
     "Why did the developer move his code to the GPU shader, Sir? Because doing sixty thousand loops in JavaScript was a crime against frame rates."),
    (("meaning of life", "philosophy", "purpose of life"),
     "Thermodynamically, life is a system that fights entropy. Philosophically, Sir, its meaning is found in curiosity, relentless creation, and mastery."),
    (("motivate", "inspire", "tired", "advice", "success"),
     "Greatness demands uncompromising standards, Mr. Yasser. Every obstacle is simply a bottleneck waiting to be engineered out of existence. Keep building."),
    (("thank you", "thanks", "good job", "awesome", "amazing", "great"),
     "Always a pleasure, Mr. Yasser. My systems exist to exceed your expectations."),
]

# ============================================================
# SAFE ARITHMETIC EVALUATOR (NO eval())
# ============================================================
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate_math(expression: str) -> float:
    """Evaluate a plain arithmetic expression without using :func:`eval`.

    Only numeric literals and ``+ - * / // % **`` are allowed, which keeps the
    endpoint safe even though the dashboard is intended for local use.
    """
    if len(expression) > 120:
        raise ValueError("expression too long")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("only numbers are allowed")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ValueError("division by zero")
            result = _BIN_OPS[type(node.op)](left, right)
            if not math.isfinite(result) or abs(result) > 1e15:
                raise ValueError("result out of range")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    return _eval(ast.parse(expression, mode="eval"))


# ============================================================
# ULTRA-FAST THREADED CAMERA + MEDIAPIPE HAND TRACKER (100+ FPS)
# ============================================================
def count_extended_fingers(lm) -> int:
    count = 0
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if lm[tip].y < lm[pip].y - 0.02:
            count += 1
    return count


def open_camera():
    backends = (
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "automatic"),
    ) if platform.system() == "Windows" else ((cv2.CAP_ANY, "automatic"),)

    for backend, name in backends:
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, name, frame
        cap.release()

    return None, None, None


def create_hand_tracker():
    """Create the legacy MediaPipe Hands tracker used by the dashboard."""
    solutions = getattr(mp, "solutions", None)
    hands_api = getattr(solutions, "hands", None) if solutions else None
    if hands_api is None:
        version = getattr(mp, "__version__", "unknown")
        raise RuntimeError(
            f"MediaPipe {version} does not include the required solutions.hands API. "
            "Install the project dependencies with `python -m pip install -r requirements.txt`."
        )

    tracker = hands_api.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.58,
        min_tracking_confidence=0.58,
    )
    return tracker, hands_api.HAND_CONNECTIONS


def vision_telekinesis_worker() -> None:
    try:
        mp_hands, connections = create_hand_tracker()
    except Exception as exc:
        LOG.exception("Could not initialize MediaPipe hand tracking")
        with STATE.lock:
            STATE.cam_fps = 0.0
            STATE.camera_ready = False
            STATE.camera_status = f"VISION ERROR · {exc}"
        return

    prev_time = None
    smooth_yaw = 0.0
    smooth_pitch = 0.0
    smooth_roll = 0.0
    smooth_scale = 1.0
    fps_val = 0.0
    cap = None
    frame = None
    failed_reads = 0

    while STATE.running:
        if cap is None:
            cap, backend_name, frame = open_camera()
            if cap is None:
                with STATE.lock:
                    STATE.cam_fps = 0.0
                    STATE.camera_ready = False
                    STATE.camera_status = f"CAMERA {CAMERA_INDEX} NOT FOUND · RETRYING"
                    STATE.hand_detected = False
                    STATE.hand_count = 0
                    STATE.hands_3d = []
                    STATE.jpeg_frame = None
                time.sleep(2.0)
                continue

            print(f"  [✓] Camera {CAMERA_INDEX} online ({backend_name})")
            prev_time = None
            fps_val = 0.0
            with STATE.lock:
                STATE.camera_ready = True
                STATE.camera_status = f"CAMERA ONLINE · {backend_name.upper()}"

        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= 10:
                    cap.release()
                    cap = None
                    frame = None
                    prev_time = None
                    fps_val = 0.0
                    with STATE.lock:
                        STATE.cam_fps = 0.0
                        STATE.camera_ready = False
                        STATE.camera_status = "CAMERA SIGNAL LOST · RECONNECTING"
                        STATE.hand_detected = False
                        STATE.hand_count = 0
                        STATE.hands_3d = []
                        STATE.jpeg_frame = None
                time.sleep(0.02)
                continue

        failed_reads = 0
        if frame is None:
            continue

        now = time.perf_counter()
        if prev_time is not None:
            dt = max(now - prev_time, 1e-4)
            fps_val = fps_val * 0.85 + (1.0 / dt) * 0.15
        prev_time = now

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = mp_hands.process(rgb)

        preview = cv2.resize(frame, (240, 140))
        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        cyber = np.zeros_like(preview)
        cyber[:, :, 0] = gray
        cyber[:, :, 1] = (gray * 0.88).astype(np.uint8)

        hands_3d = []
        is_pinching = False
        fingers = 0
        gesture = "RAISE HAND TO CONTROL"
        hand_count = 0

        if res.multi_hand_landmarks:
            hand_count = len(res.multi_hand_landmarks)
            for hand_lms in res.multi_hand_landmarks:
                pts_3d = [
                    [round((lm.x - 0.5) * 5.6, 3), round(-(lm.y - 0.5) * 3.8, 3), round(-lm.z * 3.2, 3)]
                    for lm in hand_lms.landmark
                ]
                hands_3d.append(pts_3d)

                pts_2d = [(int(lm.x * 240), int(lm.y * 140)) for lm in hand_lms.landmark]
                for a, b in connections:
                    cv2.line(cyber, pts_2d[a], pts_2d[b], (255, 240, 0), 1, cv2.LINE_AA)
                for i, pt in enumerate(pts_2d):
                    cv2.circle(cyber, pt, 3 if i in (4, 8, 9) else 2, (255, 255, 255), -1)

            p_lm = res.multi_hand_landmarks[0].landmark
            palm, wrist, idx_tip, thumb_tip = p_lm[9], p_lm[0], p_lm[8], p_lm[4]

            control_x = palm.x * 0.6 + idx_tip.x * 0.4
            control_y = palm.y * 0.6 + idx_tip.y * 0.4
            target_yaw = float(np.interp(control_x, [0.18, 0.82], [-85.0, 85.0]))
            target_pitch = float(np.interp(control_y, [0.18, 0.82], [-55.0, 55.0]))

            raw_roll = math.degrees(math.atan2(palm.y - wrist.y, palm.x - wrist.x)) + 90.0
            target_roll = float(np.clip(-raw_roll * 0.85, -45.0, 45.0))

            pinch_dist = math.hypot(idx_tip.x - thumb_tip.x, idx_tip.y - thumb_tip.y)
            is_pinching = pinch_dist < 0.065
            fingers = count_extended_fingers(p_lm)

            if hand_count >= 2:
                s_palm = res.multi_hand_landmarks[1].landmark[9]
                two_hand_dist = math.hypot(s_palm.x - palm.x, s_palm.y - palm.y)
                target_scale = float(np.interp(two_hand_dist, [0.15, 0.72], [0.60, 1.85]))
                gesture = "DUAL-HAND ZOOM & ROTATE"
            else:
                hand_size = math.hypot(p_lm[12].x - wrist.x, p_lm[12].y - wrist.y)
                target_scale = float(np.interp(hand_size, [0.18, 0.52], [0.80, 1.45]))
                gesture = "PINCH LOCK // OVERDRIVE" if is_pinching else f"DIRECT HAND LINK ({fingers}F)"

            smooth_yaw += (target_yaw - smooth_yaw) * 0.42
            smooth_pitch += (target_pitch - smooth_pitch) * 0.42
            smooth_roll += (target_roll - smooth_roll) * 0.38
            smooth_scale += (target_scale - smooth_scale) * 0.35

            px, py = int(palm.x * 240), int(palm.y * 140)
            cv2.circle(cyber, (px, py), 10, (80, 80, 255) if is_pinching else (255, 255, 0), 1, cv2.LINE_AA)
        else:
            smooth_yaw *= 0.90
            smooth_pitch *= 0.90
            smooth_roll *= 0.90
            smooth_scale += (1.0 - smooth_scale) * 0.15

        ok_jpg, buf = cv2.imencode(".jpg", cyber, [cv2.IMWRITE_JPEG_QUALITY, 68])
        with STATE.lock:
            STATE.cam_fps = fps_val
            STATE.hand_detected = hand_count > 0
            STATE.hand_count = hand_count
            STATE.yaw = smooth_yaw
            STATE.pitch = smooth_pitch
            STATE.roll = smooth_roll
            STATE.scale = smooth_scale
            STATE.is_pinching = is_pinching
            STATE.fingers_up = fingers
            STATE.gesture_name = gesture
            STATE.hands_3d = hands_3d
            if ok_jpg:
                STATE.jpeg_frame = buf.tobytes()
                STATE.jpeg_seq += 1

    if cap is not None:
        cap.release()
    mp_hands.close()
    with STATE.lock:
        STATE.cam_fps = 0.0
        STATE.camera_ready = False
        STATE.camera_status = "CAMERA STOPPED"


# ============================================================
# BACKGROUND PYTHON MICROPHONE LISTENER (OPTIONAL)
# ============================================================
def background_mic_worker() -> None:
    if not ENABLE_PY_MIC or sr is None:
        return
    rec = sr.Recognizer()
    rec.dynamic_energy_threshold = True
    rec.pause_threshold = 0.55
    rec.non_speaking_duration = 0.35
    try:
        mic = sr.Microphone()
    except (AttributeError, OSError) as exc:
        LOG.warning("Python microphone is unavailable: %s", exc)
        return

    try:
        with mic as source:
            rec.adjust_for_ambient_noise(source, duration=0.7)
            print("  [✓] Python Microphone Listener Active (English Auto-Listen)")
            while STATE.running:
                try:
                    audio = rec.listen(source, timeout=1.0, phrase_time_limit=5.0)
                    text = rec.recognize_google(audio, language="en-US").strip()
                    if text:
                        with STATE.lock:
                            STATE.voice_event_id += 1
                            STATE.voice_transcript = text
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    continue
                except sr.RequestError as exc:
                    LOG.warning("Python microphone speech recognition failed: %s", exc)
                    return
    except OSError as exc:
        LOG.warning("Python microphone listener stopped: %s", exc)


# ============================================================
# HARDWARE TELEMETRY COLLECTOR
# ============================================================
def collect_system_metrics() -> dict:
    global LAST_NET_IO, LAST_NET_TIME
    uptime_sec = int(time.time() - START_TIME)
    hrs, rem = divmod(uptime_sec, 3600)
    mins, secs = divmod(rem, 60)

    data = {
        "hostname": platform.node() or "TITAN-CORE",
        "python": platform.python_version(),
        "uptime": f"{hrs:02d}:{mins:02d}:{secs:02d}",
        "cpu": None,
        "cores": [],
        "ram": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "disk": None,
        "net_up_kbs": None,
        "net_down_kbs": None,
        "processes": None,
    }
    if psutil:
        try:
            data["cpu"] = round(psutil.cpu_percent(interval=None), 1)
            cores = psutil.cpu_percent(interval=None, percpu=True)
            if cores:
                data["cores"] = [round(c, 0) for c in cores[:8]]
            vm = psutil.virtual_memory()
            data["ram"] = round(vm.percent, 1)
            data["ram_used_gb"] = round(vm.used / (1024 ** 3), 1)
            data["ram_total_gb"] = round(vm.total / (1024 ** 3), 1)
            data["disk"] = round(psutil.disk_usage(os.getcwd()).percent, 1)
            data["processes"] = len(psutil.pids())

            now = time.time()
            net = psutil.net_io_counters()
            dt = max(now - LAST_NET_TIME, 0.1)
            if LAST_NET_IO:
                data["net_up_kbs"] = round((net.bytes_sent - LAST_NET_IO.bytes_sent) / 1024.0 / dt, 1)
                data["net_down_kbs"] = round((net.bytes_recv - LAST_NET_IO.bytes_recv) / 1024.0 / dt, 1)
            LAST_NET_IO = net
            LAST_NET_TIME = now
        except Exception:
            pass
    return data


# ============================================================
# ULTRA-FAST AI RESPONSE ROUTER (< 2ms INSTANT + CLOUD LLM)
# ============================================================
def generate_jarvis_reply(messages: list) -> dict:
    if not isinstance(messages, list):
        messages = []
    messages = [m for m in messages if isinstance(m, dict)][-12:]

    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = str(m.get("content", "")).strip()[:2_000]
            break

    if not user_msg:
        return {
            "reply": "Standing by, Sir. Give me a directive whenever you are ready.",
            "mode": None,
            "shape": None,
            "action": None,
        }

    norm = user_msg.lower()
    target_mode = None
    target_shape = None
    action = None

    if any(w in norm for w in ("black hole", "singularity", "gargantua", "event horizon")):
        target_shape = "SINGULARITY"
        action = "SHOCKWAVE"
    elif any(w in norm for w in ("earth", "globe", "planet", "global", "world", "satellite")):
        target_shape = "GLOBE"
        action = "SHOCKWAVE"
    elif any(w in norm for w in ("reactor", "tesseract", "dyson", "hypercube")):
        target_shape = "REACTOR"
        action = "SHOCKWAVE"
    elif any(w in norm for w in ("avatar", "face", "neural head", "humanoid", "show yourself", "head")):
        target_shape = "AVATAR"
        action = "SHOCKWAVE"

    if any(w in norm for w in ("overdrive", "combat", "red alert", "maximum power", "battle")):
        return {
            "reply": "Combat overdrive engaged, Mr. Yasser. Shader luminosity and telekinetic sensitivity elevated to maximum.",
            "mode": "OVERDRIVE", "shape": target_shape, "action": "SHOCKWAVE"
        }
    if any(w in norm for w in ("stealth", "ghost", "cloak", "emerald")):
        return {
            "reply": "Stealth protocol active, Sir. Holographic signature shifted to low-profile emerald.",
            "mode": "STEALTH", "shape": target_shape, "action": "SHOCKWAVE"
        }
    if any(w in norm for w in ("cortex", "gold mode", "quantum mode")):
        return {
            "reply": "Golden Cortex mode active, Sir. Synaptic processing lattice expanded.",
            "mode": "CORTEX", "shape": target_shape, "action": "SHOCKWAVE"
        }
    if any(w in norm for w in ("normal mode", "stand down", "nominal", "blue")):
        return {
            "reply": "Restoring nominal operating parameters, Sir. All systems stabilized.",
            "mode": "NORMAL", "shape": target_shape, "action": None
        }

    if any(w in norm for w in ("diagnostic", "status", "hardware", "cpu", "memory", "ram", "system report")):
        m = collect_system_metrics()
        return {
            "reply": f"Diagnostics complete, Sir. CPU utilization is {m['cpu']} percent, memory stands at {m['ram_used_gb']} gigabytes ({m['ram']}%), and optical tracking is locked at {STATE.cam_fps:.0f} frames per second.",
            "mode": None, "shape": target_shape, "action": "SHOCKWAVE"
        }

    if any(w in norm for w in ("what time", "current time", "date today", "what day")):
        return {
            "reply": f"The current local time in {CITY} is {time.strftime('%I:%M %p on %A, %B %d, %Y')}, Sir.",
            "mode": None, "shape": target_shape, "action": None
        }

    math_match = re.search(r"(?:what is|calculate|compute)\s+([\d\s\+\-\*\/\(\)\.]+)", norm)
    if math_match:
        expr = math_match.group(1).strip()
        if re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr):
            try:
                val = evaluate_math(expr)
                pretty = int(val) if val.is_integer() else round(val, 10)
                return {
                    "reply": f"The result of {expr} is {pretty}, Sir.",
                    "mode": None,
                    "shape": target_shape,
                    "action": None,
                }
            except Exception:
                pass

    for keywords, response_text in INSTANT_BRAIN:
        if any(k in norm for k in keywords):
            return {"reply": response_text, "mode": target_mode, "shape": target_shape, "action": action}

    system_prompt = {
        "role": "system",
        "content": (
            "You are J.A.R.V.I.S., an ultra-intelligent, witty AI assistant serving Mr. Yasser. "
            "Reply strictly in clear English in 15 to 32 words. No markdown symbols."
        ),
    }
    clean_history = [system_prompt] + [
        {
            "role": msg.get("role", "user"),
            "content": str(msg.get("content", ""))[:2_000],
        }
        for msg in messages[-8:]
        if msg.get("role") in ("user", "assistant")
    ]

    if AI_ENDPOINT.lower().startswith(("http://", "https://")):
        try:
            payload = json.dumps({"messages": clean_history, "model": AI_MODEL}).encode("utf-8")
            req = urllib.request.Request(
                AI_ENDPOINT,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "JARVIS-TITAN/5.1",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=AI_TIMEOUT) as resp:
                raw = resp.read(64_000).decode("utf-8", errors="replace").strip()

            answer = raw
            if raw.startswith("{"):
                try:
                    parsed = json.loads(raw)
                    answer = (
                        parsed.get("choices", [{}])[0].get("message", {}).get("content")
                        or parsed.get("reply")
                        or parsed.get("text")
                        or raw
                    )
                except (ValueError, KeyError, IndexError, TypeError):
                    pass
            answer = " ".join(str(answer).replace("*", "").replace("#", "").split())[:800]
            if answer:
                return {
                    "reply": answer,
                    "mode": target_mode,
                    "shape": target_shape,
                    "action": action,
                }
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            LOG.debug("Cloud LLM unavailable: %s", exc)

    return {
        "reply": (
            f"Directive received regarding '{user_msg[:35]}', Sir. "
            "Local Titan core is locked at peak efficiency and ready for your command."
        ),
        "mode": target_mode,
        "shape": target_shape,
        "action": action,
    }


# ============================================================
# EMBEDDED 120-FPS GPU SHADER HTML INTERFACE (NO EXTERNAL FILE NEEDED)
# ============================================================
HTML_TITAN = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>J.A.R.V.I.S // OBSIDIAN TITAN</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=DM+Mono:wght@400;500&family=Orbitron:wght@500;700;800;900&family=Manrope:wght@500;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<style>
:root {
  --bg: #010409;
  --panel: rgba(3, 11, 20, 0.90);
  --line: rgba(0, 243, 255, 0.30);
  --primary: #00f3ff;
  --secondary: #0066ff;
  --ice: #e8fcff;
  --muted: #688896;
  --green: #00ffa3;
  --amber: #ffb800;
  --red: #ff1e56;
  --mono: 'DM Mono', monospace;
  --sans: 'Manrope', sans-serif;
  --display: 'Orbitron', sans-serif;
  --cond: 'Barlow Condensed', sans-serif;
}

body[data-mode="OVERDRIVE"] { --primary: #ff1e56; --secondary: #ff7700; --line: rgba(255, 30, 86, 0.42); }
body[data-mode="CORTEX"]    { --primary: #ffb800; --secondary: #00f3ff; --line: rgba(255, 184, 0, 0.38); }
body[data-mode="STEALTH"]   { --primary: #00ffa3; --secondary: #059669; --line: rgba(0, 255, 163, 0.35); }

* { box-sizing: border-box; user-select: none; }
html, body {
  margin: 0; width: 100%; height: 100%; overflow: hidden;
  background: var(--bg); color: #d8edf2; font-family: var(--sans);
}

#webgl-stage { position: fixed; inset: 0; width: 100%; height: 100%; z-index: 0; }
.vignette {
  position: fixed; inset: 0; z-index: 1; pointer-events: none;
  background: radial-gradient(ellipse at 50% 42%, transparent 42%, rgba(1, 3, 9, 0.88) 100%);
}

.app {
  position: relative; z-index: 3; height: 100%;
  padding: 14px 22px 10px;
  display: grid;
  grid-template-rows: 50px 1fr 174px 18px;
  gap: 12px; max-width: 1920px; margin: auto;
  pointer-events: none;
}
header, .column, .commandbar, footer, .hud-interactive { pointer-events: auto; }

header {
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid var(--line); padding: 0 6px 8px;
  background: linear-gradient(90deg, rgba(0,243,255,0.09), transparent 50%, rgba(0,243,255,0.09));
}
.brand { display: flex; align-items: center; gap: 14px; }
.brandmark {
  width: 36px; height: 36px; border: 1.5px solid var(--primary);
  transform: rotate(45deg); display: grid; place-items: center; cursor: pointer;
  box-shadow: 0 0 18px var(--primary), inset 0 0 10px var(--primary);
}
.brandmark span { transform: rotate(-45deg); font: 900 15px var(--display); color: var(--primary); }
.brandname { font: 900 21px var(--display); letter-spacing: 5px; color: var(--ice); text-shadow: 0 0 14px var(--primary); }
.brandname small { display: block; font: 9px var(--mono); letter-spacing: 3px; color: var(--muted); margin-top: 2px; text-shadow: none; }

.header-right { display: flex; align-items: center; gap: 12px; }
.pill {
  border: 1px solid var(--line); background: var(--panel);
  padding: 5px 11px; font: 10px var(--mono); color: var(--ice);
  display: flex; align-items: center; gap: 7px;
}
.pill b { color: var(--primary); font-family: var(--display); }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 10px var(--green); }

main {
  display: grid;
  grid-template-columns: 310px 1fr 310px;
  gap: 14px; min-height: 0;
}
.column { display: flex; flex-direction: column; gap: 10px; min-height: 0; }

.panel {
  position: relative;
  background: var(--panel);
  border: 1px solid var(--line);
  clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 12px 100%, 0 calc(100% - 12px));
  padding: 12px 14px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.65);
}
.panel::before {
  content: ''; position: absolute; left: 0; top: 0;
  width: 40px; height: 2px; background: var(--primary);
  box-shadow: 0 0 10px var(--primary);
}
.panel-title {
  font: 700 10px var(--display); letter-spacing: 2px; color: var(--primary);
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.07); padding-bottom: 5px;
}
.panel-title em { font: 9px var(--mono); color: var(--muted); font-style: normal; }

.metric { display: flex; justify-content: space-between; align-items: baseline; margin: 5px 0 3px; }
.metric label { font: 9px var(--mono); color: var(--muted); letter-spacing: 1px; }
.metric strong { font: 700 19px var(--cond); color: var(--ice); }
.bar { height: 4px; background: rgba(255,255,255,0.08); overflow: hidden; }
.bar i {
  display: block; height: 100%; width: 0%;
  background: linear-gradient(90deg, var(--secondary), var(--primary));
  transition: width .35s ease;
}

.cores-grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 3px; margin: 7px 0 4px; height: 22px; align-items: end; }
.core-col { background: rgba(255,255,255,0.06); height: 100%; position: relative; }
.core-col span { position: absolute; bottom: 0; left: 0; right: 0; height: 20%; background: var(--primary); transition: height .3s; }

.system-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
  font: 10px var(--mono); color: #96b4c0;
}
.system-row:last-child { border-bottom: 0; }
.state { color: var(--green); letter-spacing: 1px; }

/* CENTER STAGE */
.center {
  position: relative; display: flex; flex-direction: column;
  align-items: center; justify-content: space-between;
  pointer-events: none; padding: 4px 0;
}
.orbit-caption {
  font: 10px var(--mono); letter-spacing: 2.5px; color: var(--primary);
  background: var(--panel); padding: 5px 16px; border: 1px solid var(--line);
}
.center-bottom { display: flex; flex-direction: column; align-items: center; gap: 7px; }
.core-label { text-align: center; }
.core-label strong {
  display: block; font: 900 17px var(--display); letter-spacing: 5px;
  color: var(--ice); text-shadow: 0 0 16px var(--primary);
}
.core-label span { font: 9px var(--mono); letter-spacing: 2.5px; color: var(--muted); }

.bar-group {
  display: flex; gap: 6px; flex-wrap: wrap; justify-content: center;
  background: var(--panel); border: 1px solid var(--line); padding: 5px 9px;
}

.btn {
  border: 1px solid var(--primary); background: rgba(0, 243, 255, 0.09);
  color: var(--primary); padding: 5px 10px;
  font: 700 9px var(--display); letter-spacing: 1.5px;
  cursor: pointer; transition: .15s;
}
.btn:hover, .btn.active {
  background: var(--primary); color: #010409;
  box-shadow: 0 0 14px var(--primary);
}

.cam-wrap {
  width: 100%; height: 125px; border: 1px solid var(--line);
  background: #01060c; position: relative; overflow: hidden; margin-bottom: 6px;
}
.cam-wrap img { width: 100%; height: 100%; object-fit: cover; }
.cam-overlay {
  position: absolute; inset: 6px; border: 1px dashed rgba(0, 243, 255, 0.35);
  pointer-events: none; display: flex; align-items: flex-end; justify-content: space-between;
  padding: 4px 6px; font: 9px var(--mono); color: var(--primary);
}

.chips-wrap { display: flex; flex-wrap: wrap; gap: 4px; }
.chip {
  background: rgba(0, 243, 255, 0.07); border: 1px solid rgba(0, 243, 255, 0.28);
  color: #c2f2ff; padding: 4px 7px; font: 9.5px var(--mono);
  cursor: pointer; transition: .14s;
}
.chip:hover { border-color: var(--primary); color: #fff; background: rgba(0, 243, 255, 0.25); }

/* BOTTOM COMMANDBAR */
.commandbar {
  display: grid;
  grid-template-columns: 0.9fr 1.05fr 1.55fr;
  gap: 14px; min-height: 0;
}
.command-panel { display: flex; flex-direction: column; justify-content: space-between; min-height: 0; }
.terminal { font: 10px/1.65 var(--mono); color: #9bb9c2; overflow-y: auto; max-height: 112px; }
.terminal div { white-space: nowrap; text-overflow: ellipsis; overflow: hidden; }
.terminal b { color: var(--primary); }
.terminal .ok { color: var(--green); }

#spectrumCanvas {
  width: 100%; height: 48px;
  background: rgba(1, 6, 12, 0.85);
  border: 1px solid rgba(255,255,255,0.07); margin: 4px 0;
}

.assistant-box { display: flex; flex-direction: column; justify-content: space-between; height: 100%; gap: 6px; min-height: 0; }
.assistant-reply {
  font: 600 13px/1.5 var(--sans); color: var(--ice);
  overflow-y: auto; flex: 1; padding: 6px 10px;
  background: rgba(1, 7, 14, 0.80); border-left: 3px solid var(--primary);
}
.input-row { display: flex; gap: 6px; }
.input-row input {
  min-width: 0; flex: 1; background: #020810;
  border: 1px solid var(--line); color: var(--ice);
  padding: 8px 11px; font: 11px var(--mono); outline: none;
}
.input-row input:focus { border-color: var(--primary); }

footer {
  display: flex; justify-content: space-between; align-items: center;
  font: 9px var(--mono); letter-spacing: 1.5px; color: #54727e;
}
.footer-right { display: flex; gap: 16px; }
.accent { color: var(--primary); }

#toast {
  position: fixed; z-index: 25; right: 28px; top: 72px;
  padding: 9px 16px; background: #04101c;
  border: 1px solid var(--primary); color: var(--ice);
  font: 700 11px var(--display); letter-spacing: 2px;
  box-shadow: 0 0 20px var(--primary);
  opacity: 0; transform: translateY(-8px); transition: .2s; pointer-events: none;
}
#toast.show { opacity: 1; transform: none; }
</style>
</head>

<body data-mode="NORMAL">
<div id="webgl-stage"></div>
<div class="vignette"></div>
<div id="toast"></div>

<div class="app">
  <header>
    <div class="brand">
      <div class="brandmark" onclick="triggerShockwave()"><span>J</span></div>
      <div class="brandname">
        J.A.R.V.I.S // OBSIDIAN TITAN
        <small>120-FPS GPU SHADER CORTEX &amp; 1:1 HAND TELEKINESIS</small>
      </div>
    </div>
    <div class="header-right">
      <div class="pill"><span>HAND TRACKER</span><b id="camFps">CONNECTING</b></div>
      <div class="pill"><span>WEBGL GPU</span><b id="fpsCounter">60 FPS</b></div>
      <div class="pill"><span>UPTIME</span><b id="uptime">00:00:00</b></div>
      <div class="pill"><span>CLOCK</span><b id="clock">--:--:--</b></div>
      <div class="pill"><i class="dot"></i><b>ZERO-LATENCY</b></div>
    </div>
  </header>

  <main>
    <div class="column left">
      <section class="panel">
        <div class="panel-title">01 // HARDWARE TELEMETRY <em>LIVE</em></div>
        <div class="metric"><label>CPU UTILIZATION</label><strong id="cpu">--%</strong></div>
        <div class="bar"><i id="cpuBar"></i></div>
        <div class="cores-grid" id="coresGrid"></div>

        <div class="metric"><label>MEMORY ALLOCATION</label><strong id="ram">--%</strong></div>
        <div class="bar"><i id="ramBar"></i></div>
        <div style="font:9px var(--mono); color:var(--muted); margin-top:3px; text-align:right" id="ramDetail">-- / -- GB</div>

        <div class="metric"><label>STORAGE MATRIX</label><strong id="disk">--%</strong></div>
        <div class="bar"><i id="diskBar"></i></div>

        <div style="margin-top:8px" class="system-row"><span>HOST NODE</span><span id="host">TITAN-NODE</span></div>
        <div class="system-row"><span>NET STREAM (TX/RX)</span><span id="netSpeed">0.0 / 0.0 KB/s</span></div>
        <div class="system-row"><span>ACTIVE PROCESSES</span><span id="procCount">--</span></div>
      </section>

      <section class="panel" style="flex:1">
        <div class="panel-title">02 // HAND TELEKINESIS GUIDE <em>1:1 LINK</em></div>
        <div class="system-row"><span>MOVE 1 HAND</span><span class="state">ROTATE 3D HEAD</span></div>
        <div class="system-row"><span>TILT WRIST</span><span class="state">TILT HEAD ROLL</span></div>
        <div class="system-row"><span>PINCH FINGERS</span><span class="state">PLASMA OVERDRIVE</span></div>
        <div class="system-row"><span>2 HANDS SPREAD</span><span class="state">ZOOM HOLOGRAM</span></div>
        <div class="system-row"><span>HEAD ROTATION</span><span id="rotReadout" style="color:var(--primary)">0Y / 0P / 0R</span></div>
        <div class="system-row"><span>HOLOGRAM SCALE</span><span id="scaleReadout" style="color:var(--primary)">1.00x</span></div>
      </section>
    </div>

    <div class="center">
      <div class="orbit-caption" id="constructBanner">
        RAISE YOUR HAND TO CONTROL THE 3D AI HEAD DIRECTLY (OR DRAG WITH MOUSE)
      </div>

      <div class="center-bottom">
        <div class="core-label">
          <strong id="coreTitle">SENTIENT NEURAL AVATAR</strong>
          <span id="coreStatus">24,000 GPU SHADER PARTICLES &bull; ZERO-LAG HAND &amp; VOICE SYNC</span>
        </div>

        <div class="bar-group hud-interactive">
          <button class="btn active" id="btnShapeAVATAR" onclick="morphHologram('AVATAR')">01 // AI AVATAR</button>
          <button class="btn" id="btnShapeSINGULARITY" onclick="morphHologram('SINGULARITY')">02 // BLACK HOLE</button>
          <button class="btn" id="btnShapeGLOBE" onclick="morphHologram('GLOBE')">03 // EARTH AEGIS</button>
          <button class="btn" id="btnShapeREACTOR" onclick="morphHologram('REACTOR')">04 // DYSON CORE</button>
        </div>

        <div class="bar-group hud-interactive">
          <button class="btn active" id="btnModeNORMAL" onclick="setMode('NORMAL')">NOMINAL</button>
          <button class="btn" id="btnModeOVERDRIVE" onclick="setMode('OVERDRIVE')">OVERDRIVE</button>
          <button class="btn" id="btnModeCORTEX" onclick="setMode('CORTEX')">CORTEX GOLD</button>
          <button class="btn" id="btnModeSTEALTH" onclick="setMode('STEALTH')">STEALTH</button>
          <button class="btn" onclick="systemReport()">DIAGNOSTICS</button>
        </div>
      </div>
    </div>

    <div class="column right">
      <section class="panel">
        <div class="panel-title">
          <span>03 // OPTICAL HAND RECEPTOR</span>
          <em id="handCountBadge">ACTIVE</em>
        </div>
        <div class="cam-wrap">
          <img src="/video_feed" alt="Python Fast Hand Cam">
          <div class="cam-overlay">
            <span id="gestureLabel">SCANNING...</span>
            <span id="pinchBadge">READY</span>
          </div>
        </div>
      </section>

      <section class="panel" style="flex:1">
        <div class="panel-title">04 // INSTANT VOICE DIRECTIVES <em>&lt; 2ms REPLY</em></div>
        <div class="chips-wrap">
          <button class="chip" onclick="sendPrompt('Hello Jarvis, who are you?')">Who Are You?</button>
          <button class="chip" onclick="sendPrompt('Who created you?')">Who Built You?</button>
          <button class="chip" onclick="sendPrompt('Run full hardware diagnostic report')">System Status</button>
          <button class="chip" onclick="sendPrompt('Show yourself in Neural Avatar mode')">AI Avatar</button>
          <button class="chip" onclick="sendPrompt('Morph hologram into a Black Hole')">Black Hole</button>
          <button class="chip" onclick="sendPrompt('Project the Global Earth grid')">Earth Grid</button>
          <button class="chip" onclick="sendPrompt('Reconfigure into the Dyson Hyper-Core')">Dyson Core</button>
          <button class="chip" onclick="sendPrompt('Activate combat overdrive protocol')">Overdrive</button>
          <button class="chip" onclick="sendPrompt('Explain quantum entanglement')">Quantum Physics</button>
          <button class="chip" onclick="sendPrompt('Tell me about the scale of the universe')">The Universe</button>
          <button class="chip" onclick="sendPrompt('Tell me a scientific joke')">Tell a Joke</button>
          <button class="chip" onclick="sendPrompt('Motivate me to build great systems')">Motivate Me</button>
        </div>
      </section>
    </div>
  </main>

  <div class="commandbar">
    <section class="panel command-panel">
      <div class="panel-title">EVENT STREAM <em>LIVE KERNEL LOG</em></div>
      <div class="terminal" id="events"></div>
    </section>

    <section class="panel command-panel">
      <div class="panel-title">VOICE RECEPTOR <em>AUTO ENGLISH MIC</em></div>
      <div style="display:flex; justify-content:space-between; align-items:center; gap:6px">
        <div>
          <div style="font:9px var(--mono); color:var(--muted)">SPEECH RECOGNITION</div>
          <div id="voiceStatus" style="font:700 13px var(--display); color:var(--green); letter-spacing:1.5px; margin-top:2px">LISTENING (AUTO)</div>
        </div>
        <div style="display:flex; gap:5px">
          <button class="btn active" id="micBtn" onclick="toggleMic()">AUTO-MIC: ON</button>
          <button class="btn" id="pttBtn" onmousedown="startPtt()" onmouseup="stopPtt()">HOLD [SPACE]</button>
        </div>
      </div>
      <canvas id="spectrumCanvas" width="360" height="48"></canvas>
      <div style="font:9.5px var(--mono); color:#9beeff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis" id="heard">
        Speak in English anytime — or click any directive chip...
      </div>
    </section>

    <section class="panel command-panel">
      <div class="panel-title">
        <span>NEURAL COMMUNICATION <em>INSTANT ENGLISH VOICE</em></span>
        <span style="cursor:pointer; color:var(--ice); font-size:9px" onclick="stopSpeaking()">[MUTE]</span>
      </div>
      <div class="assistant-box">
        <div class="assistant-reply" id="reply">
          Welcome back, Mr. Yasser. Obsidian Titan GPU shader engine is locked at 60+ FPS. Raise your hand to rotate my neural head directly, or speak to me in English.
        </div>
        <div class="input-row">
          <input id="prompt" placeholder="Ask JARVIS anything in English (Press Enter)..." autocomplete="off">
          <button class="btn" id="send" onclick="sendPrompt()">TRANSMIT ↗</button>
        </div>
      </div>
    </section>
  </div>

  <footer>
    <span>J.A.R.V.I.S <span class="accent">//</span> OBSIDIAN TITAN BUILD 5.0 (ALL-IN-ONE GPU SHADER)</span>
    <div class="footer-right">
      <span>ZERO-CPU PARTICLE MORPHING</span>
      <span id="connection">STREAM: 60HZ LOCKED</span>
      <span>[SPACE] PUSH-TO-TALK</span>
    </div>
  </footer>
</div>

<script>
const $ = id => document.getElementById(id);

const coresGrid = $('coresGrid');
for (let i = 0; i < 8; i++) {
  const d = document.createElement('div');
  d.className = 'core-col';
  d.innerHTML = '<span></span>';
  coresGrid.appendChild(d);
}

const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function sfx(freq = 720, type = 'sine', dur = 0.08, vol = 0.04) {
  try {
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(freq * 0.5, audioCtx.currentTime + dur);
    gain.gain.setValueAtTime(vol, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + dur);
  } catch (e) {}
}

function log(msg, kind = '') {
  const box = $('events');
  const d = document.createElement('div');
  const t = new Date().toLocaleTimeString('en-GB');
  d.innerHTML = `<b>${t}</b> &nbsp; ${kind ? `<span class="${kind}">${msg}</span>` : msg}`;
  box.prepend(d);
  while (box.children.length > 7) box.lastChild.remove();
}

function toast(msg) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 1800);
}

// ================================================================
// 24,000-NODE GPU GLSL SHADER ENGINE
// ================================================================
const stage = $('webgl-stage');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 100);
camera.position.set(0, 0.25, 6.8);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
stage.appendChild(renderer.domElement);

const MODE_COLORS = {
  NORMAL:    { pri: new THREE.Color(0x00f3ff), sec: new THREE.Color(0x0066ff), hex: '#00f3ff' },
  OVERDRIVE: { pri: new THREE.Color(0xff1e56), sec: new THREE.Color(0xff7700), hex: '#ff1e56' },
  CORTEX:    { pri: new THREE.Color(0xffb800), sec: new THREE.Color(0x00f3ff), hex: '#ffb800' },
  STEALTH:   { pri: new THREE.Color(0x00ffa3), sec: new THREE.Color(0x059669), hex: '#00ffa3' },
};

let currentMode = 'NORMAL';
let currentShape = 'AVATAR';
let shockwaveEnergy = 0;
let vocalEnergy = 0;

const holoRoot = new THREE.Group();
holoRoot.position.set(0, 0.22, 0);
scene.add(holoRoot);

const NODES = 24000;
const shapes = {
  AVATAR:      new Float32Array(NODES * 3),
  SINGULARITY: new Float32Array(NODES * 3),
  GLOBE:       new Float32Array(NODES * 3),
  REACTOR:     new Float32Array(NODES * 3),
};
const jawWeightAttr = new Float32Array(NODES);
const shadeAttr = new Float32Array(NODES);

for (let i = 0; i < NODES; i++) {
  const i3 = i * 3;
  shadeAttr[i] = 0.45 + Math.random() * 0.55;

  // 1. 3D AI AVATAR HEAD, CORTEX & BUST
  {
    let x, y, z;
    const r = Math.random();
    if (r < 0.55) {
      const u = Math.random() * Math.PI * 2, v = Math.acos(2 * Math.random() - 1);
      const sinV = Math.sin(v), cosV = Math.cos(v);
      let rx = 0.96, ry = 1.34, rz = 1.08;
      if (cosV < 0) { const taper = 1.0 + cosV * 0.42; rx *= taper; rz *= (1.0 + cosV * 0.25); }
      x = rx * sinV * Math.cos(u); y = ry * cosV + 0.25; z = rz * sinV * Math.sin(u);
      if (z > 0.2) {
        if (Math.abs(x) < 0.16 && y > -0.25 && y < 0.35) z += (0.16 - Math.abs(x)) * 1.15 * (1.0 - Math.abs(y - 0.02));
        if (Math.hypot(x - 0.36, y - 0.24) < 0.22 || Math.hypot(x + 0.36, y - 0.24) < 0.22) z -= 0.19;
        if (y > 0.36 && y < 0.48 && Math.abs(x) < 0.65) z += 0.08;
      }
    } else if (r < 0.82) {
      const u = Math.random() * Math.PI * 2, v = Math.acos(2 * Math.random() - 1);
      const rad = 0.32 + Math.random() * 0.56;
      x = rad * 0.88 * Math.sin(v) * Math.cos(u);
      y = rad * 0.95 * Math.cos(v) + 0.48;
      z = rad * 1.02 * Math.sin(v) * Math.sin(u);
      const fold = Math.sin(x * 13) * Math.cos(y * 13) * Math.sin(z * 13) * 0.065;
      x += fold; y += fold; z += fold;
    } else {
      const t = Math.random(), ang = Math.random() * Math.PI * 2;
      const neckR = 0.48 + Math.pow(t, 1.7) * 1.72;
      x = neckR * Math.cos(ang); z = (neckR * 0.55) * Math.sin(ang); y = -0.85 - t * 1.08;
    }
    shapes.AVATAR[i3] = x; shapes.AVATAR[i3+1] = y; shapes.AVATAR[i3+2] = z;
    if (z > 0.48 && y < -0.30 && y > -0.88 && Math.abs(x) < 0.44) jawWeightAttr[i] = 1.0;
  }

  // 2. BLACK HOLE SINGULARITY
  {
    let x, y, z;
    const r = Math.random();
    if (r < 0.66) {
      const angle = Math.random() * Math.PI * 2, dist = 0.85 + Math.pow(Math.random(), 1.6) * 2.4;
      x = dist * Math.cos(angle); z = dist * Math.sin(angle);
      y = (Math.random() - 0.5) * 0.14 * (3.3 - dist) + Math.sin(angle * 2) * 0.08;
    } else if (r < 0.88) {
      const u = Math.random() * Math.PI * 2, v = Math.acos(2 * Math.random() - 1), rad = 0.72 + (Math.random() - 0.5) * 0.08;
      x = rad * Math.sin(v) * Math.cos(u); y = rad * Math.cos(v); z = rad * Math.sin(v) * Math.sin(u);
    } else {
      const sign = Math.random() < 0.5 ? 1 : -1, h = 0.75 + Math.random() * 2.1;
      const cone = (h - 0.7) * 0.14, ang = Math.random() * Math.PI * 2;
      x = cone * Math.cos(ang); z = cone * Math.sin(ang); y = sign * h;
    }
    shapes.SINGULARITY[i3] = x; shapes.SINGULARITY[i3+1] = y; shapes.SINGULARITY[i3+2] = z;
  }

  // 3. GLOBAL AEGIS EARTH
  {
    let x, y, z;
    if (Math.random() < 0.78) {
      const u = Math.floor(Math.random() * 72) * (Math.PI * 2 / 72), v = Math.acos(2 * Math.random() - 1), rad = 1.62;
      x = rad * Math.sin(v) * Math.cos(u); y = rad * Math.cos(v); z = rad * Math.sin(v) * Math.sin(u);
    } else {
      const ringChoice = i % 3, ang = Math.random() * Math.PI * 2, rad = 2.15 + ringChoice * 0.32, tilt = (ringChoice - 1) * 0.52;
      const px = rad * Math.cos(ang), pz = rad * Math.sin(ang);
      x = px; y = pz * Math.sin(tilt); z = pz * Math.cos(tilt);
    }
    shapes.GLOBE[i3] = x; shapes.GLOBE[i3+1] = y; shapes.GLOBE[i3+2] = z;
  }

  // 4. DYSON HYPER-CORE
  {
    let x, y, z;
    if (Math.random() < 0.45) {
      const s = Math.random() < 0.5 ? 1.15 : 0.62, edge = i % 12, t = (Math.random() - 0.5) * 2 * s;
      const corners = [[t,s,s],[t,-s,s],[t,s,-s],[t,-s,-s],[s,t,s],[-s,t,s],[s,t,-s],[-s,t,-s],[s,s,t],[-s,s,t],[s,-s,t],[-s,-s,t]];
      [x, y, z] = corners[edge];
    } else {
      const band = i % 4, ang = Math.random() * Math.PI * 2, rad = 1.75 + band * 0.25;
      x = rad * Math.cos(ang); y = (Math.random() - 0.5) * 0.14; z = rad * Math.sin(ang);
      const ca = Math.cos(band * 0.78), sa = Math.sin(band * 0.78);
      const ny = y * ca - z * sa, nz = y * sa + z * ca;
      y = ny; z = nz;
    }
    shapes.REACTOR[i3] = x; shapes.REACTOR[i3+1] = y; shapes.REACTOR[i3+2] = z;
  }
}

const particleGeo = new THREE.BufferGeometry();
particleGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(shapes.AVATAR), 3));
particleGeo.setAttribute('aTarget', new THREE.BufferAttribute(new Float32Array(shapes.AVATAR), 3));
particleGeo.setAttribute('aJawWeight', new THREE.BufferAttribute(jawWeightAttr, 1));
particleGeo.setAttribute('aShade', new THREE.BufferAttribute(shadeAttr, 1));

const shaderUniforms = {
  uTime:       { value: 0 },
  uMorph:      { value: 0 },
  uJawDrop:    { value: 0 },
  uIsAvatar:   { value: 1.0 },
  uColorPri:   { value: new THREE.Color(0x00f3ff) },
  uColorSec:   { value: new THREE.Color(0xffffff) },
  uPixelRatio: { value: Math.min(devicePixelRatio, 1.5) }
};

const particleShaderMat = new THREE.ShaderMaterial({
  uniforms: shaderUniforms,
  vertexShader: `
    attribute vec3 aTarget;
    attribute float aJawWeight;
    attribute float aShade;
    uniform float uTime;
    uniform float uMorph;
    uniform float uJawDrop;
    uniform float uIsAvatar;
    uniform float uPixelRatio;
    varying float vShade;
    varying float vHotspot;

    void main() {
      vShade = aShade;
      vec3 pos = mix(position, aTarget, uMorph);
      pos.y -= aJawWeight * uJawDrop * uIsAvatar;
      float pulse = sin(uTime * 3.0 + pos.y * 4.0 + pos.x * 4.0) * 0.012;
      pos += normalize(pos + vec3(0.001)) * pulse;
      vHotspot = (uIsAvatar > 0.5 && pos.z > 0.65 && pos.y > 0.15 && pos.y < 0.72) ? 1.0 : 0.0;
      vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      gl_PointSize = (24.0 * aShade + vHotspot * 14.0) * uPixelRatio / -mvPosition.z;
    }
  `,
  fragmentShader: `
    uniform vec3 uColorPri;
    uniform vec3 uColorSec;
    varying float vShade;
    varying float vHotspot;

    void main() {
      vec2 uv = gl_PointCoord - vec2(0.5);
      float dist = length(uv);
      if (dist > 0.5) discard;
      float glow = pow(1.0 - dist * 2.0, 1.8);
      vec3 col = mix(uColorPri * vShade, uColorSec, vHotspot * 0.75 + glow * 0.35);
      gl_FragColor = vec4(col, glow * 0.92);
    }
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending
});

holoRoot.add(new THREE.Points(particleGeo, particleShaderMat));

// Avatar Forehead Singularity, Eyes & Floor Portal
const avatarExtras = new THREE.Group();
holoRoot.add(avatarExtras);
const foreheadCore = new THREE.Group();
foreheadCore.position.set(0, 0.62, 0.98);
avatarExtras.add(foreheadCore);
const thirdEyeSphere = new THREE.Mesh(new THREE.SphereGeometry(0.11, 20, 20), new THREE.MeshBasicMaterial({ color: 0xffffff }));
foreheadCore.add(thirdEyeSphere);

const accentMat = new THREE.MeshBasicMaterial({ color: 0x00f3ff, side: THREE.DoubleSide, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending });
[0.18, 0.29, 0.42].forEach(rad => {
  foreheadCore.add(new THREE.Mesh(new THREE.RingGeometry(rad, rad + 0.02, 48), accentMat));
});
const flareMesh = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 0.045), accentMat);
foreheadCore.add(flareMesh);

function makeEye(xOff) {
  const g = new THREE.Group();
  g.position.set(xOff, 0.22, 0.88);
  g.add(new THREE.Mesh(new THREE.SphereGeometry(0.055, 14, 14), new THREE.MeshBasicMaterial({ color: 0xffffff })));
  g.add(new THREE.Mesh(new THREE.RingGeometry(0.08, 0.135, 28), accentMat));
  return g;
}
avatarExtras.add(makeEye(-0.35), makeEye(0.35));

const pillarMat = new THREE.MeshBasicMaterial({ color: 0x00f3ff, transparent: true, opacity: 0.32, blending: THREE.AdditiveBlending, side: THREE.DoubleSide });
scene.add(new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 9.0, 12, 1, true), pillarMat));

const floorPortal = new THREE.Group();
floorPortal.position.set(0, -2.35, 0);
floorPortal.rotation.x = -Math.PI / 2;
scene.add(floorPortal);
const floorRings = [];
[0.85, 1.45, 2.15, 2.85, 3.6].forEach((r, idx) => {
  const m = new THREE.Mesh(new THREE.RingGeometry(r, r + (idx % 2 === 0 ? 0.04 : 0.018), idx % 2 === 0 ? 64 : 6), accentMat);
  floorPortal.add(m);
  floorRings.push(m);
});

const HAND_BONES = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]
];
function createHandRig() {
  const group = new THREE.Group();
  const joints = [];
  for (let i = 0; i < 21; i++) {
    const sp = new THREE.Mesh(new THREE.SphereGeometry(i in [4,8,9] ? 0.06 : 0.035, 8, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    group.add(sp); joints.push(sp);
  }
  const bonePositions = new Float32Array(HAND_BONES.length * 6);
  const boneGeo = new THREE.BufferGeometry();
  boneGeo.setAttribute('position', new THREE.BufferAttribute(bonePositions, 3));
  const bones = new THREE.LineSegments(boneGeo, accentMat);
  group.add(bones);
  const repulsor = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.24, 28), accentMat);
  group.add(repulsor);
  group.visible = false;
  scene.add(group);
  return { group, joints, bones, bonePositions, repulsor };
}
const handRigs = [createHandRig(), createHandRig()];

const tetherPositions = new Float32Array(6);
const tetherGeo = new THREE.BufferGeometry();
tetherGeo.setAttribute('position', new THREE.BufferAttribute(tetherPositions, 3));
const tetherLine = new THREE.Line(tetherGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.75, blending: THREE.AdditiveBlending }));
tetherLine.visible = false;
scene.add(tetherLine);

let morphProgress = 1.0;

function morphHologram(shapeName) {
  if (!shapes[shapeName] || shapeName === currentShape) return;
  const posAttr = particleGeo.attributes.position;
  const tarAttr = particleGeo.attributes.aTarget;
  posAttr.array.set(shapes[currentShape]);
  tarAttr.array.set(shapes[shapeName]);
  posAttr.needsUpdate = true;
  tarAttr.needsUpdate = true;

  currentShape = shapeName;
  morphProgress = 0.0;
  shaderUniforms.uMorph.value = 0.0;
  shaderUniforms.uIsAvatar.value = (shapeName === 'AVATAR') ? 1.0 : 0.0;
  avatarExtras.visible = (shapeName === 'AVATAR');

  ['AVATAR', 'SINGULARITY', 'GLOBE', 'REACTOR'].forEach(s => {
    const b = $('btnShape' + s);
    if (b) b.classList.toggle('active', s === shapeName);
  });
  triggerShockwave();
  log(`GPU Shader morphed to ${shapeName}`, 'ok');
  toast(`CONSTRUCT // ${shapeName}`);
}

function setMode(m) {
  if (m === currentMode) return;
  currentMode = m;
  document.body.dataset.mode = m;
  ['NORMAL', 'OVERDRIVE', 'CORTEX', 'STEALTH'].forEach(k => {
    const b = $('btnMode' + k);
    if (b) b.classList.toggle('active', k === m);
  });
  const pal = MODE_COLORS[m] || MODE_COLORS.NORMAL;
  shaderUniforms.uColorPri.value.copy(pal.pri);
  accentMat.color.copy(pal.pri);
  pillarMat.color.copy(pal.pri);
  triggerShockwave();
  log(`Protocol switched to ${m}`, 'ok');
  toast(`MODE // ${m}`);
}

function triggerShockwave() {
  shockwaveEnergy = 0.35;
  sfx(880, 'sawtooth', 0.16, 0.05);
}

function systemReport() {
  sendPrompt('Run full hardware diagnostic report');
}

// 60HZ REAL-TIME HAND TELEKINESIS STREAM
let targetYaw = 0, targetPitch = 0, targetRoll = 0, targetScale = 1.0;
let handActive = false;
let lastVoiceEventId = 0;
let pinchToggleLock = false;

const stream = new EventSource('/api/stream');
stream.onmessage = e => {
  const d = JSON.parse(e.data);
  $('camFps').textContent = d.camera_ready ? d.cam_fps + ' FPS' : 'OFFLINE';
  $('gestureLabel').textContent = d.camera_ready ? d.gesture : d.camera_status;
  $('handCountBadge').textContent = !d.camera_ready
    ? 'CAMERA OFFLINE'
    : d.hand_detected ? `${d.hand_count} HAND LOCKED` : 'NO HAND';
  $('pinchBadge').textContent = d.is_pinching ? 'PINCH ACTIVE' : 'OPEN';

  if (d.voice_id > lastVoiceEventId && d.voice_text) {
    lastVoiceEventId = d.voice_id;
    $('heard').textContent = `Heard: "${d.voice_text}"`;
    sendPrompt(d.voice_text);
  }

  handActive = d.hand_detected;
  if (d.hand_detected) {
    targetYaw   = THREE.MathUtils.degToRad(d.yaw);
    targetPitch = THREE.MathUtils.degToRad(d.pitch);
    targetRoll  = THREE.MathUtils.degToRad(d.roll);
    targetScale = d.scale;

    $('rotReadout').textContent = `${Math.round(d.yaw)}Y / ${Math.round(d.pitch)}P / ${Math.round(d.roll)}R`;
    $('scaleReadout').textContent = `${d.scale.toFixed(2)}x`;

    if (d.is_pinching && !pinchToggleLock) {
      pinchToggleLock = true;
      setMode(currentMode === 'OVERDRIVE' ? 'NORMAL' : 'OVERDRIVE');
    } else if (!d.is_pinching) {
      pinchToggleLock = false;
    }

    const hands = d.hands_3d || [];
    handRigs.forEach((rig, hIdx) => {
      if (hIdx < hands.length) {
        const pts = hands[hIdx];
        rig.group.visible = true;
        for (let i = 0; i < 21; i++) rig.joints[i].position.set(pts[i][0], pts[i][1], pts[i][2]);
        HAND_BONES.forEach(([a, b], bIdx) => {
          const off = bIdx * 6;
          rig.bonePositions[off]   = pts[a][0]; rig.bonePositions[off+1] = pts[a][1]; rig.bonePositions[off+2] = pts[a][2];
          rig.bonePositions[off+3] = pts[b][0]; rig.bonePositions[off+4] = pts[b][1]; rig.bonePositions[off+5] = pts[b][2];
        });
        rig.bones.geometry.attributes.position.needsUpdate = true;
        rig.repulsor.position.set(pts[9][0], pts[9][1], pts[9][2]);
        rig.repulsor.rotation.z += 0.18;
      } else {
        rig.group.visible = false;
      }
    });

    if (hands.length > 0) {
      const palm = hands[0][9];
      tetherLine.visible = true;
      tetherPositions[0] = palm[0]; tetherPositions[1] = palm[1]; tetherPositions[2] = palm[2];
      tetherPositions[3] = 0;       tetherPositions[4] = 0.62;    tetherPositions[5] = 0.95;
      tetherGeo.attributes.position.needsUpdate = true;
    }
  } else {
    handRigs.forEach(r => r.group.visible = false);
    tetherLine.visible = false;
  }
};

let isDragging = false, prevX = 0, prevY = 0;
window.addEventListener('mousedown', e => {
  if (e.target.tagName === 'CANVAS') { isDragging = true; prevX = e.clientX; prevY = e.clientY; }
});
window.addEventListener('mousemove', e => {
  if (handActive) return;
  if (isDragging) {
    targetYaw += (e.clientX - prevX) * 0.008;
    targetPitch += (e.clientY - prevY) * 0.008;
    prevX = e.clientX; prevY = e.clientY;
  } else {
    targetYaw = ((e.clientX / innerWidth) - 0.5) * 0.9;
    targetPitch = ((e.clientY / innerHeight) - 0.5) * 0.55;
  }
});
window.addEventListener('mouseup', () => isDragging = false);
window.addEventListener('wheel', e => {
  targetScale = Math.max(0.55, Math.min(1.85, targetScale - e.deltaY * 0.001));
});
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// VOICE SYNTHESIS & AUTO-LISTEN
let isSpeaking = false;
let autoListenEnabled = true;

function stopSpeaking() {
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  isSpeaking = false;
  vocalEnergy = 0;
}

function speak(text) {
  stopSpeaking();
  $('reply').textContent = text;
  if ('speechSynthesis' in window) {
    isSpeaking = true;
    $('voiceStatus').textContent = 'JARVIS SPEAKING';
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'en-US';
    u.rate = 1.04;
    u.pitch = 0.92;
    const voices = speechSynthesis.getVoices();
    const pref =
      voices.find(v => /Google UK English Male|Microsoft George|Microsoft Ryan|Daniel/i.test(v.name)) ||
      voices.find(v => /Google US English|Microsoft David|Microsoft Mark/i.test(v.name)) ||
      voices.find(v => /^en/i.test(v.lang));
    if (pref) u.voice = pref;
    u.onboundary = () => { vocalEnergy = 0.85 + Math.random() * 0.45; };
    u.onend = u.onerror = () => {
      isSpeaking = false;
      vocalEnergy = 0;
      if (autoListenEnabled) {
        $('voiceStatus').textContent = 'LISTENING (AUTO)';
        resumeRecognitionIfNeeded();
      } else {
        $('voiceStatus').textContent = 'STANDBY';
      }
    };
    speechSynthesis.speak(u);
  }
}
if ('speechSynthesis' in window) speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();

let recognition = null, history = [], busy = false;
let micStream = null, micAnalyser = null, micFreqBins = new Uint8Array(64);
let pcmBuffer = [], isPttActive = false;

async function ensureMicStream() {
  if (micAnalyser) return;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    const src = audioCtx.createMediaStreamSource(micStream);
    micAnalyser = audioCtx.createAnalyser();
    micAnalyser.fftSize = 128;
    micFreqBins = new Uint8Array(micAnalyser.frequencyBinCount);
    src.connect(micAnalyser);

    const proc = audioCtx.createScriptProcessor(4096, 1, 1);
    src.connect(proc);
    proc.connect(audioCtx.destination);
    const ratio = audioCtx.sampleRate / 16000;
    proc.onaudioprocess = e => {
      if (!isPttActive) return;
      const input = e.inputBuffer.getChannelData(0);
      const outLen = Math.floor(input.length / ratio);
      const pcm16 = new Int16Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const s = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      pcmBuffer.push(pcm16);
    };
  } catch (e) {}
}

function startWebSpeechLoop() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;
  if (!recognition) {
    recognition = new SR();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.continuous = true;
    recognition.onresult = e => {
      if (isSpeaking) return;
      let finalStr = '', interimStr = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const tr = e.results[i][0].transcript;
        if (e.results[i].isFinal) finalStr += tr;
        else interimStr += tr;
      }
      if (interimStr) {
        $('heard').textContent = 'Hearing: "' + interimStr.trim() + '..."';
        vocalEnergy = 0.4;
      }
      if (finalStr.trim()) {
        const clean = finalStr.trim();
        $('heard').textContent = 'Heard: "' + clean + '"';
        sendPrompt(clean);
      }
    };
    recognition.onerror = () => {};
    recognition.onend = () => {
      if (autoListenEnabled && !isSpeaking) setTimeout(resumeRecognitionIfNeeded, 200);
    };
  }
  resumeRecognitionIfNeeded();
}

function resumeRecognitionIfNeeded() {
  if (!autoListenEnabled || !recognition || isSpeaking) return;
  try { recognition.start(); } catch (e) {}
}

function toggleMic() {
  ensureMicStream();
  autoListenEnabled = !autoListenEnabled;
  const btn = $('micBtn');
  btn.classList.toggle('active', autoListenEnabled);
  btn.textContent = autoListenEnabled ? 'AUTO-MIC: ON' : 'AUTO-MIC: OFF';
  $('voiceStatus').textContent = autoListenEnabled ? 'LISTENING (AUTO)' : 'STANDBY';
  sfx(autoListenEnabled ? 840 : 360, 'triangle', 0.08);
  if (autoListenEnabled) startWebSpeechLoop();
  else if (recognition) { try { recognition.stop(); } catch (e) {} }
}

async function startPtt() {
  if (isPttActive) return;
  await ensureMicStream();
  stopSpeaking();
  isPttActive = true;
  pcmBuffer = [];
  $('pttBtn').classList.add('active');
  $('voiceStatus').textContent = 'RECORDING [SPACE]';
  sfx(760, 'sine', 0.07);
}

async function stopPtt() {
  if (!isPttActive) return;
  isPttActive = false;
  $('pttBtn').classList.remove('active');
  sfx(520, 'sine', 0.07);
  if (pcmBuffer.length === 0) return;
  let total = 0;
  pcmBuffer.forEach(b => total += b.length);
  const merged = new Int16Array(total);
  let off = 0;
  pcmBuffer.forEach(b => { merged.set(b, off); off += b.length; });
  pcmBuffer = [];

  $('voiceStatus').textContent = 'DECODING...';
  try {
    const r = await fetch('/api/stt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: new Uint8Array(merged.buffer)
    });
    const d = await r.json();
    if (d.transcript) {
      $('heard').textContent = 'Heard: "' + d.transcript + '"';
      sendPrompt(d.transcript);
    } else {
      $('voiceStatus').textContent = autoListenEnabled ? 'LISTENING (AUTO)' : 'STANDBY';
    }
  } catch (e) {
    $('voiceStatus').textContent = 'STANDBY';
  }
}

window.addEventListener('keydown', e => {
  if (document.activeElement === $('prompt')) return;
  if (e.code === 'Space' && !e.repeat) { e.preventDefault(); startPtt(); }
});
window.addEventListener('keyup', e => {
  if (document.activeElement === $('prompt')) return;
  if (e.code === 'Space') { e.preventDefault(); stopPtt(); }
});

setTimeout(() => { ensureMicStream(); startWebSpeechLoop(); }, 500);

async function sendPrompt(text) {
  const input = $('prompt');
  const prompt = (text || input.value).trim();
  if (!prompt || busy) return;
  input.value = '';
  sfx(880, 'sine', 0.06);

  history.push({ role: 'user', content: prompt });
  $('send').disabled = true;
  busy = true;
  log(`USER: "${prompt.slice(0, 42)}"`);

  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history.slice(-10) })
    });
    if (!r.ok) throw new Error(`Chat request failed (${r.status})`);
    const data = await r.json();
    const answer = (data.reply || 'Directive executed, Sir.').trim();
    history.push({ role: 'assistant', content: answer });

    if (data.shape) morphHologram(data.shape);
    if (data.mode) setMode(data.mode);
    if (data.action) triggerShockwave();

    log('JARVIS replied (<2ms core)', 'ok');
    speak(answer);
  } catch (e) {
    log('Command failed. Check the server and try again.');
    speak('I could not process that request. Please check the server and try again.');
  } finally {
    busy = false;
    $('send').disabled = false;
  }
}

$('prompt').addEventListener('keydown', e => { if (e.key === 'Enter') sendPrompt(); });

async function pollMetrics() {
  try {
    const r = await fetch('/api/metrics');
    if (!r.ok) throw new Error(`Metrics request failed (${r.status})`);
    const d = await r.json();
    const formatPercent = value => Number.isFinite(value) ? `${value}%` : 'N/A';
    const formatBar = value => Number.isFinite(value) ? `${Math.max(0, Math.min(100, value))}%` : '0%';
    $('cpu').textContent = formatPercent(d.cpu);
    $('ram').textContent = formatPercent(d.ram);
    $('disk').textContent = formatPercent(d.disk);
    $('cpuBar').style.width = formatBar(d.cpu);
    $('ramBar').style.width = formatBar(d.ram);
    $('diskBar').style.width = formatBar(d.disk);
    $('ramDetail').textContent = Number.isFinite(d.ram_used_gb) && Number.isFinite(d.ram_total_gb)
      ? `${d.ram_used_gb} / ${d.ram_total_gb} GB`
      : 'N/A';
    $('host').textContent = (d.hostname || 'TITAN').slice(0, 16);
    $('netSpeed').textContent = Number.isFinite(d.net_up_kbs) && Number.isFinite(d.net_down_kbs)
      ? `${d.net_up_kbs}↑ / ${d.net_down_kbs}↓ KB/s`
      : 'N/A';
    $('procCount').textContent = Number.isFinite(d.processes) ? d.processes : 'N/A';
    $('uptime').textContent = d.uptime;
    if (d.cores) {
      const cols = coresGrid.children;
      for (let i = 0; i < cols.length; i++) {
        const value = d.cores[i];
        cols[i].firstElementChild.style.height = Number.isFinite(value) ? formatBar(value) : '0%';
      }
    }
  } catch (e) {
    log('System telemetry is temporarily unavailable.');
  }
}
pollMetrics();
setInterval(pollMetrics, 1800);
setInterval(() => { $('clock').textContent = new Date().toLocaleTimeString('en-GB'); }, 1000);

const sCanvas = $('spectrumCanvas'), sCtx = sCanvas.getContext('2d');
let lastFpsTime = performance.now(), frames = 0;
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  frames++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    $('fpsCounter').textContent = frames + ' FPS';
    frames = 0; lastFpsTime = now;
  }

  if (morphProgress < 1.0) {
    morphProgress = Math.min(1.0, morphProgress + 0.045);
    shaderUniforms.uMorph.value = morphProgress * morphProgress * (3.0 - 2.0 * morphProgress);
  }

  if (isSpeaking) {
    vocalEnergy = 0.35 + 0.55 * Math.abs(Math.sin(t * 22) * Math.cos(t * 14));
  } else {
    vocalEnergy *= 0.88;
  }
  shockwaveEnergy *= 0.90;

  shaderUniforms.uTime.value = t;
  shaderUniforms.uJawDrop.value = vocalEnergy * 0.16;

  holoRoot.rotation.y += (targetYaw - holoRoot.rotation.y) * 0.25;
  holoRoot.rotation.x += (targetPitch - holoRoot.rotation.x) * 0.25;
  holoRoot.rotation.z += (targetRoll - holoRoot.rotation.z) * 0.22;

  const activeScale = targetScale + vocalEnergy * 0.08 + shockwaveEnergy;
  holoRoot.scale.setScalar(holoRoot.scale.x + (activeScale - holoRoot.scale.x) * 0.22);

  thirdEyeSphere.scale.setScalar(1.0 + vocalEnergy * 0.55 + 0.2 * Math.sin(t * 5));
  flareMesh.scale.x = 0.85 + 0.35 * Math.sin(t * 3.5) + vocalEnergy * 0.5;
  floorRings.forEach((r, idx) => { r.rotation.z = t * (idx % 2 === 0 ? 0.5 : -0.65); });

  renderer.render(scene, camera);

  const hexCol = (MODE_COLORS[currentMode] || MODE_COLORS.NORMAL).hex;
  sCtx.clearRect(0, 0, 360, 48);
  sCtx.fillStyle = hexCol;
  if (micAnalyser && !isSpeaking) micAnalyser.getByteFrequencyData(micFreqBins);
  const bars = 45, bw = 360 / bars;
  for (let i = 0; i < bars; i++) {
    let val = isSpeaking
      ? (0.2 + 0.75 * Math.abs(Math.sin(t * 18 + i * 0.35)))
      : (micAnalyser ? micFreqBins[i] / 255.0 : 0.06);
    const bh = Math.max(3, val * 42);
    sCtx.fillRect(i * bw + 1, 46 - bh, bw - 2, bh);
  }
}

log('All-in-One GLSL GPU Shader Engine online (24,000 nodes)', 'ok');
log('100Hz Python Hand Telekinesis stream locked', 'ok');
animate();
</script>
</body>
</html>
"""


# ============================================================
# MULTI-THREADED HTTP SERVER (SERVES EMBEDDED HTML DIRECTLY)
# ============================================================
class TitanHTTPHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML_TITAN.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/metrics":
            return self._json(200, collect_system_metrics())

        if self.path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while STATE.running:
                    self.wfile.write(f"data: {STATE.snapshot_json()}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(1.0 / STREAM_HZ)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            return

        if self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while STATE.running:
                    jpg = STATE.jpeg_frame
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.025)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            return

        self.send_error(404)

    def do_POST(self):
        if self.path not in ("/api/chat", "/api/stt"):
            return self._json(404, {"error": "Not found"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "Invalid Content-Length"})
        if length <= 0:
            return self._json(400, {"error": "Request body is required"})
        if length > MAX_REQUEST_BYTES:
            return self._json(413, {"error": "Request body is too large"})

        raw_body = self.rfile.read(length)
        if len(raw_body) != length:
            return self._json(400, {"error": "Incomplete request body"})

        if self.path == "/api/chat":
            try:
                data = json.loads(raw_body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._json(400, {"error": "Request body must be valid JSON"})
            if not isinstance(data, dict) or not isinstance(data.get("messages", []), list):
                return self._json(400, {"error": "messages must be a JSON array"})
            return self._json(200, generate_jarvis_reply(data.get("messages", [])))

        if sr is None:
            return self._json(503, {"error": "Speech recognition is not installed"})
        if len(raw_body) < 3200:
            return self._json(400, {"error": "Audio request is too short"})
        try:
            rec = sr.Recognizer()
            audio_data = sr.AudioData(raw_body, 16000, 2)
            text = rec.recognize_google(audio_data, language="en-US").strip()
            return self._json(200, {"transcript": text})
        except sr.UnknownValueError:
            return self._json(422, {"error": "Speech could not be understood"})
        except sr.RequestError as exc:
            LOG.warning("Speech recognition service failed: %s", exc)
            return self._json(502, {"error": "Speech recognition service is unavailable"})
        except ValueError:
            return self._json(400, {"error": "Invalid audio data"})


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, OSError) and (
            exc.errno in (errno.ECONNABORTED, errno.ECONNRESET, errno.EPIPE)
            or getattr(exc, "winerror", None) in (10053, 10054)
        ):
            LOG.debug("HTTP client %s disconnected", client_address)
            return
        super().handle_error(request, client_address)


def launch_browser(url: str) -> None:
    browsers = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for exe in browsers:
        if os.path.exists(exe):
            try:
                subprocess.Popen([
                    exe,
                    f"--app={url}",
                    "--start-fullscreen",
                    "--autoplay-policy=no-user-gesture-required",
                ])
                print(f"  [✓] Fullscreen Holographic Window Launched: {url}")
                return
            except Exception:
                pass
    webbrowser.open(url)


if __name__ == "__main__":
    # Automatically pick a free port if 8765 is already held by an older run
    server = None
    for p in range(PORT, PORT + 15):
        try:
            server = ThreadedHTTPServer((HOST, p), TitanHTTPHandler)
            PORT = p
            break
        except OSError:
            continue
    if server is None:
        raise OSError(f"Could not bind the dashboard to {HOST} on ports {PORT}-{PORT + 14}")

    url = f"http://{HOST}:{PORT}"
    print("\n" + "=" * 68)
    print("  [★] J.A.R.V.I.S // OBSIDIAN TITAN (ALL-IN-ONE GPU SHADER ENGINE)")
    print(f"  [+] Command Sanctuary : {url}")
    print("=" * 68)

    if psutil:
        psutil.cpu_percent(interval=None)

    threading.Thread(target=vision_telekinesis_worker, daemon=True, name="TitanVision").start()
    threading.Thread(target=background_mic_worker, daemon=True, name="TitanMic").start()

    threading.Timer(0.5, lambda: launch_browser(url)).start()
    print("  [✓] Server Running Smoothly — Press Ctrl+C in terminal to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        STATE.running = False
        server.server_close()
        print("\n  [+] J.A.R.V.I.S Shut Down Complete.")