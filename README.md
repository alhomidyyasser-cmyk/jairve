# J.A.R.V.I.S. — Obsidian Titan

A local-first, single-file Python dashboard with webcam hand tracking, live
system telemetry, voice input, and an interactive Three.js hologram. The
dashboard is embedded in `jarvis.py`; no separate frontend build step is
required.

## Features

- Hand tracking with one- and two-hand hologram controls.
- Live CPU, memory, disk, network, process, and uptime metrics when `psutil`
  is installed.
- Browser-based speech input and speech synthesis where supported.
- Optional Python microphone listener and speech-to-text endpoint.
- Local keyword responses and safe arithmetic evaluation, with an optional
  remote language-model fallback.
- A threaded local HTTP server with a configurable host and port.

## Requirements

- Python 3.10 or later.
- A webcam for hand tracking.
- A modern browser with WebGL for the animated dashboard.
- Internet access for the externally hosted Three.js library, fonts, browser
  speech recognition (where provided by the browser), and remote AI responses.

The Python microphone and speech-to-text features also use Google's online
speech-recognition service. Camera and microphone access are sensitive: the
camera is opened by the Python process, and browser microphone access is
requested by the browser.

## Install

From the project directory, create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The MediaPipe and OpenCV versions are constrained to keep the `mp.solutions.hands`
camera API available and avoid conflicting OpenCV wheels.

On macOS or Linux, activate the environment with:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```sh
python jarvis.py
```

The server prints its address and attempts to open the dashboard in a browser.
If it does not open automatically, navigate to the printed address (normally
`http://127.0.0.1:8765`). Keep the terminal open while using the dashboard and
press `Ctrl+C` to stop the server. `index.html` is a small standalone launcher;
the Python server serves the full dashboard from the embedded HTML.

The dashboard uses the first available port from the configured port through
the next 14 ports if the default is already occupied.

### Repair an existing Windows environment

If you previously installed a newer MediaPipe or both OpenCV wheel variants,
remove those conflicting packages and reinstall the pinned project requirements
from the activated `.venv`:

```powershell
python -m pip uninstall -y mediapipe opencv-python opencv-contrib-python
python -m pip install -r requirements.txt
python -c "import mediapipe as mp; print(mp.__version__, hasattr(mp, 'solutions'))"
```

The last command should print a MediaPipe `0.10.x` version followed by `True`.
Then start the dashboard with `python jarvis.py`.

## Configuration

Set environment variables before starting the application:

- `JARVIS_HOST` — interface to bind to; defaults to `127.0.0.1`. Keep this
  loopback address unless you deliberately need LAN access. The server has no
  authentication and should not be exposed to untrusted networks.
- `JARVIS_PORT` — first port to try; defaults to `8765`.
- `JARVIS_CAM` — OpenCV camera index; defaults to `0`.
- `JARVIS_CITY` — city name used in local-time responses; defaults to `Sana'a`.
- `JARVIS_API_URL` — remote chat-completion endpoint; defaults to
  `https://text.pollinations.ai/`.
- `JARVIS_MODEL` — model identifier sent to that endpoint; defaults to
  `openai`.
- `JARVIS_AI_TIMEOUT` — remote request timeout in seconds; defaults to `4.0`.
- `JARVIS_STREAM_HZ` — server-sent event update rate in hertz; defaults to
  `60` and is clamped to at least `1`.
- `JARVIS_PY_MIC` — set to `1` to enable the optional Python microphone
  listener. It is disabled by default.

For example, in PowerShell:

```powershell
$env:JARVIS_PORT = "8766"
$env:JARVIS_CAM = "1"
python jarvis.py
```

Chat messages sent to a configured remote model leave the local machine. Review
the endpoint and its privacy policy before using it with sensitive information.
Voice recognition also sends audio to Google's recognition service.

## Optional Python microphone listener

The default browser speech controls do not require the Python microphone
listener. To enable the listener, install a compatible PyAudio package for your
platform, then set `JARVIS_PY_MIC=1` before launching. Microphone device and
PyAudio installation issues are reported in the terminal.

## Tests

Run the built-in tests with:

```sh
python -m unittest discover -s tests -v
```

## Project files

- `jarvis.py` — application, HTTP API, hand tracking, and embedded dashboard.
- `index.html` — standalone page with launch instructions.
- `requirements.txt` — Python runtime dependencies.
- `tests/` — focused regression tests.
