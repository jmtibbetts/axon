"""
AXON — Flask + SocketIO UI server
"""
import os, sys, threading
try:
    import torch as _torch
    _CUDA_NAME = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None
except ImportError:
    _CUDA_NAME = None
from pathlib import Path
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from axon.core.engine import AxonEngine

app      = Flask(__name__, template_folder="../../web/templates")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
_engine: AxonEngine = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify(_engine.get_status() if _engine else {"running": False})

@app.route("/api/audio_diag")
def audio_diag():
    """Returns full audio diagnostics: TTS engine, playback method, mic devices."""
    from axon.sensory.auditory import AuditorySystem
    from axon.cognition.voice_output import EDGE_TTS_OK, PLAYBACK
    mics = []
    try:
        mics = AuditorySystem.list_devices()
    except Exception as e:
        pass
    voice_status = {}
    if _engine:
        voice_status = _engine.voice.get_status()
    return jsonify({
        "edge_tts":    EDGE_TTS_OK,
        "playback":    PLAYBACK or "none",
        "mics":        mics,
        "voice":       voice_status,
        "mic_running": _engine.auditory.running if _engine else False,
        "whisper_loaded": _engine.auditory._whisper is not None if _engine else False,
    })

@app.route("/api/mics")
def list_mics():
    from axon.sensory.auditory import AuditorySystem
    try:
        devs = AuditorySystem.list_devices()
    except Exception as e:
        devs = []
        print(f"  [Mics] Error: {e}")
    return jsonify({"mics": devs})

@app.route("/api/cameras")
def list_cameras():
    from axon.sensory.optic import OpticSystem
    try:
        cams = OpticSystem.list_cameras()
    except Exception as e:
        cams = []
        print(f"  [Cameras] Error: {e}")
    return jsonify({"cameras": cams})

@app.route("/api/user_profile")
def user_profile():
    if _engine and hasattr(_engine.language, "user_model"):
        return jsonify(_engine.language.user_model.get_profile())
    return jsonify({})

@app.route("/api/memory_summary")
def memory_summary():
    if _engine and hasattr(_engine, "memory"):
        return jsonify(_engine.memory.memory_summary())
    return jsonify({"episodes": 0, "facts": {}, "top_topics": [], "top_connections": []})

@socketio.on("connect")
def on_connect():
    emit("log", {"msg": "Connected to AXON."})
    if _engine:
        emit("lm_status", _engine.language.get_status())

@socketio.on("chat")
def on_chat(data):
    if _engine:
        _engine.chat(data.get("text", ""))
    else:
        emit("log", {"msg": "Engine not started — hit ACTIVATE first."})

@socketio.on("user_text")
def on_user_text(data):
    if _engine:
        _engine.chat(data.get("text", ""))

@socketio.on("reprobe_lm")
def on_reprobe():
    if _engine:
        _engine.language._probe_lm_studio()
        emit("lm_status", _engine.language.get_status())

@socketio.on("start_engine")
def on_start(config):
    global _engine
    if _engine and _engine.running:
        emit("log", {"msg": "Already running."}); return
    api_key      = config.get("api_key")    or os.getenv("ANTHROPIC_API_KEY", "")
    lm_url       = config.get("lm_url",       "http://localhost:1234")
    lm_model     = config.get("lm_model",     None) or None
    prefer_local = config.get("prefer_local", True)

    _engine = AxonEngine(
        socketio=socketio,
        api_key=api_key,
        lm_studio_url=lm_url,
        lm_studio_model=lm_model,
        prefer_local=prefer_local,
    )
    if not config.get("voice", True):
        _engine.voice.enabled = False

    _engine.start(
        enable_camera=config.get("camera",       True),
        enable_mic=config.get("mic",             True),
        camera_index=config.get("camera_index",  -1),
        mic_index=config.get("mic_index", None),
    )

@socketio.on("stop_engine")
def on_stop():
    global _engine
    if _engine:
        _engine.stop()
        _engine = None
    emit("log", {"msg": "AXON stopped."})

def _sanitize(obj):
    """Recursively convert numpy/torch scalar types to native Python for JSON."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
    except ImportError:
        pass
    return obj


@socketio.on("diagnostic")
def on_diagnostic():
    global _engine
    if not _engine or not _engine.running:
        emit("diagnostic_result", {"error": "Engine not running."})
        return
    try:
        data = _engine.get_diagnostic()
        emit("diagnostic_result", _sanitize(data))
    except Exception as e:
        import traceback
        emit("diagnostic_result", {"error": str(e) + "\n" + traceback.format_exc()[-800:]})


@socketio.on("get_voice_config")
def on_get_voice_config():
    global _engine
    if not _engine or not _engine.voice:
        emit("voice_config", {"voice": "en-US-AriaNeural", "rate": "-5%",
                               "pitch": "-3Hz", "catalogue": []})
        return
    emit("voice_config", _engine.voice.get_voice_config())


@socketio.on("set_voice")
def on_set_voice(data):
    global _engine
    if not _engine or not _engine.voice:
        emit("voice_config_ack", {"ok": False, "error": "Engine not running"})
        return
    voice_id = data.get("voice_id")
    rate     = data.get("rate")
    pitch    = data.get("pitch")
    _engine.voice.set_voice(voice_id=voice_id, rate=rate, pitch=pitch)
    cfg = _engine.voice.get_voice_config()
    emit("voice_config_ack", {"ok": True, **cfg})
    # Confirm with a test phrase
    if data.get("test", False) and voice_id:
        label = next((v["label"] for v in cfg["catalogue"] if v["id"] == voice_id), voice_id)
        import threading
        threading.Thread(
            target=lambda: _engine.voice.speak(f"Hello, I'm {label}. This is my voice.", interrupt=True),
            daemon=True
        ).start()

if __name__ == "__main__":
    import signal, sys

    def _shutdown(sig=None, frame=None):
        print("\n  [AXON] Shutting down…")
        global _engine
        if _engine:
            try:
                _engine.stop()
            except Exception:
                pass
            _engine = None
        # Give threads a moment then hard-exit
        import threading
        t = threading.Thread(target=lambda: (__import__("time").sleep(1.5), os._exit(0)), daemon=True)
        t.start()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("\n  AXON — Emerging Intelligence\n  Open: http://localhost:7777\n")
    print("  Press Ctrl+C to exit\n")
    try:
        socketio.run(app, host="0.0.0.0", port=7777, debug=False,
                     allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        _shutdown()
