"""
AXON — Flask + SocketIO UI server
"""
import os, sys, threading
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

@app.route("/api/user_profile")
def user_profile():
    if _engine and hasattr(_engine.language, "user_model"):
        return jsonify(_engine.language.user_model.get_profile())
    return jsonify({})

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
    )

@socketio.on("stop_engine")
def on_stop():
    global _engine
    if _engine:
        _engine.stop()
        _engine = None
    emit("log", {"msg": "AXON stopped."})

if __name__ == "__main__":
    print("\n  AXON — Emerging Intelligence\n  Open: http://localhost:7777\n")
    socketio.run(app, host="0.0.0.0", port=7777, debug=False)
