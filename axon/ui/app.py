"""
AXON — Flask + SocketIO UI server
"""
import os, sys, threading
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from axon.core.engine import AxonEngine

app      = Flask(__name__, template_folder="../../web/templates")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
_engine  = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    if _engine:
        return jsonify(_engine.get_status())
    return jsonify({"running": False})

@socketio.on("connect")
def on_connect():
    emit("log", {"msg": "Connected to AXON."})

@socketio.on("user_text")
def on_user_text(data):
    if _engine:
        _engine.process_text(data.get("text",""))

@socketio.on("start_engine")
def on_start(config):
    global _engine
    if _engine and _engine.running:
        emit("log", {"msg": "Already running."}); return
    api_key        = config.get("api_key")        or os.getenv("ANTHROPIC_API_KEY","")
    lm_url         = config.get("lm_url",         "http://localhost:1234")
    lm_model       = config.get("lm_model",       None) or None
    prefer_local   = config.get("prefer_local",   True)
    _engine = AxonEngine(
        socketio=socketio,
        api_key=api_key,
        lm_studio_url=lm_url,
        lm_studio_model=lm_model,
        prefer_local=prefer_local,
    )
    _engine.start(
        enable_camera=config.get("camera", True),
        enable_mic=config.get("mic", True),
        camera_index=config.get("camera_index", -1),
    )

@socketio.on("stop_engine")
def on_stop():
    global _engine
    if _engine: _engine.stop(); _engine = None
    emit("log", {"msg": "AXON stopped."})

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--no-mic",    action="store_true")
    p.add_argument("--api-key",   default="")
    args = p.parse_args()

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY","")
    print("\n  AXON — Emerging Intelligence\n  Open: http://localhost:7777\n")

    _engine = AxonEngine(socketio=socketio, api_key=api_key,
                          lm_studio_url="http://localhost:1234",
                          prefer_local=True)
    threading.Thread(
        target=lambda: _engine.start(
            enable_camera=not args.no_camera,
            enable_mic=not args.no_mic,
        ), daemon=True
    ).start()

    socketio.run(app, host="0.0.0.0", port=7777, debug=False)
