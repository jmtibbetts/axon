"""
AXON — Flask + SocketIO UI server
"""
import eventlet
eventlet.monkey_patch()
import os, sys, threading
try:
    import torch as _torch
    _CUDA_NAME = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None
except ImportError:
    _CUDA_NAME = None
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
import os, tempfile
from flask_socketio import SocketIO, emit
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from axon.core.engine   import AxonEngine
from axon.core.brain_api import AxonBrain
_brain: AxonBrain = None

app      = Flask(__name__,
                  template_folder="../../web/templates")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet", logger=False, engineio_logger=False, ping_timeout=120, ping_interval=30, max_http_buffer_size=10_000_000)

# Thread-safe emit bridge.
# flask-socketio's SocketIO.emit() (on the instance, not flask_socketio.emit())
# is documented as safe to call from background threads when using eventlet.
# We wrap it in a try/except to absorb any transient errors.
def _safe_emit(event: str, data: dict):
    """Emit from ANY context — greenlet, OS thread, or tpool — safely."""
    try:
        socketio.emit(event, data, broadcast=True)
    except Exception:
        pass

_engine: AxonEngine = None
_engine_lock     = threading.Lock()  # prevents double-init race
_engine_starting = False               # True while start_engine is in progress
_last_start_error: str = None          # last startup traceback for diagnostics

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


def _apply_deferred_onboarding(brain):
    """After engine starts, apply any onboarding choices that were saved pre-engine."""
    import json as _json
    from pathlib import Path as _Path
    ob_path = _Path("data/onboarding.json")
    if not ob_path.exists():
        return
    try:
        state = (_json.loads(ob_path.read_text()) if ob_path.read_text().strip() else {})
    except Exception:
        return
    if not state.get("completed"):
        return
    # Apply personality preset if engine doesn't already have it
    preset = state.get("preset", "")
    if preset:
        try:
            brain.onboarding_set_preset(preset)
            print(f"  [Onboarding] Applied deferred preset: {preset}")
        except Exception as e:
            print(f"  [Onboarding] Could not apply preset: {e}")
    # Ingest deferred sample
    sample_id = state.get("sample_id", "")
    if sample_id:
        try:
            brain.onboarding_ingest_sample(sample_id)
            print(f"  [Onboarding] Ingested deferred sample: {sample_id}")
        except Exception as e:
            print(f"  [Onboarding] Could not ingest sample: {e}")
    # Ingest deferred custom text
    custom_text = state.get("custom_text", "")
    if custom_text:
        try:
            brain.onboarding_ingest_text(custom_text)
            print(f"  [Onboarding] Ingested deferred custom text ({len(custom_text)} chars)")
        except Exception as e:
            print(f"  [Onboarding] Could not ingest custom text: {e}")


import os as _os
_UI_BUILD_DIR = _os.path.normpath(_os.path.join(_os.path.dirname(__file__), "../../web/static/ui"))

def _react_exists():
    return _os.path.exists(_os.path.join(_UI_BUILD_DIR, "index.html"))

@app.route("/")
@app.route("/react")
@app.route("/react/<path:path>")
def index(path=None):
    # Serve the React/Vite build
    if _react_exists():
        return send_from_directory(_UI_BUILD_DIR, "index.html")
    return render_template("monitor.html")

@app.route("/assets/<path:filename>")
def react_assets(filename):
    return send_from_directory(_os.path.join(_UI_BUILD_DIR, "assets"), filename)

@app.route("/favicon.svg")
@app.route("/icons.svg")
def react_static(filename=None):
    name = _os.path.basename(request.path)
    return send_from_directory(_UI_BUILD_DIR, name)

@app.route("/dashboard")
@app.route("/dashboard/<path:path>")
def dashboard(path=None):
    return render_template("monitor.html")

@app.route("/legacy")
def legacy():
    return render_template("index.html")

@app.route("/monitor")
def monitor():
    return render_template("monitor.html")

@app.route("/api/ready")
def ready():
    """Lightweight liveness probe — returns 200 as soon as the server is up."""
    return jsonify({"ready": True})

@app.route("/api/start_error")
def start_error():
    """Return the last engine startup traceback for debugging."""
    return jsonify({"error": _last_start_error, "has_engine": bool(_engine), "starting": _engine_starting})

@app.route("/api/status")
def status():
    if not _engine:
        return jsonify({"running": False})
    try:
        return jsonify(_sanitize(_engine.get_status()))
    except Exception as e:
        # Fallback — just confirm the engine is alive without the full state
        return jsonify({"running": bool(_engine and _engine.running), "error": str(e)})

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
    if _engine and _engine.running:
        emit("lm_status", _engine.language.get_status())
        # Tell monitor.html the engine is already online (handles page refresh)
        emit("log", {"msg": "AXON online — engine running."})
        # Push a full state snapshot directly to the connecting client
        try:
            snap = _sanitize(_engine.fabric.get_state_snapshot())
            emit("neural_state", snap)
        except Exception:
            pass
        # Push current vision frame if available
        try:
            vf = getattr(_engine, "_last_visual_ctx", None)
            if vf and vf.get("frame_b64"):
                emit("frame", _sanitize(vf))
        except Exception:
            pass

@socketio.on("disconnect")
def on_disconnect():
    global _observe_active
    _observe_active = False   # kill any running observe loop on disconnect

@socketio.on("chat")
def on_chat(data):
    if _engine:
        eventlet.spawn(_engine.chat, data.get("text", ""))
    else:
        emit("log", {"msg": "Engine not started — hit ACTIVATE first."})

@socketio.on("user_text")
def on_user_text(data):
    if _engine:
        eventlet.spawn(_engine.chat, data.get("text", ""))

@socketio.on("reprobe_lm")
def on_reprobe():
    if _engine:
        _engine.language._probe_lm_studio()
        emit("lm_status", _engine.language.get_status())

@socketio.on("get_provider_status")
def on_get_provider_status():
    if _engine:
        emit("provider_status", _engine.language.get_provider_status())
    else:
        from axon.cognition.providers import load_config, provider_status
        emit("provider_status", provider_status(load_config()))

@socketio.on("update_provider")
def on_update_provider(data):
    """data: {provider, key?, model?, set_active?, prefer_local?, lmstudio_url?}"""
    provider = data.get("provider", "lmstudio")
    kwargs   = {k: v for k, v in data.items() if k != "provider" and v is not None}
    if _engine:
        status = _engine.language.update_provider(provider, **kwargs)
    else:
        # Engine not started yet — update config on disk so it persists
        from axon.cognition.providers import load_config, save_config, provider_status
        cfg = load_config()
        if "key" in kwargs:         cfg[f"{provider}_key"]   = kwargs["key"]
        if "model" in kwargs:       cfg[f"{provider}_model"] = kwargs["model"]
        if "set_active" in kwargs and kwargs["set_active"]:
            cfg["active_provider"] = provider
        if "prefer_local" in kwargs: cfg["prefer_local"]      = kwargs["prefer_local"]
        if "lmstudio_url" in kwargs: cfg["lmstudio_url"]      = kwargs["lmstudio_url"]
        save_config(cfg)
        status = provider_status(cfg)
    emit("provider_status", status)

@socketio.on("start_engine")
def on_start(config):
    global _engine, _brain, _engine_starting
    with _engine_lock:
        if (_engine and _engine.running) or _engine_starting:
            emit("log", {"msg": "Already running."})
            # Still push onboarding state so page refreshes get the correct overlay state
            if _brain:
                emit("onboarding_state", _sanitize(_brain.get_onboarding_state()))
            return
        _engine_starting = True
    try:
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

        # Override engine's _emit to use the thread-safe queue
        def _engine_emit(event: str, data: dict):
            import numpy as _np
            def _san(o):
                if isinstance(o, dict):  return {k: _san(v) for k,v in o.items()}
                if isinstance(o, (list, tuple)): return [_san(v) for v in o]
                if isinstance(o, _np.integer):   return int(o)
                if isinstance(o, (_np.floating, _np.float32, _np.float64)): return float(o)
                if isinstance(o, _np.ndarray):   return o.tolist()
                try:
                    import torch as _t
                    if isinstance(o, _t.Tensor): return o.item() if o.numel()==1 else o.tolist()
                except ImportError: pass
                return o
            _safe_emit(event, _san(data))
        _engine._emit = _engine_emit
        _engine.fabric._socket_emit = lambda ev, d: _safe_emit(ev, _sanitize(d))

        # Run engine start in a greenlet so we don't block the eventlet hub
        # during the long GPU/model init sequence.
        sid = request.sid

        def _do_start():
            global _engine, _brain, _engine_starting, _last_start_error
            try:
                _engine.start(
                    enable_camera=config.get("camera",       True),
                    enable_mic=config.get("mic",             True),
                    camera_index=config.get("camera_index",  -1),
                    mic_index=config.get("mic_index", None),
                )
                # Wire public API layer
                _brain = AxonBrain(engine=_engine)
                _engine.fabric._socket_emit = lambda ev, d: _safe_emit(ev, _sanitize(d))
                _apply_deferred_onboarding(_brain)
                _safe_emit("onboarding_state", _sanitize(_brain.get_onboarding_state()))
                _safe_emit("log", {"msg": "✅ AXON online — engine running."})
            except Exception as _start_exc:
                import traceback
                tb = traceback.format_exc()
                _last_start_error = tb
                print(f"[AXON] Engine startup FAILED:\n{tb}")
                _engine = None
                _brain  = None
                _safe_emit("log", {"msg": f"❌ Engine startup failed: {_start_exc}"})
                for line in tb.splitlines():
                    _safe_emit("log", {"msg": line})
            finally:
                _engine_starting = False

        eventlet.spawn(_do_start)
        emit("log", {"msg": "⏳ Engine initializing…"})
        return  # handler returns immediately; _do_start runs in background
    except Exception as _outer_exc:
        print(f"[AXON] Unexpected outer error in start handler: {_outer_exc}")
        _engine_starting = False

@socketio.on("stop_engine")
def on_stop():
    global _engine, _brain
    if _brain:
        _brain.save_brain("autosave")
        emit("log", {"msg": "💾 Brain autosaved."})
    if _engine:
        _engine.stop()
        _engine = None
        _brain  = None
    emit("log", {"msg": "AXON stopped."})

@socketio.on("set_personality")
def on_set_personality(data):
    if not _brain:
        return
    result = _brain.set_personality(data.get("traits", {}))
    emit("personality_update", _sanitize({"traits": result.get("traits", {})}))

@socketio.on("run_autonomous")
def on_run_autonomous(data):
    if not _brain:
        return
    steps  = int(data.get("steps", 100))
    result = _brain.run_autonomous(steps=steps)
    emit("log", {"msg": f"🧠 Autonomous run started ({steps} steps)"})

# ── Observe Mode: continuous autonomous thinking loop ──────────────────────
_observe_active = False

@socketio.on("observe_mode")
def on_observe_mode(data):
    global _observe_active
    enabled = bool(data.get("enabled", False))
    _observe_active = enabled
    if not _brain:
        return
    if enabled:
        emit("log", {"msg": "👁 Observe Mode ON — AXON is thinking autonomously"})
        _start_observe_loop()
    else:
        emit("log", {"msg": "⚡ Train Mode — awaiting your input"})

def _start_observe_loop():
    import time
    def _loop():
        global _observe_active
        while _observe_active and _brain:
            try:
                _brain.run_autonomous(steps=20, interval_ms=80)
            except Exception:
                pass
            time.sleep(2.0)  # pause between bursts so UI can breathe
    import threading
    threading.Thread(target=_loop, daemon=True).start()

@socketio.on("get_explanation")
def on_get_explanation(_data=None):
    if not _brain:
        return
    exp = _brain.explain_last_decision()
    emit("decision_explanation", _sanitize(exp))

@socketio.on("save_brain")
def on_save_brain(data=None):
    if not _brain:
        return
    slot   = (data or {}).get("slot", "default")
    result = _brain.save_brain(slot)
    emit("log", {"msg": f"💾 Brain saved to slot '{slot}' — {result.get('beliefs',0)} beliefs"})
    emit("brain_saved", _sanitize(result))

@socketio.on("load_brain")
def on_load_brain(data=None):
    if not _brain:
        return
    slot   = (data or {}).get("slot", "default")
    result = _brain.load_brain(slot)
    emit("log", {"msg": f"📂 Brain restored from '{slot}'"})
    emit("brain_loaded", _sanitize(result))


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


@socketio.on("get_people")
def on_get_people():
    global _engine
    if not _engine or not _engine.running:
        emit("people_list", {"people": []})
        return
    try:
        summary = _engine.face_id.get_summary()
        emit("people_list", {"people": summary.get("all", [])})
    except Exception as e:
        emit("people_list", {"people": [], "error": str(e)})


@socketio.on("name_person")
def on_name_person(data):
    global _engine
    if not _engine:
        return
    pid  = data.get("person_id")
    name = data.get("name", "").strip()
    if pid and name:
        try:
            _engine.face_id.name_person(pid, name)
            from flask_socketio import emit as _emit
            emit("person_named", {"person_id": pid, "name": name}, broadcast=True)
        except Exception as e:
            print(f"  [App] name_person error: {e}")


@socketio.on("forget_person")
def on_forget_person(data):
    global _engine
    if not _engine:
        return
    pid = data.get("person_id")
    if pid:
        try:
            _engine.face_id.forget_person(pid)
            emit("person_forgotten", {"person_id": pid}, broadcast=True)
            print(f"  [App] Forgot person {pid}")
        except Exception as e:
            print(f"  [App] forget_person error: {e}")


@socketio.on("add_note_to_person")
def on_add_note_to_person(data):
    global _engine
    if not _engine:
        return
    pid  = data.get("person_id")
    note = data.get("note", "").strip()
    if pid and note:
        try:
            _engine.face_id.add_note(pid, note)
            # Also save a semantic memory about this person
            p = _engine.face_id.get_person(pid)
            name = p.get("name", "someone") if p else "someone"
            if _engine.memory:
                _engine.memory.store_semantic(f"Note about {name}: {note}", "people", confidence=0.9)
            emit("note_saved", {"person_id": pid}, broadcast=True)
        except Exception as e:
            print(f"  [App] add_note error: {e}")


@socketio.on("get_voice_config")
def on_get_voice_config():
    global _engine
    if not _engine or not _engine.voice:
        from axon.cognition.voice_output import VOICE_CATALOGUE, DEFAULT_VOICE, DEFAULT_RATE, DEFAULT_PITCH
        emit("voice_config", {"voice": DEFAULT_VOICE, "rate": DEFAULT_RATE,
                               "pitch": DEFAULT_PITCH, "catalogue": VOICE_CATALOGUE})
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

# ── Identity / Knowledge socket handlers ─────────────────────────────────────

@socketio.on("get_identity")
def on_get_identity():
    global _engine
    if not _engine:
        emit("identity_state", {"error": "Engine not running"})
        return
    try:
        identity = _engine.get_identity_summary()
        emit("identity_state", identity)
    except Exception as e:
        emit("identity_state", {"error": str(e)})

@socketio.on("get_beliefs")
def on_get_beliefs():
    global _engine
    if not _engine:
        emit("beliefs_state", {"beliefs": []})
        return
    try:
        emit("beliefs_state", {"beliefs": _engine.beliefs.all_beliefs()})
    except Exception as e:
        emit("beliefs_state", {"beliefs": [], "error": str(e)})

@socketio.on("ingest_text")
def on_ingest_text(data):
    global _engine
    if not _engine:
        emit("ingest_result", {"ok": False, "error": "Engine not running"})
        return
    text       = data.get("text", "").strip()
    source     = data.get("source", "manual")
    credibility= float(data.get("credibility", 0.6))
    if not text:
        emit("ingest_result", {"ok": False, "error": "No text provided"})
        return
    try:
        result = _engine.ingest_knowledge(text, source=source, credibility=credibility)
        emit("ingest_result", {"ok": True, **result})
        # Refresh identity state for UI
        emit("identity_state", _engine.get_identity_summary())
    except Exception as e:
        emit("ingest_result", {"ok": False, "error": str(e)})

# ── File upload endpoint ──────────────────────────────────────────────────────

@app.route("/upload_knowledge", methods=["POST"])
def upload_knowledge():
    global _engine
    if not _engine:
        return jsonify({"ok": False, "error": "Engine not running"}), 400

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "No file provided"}), 400

    filename  = file.filename.lower()
    credibility = float(request.form.get("credibility", 0.6))
    source    = request.form.get("source", file.filename)

    # Save to temp
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        text = _extract_text_from_file(tmp_path, filename)
    finally:
        try: os.unlink(tmp_path)
        except: pass

    if not text or not text.strip():
        return jsonify({"ok": False, "error": "Could not extract text from file"}), 400

    try:
        result = _engine.ingest_knowledge(text, source=source, credibility=credibility)
        return jsonify({"ok": True, "chars": len(text), **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _extract_text_from_file(path: str, filename: str) -> str:
    """Extract plain text from PDF, DOCX, DOC, TXT, MD files."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            try:
                import PyPDF2
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                raise RuntimeError("PDF reading requires: pip install pdfplumber  or  pip install PyPDF2")

    elif ext in (".docx",):
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            raise RuntimeError("DOCX reading requires: pip install python-docx")

    elif ext in (".doc",):
        # Try antiword or textract
        try:
            import subprocess
            result = subprocess.run(["antiword", path], capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        try:
            import textract
            return textract.process(path).decode("utf-8", errors="ignore")
        except ImportError:
            raise RuntimeError(".doc requires antiword (system package) or: pip install textract")

    elif ext in (".txt", ".md", ".rst", ".csv"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif ext in (".epub",):
        try:
            import ebooklib
            from ebooklib import epub
            from html.parser import HTMLParser
            class _P(HTMLParser):
                def __init__(self): super().__init__(); self.parts=[]
                def handle_data(self, d): self.parts.append(d)
            book = epub.read_epub(path)
            parts = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                p = _P(); p.feed(item.get_content().decode("utf-8","ignore")); parts.append(" ".join(p.parts))
            return "\n".join(parts)
        except ImportError:
            raise RuntimeError("EPUB reading requires: pip install EbookLib")
    else:
        raise RuntimeError(f"Unsupported file type: {ext}")



# ── Public Brain API endpoints ───────────────────────────────────────────────

@app.route("/api/brain/state")
def api_brain_state():
    if not _brain:
        return jsonify({"ok": False, "error": "Engine not running"})
    return jsonify(_sanitize(_brain.get_state()))

@app.route("/api/brain/explain")
def api_brain_explain():
    if not _brain:
        return jsonify({"ok": False, "error": "Engine not running"})
    return jsonify(_sanitize(_brain.explain_last_decision()))

@app.route("/api/brain/personality", methods=["GET","POST"])
def api_personality():
    if not _brain:
        return jsonify({"ok": False, "error": "Engine not running"})
    if request.method == "POST":
        traits = request.json or {}
        return jsonify(_sanitize(_brain.set_personality(traits)))
    return jsonify({"ok": True, "traits": _sanitize(_brain.get_personality())})

@app.route("/api/brain/save", methods=["POST"])
def api_brain_save():
    if not _brain:
        return jsonify({"ok": False, "error": "Engine not running"})
    slot = (request.json or {}).get("slot", "default")
    return jsonify(_sanitize(_brain.save_brain(slot)))

@app.route("/api/brain/load", methods=["POST"])
def api_brain_load():
    if not _brain:
        return jsonify({"ok": False, "error": "Engine not running"})
    slot = (request.json or {}).get("slot", "default")
    return jsonify(_sanitize(_brain.load_brain(slot)))

@app.route("/api/brain/snapshots")
def api_brain_snapshots():
    if not _brain:
        return jsonify([])
    return jsonify(_brain.list_snapshots())

@app.route("/api/brain/autonomous", methods=["POST"])
def api_brain_autonomous():
    if not _brain:
        return jsonify({"ok": False, "error": "Engine not running"})
    data  = request.json or {}
    steps = int(data.get("steps", 100))
    return jsonify(_sanitize(_brain.run_autonomous(steps=steps)))

@app.route("/api/brain/ingest", methods=["POST"])
def api_brain_ingest():
    if not _brain:
        return jsonify({"ok": False, "error": "Engine not running"})
    data = request.json or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"ok": False, "error": "text required"})
    return jsonify(_sanitize(_brain.ingest(text, source=data.get("source","api"))))


# ─── Onboarding API ──────────────────────────────────────────────────────────

@app.route("/api/onboarding_check")
def api_onboarding_check():
    """Lightweight check that works before engine starts — reads onboarding.json directly."""
    import json as _json
    from pathlib import Path as _Path
    ob_path = _Path("data/onboarding.json")
    if ob_path.exists():
        try:
            state = (_json.loads(ob_path.read_text()) if ob_path.read_text().strip() else {})
            return jsonify({"completed": bool(state.get("completed", False))})
        except Exception:
            pass
    return jsonify({"completed": False})

@app.route("/api/onboarding")
def api_onboarding_state():
    # If engine not ready yet, read directly from disk so the overlay can show
    if not _brain:
        import json as _json
        from pathlib import Path as _Path
        from axon.cognition.onboarding import PRESETS, SAMPLE_TOPICS
        ob_path = _Path("data/onboarding.json")
        state = {}
        if ob_path.exists():
            try:
                state = (_json.loads(ob_path.read_text()) if ob_path.read_text().strip() else {})
            except Exception:
                pass
        return jsonify({
            "completed": bool(state.get("completed", False)),
            "step":      state.get("step", 0),
            "ai_name":   state.get("ai_name", ""),
            "preset":    state.get("preset", ""),
            "sample_id": state.get("sample_id", ""),
            "presets":   {k: {"description": v["description"]} for k, v in PRESETS.items()},
            "samples":   [{"id": s["id"], "label": s["label"]} for s in SAMPLE_TOPICS],
        })
    return jsonify(_sanitize(_brain.get_onboarding_state()))

@app.route("/api/onboarding/name", methods=["POST"])
def api_onboarding_name():
    data = request.json or {}
    name = data.get("name", "")
    if _brain:
        return jsonify(_sanitize(_brain.onboarding_set_name(name)))
    # Pre-engine: persist directly to disk
    import json as _json; from pathlib import Path as _Path
    ob_path = _Path("data/onboarding.json")
    ob_path.parent.mkdir(exist_ok=True)
    state = (_json.loads(ob_path.read_text()) if ob_path.read_text().strip() else {}) if ob_path.exists() else {}
    state["ai_name"] = name; state.setdefault("step", 1)
    ob_path.write_text(_json.dumps(state))
    return jsonify({"ok": True, "ai_name": name})

@app.route("/api/onboarding/preset", methods=["POST"])
def api_onboarding_preset():
    data = request.json or {}
    preset = data.get("preset", "")
    if _brain:
        return jsonify(_sanitize(_brain.onboarding_set_preset(preset)))
    # Pre-engine: persist directly to disk
    import json as _json; from pathlib import Path as _Path
    from axon.cognition.onboarding import PRESETS
    ob_path = _Path("data/onboarding.json")
    ob_path.parent.mkdir(exist_ok=True)
    state = (_json.loads(ob_path.read_text()) if ob_path.read_text().strip() else {}) if ob_path.exists() else {}
    state["preset"] = preset; state["step"] = 2
    ob_path.write_text(_json.dumps(state))
    preset_data = PRESETS.get(preset, {})
    return jsonify({"ok": True, "preset": preset, "traits": preset_data.get("traits", {})})

@app.route("/api/onboarding/ingest_sample", methods=["POST"])
def api_onboarding_ingest_sample():
    data = request.json or {}
    if _brain:
        return jsonify(_sanitize(_brain.onboarding_ingest_sample(data.get("sample_id",""))))
    # Pre-engine: save sample selection for post-start ingestion
    import json as _json; from pathlib import Path as _Path
    ob_path = _Path("data/onboarding.json")
    ob_path.parent.mkdir(exist_ok=True)
    state = (_json.loads(ob_path.read_text()) if ob_path.read_text().strip() else {}) if ob_path.exists() else {}
    state["sample_id"] = data.get("sample_id",""); state["step"] = 3
    ob_path.write_text(_json.dumps(state))
    return jsonify({"ok": True, "deferred": True})

@app.route("/api/onboarding/ingest_text", methods=["POST"])
def api_onboarding_ingest_text():
    data = request.json or {}
    if _brain:
        return jsonify(_sanitize(_brain.onboarding_ingest_text(data.get("text",""))))
    # Pre-engine: save custom text for post-start ingestion
    import json as _json; from pathlib import Path as _Path
    ob_path = _Path("data/onboarding.json")
    ob_path.parent.mkdir(exist_ok=True)
    state = (_json.loads(ob_path.read_text()) if ob_path.read_text().strip() else {}) if ob_path.exists() else {}
    state["custom_text"] = data.get("text","")[:4000]; state["step"] = 3
    ob_path.write_text(_json.dumps(state))
    return jsonify({"ok": True, "deferred": True})

@app.route("/api/onboarding/complete", methods=["POST"])
def api_onboarding_complete():
    if _brain:
        return jsonify(_sanitize(_brain.onboarding_complete()))
    # Pre-engine: mark completed on disk so auto-start fires on next page load
    import json as _json; from pathlib import Path as _Path
    ob_path = _Path("data/onboarding.json")
    ob_path.parent.mkdir(exist_ok=True)
    state = (_json.loads(ob_path.read_text()) if ob_path.read_text().strip() else {}) if ob_path.exists() else {}
    state["completed"] = True; state["step"] = 5
    ob_path.write_text(_json.dumps(state))
    return jsonify({"ok": True})

@app.route("/api/onboarding/reset", methods=["POST"])
def api_onboarding_reset():
    """Wipe onboarding state — wizard re-appears on next page load."""
    from pathlib import Path
    ob_path = Path("data/onboarding.json")
    if ob_path.exists():
        ob_path.unlink()
    return jsonify({"ok": True, "msg": "Onboarding reset — reload the page to run setup again."})

@app.route("/api/first_opinion", methods=["POST"])
def api_first_opinion():
    """Get AXON's first opinion after onboarding ingestion."""
    if not _engine:
        return jsonify({"opinion": "I've processed the information and I'm ready to discuss it."})
    try:
        data    = request.json or {}
        context = data.get("context", "")
        prompt  = (
            f"You just processed information about: {context}. "
            f"Based on what you've ingested and your personality, "
            f"give a single sentence expressing your genuine first take or opinion. "
            f"Be direct. Show your personality. Don't hedge excessively."
        )
        opinion = _engine.language.think(prompt)
        return jsonify({"opinion": opinion})
    except Exception as ex:
        return jsonify({"opinion": "Initial processing complete. I've formed a preliminary model."})

# ─── Goals API ────────────────────────────────────────────────────────────────

@app.route("/api/goals")
def api_goals():
    if not _brain:
        return jsonify([])
    return jsonify(_sanitize(_brain.get_goals()))

@app.route("/api/goals/add", methods=["POST"])
def api_add_goal():
    if not _brain:
        return jsonify({"ok": False})
    data = request.json or {}
    return jsonify(_sanitize(_brain.add_goal(
        data.get("name",""), data.get("description",""), float(data.get("priority",0.5))
    )))

@app.route("/api/goals/remove", methods=["POST"])
def api_remove_goal():
    if not _brain:
        return jsonify({"ok": False})
    data = request.json or {}
    return jsonify(_sanitize(_brain.remove_goal(data.get("name",""))))

# ─── Surprise Events API ─────────────────────────────────────────────────────

@app.route("/api/surprise_events")
def api_surprise_events():
    if not _brain:
        return jsonify([])
    return jsonify(_sanitize(_brain.recent_surprise_events()))

# ─── Brain Fork + Share API ───────────────────────────────────────────────────

@app.route("/api/fork_brain", methods=["POST"])
def api_fork_brain():
    if not _brain:
        return jsonify({"ok": False, "error": "Engine not running"})
    data     = request.json or {}
    fork_name = data.get("fork_name", "fork")
    overrides = data.get("trait_overrides")
    return jsonify(_sanitize(_brain.fork_brain(fork_name, overrides)))

@app.route("/api/list_forks")
def api_list_forks():
    if not _brain:
        return jsonify([])
    return jsonify(_sanitize(_brain.list_forks()))

@app.route("/api/share_brain", methods=["POST"])
def api_share_brain():
    if not _brain:
        return jsonify({"ok": False, "error": "Engine not running"})
    data  = request.json or {}
    slot  = data.get("slot", "default")
    label = data.get("label", "")
    return jsonify(_sanitize(_brain.generate_share_link(slot=slot, label=label)))


# ─── Time-Scale API ───────────────────────────────────────────────────────────

@app.route("/api/brain/speed", methods=["GET"])
def api_brain_speed_get():
    if not _engine or not hasattr(_engine, "cycle"):
        return jsonify({"speed_scale": 1.0, "tick_hz": 10.0})
    c = _engine.cycle
    return jsonify({
        "speed_scale": round(c.speed_scale, 3),
        "tick_hz":     round(c.tick_hz, 2),
        "label":       _speed_label(c.speed_scale),
    })

@app.route("/api/brain/speed", methods=["POST"])
def api_brain_speed_set():
    if not _engine or not hasattr(_engine, "cycle"):
        return jsonify({"ok": False, "error": "Engine not running"})
    data  = request.json or {}
    scale = float(data.get("speed_scale", 1.0))
    _engine.cycle.speed_scale = scale
    c = _engine.cycle
    socketio.emit("speed_changed", {
        "speed_scale": round(c.speed_scale, 3),
        "tick_hz":     round(c.tick_hz, 2),
        "label":       _speed_label(c.speed_scale),
    })
    return jsonify({"ok": True, "speed_scale": round(c.speed_scale, 3),
                    "tick_hz": round(c.tick_hz, 2), "label": _speed_label(c.speed_scale)})

def _speed_label(s: float) -> str:
    if s <= 0.15:  return "Dreaming"
    if s <= 0.4:   return "Slow"
    if s <= 0.8:   return "Relaxed"
    if s <= 1.3:   return "Normal"
    if s <= 2.5:   return "Alert"
    if s <= 5.0:   return "Focused"
    return "Hyperdrive"


# ─── Reflection Engine API ─────────────────────────────────────────────────

@app.route("/api/brain/reflections", methods=["GET"])
def api_brain_reflections():
    if not _engine or not hasattr(_engine, "reflection"):
        return jsonify({"reflections": []})
    n = int(request.args.get("n", 10))
    return jsonify({"reflections": _engine.reflection.recent(n)})

# ─── Narrative Threads API ────────────────────────────────────────────────

@app.route("/api/brain/narratives", methods=["GET"])
def api_brain_narratives():
    if not _engine or not hasattr(_engine, "narratives"):
        return jsonify({"narratives": [], "dominant": "", "flips": []})
    return jsonify({
        "dominant":  _engine.narratives.dominant(),
        "top":       _engine.narratives.top_narratives(5),
        "all":       _engine.narratives.all_narratives(),
        "flips":     _engine.narratives.recent_flips(5),
        "bias":      _engine.narratives.narrative_bias(),
    })

# ─── Memory Hierarchy API ─────────────────────────────────────────────────


@app.route("/api/brain/thought_competition", methods=["GET"])
def api_brain_thought_competition():
    """Recent thought competitions — the internal struggle before each output."""
    if not _engine or not hasattr(_engine, "thought_gen"):
        return jsonify({"competitions": []})
    n = int(request.args.get("n", 5))
    return jsonify({"competitions": _sanitize(_engine.thought_gen.recent_competitions(n))})


@app.route("/api/brain/memory", methods=["GET"])
def api_brain_memory():
    """Return top Hebbian pathways for the neural canvas visualizer."""
    if not _engine or not hasattr(_engine, "memory"):
        return jsonify({"pathways": []})
    try:
        n = int(request.args.get("n", 30))
        connections = _engine.memory.top_connections(n=n)
        pathways = [
            {"src": c["a"], "dst": c["b"], "weight": c["weight"]}
            for c in connections
        ]
        return jsonify({"pathways": pathways})
    except Exception as ex:
        return jsonify({"pathways": [], "error": str(ex)})


@app.route("/api/brain/memory_hierarchy", methods=["GET"])
def api_brain_memory_hierarchy():
    if not _engine or not hasattr(_engine, "mem_hierarchy"):
        return jsonify({"stats": {}, "tiers": {}})
    tier = request.args.get("tier", None)
    n    = int(request.args.get("n", 15))
    h    = _engine.mem_hierarchy
    stats = _sanitize(h.tier_stats())
    records = {}
    if tier and tier in ("episodic","semantic","value","identity"):
        records[tier] = _sanitize(h.recall(tier, n=n))
    else:
        for t in ("episodic","semantic","value","identity"):
            records[t] = _sanitize(h.recall(t, n=5))
    return jsonify({"stats": stats, "tiers": records})

@app.route("/api/brain/memory_hierarchy/store", methods=["POST"])
def api_brain_memory_hierarchy_store():
    if not _engine or not hasattr(_engine, "mem_hierarchy"):
        return jsonify({"ok": False, "error": "Engine not running"})
    data    = request.json or {}
    tier    = data.get("tier", "episodic")
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"ok": False, "error": "content required"})
    row_id = _engine.mem_hierarchy.store(
        tier     = tier,
        content  = content,
        salience = float(data.get("salience", 0.5)),
        valence  = float(data.get("valence",  0.0)),
        tags     = data.get("tags", []),
    )
    return jsonify({"ok": True, "id": row_id, "tier": tier})


@app.route("/api/brain/interests", methods=["GET"])
def api_brain_interests():
    """Return all interests, boredom state, and search history."""
    if not _engine:
        return jsonify({"ok": False, "error": "Engine not running"})
    result = {"ok": True}
    if hasattr(_engine, "interests") and _engine.interests:
        result["interests"] = _engine.interests.all_interests()
    if hasattr(_engine, "boredom") and _engine.boredom:
        result["boredom"] = _engine.boredom.to_dict()
    if hasattr(_engine, "explorer") and _engine.explorer:
        result["search_history"] = _engine.explorer.search_history()
    return jsonify(result)

@app.route("/api/brain/interests/add", methods=["POST"])
def api_brain_interests_add():
    """Manually seed an interest."""
    if not _engine or not hasattr(_engine, "interests"):
        return jsonify({"ok": False, "error": "Engine not running"})
    data = request.json or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "name required"})
    interest = _engine.interests.add_or_strengthen(
        name,
        reward=float(data.get("strength", 0.3)),
        novelty=0.5,
        source="manual",
    )
    if interest:
        _engine._emit("new_interest", {"name": name, "source": "manual", "strength": interest.strength})
    return jsonify({"ok": True, "name": name})

@app.route("/api/brain/interests/remove", methods=["POST"])
def api_brain_interests_remove():
    """Force-remove an interest."""
    if not _engine or not hasattr(_engine, "interests"):
        return jsonify({"ok": False, "error": "Engine not running"})
    data = request.json or {}
    name = (data.get("name", "")).strip().lower()
    with _engine.interests._lock:
        if name in _engine.interests._items:
            del _engine.interests._items[name]
            _engine.interests._delete(name)
    return jsonify({"ok": True, "removed": name})

@app.route("/api/brain/boredom", methods=["GET"])
def api_brain_boredom():
    if not _engine or not hasattr(_engine, "boredom"):
        return jsonify({"ok": False, "error": "Engine not running"})
    return jsonify({"ok": True, **_engine.boredom.to_dict()})

if __name__ == "__main__":
    import signal, sys, threading as _th

    # ── Pre-launch menu — runs SYNCHRONOUSLY before anything else ──────
    # The web server has NOT started yet at this point.
    if sys.stdin.isatty():
        try:
            from axon.launch_menu import run as _launch_menu
            _launch_menu()
        except Exception as _lm_err:
            pass  # never block startup

    # Menu is done. NOW set up the server + browser open.

    def _shutdown(sig=None, frame=None):
        print("\n  [AXON] Shutting down…")
        global _engine
        if _engine:
            try:
                _engine.stop()
            except Exception:
                pass
            _engine = None
        import threading
        t = threading.Thread(target=lambda: (__import__("time").sleep(1.5), os._exit(0)), daemon=True)
        t.start()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("\n  AXON — Emerging Intelligence")
    print("  Main UI:      http://localhost:7777")
    print("  Neural Monitor: http://localhost:7777/monitor\n")
    print("  Axon Non-Commercial License | Copyright (c) 2026 Jon Tibbetts")
    print("  Commercial use requires a license: jon@jontibbetts.com\n")
    print("  Press Ctrl+C to exit\n")

    # Open browser after a flat 2-second delay — socketio.run() binds in ~1s.
    # NO polling: polling hits /api/ready which Flask answers instantly at
    # import time, causing the browser to open before the menu.
    def _open_browser():
        import time as _t, webbrowser as _wb
        _t.sleep(2.0)          # give socketio.run() time to bind
        _wb.open_new_tab("http://localhost:7777")

    # Thread starts HERE — after menu has already returned — so it is
    # physically impossible for the browser to open before the user picks.
    _th.Thread(target=_open_browser, daemon=True).start()

    try:
        socketio.run(app, host="0.0.0.0", port=7777, debug=False)
    except KeyboardInterrupt:
        _shutdown()
    except Exception as _server_exc:
        import traceback
        print(f"\n  [AXON] Server crashed: {_server_exc}")
        traceback.print_exc()
        _shutdown()
