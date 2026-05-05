"""
AXON — LLM Provider Registry
Supports: LM Studio (local), OpenAI, Anthropic Claude, Google Gemini, Groq
Config stored in data/providers.json — never committed to git.
"""

import os
import json
import pathlib
import urllib.request
import urllib.error
from typing import Optional

PROVIDERS_FILE = pathlib.Path("data/providers.json")

DEFAULT_MODELS = {
    "openai":    "gpt-4o",
    "anthropic": "claude-opus-4-5",
    "gemini":    "gemini-2.0-flash",
    "groq":      "llama3-70b-8192",
    "lmstudio":  None,   # auto-detected
}

ALL_MODELS = {
    "openai":    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    "anthropic": ["claude-opus-4-5", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
    "gemini":    ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    "groq":      ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
    "lmstudio":  [],  # populated at runtime
}


def load_config() -> dict:
    """Load provider config from disk, falling back to env vars + defaults."""
    cfg = {
        "active_provider": "lmstudio",
        "prefer_local":    True,
        "lmstudio_url":    "http://localhost:1234",
        "lmstudio_model":  None,
        "openai_key":      os.getenv("OPENAI_API_KEY", ""),
        "openai_model":    DEFAULT_MODELS["openai"],
        "anthropic_key":   os.getenv("ANTHROPIC_API_KEY", ""),
        "anthropic_model": DEFAULT_MODELS["anthropic"],
        "gemini_key":      os.getenv("GEMINI_API_KEY", ""),
        "gemini_model":    DEFAULT_MODELS["gemini"],
        "groq_key":        os.getenv("GROQ_API_KEY", ""),
        "groq_model":      DEFAULT_MODELS["groq"],
    }
    if PROVIDERS_FILE.exists():
        try:
            saved = json.loads(PROVIDERS_FILE.read_text())
            cfg.update(saved)
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    PROVIDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Never write blank keys over env-var keys
    to_save = {k: v for k, v in cfg.items() if v not in (None, "")}
    # But always save structural fields even if empty
    for field in ("active_provider", "prefer_local", "lmstudio_url", "lmstudio_model",
                  "openai_model", "anthropic_model", "gemini_model", "groq_model"):
        to_save[field] = cfg.get(field)
    PROVIDERS_FILE.write_text(json.dumps(to_save, indent=2))


def provider_status(cfg: dict, lm_detected_model: Optional[str] = None) -> dict:
    """Return a status dict for the UI — which providers are configured."""
    return {
        "active_provider": cfg.get("active_provider", "lmstudio"),
        "prefer_local":    cfg.get("prefer_local", True),
        "lmstudio": {
            "available":      bool(lm_detected_model),
            "model":          lm_detected_model or cfg.get("lmstudio_model") or "none",
            "url":            cfg.get("lmstudio_url"),
        },
        "openai": {
            "configured":     bool(cfg.get("openai_key")),
            "model":          cfg.get("openai_model", DEFAULT_MODELS["openai"]),
            "available_models": ALL_MODELS["openai"],
        },
        "anthropic": {
            "configured":     bool(cfg.get("anthropic_key")),
            "model":          cfg.get("anthropic_model", DEFAULT_MODELS["anthropic"]),
            "available_models": ALL_MODELS["anthropic"],
        },
        "gemini": {
            "configured":     bool(cfg.get("gemini_key")),
            "model":          cfg.get("gemini_model", DEFAULT_MODELS["gemini"]),
            "available_models": ALL_MODELS["gemini"],
        },
        "groq": {
            "configured":     bool(cfg.get("groq_key")),
            "model":          cfg.get("groq_model", DEFAULT_MODELS["groq"]),
            "available_models": ALL_MODELS["groq"],
        },
    }
