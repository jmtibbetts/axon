"""
AXON — Speech Output
edge-tts → mp3 → pygame playback (works cross-platform, no system tools needed).
"""

import asyncio
import threading
import tempfile
import os
import time


class SpeechSystem:
    def __init__(self, on_speaking=None, on_done=None,
                 voice="en-US-AriaNeural", rate="+5%", pitch="+0Hz"):
        self.voice       = voice
        self.rate        = rate
        self.pitch       = pitch
        self.on_speaking = on_speaking
        self.on_done     = on_done
        self.speaking    = False
        self._queue      = []
        self._lock       = threading.Lock()
        self._pygame_ok  = False
        self._init_pygame()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _init_pygame(self):
        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self._pygame_ok = True
        except Exception as e:
            print(f"  [Speech] pygame not available: {e}")

    def say(self, text: str, priority: bool = False):
        with self._lock:
            if priority:
                self._queue.insert(0, text)
            else:
                self._queue.append(text)

    def _worker(self):
        while True:
            text = None
            with self._lock:
                if self._queue:
                    text = self._queue.pop(0)
            if text:
                self._speak_sync(text)
            else:
                time.sleep(0.05)

    def _speak_sync(self, text: str):
        self.speaking = True
        if self.on_speaking:
            self.on_speaking(text)
        try:
            asyncio.run(self._speak_async(text))
        except Exception as e:
            print(f"  [Speech] Error: {e}")
        self.speaking = False
        if self.on_done:
            self.on_done()

    async def _speak_async(self, text: str):
        try:
            import edge_tts
        except ImportError:
            print("  [Speech] edge-tts not installed.")
            return

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            tmp = f.name
        try:
            comm = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
            await comm.save(tmp)
            self._play_file(tmp)
        finally:
            try: os.unlink(tmp)
            except: pass

    def _play_file(self, path: str):
        if not self._pygame_ok:
            return
        try:
            import pygame
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        except Exception as e:
            print(f"  [Speech] Playback error: {e}")

    def get_status(self):
        return {"speaking": self.speaking, "queued": len(self._queue), "voice": self.voice}
