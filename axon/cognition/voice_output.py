"""
AXON — Voice Output
Text-to-speech using edge-tts (Microsoft Neural TTS, offline-capable).
Runs async in a background thread with a queue so responses don't block.
"""

import asyncio
import threading
import queue
import os
import time
import tempfile

try:
    import edge_tts
    EDGE_TTS_OK = True
except ImportError:
    EDGE_TTS_OK = False

try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False


# Neural-sounding voices — slightly slower, more thoughtful
VOICE      = "en-US-AriaNeural"
RATE       = "-5%"    # slightly slower = more deliberate
PITCH      = "-3Hz"   # slightly lower = calmer

class VoiceOutput:
    def __init__(self):
        self._queue   = queue.Queue()
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self.enabled  = EDGE_TTS_OK and PYGAME_OK
        self.speaking = False
        print(f"  [Voice] edge-tts={EDGE_TTS_OK}, pygame={PYGAME_OK}, enabled={self.enabled}")

    def speak(self, text: str, interrupt: bool = False):
        if not self.enabled:
            return
        # Strip internal monologue markers
        text = text.replace("...", " ").strip()
        if not text:
            return
        if interrupt:
            # Clear queue
            while not self._queue.empty():
                try: self._queue.get_nowait()
                except: break
        self._queue.put(text)

    def _worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            text = self._queue.get()
            if text is None:
                break
            self.speaking = True
            try:
                loop.run_until_complete(self._synthesize(text))
            except Exception as e:
                print(f"  [Voice] TTS error: {e}")
            self.speaking = False

    async def _synthesize(self, text: str):
        if not EDGE_TTS_OK:
            return
        communicate = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        try:
            await communicate.save(tmp.name)
            if PYGAME_OK and os.path.exists(tmp.name):
                pygame.mixer.music.load(tmp.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.05)
        finally:
            try:
                os.unlink(tmp.name)
            except:
                pass

    def stop(self):
        if PYGAME_OK:
            try: pygame.mixer.music.stop()
            except: pass
        self._queue.put(None)

    def get_status(self) -> dict:
        return {"enabled": self.enabled, "speaking": self.speaking,
                "voice": VOICE, "queue": self._queue.qsize()}
