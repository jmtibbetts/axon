"""
AXON — Voice Output
Text-to-speech using edge-tts (Microsoft Neural TTS).
Playback chain: pygame -> winsound (Windows) -> playsound -> skip
Runs async in background thread with a queue so responses don't block.
"""

import asyncio
import threading
import queue
import os
import sys
import time
import tempfile

# ── TTS engine ───────────────────────────────────────────────
try:
    import edge_tts
    EDGE_TTS_OK = True
except ImportError:
    EDGE_TTS_OK = False
    print("  [Voice] edge-tts not installed — pip install edge-tts")

# ── Audio playback: try pygame first, fall back to winsound/playsound ──
PLAYBACK = None

try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=2048)
    PLAYBACK = "pygame"
    print("  [Voice] Playback: pygame")
except Exception as e:
    print(f"  [Voice] pygame init failed: {e}")
    if sys.platform == "win32":
        try:
            import winsound
            PLAYBACK = "winsound"
            print("  [Voice] Playback: winsound")
        except ImportError:
            pass
    if PLAYBACK is None:
        try:
            import playsound
            PLAYBACK = "playsound"
            print("  [Voice] Playback: playsound")
        except ImportError:
            print("  [Voice] No audio playback available — voice output disabled")

# ── Voice config ─────────────────────────────────────────────
VOICE = "en-US-AriaNeural"
RATE  = "-5%"
PITCH = "-3Hz"


class VoiceOutput:
    def __init__(self):
        self._queue  = queue.Queue()
        self.enabled = EDGE_TTS_OK and (PLAYBACK is not None)
        self.speaking = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"  [Voice] edge-tts={EDGE_TTS_OK}, playback={PLAYBACK}, enabled={self.enabled}")

    def speak(self, text: str, interrupt: bool = False):
        if not self.enabled:
            return
        text = text.strip()
        if not text:
            return
        if interrupt:
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
            if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) < 100:
                print("  [Voice] TTS generated empty file")
                return
            await self._play(tmp.name)
        except Exception as e:
            print(f"  [Voice] Synth/play error: {e}")
        finally:
            await asyncio.sleep(0.1)
            try:
                os.unlink(tmp.name)
            except:
                pass

    async def _play(self, path: str):
        if PLAYBACK == "pygame":
            try:
                # Re-init mixer if it died
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=2048)
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.05)
            except Exception as e:
                print(f"  [Voice] pygame playback error: {e}")
                # Try winsound fallback
                await self._play_winsound(path)

        elif PLAYBACK == "winsound":
            await self._play_winsound(path)

        elif PLAYBACK == "playsound":
            import playsound as ps
            await asyncio.get_event_loop().run_in_executor(None, ps.playsound, path)

    async def _play_winsound(self, mp3_path: str):
        """winsound only plays WAV — convert mp3→wav via pydub or ffmpeg."""
        wav_path = mp3_path.replace(".mp3", ".wav")
        converted = False
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(mp3_path)
            seg.export(wav_path, format="wav")
            converted = True
        except Exception:
            pass

        if not converted:
            # Try ffmpeg directly
            import subprocess
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_path, wav_path],
                capture_output=True
            )
            converted = r.returncode == 0

        if converted and os.path.exists(wav_path):
            import winsound
            try:
                winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            except Exception as e:
                print(f"  [Voice] winsound error: {e}")
            try:
                os.unlink(wav_path)
            except:
                pass
        else:
            print("  [Voice] mp3->wav conversion failed; no audio output")

    def stop(self):
        if PLAYBACK == "pygame":
            try:
                pygame.mixer.music.stop()
            except:
                pass
        self._queue.put(None)

    def get_status(self) -> dict:
        return {
            "enabled":  self.enabled,
            "speaking": self.speaking,
            "voice":    VOICE,
            "playback": PLAYBACK or "none",
            "queue":    self._queue.qsize(),
        }
