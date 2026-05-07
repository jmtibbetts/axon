import { useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAxonStore } from '../store/axonStore';

const SERVER = window.location.origin;

const DEFAULT_START_CONFIG = {
  api_key:      '',
  lm_url:       'http://localhost:1234',
  lm_model:     null,
  prefer_local: true,
  voice:        true,
  camera:       true,
  mic:          true,
  mic_index:    null,
  camera_index: -1,
};

// ── Module-level singleton — one socket shared across ALL hook calls ──────
let _socket: Socket | null = null;
let _startFired  = false;
let _handlersSet = false;

function getSocket(): Socket {
  if (!_socket) {
    _socket = io(SERVER, {
      transports: ['websocket'],
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
    });
  }
  return _socket;
}

// ─────────────────────────────────────────────────────────────────────────
// Hook — any component can call this; only one socket is ever created
// ─────────────────────────────────────────────────────────────────────────
export function useAxonSocket() {
  const set = useAxonStore((s) => s.set);
  // Keep a stable ref so callers can access the socket synchronously
  const socketRef = useRef<Socket>(getSocket());

  useEffect(() => {
    const socket = getSocket();
    socketRef.current = socket;

    // Register ALL event handlers exactly once, globally
    if (_handlersSet) return;
    _handlersSet = true;

    socket.on('connect', () => {
      set({ connected: true });
      if (_startFired) return;
      _startFired = true;

      fetch('/api/status')
        .then((r) => r.json())
        .then((d) => {
          if (d.running) {
            set({ engineRunning: true });
          } else {
            fetch('/api/start_error')
              .then((r) => r.json())
              .then((err) => {
                if (err.error) console.error('[AXON] Last startup error:\n', err.error);
              }).catch(() => {});
            socket.emit('start_engine', DEFAULT_START_CONFIG);
          }
        })
        .catch(() => socket.emit('start_engine', DEFAULT_START_CONFIG));
    });

    socket.on('disconnect', () => {
      set({ connected: false });
      _startFired = false;
    });

    socket.on('brain_state', (d) => {
      const state = useAxonStore.getState();
      const prev  = state.neuralState;

      // ── Rolling history buffers ────────────────────────────────────────
      const rewardVal   = (d.temporal_reward as any)?.mean ?? 0;
      const surpriseVal = d.prediction_surprise ?? 0;

      const HIST = 60;
      const newReward   = [...(state.rewardHistory   ?? []), rewardVal  ].slice(-HIST);
      const newSurprise = [...(state.surpriseHistory ?? []), surpriseVal].slice(-HIST);

      // Region history
      const regions = (d.regions ?? {}) as Record<string, number>;
      const newRegionHist: Record<string, number[]> = {};
      for (const [k, v] of Object.entries(regions)) {
        const old = state.regionHistory[k] ?? [];
        newRegionHist[k] = [...old, v as number].slice(-HIST);
      }

      // Neuromod history
      const nm = (d.neuromod ?? {}) as Record<string, number>;
      const newNmHist: Record<string, number[]> = {};
      for (const [k, v] of Object.entries(nm)) {
        const old = state.nmHistory[k] ?? [];
        newNmHist[k] = [...old, v as number].slice(-HIST);
      }

      // Spike detection — regions that jumped >0.15
      const newSpikes: any[] = [];
      for (const [k, v] of Object.entries(regions)) {
        const prevAct = (prev.regions as any)?.[k] ?? 0;
        if ((v as number) - prevAct > 0.15) {
          newSpikes.push({ region: k, activation: v, ts: Date.now() });
        }
      }

      set((state) => ({
        neuralState:    { ...prev, ...d },
        lastTick:       d.tick ?? state.lastTick,
        rewardHistory:  newReward,
        surpriseHistory: newSurprise,
        regionHistory:  { ...state.regionHistory, ...newRegionHist },
        nmHistory:      { ...state.nmHistory, ...newNmHist },
        regionSpikes:   newSpikes.length > 0
          ? [...newSpikes, ...state.regionSpikes].slice(0, 40)
          : state.regionSpikes,
      }));
    });

    socket.on('thinking', (d) => {
      set({ thinking: !!d.state });
    });

    socket.on('response', (d) => {
      set((state) => ({
        messages: [...state.messages, { role: 'axon', text: d.text ?? d, ts: Date.now() }],
        thinking: false,
      }));
    });

    socket.on('chat_message', (d) => {
      set((state) => ({
        messages: [...state.messages, { role: 'axon', text: d.text ?? d, ts: Date.now() }],
      }));
    });

    socket.on('user_input_echo', (d) => {
      set((state) => ({
        messages: [...state.messages, { role: 'user', text: d.text ?? d, ts: Date.now() }],
      }));
    });

    socket.on('lm_status', (d) => {
      set({ lmStatus: d });
    });

    socket.on('vision_frame', (d) => {
      set({ visionFrame: d.frame ?? d });
    });

    socket.on('face_data', (d) => {
      set({ faceData: d });
    });

    socket.on('known_face', (d) => {
      set({ faceData: { ...d, ts: Date.now() } });
    });

    socket.on('hebbian_event', (d) => {
      set((state) => ({
        hebbianEvents: [{ ...d, ts: Date.now() }, ...state.hebbianEvents].slice(0, 30),
      }));
    });

    socket.on('memory_event', (d) => {
      set((state) => ({
        memoryEvents: [{ ...d, ts: Date.now() }, ...state.memoryEvents].slice(0, 50),
      }));
    });

    socket.on('log', (d) => {
      set((state) => ({
        logs: [{ ...d, ts: Date.now() }, ...state.logs].slice(0, 80),
      }));
    });

    socket.on('surprise_event', (d) => {
      set((state) => ({
        surpriseEvents: [{ ...d, ts: Date.now() }, ...state.surpriseEvents].slice(0, 30),
      }));
    });

    socket.on('reflection', (d) => {
      set((state) => ({
        reflections: [{ ...d, ts: Date.now() }, ...state.reflections].slice(0, 30),
      }));
    });

    socket.on('knowledge_ingested', (d) => {
      set((state) => ({
        ingestions: state.ingestions.map((i) =>
          i.filename === d.filename
            ? { ...i, status: d.ok ? 'done' : 'error', concepts: d.concepts, opinions: d.opinions, error: d.error }
            : i
        ),
      }));
    });

    socket.on('thought_competition', (d) => {
      set((state) => ({
        thoughtCompetition: [{ ...d, ts: Date.now() }, ...state.thoughtCompetition].slice(0, 20),
      }));
    });

    socket.on('profile_update', (d) => {
      set({ userProfile: d });
    });

    socket.on('voice_speaking', (d) => {
      set({ voiceSpeaking: !!d.speaking });
    });

    socket.on('audio_emotion', (d) => {
      set({ audioEmotion: d });
    });

    socket.on('mic_volume', (d) => {
      set({ micVolume: d.db ?? d.volume ?? 0 });
    });

    socket.on('ingestion_progress', (d) => {
      set((state) => ({
        ingestions: state.ingestions.map((i) =>
          i.filename === d.filename ? { ...i, ...d } : i
        ),
      }));
    });

    socket.on('provider_status', (d) => {
      set({ lmStatus: d });
    });

    socket.on('engine_started', () => {
      set({ engineRunning: true });
    });

    socket.on('engine_error', (d) => {
      set((state) => ({
        logs: [{ level: 'error', msg: `Engine error: ${d.error ?? d}`, ts: Date.now() }, ...state.logs],
      }));
    });

    socket.on('autonomous_mode', (d) => {
      set({ autonomousMode: !!d.active });
    });

    socket.on('open_diagnostic', () => {
      // Could open a modal; for now just log
      set((state) => ({
        logs: [{ msg: '🔬 Diagnostic panel requested', ts: Date.now() }, ...state.logs],
      }));
    });

    // No cleanup — singleton socket lives for the page lifetime
  }, []);

  // ── Public API ────────────────────────────────────────────────────────
  const send = (text: string) => {
    const socket = getSocket();
    socket.emit('chat', { text });
    // Optimistic local echo
    useAxonStore.getState().set((state) => ({
      messages: [...state.messages, { role: 'user', text, ts: Date.now() }],
    }));
  };

  const emit = (event: string, data?: any) => {
    getSocket().emit(event, data);
  };

  return { socket: socketRef.current, send, emit };
}
