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

// Module-level guard so React StrictMode double-mount doesn't double-start
let _globalStartFired = false;

export function useAxonSocket() {
  const socketRef = useRef<Socket | null>(null);
  const set = useAxonStore((s) => s.set);

  useEffect(() => {
    const socket = io(SERVER, {
      transports: ['websocket'],
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
    });
    socketRef.current = socket;

    socket.on('connect', () => {
      set({ connected: true });

      if (_globalStartFired) return;
      _globalStartFired = true;

      fetch('/api/status')
        .then((r) => r.json())
        .then((d) => {
          if (d.running) {
            set({ engineRunning: true });
          } else {
            // Check if there's a previous startup error
            fetch('/api/start_error')
              .then((r) => r.json())
              .then((err) => {
                if (err.error) {
                  console.error('[AXON] Last startup error:\n', err.error);
                  set((state) => ({
                    logs: [{ msg: '⚠️ Previous startup failed — retrying...', ts: Date.now() }, ...state.logs],
                  }));
                }
              }).catch(() => {});
            socket.emit('start_engine', DEFAULT_START_CONFIG);
          }
        })
        .catch(() => {
          socket.emit('start_engine', DEFAULT_START_CONFIG);
        });
    });

    socket.on('disconnect', () => {
      set({ connected: false });
      _globalStartFired = false; // allow restart on reconnect
    });

    socket.on('brain_state', (d) => {
      // Apply same normalization as neural_state
      const normalized = { ...d };
      if (normalized.emotion && normalized.emotion.emotion !== undefined && normalized.emotion.current === undefined) {
        normalized.emotion = { ...normalized.emotion, current: normalized.emotion.emotion };
      }
      if (normalized.conflict && normalized.conflict.score === undefined) {
        normalized.conflict = { ...normalized.conflict, score: normalized.conflict.dominance_mean ?? 0, winner_set: normalized.conflict.top_dominant ?? [] };
      }
      set((state) => {
        const reward = normalized.temporal_reward?.mean ?? 0;
        const surprise = normalized.prediction_surprise ?? 0;
        const rh = [...state.rewardHistory.slice(-99), reward];
        const sh = [...state.surpriseHistory.slice(-99), surprise];
        const rHistory = { ...state.regionHistory };
        Object.entries(normalized.regions ?? {}).forEach(([k, v]) => {
          rHistory[k] = [...(rHistory[k] ?? []).slice(-49), v as number];
        });
        const nmHistory = { ...state.nmHistory };
        Object.entries(normalized.neuromod ?? {}).forEach(([k, v]) => {
          nmHistory[k] = [...(nmHistory[k] ?? []).slice(-49), v as number];
        });
        return {
          neuralState: normalized,
          lastTick: Date.now(),
          engineRunning: true,
          rewardHistory: rh,
          surpriseHistory: sh,
          regionHistory: rHistory,
          nmHistory: nmHistory,
        };
      });
    });

    socket.on('neural_state', (d) => {
      // Normalize field name mismatches between Python backend and frontend store
      const normalized = { ...d };
      // emotion: backend sends {emotion: "calm"}, frontend expects {current: "calm"}
      if (normalized.emotion && normalized.emotion.emotion !== undefined && normalized.emotion.current === undefined) {
        normalized.emotion = { ...normalized.emotion, current: normalized.emotion.emotion };
      }
      // conflict: backend sends {dominance_mean: 0.5}, frontend expects {score: 0.5}
      if (normalized.conflict && normalized.conflict.score === undefined) {
        normalized.conflict = {
          ...normalized.conflict,
          score: normalized.conflict.dominance_mean ?? 0,
          dominant: (normalized.conflict.top_dominant ?? [])[0] ?? '',
          winner_set: normalized.conflict.top_dominant ?? [],
        };
      }
      // top_routes: if empty, synthesize from top active regions
      if ((!normalized.top_routes || normalized.top_routes.length === 0) && normalized.top_clusters && normalized.top_clusters.length >= 2) {
        const top = (normalized.top_clusters as any[]).slice(0, 4);
        normalized.top_routes = top.slice(0, 3).map((src: any, i: number) => {
          const dst = top[(i + 1) % top.length];
          return {
            src: src.name, dst: dst.name,
            src_region: src.region, dst_region: dst.region,
            weight: (src.activation + dst.activation) / 2,
          };
        });
      }

      set((state) => {
        const rHistory = { ...state.regionHistory };
        Object.entries(normalized.regions ?? {}).forEach(([k, v]) => {
          rHistory[k] = [...(rHistory[k] ?? []).slice(-49), v as number];
        });
        const nmHistory = { ...state.nmHistory };
        Object.entries(normalized.neuromod ?? {}).forEach(([k, v]) => {
          nmHistory[k] = [...(nmHistory[k] ?? []).slice(-49), v as number];
        });
        const reward   = normalized.temporal_reward?.mean ?? 0;
        const surprise = normalized.prediction_surprise ?? 0;
        return {
          neuralState: { ...state.neuralState, ...normalized },
          lastTick: Date.now(),
          engineRunning: true,
          regionHistory: rHistory,
          nmHistory: nmHistory,
          rewardHistory:  [...state.rewardHistory.slice(-99), reward],
          surpriseHistory:[...state.surpriseHistory.slice(-99), surprise],
        };
      });
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

    socket.on('thinking', (d) => {
      set({ thinking: d?.state ?? d?.active ?? true });
    });

    socket.on('transcript', (d) => {
      set((state) => ({
        messages: [...state.messages, { role: 'user', text: d.text ?? d, ts: Date.now() }],
      }));
    });

    socket.on('lm_status', (d) => {
      set({ lmStatus: d });
    });

    socket.on('frame', (d) => {
      set({ visionFrame: d.frame_b64 ?? d.image ?? null });
    });

    socket.on('face', (d) => {
      set({ faceData: d });
    });

    socket.on('known_face', (d) => {
      set({ faceData: { ...d, known: true } });
    });

    socket.on('hebbian_event', (d) => {
      set((state) => ({
        hebbianEvents: [{ ...d, ts: Date.now() }, ...state.hebbianEvents].slice(0, 50),
      }));
    });

    socket.on('memory_event', (d) => {
      set((state) => ({
        memoryEvents: [{ ...d, ts: Date.now() }, ...state.memoryEvents].slice(0, 50),
      }));
    });

    socket.on('region_spike', (d) => {
      set((state) => ({
        regionSpikes: [d, ...state.regionSpikes].slice(0, 100),
      }));
    });

    socket.on('log', (d) => {
      const msg = typeof d === 'string' ? d : d.msg ?? JSON.stringify(d);
      set((state) => ({
        logs: [{ msg, ts: Date.now() }, ...state.logs].slice(0, 200),
      }));
      if (msg.includes('Engine startup failed') || msg.includes('startup FAILED')) {
        set({ engineRunning: false });
        console.error('[AXON startup]', msg);
      }
    });

    socket.on('thought_competition', (d) => {
      set((state) => ({
        thoughtCompetition: [{ ...d, ts: Date.now() }, ...state.thoughtCompetition].slice(0, 20),
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

    socket.on('prediction_error', (d) => {
      set((state) => ({
        surpriseEvents: [{ type: 'prediction_error', ...d, ts: Date.now() }, ...state.surpriseEvents].slice(0, 30),
      }));
    });

    socket.on('voice_speaking', (d) => {
      set({ voiceSpeaking: d?.speaking ?? false });
    });

    socket.on('audio_emotion', (d) => {
      set({ audioEmotion: d ?? null });
    });

    socket.on('mic_volume', (d) => {
      set({ micVolume: d?.db ?? 0 });
    });

    socket.on('profile_update', (d) => {
      set({ userProfile: d });
    });

    socket.on('new_hobby', (d) => {
      set((state) => ({
        logs: [{ msg: `🎮 New hobby cluster: ${d?.cluster ?? ''}`, ts: Date.now() }, ...state.logs].slice(0, 200),
      }));
    });

    socket.on('person_named', (d) => {
      set((state) => ({
        logs: [{ msg: `👤 Face named: ${d?.name ?? d?.person_id ?? '?'}`, ts: Date.now() }, ...state.logs].slice(0, 200),
      }));
    });

    socket.on('new_face', (_d) => {
      set((state) => ({
        logs: [{ msg: `👤 New face detected`, ts: Date.now() }, ...state.logs].slice(0, 200),
      }));
    });

    socket.on('open_diagnostic', () => {
      set((state) => ({
        logs: [{ msg: `🔬 Diagnostic mode triggered`, ts: Date.now() }, ...state.logs].slice(0, 200),
      }));
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const send = (text: string) => {
    socketRef.current?.emit('chat', { text });
    set((state) => ({
      messages: [...state.messages, { role: 'user', text, ts: Date.now() }],
    }));
  };

  const emit = (event: string, data?: any) => {
    socketRef.current?.emit(event, data);
  };

  return { socket: socketRef.current, send, emit };
}
