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
      set((state) => {
        const reward = d.temporal_reward?.mean ?? 0;
        const surprise = d.prediction_surprise ?? 0;
        const rh = [...state.rewardHistory.slice(-99), reward];
        const sh = [...state.surpriseHistory.slice(-99), surprise];

        const rHistory = { ...state.regionHistory };
        Object.entries(d.regions ?? {}).forEach(([k, v]) => {
          rHistory[k] = [...(rHistory[k] ?? []).slice(-49), v as number];
        });

        const nmHistory = { ...state.nmHistory };
        Object.entries(d.neuromod ?? {}).forEach(([k, v]) => {
          nmHistory[k] = [...(nmHistory[k] ?? []).slice(-49), v as number];
        });

        return {
          neuralState: d,
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
      set((state) => {
        const rHistory = { ...state.regionHistory };
        Object.entries(d.regions ?? {}).forEach(([k, v]) => {
          rHistory[k] = [...(rHistory[k] ?? []).slice(-49), v as number];
        });
        const nmHistory = { ...state.nmHistory };
        Object.entries(d.neuromod ?? {}).forEach(([k, v]) => {
          nmHistory[k] = [...(nmHistory[k] ?? []).slice(-49), v as number];
        });
        const reward   = d.temporal_reward?.mean ?? 0;
        const surprise = d.prediction_surprise ?? 0;
        return {
          neuralState: { ...state.neuralState, ...d },
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
      set({ thinking: d?.active ?? true });
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
