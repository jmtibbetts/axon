import { useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAxonStore } from '../store/axonStore';

const SERVER = window.location.origin;

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
    });

    socket.on('disconnect', () => {
      set({ connected: false });
    });

    // Full brain state (rich tick)
    socket.on('brain_state', (d) => {
      set((state) => {
        const reward = d.cognitive_state?.temporal_reward?.mean ?? d.temporal_reward?.mean ?? 0;
        const surprise = d.prediction_surprise ?? 0;
        const rh = [...(state.rewardHistory.slice(-99)), reward];
        const sh = [...(state.surpriseHistory.slice(-99)), surprise];
        // track top-5 region history
        const rHistory = { ...state.regionHistory };
        const regions = d.regions ?? {};
        Object.entries(regions).forEach(([k, v]) => {
          rHistory[k] = [...((rHistory[k] ?? []).slice(-49)), v as number];
        });
        const nmHistory = { ...state.nmHistory };
        const nm = d.neuromod ?? {};
        Object.entries(nm).forEach(([k, v]) => {
          nmHistory[k] = [...((nmHistory[k] ?? []).slice(-49)), v as number];
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

    // Lightweight neural snapshot
    socket.on('neural_state', (d) => {
      set({ neuralState: d, lastTick: Date.now() });
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
      set({ visionFrame: d.image ?? null });
    });

    socket.on('face', (d) => {
      set({ faceData: d });
    });

    socket.on('known_face', (d) => {
      set({ faceData: { ...d, known: true } });
    });

    socket.on('hebbian_event', (d) => {
      set((state) => ({ hebbianEvents: [{ ...d, ts: Date.now() }, ...state.hebbianEvents].slice(0, 50) }));
    });

    socket.on('memory_event', (d) => {
      set((state) => ({ memoryEvents: [{ ...d, ts: Date.now() }, ...state.memoryEvents].slice(0, 50) }));
    });

    socket.on('region_spike', (d) => {
      set((state) => ({ regionSpikes: [d, ...state.regionSpikes].slice(0, 100) }));
    });

    socket.on('log', (d) => {
      set((state) => ({ logs: [d, ...state.logs].slice(0, 200) }));
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

    return () => { socket.disconnect(); };
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
