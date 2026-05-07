import { useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAxonStore } from '../store/axonStore';

const SERVER = window.location.origin; // same host as Flask (port 7777)

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
      console.log('[AXON] socket connected', socket.id);
      set({ connected: true });
    });

    socket.on('disconnect', () => {
      console.warn('[AXON] socket disconnected');
      set({ connected: false });
    });

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
      set({ thinking: d.active ?? true });
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

    socket.on('hebbian_event', (d) => {
      set((state) => ({ hebbianEvents: [d, ...state.hebbianEvents].slice(0, 20) }));
    });

    socket.on('memory_event', (d) => {
      set((state) => ({ memoryEvents: [d, ...state.memoryEvents].slice(0, 30) }));
    });

    socket.on('region_spike', (d) => {
      set((state) => ({ regionSpikes: [d, ...state.regionSpikes].slice(0, 50) }));
    });

    socket.on('log', (d) => {
      set((state) => ({ logs: [d, ...state.logs].slice(0, 100) }));
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const send = (text: string) => {
    socketRef.current?.emit('message', { text });
    set((state) => ({
      messages: [...state.messages, { role: 'user', text, ts: Date.now() }],
    }));
  };

  const setSpeed = (hz: number) => {
    socketRef.current?.emit('set_speed', { hz });
  };

  return { socket: socketRef.current, send, setSpeed };
}
