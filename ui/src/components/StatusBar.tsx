import { useAxonStore } from '../store/axonStore';
import { useAxonSocket } from '../hooks/useAxonSocket';

export default function StatusBar() {
  const connected = useAxonStore((s) => s.connected);
  const ns        = useAxonStore((s) => s.neuralState);
  const lmStatus  = useAxonStore((s) => s.lmStatus);
  const thinking  = useAxonStore((s) => s.thinking);
  const lastTick  = useAxonStore((s) => s.lastTick);
  const { emit } = useAxonSocket();

  const secAgo = lastTick ? Math.round((Date.now() - lastTick) / 1000) : null;
  const alive  = secAgo !== null && secAgo < 10;

  const dotColor = !connected ? '#ef4444' : alive ? '#00ff88' : '#fbbf24';

  return (
    <div style={{
      height: 32, display: 'flex', alignItems: 'center', gap: 12,
      padding: '0 12px', background: '#020205',
      borderBottom: '1px solid #1e1b4e', flexShrink: 0,
      fontSize: 10, fontFamily: 'monospace', color: '#64748b',
      userSelect: 'none',
    }}>
      {/* Status dot + connection */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
        <div style={{ width: 7, height: 7, borderRadius: '50%', background: dotColor, boxShadow: `0 0 6px ${dotColor}` }} />
        <span style={{ color: dotColor, fontWeight: 700 }}>
          {!connected ? 'OFFLINE' : alive ? 'LIVE' : 'STALE'}
        </span>
      </div>

      <Sep />

      {/* Tick */}
      <span>tick: <span style={{ color: '#a78bfa' }}>{ns.tick ?? '—'}</span></span>

      {/* Neurons */}
      <span>neurons: <span style={{ color: '#22d3ee' }}>{ns.total_neurons?.toLocaleString() ?? '—'}</span></span>

      {/* Emotion */}
      {ns.emotion?.current && (
        <>
          <Sep />
          <span>
            {getEmoji(ns.emotion.current)}{' '}
            <span style={{ color: '#e2e8f0' }}>{ns.emotion.current}</span>
            {' '}
            <span style={{ color: '#94a3b8', fontSize: 8 }}>
              v:{(ns.emotion.valence ?? 0).toFixed(2)}
            </span>
          </span>
        </>
      )}

      {/* Thinking */}
      {thinking && (
        <>
          <Sep />
          <span style={{ color: '#6366f1', animation: 'blink 1s infinite' }}>⚙ thinking…</span>
        </>
      )}

      {/* LM status */}
      {lmStatus && (
        <>
          <Sep />
          <span style={{ color: '#fbbf24' }}>
            LM: {lmStatus.provider ?? lmStatus.model ?? 'connected'}
          </span>
        </>
      )}

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Speed presets */}
      <span style={{ color: '#374151' }}>speed:</span>
      {[['💤', 0.5], ['🐢', 5], ['✅', 10], ['⚡', 20], ['🚀', 50]].map(([emoji, hz]) => (
        <button
          key={String(hz)}
          onClick={() => emit('set_speed', { hz: Number(hz) })}
          style={{
            padding: '1px 5px', borderRadius: 4,
            background: '#0d1117', border: '1px solid #1e293b',
            color: '#94a3b8', cursor: 'pointer', fontSize: 9,
          }}
        >
          {emoji}
        </button>
      ))}

      <style>{`
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
      `}</style>
    </div>
  );
}

function Sep() {
  return <div style={{ width: 1, height: 14, background: '#1e1b4e', flexShrink: 0 }} />;
}

function getEmoji(emotion: string): string {
  const map: Record<string, string> = {
    happy: '😊', excited: '🤩', curious: '🤔', calm: '😌',
    neutral: '😐', sad: '😔', anxious: '😰', angry: '😠',
    bored: '😑', focused: '🎯', surprised: '😲', afraid: '😨',
  };
  return map[emotion.toLowerCase()] ?? '🧠';
}
