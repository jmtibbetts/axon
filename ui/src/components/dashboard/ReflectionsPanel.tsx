import { useAxonStore } from '../../store/axonStore';

export default function ReflectionsPanel() {
  const reflections = useAxonStore((s) => s.reflections);
  const ns = useAxonStore((s) => s.neuralState);

  const storeReflections: any[] = ns.reflections ?? [];
  const allReflections = [...reflections, ...storeReflections].slice(0, 30);

  const hebbianEvents = useAxonStore((s) => s.hebbianEvents);
  const regionSpikes = useAxonStore((s) => s.regionSpikes);

  return (
    <div style={{ padding: 12, overflowY: 'auto', height: '100%' }}>

      {/* Autonomous reflections */}
      <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #c084fc33', borderRadius: 8 }}>
        <div style={{ ...sTitle, color: '#c084fc' }}>🌀 Autonomous Reflections</div>
        <div style={{ fontSize: 8, color: '#374151', marginBottom: 8 }}>AXON reflects every ~15s from its own activation patterns</div>
        {allReflections.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>waiting for first reflection…</div>}
        {allReflections.map((r: any, i) => {
          const text = typeof r === 'string' ? r : r.text ?? r.content ?? r.reflection ?? JSON.stringify(r).slice(0, 120);
          return (
            <div key={i} style={{
              marginBottom: 8, padding: '8px 10px',
              background: '#0a0014', border: '1px solid #c084fc22',
              borderRadius: 6, position: 'relative',
            }}>
              <div style={{ position: 'absolute', top: 8, right: 8, fontSize: 7, color: '#374151' }}>
                {r.ts ? new Date(r.ts).toLocaleTimeString() : ''}
              </div>
              <div style={{ width: 4, height: 4, borderRadius: '50%', background: '#c084fc', position: 'absolute', top: 10, left: 6 }} />
              <div style={{ fontSize: 9, color: '#94a3b8', lineHeight: 1.7, paddingLeft: 10 }}>{text}</div>
            </div>
          );
        })}
      </div>

      {/* Hebbian events */}
      <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>🧬 Hebbian Learning Events</div>
        <div style={{ fontSize: 8, color: '#374151', marginBottom: 6 }}>New synaptic connections being formed</div>
        {hebbianEvents.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no events yet</div>}
        {hebbianEvents.slice(0, 15).map((e: any, i) => {
          const strength = e.strength ?? e.delta ?? e.weight ?? 0;
          const isNew = e.type === 'new' || e.type === 'formed';
          const isPruned = e.type === 'pruned' || e.type === 'prune';
          const col = isNew ? '#4ade80' : isPruned ? '#f43f5e' : '#6366f1';
          return (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              marginBottom: 4, padding: '3px 6px',
              background: '#0a0014', borderRadius: 4,
              borderLeft: `2px solid ${col}`,
            }}>
              <span style={{ fontSize: 7, color: col, fontWeight: 700, width: 40, flexShrink: 0 }}>
                {isNew ? 'FORM' : isPruned ? 'PRUNE' : 'STRN'}
              </span>
              <span style={{ fontSize: 8, color: '#64748b', flex: 1 }}>
                {e.src ?? '?'} → {e.dst ?? '?'}
              </span>
              <div style={{ width: 40, background: '#0d1117', borderRadius: 2, height: 4, overflow: 'hidden' }}>
                <div style={{ width: `${Math.abs(strength) * 100}%`, height: '100%', background: col, borderRadius: 2 }} />
              </div>
              <span style={{ fontSize: 7, color: col, fontFamily: 'monospace' }}>{strength.toFixed(3)}</span>
            </div>
          );
        })}
      </div>

      {/* Region spikes */}
      <div style={{ padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>⚡ Region Spikes</div>
        {regionSpikes.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no spikes detected</div>}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {regionSpikes.slice(0, 20).map((s: any, i) => {
            const magnitude = s.magnitude ?? s.activation ?? 0;
            const bigSpike = magnitude > 0.7;
            return (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '3px 6px', borderRadius: 4,
                background: bigSpike ? '#1a0a00' : '#0a0014',
                border: `1px solid ${bigSpike ? '#f9731633' : '#1e1b4e22'}`,
              }}>
                <div style={{
                  width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                  background: bigSpike ? '#f97316' : '#374151',
                  boxShadow: bigSpike ? '0 0 6px #f97316' : 'none',
                }} />
                <span style={{ fontSize: 8, color: bigSpike ? '#f97316' : '#64748b', flex: 1 }}>
                  {s.region ?? s.cluster ?? '?'}
                </span>
                <span style={{ fontSize: 7, color: '#374151', fontFamily: 'monospace' }}>
                  {magnitude.toFixed(3)}
                </span>
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
}

const sTitle: React.CSSProperties = {
  fontSize: 9, fontWeight: 700, letterSpacing: '0.1em',
  textTransform: 'uppercase', color: '#6366f1', marginBottom: 8,
};
