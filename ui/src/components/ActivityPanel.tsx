import { useAxonStore } from '../store/axonStore';

export default function ActivityPanel() {
  const ns             = useAxonStore((s) => s.neuralState);
  const hebbianEvents  = useAxonStore((s) => s.hebbianEvents);
  const memoryEvents   = useAxonStore((s) => s.memoryEvents);
  const logs           = useAxonStore((s) => s.logs);

  const topClusters = ns.top_clusters ?? [];
  const topRoutes   = ns.top_routes ?? [];

  return (
    <div style={{ overflowY: 'auto', height: '100%', padding: '6px 4px', fontSize: 11, color: '#e2e8f0' }}>

      {/* Top Active Clusters */}
      <div style={{ marginBottom: 12, padding: '6px 8px', background: '#0a0014', border: '1px solid #1e1b4e', borderRadius: 6 }}>
        <div style={sTitle}>⚡ Top Active Clusters</div>
        {topClusters.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>waiting for signal…</div>}
        {topClusters.slice(0, 10).map((c, i) => (
          <div key={i} style={{ display: 'grid', gridTemplateColumns: '1fr 60px 30px', gap: 4, marginBottom: 3, alignItems: 'center' }}>
            <span style={{ fontSize: 8, color: '#94a3b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {c.name}
            </span>
            <div style={{ background: '#0d1117', borderRadius: 3, height: 5, overflow: 'hidden' }}>
              <div style={{ width: `${Math.round(c.activation * 100)}%`, height: '100%', background: '#6366f1', borderRadius: 3, transition: 'width 0.4s' }} />
            </div>
            <span style={{ fontSize: 7, color: '#6366f1', textAlign: 'right', fontFamily: 'monospace' }}>{c.activation.toFixed(2)}</span>
          </div>
        ))}
      </div>

      {/* Top Routes */}
      {topRoutes.length > 0 && (
        <div style={{ marginBottom: 12, padding: '6px 8px', background: '#0a0014', border: '1px solid #1e1b4e', borderRadius: 6 }}>
          <div style={sTitle}>🔗 Active Pathways</div>
          {topRoutes.slice(0, 5).map((r: any, i) => (
            <div key={i} style={{ fontSize: 8, color: '#94a3b8', marginBottom: 3, display: 'flex', justifyContent: 'space-between' }}>
              <span>{r.src_region ?? r.src} → {r.dst_region ?? r.dst}</span>
              <span style={{ color: '#6366f1', fontFamily: 'monospace' }}>{(r.weight ?? 0).toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Hebbian Events */}
      <div style={{ marginBottom: 12, padding: '6px 8px', background: '#0a0014', border: '1px solid #1e1b4e', borderRadius: 6 }}>
        <div style={sTitle}>🧬 Hebbian Events</div>
        {hebbianEvents.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no events yet</div>}
        {hebbianEvents.map((e: any, i) => (
          <div key={i} style={{ fontSize: 8, color: '#94a3b8', marginBottom: 3, paddingLeft: 4, borderLeft: '2px solid #4f46e5', lineHeight: 1.5 }}>
            {e.type ?? 'event'}: {e.src} → {e.dst} ({(e.strength ?? 0).toFixed(3)})
          </div>
        ))}
      </div>

      {/* Memory Events */}
      <div style={{ marginBottom: 12, padding: '6px 8px', background: '#0a0014', border: '1px solid #1e1b4e', borderRadius: 6 }}>
        <div style={sTitle}>💾 Memory Events</div>
        {memoryEvents.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no events yet</div>}
        {memoryEvents.map((e: any, i) => (
          <div key={i} style={{ fontSize: 8, color: '#94a3b8', marginBottom: 3, lineHeight: 1.5 }}>
            <span style={{ color: '#22d3ee' }}>{e.type ?? 'memory'}</span>: {e.content ?? e.text ?? JSON.stringify(e).slice(0, 60)}
          </div>
        ))}
      </div>

      {/* Logs */}
      <div style={{ padding: '6px 8px', background: '#0a0014', border: '1px solid #1e1b4e', borderRadius: 6 }}>
        <div style={sTitle}>📋 System Log</div>
        {logs.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no log output</div>}
        {logs.slice(0, 30).map((l: any, i) => (
          <div key={i} style={{ fontSize: 7, color: '#4b5563', marginBottom: 1, fontFamily: 'monospace', lineHeight: 1.5 }}>
            <span style={{ color: l.level === 'error' ? '#ef4444' : l.level === 'warn' ? '#f59e0b' : '#374151' }}>
              [{l.level ?? 'log'}]
            </span>{' '}
            {l.message ?? l.text ?? JSON.stringify(l).slice(0, 80)}
          </div>
        ))}
      </div>

    </div>
  );
}

const sTitle: React.CSSProperties = {
  fontSize: 9, fontWeight: 700, letterSpacing: '0.08em',
  textTransform: 'uppercase', color: '#6366f1', marginBottom: 6,
};
