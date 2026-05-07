import { useAxonStore } from '../../store/axonStore';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer } from 'recharts';

export default function MemoryPanel() {
  const ns = useAxonStore((s) => s.neuralState);
  const memoryEvents = useAxonStore((s) => s.memoryEvents);

  const hierarchy = ns.memory_hierarchy ?? {};
  const tiers = ['episodic', 'semantic', 'value', 'identity'];
  const tierColors: Record<string, string> = {
    episodic: '#22d3ee', semantic: '#6366f1', value: '#f43f5e', identity: '#fbbf24',
  };

  // Radar data for memory tier balance
  const radarData = tiers.map((t) => ({
    tier: t.charAt(0).toUpperCase() + t.slice(1),
    value: Math.min(100, ((hierarchy[t] as any)?.count ?? 0) / 5),
  }));

  // Narrative worldview data
  const narratives = ns.narratives ?? {};

  // Interests
  const interests = ns.interests ?? {};

  return (
    <div style={{ padding: 12, overflowY: 'auto', height: '100%' }}>

      {/* Memory tiers */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        <div style={{ padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
          <div style={sTitle}>💾 Memory Hierarchy</div>
          {tiers.map((t) => {
            const d = (hierarchy[t] as any) ?? {};
            const col = tierColors[t];
            return (
              <div key={t} style={{ marginBottom: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                  <span style={{ fontSize: 8, color: col, textTransform: 'capitalize', fontWeight: 600 }}>{t}</span>
                  <span style={{ fontSize: 7, color: '#64748b', fontFamily: 'monospace' }}>
                    {d.count ?? 0} records · sal: {(d.salience ?? 0).toFixed(2)}
                  </span>
                </div>
                <div style={{ background: '#0d1117', borderRadius: 3, height: 6, overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.min(100, ((d.count ?? 0) / 200) * 100)}%`,
                    height: '100%', background: col, borderRadius: 3, transition: 'width 0.5s',
                    boxShadow: `0 0 4px ${col}`,
                  }} />
                </div>
              </div>
            );
          })}
        </div>

        {/* Memory radar */}
        <div style={{ padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
          <div style={sTitle}>📊 Tier Balance</div>
          <ResponsiveContainer width="100%" height={130}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#1e1b4e" />
              <PolarAngleAxis dataKey="tier" tick={{ fontSize: 8, fill: '#64748b' }} />
              <Radar dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Narrative worldviews */}
      <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>🌍 Narrative Worldviews</div>
        {Object.keys(narratives).length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no narrative data yet</div>}
        {Object.entries(narratives)
          .sort((a, b) => (b[1] as number) - (a[1] as number))
          .map(([name, score]) => {
            const pct = Math.round((score as number) * 100);
            const hue = Math.round((score as number) * 120);
            const col = `hsl(${hue}, 70%, 55%)`;
            return (
              <div key={name} style={{ marginBottom: 5 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                  <span style={{ fontSize: 8, color: '#94a3b8' }}>{name}</span>
                  <span style={{ fontSize: 7, color: col, fontFamily: 'monospace' }}>{pct}%</span>
                </div>
                <div style={{ background: '#0d1117', borderRadius: 3, height: 5, overflow: 'hidden' }}>
                  <div style={{ width: `${pct}%`, height: '100%', background: col, borderRadius: 3, transition: 'width 0.5s' }} />
                </div>
              </div>
            );
          })}
      </div>

      {/* Interests */}
      <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>🎯 Tracked Interests</div>
        {Object.keys(interests).length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no interests tracked yet</div>}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {Object.entries(interests)
            .sort((a, b) => (b[1] as number) - (a[1] as number))
            .slice(0, 20)
            .map(([name, weight]) => {
              const w = weight as number;
              const size = 8 + Math.round(w * 8);
              return (
                <div key={name} style={{
                  padding: '2px 7px', borderRadius: 12,
                  background: `rgba(99,102,241,${0.1 + w * 0.3})`,
                  border: `1px solid rgba(99,102,241,${0.2 + w * 0.4})`,
                  fontSize: size, color: '#a5b4fc',
                }}>
                  {name}
                </div>
              );
            })}
        </div>
      </div>

      {/* Memory event stream */}
      <div style={{ padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>📡 Memory Event Stream</div>
        {memoryEvents.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no events yet</div>}
        {memoryEvents.slice(0, 20).map((e: any, i) => (
          <div key={i} style={{
            marginBottom: 5, padding: '4px 6px',
            background: '#0a0014', borderLeft: `2px solid #22d3ee33`, borderRadius: '0 4px 4px 0',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
              <span style={{ fontSize: 7, color: '#22d3ee', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{e.type ?? 'encode'}</span>
              <span style={{ fontSize: 7, color: '#374151' }}>{e.ts ? new Date(e.ts).toLocaleTimeString() : ''}</span>
            </div>
            <div style={{ fontSize: 8, color: '#64748b', lineHeight: 1.5 }}>
              {e.content ?? e.text ?? JSON.stringify(e).slice(0, 80)}
            </div>
          </div>
        ))}
      </div>

    </div>
  );
}

const sTitle: React.CSSProperties = {
  fontSize: 9, fontWeight: 700, letterSpacing: '0.1em',
  textTransform: 'uppercase', color: '#6366f1', marginBottom: 8,
};
