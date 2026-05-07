import { useAxonStore } from '../../store/axonStore';

export default function BeliefsPanel() {
  const ns = useAxonStore((s) => s.neuralState);

  const beliefs = ns.beliefs ?? [];
  const drives = ns.drives ?? {};
  const goals = ns.goals ?? [];
  const selfModel = ns.self_model ?? {};
  const surpriseEvents = useAxonStore((s) => s.surpriseEvents);

  const driveColors: Record<string, string> = {
    curiosity: '#22d3ee', social: '#f472b6', competence: '#4ade80', stability: '#fbbf24',
  };

  return (
    <div style={{ padding: 12, overflowY: 'auto', height: '100%' }}>

      {/* Self-Model */}
      <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #fbbf2433', borderRadius: 8 }}>
        <div style={{ ...sTitle, color: '#fbbf24' }}>🪞 Self-Model</div>
        {Object.entries(selfModel).map(([k, vals]) => {
          if (!Array.isArray(vals) || vals.length === 0) return null;
          const colors: Record<string, string> = {
            I_am: '#6366f1', I_believe: '#22d3ee', I_like: '#4ade80', I_avoid: '#f43f5e', I_want: '#fbbf24',
          };
          return (
            <div key={k} style={{ marginBottom: 6 }}>
              <div style={{ fontSize: 8, fontWeight: 700, color: colors[k] ?? '#94a3b8', marginBottom: 3 }}>{k.replace('_', ' ')}</div>
              {vals.slice(0, 3).map((v: string, i: number) => (
                <div key={i} style={{
                  fontSize: 8, color: '#64748b', marginBottom: 2,
                  paddingLeft: 6, borderLeft: `2px solid ${colors[k] ?? '#1e1b4e'}33`,
                  lineHeight: 1.5,
                }}>{v}</div>
              ))}
            </div>
          );
        })}
        {Object.keys(selfModel).length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>self-model rebuilding…</div>}
      </div>

      {/* Drive System */}
      <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>⚡ Drive System</div>
        {Object.keys(drives).length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no drive data</div>}
        {Object.entries(drives).map(([name, d]: [string, any]) => {
          const col = driveColors[name] ?? '#6366f1';
          const pressure = d.pressure ?? 0;
          const urgency = d.urgency ?? 0;
          return (
            <div key={name} style={{ marginBottom: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 3 }}>
                <span style={{ fontSize: 9, color: col, fontWeight: 600, textTransform: 'capitalize' }}>{name}</span>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  {d.satisfied && <span style={{ fontSize: 7, color: '#4ade80', fontWeight: 700 }}>SATISFIED</span>}
                  {urgency > 0.5 && <span style={{ fontSize: 7, color: '#ef4444', fontWeight: 700 }}>PRESSING</span>}
                  <span style={{ fontSize: 7, color: '#64748b', fontFamily: 'monospace' }}>urg:{urgency.toFixed(2)}</span>
                </div>
              </div>
              <div style={{ background: '#0d1117', borderRadius: 4, height: 8, overflow: 'hidden', position: 'relative' }}>
                <div style={{
                  width: `${Math.round(pressure * 100)}%`, height: '100%',
                  background: col, borderRadius: 4, transition: 'width 0.5s',
                  boxShadow: urgency > 0.5 ? `0 0 8px ${col}` : 'none',
                }} />
                {urgency > 0 && (
                  <div style={{
                    position: 'absolute', top: 0, left: `${Math.round(pressure * 100)}%`,
                    width: `${Math.round(urgency * 100)}%`, height: '100%',
                    background: `${col}44`, borderRadius: '0 4px 4px 0',
                  }} />
                )}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 1 }}>
                <span style={{ fontSize: 7, color: '#374151' }}>pressure</span>
                <span style={{ fontSize: 7, color: col, fontFamily: 'monospace' }}>{(pressure * 100).toFixed(0)}%</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Belief System */}
      <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>🧩 Belief System</div>
        {beliefs.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no beliefs formed yet</div>}
        {[...beliefs].sort((a: any, b: any) => (b.strength ?? 0) - (a.strength ?? 0)).slice(0, 10).map((b: any, i) => {
          const valCol = (b.valence ?? 0) > 0 ? '#4ade80' : (b.valence ?? 0) < 0 ? '#f43f5e' : '#94a3b8';
          const dissonance = b.dissonance ?? 0;
          return (
            <div key={i} style={{
              marginBottom: 6, padding: '5px 7px',
              background: dissonance > 0.3 ? '#2d0a0a' : '#0a0014',
              border: `1px solid ${dissonance > 0.3 ? '#f43f5e33' : '#1e1b4e'}`,
              borderRadius: 5,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span style={{ fontSize: 8, color: '#94a3b8', flex: 1, marginRight: 6, lineHeight: 1.4 }}>
                  {b.key ?? b.text ?? b.claim ?? JSON.stringify(b).slice(0, 60)}
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', flexShrink: 0 }}>
                  <span style={{ fontSize: 7, color: '#6366f1', fontFamily: 'monospace' }}>{((b.strength ?? 0) * 100).toFixed(0)}%</span>
                  {dissonance > 0.2 && <span style={{ fontSize: 7, color: '#f43f5e' }}>⚡ dissonant</span>}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                <div style={{ flex: 1, background: '#0d1117', borderRadius: 2, height: 4, overflow: 'hidden' }}>
                  <div style={{ width: `${(b.strength ?? 0) * 100}%`, height: '100%', background: '#6366f1', borderRadius: 2 }} />
                </div>
                <span style={{ fontSize: 7, color: valCol, fontFamily: 'monospace' }}>v:{(b.valence ?? 0).toFixed(2)}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Goals */}
      <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>🎯 Active Goals</div>
        {goals.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no goals set</div>}
        {goals.slice(0, 8).map((g: any, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <div style={{
              width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
              background: g.active ? '#4ade80' : '#374151',
              boxShadow: g.active ? '0 0 4px #4ade80' : 'none',
            }} />
            <span style={{ fontSize: 9, color: g.active ? '#e2e8f0' : '#64748b', flex: 1 }}>
              {typeof g === 'string' ? g : g.text ?? g.goal ?? JSON.stringify(g).slice(0, 50)}
            </span>
            {g.priority && (
              <span style={{ fontSize: 7, color: '#fbbf24', fontFamily: 'monospace' }}>p:{g.priority}</span>
            )}
          </div>
        ))}
      </div>

      {/* Surprise events */}
      <div style={{ padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>⚡ Surprise Events</div>
        {surpriseEvents.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no surprise events</div>}
        {surpriseEvents.slice(0, 10).map((e: any, i) => (
          <div key={i} style={{ marginBottom: 5, padding: '4px 6px', background: '#0a0014', borderLeft: '2px solid #f97316', borderRadius: '0 4px 4px 0' }}>
            <div style={{ fontSize: 7, color: '#f97316', fontWeight: 700, marginBottom: 1 }}>
              {e.type ?? 'surprise'} {e.magnitude ? `· mag: ${(e.magnitude).toFixed(3)}` : ''}
            </div>
            <div style={{ fontSize: 8, color: '#64748b' }}>{e.title ?? e.detail ?? e.text ?? ''}</div>
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
