import { useAxonStore } from '../../store/axonStore';

export default function ThoughtCompetition() {
  const competition = useAxonStore((s) => s.thoughtCompetition);
  const ns = useAxonStore((s) => s.neuralState);
  const thoughts = ns.thoughts ?? [];

  return (
    <div style={{ padding: 12, overflowY: 'auto', height: '100%' }}>

      {/* Live thought stream */}
      <div style={{ marginBottom: 16, padding: '8px 10px', background: '#080818', border: '1px solid #4f46e533', borderRadius: 8 }}>
        <div style={sTitle}>💭 Active Thought Stream</div>
        {thoughts.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>waiting for thought data…</div>}
        {thoughts.slice(0, 5).map((t: any, i) => {
          const text = typeof t === 'string' ? t : t.text ?? JSON.stringify(t);
          const score = typeof t === 'object' ? t.score ?? t.neural_score : null;
          const winner = i === 0;
          return (
            <div key={i} style={{
              padding: '8px 10px', marginBottom: 6,
              background: winner ? '#1e1b4e33' : '#0a0a1a',
              border: `1px solid ${winner ? '#6366f1' : '#1e1b4e'}`,
              borderRadius: 6, position: 'relative',
            }}>
              {winner && (
                <div style={{
                  position: 'absolute', top: -8, right: 8,
                  fontSize: 7, fontWeight: 700, letterSpacing: '0.1em',
                  color: '#6366f1', background: '#080818', padding: '1px 5px', borderRadius: 3,
                  border: '1px solid #6366f1',
                }}>WINNER</div>
              )}
              <div style={{ fontSize: 10, color: winner ? '#e2e8f0' : '#94a3b8', lineHeight: 1.6 }}>{text}</div>
              {score !== null && score !== undefined && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
                  <div style={{ flex: 1, background: '#0d1117', borderRadius: 3, height: 4, overflow: 'hidden' }}>
                    <div style={{
                      width: `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`,
                      height: '100%', background: winner ? '#6366f1' : '#374151', borderRadius: 3,
                      transition: 'width 0.4s',
                    }} />
                  </div>
                  <span style={{ fontSize: 7, color: winner ? '#6366f1' : '#4b5563', fontFamily: 'monospace' }}>
                    {(score as number).toFixed(3)}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Historical competition rounds */}
      <div style={{ padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>⚔ Competition History</div>
        {competition.length === 0 && <div style={{ fontSize: 9, color: '#374151' }}>no competition rounds yet</div>}
        {competition.slice(0, 8).map((round: any, ri) => (
          <div key={ri} style={{ marginBottom: 10, padding: '6px 8px', background: '#0a0014', border: '1px solid #1e1b4e22', borderRadius: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span style={{ fontSize: 8, color: '#6366f1', fontFamily: 'monospace' }}>Round #{competition.length - ri}</span>
              <span style={{ fontSize: 7, color: '#374151' }}>
                {round.ts ? new Date(round.ts).toLocaleTimeString() : ''}
              </span>
            </div>
            {/* Winner */}
            {round.winner && (
              <div style={{ padding: '4px 6px', marginBottom: 4, background: '#1e1b4e22', borderLeft: '2px solid #6366f1', borderRadius: '0 4px 4px 0' }}>
                <div style={{ fontSize: 7, color: '#6366f1', marginBottom: 1 }}>WINNER</div>
                <div style={{ fontSize: 9, color: '#e2e8f0', lineHeight: 1.5 }}>
                  {typeof round.winner === 'string' ? round.winner : round.winner?.text ?? JSON.stringify(round.winner).slice(0, 80)}
                </div>
              </div>
            )}
            {/* Candidates */}
            {Array.isArray(round.candidates) && round.candidates.slice(0, 3).map((c: any, ci: number) => {
              const txt = typeof c === 'string' ? c : c.text ?? c.thought ?? '';
              const sc = typeof c === 'object' ? (c.score ?? c.neural_score ?? 0) : 0;
              return (
                <div key={ci} style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 2 }}>
                  <div style={{ flex: 1, background: '#0d1117', borderRadius: 3, height: 4, overflow: 'hidden' }}>
                    <div style={{ width: `${Math.round(sc * 100)}%`, height: '100%', background: '#374151', borderRadius: 3 }} />
                  </div>
                  <span style={{ fontSize: 7, color: '#4b5563', fontFamily: 'monospace', width: 28 }}>{(sc).toFixed(2)}</span>
                  <span style={{ fontSize: 7, color: '#374151', flex: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{txt.slice(0, 40)}</span>
                </div>
              );
            })}
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
