import { useAxonStore } from '../../store/axonStore';

export default function SystemLog() {
  const logs = useAxonStore((s) => s.logs);
  const engineRunning = useAxonStore((s) => s.engineRunning);
  const connected = useAxonStore((s) => s.connected);
  const lmStatus = useAxonStore((s) => s.lmStatus);

  return (
    <div style={{ padding: 12, overflowY: 'auto', height: '100%', display: 'flex', flexDirection: 'column', gap: 10 }}>

      {/* Status indicators */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
        {[
          { label: 'WebSocket', ok: connected, text: connected ? 'CONNECTED' : 'OFFLINE' },
          { label: 'Engine', ok: engineRunning, text: engineRunning ? 'RUNNING' : 'STOPPED' },
          { label: 'LM Studio', ok: lmStatus?.lm_studio, text: lmStatus?.lm_studio ? `online · ${lmStatus?.lm_model ?? '?'}` : 'offline' },
        ].map(({ label, ok, text }) => (
          <div key={label} style={{
            padding: '8px 10px', background: '#080818',
            border: `1px solid ${ok ? '#4ade8033' : '#f43f5e33'}`,
            borderRadius: 8,
          }}>
            <div style={{ fontSize: 7, color: '#64748b', textTransform: 'uppercase', marginBottom: 3 }}>{label}</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{
                width: 6, height: 6, borderRadius: '50%',
                background: ok ? '#4ade80' : '#f43f5e',
                boxShadow: `0 0 5px ${ok ? '#4ade80' : '#f43f5e'}`,
              }} />
              <span style={{ fontSize: 9, color: ok ? '#4ade80' : '#f43f5e', fontFamily: 'monospace', fontWeight: 600 }}>{text}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Startup error banner */}
      {!engineRunning && logs.some((l: any) => (l.msg ?? '').includes('failed')) && (
        <div style={{
          padding: '10px 12px', background: '#2d0a0a',
          border: '1px solid #f43f5e', borderRadius: 8,
        }}>
          <div style={{ fontSize: 10, color: '#f43f5e', fontWeight: 700, marginBottom: 6 }}>⚠️ Engine Startup Failed</div>
          <div style={{ fontSize: 8, color: '#94a3b8', lineHeight: 1.7 }}>
            Check the log below. Common causes:<br />
            • A Python dependency is missing (run <code style={{ color: '#22d3ee' }}>pip install -r requirements.txt</code>)<br />
            • CUDA/torch version mismatch<br />
            • SQLite DB locked from previous crash<br />
            • Import error in a cognition module
          </div>
        </div>
      )}

      {/* Live log */}
      <div style={{ flex: 1, padding: '8px 10px', background: '#030712', border: '1px solid #1e1b4e', borderRadius: 8, overflowY: 'auto' }}>
        <div style={{ fontSize: 9, fontWeight: 700, color: '#6366f1', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6 }}>
          System Log
        </div>
        {logs.length === 0 && (
          <div style={{ fontSize: 9, color: '#374151' }}>No log entries yet…</div>
        )}
        {logs.map((entry: any, i: number) => {
          const msg = typeof entry === 'string' ? entry : entry.msg ?? JSON.stringify(entry);
          const isError = msg.includes('❌') || msg.includes('Error') || msg.includes('FAILED') || msg.includes('Traceback') || msg.includes('exception');
          const isWarn = msg.includes('⚠️') || msg.includes('Warning') || msg.includes('not detected');
          const isGood = msg.includes('✓') || msg.includes('online') || msg.includes('Connected');
          const col = isError ? '#f43f5e' : isWarn ? '#fbbf24' : isGood ? '#4ade80' : '#64748b';
          return (
            <div key={i} style={{
              fontFamily: 'monospace', fontSize: 9, color: col,
              marginBottom: 3, lineHeight: 1.5,
              borderLeft: isError ? '2px solid #f43f5e33' : 'none',
              paddingLeft: isError ? 6 : 0,
            }}>
              <span style={{ color: '#1e293b', marginRight: 6 }}>
                {entry.ts ? new Date(entry.ts).toLocaleTimeString() : ''}
              </span>
              {msg}
            </div>
          );
        })}
      </div>
    </div>
  );
}
