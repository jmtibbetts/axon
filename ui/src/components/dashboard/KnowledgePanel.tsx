import { useState, useRef, useCallback } from 'react';
import { useAxonStore } from '../../store/axonStore';
import { useAxonSocket } from '../../hooks/useAxonSocket';

const ACCEPT = '.pdf,.docx,.doc,.txt,.md,.rst,.csv,.epub';

type IngestionStatus = 'processing' | 'done' | 'error';

interface LocalIngestion {
  filename: string;
  status: IngestionStatus;
  concepts?: number;
  opinions?: number;
  ts: number;
  error?: string;
}

export default function KnowledgePanel() {
  const { emit } = useAxonSocket();
  const [dragging, setDragging] = useState(false);
  const [opinion, setOpinion] = useState('');
  const [opinionResult, setOpinionResult] = useState<string | null>(null);
  const [autonomousSteps, setAutonomousSteps] = useState(100);
  const [autonomousRunning, setAutonomousRunning] = useState(false);
  const [localIngestions, setLocalIngestions] = useState<LocalIngestion[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const ingestions = useAxonStore((s) => s.ingestions);

  // Merge store + local
  const allIngestions = [...ingestions, ...localIngestions]
    .reduce((acc: LocalIngestion[], cur) => {
      if (!acc.find((a) => a.filename === cur.filename)) acc.push(cur);
      return acc;
    }, [])
    .sort((a, b) => b.ts - a.ts);

  const uploadFiles = useCallback(async (files: FileList | File[]) => {
    for (const file of Array.from(files)) {
      setLocalIngestions((prev) => [
        { filename: file.name, status: 'processing', ts: Date.now() },
        ...prev,
      ]);
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await fetch('/upload_knowledge', { method: 'POST', body: formData });
        const data = await res.json();
        setLocalIngestions((prev) =>
          prev.map((i) =>
            i.filename === file.name
              ? { ...i, status: data.ok ? 'done' : 'error', concepts: data.concepts, opinions: data.opinions, error: data.error }
              : i
          )
        );
      } catch (e: any) {
        setLocalIngestions((prev) =>
          prev.map((i) =>
            i.filename === file.name ? { ...i, status: 'error', error: e.message } : i
          )
        );
      }
    }
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files.length > 0) uploadFiles(e.dataTransfer.files);
  };

  const handleOpinion = async () => {
    if (!opinion.trim()) return;
    setOpinionResult('thinking…');
    try {
      const res = await fetch('/api/first_opinion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: opinion }),
      });
      const data = await res.json();
      setOpinionResult(data.opinion ?? data.result ?? JSON.stringify(data));
    } catch (e: any) {
      setOpinionResult(`Error: ${e.message}`);
    }
  };

  const handleAutonomous = () => {
    setAutonomousRunning(true);
    emit('run_autonomous', { steps: autonomousSteps });
    setTimeout(() => setAutonomousRunning(false), autonomousSteps * 100 + 1000);
  };

  const statusColor = (s: IngestionStatus) =>
    s === 'done' ? '#4ade80' : s === 'error' ? '#f43f5e' : '#fbbf24';
  const statusLabel = (s: IngestionStatus) =>
    s === 'done' ? '✓' : s === 'error' ? '✗' : '⟳';

  return (
    <div style={{ padding: 12, overflowY: 'auto', height: '100%' }}>

      {/* Drop zone */}
      <div
        onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
        style={{
          marginBottom: 12, padding: '24px 16px',
          border: `2px dashed ${dragging ? '#6366f1' : '#1e1b4e'}`,
          borderRadius: 10, textAlign: 'center', cursor: 'pointer',
          background: dragging ? '#1e1b4e22' : '#080818',
          transition: 'all 0.2s',
        }}
      >
        <div style={{ fontSize: 28, marginBottom: 6 }}>📚</div>
        <div style={{ fontSize: 11, color: '#6366f1', fontWeight: 600 }}>
          {dragging ? 'Drop files to feed AXON' : 'Drag & drop or click to upload'}
        </div>
        <div style={{ fontSize: 8, color: '#374151', marginTop: 4 }}>
          PDF · DOCX · TXT · MD · EPUB · CSV
        </div>
        <input
          ref={fileRef} type="file" accept={ACCEPT} multiple
          onChange={(e) => e.target.files && uploadFiles(e.target.files)}
          style={{ display: 'none' }}
        />
      </div>

      {/* Ingestion queue */}
      {allIngestions.length > 0 && (
        <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
          <div style={sTitle}>📥 Knowledge Queue</div>
          {allIngestions.map((item, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5,
              padding: '5px 7px', background: '#0a0014', borderRadius: 5,
              border: `1px solid ${statusColor(item.status)}22`,
            }}>
              <span style={{
                fontSize: 11, color: statusColor(item.status),
                animation: item.status === 'processing' ? 'spin 1s linear infinite' : 'none',
              }}>{statusLabel(item.status)}</span>
              <span style={{ fontSize: 9, color: '#94a3b8', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {item.filename}
              </span>
              {item.status === 'done' && (
                <span style={{ fontSize: 7, color: '#4ade80', fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
                  {item.concepts ?? 0}c · {item.opinions ?? 0}op
                </span>
              )}
              {item.status === 'error' && (
                <span style={{ fontSize: 7, color: '#f43f5e' }}>{item.error?.slice(0, 30)}</span>
              )}
              {item.status === 'processing' && (
                <span style={{ fontSize: 7, color: '#fbbf24' }}>processing…</span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Force opinion */}
      <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>💡 Force Opinion Formation</div>
        <div style={{ fontSize: 8, color: '#374151', marginBottom: 8 }}>
          Make AXON form an opinion on any topic, updating its belief system immediately.
        </div>
        <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
          <input
            value={opinion}
            onChange={(e) => setOpinion(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleOpinion()}
            placeholder="e.g. 'consciousness', 'quantum computing', 'empathy'…"
            style={{
              flex: 1, padding: '6px 8px', background: '#0a0014',
              border: '1px solid #1e1b4e', borderRadius: 6,
              color: '#e2e8f0', fontSize: 10, outline: 'none',
            }}
          />
          <button onClick={handleOpinion} style={btnStyle('#6366f1', '#1e1b4e')}>Form Opinion</button>
        </div>
        {opinionResult && (
          <div style={{
            padding: '8px 10px', background: '#0a0014',
            border: '1px solid #6366f133', borderRadius: 6,
            fontSize: 9, color: '#94a3b8', lineHeight: 1.7,
          }}>
            {opinionResult}
          </div>
        )}
      </div>

      {/* Autonomous cognition */}
      <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>🤖 Autonomous Cognition</div>
        <div style={{ fontSize: 8, color: '#374151', marginBottom: 8 }}>
          Run N steps of unsupervised thought — AXON explores, reflects, and learns without input.
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
          <span style={{ fontSize: 8, color: '#64748b' }}>Steps:</span>
          {[50, 100, 200, 500].map((n) => (
            <button
              key={n}
              onClick={() => setAutonomousSteps(n)}
              style={{
                padding: '3px 8px', borderRadius: 4,
                background: autonomousSteps === n ? '#4f46e522' : '#080818',
                border: `1px solid ${autonomousSteps === n ? '#6366f1' : '#1e1b4e'}`,
                color: autonomousSteps === n ? '#6366f1' : '#64748b',
                fontSize: 8, cursor: 'pointer',
              }}
            >{n}</button>
          ))}
        </div>
        <button
          onClick={handleAutonomous}
          disabled={autonomousRunning}
          style={{
            width: '100%', padding: '8px',
            background: autonomousRunning ? '#1e293b' : '#4f46e522',
            border: `1px solid ${autonomousRunning ? '#374151' : '#6366f1'}`,
            borderRadius: 6, color: autonomousRunning ? '#374151' : '#6366f1',
            fontSize: 10, fontWeight: 600, cursor: autonomousRunning ? 'not-allowed' : 'pointer',
          }}
        >
          {autonomousRunning ? `⟳ Running ${autonomousSteps} steps…` : `▶ Run ${autonomousSteps} Autonomous Steps`}
        </button>
      </div>

      {/* Brain snapshot controls */}
      <div style={{ padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>💾 Brain Snapshots</div>
        <div style={{ display: 'flex', gap: 6 }}>
          <button
            onClick={async () => {
              await fetch('/api/brain/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            }}
            style={btnStyle('#4ade80', '#052e16')}
          >💾 Save State</button>
          <button
            onClick={async () => {
              await fetch('/api/brain/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            }}
            style={btnStyle('#22d3ee', '#0a2540')}
          >⏫ Load State</button>
        </div>
      </div>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}

function btnStyle(color: string, _bg?: string): React.CSSProperties {
  return {
    padding: '5px 12px', borderRadius: 6,
    background: `${color}18`, border: `1px solid ${color}44`,
    color, fontSize: 9, fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap',
  };
}

const sTitle: React.CSSProperties = {
  fontSize: 9, fontWeight: 700, letterSpacing: '0.1em',
  textTransform: 'uppercase', color: '#6366f1', marginBottom: 8,
};
