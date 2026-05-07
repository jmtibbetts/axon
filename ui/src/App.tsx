import { Suspense, lazy } from 'react';
import { useAxonSocket } from './hooks/useAxonSocket';
import { useAxonStore } from './store/axonStore';
import StatusBar from './components/StatusBar';
import NeuralPanel from './components/NeuralPanel';
import ChatPanel from './components/ChatPanel';
import ActivityPanel from './components/ActivityPanel';

// Lazy-load 3D canvas (heavy)
const BrainCanvas = lazy(() => import('./components/BrainCanvas'));

const TABS = [
  { id: 'brain',    label: '🧠 Brain'    },
  { id: 'activity', label: '⚡ Activity'  },
  { id: 'memory',   label: '💾 Memory'   },
];

export default function App() {
  // Bootstrap the socket on mount
  useAxonSocket();

  const activeTab = useAxonStore((s) => s.activeTab);
  const set       = useAxonStore((s) => s.set);

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100vh', background: '#020205',
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
      color: '#e2e8f0', overflow: 'hidden',
    }}>
      {/* Top status bar */}
      <StatusBar />

      {/* Main 3-column layout */}
      <div style={{
        flex: 1, display: 'grid',
        gridTemplateColumns: '260px 1fr 280px',
        gridTemplateRows: '1fr',
        gap: 4, padding: 4, minHeight: 0, overflow: 'hidden',
      }}>

        {/* COL 1: Right panel tabs */}
        <div style={{
          display: 'flex', flexDirection: 'column', minHeight: 0,
          background: '#030712', border: '1px solid #1e1b4e', borderRadius: 8, overflow: 'hidden',
        }}>
          {/* Tab buttons */}
          <div style={{ display: 'flex', borderBottom: '1px solid #1e1b4e', flexShrink: 0 }}>
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => set({ activeTab: tab.id })}
                style={{
                  flex: 1, padding: '7px 4px', fontSize: 9, fontWeight: 600,
                  border: 'none', cursor: 'pointer', transition: 'all 0.15s',
                  background: activeTab === tab.id ? '#0f172a' : 'transparent',
                  color: activeTab === tab.id ? '#6366f1' : '#4b5563',
                  borderBottom: activeTab === tab.id ? '2px solid #6366f1' : '2px solid transparent',
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
          {/* Tab content */}
          <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
            {activeTab === 'brain'    && <NeuralPanel />}
            {activeTab === 'activity' && <ActivityPanel />}
            {activeTab === 'memory'   && <MemoryPlaceholder />}
          </div>
        </div>

        {/* COL 2: Brain canvas (top 60%) + Chat (bottom 40%) */}
        <div style={{
          display: 'flex', flexDirection: 'column', minHeight: 0,
          border: '1px solid #1e1b4e', borderRadius: 8, overflow: 'hidden',
        }}>
          {/* Brain Canvas */}
          <div style={{ flex: 3, minHeight: 0, position: 'relative' }}>
            <Suspense fallback={
              <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#020205', color: '#374151', fontSize: 13 }}>
                loading 3D brain…
              </div>
            }>
              <BrainCanvas />
            </Suspense>
          </div>

          {/* Chat */}
          <div style={{ flex: 2, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <ChatPanel />
          </div>
        </div>

        {/* COL 3: Vision + log sidebar */}
        <div style={{
          display: 'flex', flexDirection: 'column', minHeight: 0, gap: 4,
        }}>
          <VisionPanel />
          <LogPanel />
        </div>

      </div>
    </div>
  );
}

// ── Vision feed ────────────────────────────────────────────────────────────
function VisionPanel() {
  const frame    = useAxonStore((s) => s.visionFrame);
  const faceData = useAxonStore((s) => s.faceData);

  return (
    <div style={{
      flex: 1, background: '#030712', border: '1px solid #1e1b4e',
      borderRadius: 8, overflow: 'hidden', position: 'relative',
    }}>
      <div style={{ padding: '4px 8px', borderBottom: '1px solid #1e1b4e', fontSize: 9, color: '#4b5563', fontFamily: 'monospace' }}>
        👁 Vision Feed
      </div>
      {frame ? (
        <img
          src={`data:image/jpeg;base64,${frame}`}
          style={{ width: '100%', height: 'calc(100% - 26px)', objectFit: 'cover' }}
          alt="vision"
        />
      ) : (
        <div style={{ height: 'calc(100% - 26px)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#374151', fontSize: 10 }}>
          no camera feed
        </div>
      )}
      {faceData?.name && (
        <div style={{
          position: 'absolute', bottom: 6, left: 6, right: 6,
          background: 'rgba(0,0,0,0.7)', borderRadius: 4, padding: '3px 6px',
          fontSize: 9, color: '#22d3ee', fontFamily: 'monospace',
        }}>
          👤 {faceData.name} {faceData.emotion ? `| ${faceData.emotion}` : ''}
        </div>
      )}
    </div>
  );
}

// ── Log panel ──────────────────────────────────────────────────────────────
function LogPanel() {
  const logs = useAxonStore((s) => s.logs);
  return (
    <div style={{
      flex: 1, background: '#030712', border: '1px solid #1e1b4e',
      borderRadius: 8, overflow: 'hidden', display: 'flex', flexDirection: 'column',
    }}>
      <div style={{ padding: '4px 8px', borderBottom: '1px solid #1e1b4e', fontSize: 9, color: '#4b5563', fontFamily: 'monospace' }}>
        📋 Log
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '4px 6px' }}>
        {logs.slice(0, 40).map((l: any, i) => (
          <div key={i} style={{ fontSize: 7, fontFamily: 'monospace', color: '#374151', marginBottom: 1, lineHeight: 1.5 }}>
            <span style={{ color: l.level === 'error' ? '#ef4444' : l.level === 'warn' ? '#f59e0b' : '#1e293b' }}>
              [{l.level ?? 'log'}]
            </span>{' '}
            {l.message ?? l.text ?? JSON.stringify(l).slice(0, 100)}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Memory placeholder ─────────────────────────────────────────────────────
function MemoryPlaceholder() {
  const memoryEvents = useAxonStore((s) => s.memoryEvents);
  return (
    <div style={{ padding: 8, overflowY: 'auto', height: '100%' }}>
      <div style={{ fontSize: 9, color: '#6366f1', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 8 }}>
        💾 Memory Stream
      </div>
      {memoryEvents.length === 0 && (
        <div style={{ fontSize: 9, color: '#374151' }}>no memory events yet</div>
      )}
      {memoryEvents.map((e: any, i) => (
        <div key={i} style={{
          fontSize: 8, color: '#94a3b8', marginBottom: 6, padding: '4px 6px',
          background: '#0a0014', border: '1px solid #1e1b4e', borderRadius: 4, lineHeight: 1.6,
        }}>
          <span style={{ color: '#22d3ee' }}>{e.type ?? 'memory'}</span>{' '}
          {e.content ?? e.text ?? JSON.stringify(e).slice(0, 80)}
        </div>
      ))}
    </div>
  );
}
