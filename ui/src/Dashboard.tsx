import { Suspense, lazy, useState } from 'react';
import { useAxonSocket } from './hooks/useAxonSocket';
import { useAxonStore } from './store/axonStore';
import BrainOverview from './components/dashboard/BrainOverview';
import ThoughtCompetition from './components/dashboard/ThoughtCompetition';
import MemoryPanel from './components/dashboard/MemoryPanel';
import BeliefsPanel from './components/dashboard/BeliefsPanel';
import ReflectionsPanel from './components/dashboard/ReflectionsPanel';
import PersonalityPanel from './components/dashboard/PersonalityPanel';
import KnowledgePanel from './components/dashboard/KnowledgePanel';
import ProvidersPanel from './components/dashboard/ProvidersPanel';
import ChatPanel from './components/ChatPanel';
import SystemLog from './components/dashboard/SystemLog';

const BrainCanvas = lazy(() => import('./components/BrainCanvas'));

// ── Tab definitions ──────────────────────────────────────────────────────
const TABS = [
  { id: 'overview',    label: '🧠 Overview',     desc: 'Neural state, emotion, regions' },
  { id: 'thoughts',    label: '💭 Thoughts',      desc: 'Thought competition & stream' },
  { id: 'memory',      label: '💾 Memory',        desc: 'Hierarchy, narratives, interests' },
  { id: 'beliefs',     label: '🧩 Beliefs',       desc: 'Drives, beliefs, goals, self-model' },
  { id: 'reflections', label: '🌀 Reflections',   desc: 'Autonomous thought & Hebbian events' },
  { id: 'personality', label: '🎭 Personality',   desc: 'Live trait sliders' },
  { id: 'knowledge',   label: '📚 Knowledge',     desc: 'Upload docs, force opinions' },
  { id: 'providers',   label: '🔌 Providers',     desc: 'LLM provider & model config' },
  { id: 'syslog',      label: '🖥 System Log',     desc: 'Startup errors & live logs' },
];

function TopBar() {
  const connected = useAxonStore((s) => s.connected);
  const ns        = useAxonStore((s) => s.neuralState);
  const thinking  = useAxonStore((s) => s.thinking);
  const lastTick  = useAxonStore((s) => s.lastTick);
  const { emit }  = useAxonSocket();

  const secAgo  = lastTick ? Math.round((Date.now() - lastTick) / 1000) : null;
  const alive   = secAgo !== null && secAgo < 10;
  const dotCol  = !connected ? '#ef4444' : alive ? '#00ff88' : '#fbbf24';
  const emo     = ns.emotion ?? {};

  const emotionEmoji: Record<string, string> = {
    happy: '😊', excited: '🤩', curious: '🤔', calm: '😌',
    neutral: '😐', sad: '😔', anxious: '😰', angry: '😠',
    bored: '😑', focused: '🎯', surprised: '😲', afraid: '😨',
  };

  return (
    <div style={{
      height: 44, display: 'flex', alignItems: 'center', gap: 16,
      padding: '0 16px', background: '#020205',
      borderBottom: '1px solid #1e1b4e', flexShrink: 0,
      fontFamily: 'monospace', userSelect: 'none',
    }}>
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{
          width: 28, height: 28, borderRadius: '50%',
          background: 'radial-gradient(circle at 40% 40%, #6366f133, #020205)',
          border: `1px solid ${dotCol}`, boxShadow: `0 0 8px ${dotCol}44`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 13,
        }}>🧠</div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 800, color: '#6366f1', letterSpacing: '0.15em' }}>AXON</div>
          <div style={{ fontSize: 7, color: '#374151', letterSpacing: '0.05em' }}>Neural Dashboard</div>
        </div>
      </div>

      <div style={{ width: 1, height: 24, background: '#1e1b4e' }} />

      {/* Connection */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
        <div style={{ width: 7, height: 7, borderRadius: '50%', background: dotCol, boxShadow: `0 0 6px ${dotCol}` }} />
        <span style={{ fontSize: 9, color: dotCol, fontWeight: 700 }}>
          {!connected ? 'OFFLINE' : alive ? `LIVE · tick ${ns.tick ?? '?'}` : 'STALE'}
        </span>
      </div>

      {/* Neurons */}
      <span style={{ fontSize: 9, color: '#22d3ee' }}>
        {ns.total_neurons
          ? `${(ns.total_neurons / 1e9).toFixed(2)}B neurons`
          : '—'}
      </span>

      {/* Emotion */}
      {emo.current && (
        <>
          <div style={{ width: 1, height: 24, background: '#1e1b4e' }} />
          <span style={{ fontSize: 9, color: '#94a3b8' }}>
            {emotionEmoji[emo.current?.toLowerCase()] ?? '🧠'} {emo.current}
            <span style={{ color: '#374151' }}> v:{(emo.valence ?? 0).toFixed(2)} a:{(emo.arousal ?? 0).toFixed(2)}</span>
          </span>
        </>
      )}

      {/* Thinking */}
      {thinking && (
        <span style={{ fontSize: 9, color: '#6366f1', animation: 'blink 1s infinite' }}>⚙ thinking…</span>
      )}

      <div style={{ flex: 1 }} />

      {/* Speed controls */}
      <span style={{ fontSize: 8, color: '#374151' }}>speed:</span>
      {[['💤', 0.5], ['🐢', 5], ['✅', 10], ['⚡', 20], ['🚀', 50]].map(([emoji, hz]) => (
        <button key={String(hz)} onClick={() => emit('set_speed', { hz: Number(hz) })} style={{
          padding: '2px 6px', borderRadius: 4, background: '#080818',
          border: '1px solid #1e1b4e', color: '#64748b', cursor: 'pointer', fontSize: 9,
        }}>{emoji}</button>
      ))}

      {/* Nav to main */}
      <a href="/" style={{
        padding: '3px 10px', borderRadius: 5,
        background: '#1e1b4e22', border: '1px solid #6366f133',
        color: '#6366f1', fontSize: 8, textDecoration: 'none', fontWeight: 600,
      }}>← Main UI</a>

      <style>{`
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
      `}</style>
    </div>
  );
}

export default function Dashboard() {
  useAxonSocket();
  const [activeTab, setActiveTab] = useState('overview');
  const [showCanvas, setShowCanvas] = useState(true);

  const PANEL_MAP: Record<string, React.ReactNode> = {
    overview:    <BrainOverview />,
    thoughts:    <ThoughtCompetition />,
    memory:      <MemoryPanel />,
    beliefs:     <BeliefsPanel />,
    reflections: <ReflectionsPanel />,
    personality: <PersonalityPanel />,
    knowledge:   <KnowledgePanel />,
    providers:   <ProvidersPanel />,
    syslog:      <SystemLog />,
  };

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100vh',
      background: '#020205', fontFamily: "'Inter','Segoe UI',sans-serif",
      color: '#e2e8f0', overflow: 'hidden',
    }}>
      <TopBar />

      {/* Main layout: sidebar tabs | content | brain canvas + chat */}
      <div style={{
        flex: 1, display: 'grid',
        gridTemplateColumns: '180px 1fr 320px',
        gap: 4, padding: 4, minHeight: 0, overflow: 'hidden',
      }}>

        {/* ── LEFT: Tab navigation ── */}
        <div style={{
          display: 'flex', flexDirection: 'column', gap: 2,
          background: '#030712', border: '1px solid #1e1b4e',
          borderRadius: 8, padding: '8px 4px', overflowY: 'auto',
        }}>
          {TABS.map((tab) => {
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  padding: '8px 10px', borderRadius: 6, cursor: 'pointer',
                  background: active ? '#1e1b4e' : 'transparent',
                  border: `1px solid ${active ? '#6366f1' : 'transparent'}`,
                  textAlign: 'left', transition: 'all 0.15s',
                  boxShadow: active ? '0 0 8px #6366f122' : 'none',
                }}
              >
                <div style={{ fontSize: 10, color: active ? '#a5b4fc' : '#64748b', fontWeight: active ? 600 : 400 }}>
                  {tab.label}
                </div>
                <div style={{ fontSize: 7, color: active ? '#4b5563' : '#1e293b', marginTop: 1 }}>
                  {tab.desc}
                </div>
              </button>
            );
          })}

          <div style={{ flex: 1 }} />

          {/* Canvas toggle */}
          <button
            onClick={() => setShowCanvas((v) => !v)}
            style={{
              padding: '6px 10px', borderRadius: 6, cursor: 'pointer',
              background: showCanvas ? '#1e1b4e' : 'transparent',
              border: '1px solid #1e1b4e', fontSize: 9, color: '#64748b',
            }}
          >
            {showCanvas ? '🧠 Hide 3D' : '🧠 Show 3D'}
          </button>
        </div>

        {/* ── CENTER: Active panel ── */}
        <div style={{
          background: '#030712', border: '1px solid #1e1b4e',
          borderRadius: 8, overflow: 'hidden', minHeight: 0,
        }}>
          {PANEL_MAP[activeTab]}
        </div>

        {/* ── RIGHT: Brain canvas (top) + Chat (bottom) ── */}
        <div style={{
          display: 'flex', flexDirection: 'column', gap: 4, minHeight: 0,
        }}>
          {/* 3D Brain */}
          {showCanvas && (
            <div style={{
              flex: 2, border: '1px solid #1e1b4e', borderRadius: 8,
              overflow: 'hidden', position: 'relative', minHeight: 0,
            }}>
              <Suspense fallback={
                <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#020205', color: '#374151', fontSize: 11 }}>
                  loading 3D…
                </div>
              }>
                <BrainCanvas />
              </Suspense>
              {/* Live overlay */}
              <LiveBrainOverlay />
            </div>
          )}

          {/* Chat */}
          <div style={{ flex: showCanvas ? 1 : 3, border: '1px solid #1e1b4e', borderRadius: 8, overflow: 'hidden', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <ChatPanel />
          </div>
        </div>

      </div>
    </div>
  );
}

// ── Live overlay on top of 3D brain ─────────────────────────────────────
function LiveBrainOverlay() {
  const ns = useAxonStore((s) => s.neuralState);
  const emo = ns.emotion ?? {};
  const topC = (ns.top_clusters ?? []).slice(0, 3);

  return (
    <div style={{
      position: 'absolute', inset: 0, pointerEvents: 'none',
      display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: 8,
    }}>
      {/* Top HUD */}
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <div style={{
          padding: '3px 7px', background: 'rgba(2,2,5,0.8)',
          border: '1px solid #1e1b4e', borderRadius: 5, fontSize: 8, color: '#6366f1', fontFamily: 'monospace',
        }}>
          tick:{ns.tick ?? '?'} · ε:{(ns.explore_eps ?? 0).toFixed(2)}
        </div>
        <div style={{
          padding: '3px 7px', background: 'rgba(2,2,5,0.8)',
          border: '1px solid #1e1b4e', borderRadius: 5, fontSize: 8, fontFamily: 'monospace',
          color: (emo.valence ?? 0) > 0 ? '#4ade80' : '#f43f5e',
        }}>
          {emo.current ?? 'neutral'} v:{(emo.valence ?? 0).toFixed(2)}
        </div>
      </div>

      {/* Bottom: active clusters */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
        {topC.map((c, i) => (
          <div key={i} style={{
            padding: '2px 6px', background: 'rgba(2,2,5,0.85)',
            border: '1px solid #6366f144', borderRadius: 4,
            fontSize: 7, color: '#a5b4fc', fontFamily: 'monospace',
          }}>
            {c.name} {(c.activation * 100).toFixed(0)}%
          </div>
        ))}
      </div>
    </div>
  );
}
