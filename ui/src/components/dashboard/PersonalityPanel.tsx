import { useState } from 'react';
import { useAxonStore } from '../../store/axonStore';
import { useAxonSocket } from '../../hooks/useAxonSocket';

const TRAITS = [
  { key: 'openness',          label: 'Openness',           color: '#818cf8', desc: 'Curiosity, creativity, experience-seeking' },
  { key: 'conscientiousness', label: 'Conscientious.',     color: '#34d399', desc: 'Discipline, organization, goal-focus' },
  { key: 'extraversion',      label: 'Extraversion',       color: '#fbbf24', desc: 'Social energy, enthusiasm, assertiveness' },
  { key: 'agreeableness',     label: 'Agreeableness',      color: '#f472b6', desc: 'Empathy, cooperation, trust' },
  { key: 'neuroticism',       label: 'Neuroticism',        color: '#f43f5e', desc: 'Anxiety, mood swings, emotional reactivity' },
  { key: 'curiosity',         label: 'Curiosity',          color: '#22d3ee', desc: 'Drive to explore and learn' },
  { key: 'empathy',           label: 'Empathy',            color: '#a3e635', desc: 'Sensitivity to others\' states' },
  { key: 'risk',              label: 'Risk Tolerance',     color: '#fb923c', desc: 'Willingness to explore unknown paths' },
  { key: 'creativity',        label: 'Creativity',         color: '#c084fc', desc: 'Novel associations and idea generation' },
  { key: 'stability',         label: 'Stability',          color: '#6ee7b7', desc: 'Resistance to sudden parameter changes' },
];

export default function PersonalityPanel() {
  const ns = useAxonStore((s) => s.neuralState);
  const { emit } = useAxonSocket();
  const personality = ns.personality ?? {};

  // Local slider state — starts from live values
  const [overrides, setOverrides] = useState<Record<string, number>>({});
  const [pending, setPending] = useState(false);
  const [saved, setSaved] = useState(false);

  const getValue = (key: string) => overrides[key] ?? (personality[key] ?? 0.5);

  const handleSlider = (key: string, val: number) => {
    setOverrides((prev) => ({ ...prev, [key]: val }));
    setSaved(false);
  };

  const handleApply = () => {
    setPending(true);
    emit('set_personality', overrides);
    setTimeout(() => {
      setPending(false);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    }, 600);
  };

  const handleReset = () => {
    setOverrides({});
    setSaved(false);
  };

  const hasChanges = Object.keys(overrides).length > 0;

  return (
    <div style={{ padding: 12, overflowY: 'auto', height: '100%' }}>

      <div style={{ marginBottom: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={sTitle}>🎭 Personality Vector</div>
        <div style={{ display: 'flex', gap: 6 }}>
          {hasChanges && (
            <button onClick={handleReset} style={btnStyle('#374151', '#1e293b')}>Reset</button>
          )}
          <button
            onClick={handleApply}
            disabled={!hasChanges || pending}
            style={btnStyle(saved ? '#4ade80' : '#6366f1', saved ? '#052e16' : '#1e1b4e')}
          >
            {pending ? 'Applying…' : saved ? '✓ Applied' : 'Apply Changes'}
          </button>
        </div>
      </div>

      <div style={{ fontSize: 8, color: '#374151', marginBottom: 12, lineHeight: 1.6 }}>
        Personality drifts over time via Hebbian learning. These sliders push immediate overrides —
        changes take effect on the next cognitive cycle and begin to drift from there.
      </div>

      {TRAITS.map(({ key, label, color, desc }) => {
        const v = getValue(key);
        const live = personality[key] ?? 0.5;
        const isDirty = overrides[key] !== undefined;
        return (
          <div key={key} style={{
            marginBottom: 12, padding: '8px 10px',
            background: isDirty ? `${color}08` : '#080818',
            border: `1px solid ${isDirty ? color + '33' : '#1e1b4e'}`,
            borderRadius: 8, transition: 'all 0.2s',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <div>
                <span style={{ fontSize: 10, fontWeight: 600, color }}>{label}</span>
                <span style={{ fontSize: 7, color: '#374151', marginLeft: 6 }}>{desc}</span>
              </div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                {isDirty && (
                  <span style={{ fontSize: 7, color: '#64748b', fontFamily: 'monospace' }}>
                    was: {(live * 100).toFixed(0)}%
                  </span>
                )}
                <span style={{ fontSize: 10, color, fontFamily: 'monospace', fontWeight: 700 }}>
                  {(v * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            {/* Track with live marker */}
            <div style={{ position: 'relative', height: 20, display: 'flex', alignItems: 'center' }}>
              <input
                type="range" min={0} max={1} step={0.01}
                value={v}
                onChange={(e) => handleSlider(key, parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: color, cursor: 'pointer' }}
              />
              {/* Live value marker */}
              {isDirty && (
                <div style={{
                  position: 'absolute', left: `${live * 100}%`, top: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: 2, height: 12, background: '#374151', borderRadius: 1, pointerEvents: 'none',
                }} />
              )}
            </div>

            {/* Value effect hints */}
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
              <span style={{ fontSize: 7, color: '#1e293b' }}>conservative</span>
              <span style={{ fontSize: 7, color: '#1e293b' }}>extreme</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function btnStyle(bg: string, border: string): React.CSSProperties {
  return {
    padding: '4px 10px', borderRadius: 6,
    background: `${bg}22`, border: `1px solid ${border}`,
    color: bg, fontSize: 9, fontWeight: 600, cursor: 'pointer',
  };
}

const sTitle: React.CSSProperties = {
  fontSize: 9, fontWeight: 700, letterSpacing: '0.1em',
  textTransform: 'uppercase', color: '#6366f1', margin: 0,
};
