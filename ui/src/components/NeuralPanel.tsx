import { useAxonStore } from '../store/axonStore';

const REGION_COLORS: Record<string, string> = {
  prefrontal:     '#6366f1',
  hippocampus:    '#22d3ee',
  amygdala:       '#f43f5e',
  visual:         '#a3e635',
  auditory:       '#fb923c',
  language:       '#e879f9',
  thalamus:       '#fbbf24',
  cerebellum:     '#34d399',
  association:    '#818cf8',
  default_mode:   '#94a3b8',
  social:         '#f472b6',
  metacognition:  '#c084fc',
};

const NM_COLORS: Record<string, string> = {
  dopamine:       '#a855f7',
  serotonin:      '#22c55e',
  norepinephrine: '#f97316',
  acetylcholine:  '#38bdf8',
  gaba:           '#f43f5e',
  glutamate:      '#fbbf24',
};

const NM_LABELS: Record<string, string> = {
  dopamine:       'Dopamine',
  serotonin:      'Serotonin',
  norepinephrine: 'Norepi',
  acetylcholine:  'ACholine',
  gaba:           'GABA',
  glutamate:      'Glut.',
};

const TRAIT_COLORS: Record<string, string> = {
  openness:          '#818cf8',
  conscientiousness: '#34d399',
  extraversion:      '#fbbf24',
  agreeableness:     '#f472b6',
  neuroticism:       '#f43f5e',
  curiosity:         '#22d3ee',
  empathy:           '#a3e635',
  risk:              '#fb923c',
  creativity:        '#c084fc',
  stability:         '#6ee7b7',
};

function Bar({ label, value, color, max = 1 }: { label: string; value: number; color: string; max?: number }) {
  const pct = Math.round((value / max) * 100);
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '70px 1fr 28px', alignItems: 'center', gap: '3px 6px', marginBottom: 3 }}>
      <span style={{ fontSize: 9, color: '#94a3b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
      <div style={{ background: '#0d1117', borderRadius: 3, height: 6, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.4s ease' }} />
      </div>
      <span style={{ fontSize: 8, color, textAlign: 'right', fontFamily: 'monospace' }}>{pct}%</span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 12, padding: '6px 8px', background: '#0a0014', border: '1px solid #1e1b4e', borderRadius: 6 }}>
      <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#6366f1', marginBottom: 6 }}>{title}</div>
      {children}
    </div>
  );
}

export default function NeuralPanel() {
  const ns = useAxonStore((s) => s.neuralState);
  const connected = useAxonStore((s) => s.connected);
  const lastTick = useAxonStore((s) => s.lastTick);

  const secAgo = lastTick ? Math.round((Date.now() - lastTick) / 1000) : null;
  const alive = secAgo !== null && secAgo < 10;

  const emo = ns.emotion ?? {};
  const nm  = ns.neuromod ?? {};
  const regions = ns.regions ?? {};
  const personality = ns.personality ?? {};
  const cog = ns.cognitive_state ?? {};
  const conflict = ns.conflict ?? {};

  const emotionColor = (v: number) => {
    if (v > 0.3) return '#4ade80';
    if (v < -0.3) return '#f43f5e';
    return '#94a3b8';
  };

  return (
    <div style={{ overflowY: 'auto', height: '100%', padding: '6px 4px', fontSize: 11, color: '#e2e8f0' }}>

      {/* Header / Status */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, padding: '4px 6px', background: '#0a0014', border: '1px solid #1e1b4e', borderRadius: 6 }}>
        <span style={{ fontSize: 8, fontFamily: 'monospace', color: alive ? '#00ff88' : '#ef4444' }}>
          {connected ? (alive ? `🟢 tick:${ns.tick ?? '?'} (+${secAgo}s)` : `🟡 stale`) : '🔴 disconnected'}
        </span>
        <span style={{ fontSize: 8, color: '#475569', fontFamily: 'monospace' }}>
          {ns.total_neurons?.toLocaleString() ?? '—'} neurons
        </span>
      </div>

      {/* Emotion */}
      <Section title="⚡ Emotional State">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <div style={{
            width: 52, height: 52, borderRadius: '50%',
            background: `radial-gradient(circle, ${emotionColor(emo.valence ?? 0)}33, #0a0014)`,
            border: `2px solid ${emotionColor(emo.valence ?? 0)}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 18, flexShrink: 0,
          }}>
            {getEmoji((emo.current ?? (emo as any)?.emotion) ?? 'neutral')}
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 13, color: emotionColor(emo.valence ?? 0), letterSpacing: '0.04em' }}>
              {(emo.current ?? (emo as any)?.emotion) ?? 'neutral'}
            </div>
            <div style={{ fontSize: 8, color: '#64748b', fontFamily: 'monospace' }}>
              v:{(emo.valence ?? 0).toFixed(2)} a:{(emo.arousal ?? 0).toFixed(2)} i:{(emo.intensity ?? 0).toFixed(2)}
            </div>
          </div>
        </div>
        <Bar label="Valence"  value={(emo.valence  ?? 0) / 2 + 0.5} color={emotionColor(emo.valence ?? 0)} />
        <Bar label="Arousal"  value={(emo.arousal  ?? 0) / 2 + 0.5} color="#fb923c" />
        <Bar label="Intensity" value={emo.intensity ?? 0}            color="#a855f7" />
      </Section>

      {/* Neuromodulators */}
      <Section title="🧪 Neuromodulators">
        {Object.entries(NM_LABELS).map(([k, label]) => (
          <Bar key={k} label={label} value={nm[k] ?? 0} color={NM_COLORS[k] ?? '#94a3b8'} />
        ))}
      </Section>

      {/* Brain Regions */}
      <Section title="🧠 Brain Regions">
        {Object.entries(regions).sort((a, b) => b[1] - a[1]).map(([rk, val]) => (
          <Bar key={rk} label={rk.replace(/_/g, ' ')} value={val} color={REGION_COLORS[rk] ?? '#6366f1'} />
        ))}
      </Section>

      {/* Cognitive State */}
      <Section title="⚙ Cognitive State">
        <Bar label="Confidence"  value={cog.confidence  ?? 0.5} color="#4ade80" />
        <Bar label="Uncertainty" value={cog.uncertainty ?? 0.5} color="#f97316" />
        <Bar label="Urgency"     value={cog.urgency     ?? 0.1} color="#ef4444" />
        <Bar label="Coherence"   value={ns.temporal_momentum ?? 0} color="#a78bfa" />
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 10px', marginTop: 6, fontSize: 8, fontFamily: 'monospace', color: '#64748b' }}>
          <span>Surprise <span style={{ color: '#f97316' }}>{(ns.prediction_surprise ?? 0).toFixed(3)}</span></span>
          <span>ε <span style={{ color: '#4ade80' }}>{(ns.explore_eps ?? 0).toFixed(3)}</span></span>
          <span>Reward <span style={{ color: '#a855f7' }}>{((ns.temporal_reward as any)?.mean ?? 0).toFixed(3)}</span></span>
          <span>Regret <span style={{ color: '#fb923c' }}>{(ns.critic?.regret ?? 0).toFixed(3)}</span></span>
          <span>MemDepth <span style={{ color: '#818cf8' }}>{ns.temporal_depth ?? 0}</span></span>
        </div>
      </Section>

      {/* Conflict */}
      {(conflict.dominant || (conflict.winner_set?.length ?? 0) > 0) && (
        <Section title="⚔ Conflict Engine">
          <div style={{ fontSize: 8, color: '#c4b5fd' }}>
            {conflict.dominant ?? conflict.winner_set?.join(', ') ?? '—'}
          </div>
        </Section>
      )}

      {/* Personality */}
      <Section title="🎭 Personality">
        {Object.entries(personality).slice(0, 10).map(([k, v]) => (
          <Bar key={k} label={k.slice(0, 9)} value={v as number} color={TRAIT_COLORS[k] ?? '#818cf8'} />
        ))}
      </Section>

      {/* Active thoughts */}
      {(ns.thoughts?.length ?? 0) > 0 && (
        <Section title="💭 Recent Thoughts">
          {(ns.thoughts ?? []).map((t, i) => (
            <div key={i} style={{ fontSize: 8, color: '#94a3b8', marginBottom: 4, paddingLeft: 4, borderLeft: '2px solid #6366f1', lineHeight: 1.5 }}>
              {typeof t === 'string' ? t : (t as any).text ?? JSON.stringify(t)}
            </div>
          ))}
        </Section>
      )}

    </div>
  );
}

function getEmoji(emotion: string): string {
  const map: Record<string, string> = {
    happy: '😊', excited: '🤩', curious: '🤔', calm: '😌',
    neutral: '😐', sad: '😔', anxious: '😰', angry: '😠',
    bored: '😑', focused: '🎯', surprised: '😲', afraid: '😨',
  };
  return map[emotion.toLowerCase()] ?? '🧠';
}
