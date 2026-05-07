import { useAxonStore } from '../../store/axonStore';
import { LineChart, Line, ResponsiveContainer, Tooltip, YAxis } from 'recharts';

const REGION_COLORS: Record<string, string> = {
  prefrontal: '#6366f1', hippocampus: '#22d3ee', amygdala: '#f43f5e',
  visual: '#a3e635', auditory: '#fb923c', language: '#e879f9',
  thalamus: '#fbbf24', cerebellum: '#34d399', association: '#818cf8',
  default_mode: '#94a3b8', social: '#f472b6', metacognition: '#c084fc',
};

const NM_COLORS: Record<string, string> = {
  dopamine: '#a855f7', serotonin: '#22c55e', norepinephrine: '#f97316',
  acetylcholine: '#38bdf8', gaba: '#f43f5e', glutamate: '#fbbf24',
};

function MiniSparkline({ data, color }: { data: number[]; color: string }) {
  const pts = data.map((v, i) => ({ i, v }));
  return (
    <ResponsiveContainer width="100%" height={28}>
      <LineChart data={pts}>
        <YAxis domain={[0, 1]} hide />
        <Line type="monotone" dataKey="v" stroke={color} dot={false} strokeWidth={1.5} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function StatCard({ label, value, sub, color = '#6366f1', glow }: {
  label: string; value: string | number; sub?: string; color?: string; glow?: boolean;
}) {
  return (
    <div style={{
      padding: '10px 12px', background: '#080818', border: `1px solid ${color}33`,
      borderRadius: 8, boxShadow: glow ? `0 0 12px ${color}33` : 'none',
    }}>
      <div style={{ fontSize: 8, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700, color, fontFamily: 'monospace', letterSpacing: '0.04em' }}>{value}</div>
      {sub && <div style={{ fontSize: 8, color: '#475569', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function EmotionOrb({ emotion, valence, arousal }: { emotion: string; valence: number; arousal: number }) {
  const col = valence > 0.2 ? '#4ade80' : valence < -0.2 ? '#f43f5e' : '#94a3b8';
  const emoji: Record<string, string> = {
    happy: '😊', excited: '🤩', curious: '🤔', calm: '😌',
    neutral: '😐', sad: '😔', anxious: '😰', angry: '😠',
    bored: '😑', focused: '🎯', surprised: '😲', afraid: '😨',
  };
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      width: 90, height: 90, borderRadius: '50%',
      background: `radial-gradient(circle at 40% 40%, ${col}22, #020205)`,
      border: `2px solid ${col}66`,
      boxShadow: `0 0 20px ${col}44`,
      flexShrink: 0,
    }}>
      <div style={{ fontSize: 28 }}>{emoji[emotion?.toLowerCase()] ?? '🧠'}</div>
      <div style={{ fontSize: 9, color: col, fontWeight: 700, letterSpacing: '0.05em', marginTop: 2 }}>
        {emotion?.toUpperCase()}
      </div>
      <div style={{ fontSize: 7, color: '#475569', fontFamily: 'monospace' }}>
        v:{valence?.toFixed(2)} a:{arousal?.toFixed(2)}
      </div>
    </div>
  );
}

export default function BrainOverview() {
  const ns = useAxonStore((s) => s.neuralState);
  const rh = useAxonStore((s) => s.rewardHistory);
  const sh = useAxonStore((s) => s.surpriseHistory);
  const regionHistory = useAxonStore((s) => s.regionHistory);


  const emo = ns.emotion ?? {};
  const regions = ns.regions ?? {};
  const nm = ns.neuromod ?? {};
  const cog = ns.cognitive_state ?? {};
  const meta = ns.meta ?? {};

  return (
    <div style={{ padding: 12, overflowY: 'auto', height: '100%' }}>

      {/* Top stat row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8, marginBottom: 12 }}>
        <StatCard label="Neurons" value={(ns.total_neurons ?? 0) >= 1e9
          ? `${((ns.total_neurons ?? 0) / 1e9).toFixed(2)}B`
          : (ns.total_neurons?.toLocaleString() ?? '—')}
          color="#22d3ee" glow />
        <StatCard label="Tick" value={ns.tick ?? '—'} color="#6366f1" />
        <StatCard label="Exploration ε" value={(ns.explore_eps ?? 0).toFixed(3)} color="#a855f7"
          sub={`meta: ${(meta.explore_rate ?? 0).toFixed(2)}`} />
        <StatCard label="Reward μ" value={((ns.temporal_reward as any)?.mean ?? 0).toFixed(3)} color="#4ade80"
          sub={`regret: ${(ns.critic?.regret ?? 0).toFixed(3)}`} />
        <StatCard label="Surprise" value={(ns.prediction_surprise ?? 0).toFixed(3)} color="#f97316"
          sub={`depth: ${ns.temporal_depth ?? 0}`} />
      </div>

      {/* Emotion + NM row */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
        <EmotionOrb emotion={emo.current ?? 'neutral'} valence={emo.valence ?? 0} arousal={emo.arousal ?? 0} />

        {/* NM bars */}
        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px' }}>
          {Object.entries(NM_COLORS).map(([k, col]) => (
            <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ fontSize: 8, color: '#64748b', width: 65, flexShrink: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {k.charAt(0).toUpperCase() + k.slice(1, 5)}
              </div>
              <div style={{ flex: 1, background: '#0d1117', borderRadius: 3, height: 6, overflow: 'hidden' }}>
                <div style={{
                  width: `${Math.round((nm[k] ?? 0) * 100)}%`, height: '100%',
                  background: col, borderRadius: 3, transition: 'width 0.5s ease',
                  boxShadow: `0 0 4px ${col}`,
                }} />
              </div>
              <span style={{ fontSize: 7, color: col, width: 28, textAlign: 'right', fontFamily: 'monospace' }}>
                {(nm[k] ?? 0).toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Cognitive state */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8, marginBottom: 12 }}>
        {[
          { k: 'confidence', v: cog.confidence ?? 0.5, c: '#4ade80' },
          { k: 'uncertainty', v: cog.uncertainty ?? 0.5, c: '#f97316' },
          { k: 'urgency', v: cog.urgency ?? 0, c: '#ef4444' },
        ].map(({ k, v, c }) => (
          <div key={k} style={{ padding: '6px 10px', background: '#080818', border: `1px solid ${c}22`, borderRadius: 6 }}>
            <div style={{ fontSize: 8, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 4 }}>{k}</div>
            <div style={{ background: '#0d1117', borderRadius: 4, height: 8, overflow: 'hidden' }}>
              <div style={{ width: `${Math.round(v * 100)}%`, height: '100%', background: c, borderRadius: 4, transition: 'width 0.5s', boxShadow: `0 0 6px ${c}` }} />
            </div>
            <div style={{ fontSize: 8, color: c, textAlign: 'right', fontFamily: 'monospace', marginTop: 2 }}>{(v * 100).toFixed(0)}%</div>
          </div>
        ))}
      </div>

      {/* Region grid */}
      <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>🧠 Brain Regions — Live Activation</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 20px' }}>
          {Object.entries(regions).sort((a, b) => (b[1] as number) - (a[1] as number)).map(([rk, v]) => {
            const col = REGION_COLORS[rk] ?? '#6366f1';
            const hist = regionHistory[rk] ?? [];
            return (
              <div key={rk}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                  <span style={{ fontSize: 8, color: '#64748b' }}>{rk.replace(/_/g, ' ')}</span>
                  <span style={{ fontSize: 7, color: col, fontFamily: 'monospace' }}>{((v as number) * 100).toFixed(0)}%</span>
                </div>
                <div style={{ position: 'relative', height: 28, background: '#0a0a1a', borderRadius: 4, overflow: 'hidden', marginBottom: 4 }}>
                  {/* activation fill */}
                  <div style={{ position: 'absolute', bottom: 0, left: 0, width: `${(v as number) * 100}%`, height: '100%', background: `${col}22`, transition: 'width 0.5s' }} />
                  {/* sparkline */}
                  <div style={{ position: 'absolute', inset: 0 }}>
                    <MiniSparkline data={hist} color={col} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Reward + Surprise history */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>
        {[
          { title: 'Reward History', data: rh, color: '#4ade80' },
          { title: 'Surprise History', data: sh, color: '#f97316' },
        ].map(({ title, data, color }) => (
          <div key={title} style={{ padding: '6px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
            <div style={sTitle}>{title}</div>
            <ResponsiveContainer width="100%" height={60}>
              <LineChart data={data.map((v, i) => ({ i, v }))}>
                <YAxis domain={[-0.1, 1]} hide />
                <Tooltip
                  contentStyle={{ background: '#0a0014', border: '1px solid #1e1b4e', fontSize: 9 }}
                  formatter={(v: any) => [(v as number).toFixed(3)]}
                  labelFormatter={() => ''}
                />
                <Line type="monotone" dataKey="v" stroke={color} dot={false} strokeWidth={1.5} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>

      {/* Meta + Strategy */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <div style={{ padding: '6px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
          <div style={sTitle}>🎛 Meta-Controller</div>
          {[
            { k: 'Mood', v: meta.mood ?? (ns as any).cognitive_state?.mood ?? 'stable', isText: true },
            { k: 'Explore ε', v: (meta.explore_rate ?? 0).toFixed(3), isText: true },
            { k: 'Stability', v: (meta.stability ?? 0).toFixed(3), isText: true },
            { k: 'Cluster Wear', v: (ns.cluster_wear ?? 0).toFixed(3), isText: true },
          ].map(({ k, v }) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
              <span style={{ fontSize: 8, color: '#64748b' }}>{k}</span>
              <span style={{ fontSize: 8, color: '#a78bfa', fontFamily: 'monospace' }}>{v}</span>
            </div>
          ))}
        </div>
        <div style={{ padding: '6px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
          <div style={sTitle}>📚 Strategy Library</div>
          {[
            { k: 'Strategies', v: ns.strategy_lib?.count ?? 0 },
            { k: 'Avg Score', v: (ns.strategy_lib?.avg_score ?? 0).toFixed(3) },
            { k: 'Hesitations', v: ns.critic?.hesitations ?? 0 },
            { k: 'Temporal Depth', v: ns.temporal_depth ?? 0 },
          ].map(({ k, v }) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
              <span style={{ fontSize: 8, color: '#64748b' }}>{k}</span>
              <span style={{ fontSize: 8, color: '#22d3ee', fontFamily: 'monospace' }}>{v}</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}

const sTitle: React.CSSProperties = {
  fontSize: 9, fontWeight: 700, letterSpacing: '0.1em',
  textTransform: 'uppercase', color: '#6366f1', marginBottom: 8,
};
