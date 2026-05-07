import { useState, useEffect } from 'react';
import { useAxonSocket } from '../../hooks/useAxonSocket';
import { useAxonStore } from '../../store/axonStore';

interface Provider {
  id: string;
  label: string;
  icon: string;
  models: string[];
  needsKey: boolean;
  placeholder: string;
  color: string;
}

const PROVIDERS: Provider[] = [
  {
    id: 'lmstudio', label: 'LM Studio', icon: '🖥️',
    models: ['local-model'],
    needsKey: false, placeholder: 'http://localhost:1234',
    color: '#6366f1',
  },
  {
    id: 'openai', label: 'OpenAI', icon: '🟢',
    models: ['gpt-4o', 'gpt-4-turbo', 'gpt-4o-mini', 'gpt-3.5-turbo'],
    needsKey: true, placeholder: 'sk-…',
    color: '#4ade80',
  },
  {
    id: 'anthropic', label: 'Anthropic', icon: '🔷',
    models: ['claude-opus-4-5', 'claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307'],
    needsKey: true, placeholder: 'sk-ant-…',
    color: '#fb923c',
  },
  {
    id: 'gemini', label: 'Google Gemini', icon: '💫',
    models: ['gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash'],
    needsKey: true, placeholder: 'AIza…',
    color: '#22d3ee',
  },
  {
    id: 'groq', label: 'Groq', icon: '⚡',
    models: ['llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768'],
    needsKey: true, placeholder: 'gsk_…',
    color: '#f472b6',
  },
];

export default function ProvidersPanel() {
  const { emit } = useAxonSocket();
  const lmStatus = useAxonStore((s) => s.lmStatus);
  const [status, setStatus] = useState<any>(null);
  const [selected, setSelected] = useState<string>('lmstudio');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState('http://localhost:1234');
  const [applying, setApplying] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  // Fetch current provider status
  useEffect(() => {
    fetch('/api/status').then((r) => r.json()).then((d) => {
      if (d.provider) setSelected(d.provider);
      if (d.model) setSelectedModel(d.model);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (lmStatus) setStatus(lmStatus);
  }, [lmStatus]);

  const handleReprobe = () => {
    emit('reprobe_lm');
    setResult('Probing LM Studio…');
    setTimeout(() => setResult(null), 3000);
  };

  const handleApply = async () => {
    setApplying(true);
    setResult(null);
    const prov = PROVIDERS.find((p) => p.id === selected)!;
    const payload: any = { provider: selected };
    if (selectedModel) payload.model = selectedModel;
    if (prov.needsKey && apiKey) payload.api_key = apiKey;
    if (!prov.needsKey) payload.base_url = baseUrl;
    try {
      const res = await fetch('/api/status'); // check if engine up
      const st = await res.json();
      if (st.running) {
        emit('update_provider', payload);
        setResult('✓ Provider switching…');
      } else {
        // Store in providers.json via direct API
        await fetch('/api/brain/personality', { method: 'GET' });
        setResult('Engine not running — provider will apply on next start');
      }
    } catch (e: any) {
      setResult(`Error: ${e.message}`);
    } finally {
      setApplying(false);
    }
  };

  const activeProvider = PROVIDERS.find((p) => p.id === selected)!;

  return (
    <div style={{ padding: 12, overflowY: 'auto', height: '100%' }}>

      {/* Current status */}
      <div style={{ marginBottom: 12, padding: '8px 10px', background: '#080818', border: '1px solid #1e1b4e', borderRadius: 8 }}>
        <div style={sTitle}>🔌 Current LLM Status</div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { k: 'Provider', v: status?.provider ?? lmStatus?.provider ?? '—' },
            { k: 'Model', v: status?.model ?? lmStatus?.model ?? '—' },
            { k: 'Status', v: status?.connected ? '🟢 connected' : '🔴 disconnected' },
            { k: 'Latency', v: status?.latency ? `${status.latency}ms` : '—' },
          ].map(({ k, v }) => (
            <div key={k}>
              <div style={{ fontSize: 7, color: '#374151', textTransform: 'uppercase' }}>{k}</div>
              <div style={{ fontSize: 10, color: '#e2e8f0', fontFamily: 'monospace' }}>{v}</div>
            </div>
          ))}
        </div>
        <button onClick={handleReprobe} style={{ ...btnStyle('#22d3ee', '#0a2540'), marginTop: 8 }}>
          🔍 Re-probe LM Studio
        </button>
      </div>

      {/* Provider selector */}
      <div style={{ marginBottom: 12 }}>
        <div style={sTitle}>🔀 Switch Provider</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
          {PROVIDERS.map((p) => (
            <button
              key={p.id}
              onClick={() => { setSelected(p.id); setSelectedModel(p.models[0]); setResult(null); }}
              style={{
                padding: '8px 10px', borderRadius: 8, cursor: 'pointer',
                background: selected === p.id ? `${p.color}15` : '#080818',
                border: `1px solid ${selected === p.id ? p.color : '#1e1b4e'}`,
                boxShadow: selected === p.id ? `0 0 10px ${p.color}33` : 'none',
                textAlign: 'left', transition: 'all 0.15s',
              }}
            >
              <div style={{ fontSize: 11, marginBottom: 2 }}>{p.icon} <span style={{ color: selected === p.id ? p.color : '#94a3b8', fontWeight: 600 }}>{p.label}</span></div>
              <div style={{ fontSize: 7, color: '#374151' }}>{p.needsKey ? 'API key required' : 'Local / no key'}</div>
            </button>
          ))}
        </div>

        {/* Config for selected */}
        <div style={{ padding: '10px 12px', background: `${activeProvider.color}08`, border: `1px solid ${activeProvider.color}33`, borderRadius: 8 }}>
          <div style={{ fontSize: 9, color: activeProvider.color, fontWeight: 600, marginBottom: 8 }}>
            {activeProvider.icon} {activeProvider.label} Configuration
          </div>

          {/* Model selector */}
          <div style={{ marginBottom: 8 }}>
            <label style={{ fontSize: 8, color: '#64748b', display: 'block', marginBottom: 3 }}>Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                width: '100%', padding: '5px 8px',
                background: '#0a0014', border: '1px solid #1e1b4e',
                borderRadius: 5, color: '#e2e8f0', fontSize: 10, outline: 'none',
              }}
            >
              {activeProvider.models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
              <option value="__custom__">Custom…</option>
            </select>
            {selectedModel === '__custom__' && (
              <input
                placeholder="Enter model name"
                onChange={(e) => setSelectedModel(e.target.value)}
                style={{ ...inputStyle, marginTop: 4 }}
              />
            )}
          </div>

          {/* Key or URL */}
          {activeProvider.needsKey ? (
            <div>
              <label style={{ fontSize: 8, color: '#64748b', display: 'block', marginBottom: 3 }}>API Key</label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={activeProvider.placeholder}
                style={inputStyle}
              />
            </div>
          ) : (
            <div>
              <label style={{ fontSize: 8, color: '#64748b', display: 'block', marginBottom: 3 }}>Base URL</label>
              <input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder={activeProvider.placeholder}
                style={inputStyle}
              />
            </div>
          )}

          <button
            onClick={handleApply}
            disabled={applying}
            style={{
              marginTop: 10, width: '100%', padding: '7px',
              background: `${activeProvider.color}22`,
              border: `1px solid ${activeProvider.color}44`,
              borderRadius: 6, color: activeProvider.color,
              fontSize: 10, fontWeight: 600, cursor: 'pointer',
            }}
          >
            {applying ? 'Switching…' : `Apply — Use ${activeProvider.label}`}
          </button>

          {result && (
            <div style={{ marginTop: 6, fontSize: 9, color: result.startsWith('✓') ? '#4ade80' : '#f43f5e' }}>
              {result}
            </div>
          )}
        </div>
      </div>

    </div>
  );
}

const inputStyle: React.CSSProperties = {
  width: '100%', padding: '5px 8px',
  background: '#0a0014', border: '1px solid #1e1b4e',
  borderRadius: 5, color: '#e2e8f0', fontSize: 10,
  outline: 'none', boxSizing: 'border-box',
};

function btnStyle(color: string, _bg?: string): React.CSSProperties {
  return {
    padding: '4px 10px', borderRadius: 5,
    background: `${color}18`, border: `1px solid ${color}44`,
    color, fontSize: 8, fontWeight: 600, cursor: 'pointer',
  };
}

const sTitle: React.CSSProperties = {
  fontSize: 9, fontWeight: 700, letterSpacing: '0.1em',
  textTransform: 'uppercase', color: '#6366f1', marginBottom: 8,
};
