import { useRef, useEffect, useState } from 'react';
import { useAxonStore } from '../store/axonStore';
import { useAxonSocket } from '../hooks/useAxonSocket';

export default function ChatPanel() {
  const messages  = useAxonStore((s) => s.messages);
  const thinking  = useAxonStore((s) => s.thinking);
  const connected = useAxonStore((s) => s.connected);
  const { send }  = useAxonSocket();
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, thinking]);

  const handleSend = () => {
    const txt = input.trim();
    if (!txt || !connected) return;
    send(txt);
    setInput('');
  };

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: '#030712', borderTop: '1px solid #1e1b4e',
    }}>
      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}>
        {messages.length === 0 && (
          <div style={{ margin: 'auto', color: '#374151', fontSize: 11, textAlign: 'center' }}>
            <div style={{ fontSize: 28, marginBottom: 8 }}>🧠</div>
            AXON is {connected ? 'online and waiting' : 'offline — start the server'}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} style={{
            alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '85%',
          }}>
            {m.role === 'axon' && (
              <div style={{ fontSize: 8, color: '#6366f1', marginBottom: 2, fontFamily: 'monospace' }}>AXON</div>
            )}
            <div style={{
              padding: '6px 10px',
              borderRadius: m.role === 'user' ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
              background: m.role === 'user' ? '#1e1b4e' : '#0f172a',
              border: `1px solid ${m.role === 'user' ? '#4338ca' : '#1e293b'}`,
              fontSize: 12, color: '#e2e8f0', lineHeight: 1.6,
              wordBreak: 'break-word',
            }}>
              {m.text}
            </div>
            <div style={{ fontSize: 7, color: '#374151', marginTop: 2, textAlign: m.role === 'user' ? 'right' : 'left', fontFamily: 'monospace' }}>
              {new Date(m.ts).toLocaleTimeString()}
            </div>
          </div>
        ))}

        {thinking && (
          <div style={{ alignSelf: 'flex-start', maxWidth: '80%' }}>
            <div style={{ fontSize: 8, color: '#6366f1', marginBottom: 2, fontFamily: 'monospace' }}>AXON</div>
            <div style={{
              padding: '8px 12px', borderRadius: '12px 12px 12px 2px',
              background: '#0f172a', border: '1px solid #1e293b',
            }}>
              <span style={{ display: 'inline-flex', gap: 4 }}>
                {[0, 1, 2].map((i) => (
                  <span key={i} style={{
                    width: 6, height: 6, borderRadius: '50%',
                    background: '#6366f1',
                    animation: `pulse 1.2s ${i * 0.2}s infinite`,
                  }} />
                ))}
              </span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{
        padding: '8px 10px', borderTop: '1px solid #1e1b4e',
        display: 'flex', gap: 6, background: '#020205',
      }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
          placeholder={connected ? 'Talk to AXON…' : 'Server offline'}
          disabled={!connected}
          style={{
            flex: 1, padding: '7px 10px',
            background: '#0d1117', border: '1px solid #1e293b',
            borderRadius: 8, color: '#e2e8f0', fontSize: 12, outline: 'none',
          }}
        />
        <button
          onClick={handleSend}
          disabled={!connected || !input.trim()}
          style={{
            padding: '7px 14px', borderRadius: 8,
            background: connected ? '#4f46e5' : '#1e1b4e',
            border: 'none', color: '#fff', cursor: connected ? 'pointer' : 'not-allowed',
            fontSize: 12, fontWeight: 600, transition: 'background 0.2s',
          }}
        >
          Send
        </button>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
          40% { opacity: 1; transform: scale(1.1); }
        }
      `}</style>
    </div>
  );
}
