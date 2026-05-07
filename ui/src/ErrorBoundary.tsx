import React from 'react';

interface State { hasError: boolean; error: Error | null; info: string }

export class ErrorBoundary extends React.Component<{ children: React.ReactNode }, State> {
  state: State = { hasError: false, error: null, info: '' };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    this.setState({ info: info.componentStack ?? '' });
    console.error('[AXON ErrorBoundary]', error, info);
  }

  render() {
    if (!this.state.hasError) return this.props.children;
    return (
      <div style={{
        width: '100vw', height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexDirection: 'column', background: '#020205', color: '#ef4444', fontFamily: 'monospace', padding: 32,
      }}>
        <div style={{ fontSize: 28, marginBottom: 16 }}>💥 AXON crashed</div>
        <div style={{ fontSize: 13, color: '#f87171', marginBottom: 12, maxWidth: 800, textAlign: 'center' }}>
          {this.state.error?.message ?? 'Unknown error'}
        </div>
        <pre style={{
          fontSize: 10, color: '#6b7280', background: '#0a0014', border: '1px solid #1e1b4e',
          borderRadius: 8, padding: 16, maxWidth: 900, overflow: 'auto', maxHeight: 300,
          whiteSpace: 'pre-wrap',
        }}>
          {this.state.info}
        </pre>
        <button
          onClick={() => window.location.reload()}
          style={{
            marginTop: 20, padding: '8px 24px', background: '#6366f1', border: 'none',
            borderRadius: 6, color: '#fff', cursor: 'pointer', fontSize: 13,
          }}
        >
          🔄 Reload
        </button>
      </div>
    );
  }
}
