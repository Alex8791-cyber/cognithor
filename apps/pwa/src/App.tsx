import { useState, useMemo, useCallback } from 'preact/hooks';
import { JarvisAPI } from './services/api';
import { Chat } from './components/Chat';
import { Canvas } from './components/Canvas';
import { Settings } from './components/Settings';
import { VoiceButton } from './components/VoiceButton';
import { ApprovalDialog } from './components/ApprovalDialog';
import { addMessage, removeApproval, addApproval, getState } from './stores/session';
import './app.css';

type View = 'chat' | 'settings';

interface PendingApproval {
  id: string;
  tool: string;
  reason: string;
  params: string;
}

const DEFAULT_SERVER = localStorage.getItem('jarvis_server_url') || 'ws://localhost:8741';

export function App() {
  const [view, setView] = useState<View>('chat');
  const [serverUrl, setServerUrl] = useState(DEFAULT_SERVER);
  const [canvasHtml, setCanvasHtml] = useState('');
  const [canvasVisible, setCanvasVisible] = useState(false);
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);
  const [connected, setConnected] = useState(false);

  const api = useMemo(() => {
    const instance = new JarvisAPI(serverUrl);
    instance.onMessage((msg) => {
      if (msg.type === 'system') {
        setConnected(msg.text === 'Verbunden');
      }
    });
    return instance;
  }, [serverUrl]);

  const handleCanvasUpdate = useCallback((html: string) => {
    setCanvasHtml(html);
    setCanvasVisible(!!html);
  }, []);

  const handleApprovalRequest = useCallback((request: PendingApproval) => {
    addApproval(request);
    setPendingApproval(request);
  }, []);

  const handleApprove = useCallback((id: string) => {
    api.sendApproval(id, true);
    removeApproval(id);
    setPendingApproval(null);
  }, [api]);

  const handleReject = useCallback((id: string) => {
    api.sendApproval(id, false);
    removeApproval(id);
    setPendingApproval(null);
  }, [api]);

  const handleServerSave = useCallback((url: string) => {
    api.disconnect();
    setServerUrl(url);
  }, [api]);

  const handleVoiceTranscript = useCallback((text: string) => {
    addMessage('user', text);
    api.send(text, getState().sessionId);
  }, [api]);

  return (
    <div class="app-root">
      {/* Header */}
      <header class="app-header" role="banner">
        <div class="app-header-left">
          <h1 class="app-logo">Jarvis</h1>
          <span class={`connection-dot ${connected ? 'connected' : 'disconnected'}`}
                title={connected ? 'Verbunden' : 'Getrennt'}
                aria-label={connected ? 'Server verbunden' : 'Server getrennt'} />
        </div>
        <nav class="app-nav" role="navigation" aria-label="Hauptnavigation">
          <button
            onClick={() => setView('chat')}
            class={`nav-btn ${view === 'chat' ? 'nav-btn-active' : ''}`}
            aria-current={view === 'chat' ? 'page' : undefined}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
            </svg>
            Chat
          </button>
          <button
            onClick={() => setView('settings')}
            class={`nav-btn ${view === 'settings' ? 'nav-btn-active' : ''}`}
            aria-current={view === 'settings' ? 'page' : undefined}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.32 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
            </svg>
            Settings
          </button>
        </nav>
      </header>

      {/* Main Content */}
      <main class="app-main" role="main">
        {view === 'chat' && (
          <div class="chat-layout">
            <div class={`chat-panel ${canvasVisible ? 'with-canvas' : ''}`}>
              <Chat
                api={api}
                onCanvasUpdate={handleCanvasUpdate}
                onApproval={handleApprovalRequest}
              />
            </div>
            {canvasVisible && (
              <div class="canvas-panel">
                <Canvas html={canvasHtml} onClose={() => setCanvasVisible(false)} />
              </div>
            )}
          </div>
        )}
        {view === 'settings' && (
          <Settings serverUrl={serverUrl} onSave={handleServerSave} />
        )}
      </main>

      {/* Floating Voice Button */}
      {view === 'chat' && (
        <VoiceButton
          onTranscript={handleVoiceTranscript}
          serverUrl={serverUrl}
          disabled={!connected}
        />
      )}

      {/* Approval Dialog */}
      {pendingApproval && (
        <ApprovalDialog
          request={pendingApproval}
          onApprove={handleApprove}
          onReject={handleReject}
        />
      )}
    </div>
  );
}
