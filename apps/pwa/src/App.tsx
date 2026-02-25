import { useState } from 'preact/hooks';
import { Chat } from './components/Chat';
import { Canvas } from './components/Canvas';
import { Settings } from './components/Settings';
import { VoiceButton } from './components/VoiceButton';

type View = 'chat' | 'settings';

export function App() {
  const [view, setView] = useState<View>('chat');
  const [canvasHtml, setCanvasHtml] = useState('');
  const [canvasVisible, setCanvasVisible] = useState(false);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header */}
      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '12px 16px', background: '#1a1a2e', borderBottom: '1px solid #333',
      }}>
        <h1 style={{ fontSize: '18px', fontWeight: 600, color: '#00d4ff' }}>Jarvis</h1>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={() => setView('chat')}
            style={{ padding: '6px 12px', background: view === 'chat' ? '#00d4ff' : '#333',
                     color: view === 'chat' ? '#000' : '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
            Chat
          </button>
          <button onClick={() => setView('settings')}
            style={{ padding: '6px 12px', background: view === 'settings' ? '#00d4ff' : '#333',
                     color: view === 'settings' ? '#000' : '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
            Settings
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {view === 'chat' && (
          <>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <Chat onCanvasUpdate={(html: string) => { setCanvasHtml(html); setCanvasVisible(!!html); }} />
            </div>
            {canvasVisible && (
              <div style={{ width: '50%', borderLeft: '1px solid #333', minWidth: '300px' }}>
                <Canvas html={canvasHtml} onClose={() => setCanvasVisible(false)} />
              </div>
            )}
          </>
        )}
        {view === 'settings' && <Settings />}
      </main>

      {/* Voice Button (floating) */}
      <VoiceButton />
    </div>
  );
}
