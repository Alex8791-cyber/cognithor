import { useState, useEffect } from 'preact/hooks';
import { FunctionComponent } from 'preact';

interface SettingsProps {
  onSave: (serverUrl: string) => void;
  currentUrl: string;
}

const Settings: FunctionComponent<SettingsProps> = ({ onSave, currentUrl }) => {
  const [url, setUrl] = useState(currentUrl);
  const [status, setStatus] = useState<'idle' | 'testing' | 'ok' | 'error'>('idle');

  useEffect(() => {
    setUrl(currentUrl);
  }, [currentUrl]);

  const testConnection = async () => {
    setStatus('testing');
    try {
      const httpUrl = url.replace('ws://', 'http://').replace('wss://', 'https://');
      const resp = await fetch(`${httpUrl}/api/v1/health`, { signal: AbortSignal.timeout(5000) });
      setStatus(resp.ok ? 'ok' : 'error');
    } catch {
      setStatus('error');
    }
  };

  const handleSave = () => {
    localStorage.setItem('jarvis_server_url', url);
    onSave(url);
  };

  return (
    <div class="settings-panel">
      <h2>Einstellungen</h2>

      <div class="settings-group">
        <label>Server-URL</label>
        <input
          type="text"
          value={url}
          onInput={(e) => setUrl((e.target as HTMLInputElement).value)}
          placeholder="ws://localhost:8080"
        />
        <div class="settings-actions">
          <button onClick={testConnection} disabled={status === 'testing'}>
            {status === 'testing' ? 'Teste...' : 'Verbindung testen'}
          </button>
          <button onClick={handleSave} class="primary">
            Speichern
          </button>
        </div>
        {status === 'ok' && <span class="status-ok">Verbunden</span>}
        {status === 'error' && <span class="status-error">Nicht erreichbar</span>}
      </div>

      <div class="settings-group">
        <h3>Info</h3>
        <p>Jarvis PWA v0.22.0</p>
        <p>Verbinde dich mit deinem lokalen Jarvis-Server.</p>
      </div>
    </div>
  );
};

export default Settings;
