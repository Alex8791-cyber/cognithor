import { useState, useEffect } from 'preact/hooks';

interface SettingsProps {
  serverUrl: string;
  onSave: (serverUrl: string) => void;
}

export function Settings({ serverUrl, onSave }: SettingsProps) {
  const [url, setUrl] = useState(serverUrl);
  const [status, setStatus] = useState<'idle' | 'testing' | 'ok' | 'error'>('idle');
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    setUrl(serverUrl);
    setHasChanges(false);
  }, [serverUrl]);

  const handleUrlChange = (newUrl: string) => {
    setUrl(newUrl);
    setHasChanges(newUrl !== serverUrl);
    setStatus('idle');
  };

  const testConnection = async () => {
    setStatus('testing');
    try {
      const httpUrl = url
        .replace('ws://', 'http://')
        .replace('wss://', 'https://');
      const resp = await fetch(`${httpUrl}/api/v1/health`, {
        signal: AbortSignal.timeout(5000),
      });
      setStatus(resp.ok ? 'ok' : 'error');
    } catch {
      setStatus('error');
    }
  };

  const handleSave = () => {
    localStorage.setItem('jarvis_server_url', url);
    onSave(url);
    setHasChanges(false);
  };

  return (
    <div class="settings-container" role="region" aria-label="Einstellungen">
      <h2 class="settings-title">Einstellungen</h2>

      <div class="settings-group">
        <label class="settings-label" for="server-url">Server-URL</label>
        <p class="settings-desc">
          WebSocket-Adresse deines Jarvis-Servers.
        </p>
        <input
          id="server-url"
          type="text"
          class="settings-input"
          value={url}
          onInput={(e) => handleUrlChange((e.target as HTMLInputElement).value)}
          placeholder="ws://localhost:8741"
          aria-describedby="server-url-status"
        />

        <div class="settings-actions">
          <button
            onClick={testConnection}
            disabled={status === 'testing'}
            class="settings-btn settings-btn-secondary"
          >
            {status === 'testing' ? 'Teste...' : 'Verbindung testen'}
          </button>
          <button
            onClick={handleSave}
            disabled={!hasChanges}
            class="settings-btn settings-btn-primary"
          >
            Speichern
          </button>
        </div>

        <div id="server-url-status" class="settings-status" role="status" aria-live="polite">
          {status === 'ok' && (
            <span class="status-ok">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 6L9 17l-5-5" />
              </svg>
              Verbunden
            </span>
          )}
          {status === 'error' && (
            <span class="status-error">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M15 9l-6 6M9 9l6 6" />
              </svg>
              Nicht erreichbar
            </span>
          )}
        </div>
      </div>

      <div class="settings-group">
        <h3 class="settings-subtitle">Info</h3>
        <p class="settings-version">Jarvis PWA v0.26.6</p>
        <p class="settings-desc">
          Verbinde dich mit deinem lokalen oder Remote-Jarvis-Server.
          Der Server muss auf dem angegebenen Port laufen.
        </p>
      </div>

      <div class="settings-group">
        <h3 class="settings-subtitle">Tastaturkürzel</h3>
        <div class="settings-shortcuts">
          <div class="shortcut-row">
            <kbd>Enter</kbd>
            <span>Nachricht senden</span>
          </div>
          <div class="shortcut-row">
            <kbd>Shift + Enter</kbd>
            <span>Neue Zeile</span>
          </div>
        </div>
      </div>
    </div>
  );
}
