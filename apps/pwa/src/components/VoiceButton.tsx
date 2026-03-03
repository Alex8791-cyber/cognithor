import { useState, useCallback } from 'preact/hooks';

interface VoiceButtonProps {
  onTranscript: (text: string) => void;
  serverUrl: string;
  disabled?: boolean;
}

export function VoiceButton({ onTranscript, serverUrl, disabled = false }: VoiceButtonProps) {
  const [recording, setRecording] = useState(false);
  const [talkMode, setTalkMode] = useState(false);

  const httpUrl = serverUrl
    .replace('ws://', 'http://')
    .replace('wss://', 'https://');

  const toggleRecording = useCallback(async () => {
    if (recording) {
      setRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm',
      });
      const chunks: Blob[] = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks, { type: 'audio/webm' });

        try {
          const resp = await fetch(`${httpUrl}/api/v1/voice/transcribe`, {
            method: 'POST',
            body: blob,
            headers: { 'Content-Type': 'audio/webm' },
          });
          if (resp.ok) {
            const data = await resp.json();
            if (data.text) onTranscript(data.text);
          }
        } catch (err) {
          console.error('Transcription failed:', err);
        }
      };

      setRecording(true);
      mediaRecorder.start();

      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
          setRecording(false);
        }
      }, 15000);
    } catch (err) {
      console.error('Microphone access failed:', err);
    }
  }, [recording, httpUrl, onTranscript]);

  const toggleTalkMode = useCallback(() => {
    setTalkMode((prev) => !prev);
  }, []);

  return (
    <div class="voice-controls" role="toolbar" aria-label="Sprachsteuerung">
      <button
        class={`voice-btn ${recording ? 'recording' : ''}`}
        onClick={toggleRecording}
        disabled={disabled}
        title={recording ? 'Aufnahme stoppen' : 'Sprachaufnahme'}
        aria-label={recording ? 'Aufnahme stoppen' : 'Sprachaufnahme starten'}
        aria-pressed={recording}
      >
        {recording ? (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        ) : (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
            <path d="M19 10v2a7 7 0 01-14 0v-2M12 19v4M8 23h8" />
          </svg>
        )}
      </button>
      <button
        class={`talk-mode-btn ${talkMode ? 'active' : ''}`}
        onClick={toggleTalkMode}
        disabled={disabled}
        title={talkMode ? 'Talk Mode deaktivieren' : 'Talk Mode aktivieren'}
        aria-label={talkMode ? 'Talk Mode deaktivieren' : 'Talk Mode aktivieren'}
        aria-pressed={talkMode}
      >
        {talkMode ? (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
            <path d="M19.07 4.93a10 10 0 010 14.14M15.54 8.46a5 5 0 010 7.07" />
          </svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
            <line x1="23" y1="9" x2="17" y2="15" />
            <line x1="17" y1="9" x2="23" y2="15" />
          </svg>
        )}
      </button>
    </div>
  );
}
