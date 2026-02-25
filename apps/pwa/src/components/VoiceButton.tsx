import { useState, useCallback } from 'preact/hooks';
import { FunctionComponent } from 'preact';

interface VoiceButtonProps {
  onTranscript: (text: string) => void;
  wsUrl: string;
  disabled?: boolean;
}

const VoiceButton: FunctionComponent<VoiceButtonProps> = ({
  onTranscript,
  wsUrl,
  disabled = false,
}) => {
  const [recording, setRecording] = useState(false);
  const [talkMode, setTalkMode] = useState(false);

  const toggleRecording = useCallback(async () => {
    if (recording) {
      setRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });
      const chunks: Blob[] = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks, { type: 'audio/webm' });

        // Send to backend for STT
        try {
          const resp = await fetch(`${wsUrl.replace('ws', 'http')}/api/v1/voice/transcribe`, {
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

      // Auto-stop after 15s
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
          setRecording(false);
        }
      }, 15000);
    } catch (err) {
      console.error('Microphone access failed:', err);
    }
  }, [recording, wsUrl, onTranscript]);

  const toggleTalkMode = useCallback(() => {
    setTalkMode(!talkMode);
    // TODO: Send talk_mode toggle to backend via WebSocket
  }, [talkMode]);

  return (
    <div class="voice-controls">
      <button
        class={`voice-btn ${recording ? 'recording' : ''}`}
        onClick={toggleRecording}
        disabled={disabled}
        title={recording ? 'Stop Recording' : 'Push to Talk'}
      >
        {recording ? '⏹' : '🎤'}
      </button>
      <button
        class={`talk-mode-btn ${talkMode ? 'active' : ''}`}
        onClick={toggleTalkMode}
        disabled={disabled}
        title={talkMode ? 'Talk Mode aktiv' : 'Talk Mode aktivieren'}
      >
        {talkMode ? '🔊' : '🔇'}
      </button>
    </div>
  );
};

export default VoiceButton;
