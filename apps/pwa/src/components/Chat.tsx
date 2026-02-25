import { useState, useRef, useEffect } from 'preact/hooks';
import { connectWebSocket, sendMessage } from '../services/api';
import { sessionStore } from '../stores/session';

interface Message {
  role: 'user' | 'assistant';
  text: string;
  timestamp: string;
}

interface ChatProps {
  onCanvasUpdate: (html: string) => void;
}

export function Chat({ onCanvasUpdate }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamText, setStreamText] = useState('');
  const messagesEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ws = connectWebSocket(sessionStore.sessionId, {
      onMessage: (text: string) => {
        setMessages(prev => [...prev, { role: 'assistant', text, timestamp: new Date().toISOString() }]);
        setIsLoading(false);
        setStreamText('');
      },
      onStreamToken: (token: string) => {
        setStreamText(prev => prev + token);
      },
      onCanvasPush: (html: string) => {
        onCanvasUpdate(html);
      },
      onCanvasReset: () => {
        onCanvasUpdate('');
      },
    });
    return () => ws?.close();
  }, []);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamText]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isLoading) return;
    setInput('');
    setIsLoading(true);
    setMessages(prev => [...prev, { role: 'user', text, timestamp: new Date().toISOString() }]);
    sendMessage(sessionStore.sessionId, text);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1, overflow: 'auto', padding: '16px' }}>
        {messages.map((msg, i) => (
          <div key={i} style={{
            marginBottom: '12px', padding: '10px 14px', borderRadius: '12px',
            maxWidth: '85%', wordBreak: 'break-word',
            marginLeft: msg.role === 'user' ? 'auto' : '0',
            background: msg.role === 'user' ? '#00d4ff22' : '#1a1a2e',
            border: `1px solid ${msg.role === 'user' ? '#00d4ff44' : '#333'}`,
          }}>
            <div style={{ fontSize: '14px', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>{msg.text}</div>
          </div>
        ))}
        {streamText && (
          <div style={{ marginBottom: '12px', padding: '10px 14px', borderRadius: '12px',
                        background: '#1a1a2e', border: '1px solid #333' }}>
            <div style={{ fontSize: '14px', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>{streamText}</div>
          </div>
        )}
        <div ref={messagesEnd} />
      </div>
      <div style={{ padding: '12px 16px', borderTop: '1px solid #333', display: 'flex', gap: '8px' }}>
        <input
          value={input}
          onInput={(e: any) => setInput(e.target.value)}
          onKeyDown={(e: any) => e.key === 'Enter' && handleSend()}
          placeholder="Nachricht an Jarvis..."
          style={{
            flex: 1, padding: '10px 14px', background: '#1a1a2e', color: '#e0e0e0',
            border: '1px solid #333', borderRadius: '8px', fontSize: '14px', outline: 'none',
          }}
        />
        <button onClick={handleSend} disabled={isLoading}
          style={{
            padding: '10px 20px', background: isLoading ? '#555' : '#00d4ff',
            color: isLoading ? '#999' : '#000', border: 'none', borderRadius: '8px',
            cursor: isLoading ? 'default' : 'pointer', fontWeight: 600,
          }}>
          {isLoading ? '...' : 'Senden'}
        </button>
      </div>
    </div>
  );
}
