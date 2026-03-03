import { useState, useRef, useEffect } from 'preact/hooks';
import { JarvisAPI, JarvisMessage } from '../services/api';
import { addMessage, getState, subscribe, appendStreamToken, flushStream } from '../stores/session';

interface ChatProps {
  api: JarvisAPI;
  onCanvasUpdate: (html: string) => void;
  onApproval: (request: { id: string; tool: string; reason: string; params: string }) => void;
}

export function Chat({ api, onCanvasUpdate, onApproval }: ChatProps) {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamText, setStreamText] = useState('');
  const [statusText, setStatusText] = useState('');
  const [messages, setMessages] = useState(getState().messages);
  const messagesEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const unsub = subscribe((state) => {
      setMessages(state.messages);
      setStreamText(state.streamBuffer);
    });
    return unsub;
  }, []);

  useEffect(() => {
    const unsubscribe = api.onMessage((msg: JarvisMessage) => {
      switch (msg.type) {
        case 'message':
          if (msg.text) {
            addMessage('assistant', msg.text);
          }
          setIsLoading(false);
          setStatusText('');
          flushStream();
          break;

        case 'streaming_token':
          if (msg.text) {
            appendStreamToken(msg.text);
          }
          break;

        case 'canvas_push':
          if (msg.html) {
            onCanvasUpdate(msg.html);
          }
          break;

        case 'canvas_reset':
          onCanvasUpdate('');
          break;

        case 'approval_request':
          if (msg.approval_id && msg.tool && msg.reason) {
            onApproval({
              id: msg.approval_id,
              tool: msg.tool,
              reason: msg.reason,
              params: msg.params || '',
            });
          }
          break;

        case 'system':
          if (msg.text) {
            setStatusText(msg.text);
          }
          break;
      }
    });

    api.connect();
    return () => {
      unsubscribe();
    };
  }, [api]);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamText]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isLoading) return;
    setInput('');
    setIsLoading(true);
    setStatusText('');
    addMessage('user', text);
    api.send(text, getState().sessionId);
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div class="chat-container">
      <div class="chat-messages">
        {messages.length === 0 && !isLoading && (
          <div class="chat-empty">
            <div class="chat-empty-icon">J</div>
            <p>Hallo! Wie kann ich dir helfen?</p>
          </div>
        )}
        {messages.map((msg) => (
          <div
            key={msg.id}
            class={`chat-bubble ${msg.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'}`}
          >
            <div class="chat-bubble-text">{msg.text}</div>
            <div class="chat-bubble-time">
              {new Date(msg.timestamp).toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>
        ))}
        {streamText && (
          <div class="chat-bubble chat-bubble-assistant">
            <div class="chat-bubble-text">{streamText}</div>
          </div>
        )}
        {isLoading && !streamText && (
          <div class="chat-bubble chat-bubble-assistant chat-typing">
            <span class="dot" /><span class="dot" /><span class="dot" />
          </div>
        )}
        <div ref={messagesEnd} />
      </div>

      {statusText && (
        <div class="chat-status" role="status" aria-live="polite">
          {statusText}
        </div>
      )}

      <div class="chat-input-bar">
        <textarea
          value={input}
          onInput={(e: any) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Nachricht an Jarvis..."
          class="chat-input"
          rows={1}
          aria-label="Nachricht eingeben"
        />
        <button
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
          class="chat-send-btn"
          aria-label="Senden"
        >
          {isLoading ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10" stroke-dasharray="31.4" stroke-dashoffset="10">
                <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite" />
              </circle>
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 2L11 13M22 2l-7 20-4-9-9-4z" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}
