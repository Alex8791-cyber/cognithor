import { useCallback, useEffect, useRef, useState } from "react";
import { getToken } from "../utils/api";
import { useJarvisChat } from "../hooks/useJarvisChat";
import { useVoiceMode } from "../hooks/useVoiceMode";
import { t } from "../utils/i18n";
import { MessageList } from "../components/chat/MessageList";
import { ChatInput } from "../components/chat/ChatInput";
import { ChatCanvas } from "../components/chat/ChatCanvas";
import { ToolIndicator } from "../components/chat/ToolIndicator";
import { ApprovalBanner } from "../components/chat/ApprovalBanner";
import { VoiceIndicator } from "../components/chat/VoiceIndicator";
import PipelineCanvas from "../components/chat/PipelineCanvas";

export default function ChatPage() {
  const {
    messages,
    streamText,
    isStreaming,
    isConnected,
    canvasHtml,
    canvasTitle,
    activeTool,
    pendingApproval,
    pipelineState,
    sendMessage,
    sendFile,
    sendVoice,
    respondApproval,
    clearMessages,
  } = useJarvisChat();

  // Load wake word from config API
  const [wakeWord, setWakeWord] = useState("jarvis");
  useEffect(() => {
    (async () => {
      const token = await getToken();
      const headers = token ? { Authorization: `Bearer ${token}` } : {};
      fetch("/api/v1/config", { headers })
        .then(r => r.ok ? r.json() : null)
        .then(cfg => {
          const ww = cfg?.channels?.voice_config?.wake_word;
          if (ww) setWakeWord(ww);
        })
        .catch(() => {});
    })();
  }, []);

  // Track previous message count to detect new assistant messages
  const prevMsgCountRef = useRef(messages.length);

  const handleVoiceCommand = useCallback((command) => {
    sendMessage(command);
  }, [sendMessage]);

  const voice = useVoiceMode({ onCommand: handleVoiceCommand, wakeWord });

  // Auto-activate voice mode after page load.
  // No ref guard — StrictMode does mount/cleanup/re-mount, so the second
  // mount creates a fresh timer that actually fires.  voice.activate()
  // is idempotent (checks activeRef internally).
  useEffect(() => {
    if (!voice.isSupported) return;
    const timer = setTimeout(() => voice.activate(), 1500);
    return () => clearTimeout(timer);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // When a new assistant message arrives and voice mode is active, speak it
  useEffect(() => {
    if (messages.length > prevMsgCountRef.current) {
      const lastMsg = messages[messages.length - 1];
      if (lastMsg && lastMsg.role === "assistant" && voice.isActive) {
        voice.speakResponse(lastMsg.text);
      }
    }
    prevMsgCountRef.current = messages.length;
  }, [messages, voice.isActive, voice.speakResponse]);

  const showCanvas = !!canvasHtml;

  return (
    <div className="cc-chat-layout">
      {/* Chat Panel */}
      <div className="cc-chat-panel">
        {/* Chat Header */}
        <div className="cc-chat-header">
          <div className="cc-chat-header-left">
            <span className={`cc-chat-status-dot ${isConnected ? "cc-connected" : ""}`} />
            <span className="cc-chat-title">{t("chat.title")}</span>
          </div>
          <div className="cc-chat-header-right">
            {/* Voice Mode Toggle */}
            {voice.isSupported && (
              <button
                className={`cc-chat-header-btn ${voice.isActive ? "cc-voice-active" : ""}`}
                onClick={voice.toggle}
                type="button"
                title={voice.isActive ? "Disable voice mode" : `Enable voice mode (wake word: "${wakeWord}")`}
              >
                <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z"/>
                  <path d="M19 10v2a7 7 0 01-14 0v-2M12 19v4M8 23h8"/>
                </svg>
                {voice.isActive ? t("chat.voice_on") : t("chat.voice")}
              </button>
            )}
            {showCanvas && (
              <button
                className="cc-chat-header-btn"
                onClick={() => {}}
                type="button"
                title="Canvas is displayed on the right"
              >
                {t("chat.canvas_active")}
              </button>
            )}
            <button
              className="cc-chat-header-btn"
              onClick={clearMessages}
              type="button"
              title="Clear chat"
            >
              <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
              </svg>
              {t("chat.clear")}
            </button>
          </div>
        </div>

        {/* Voice Indicator */}
        {voice.isActive && (
          <VoiceIndicator
            voiceState={voice.voiceState}
            lastHeard={voice.lastHeard}
            VoiceState={voice.VoiceState}
          />
        )}

        {/* Message List */}
        <MessageList
          messages={messages}
          streamText={streamText}
          isStreaming={isStreaming}
        />

        {/* Tool Indicator */}
        <ToolIndicator tool={activeTool} />

        {/* Pipeline Canvas — live PGE visualization */}
        {pipelineState && (
          <PipelineCanvas
            pipeline={pipelineState}
            collapsed={!pipelineState.active}
            onToggle={() => {}}
          />
        )}

        {/* Approval Banner */}
        <ApprovalBanner
          approval={pendingApproval}
          onRespond={respondApproval}
        />

        {/* Input */}
        <ChatInput
          onSend={sendMessage}
          onFile={sendFile}
          onVoice={sendVoice}
          disabled={isStreaming}
        />
      </div>

      {/* Canvas Panel (conditional) */}
      {showCanvas && (
        <ChatCanvas
          html={canvasHtml}
          title={canvasTitle}
          onClose={() => {}}
        />
      )}
    </div>
  );
}
