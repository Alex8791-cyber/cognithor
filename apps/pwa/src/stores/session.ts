/**
 * Session State Management für Jarvis PWA.
 *
 * Verwaltet den globalen Zustand der Anwendung:
 * - Verbindungsstatus
 * - Chat-Nachrichten
 * - Canvas-Zustand
 * - Approval-Requests
 */

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  text: string;
  timestamp: number;
}

export interface CanvasState {
  html: string;
  title: string;
}

export interface ApprovalRequest {
  id: string;
  tool: string;
  reason: string;
  params: string;
}

export interface SessionState {
  sessionId: string;
  connected: boolean;
  messages: ChatMessage[];
  canvas: CanvasState;
  pendingApprovals: ApprovalRequest[];
  isStreaming: boolean;
  streamBuffer: string;
}

let state: SessionState = {
  sessionId: `session_${Date.now()}`,
  connected: false,
  messages: [],
  canvas: { html: '', title: '' },
  pendingApprovals: [],
  isStreaming: false,
  streamBuffer: '',
};

type Listener = (state: SessionState) => void;
const listeners = new Set<Listener>();

function notify(): void {
  listeners.forEach((l) => l({ ...state }));
}

export function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  listener({ ...state });
  return () => listeners.delete(listener);
}

export function getState(): SessionState {
  return { ...state };
}

export function setConnected(connected: boolean): void {
  state = { ...state, connected };
  notify();
}

export function addMessage(role: ChatMessage['role'], text: string): void {
  const msg: ChatMessage = {
    id: `msg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    role,
    text,
    timestamp: Date.now(),
  };
  state = { ...state, messages: [...state.messages, msg] };
  notify();
}

export function appendStreamToken(token: string): void {
  state = {
    ...state,
    isStreaming: true,
    streamBuffer: state.streamBuffer + token,
  };
  notify();
}

export function flushStream(): void {
  if (state.streamBuffer) {
    addMessage('assistant', state.streamBuffer);
  }
  state = { ...state, isStreaming: false, streamBuffer: '' };
  notify();
}

export function setCanvas(html: string, title: string): void {
  state = { ...state, canvas: { html, title } };
  notify();
}

export function resetCanvas(): void {
  state = { ...state, canvas: { html: '', title: '' } };
  notify();
}

export function addApproval(approval: ApprovalRequest): void {
  state = {
    ...state,
    pendingApprovals: [...state.pendingApprovals, approval],
  };
  notify();
}

export function removeApproval(id: string): void {
  state = {
    ...state,
    pendingApprovals: state.pendingApprovals.filter((a) => a.id !== id),
  };
  notify();
}

export function clearMessages(): void {
  state = { ...state, messages: [] };
  notify();
}
