/**
 * WebSocket + REST Client für Jarvis PWA.
 *
 * Stellt die Verbindung zum Jarvis-Server her und verwaltet
 * Echtzeit-Kommunikation via WebSocket sowie REST-Aufrufe.
 */

export interface JarvisMessage {
  type: 'message' | 'streaming_token' | 'approval_request' | 'canvas_push' | 'canvas_reset' | 'canvas_eval' | 'system';
  text?: string;
  session_id?: string;
  html?: string;
  title?: string;
  js?: string;
  approval_id?: string;
  tool?: string;
  reason?: string;
  params?: string;
}

export type MessageHandler = (msg: JarvisMessage) => void;

export class JarvisAPI {
  private ws: WebSocket | null = null;
  private handlers: Set<MessageHandler> = new Set();
  private reconnectTimer: number | null = null;
  private _reconnectAttempts = 0;
  private _connected = false;
  private _token: string | null = null;

  constructor(private serverUrl: string) {}

  get connected(): boolean {
    return this._connected;
  }

  get httpUrl(): string {
    return this.serverUrl
      .replace('ws://', 'http://')
      .replace('wss://', 'https://');
  }

  private async fetchToken(): Promise<string | null> {
    if (this._token) return this._token;
    try {
      const resp = await fetch(`${this.httpUrl}/api/v1/bootstrap`);
      if (resp.ok) {
        const data = await resp.json();
        this._token = data?.token || null;
      }
    } catch { /* bootstrap not available */ }
    return this._token;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    try {
      this.ws = new WebSocket(`${this.serverUrl}/ws`);

      this.ws.onopen = async () => {
        this._connected = true;
        this._reconnectAttempts = 0;
        // Send auth token as first message
        const token = await this.fetchToken();
        if (token && this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: 'auth', token }));
        }
        this.notify({ type: 'system', text: 'Verbunden' });
      };

      this.ws.onmessage = (event) => {
        try {
          const msg: JarvisMessage = JSON.parse(event.data);
          this.notify(msg);
        } catch {
          this.notify({ type: 'message', text: event.data });
        }
      };

      this.ws.onclose = () => {
        this._connected = false;
        this.notify({ type: 'system', text: 'Verbindung getrennt' });
        this.scheduleReconnect();
      };

      this.ws.onerror = () => {
        this._connected = false;
      };
    } catch (err) {
      console.error('WebSocket connection failed:', err);
      this.scheduleReconnect();
    }
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
    this._connected = false;
  }

  send(text: string, sessionId: string = 'default'): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected');
      return;
    }
    this.ws.send(JSON.stringify({
      type: 'message',
      text,
      session_id: sessionId,
    }));
  }

  sendApproval(approvalId: string, approved: boolean): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({
      type: 'approval_response',
      approval_id: approvalId,
      approved,
    }));
  }

  onMessage(handler: MessageHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  async healthCheck(): Promise<boolean> {
    try {
      const resp = await fetch(`${this.httpUrl}/api/v1/health`, {
        signal: AbortSignal.timeout(5000),
      });
      return resp.ok;
    } catch {
      return false;
    }
  }

  private notify(msg: JarvisMessage): void {
    this.handlers.forEach((h) => h(msg));
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    const delay = Math.min(1000 * Math.pow(2, this._reconnectAttempts), 30000);
    this._reconnectAttempts++;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }
}
