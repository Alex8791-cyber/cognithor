/**
 * Push Notifications Service für Jarvis PWA.
 *
 * Nutzt Capacitor Push Notifications Plugin für native Push
 * und Web Push API als Fallback im Browser.
 */

export interface PushConfig {
  serverUrl: string;
  sessionId: string;
}

export class PushService {
  private config: PushConfig;
  private registered = false;

  constructor(config: PushConfig) {
    this.config = config;
  }

  async register(): Promise<boolean> {
    // Try Capacitor first
    if (await this.registerCapacitor()) return true;

    // Fallback to Web Push
    return this.registerWebPush();
  }

  private async registerCapacitor(): Promise<boolean> {
    try {
      const { PushNotifications } = await import('@capacitor/push-notifications');

      const permission = await PushNotifications.requestPermissions();
      if (permission.receive !== 'granted') return false;

      await PushNotifications.register();

      PushNotifications.addListener('registration', async (token) => {
        await this.sendTokenToServer(token.value, 'fcm');
        this.registered = true;
      });

      PushNotifications.addListener('pushNotificationReceived', (notification) => {
        console.log('Push received:', notification);
      });

      return true;
    } catch {
      return false;
    }
  }

  private async registerWebPush(): Promise<boolean> {
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
      return false;
    }

    try {
      const registration = await navigator.serviceWorker.ready;
      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: await this.getVAPIDKey(),
      });

      await this.sendSubscriptionToServer(subscription);
      this.registered = true;
      return true;
    } catch {
      return false;
    }
  }

  private async sendTokenToServer(token: string, type: string): Promise<void> {
    const httpUrl = this.config.serverUrl
      .replace('ws://', 'http://')
      .replace('wss://', 'https://');

    await fetch(`${httpUrl}/api/v1/push/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        token,
        type,
        session_id: this.config.sessionId,
      }),
    });
  }

  private async sendSubscriptionToServer(subscription: PushSubscription): Promise<void> {
    const httpUrl = this.config.serverUrl
      .replace('ws://', 'http://')
      .replace('wss://', 'https://');

    await fetch(`${httpUrl}/api/v1/push/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        subscription: subscription.toJSON(),
        type: 'web',
        session_id: this.config.sessionId,
      }),
    });
  }

  private async getVAPIDKey(): Promise<Uint8Array> {
    const httpUrl = this.config.serverUrl
      .replace('ws://', 'http://')
      .replace('wss://', 'https://');

    const resp = await fetch(`${httpUrl}/api/v1/push/vapid-key`);
    const data = await resp.json();
    return Uint8Array.from(atob(data.key), (c) => c.charCodeAt(0));
  }

  get isRegistered(): boolean {
    return this.registered;
  }
}
