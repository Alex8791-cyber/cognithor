/**
 * Kamera-Service für Jarvis PWA.
 *
 * Nutzt Capacitor Camera Plugin für native Kamera-Zugriff
 * und Web API als Fallback. Ermöglicht das Senden von
 * Fotos/Screenshots an Jarvis zur Analyse.
 */

export interface CaptureResult {
  dataUrl: string;
  blob: Blob;
}

export class CameraService {
  /**
   * Nimmt ein Foto auf (Capacitor oder Web Fallback).
   */
  async capture(): Promise<CaptureResult | null> {
    // Try Capacitor first
    const capacitorResult = await this.captureCapacitor();
    if (capacitorResult) return capacitorResult;

    // Web API fallback
    return this.captureWeb();
  }

  private async captureCapacitor(): Promise<CaptureResult | null> {
    try {
      const { Camera, CameraResultType } = await import('@capacitor/camera');

      const photo = await Camera.getPhoto({
        quality: 80,
        allowEditing: false,
        resultType: CameraResultType.DataUrl,
      });

      if (!photo.dataUrl) return null;

      const resp = await fetch(photo.dataUrl);
      const blob = await resp.blob();

      return { dataUrl: photo.dataUrl, blob };
    } catch {
      return null;
    }
  }

  private async captureWeb(): Promise<CaptureResult | null> {
    return new Promise((resolve) => {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.capture = 'environment';

      input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) {
          resolve(null);
          return;
        }

        const reader = new FileReader();
        reader.onload = () => {
          resolve({
            dataUrl: reader.result as string,
            blob: file,
          });
        };
        reader.readAsDataURL(file);
      };

      input.click();
    });
  }

  /**
   * Sendet ein Bild an den Jarvis-Server zur Analyse.
   */
  async sendToJarvis(
    blob: Blob,
    serverUrl: string,
    prompt: string = 'Was siehst du auf diesem Bild?',
  ): Promise<string> {
    const httpUrl = serverUrl
      .replace('ws://', 'http://')
      .replace('wss://', 'https://');

    const formData = new FormData();
    formData.append('image', blob, 'capture.jpg');
    formData.append('prompt', prompt);

    const resp = await fetch(`${httpUrl}/api/v1/vision/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!resp.ok) throw new Error(`Vision API error: ${resp.status}`);

    const data = await resp.json();
    return data.response || data.text || '';
  }
}
