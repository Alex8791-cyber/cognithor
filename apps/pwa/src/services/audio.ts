/**
 * Audio/Mikrofon-Service für Jarvis PWA.
 *
 * Verwaltet Mikrofon-Zugriff, Audio-Aufnahme und Streaming
 * zum Jarvis-Server für Speech-to-Text.
 */

export class AudioService {
  private stream: MediaStream | null = null;
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];

  async requestMicrophoneAccess(): Promise<boolean> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      return true;
    } catch {
      return false;
    }
  }

  startRecording(onChunk?: (data: Blob) => void): void {
    if (!this.stream) return;

    this.chunks = [];
    this.recorder = new MediaRecorder(this.stream, {
      mimeType: 'audio/webm;codecs=opus',
    });

    this.recorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        this.chunks.push(e.data);
        onChunk?.(e.data);
      }
    };

    this.recorder.start(250); // Chunk every 250ms for streaming
  }

  stopRecording(): Blob | null {
    if (!this.recorder || this.recorder.state !== 'recording') return null;

    return new Promise<Blob | null>((resolve) => {
      this.recorder!.onstop = () => {
        const blob = new Blob(this.chunks, { type: 'audio/webm' });
        this.chunks = [];
        resolve(blob);
      };
      this.recorder!.stop();
    }) as unknown as Blob | null;
  }

  async stopRecordingAsync(): Promise<Blob | null> {
    if (!this.recorder || this.recorder.state !== 'recording') return null;

    return new Promise<Blob>((resolve) => {
      this.recorder!.onstop = () => {
        const blob = new Blob(this.chunks, { type: 'audio/webm' });
        this.chunks = [];
        resolve(blob);
      };
      this.recorder!.stop();
    });
  }

  release(): void {
    if (this.recorder?.state === 'recording') {
      this.recorder.stop();
    }
    this.recorder = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }

  get isRecording(): boolean {
    return this.recorder?.state === 'recording';
  }

  get hasAccess(): boolean {
    return this.stream !== null;
  }
}
