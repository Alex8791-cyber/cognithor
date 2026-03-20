import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'dev.cognithor.app',
  appName: 'Cognithor',
  webDir: '../../flutter_app/build/web',  // Point to Flutter build
  server: {
    url: 'http://localhost:8741',
    cleartext: true,
    allowNavigation: ['localhost', '127.0.0.1', '*.local'],
  },
  plugins: {
    PushNotifications: {
      presentationOptions: ['badge', 'sound', 'alert'],
    },
    Camera: {
      permissions: ['camera', 'photos'],
    },
  },
};

export default config;
