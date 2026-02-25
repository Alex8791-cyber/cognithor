import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'dev.jarvis.app',
  appName: 'Jarvis',
  webDir: 'dist',
  server: {
    // In development, connect to local Jarvis backend
    // url: 'http://localhost:8741',
    cleartext: true,
    allowNavigation: ['*'],
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
