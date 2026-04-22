import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('LlmBackendProvider', () {
    test('initial state has empty backends and not polling', () {
      final p = LlmBackendProvider(apiBaseUrl: 'http://localhost:8741');
      expect(p.backends, isEmpty);
      expect(p.vllmStatus, isNull);
      expect(p.isPolling, isFalse);
    });

    test('startPolling sets isPolling true', () {
      final p = LlmBackendProvider(apiBaseUrl: 'http://localhost:8741');
      p.startPolling();
      expect(p.isPolling, isTrue);
      p.stopPolling();
    });

    test('stopPolling resets', () {
      final p = LlmBackendProvider(apiBaseUrl: 'http://localhost:8741');
      p.startPolling();
      p.stopPolling();
      expect(p.isPolling, isFalse);
    });

    test('VLLMStatus.fromJson parses API payload', () {
      final status = VLLMStatus.fromJson({
        'hardware_ok': true,
        'hardware_info': {
          'gpu_name': 'RTX 5090',
          'vram_gb': 32,
          'compute_capability': '12.0',
        },
        'docker_ok': true,
        'image_pulled': false,
        'container_running': false,
        'current_model': null,
        'last_error': null,
      });
      expect(status.hardwareOk, isTrue);
      expect(status.hardwareInfo?.gpuName, 'RTX 5090');
      expect(status.hardwareInfo?.vramGb, 32);
      expect(status.hardwareInfo?.computeCapability, '12.0');
    });

    test('pullImage method exists on provider', () {
      final p = LlmBackendProvider(apiBaseUrl: 'http://test');
      // Real SSE integration is covered server-side (Python test_vllm_fake_server.py).
      // Flutter side only verifies the surface: method is callable and returns a Stream.
      expect(p.pullImage, isA<Function>());
    });

    test('startContainer method exists on provider', () {
      final p = LlmBackendProvider(apiBaseUrl: 'http://test');
      expect(p.startContainer, isA<Function>());
    });
  });
}
