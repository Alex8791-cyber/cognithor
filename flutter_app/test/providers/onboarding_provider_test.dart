// Tests for OnboardingProvider — exercises stage transitions and JSON parsing
// against a mocked http.Client.

import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:cognithor_ui/providers/onboarding_provider.dart';

http.Response _ok(Object body) => http.Response(
  jsonEncode(body),
  200,
  headers: {'content-type': 'application/json'},
);

http.Response _bad(int code) => http.Response(
  '{"detail":"err"}',
  code,
  headers: {'content-type': 'application/json'},
);

void main() {
  group('OnboardingProvider', () {
    test('initial state is idle', () {
      final p = OnboardingProvider(
        apiBaseUrl: 'http://x',
        httpClient: MockClient((req) async => _ok({})),
      );
      expect(p.stage, WizardStage.idle);
      expect(p.solutions, isEmpty);
      expect(p.appliedTierId, isNull);
    });

    test('detect happy path → stage=detected', () async {
      final mock = MockClient((req) async {
        if (req.url.path == '/api/system/profile') {
          return _ok({
            'detected_at': '2026-05-07T20:00:00Z',
            'tier': 'enterprise',
            'recommended_mode': 'offline',
            'sanity_warnings': [],
            'components': {
              'gpu': {'value': 'RTX 5090', 'status': 'ok'},
            },
          });
        }
        if (req.url.path == '/api/system/capabilities') {
          return _ok({
            'can_run_nvfp4': true,
            'can_run_fp8_marlin': true,
            'can_run_gguf_cuda': true,
            'can_run_gguf_metal': false,
            'can_run_vllm_container': true,
            'can_run_ollama_native': true,
            'vram_class': 'xlarge',
            'ram_class': 'extreme',
            'disk_class': 'medium',
            'has_internet': true,
            'profile_hash': 'sha256:abc',
            'multi_gpu_count': 1,
          });
        }
        return _bad(404);
      });
      final p = OnboardingProvider(apiBaseUrl: 'http://x', httpClient: mock);
      await p.detect();
      expect(p.stage, WizardStage.detected);
      expect(p.profile?.tier, 'enterprise');
      expect(p.capabilities?.canRunNvfp4, true);
      expect(p.capabilities?.vramClass, 'xlarge');
    });

    test('detect failure → stage=failed', () async {
      final mock = MockClient((req) async => _bad(500));
      final p = OnboardingProvider(apiBaseUrl: 'http://x', httpClient: mock);
      await p.detect();
      expect(p.stage, WizardStage.failed);
      expect(p.errorMessage, isNotNull);
    });

    test('setObjective fetches and parses solutions', () async {
      final mock = MockClient((req) async {
        if (req.url.path == '/api/system/recommendations') {
          return _ok({
            'manifest_version': '2026.05.07.01',
            'manifest_origin': 'embedded',
            'manifest_signature_verified': false,
            'objective_preset': 'balanced',
            'capabilities_hash': 'sha256:abc',
            'solutions': [
              {
                'tier_id': 'power-vllm-fp8-ada',
                'display_name': 'Power vLLM FP8',
                'rationale_de': 'RTX 4090 mit FP8',
                'rationale_en': '...',
                'score': 0.708,
                'score_breakdown': {
                  'quality': 0.83,
                  'speed': 0.25,
                  'cost': 1.0,
                  'privacy': 1.0,
                },
                'blockers': [],
                'warnings': [],
                'is_immediately_runnable': true,
                'estimated_first_response_s': 0.5,
                'estimated_disk_gb': 50,
                'estimated_setup_minutes': 15,
                'estimated_cost_eur_per_month': 0,
                'backend': 'vllm',
                'model_set': {
                  'planner': 'qwen3.6-27b-fp8',
                  'executor': 'qwen3.5-9b',
                  'coder': 'qwen3.6-27b-fp8',
                  'embedding': 'qwen3-embedding-0.6b-ollama',
                  'formulate': 'qwen3.5-9b',
                  'fast_path_validator': 'qwen3.5-9b',
                },
                'rule_id': 'solver.match.exact',
              },
            ],
          });
        }
        return _bad(404);
      });
      final p = OnboardingProvider(apiBaseUrl: 'http://x', httpClient: mock);
      await p.setObjective('balanced');
      expect(p.stage, WizardStage.awaitingChoice);
      expect(p.solutions, hasLength(1));
      expect(p.solutions.first.tierId, 'power-vllm-fp8-ada');
      expect(p.solutions.first.isImmediatelyRunnable, true);
      expect(p.solutions.first.scoreBreakdown['quality'], closeTo(0.83, 1e-6));
      expect(p.solutions.first.modelSet['planner'], 'qwen3.6-27b-fp8');
    });

    test('apply happy path → stage=applied + tierId set', () async {
      final mock = MockClient((req) async {
        if (req.method == 'POST' && req.url.path == '/api/system/apply') {
          final body = jsonDecode(req.body) as Map<String, dynamic>;
          expect(body['user_confirmed'], true);
          expect(body['tier_id'], 'power-vllm-fp8-ada');
          return _ok({
            'success': true,
            'selected_tier_id': 'power-vllm-fp8-ada',
            'config_path': '/home/u/.cognithor/config.yaml',
            'backup_path': null,
            'sidecar_path': '/home/u/.cognithor/.hardware_aware.json',
            'initialized_marker_path':
                '/home/u/.cognithor/.cognithor_initialized',
            'capabilities_hash': 'sha256:abc',
          });
        }
        return _bad(404);
      });
      final p = OnboardingProvider(apiBaseUrl: 'http://x', httpClient: mock);
      await p.apply('power-vllm-fp8-ada');
      expect(p.stage, WizardStage.applied);
      expect(p.appliedTierId, 'power-vllm-fp8-ada');
    });

    test('reset returns to idle', () async {
      final mock = MockClient(
        (req) async => _ok({
          'success': true,
          'selected_tier_id': 'x',
          'config_path': '/c',
          'backup_path': null,
          'sidecar_path': '/s',
          'initialized_marker_path': '/m',
          'capabilities_hash': 'h',
        }),
      );
      final p = OnboardingProvider(apiBaseUrl: 'http://x', httpClient: mock);
      await p.apply('x');
      expect(p.stage, WizardStage.applied);
      p.reset();
      expect(p.stage, WizardStage.idle);
      expect(p.appliedTierId, isNull);
      expect(p.solutions, isEmpty);
    });

    test('auth token is sent in header', () async {
      String? capturedAuth;
      final mock = MockClient((req) async {
        capturedAuth = req.headers['Authorization'];
        if (req.url.path == '/api/system/profile') {
          return _ok({
            'detected_at': 't',
            'tier': 'x',
            'recommended_mode': 'm',
            'sanity_warnings': [],
            'components': {},
          });
        }
        if (req.url.path == '/api/system/capabilities') {
          return _ok({
            'can_run_nvfp4': false,
            'can_run_fp8_marlin': false,
            'can_run_gguf_cuda': false,
            'can_run_gguf_metal': false,
            'can_run_vllm_container': false,
            'can_run_ollama_native': true,
            'vram_class': 'none',
            'ram_class': 'low',
            'disk_class': 'low',
            'has_internet': true,
            'profile_hash': 'h',
            'multi_gpu_count': 1,
          });
        }
        return _bad(404);
      });
      final p = OnboardingProvider(apiBaseUrl: 'http://x', httpClient: mock);
      p.setAuthToken('test-token');
      await p.detect();
      expect(capturedAuth, 'Bearer test-token');
    });
  });

  group('HardwareSolution.fromJson', () {
    test('handles missing optional fields gracefully', () {
      final s = HardwareSolution.fromJson({
        'tier_id': 't',
        'display_name': 'd',
        'rationale_de': '',
        'rationale_en': '',
        'score': 0.5,
        'score_breakdown': {},
        'blockers': [],
        'warnings': [],
        'is_immediately_runnable': false,
        'estimated_first_response_s': 0,
        'estimated_disk_gb': 0,
        'estimated_setup_minutes': 0,
        'estimated_cost_eur_per_month': 0,
        'backend': 'ollama',
        'model_set': {
          'planner': 'qwen',
          'executor': 'qwen-fast',
          'coder': 'qwen',
          'embedding': 'qwen-emb',
          'formulate': 'qwen-fast',
          'fast_path_validator': 'qwen-fast',
        },
      });
      expect(s.tierId, 't');
      expect(s.modelSet['planner'], 'qwen');
      expect(s.score, 0.5);
    });
  });
}
