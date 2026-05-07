// Hardware-Aware Onboarding Provider.
//
// Wraps the /api/system/* endpoints (system_api.py) for the
// hardware-wizard Flutter screens. Mirrors the Python `cognithor doctor`
// flow: detection → objective → solutions → apply → done.

import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class HardwareSolution {
  HardwareSolution({
    required this.tierId,
    required this.displayName,
    required this.rationaleDe,
    required this.rationaleEn,
    required this.score,
    required this.scoreBreakdown,
    required this.blockers,
    required this.warnings,
    required this.isImmediatelyRunnable,
    required this.estimatedFirstResponseS,
    required this.estimatedDiskGb,
    required this.estimatedSetupMinutes,
    required this.estimatedCostEurPerMonth,
    required this.backend,
    required this.modelSet,
  });

  final String tierId;
  final String displayName;
  final String rationaleDe;
  final String rationaleEn;
  final double score;
  final Map<String, double> scoreBreakdown;
  final List<String> blockers;
  final List<String> warnings;
  final bool isImmediatelyRunnable;
  final double estimatedFirstResponseS;
  final double estimatedDiskGb;
  final int estimatedSetupMinutes;
  final double estimatedCostEurPerMonth;
  final String backend;
  final Map<String, String> modelSet;

  factory HardwareSolution.fromJson(Map<String, dynamic> j) {
    final breakdown = <String, double>{};
    final raw = j['score_breakdown'];
    if (raw is Map) {
      raw.forEach((k, v) => breakdown[k.toString()] = (v as num).toDouble());
    }
    final ms = <String, String>{};
    final rawMs = j['model_set'];
    if (rawMs is Map) {
      rawMs.forEach((k, v) => ms[k.toString()] = v.toString());
    }
    return HardwareSolution(
      tierId: j['tier_id'] as String,
      displayName: j['display_name'] as String,
      rationaleDe: j['rationale_de'] as String? ?? '',
      rationaleEn: j['rationale_en'] as String? ?? '',
      score: (j['score'] as num).toDouble(),
      scoreBreakdown: breakdown,
      blockers: (j['blockers'] as List? ?? []).map((e) => e.toString()).toList(),
      warnings: (j['warnings'] as List? ?? []).map((e) => e.toString()).toList(),
      isImmediatelyRunnable: j['is_immediately_runnable'] as bool? ?? false,
      estimatedFirstResponseS:
          (j['estimated_first_response_s'] as num? ?? 0).toDouble(),
      estimatedDiskGb: (j['estimated_disk_gb'] as num? ?? 0).toDouble(),
      estimatedSetupMinutes: (j['estimated_setup_minutes'] as num? ?? 0).toInt(),
      estimatedCostEurPerMonth:
          (j['estimated_cost_eur_per_month'] as num? ?? 0).toDouble(),
      backend: j['backend'] as String? ?? '',
      modelSet: ms,
    );
  }
}

class HardwareCapabilities {
  HardwareCapabilities({
    required this.canRunNvfp4,
    required this.canRunFp8Marlin,
    required this.canRunGgufCuda,
    required this.canRunGgufMetal,
    required this.canRunVllmContainer,
    required this.canRunOllamaNative,
    required this.vramClass,
    required this.ramClass,
    required this.diskClass,
    required this.hasInternet,
    required this.profileHash,
    required this.multiGpuCount,
  });

  final bool canRunNvfp4;
  final bool canRunFp8Marlin;
  final bool canRunGgufCuda;
  final bool canRunGgufMetal;
  final bool canRunVllmContainer;
  final bool canRunOllamaNative;
  final String vramClass;
  final String ramClass;
  final String diskClass;
  final bool hasInternet;
  final String profileHash;
  final int multiGpuCount;

  factory HardwareCapabilities.fromJson(Map<String, dynamic> j) {
    return HardwareCapabilities(
      canRunNvfp4: j['can_run_nvfp4'] as bool? ?? false,
      canRunFp8Marlin: j['can_run_fp8_marlin'] as bool? ?? false,
      canRunGgufCuda: j['can_run_gguf_cuda'] as bool? ?? false,
      canRunGgufMetal: j['can_run_gguf_metal'] as bool? ?? false,
      canRunVllmContainer: j['can_run_vllm_container'] as bool? ?? false,
      canRunOllamaNative: j['can_run_ollama_native'] as bool? ?? false,
      vramClass: j['vram_class'] as String? ?? 'none',
      ramClass: j['ram_class'] as String? ?? 'low',
      diskClass: j['disk_class'] as String? ?? 'very_low',
      hasInternet: j['has_internet'] as bool? ?? false,
      profileHash: j['profile_hash'] as String? ?? '',
      multiGpuCount: (j['multi_gpu_count'] as num? ?? 1).toInt(),
    );
  }
}

class HardwareProfileSummary {
  HardwareProfileSummary({
    required this.tier,
    required this.recommendedMode,
    required this.sanityWarnings,
    required this.components,
  });

  final String tier;
  final String recommendedMode;
  final List<Map<String, String>> sanityWarnings;
  final Map<String, Map<String, dynamic>> components;

  factory HardwareProfileSummary.fromJson(Map<String, dynamic> j) {
    final warns = <Map<String, String>>[];
    for (final w in (j['sanity_warnings'] as List? ?? [])) {
      if (w is Map) {
        warns.add({
          'rule_id': w['rule_id']?.toString() ?? '',
          'severity': w['severity']?.toString() ?? '',
          'message': w['message']?.toString() ?? '',
        });
      }
    }
    final comps = <String, Map<String, dynamic>>{};
    final rawComps = j['components'];
    if (rawComps is Map) {
      rawComps.forEach((k, v) {
        if (v is Map) {
          comps[k.toString()] = Map<String, dynamic>.from(v);
        }
      });
    }
    return HardwareProfileSummary(
      tier: j['tier'] as String? ?? '',
      recommendedMode: j['recommended_mode'] as String? ?? '',
      sanityWarnings: warns,
      components: comps,
    );
  }
}

enum WizardStage {
  idle,
  detecting,
  detected,
  loadingRecommendations,
  awaitingChoice,
  applying,
  applied,
  failed,
}

class OnboardingProvider extends ChangeNotifier {
  OnboardingProvider({required this.apiBaseUrl, http.Client? httpClient})
      : _http = httpClient ?? http.Client();

  final String apiBaseUrl;
  final http.Client _http;

  WizardStage _stage = WizardStage.idle;
  HardwareProfileSummary? _profile;
  HardwareCapabilities? _capabilities;
  List<HardwareSolution> _solutions = [];
  String _objectivePreset = 'balanced';
  String? _appliedTierId;
  String? _errorMessage;
  String? _authToken;

  WizardStage get stage => _stage;
  HardwareProfileSummary? get profile => _profile;
  HardwareCapabilities? get capabilities => _capabilities;
  List<HardwareSolution> get solutions => List.unmodifiable(_solutions);
  String get objectivePreset => _objectivePreset;
  String? get appliedTierId => _appliedTierId;
  String? get errorMessage => _errorMessage;

  void setAuthToken(String? token) {
    _authToken = token;
  }

  Map<String, String> get _headers => {
        'Content-Type': 'application/json',
        if (_authToken != null) 'Authorization': 'Bearer $_authToken',
      };

  Future<void> detect() async {
    _stage = WizardStage.detecting;
    _errorMessage = null;
    notifyListeners();
    try {
      final pf = await _http
          .get(Uri.parse('$apiBaseUrl/api/system/profile'), headers: _headers)
          .timeout(const Duration(seconds: 20));
      if (pf.statusCode != 200) {
        throw Exception('profile HTTP ${pf.statusCode}');
      }
      _profile = HardwareProfileSummary.fromJson(
        jsonDecode(pf.body) as Map<String, dynamic>,
      );

      final cp = await _http
          .get(Uri.parse('$apiBaseUrl/api/system/capabilities'),
              headers: _headers)
          .timeout(const Duration(seconds: 15));
      if (cp.statusCode != 200) {
        throw Exception('capabilities HTTP ${cp.statusCode}');
      }
      _capabilities = HardwareCapabilities.fromJson(
        jsonDecode(cp.body) as Map<String, dynamic>,
      );
      _stage = WizardStage.detected;
    } on Exception catch (e) {
      _errorMessage = e.toString();
      _stage = WizardStage.failed;
    } finally {
      notifyListeners();
    }
  }

  Future<void> setObjective(String preset) async {
    _objectivePreset = preset;
    _stage = WizardStage.loadingRecommendations;
    notifyListeners();
    try {
      final r = await _http.get(
        Uri.parse(
          '$apiBaseUrl/api/system/recommendations?objective=$preset&max_solutions=5',
        ),
        headers: _headers,
      ).timeout(const Duration(seconds: 20));
      if (r.statusCode != 200) {
        throw Exception('recommendations HTTP ${r.statusCode}');
      }
      final data = jsonDecode(r.body) as Map<String, dynamic>;
      _solutions = ((data['solutions'] as List?) ?? [])
          .whereType<Map<String, dynamic>>()
          .map(HardwareSolution.fromJson)
          .toList();
      _stage = WizardStage.awaitingChoice;
    } on Exception catch (e) {
      _errorMessage = e.toString();
      _stage = WizardStage.failed;
    } finally {
      notifyListeners();
    }
  }

  Future<void> apply(String tierId) async {
    _stage = WizardStage.applying;
    _errorMessage = null;
    notifyListeners();
    try {
      final r = await _http.post(
        Uri.parse('$apiBaseUrl/api/system/apply'),
        headers: _headers,
        body: jsonEncode({
          'tier_id': tierId,
          'objective_preset': _objectivePreset,
          'user_confirmed': true,
        }),
      ).timeout(const Duration(seconds: 30));
      if (r.statusCode != 200) {
        throw Exception('apply HTTP ${r.statusCode}: ${r.body}');
      }
      _appliedTierId = tierId;
      _stage = WizardStage.applied;
    } on Exception catch (e) {
      _errorMessage = e.toString();
      _stage = WizardStage.failed;
    } finally {
      notifyListeners();
    }
  }

  void reset() {
    _stage = WizardStage.idle;
    _profile = null;
    _capabilities = null;
    _solutions = [];
    _objectivePreset = 'balanced';
    _appliedTierId = null;
    _errorMessage = null;
    notifyListeners();
  }
}
