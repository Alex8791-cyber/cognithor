/// Configuration state management.
///
/// Loads all config data, tracks dirty state via JSON snapshot comparison,
/// supports deep set via dot-path, parallel save, discard, export/import.
/// Each endpoint is loaded independently so a single failure does not block
/// the rest of the UI.
library;

import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:cognithor_ui/services/api_client.dart';

class ConfigProvider extends ChangeNotifier {
  ApiClient? _api;

  Map<String, dynamic> _cfg = {};
  List<Map<String, dynamic>> _agents = [];
  List<Map<String, dynamic>> _bindings = [];
  Map<String, dynamic> _prompts = {};
  List<Map<String, dynamic>> _cronJobs = [];
  Map<String, dynamic> _mcpServers = {};
  Map<String, dynamic> _a2a = {};

  String _savedSnapshot = '';
  bool _loading = false;
  bool _saving = false;
  String? _error;
  final Map<String, String> _sectionErrors = {};

  // Public getters
  Map<String, dynamic> get cfg => _cfg;
  List<Map<String, dynamic>> get agents => _agents;
  List<Map<String, dynamic>> get bindings => _bindings;
  Map<String, dynamic> get prompts => _prompts;
  List<Map<String, dynamic>> get cronJobs => _cronJobs;
  Map<String, dynamic> get mcpServers => _mcpServers;
  Map<String, dynamic> get a2a => _a2a;
  bool get loading => _loading;
  bool get saving => _saving;
  String? get error => _error;
  Map<String, String> get sectionErrors => _sectionErrors;

  bool get hasChanges => _currentSnapshot() != _savedSnapshot;

  /// True if ANY lead source (Reddit, HN, Discord, RSS) is enabled.
  /// Used to gate the Leads tab in the sidebar.
  bool get leadsEngineEnabled {
    final social = _cfg['social'] as Map<String, dynamic>? ?? {};
    return social['reddit_scan_enabled'] == true ||
        social['hn_enabled'] == true ||
        social['discord_scanner_enabled'] == true ||
        social['rss_enabled'] == true;
  }

  /// Trigger rebuild for direct mutations on [cfg], [prompts], etc.
  void notify() => notifyListeners();

  void setApi(ApiClient api) {
    _api = api;
  }

  // ---------------------------------------------------------------------------
  // Defaults & deep merge
  // ---------------------------------------------------------------------------

  static Map<String, dynamic> _defaults() => {
    'owner_name': 'User',
    'version': '',
    'operation_mode': 'auto',
    'llm_backend_type': 'ollama',
    'cost_tracking_enabled': false,
    'daily_budget_usd': 0,
    'monthly_budget_usd': 0,
    'openai_api_key': '',
    'openai_base_url': '',
    'anthropic_api_key': '',
    'anthropic_max_tokens': 4096,
    'gemini_api_key': '',
    'groq_api_key': '',
    'deepseek_api_key': '',
    'mistral_api_key': '',
    'together_api_key': '',
    'openrouter_api_key': '',
    'xai_api_key': '',
    'cerebras_api_key': '',
    'github_api_key': '',
    'bedrock_api_key': '',
    'huggingface_api_key': '',
    'moonshot_api_key': '',
    'lmstudio_api_key': 'lm-studio',
    'lmstudio_base_url': 'http://localhost:1234/v1',
    'vllm_api_key': '',
    'vllm_base_url': 'http://localhost:8000/v1',
    'llama_cpp_api_key': '',
    'llama_cpp_base_url': 'http://localhost:8080/v1',
    'vision_model': '',
    'vision_model_detail': '',
    'language': 'de',
    'ollama': {
      'base_url': 'http://localhost:11434',
      'timeout_seconds': 120,
      'keep_alive': '30m',
    },
    'models': {
      'planner': {
        'name': 'qwen3:32b',
        'context_window': 32768,
        'vram_gb': 20,
        'strengths': <String>[],
        'speed': 'medium',
        'backend': '',
        'temperature': 0.7,
        'top_p': 0.9,
      },
      'executor': {
        'name': 'qwen3:8b',
        'context_window': 32768,
        'vram_gb': 6,
        'strengths': <String>[],
        'speed': 'fast',
        'backend': '',
        'temperature': 0.7,
        'top_p': 0.9,
      },
      'coder': {
        'name': 'qwen3-coder:30b',
        'context_window': 32768,
        'vram_gb': 20,
        'strengths': <String>[],
        'speed': 'medium',
        'backend': '',
        'temperature': 0.7,
        'top_p': 0.9,
      },
      'embedding': {
        'name': 'qwen3-embedding:0.6b',
        'context_window': 8192,
        'vram_gb': 0.5,
        'strengths': <String>[],
        'speed': 'fast',
        'backend': '',
        'temperature': 0.7,
        'top_p': 0.9,
      },
    },
    'model_overrides': {'skill_models': <String, dynamic>{}},
    'planner': {
      'max_iterations': 25,
      'escalation_after': 3,
      'temperature': 0.7,
      'response_token_budget': 4000,
    },
    'gatekeeper': {
      'policies_dir': 'policies',
      'default_risk_level': 'yellow',
      'max_blocked_retries': 3,
    },
    'sandbox': {
      'level': 'process',
      'timeout_seconds': 30,
      'max_memory_mb': 512,
      'max_cpu_seconds': 10,
      'allowed_paths': <String>[],
      'network_access': false,
      'env_vars': <String, dynamic>{},
    },
    'memory': {
      'chunk_size_tokens': 400,
      'chunk_overlap_tokens': 80,
      'search_top_k': 6,
      'weight_vector': 0.5,
      'weight_bm25': 0.3,
      'weight_graph': 0.2,
      'recency_half_life_days': 30,
      'compaction_threshold': 0.8,
      'compaction_keep_last_n': 8,
      'episodic_retention_days': 365,
      'dynamic_weighting': false,
    },
    'vault': {
      'enabled': true,
      'path': '~/.jarvis/vault',
      'auto_save_research': false,
      'encrypt_files': false,
    },
    'channels': {
      'cli_enabled': true,
      'webui_enabled': true,
      'webui_port': 8741,
      'telegram_enabled': false,
      'telegram_whitelist': <String>[],
      'slack_enabled': false,
      'slack_default_channel': '',
      'discord_enabled': false,
      'discord_channel_id': '',
      'whatsapp_enabled': false,
      'signal_enabled': false,
      'matrix_enabled': false,
      'teams_enabled': false,
      'imessage_enabled': false,
      'google_chat_enabled': false,
      'mattermost_enabled': false,
      'feishu_enabled': false,
      'irc_enabled': false,
      'twitch_enabled': false,
      'voice_enabled': false,
      'voice_config': {
        'tts_backend': 'piper',
        'piper_voice': 'de_DE-pavoque-low',
        'piper_length_scale': 1.0,
        'elevenlabs_api_key': '',
        'elevenlabs_voice_id': '',
        'elevenlabs_model': '',
        'wake_word_enabled': false,
        'wake_word': 'jarvis',
        'wake_word_backend': 'browser',
        'talk_mode_enabled': false,
        'talk_mode_auto_listen': false,
      },
    },
    'security': {
      'max_iterations': 50,
      'max_sub_agent_depth': 5,
      'allowed_paths': <String>[],
      'blocked_commands': <String>[],
      'credential_patterns': <String>[],
    },
    'tools': {'computer_use_enabled': false, 'desktop_tools_enabled': false},
    'browser': {
      'max_text_length': 50000,
      'max_js_length': 10000,
      'default_timeout_ms': 30000,
      'default_viewport_width': 1280,
      'default_viewport_height': 720,
    },
    'calendar': {'enabled': false, 'ics_path': ''},
    'email': {
      'enabled': false,
      'imap_host': '',
      'imap_port': 993,
      'smtp_host': '',
      'smtp_port': 465,
      'username': '',
      'password_env': '',
    },
    'identity': {
      'enabled': true,
      'blockchain_enabled': false,
      'arweave_enabled': false,
    },
    'personality': {
      'warmth': 0.7,
      'humor': 0.3,
      'greeting_enabled': true,
      'success_celebration': true,
    },
    'recovery': {
      'pre_flight_enabled': true,
      'pre_flight_timeout_seconds': 3,
      'pre_flight_min_steps': 2,
      'correction_learning_enabled': true,
      'correction_proactive_threshold': 3,
    },
    'executor': {
      'default_timeout_seconds': 30,
      'max_output_chars': 10000,
      'max_retries': 3,
      'backoff_base_delay_seconds': 1,
      'max_parallel_tools': 4,
      'media_analyze_image_timeout': 180,
      'media_transcribe_audio_timeout': 120,
      'media_extract_text_timeout': 120,
      'media_tts_timeout': 120,
      'run_python_timeout': 120,
    },
    'web': {
      'searxng_url': '',
      'brave_api_key': '',
      'google_cse_api_key': '',
      'google_cse_cx': '',
      'jina_api_key': '',
      'duckduckgo_enabled': true,
      'domain_blocklist': <String>[],
      'domain_allowlist': <String>[],
      'max_fetch_bytes': 500000,
      'max_text_chars': 20000,
      'fetch_timeout_seconds': 15,
      'search_timeout_seconds': 10,
      'max_search_results': 10,
      'ddg_min_delay_seconds': 2,
      'ddg_ratelimit_wait_seconds': 30,
      'ddg_cache_ttl_seconds': 3600,
      'search_and_read_max_chars': 5000,
      'http_request_max_body_bytes': 1048576,
      'http_request_timeout_seconds': 30,
      'http_request_rate_limit_seconds': 1,
    },
    'logging': {'level': 'INFO', 'json_logs': false, 'console': true},
    'database': {
      'backend': 'sqlite',
      'encryption_enabled': false,
      'pg_host': 'localhost',
      'pg_port': 5432,
      'pg_dbname': 'jarvis',
      'pg_user': '',
      'pg_password': '',
      'pg_pool_min': 2,
      'pg_pool_max': 10,
    },
    'heartbeat': {
      'enabled': false,
      'interval_minutes': 30,
      'checklist_file': 'HEARTBEAT.md',
      'channel': 'cli',
      'model': 'qwen3:8b',
    },
    'plugins': {'skills_dir': 'skills', 'auto_update': false},
    'dashboard': {'enabled': false, 'port': 9090},
  };

  static Map<String, dynamic> _deepMerge(
    Map<String, dynamic> base,
    Map<String, dynamic> overlay,
  ) {
    final result = Map<String, dynamic>.from(base);
    for (final key in overlay.keys) {
      if (overlay[key] is Map<String, dynamic> &&
          result[key] is Map<String, dynamic>) {
        result[key] = _deepMerge(
          result[key] as Map<String, dynamic>,
          overlay[key] as Map<String, dynamic>,
        );
      } else {
        result[key] = overlay[key];
      }
    }
    return result;
  }

  Map<String, dynamic> _mergeDefaults(Map<String, dynamic> loaded) {
    return _deepMerge(_defaults(), loaded);
  }

  // ---------------------------------------------------------------------------
  // Deep get/set via dot-path
  // ---------------------------------------------------------------------------

  dynamic getPath(String dotPath) {
    final parts = dotPath.split('.');
    dynamic current = _cfg;
    for (final p in parts) {
      if (current is Map<String, dynamic> && current.containsKey(p)) {
        current = current[p];
      } else {
        return null;
      }
    }
    return current;
  }

  void set(String dotPath, dynamic value) {
    final parts = dotPath.split('.');
    Map<String, dynamic> current = _cfg;
    for (var i = 0; i < parts.length - 1; i++) {
      current.putIfAbsent(parts[i], () => <String, dynamic>{});
      final next = current[parts[i]];
      if (next is Map<String, dynamic>) {
        current = next;
      } else {
        current[parts[i]] = <String, dynamic>{};
        current = current[parts[i]] as Map<String, dynamic>;
      }
    }
    current[parts.last] = value;
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Safe loaders for optional endpoints
  // ---------------------------------------------------------------------------

  Future<List<Map<String, dynamic>>> _loadListSafe(
    String path,
    String key,
  ) async {
    try {
      final r = await _api!.get(path);
      if (!r.containsKey('error')) return _toList(r, key);
    } catch (_) {
      // Endpoint may not exist; return empty list.
    }
    return [];
  }

  Future<Map<String, dynamic>> _loadMapSafe(String path) async {
    try {
      final r = await _api!.get(path);
      if (!r.containsKey('error')) return r;
    } catch (_) {
      // Endpoint may not exist; return empty map.
    }
    return {};
  }

  // ---------------------------------------------------------------------------
  // Load all
  // ---------------------------------------------------------------------------

  Future<void> loadAll() async {
    if (_api == null) return;
    _loading = true;
    _error = null;
    notifyListeners();

    // Load config (required - this is the main one)
    try {
      final r = await _api!.get('config');
      if (!r.containsKey('error')) {
        _cfg = _mergeDefaults(r);
      } else {
        _cfg = _mergeDefaults({});
      }
    } catch (e) {
      _error = 'Failed to load config: $e';
      _cfg = _mergeDefaults({});
    }

    // Load each optional endpoint independently
    _agents = await _loadListSafe('agents', 'agents');
    _bindings = await _loadListSafe('bindings', 'bindings');
    _prompts = await _loadMapSafe('prompts');
    _cronJobs = await _loadListSafe('cron-jobs', 'jobs');
    _mcpServers = await _loadMapSafe('mcp-servers');
    _a2a = await _loadMapSafe('a2a');

    _savedSnapshot = _currentSnapshot();
    _loading = false;
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Save (parallel PATCH)
  // ---------------------------------------------------------------------------

  Future<bool> save() async {
    if (_api == null) return false;
    _saving = true;
    _sectionErrors.clear();
    notifyListeners();

    try {
      final futures = <Future>[];

      // Config sections
      for (final section in [
        'ollama',
        'models',
        'gatekeeper',
        'planner',
        'memory',
        'channels',
        'sandbox',
        'logging',
        'security',
        'heartbeat',
        'plugins',
        'dashboard',
        'model_overrides',
        'web',
        'database',
        'executor',
        'tools',
        'audit',
        'improvement',
        'prompt_evolution',
        'browser',
        'calendar',
        'email',
        'identity',
        'personality',
        'recovery',
        'evolution',
        'vault',
        'social',
        'kanban',
      ]) {
        if (_cfg.containsKey(section)) {
          var sectionData = Map<String, dynamic>.from(
            _cfg[section] as Map<String, dynamic>,
          );
          // Don't overwrite learning_goals — managed by Evolution Goals API
          if (section == 'evolution') {
            sectionData.remove('learning_goals');
          }
          futures.add(
            _api!.patch('config/$section', sectionData).then((r) {
              if (r.containsKey('error'))
                _sectionErrors[section] = r['error'].toString();
            }),
          );
        }
      }

      // Top-level config fields
      final topLevel = <String, dynamic>{};
      for (final key in [
        'owner_name',
        'llm_backend_type',
        'operation_mode',
        'cost_tracking_enabled',
        'daily_budget_usd',
        'monthly_budget_usd',
        'vision_model',
        'vision_model_detail',
        'language',
      ]) {
        if (_cfg.containsKey(key)) {
          final v = _cfg[key];
          // Don't send masked secrets
          if (v is String && v == '***') continue;
          topLevel[key] = v;
        }
      }
      // API keys: only send if not masked and non-empty
      for (final key in _cfg.keys.where(
        (k) => k.endsWith('_api_key') || k.endsWith('_base_url'),
      )) {
        final v = _cfg[key];
        if (v is String && v.isNotEmpty && v != '***') {
          topLevel[key] = v;
        }
      }
      if (topLevel.isNotEmpty) {
        futures.add(
          _api!.patch('config', topLevel).then((r) {
            if (r.containsKey('error'))
              _sectionErrors['general'] = r['error'].toString();
          }),
        );
      }

      // Agents
      for (final agent in _agents) {
        final name = agent['name']?.toString() ?? '';
        if (name.isNotEmpty) {
          futures.add(
            _api!.post('agents/$name', agent).then((r) {
              if (r.containsKey('error'))
                _sectionErrors['agents'] = r['error'].toString();
            }),
          );
        }
      }

      // Bindings
      for (final binding in _bindings) {
        final name = binding['name']?.toString() ?? '';
        if (name.isNotEmpty) {
          futures.add(
            _api!.post('bindings/$name', binding).then((r) {
              if (r.containsKey('error'))
                _sectionErrors['bindings'] = r['error'].toString();
            }),
          );
        }
      }

      // Prompts
      if (_prompts.isNotEmpty) {
        futures.add(
          _api!.put('prompts', _prompts).then((r) {
            if (r.containsKey('error'))
              _sectionErrors['prompts'] = r['error'].toString();
          }),
        );
      }

      // Cron jobs
      futures.add(
        _api!.put('cron-jobs', {'jobs': _cronJobs}).then((r) {
          if (r.containsKey('error'))
            _sectionErrors['cron'] = r['error'].toString();
        }),
      );

      // MCP servers
      if (_mcpServers.isNotEmpty) {
        futures.add(
          _api!.put('mcp-servers', _mcpServers).then((r) {
            if (r.containsKey('error'))
              _sectionErrors['mcp'] = r['error'].toString();
          }),
        );
      }

      // A2A
      if (_a2a.isNotEmpty) {
        futures.add(
          _api!.put('a2a', _a2a).then((r) {
            if (r.containsKey('error'))
              _sectionErrors['a2a'] = r['error'].toString();
          }),
        );
      }

      await Future.wait(futures);

      // Reload to pick up backend post-processing
      final fresh = await _api!.get('config');
      _cfg = _mergeDefaults(_stripError(fresh));
      _savedSnapshot = _currentSnapshot();

      return _sectionErrors.isEmpty;
    } catch (e) {
      _error = e.toString();
      return false;
    } finally {
      _saving = false;
      notifyListeners();
    }
  }

  // ---------------------------------------------------------------------------
  // Discard / Export / Import / Factory Reset
  // ---------------------------------------------------------------------------

  void discard() {
    if (_savedSnapshot.isNotEmpty) {
      final snap = jsonDecode(_savedSnapshot) as Map<String, dynamic>;
      _cfg = snap['cfg'] as Map<String, dynamic>? ?? {};
      _agents = _toListDynamic(snap['agents']);
      _bindings = _toListDynamic(snap['bindings']);
      _prompts = snap['prompts'] as Map<String, dynamic>? ?? {};
      _cronJobs = _toListDynamic(snap['cronJobs']);
      _mcpServers = snap['mcpServers'] as Map<String, dynamic>? ?? {};
      _a2a = snap['a2a'] as Map<String, dynamic>? ?? {};
      notifyListeners();
    }
  }

  String exportJson() {
    return const JsonEncoder.withIndent('  ').convert({
      'cfg': _cfg,
      'agents': _agents,
      'bindings': _bindings,
      'prompts': _prompts,
      'cronJobs': _cronJobs,
      'mcpServers': _mcpServers,
      'a2a': _a2a,
    });
  }

  Future<void> importJson(String json) async {
    try {
      final parsed = jsonDecode(json) as Map<String, dynamic>;
      // Support both full export format (with cfg/agents/etc.) and config-only JSON
      if (parsed.containsKey('cfg')) {
        _cfg = _mergeDefaults(parsed['cfg'] as Map<String, dynamic>? ?? {});
        _agents = _toListDynamic(parsed['agents']);
        _bindings = _toListDynamic(parsed['bindings']);
        _prompts = parsed['prompts'] as Map<String, dynamic>? ?? {};
        _cronJobs = _toListDynamic(parsed['cronJobs']);
        _mcpServers = parsed['mcpServers'] as Map<String, dynamic>? ?? {};
        _a2a = parsed['a2a'] as Map<String, dynamic>? ?? {};
      } else {
        // Treat the entire JSON as a config object
        _cfg = _mergeDefaults(parsed);
      }
      notifyListeners();
    } catch (e) {
      _error = 'Invalid JSON: $e';
      notifyListeners();
    }
  }

  Future<void> factoryReset() async {
    if (_api == null) return;
    await _api!.post('config/factory-reset');
    await loadAll();
  }

  // ---------------------------------------------------------------------------
  // Agent / Binding CRUD
  // ---------------------------------------------------------------------------

  void addAgent(Map<String, dynamic> agent) {
    _agents.add(agent);
    notifyListeners();
  }

  void updateAgent(int index, Map<String, dynamic> agent) {
    if (index >= 0 && index < _agents.length) {
      _agents[index] = agent;
      notifyListeners();
    }
  }

  void removeAgent(int index) {
    if (index >= 0 && index < _agents.length) {
      _agents.removeAt(index);
      notifyListeners();
    }
  }

  void addBinding(Map<String, dynamic> binding) {
    _bindings.add(binding);
    notifyListeners();
  }

  void updateBinding(int index, Map<String, dynamic> binding) {
    if (index >= 0 && index < _bindings.length) {
      _bindings[index] = binding;
      notifyListeners();
    }
  }

  void removeBinding(int index) {
    if (index >= 0 && index < _bindings.length) {
      _bindings.removeAt(index);
      notifyListeners();
    }
  }

  // ---------------------------------------------------------------------------
  // Cron CRUD
  // ---------------------------------------------------------------------------

  void addCronJob(Map<String, dynamic> job) {
    _cronJobs.add(job);
    notifyListeners();
  }

  void updateCronJob(int index, Map<String, dynamic> job) {
    if (index >= 0 && index < _cronJobs.length) {
      _cronJobs[index] = job;
      notifyListeners();
    }
  }

  void removeCronJob(int index) {
    if (index >= 0 && index < _cronJobs.length) {
      _cronJobs.removeAt(index);
      notifyListeners();
    }
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  String _currentSnapshot() {
    return jsonEncode({
      'cfg': _cfg,
      'agents': _agents,
      'bindings': _bindings,
      'prompts': _prompts,
      'cronJobs': _cronJobs,
      'mcpServers': _mcpServers,
      'a2a': _a2a,
    });
  }

  Map<String, dynamic> _stripError(Map<String, dynamic> m) {
    if (m.containsKey('error')) return {};
    return m;
  }

  List<Map<String, dynamic>> _toList(Map<String, dynamic> m, String key) {
    final list = m[key];
    if (list is List) {
      return list
          .map((e) => e is Map<String, dynamic> ? e : <String, dynamic>{})
          .toList();
    }
    return [];
  }

  List<Map<String, dynamic>> _toListDynamic(dynamic v) {
    if (v is List) {
      return v
          .map(
            (e) =>
                e is Map ? Map<String, dynamic>.from(e) : <String, dynamic>{},
          )
          .toList();
    }
    return [];
  }
}
