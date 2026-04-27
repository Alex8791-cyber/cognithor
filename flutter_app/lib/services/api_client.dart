/// Cognithor REST API client.
///
/// Reads the auth token from the HTML <meta name="cognithor-token"> tag
/// (injected by the backend at page-serve time). Falls back to
/// /api/v1/bootstrap for non-web clients (mobile, CLI).
library;

import 'dart:convert';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;

// Conditional import: web DOM only available on web platform.
import 'package:cognithor_ui/services/api_client_token_stub.dart'
    if (dart.library.js_interop) 'package:cognithor_ui/services/api_client_token_web.dart'
    as token_reader;

class ApiClient {
  ApiClient({required this.baseUrl});

  final String baseUrl;
  String? _token;
  bool _fetching = false;

  /// The cached auth token (null until bootstrapped).
  String? get token => _token;

  // ---------------------------------------------------------------------------
  // Token lifecycle
  // ---------------------------------------------------------------------------

  /// Obtain the session token.
  ///
  /// On web: reads from <meta name="cognithor-token"> in the DOM (no HTTP
  /// request needed — token was injected by the backend into index.html).
  /// On mobile/desktop: falls back to /api/v1/bootstrap (localhost-only).
  Future<String?> bootstrap() async {
    if (_token != null) return _token;
    if (_fetching) return _token;
    _fetching = true;
    try {
      // 1. Try DOM meta tag (web only, zero network)
      if (kIsWeb) {
        final metaToken = token_reader.readTokenFromMeta();
        if (metaToken != null && metaToken.isNotEmpty) {
          _token = metaToken;
          return _token;
        }
      }
      // 2. Fallback: API call (mobile, desktop, or meta tag missing)
      final res = await http
          .get(Uri.parse('$baseUrl/api/v1/bootstrap'))
          .timeout(const Duration(seconds: 10));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        _token = data['token'] as String?;
      }
    } catch (_) {
      // Bootstrap failed — will retry on next API call.
    } finally {
      _fetching = false;
    }
    return _token;
  }

  void invalidateToken() {
    _token = null;
  }

  // ---------------------------------------------------------------------------
  // HTTP helpers
  // ---------------------------------------------------------------------------

  Map<String, String> _headers() {
    final h = <String, String>{'Content-Type': 'application/json'};
    if (_token != null) h['Authorization'] = 'Bearer $_token';
    return h;
  }

  Future<Map<String, dynamic>> get(String path) async {
    return _request('GET', path);
  }

  Future<Map<String, dynamic>> post(
    String path, [
    Map<String, dynamic>? body,
  ]) async {
    return _request('POST', path, body);
  }

  Future<Map<String, dynamic>> patch(
    String path, [
    Map<String, dynamic>? body,
  ]) async {
    return _request('PATCH', path, body);
  }

  Future<Map<String, dynamic>> put(
    String path, [
    Map<String, dynamic>? body,
  ]) async {
    return _request('PUT', path, body);
  }

  Future<Map<String, dynamic>> delete(String path) async {
    return _request('DELETE', path);
  }

  /// Sends a multipart POST (for file/audio/image upload).
  Future<Map<String, dynamic>> uploadFile(
    String path,
    String fieldName,
    List<int> bytes,
    String filename, {
    String contentType = 'application/octet-stream',
    Map<String, String>? fields,
  }) async {
    final normalized = path.startsWith('/') ? path : '/$path';
    final uri = Uri.parse('$baseUrl/api/v1$normalized');
    final req = http.MultipartRequest('POST', uri);
    if (_token != null) req.headers['Authorization'] = 'Bearer $_token';
    req.files.add(
      http.MultipartFile.fromBytes(fieldName, bytes, filename: filename),
    );
    if (fields != null) req.fields.addAll(fields);
    final streamed = await req.send().timeout(const Duration(seconds: 120));
    final body = await streamed.stream.bytesToString();
    if (streamed.statusCode == 200) {
      return body.isNotEmpty
          ? jsonDecode(body) as Map<String, dynamic>
          : <String, dynamic>{};
    }
    return {
      'error': 'HTTP ${streamed.statusCode}',
      'status': streamed.statusCode,
    };
  }

  /// Raw bytes GET (for TTS audio).
  Future<List<int>?> getBytes(String path, {Map<String, dynamic>? body}) async {
    final normalized = path.startsWith('/') ? path : '/$path';
    final uri = Uri.parse('$baseUrl/api/v1$normalized');
    final res = body != null
        ? await http
              .post(uri, headers: _headers(), body: jsonEncode(body))
              .timeout(const Duration(seconds: 60))
        : await http
              .get(uri, headers: _headers())
              .timeout(const Duration(seconds: 60));
    return res.statusCode == 200 ? res.bodyBytes : null;
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> _request(
    String method,
    String path, [
    Map<String, dynamic>? body,
  ]) async {
    if (_token == null) await bootstrap();

    var res = await _doRequest(method, path, body);

    // On 401, invalidate token, re-bootstrap, retry once.
    if (res.statusCode == 401) {
      invalidateToken();
      await bootstrap();
      res = await _doRequest(method, path, body);
    }

    if (res.statusCode >= 200 && res.statusCode < 300) {
      if (res.body.isEmpty) return <String, dynamic>{};
      return jsonDecode(res.body) as Map<String, dynamic>;
    }
    return {'error': 'HTTP ${res.statusCode}', 'status': res.statusCode};
  }

  // ---------------------------------------------------------------------------
  // Config
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getConfig() => get('config');
  Future<Map<String, dynamic>> patchConfig(Map<String, dynamic> body) =>
      patch('config', body);
  Future<Map<String, dynamic>> reloadConfig() => post('config/reload', {});
  Future<Map<String, dynamic>> getConfigPresets() => get('config/presets');

  // ---------------------------------------------------------------------------
  // Identity
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getIdentityState() => get('identity/state');
  Future<Map<String, dynamic>> identityDream() => post('identity/dream', {});
  Future<Map<String, dynamic>> identityFreeze() => post('identity/freeze', {});
  Future<Map<String, dynamic>> identityUnfreeze() =>
      post('identity/unfreeze', {});
  Future<Map<String, dynamic>> identityReset() => post('identity/reset', {});

  // ---------------------------------------------------------------------------
  // Monitoring
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getMonitoringDashboard() =>
      get('monitoring/dashboard');
  Future<Map<String, dynamic>> getMonitoringEvents({int n = 50}) =>
      get('monitoring/events?n=$n');
  Future<Map<String, dynamic>> getMonitoringMetrics() =>
      get('monitoring/metrics');
  Future<Map<String, dynamic>> getMonitoringMetric(String name, {int n = 60}) =>
      get('monitoring/metrics/$name?n=$n');
  Future<Map<String, dynamic>> getMonitoringAudit({
    String? action,
    String? severity,
    int limit = 100,
  }) {
    final params = <String>['limit=$limit'];
    if (action != null) params.add('action=$action');
    if (severity != null) params.add('severity=$severity');
    return get('monitoring/audit?${params.join('&')}');
  }

  Future<Map<String, dynamic>> getMonitoringHeartbeat() =>
      get('monitoring/heartbeat');

  // ---------------------------------------------------------------------------
  // Agents & System
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getAgents() => get('agents');
  Future<Map<String, dynamic>> getModels() => get('models/available');
  Future<Map<String, dynamic>> getModelStats() => get('models/stats');
  Future<Map<String, dynamic>> getSystemStatus() => get('status');
  Future<Map<String, dynamic>> getSystemOverview() => get('overview');
  Future<Map<String, dynamic>> shutdownServer() => post('system/stop', {});

  // ---------------------------------------------------------------------------
  // Skills & Marketplace
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getMarketplaceFeatured({int n = 10}) =>
      get('marketplace/featured?n=$n');
  Future<Map<String, dynamic>> getMarketplaceTrending({
    String window = '24h',
    int n = 10,
  }) => get('marketplace/trending?window=$window&n=$n');
  Future<Map<String, dynamic>> getMarketplaceCategories() =>
      get('marketplace/categories');
  Future<Map<String, dynamic>> searchMarketplace(
    String q, {
    int maxResults = 20,
  }) => get(
    'marketplace/search?q=${Uri.encodeComponent(q)}&max_results=$maxResults',
  );
  Future<Map<String, dynamic>> getMarketplaceStats() =>
      get('marketplace/stats');
  Future<Map<String, dynamic>> getInstalledSkills() =>
      get('skill-registry/list');
  Future<Map<String, dynamic>> getSkillDetails(String id) => get('skills/$id');
  Future<Map<String, dynamic>> installSkill(String id) =>
      post('skills/$id/install', {});
  Future<Map<String, dynamic>> uninstallSkill(String id) =>
      delete('skills/$id');

  // ---------------------------------------------------------------------------
  // Memory & Knowledge Graph
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getMemoryGraphStats() =>
      get('memory/graph/stats');
  Future<Map<String, dynamic>> getMemoryGraphEntities() =>
      get('memory/graph/entities');
  Future<Map<String, dynamic>> getEntityRelations(String entityId) =>
      get('memory/graph/entities/$entityId/relations');
  Future<Map<String, dynamic>> getHygieneStats() => get('memory/hygiene/stats');
  Future<Map<String, dynamic>> scanHygiene() => post('memory/hygiene/scan', {});
  Future<Map<String, dynamic>> getQuarantine() =>
      get('memory/hygiene/quarantine');
  Future<Map<String, dynamic>> getMemoryIntegrity() => get('memory/integrity');
  Future<Map<String, dynamic>> getExplainabilityStats() =>
      get('explainability/stats');
  Future<Map<String, dynamic>> getExplainabilityTrails() =>
      get('explainability/trails');
  Future<Map<String, dynamic>> getLowTrustTrails() =>
      get('explainability/low-trust');

  // ---------------------------------------------------------------------------
  // Security & Compliance
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getRbacRoles() => get('rbac/roles');
  Future<Map<String, dynamic>> getAuthStats() => get('auth/stats');
  Future<Map<String, dynamic>> getComplianceReport() =>
      get('compliance/report');
  Future<Map<String, dynamic>> getComplianceStats() => get('compliance/stats');
  Future<Map<String, dynamic>> getComplianceDecisions() =>
      get('compliance/decisions');
  Future<Map<String, dynamic>> getComplianceRemediations() =>
      get('compliance/remediations');
  Future<Map<String, dynamic>> getRedteamStatus() =>
      get('security/redteam/status');
  Future<Map<String, dynamic>> runRedteamScan(Map<String, dynamic> policy) =>
      post('security/redteam/scan', policy);

  // ---------------------------------------------------------------------------
  // Workflows
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getWorkflowCategories() =>
      get('workflows/templates/categories');
  Future<Map<String, dynamic>> startWorkflow(String templateId) =>
      post('workflows/start', {'template_id': templateId});

  // ---------------------------------------------------------------------------
  // Vault
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getVaultStats() => get('vault/stats');
  Future<Map<String, dynamic>> getVaultAgents() => get('vault/agents');

  // ---------------------------------------------------------------------------
  // Credentials
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getCredentials() => get('credentials');
  Future<Map<String, dynamic>> addCredential(Map<String, dynamic> body) =>
      post('credentials', body);
  Future<Map<String, dynamic>> deleteCredential(String service, String key) =>
      delete('credentials/$service/$key');

  // ---------------------------------------------------------------------------
  // Bindings
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getBindings() => get('bindings');
  Future<Map<String, dynamic>> addBinding(Map<String, dynamic> body) =>
      post('bindings', body);
  Future<Map<String, dynamic>> deleteBinding(String name) =>
      delete('bindings/$name');

  // ---------------------------------------------------------------------------
  // Commands & Connectors
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getCommands() => get('commands/list');
  Future<Map<String, dynamic>> getConnectors() => get('connectors/list');
  Future<Map<String, dynamic>> getConnectorStats() => get('connectors/stats');

  // ---------------------------------------------------------------------------
  // Circles & Isolation
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getCircles() => get('circles');
  Future<Map<String, dynamic>> getCirclesStats() => get('circles/stats');
  Future<Map<String, dynamic>> getSandbox() => get('sandbox');
  Future<Map<String, dynamic>> patchSandbox(Map<String, dynamic> body) =>
      patch('sandbox', body);
  Future<Map<String, dynamic>> getIsolationStats() => get('isolation/stats');
  Future<Map<String, dynamic>> getIsolationQuotas() => get('isolation/quotas');

  // ---------------------------------------------------------------------------
  // i18n
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getLocales() => get('i18n/locales');
  Future<Map<String, dynamic>> getI18nStats() => get('i18n/stats');

  // ---------------------------------------------------------------------------
  // Wizards
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getWizards() => get('wizards');
  Future<Map<String, dynamic>> getWizard(String type) => get('wizards/$type');
  Future<Map<String, dynamic>> runWizard(
    String type,
    Map<String, dynamic> values,
  ) => post('wizards/$type/run', values);

  // ---------------------------------------------------------------------------
  // Updater
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getUpdaterStats() => get('updater/stats');
  Future<Map<String, dynamic>> getUpdaterPending() => get('updater/pending');
  Future<Map<String, dynamic>> getUpdaterRecalls() => get('updater/recalls');

  // ---------------------------------------------------------------------------
  // Agent Heartbeat
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getAgentHeartbeatDashboard() =>
      get('agent-heartbeat/dashboard');

  // ---------------------------------------------------------------------------
  // Config sections
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> patchConfigSection(
    String section,
    Map<String, dynamic> body,
  ) => patch('config/$section', body);
  // exportConfig() and importConfig() removed — no backend endpoints exist.
  // Config export/import is handled client-side in system_page.dart.
  // factoryReset() removed — no backend endpoint exists yet.

  // ---------------------------------------------------------------------------
  // Prompts
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getPrompts() => get('prompts');
  Future<Map<String, dynamic>> putPrompts(Map<String, dynamic> body) =>
      put('prompts', body);
  Future<Map<String, dynamic>> translatePrompts(Map<String, dynamic> body) =>
      post('translate-prompts', body);

  // ---------------------------------------------------------------------------
  // Cron Jobs
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getCronJobs() => get('cron-jobs');
  Future<Map<String, dynamic>> putCronJobs(Map<String, dynamic> body) =>
      put('cron-jobs', body);

  // ---------------------------------------------------------------------------
  // MCP Servers
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getMcpServers() => get('mcp-servers');
  Future<Map<String, dynamic>> putMcpServers(Map<String, dynamic> body) =>
      put('mcp-servers', body);

  // ---------------------------------------------------------------------------
  // A2A
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getA2a() => get('a2a');
  Future<Map<String, dynamic>> putA2a(Map<String, dynamic> body) =>
      put('a2a', body);

  // ---------------------------------------------------------------------------
  // Agents CRUD
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getAgent(String name) => get('agents/$name');
  Future<Map<String, dynamic>> createAgent(Map<String, dynamic> body) =>
      post('agents', body);
  Future<Map<String, dynamic>> updateAgent(
    String name,
    Map<String, dynamic> body,
  ) => put('agents/$name', body);
  Future<Map<String, dynamic>> deleteAgent(String name) =>
      delete('agents/$name');

  // ---------------------------------------------------------------------------
  // Bindings CRUD
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> createOrUpdateBinding(
    String name,
    Map<String, dynamic> body,
  ) => post('bindings/$name', body);

  // ---------------------------------------------------------------------------
  // Locales
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getAvailableLocales() => get('i18n/locales');

  // ---------------------------------------------------------------------------
  // Workflow instances & DAG
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getWorkflowInstances() =>
      get('workflows/instances');
  Future<Map<String, dynamic>> getWorkflowDagRuns() =>
      get('workflows/dag/runs');
  Future<Map<String, dynamic>> getWorkflowDagRun(String runId) =>
      get('workflows/dag/runs/$runId');
  Future<Map<String, dynamic>> getWorkflowDagNodeDetail(
    String runId,
    String nodeId,
  ) => get('workflows/dag/runs/$runId/nodes/$nodeId');

  // ---------------------------------------------------------------------------
  // System control
  // ---------------------------------------------------------------------------

  // restartServer() removed — system/restart endpoint does not exist.
  // Use shutdownServer() (system/stop) and restart manually.

  // ---------------------------------------------------------------------------
  // TTS
  // ---------------------------------------------------------------------------

  Future<List<int>?> synthesizeSpeech(String text) =>
      getBytes('/tts', body: {'text': text});

  // ---------------------------------------------------------------------------
  // Learning & Curiosity
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getLearningStats() => get('learning/stats');
  Future<Map<String, dynamic>> getLearningGaps() => get('learning/gaps');
  Future<Map<String, dynamic>> dismissGap(String gapId) =>
      post('learning/gaps/$gapId/dismiss');
  Future<Map<String, dynamic>> getConfidenceHistory() =>
      get('learning/confidence/history');
  Future<Map<String, dynamic>> submitFeedback(String entityId, String type) =>
      post('learning/confidence/$entityId/feedback', {'type': type});
  Future<Map<String, dynamic>> getLearningQueue() => get('learning/queue');
  Future<Map<String, dynamic>> triggerExploration(String gapId) =>
      post('learning/explore', {'gap_id': gapId});
  Future<Map<String, dynamic>> getLearningDirectories() =>
      get('learning/directories');
  Future<Map<String, dynamic>> updateLearningDirectories(
    List<Map<String, dynamic>> dirs,
  ) => post('learning/directories', {'directories': dirs});

  // Q&A Knowledge Base
  Future<Map<String, dynamic>> getQAPairs({String? query, int limit = 50}) {
    final params = <String>['limit=$limit'];
    if (query != null && query.isNotEmpty) {
      params.add('q=${Uri.encodeComponent(query)}');
    }
    return get('learning/qa?${params.join('&')}');
  }

  Future<Map<String, dynamic>> addQAPair(Map<String, dynamic> qa) =>
      post('learning/qa', qa);
  Future<Map<String, dynamic>> verifyQA(String id) =>
      post('learning/qa/$id/verify');
  Future<Map<String, dynamic>> deleteQA(String id) => delete('learning/qa/$id');

  // Lineage
  Future<Map<String, dynamic>> getEntityLineage(String entityId) =>
      get('learning/lineage/$entityId');
  Future<Map<String, dynamic>> getRecentLineage({int limit = 100}) =>
      get('learning/lineage?limit=$limit');

  // Exploration batch
  Future<Map<String, dynamic>> triggerExplorationBatch() =>
      post('learning/explore/run');

  // ---------------------------------------------------------------------------
  // Knowledge Ingestion (Teach)
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> learnFromFile(
    List<int> bytes,
    String filename, {
    String? description,
    String priority = 'normal',
  }) => uploadFile(
    'learn/file',
    'file',
    bytes,
    filename,
    fields: {
      if (description != null) 'description': description,
      'priority': priority,
    },
  );

  Future<Map<String, dynamic>> learnFromUrl(
    String url, {
    String? description,
    String priority = 'normal',
  }) => post('learn/url', {
    'url': url,
    if (description != null) 'description': description,
    'priority': priority,
  });

  Future<Map<String, dynamic>> learnFromYoutube(
    String url, {
    String priority = 'normal',
  }) => post('learn/youtube', {'url': url, 'priority': priority});

  Future<Map<String, dynamic>> getLearnHistory() => get('learn/history');
  Future<Map<String, dynamic>> getLearnStats() => get('learn/stats');
  Future<Map<String, dynamic>> getLearnQueue() => get('learn/queue');

  // ---------------------------------------------------------------------------
  // Backend Status
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> getBackendStatus() => get('backend/status');
  Future<Map<String, dynamic>> switchBackend(String backend) =>
      post('backend/switch', {'backend': backend});

  // ---------------------------------------------------------------------------
  // Sessions
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> listSessions({int limit = 50}) =>
      get('sessions/list?limit=$limit');
  Future<Map<String, dynamic>> getSessionHistory(
    String sessionId, {
    int limit = 100,
  }) => get('sessions/$sessionId/history?limit=$limit');
  Future<Map<String, dynamic>> createSession() => post('sessions/new', {});
  Future<Map<String, dynamic>> createIncognitoSession() =>
      post('sessions/new-incognito', {});
  Future<Map<String, dynamic>> deleteSession(String sessionId) =>
      delete('sessions/$sessionId');
  Future<Map<String, dynamic>> renameSession(String sessionId, String title) =>
      patch('sessions/$sessionId', {'title': title});
  Future<Map<String, dynamic>> moveSessionToFolder(
    String sessionId,
    String folder,
  ) => patch('sessions/$sessionId', {'folder': folder});
  Future<Map<String, dynamic>> listFolders() => get('sessions/folders');
  Future<Map<String, dynamic>> listSessionsByFolder(
    String folder, {
    int limit = 50,
  }) => get('sessions/by-folder/$folder?limit=$limit');
  Future<Map<String, dynamic>> exportSession(String sessionId) =>
      get('sessions/$sessionId/export');

  Future<Map<String, dynamic>> searchSessions(String query, {int limit = 20}) =>
      get('sessions/search?q=${Uri.encodeComponent(query)}&limit=$limit');
  Future<bool> shouldNewSession({int timeoutMinutes = 30}) async {
    final data = await get(
      'sessions/should-new?timeout_minutes=$timeoutMinutes',
    );
    return data['should_new'] == true;
  }

  // ---------------------------------------------------------------------------
  // Reddit Leads
  // ---------------------------------------------------------------------------

  Future<Map<String, dynamic>> scanRedditLeads([Map<String, dynamic>? body]) =>
      post('leads/scan', body ?? {});

  Future<Map<String, dynamic>> getRedditLeads({
    String? status,
    int? minScore,
    int limit = 50,
    int offset = 0,
  }) {
    final params = <String>['limit=$limit', 'offset=$offset'];
    if (status != null && status.isNotEmpty) params.add('status=$status');
    if (minScore != null) params.add('min_score=$minScore');
    return get('leads?${params.join('&')}');
  }

  Future<Map<String, dynamic>> getRedditLead(String id) => get('leads/$id');

  Future<Map<String, dynamic>> updateRedditLead(
    String id,
    Map<String, dynamic> body,
  ) => patch('leads/$id', body);

  Future<Map<String, dynamic>> replyToRedditLead(
    String id, {
    String mode = 'clipboard',
  }) => post('leads/$id/reply', {'mode': mode});

  Future<Map<String, dynamic>> getRedditLeadStats() => get('leads/stats');

  Future<Map<String, dynamic>> refineRedditLead(
    String id, {
    String hint = '',
    int variants = 0,
  }) => post('leads/$id/refine', {'hint': hint, 'variants': variants});

  Future<Map<String, dynamic>> getRedditLeadPerformance(String id) =>
      get('leads/$id/performance');

  Future<Map<String, dynamic>> setRedditLeadFeedback(
    String id, {
    required String tag,
    String note = '',
  }) => patch('leads/$id/feedback', {'tag': tag, 'note': note});

  Future<Map<String, dynamic>> discoverSubreddits([
    Map<String, dynamic>? body,
  ]) => post('leads/discover-subreddits', body ?? {});

  Future<Map<String, dynamic>> getRedditTemplates({String subreddit = ''}) {
    final params = subreddit.isNotEmpty ? '?subreddit=$subreddit' : '';
    return get('leads/templates$params');
  }

  Future<Map<String, dynamic>> createRedditTemplate(
    Map<String, dynamic> body,
  ) => post('leads/templates', body);

  Future<Map<String, dynamic>> deleteRedditTemplate(String id) =>
      delete('leads/templates/$id');

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  Future<http.Response> _doRequest(
    String method,
    String path, [
    Map<String, dynamic>? body,
  ]) async {
    final normalized = path.startsWith('/') ? path : '/$path';
    final uri = Uri.parse('$baseUrl/api/v1$normalized');
    final h = _headers();
    final encoded = body != null ? jsonEncode(body) : null;
    return switch (method) {
      'GET' => http.get(uri, headers: h),
      'POST' => http.post(uri, headers: h, body: encoded),
      'PATCH' => http.patch(uri, headers: h, body: encoded),
      'PUT' => http.put(uri, headers: h, body: encoded),
      'DELETE' => http.delete(uri, headers: h),
      _ => http.get(uri, headers: h),
    }.timeout(const Duration(seconds: 30));
  }
}
