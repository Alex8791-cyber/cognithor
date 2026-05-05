/// Test-only silent subclasses of providers that aggressively notify
/// from `didChangeDependencies` / `initState`.
///
/// The default Flutter test framework treats `notifyListeners()` during
/// the initial mount as a fatal `_dirty` assertion and unmounts the
/// widget tree — which makes screens that follow the
/// "load-on-mount → notifyListeners" pattern impossible to render
/// in a smoke test.
///
/// Each silent subclass below extends the real provider but overrides
/// every `loadX()` / `fetchX()` method to be a no-op. Pure state
/// (lists, maps, flags) keeps its empty defaults. Tests can still
/// call `setApi()` (a sync setter that doesn't notify) to make sure
/// the screen's wiring code doesn't crash on a null api.
///
/// New screens that hit this issue can either:
///   - extend the silent provider here with a no-op override of the
///     specific load method, or
///   - in the source code, move the `loadX()` call into
///     `WidgetsBinding.instance.addPostFrameCallback(...)` (the
///     cleaner long-term fix; see `evolution_goals_page.dart` for
///     the reference shape).
library;

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/providers/cron_provider.dart';
import 'package:cognithor_ui/providers/kanban_provider.dart';
import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/providers/memory_provider.dart';
import 'package:cognithor_ui/providers/reddit_leads_provider.dart';
import 'package:cognithor_ui/providers/robot_office_provider.dart';
import 'package:cognithor_ui/providers/security_provider.dart';
import 'package:cognithor_ui/providers/sessions_provider.dart';
import 'package:cognithor_ui/providers/skills_provider.dart';
import 'package:cognithor_ui/providers/sources_provider.dart';
import 'package:cognithor_ui/providers/workflow_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';
import 'package:cognithor_ui/services/websocket_service.dart';

class SilentSkillsProvider extends SkillsProvider {
  @override
  Future<void> loadFeatured() async {}
  @override
  Future<void> loadTrending() async {}
  @override
  Future<void> loadCategories() async {}
  @override
  Future<void> loadInstalled() async {}
  @override
  Future<void> search(String q) async {}
}

class SilentSecurityProvider extends SecurityProvider {
  @override
  Future<void> loadRoles() async {}
  @override
  Future<void> loadComplianceReport() async {}
  @override
  Future<void> loadComplianceStats() async {}
  @override
  Future<void> loadDecisions() async {}
  @override
  Future<void> loadRemediations() async {}
  @override
  Future<void> loadRedteamStatus() async {}
  @override
  Future<void> loadAudit({String? action, String? severity}) async {}
  @override
  Future<void> loadAuthStats() async {}
}

class SilentMemoryProvider extends MemoryProvider {
  @override
  Future<void> loadGraphStats() async {}
  @override
  Future<void> loadEntities() async {}
  @override
  Future<void> loadHygieneStats() async {}
  @override
  Future<void> loadQuarantine() async {}
  @override
  Future<void> loadExplainability() async {}
  @override
  Future<void> loadTrails() async {}
  @override
  Future<void> loadLowTrustTrails() async {}
}

class SilentAdminProvider extends AdminProvider {
  @override
  Future<void> loadSystemStatus() async {}
  @override
  Future<void> loadAgents() async {}
  @override
  Future<void> loadModels() async {}
  @override
  Future<void> loadModelStats() async {}
  @override
  Future<void> loadVaultStats() async {}
  @override
  Future<void> loadVaultAgents() async {}
  @override
  Future<void> loadCredentials() async {}
  @override
  Future<void> loadBindings() async {}
  @override
  Future<void> loadCommands() async {}
  @override
  Future<void> loadConnectors() async {}
  @override
  Future<void> loadIsolationStats() async {}
  @override
  Future<void> loadCircles() async {}
}

class SilentWorkflowProvider extends WorkflowProvider {
  @override
  Future<void> loadCategories() async {}
}

class SilentRedditLeadsProvider extends RedditLeadsProvider {
  @override
  void init(ApiClient api) {}
  @override
  Future<void> fetchLeads() async {}
  @override
  Future<void> fetchStats() async {}
}

class SilentSourcesProvider extends SourcesProvider {
  @override
  Future<void> refresh() async {}
}

class SilentConfigProvider extends ConfigProvider {
  @override
  Future<void> loadAll() async {}
}

class SilentKanbanProvider extends KanbanProvider {
  @override
  Future<void> fetchTasks({String? status, String? agent}) async {}
  @override
  Future<void> fetchStats() async {}
}

class SilentCronProvider extends CronProvider {
  @override
  Future<void> fetchJobs() async {}
}

/// Silent variant of [RobotOfficeProvider] — overrides [init] to skip
/// the periodic 10-second poll Timer and the WebSocket listener wiring.
/// Used by `dashboard_screen_test.dart` and `main_shell_test.dart`,
/// which mount [DashboardScreen] (it calls `init()` from `_loadData`
/// when the connection is live).
class SilentRobotOfficeProvider extends RobotOfficeProvider {
  @override
  void init(ApiClient api, WebSocketService? ws) {
    // intentionally a no-op — no Timer, no WS listeners.
  }
}

/// Silent variant of [SessionsProvider] — overrides every load method
/// fired from `ChatScreen.didChangeDependencies` so no notifyListeners
/// fires during the first build pass.
class SilentSessionsProvider extends SessionsProvider {
  @override
  Future<void> loadSessions() async {}
  @override
  Future<void> loadFolders() async {}
  @override
  Future<List<Map<String, dynamic>>?> loadHistory(String sessionId) async =>
      null;
  @override
  Future<void> searchChats(String query) async {}
}

/// Silent variant of [LlmBackendProvider] — overrides [startPolling]
/// (its only periodic Timer) to a no-op so smoke tests embedding
/// `ChatInput` (which reads `LlmBackendProvider.active`) don't leak
/// a 2-second polling Timer.
class SilentLlmBackendProvider extends LlmBackendProvider {
  SilentLlmBackendProvider() : super(apiBaseUrl: 'http://test');

  @override
  void startPolling() {
    // intentionally a no-op.
  }

  @override
  Future<void> refreshList() async {}

  @override
  Future<void> refreshVllmStatus() async {}
}
