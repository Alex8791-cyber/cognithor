import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/screens/config/bindings_page.dart';
import 'package:cognithor_ui/screens/config/channels_page.dart';
import 'package:cognithor_ui/screens/config/cron_page.dart';
import 'package:cognithor_ui/screens/config/database_page.dart';
import 'package:cognithor_ui/screens/config/evolution_config_page.dart';
import 'package:cognithor_ui/screens/config/executor_page.dart';
import 'package:cognithor_ui/screens/config/general_page.dart';
import 'package:cognithor_ui/screens/config/logging_page.dart';
import 'package:cognithor_ui/screens/config/mcp_page.dart';
import 'package:cognithor_ui/screens/config/memory_page.dart';
import 'package:cognithor_ui/screens/config/planner_page.dart';
import 'package:cognithor_ui/screens/config/prompts_page.dart';
import 'package:cognithor_ui/screens/config/security_page.dart';
import 'package:cognithor_ui/screens/config/social_page.dart';
import 'package:cognithor_ui/screens/config/tools_page.dart';
import 'package:cognithor_ui/screens/config/vault_page.dart';
import 'package:cognithor_ui/screens/config/web_page.dart';

import '../../helpers/silent_providers.dart';
import '../../helpers/test_app.dart';

/// Each config sub-page is a `Consumer<ConfigProvider>` form. They
/// should render cleanly with an empty cfg map and not crash on
/// missing keys.
void main() {
  group('Config sub-pages smoke', () {
    Widget wrap(Widget child) => localizedTestApp(
      child: ChangeNotifierProvider<ConfigProvider>.value(
        value: SilentConfigProvider(),
        child: Scaffold(body: child),
      ),
    );

    final pages = <String, Widget>{
      'LoggingPage': const LoggingPage(),
      'GeneralPage': const GeneralPage(),
      'PromptsPage': const PromptsPage(),
      'SecurityPage': const SecurityPage(),
      'DatabasePage': const DatabasePage(),
      'ToolsPage': const ToolsPage(),
      'ExecutorPage': const ExecutorPage(),
      'EvolutionConfigPage': const EvolutionConfigPage(),
      'MemoryPage': const MemoryPage(),
      'PlannerPage': const PlannerPage(),
      'BindingsConfigPage': const BindingsConfigPage(),
      'ChannelsPage': const ChannelsPage(),
      'CronPage': const CronPage(),
      'McpPage': const McpPage(),
      'SocialPage': const SocialPage(),
      'VaultPage': const VaultPage(),
      'WebPage': const WebPage(),
    };

    for (final entry in pages.entries) {
      testWidgets('${entry.key} renders without crashing on empty cfg', (
        tester,
      ) async {
        await tester.pumpWidget(wrap(entry.value));
        await tester.pump();
        tester.takeException();
        expect(find.byType(Scaffold), findsOneWidget);
      });
    }
  });
}
