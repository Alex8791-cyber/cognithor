import 'dart:async' show unawaited;
import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:cognithor_ui/providers/admin_provider.dart';
import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart'
    show ConnectionProvider, CognithorConnectionState;
import 'package:cognithor_ui/providers/memory_provider.dart';
import 'package:cognithor_ui/providers/security_provider.dart';
import 'package:cognithor_ui/providers/sessions_provider.dart';
import 'package:cognithor_ui/providers/skills_provider.dart';
import 'package:cognithor_ui/providers/workflow_provider.dart';
import 'package:cognithor_ui/providers/packs_provider.dart';
import 'package:cognithor_ui/providers/research_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/screens/main_shell.dart';
import 'package:cognithor_ui/screens/settings_screen.dart';
import 'package:cognithor_ui/screens/setup_wizard_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  bool _navigating = false;

  Future<void> _onConnected() async {
    if (_navigating || !mounted) return;
    _navigating = true;

    final conn = context.read<ConnectionProvider>();
    final api = conn.api;

    // Wire all providers
    context.read<AdminProvider>().setApi(api);
    context.read<SecurityProvider>().setApi(api);
    context.read<MemoryProvider>().setApi(api);
    context.read<SkillsProvider>().setApi(api);
    context.read<WorkflowProvider>().setApi(api);

    final packsProvider = context.read<PacksProvider>();
    packsProvider.setApi(api);
    unawaited(packsProvider.refresh());

    final researchProvider = context.read<ResearchProvider>();
    researchProvider.setApi(api);

    final sessions = context.read<SessionsProvider>();
    sessions.setApi(api);

    final chat = context.read<ChatProvider>();
    chat.attach(conn.ws);

    // Auto-session: resume recent or create new based on inactivity timeout
    final sessionId =
        await sessions.autoSessionOnStartup() ??
        'flutter_${DateTime.now().millisecondsSinceEpoch}';
    conn.ws.connect(sessionId);

    // Check if the first-run wizard has been completed.
    final prefs = await SharedPreferences.getInstance();
    final firstRunComplete = prefs.getBool(SetupWizardScreen.prefKey) ?? false;

    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(
        builder: (_) =>
            firstRunComplete ? const MainShell() : const SetupWizardScreen(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final conn = context.watch<ConnectionProvider>();
    final l = AppLocalizations.of(context);

    // Auto-navigate when connected — only once via _navigating guard
    if (conn.state == CognithorConnectionState.connected && !_navigating) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _onConnected());
    }

    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Logo / Title
              Text(
                l.appTitle,
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontSize: 48,
                  fontWeight: FontWeight.w700,
                  color: CognithorTheme.accent,
                  letterSpacing: 4,
                ),
              ),
              const SizedBox(height: 32),

              if (conn.state == CognithorConnectionState.connecting) ...[
                const CircularProgressIndicator(),
                const SizedBox(height: 16),
                Text(
                  l.connecting,
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],

              if (conn.state == CognithorConnectionState.error) ...[
                Icon(
                  conn.versionMismatch
                      ? Icons.system_update_alt
                      : Icons.cloud_off,
                  size: 48,
                  color: CognithorTheme.red,
                ),
                const SizedBox(height: 16),
                Text(
                  conn.versionMismatch ? 'Version Mismatch' : l.connectionError,
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(color: CognithorTheme.red),
                ),
                const SizedBox(height: 8),
                if (conn.versionMismatch) ...[
                  Text(
                    'Frontend version: ${conn.frontendVersion}',
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  Text(
                    'Backend version: ${conn.backendVersion ?? "unknown"}',
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'Update Cognithor via the EXE installer or run:\n'
                    'pip install --upgrade cognithor',
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ] else
                  Text(
                    l.connectionErrorDetail(conn.serverUrl),
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                const SizedBox(height: 24),
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    ElevatedButton.icon(
                      onPressed: conn.connect,
                      icon: const Icon(Icons.refresh),
                      label: Text(l.retry),
                    ),
                    const SizedBox(width: 12),
                    OutlinedButton.icon(
                      onPressed: () => Navigator.of(context).push(
                        MaterialPageRoute<void>(
                          builder: (_) => const SettingsScreen(),
                        ),
                      ),
                      icon: const Icon(Icons.settings),
                      label: Text(l.settings),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: CognithorTheme.accent,
                        side: BorderSide(color: CognithorTheme.accent),
                      ),
                    ),
                  ],
                ),
              ],

              if (conn.state == CognithorConnectionState.disconnected) ...[
                Text(
                  l.connecting,
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
