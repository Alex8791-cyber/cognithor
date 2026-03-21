import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/glass_panel.dart';
import 'package:jarvis_ui/widgets/gradient_background.dart';
import 'package:jarvis_ui/widgets/neon_card.dart';
import 'package:jarvis_ui/screens/main_shell.dart';

/// First-run setup wizard -- 3-step onboarding shown once on initial launch.
///
/// Step 1: Backend selection (Claude / Ollama / OpenAI / Anthropic)
/// Step 2: Backend-specific configuration
/// Step 3: Connection test result + launch
class SetupWizardScreen extends StatefulWidget {
  const SetupWizardScreen({super.key});

  /// SharedPreferences key that gates the wizard.
  static const prefKey = 'first_run_complete';

  @override
  State<SetupWizardScreen> createState() => _SetupWizardScreenState();
}

class _SetupWizardScreenState extends State<SetupWizardScreen> {
  int _step = 0;

  // Step 1 -- backend selection
  String? _selectedBackend;

  // Backend status from API
  Map<String, dynamic>? _backendStatus;
  bool _statusLoading = true;

  // Step 2 -- configuration
  final _ollamaUrlController =
      TextEditingController(text: 'http://localhost:11434');
  final _apiKeyController = TextEditingController();

  // Connection test
  _TestState _testState = _TestState.idle;
  String? _testMessage;

  @override
  void initState() {
    super.initState();
    _loadBackendStatus();
  }

  @override
  void dispose() {
    _ollamaUrlController.dispose();
    _apiKeyController.dispose();
    super.dispose();
  }

  // -- Load backend status from API -------------------------------------------

  Future<void> _loadBackendStatus() async {
    setState(() => _statusLoading = true);
    try {
      final conn = context.read<ConnectionProvider>();
      final result = await conn.api.getBackendStatus();
      if (mounted) {
        setState(() {
          _backendStatus = result;
          _statusLoading = false;
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _backendStatus = null;
          _statusLoading = false;
        });
      }
    }
  }

  // -- Helpers ----------------------------------------------------------------

  Map<String, dynamic> _backendInfo(String key) {
    final backends =
        _backendStatus?['backends'] as Map<String, dynamic>? ?? {};
    return backends[key] as Map<String, dynamic>? ?? {};
  }

  bool _isAuthenticated(String key) =>
      _backendInfo(key)['authenticated'] == true;

  bool _isInstalled(String key) => _backendInfo(key)['installed'] == true;

  List<dynamic> _modelsFor(String key) =>
      _backendInfo(key)['models'] as List<dynamic>? ?? [];

  // -- Navigation -------------------------------------------------------------

  void _next() {
    if (_step < 2) setState(() => _step++);
  }

  void _back() {
    if (_step > 0) {
      setState(() {
        _step--;
        _testState = _TestState.idle;
        _testMessage = null;
      });
    }
  }

  // -- Connection Test --------------------------------------------------------

  Future<void> _testConnection() async {
    final l = AppLocalizations.of(context);
    setState(() {
      _testState = _TestState.testing;
      _testMessage = null;
    });

    try {
      if (_selectedBackend == 'claude-code') {
        // Claude: just check if authenticated
        if (_isAuthenticated('claude-code')) {
          setState(() {
            _testState = _TestState.success;
            _testMessage =
                'Claude Code CLI connected. Version: ${_backendInfo('claude-code')['version'] ?? 'unknown'}';
          });
        } else {
          setState(() {
            _testState = _TestState.error;
            _testMessage = l.notInstalled;
          });
        }
      } else if (_selectedBackend == 'ollama') {
        final url = _ollamaUrlController.text.trim();
        // Reload status to re-check
        await _loadBackendStatus();
        final models = _modelsFor('ollama');
        if (_isAuthenticated('ollama')) {
          setState(() {
            _testState = _TestState.success;
            _testMessage = models.isEmpty
                ? l.ollamaNoModels
                : l.ollamaModelsAvailable(models.length);
          });
        } else {
          setState(() {
            _testState = _TestState.error;
            _testMessage = l.connectionFailed('Ollama not reachable at $url');
          });
        }
      } else {
        // OpenAI / Anthropic -- validate key format
        final key = _apiKeyController.text.trim();
        if (key.isEmpty) {
          setState(() {
            _testState = _TestState.error;
            _testMessage = l.enterApiKey;
          });
          return;
        }
        if (key.length < 20) {
          setState(() {
            _testState = _TestState.error;
            _testMessage = l.apiKeyTooShort(
                _selectedBackend == 'openai' ? 'OpenAI' : 'Anthropic');
          });
          return;
        }
        setState(() {
          _testState = _TestState.success;
          _testMessage = l.apiKeySaved(
              _selectedBackend == 'openai' ? 'OpenAI' : 'Anthropic');
        });
      }
    } catch (e) {
      setState(() {
        _testState = _TestState.error;
        _testMessage = l.connectionFailed(e.toString());
      });
    }
  }

  // -- Switch backend via API -------------------------------------------------

  Future<void> _switchBackend(String backend) async {
    try {
      final conn = context.read<ConnectionProvider>();
      await conn.api.switchBackend(backend);
    } catch (_) {
      // Best-effort; wizard continues even if API is down.
    }
  }

  // -- Finish Wizard ----------------------------------------------------------

  Future<void> _finish() async {
    // Switch backend on server
    if (_selectedBackend != null) {
      await _switchBackend(_selectedBackend!);
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(SetupWizardScreen.prefKey, true);

    if (_selectedBackend == 'ollama') {
      await prefs.setString('jarvis_server_url', 'http://localhost:8741');
      await prefs.setString('ollama_url', _ollamaUrlController.text.trim());
    }

    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(builder: (_) => const MainShell()),
    );
  }

  // -- Build ------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: JarvisTheme.dark,
      child: Scaffold(
        body: GradientBackground(
          particleColor: JarvisTheme.accent,
          child: SafeArea(
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 520),
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 24, vertical: 32),
                  child: Column(
                    children: [
                      _StepIndicator(current: _step),
                      const SizedBox(height: 32),
                      Expanded(
                        child: AnimatedSwitcher(
                          duration: JarvisTheme.animDuration,
                          child: switch (_step) {
                            0 => _buildStep1(),
                            1 => _buildStep2(),
                            2 => _buildStep3(),
                            _ => const SizedBox.shrink(),
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  // -- Step 1: Backend Selection ----------------------------------------------

  Widget _buildStep1() {
    final l = AppLocalizations.of(context);
    final claudeDetected = _isInstalled('claude-code');

    return Column(
      key: const ValueKey('step1'),
      children: [
        Text(
          'COGNITHOR',
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
                fontSize: 40,
                fontWeight: FontWeight.w700,
                color: JarvisTheme.accent,
                letterSpacing: 6,
              ),
        ),
        const SizedBox(height: 8),
        Text(
          l.wizardSubtitle,
          style: Theme.of(context)
              .textTheme
              .bodyMedium
              ?.copyWith(color: JarvisTheme.textSecondary),
        ),
        const SizedBox(height: 24),
        Text(l.chooseBackend,
            style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 16),

        if (_statusLoading)
          const Padding(
            padding: EdgeInsets.all(32),
            child: CircularProgressIndicator(),
          )
        else
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                children: [
                  // 1. Claude Subscription
                  _BackendCard(
                    icon: Icons.psychology,
                    title: l.claudeSubscription,
                    subtitle: l.claudeSubscriptionDesc,
                    tint: JarvisTheme.sectionChat,
                    selected: _selectedBackend == 'claude-code',
                    status: _isAuthenticated('claude-code')
                        ? l.connected
                        : l.notInstalled,
                    statusOk: _isAuthenticated('claude-code'),
                    badge: claudeDetected ? l.recommended : null,
                    onTap: () =>
                        setState(() => _selectedBackend = 'claude-code'),
                  ),
                  const SizedBox(height: 10),

                  // 2. Ollama (Local)
                  _BackendCard(
                    icon: Icons.computer,
                    title: l.ollamaLocal,
                    subtitle: l.ollamaLocalDesc,
                    tint: JarvisTheme.matrix,
                    selected: _selectedBackend == 'ollama',
                    status: _isAuthenticated('ollama')
                        ? '${_modelsFor('ollama').length} models'
                        : l.notInstalled,
                    statusOk: _isAuthenticated('ollama'),
                    onTap: () =>
                        setState(() => _selectedBackend = 'ollama'),
                  ),
                  const SizedBox(height: 10),

                  // 3. OpenAI API
                  _BackendCard(
                    icon: Icons.auto_awesome,
                    title: l.openaiApi,
                    subtitle: 'GPT-5, o3 -- pay-per-use with API key',
                    tint: JarvisTheme.sectionChat,
                    selected: _selectedBackend == 'openai',
                    status: _isAuthenticated('openai')
                        ? l.keyConfigured
                        : l.noKey,
                    statusOk: _isAuthenticated('openai'),
                    onTap: () =>
                        setState(() => _selectedBackend = 'openai'),
                  ),
                  const SizedBox(height: 10),

                  // 4. Anthropic API
                  _BackendCard(
                    icon: Icons.key,
                    title: l.anthropicApi,
                    subtitle: 'Claude via API -- pay-per-use with API key',
                    tint: const Color(0xFFAB68FF),
                    selected: _selectedBackend == 'anthropic',
                    status: _isAuthenticated('anthropic')
                        ? l.keyConfigured
                        : l.noKey,
                    statusOk: _isAuthenticated('anthropic'),
                    onTap: () =>
                        setState(() => _selectedBackend = 'anthropic'),
                  ),
                ],
              ),
            ),
          ),

        const SizedBox(height: 16),
        SizedBox(
          width: double.infinity,
          height: 48,
          child: _NeonButton(
            label: l.next,
            onPressed: _selectedBackend != null ? _next : null,
          ),
        ),
      ],
    );
  }

  // -- Step 2: Configuration --------------------------------------------------

  Widget _buildStep2() {
    final l = AppLocalizations.of(context);

    return Column(
      key: const ValueKey('step2'),
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          _selectedBackend == 'claude-code'
              ? l.claudeSubscription
              : _selectedBackend == 'ollama'
                  ? l.ollamaConfiguration
                  : l.cloudApiConfiguration,
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 8),
        Text(
          _configHint(),
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: 24),

        // Claude Code
        if (_selectedBackend == 'claude-code') ...[
          if (_isAuthenticated('claude-code')) ...[
            GlassPanel(
              tint: JarvisTheme.green,
              padding: const EdgeInsets.all(12),
              child: Row(
                children: [
                  Icon(Icons.check_circle, color: JarvisTheme.green, size: 20),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      '${l.connected} -- ${_backendInfo('claude-code')['version'] ?? ''}',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Text('Available models: opus, sonnet, haiku',
                style: Theme.of(context).textTheme.bodyMedium),
          ] else ...[
            GlassPanel(
              tint: JarvisTheme.red,
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.error, color: JarvisTheme.red, size: 20),
                      const SizedBox(width: 10),
                      Text(l.notInstalled,
                          style: Theme.of(context).textTheme.bodySmall),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '${l.installClaude}: npm install -g @anthropic-ai/claude-code',
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(fontFamily: 'monospace'),
                  ),
                ],
              ),
            ),
          ],
        ],

        // Ollama
        if (_selectedBackend == 'ollama') ...[
          Text(l.ollamaUrl, style: Theme.of(context).textTheme.labelLarge),
          const SizedBox(height: 8),
          TextField(
            controller: _ollamaUrlController,
            decoration: const InputDecoration(
              hintText: 'http://localhost:11434',
              prefixIcon: Icon(Icons.link),
            ),
          ),
        ],

        // OpenAI / Anthropic
        if (_selectedBackend == 'openai' ||
            _selectedBackend == 'anthropic') ...[
          Text(l.apiKey, style: Theme.of(context).textTheme.labelLarge),
          const SizedBox(height: 8),
          TextField(
            controller: _apiKeyController,
            obscureText: true,
            decoration: InputDecoration(
              hintText: _selectedBackend == 'openai' ? 'sk-...' : 'sk-ant-...',
              prefixIcon: const Icon(Icons.key),
            ),
          ),
        ],

        const SizedBox(height: 24),

        // Test connection button
        SizedBox(
          width: double.infinity,
          height: 48,
          child: OutlinedButton.icon(
            onPressed:
                _testState == _TestState.testing ? null : _testConnection,
            icon: _testState == _TestState.testing
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.wifi_tethering),
            label: Text(
              _testState == _TestState.testing
                  ? l.testingConnection
                  : l.testConnection,
            ),
            style: OutlinedButton.styleFrom(
              foregroundColor: JarvisTheme.accent,
              side: BorderSide(color: JarvisTheme.accent),
            ),
          ),
        ),

        // Test result
        if (_testMessage != null) ...[
          const SizedBox(height: 16),
          GlassPanel(
            tint: _testState == _TestState.success
                ? JarvisTheme.green
                : JarvisTheme.red,
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                Icon(
                  _testState == _TestState.success
                      ? Icons.check_circle
                      : Icons.error,
                  color: _testState == _TestState.success
                      ? JarvisTheme.green
                      : JarvisTheme.red,
                  size: 20,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    _testMessage!,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
              ],
            ),
          ),
        ],

        const Spacer(),

        // Navigation buttons
        Row(
          children: [
            TextButton.icon(
              onPressed: _back,
              icon: const Icon(Icons.arrow_back),
              label: Text(l.back),
            ),
            const Spacer(),
            SizedBox(
              height: 48,
              child: _NeonButton(
                label: l.next,
                onPressed: _testState == _TestState.success ? _next : null,
              ),
            ),
          ],
        ),
      ],
    );
  }

  String _configHint() {
    switch (_selectedBackend) {
      case 'claude-code':
        return 'Claude Code uses your existing Claude subscription. No API key needed.';
      case 'ollama':
        return 'Enter the URL where Ollama is running.';
      default:
        return 'Enter your API key to connect.';
    }
  }

  // -- Step 3: Success --------------------------------------------------------

  Widget _buildStep3() {
    final l = AppLocalizations.of(context);
    final backendLabel = switch (_selectedBackend) {
      'claude-code' => 'Claude Subscription',
      'ollama' => 'Ollama',
      'openai' => 'OpenAI',
      'anthropic' => 'Anthropic',
      _ => '',
    };

    return Column(
      key: const ValueKey('step3'),
      children: [
        const Spacer(),
        Icon(Icons.rocket_launch, size: 72, color: JarvisTheme.accent),
        const SizedBox(height: 24),
        Text(
          l.youreAllSet,
          style: Theme.of(context)
              .textTheme
              .titleLarge
              ?.copyWith(fontSize: 28, fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 12),
        Text(
          '$backendLabel is configured. Cognithor will use it for planning and execution.',
          textAlign: TextAlign.center,
          style: Theme.of(context)
              .textTheme
              .bodyMedium
              ?.copyWith(color: JarvisTheme.textSecondary),
        ),
        const SizedBox(height: 8),
        if (_selectedBackend != 'claude-code')
          GlassPanel(
            tint: JarvisTheme.accent,
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                Icon(Icons.info_outline,
                    color: JarvisTheme.accent, size: 18),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(l.restartRequired,
                      style: Theme.of(context).textTheme.bodySmall),
                ),
              ],
            ),
          ),
        const SizedBox(height: 8),
        Text(
          l.changeSettingsAnytime,
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const Spacer(),
        SizedBox(
          width: double.infinity,
          height: 52,
          child: _NeonButton(
            label: l.startUsingCognithor,
            onPressed: _finish,
            glow: true,
          ),
        ),
        const SizedBox(height: 12),
        TextButton.icon(
          onPressed: _back,
          icon: const Icon(Icons.arrow_back),
          label: Text(l.back),
        ),
      ],
    );
  }
}

// -- Helper Types -------------------------------------------------------------

enum _TestState { idle, testing, success, error }

// -- Reusable Widgets ---------------------------------------------------------

/// Backend selection card with status indicator and optional badge.
class _BackendCard extends StatelessWidget {
  const _BackendCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.tint,
    required this.selected,
    required this.status,
    required this.statusOk,
    required this.onTap,
    this.badge,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final Color tint;
  final bool selected;
  final String status;
  final bool statusOk;
  final VoidCallback onTap;
  final String? badge;

  @override
  Widget build(BuildContext context) {
    return NeonCard(
      tint: selected ? tint : null,
      glowOnHover: true,
      onTap: onTap,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      child: Row(
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: (selected ? tint : JarvisTheme.textTertiary)
                  .withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              icon,
              color: selected ? tint : JarvisTheme.textSecondary,
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Flexible(
                      child: Text(
                        title,
                        style:
                            Theme.of(context).textTheme.titleMedium?.copyWith(
                                  color: selected ? tint : null,
                                  fontWeight: FontWeight.w600,
                                ),
                      ),
                    ),
                    if (badge != null) ...[
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: tint.withValues(alpha: 0.18),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          badge!,
                          style: TextStyle(
                              color: tint,
                              fontSize: 10,
                              fontWeight: FontWeight.w700),
                        ),
                      ),
                    ],
                  ],
                ),
                const SizedBox(height: 2),
                Text(subtitle,
                    style: Theme.of(context).textTheme.bodySmall),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Icon(
                      statusOk ? Icons.check_circle : Icons.cancel,
                      size: 14,
                      color: statusOk ? JarvisTheme.green : JarvisTheme.red,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      status,
                      style: Theme.of(context).textTheme.labelSmall?.copyWith(
                            color:
                                statusOk ? JarvisTheme.green : JarvisTheme.red,
                          ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          if (selected) Icon(Icons.check_circle, color: tint, size: 24),
        ],
      ),
    );
  }
}

/// Neon-glowing primary button matching the Sci-Fi aesthetic.
class _NeonButton extends StatelessWidget {
  const _NeonButton({
    required this.label,
    required this.onPressed,
    this.glow = false,
  });

  final String label;
  final VoidCallback? onPressed;
  final bool glow;

  @override
  Widget build(BuildContext context) {
    final enabled = onPressed != null;
    return AnimatedContainer(
      duration: JarvisTheme.animDuration,
      decoration: glow && enabled
          ? BoxDecoration(
              borderRadius: BorderRadius.circular(JarvisTheme.buttonRadius),
              boxShadow: [
                BoxShadow(
                  color: JarvisTheme.accent.withValues(alpha: 0.35),
                  blurRadius: 18,
                  spreadRadius: -2,
                ),
              ],
            )
          : null,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor:
              enabled ? JarvisTheme.accent : JarvisTheme.surface,
          foregroundColor:
              enabled ? JarvisTheme.bg : JarvisTheme.textTertiary,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(JarvisTheme.buttonRadius),
          ),
        ),
        child: Text(
          label,
          style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
        ),
      ),
    );
  }
}

/// Three-dot step indicator at the top of the wizard.
class _StepIndicator extends StatelessWidget {
  const _StepIndicator({required this.current});

  final int current;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: List.generate(3, (i) {
        final active = i <= current;
        return AnimatedContainer(
          duration: JarvisTheme.animDuration,
          margin: const EdgeInsets.symmetric(horizontal: 4),
          width: i == current ? 28 : 10,
          height: 10,
          decoration: BoxDecoration(
            color: active
                ? JarvisTheme.accent
                : JarvisTheme.accent.withValues(alpha: 0.18),
            borderRadius: BorderRadius.circular(5),
          ),
        );
      }),
    );
  }
}
