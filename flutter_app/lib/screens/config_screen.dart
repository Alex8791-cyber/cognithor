import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/providers/config_provider.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/glass_panel.dart';
import 'package:jarvis_ui/widgets/jarvis_toast.dart';

import 'package:jarvis_ui/screens/config/general_page.dart';
import 'package:jarvis_ui/screens/config/language_page.dart';
import 'package:jarvis_ui/screens/config/providers_page.dart';
import 'package:jarvis_ui/screens/config/models_page.dart';
import 'package:jarvis_ui/screens/config/planner_page.dart';
import 'package:jarvis_ui/screens/config/executor_page.dart';
import 'package:jarvis_ui/screens/config/memory_page.dart';
import 'package:jarvis_ui/screens/config/channels_page.dart';
import 'package:jarvis_ui/screens/config/security_page.dart';
import 'package:jarvis_ui/screens/config/web_page.dart';
import 'package:jarvis_ui/screens/config/mcp_page.dart';
import 'package:jarvis_ui/screens/config/cron_page.dart';
import 'package:jarvis_ui/screens/config/database_page.dart';
import 'package:jarvis_ui/screens/config/logging_page.dart';
import 'package:jarvis_ui/screens/config/prompts_page.dart';
import 'package:jarvis_ui/screens/config/agents_page.dart';
import 'package:jarvis_ui/screens/config/bindings_page.dart';
import 'package:jarvis_ui/screens/config/system_page.dart';
import 'package:jarvis_ui/screens/config/tools_page.dart';

// ── Category definition ──────────────────────────────────────────────────────

class _Category {
  const _Category(this.labelKey, this.icon, this.pageKeys);
  final String Function(AppLocalizations l) labelKey;
  final IconData icon;
  final List<String> pageKeys;
}

final _categories = [
  _Category((l) => l.catAiEngine, Icons.psychology, [
    'providers', 'models', 'planner', 'executor', 'prompts',
  ]),
  _Category((l) => l.catChannels, Icons.cell_tower, [
    'channels',
  ]),
  _Category((l) => l.catKnowledge, Icons.storage, [
    'memory', 'agents', 'bindings', 'web',
  ]),
  _Category((l) => l.catSecurity, Icons.shield, [
    'security', 'tools', 'database',
  ]),
  _Category((l) => l.catSystem, Icons.settings, [
    'general', 'language', 'logging', 'cron', 'mcp', 'system',
  ]),
];

// ── Page key → builder + label + icon ────────────────────────────────────────

class _SubPageDef {
  const _SubPageDef(this.icon, this.labelKey, this.builder);
  final IconData icon;
  final String Function(AppLocalizations l) labelKey;
  final Widget Function() builder;
}

final _pageRegistry = <String, _SubPageDef>{
  'general': _SubPageDef(
      Icons.settings, (l) => l.configPageGeneral, () => const GeneralPage()),
  'language': _SubPageDef(
      Icons.language, (l) => l.configPageLanguage, () => const LanguagePage()),
  'providers': _SubPageDef(
      Icons.cloud, (l) => l.configPageProviders, () => const ProvidersPage()),
  'models': _SubPageDef(Icons.model_training, (l) => l.configPageModels,
      () => const ModelsPage()),
  'planner': _SubPageDef(Icons.architecture, (l) => l.configPagePlanner,
      () => const PlannerPage()),
  'executor': _SubPageDef(Icons.play_arrow, (l) => l.configPageExecutor,
      () => const ExecutorPage()),
  'memory': _SubPageDef(
      Icons.memory, (l) => l.configPageMemory, () => const MemoryPage()),
  'channels': _SubPageDef(
      Icons.chat, (l) => l.configPageChannels, () => const ChannelsPage()),
  'security': _SubPageDef(
      Icons.shield, (l) => l.configPageSecurity, () => const SecurityPage()),
  'tools': _SubPageDef(
      Icons.desktop_windows, (l) => l.configPageTools, () => const ToolsPage()),
  'web': _SubPageDef(
      Icons.public, (l) => l.configPageWeb, () => const WebPage()),
  'mcp':
      _SubPageDef(Icons.dns, (l) => l.configPageMcp, () => const McpPage()),
  'cron': _SubPageDef(
      Icons.schedule, (l) => l.configPageCron, () => const CronPage()),
  'database': _SubPageDef(Icons.storage, (l) => l.configPageDatabase,
      () => const DatabasePage()),
  'logging': _SubPageDef(
      Icons.article, (l) => l.configPageLogging, () => const LoggingPage()),
  'prompts': _SubPageDef(Icons.edit_note, (l) => l.configPagePrompts,
      () => const PromptsPage()),
  'agents': _SubPageDef(Icons.smart_toy, (l) => l.configPageAgents,
      () => const AgentsConfigPage()),
  'bindings': _SubPageDef(
      Icons.link, (l) => l.configPageBindings, () => const BindingsConfigPage()),
  'system': _SubPageDef(
      Icons.build, (l) => l.configPageSystem, () => const SystemConfigPage()),
};

// ── Config Screen ────────────────────────────────────────────────────────────

class ConfigScreen extends StatefulWidget {
  const ConfigScreen({super.key});

  @override
  State<ConfigScreen> createState() => _ConfigScreenState();
}

class _ConfigScreenState extends State<ConfigScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabController;
  int _selectedCategory = 0;
  int _selectedSubPage = 0;
  bool _initialized = false;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: _categories.length, vsync: this);
    _tabController.addListener(_onTabChanged);
  }

  @override
  void dispose() {
    _tabController.removeListener(_onTabChanged);
    _tabController.dispose();
    super.dispose();
  }

  void _onTabChanged() {
    if (!_tabController.indexIsChanging) {
      setState(() {
        _selectedCategory = _tabController.index;
        _selectedSubPage = 0;
      });
    }
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      _initialized = true;
      final cfg = context.read<ConfigProvider>();
      final conn = context.read<ConnectionProvider>();
      cfg.setApi(conn.api);
      if (cfg.cfg.isEmpty) {
        cfg.loadAll();
      }
    }
  }

  List<String> get _currentPageKeys =>
      _categories[_selectedCategory].pageKeys;

  String get _currentPageKey => _currentPageKeys[_selectedSubPage];

  Future<void> _save() async {
    final cfg = context.read<ConfigProvider>();
    final l = AppLocalizations.of(context);
    final ok = await cfg.save();
    if (!mounted) return;
    if (cfg.sectionErrors.isNotEmpty) {
      final errSections = cfg.sectionErrors.keys.join(', ');
      JarvisToast.show(
        context,
        l.savedWithErrors(errSections),
        type: ToastType.warning,
      );
    } else {
      JarvisToast.show(
        context,
        ok ? l.configurationSaved : l.saveFailed,
        type: ok ? ToastType.success : ToastType.error,
      );
    }
  }

  void _navigateToSubPage(int index) {
    if (index >= 0 && index < _currentPageKeys.length) {
      setState(() => _selectedSubPage = index);
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final isWide = MediaQuery.sizeOf(context).width > 700;

    // Keyboard shortcuts: Ctrl+S save, Ctrl+1-0 for sub-pages within category
    final bindings = <ShortcutActivator, VoidCallback>{
      const SingleActivator(LogicalKeyboardKey.keyS, control: true): _save,
    };
    final digitKeys = [
      LogicalKeyboardKey.digit1, LogicalKeyboardKey.digit2,
      LogicalKeyboardKey.digit3, LogicalKeyboardKey.digit4,
      LogicalKeyboardKey.digit5, LogicalKeyboardKey.digit6,
      LogicalKeyboardKey.digit7, LogicalKeyboardKey.digit8,
      LogicalKeyboardKey.digit9, LogicalKeyboardKey.digit0,
    ];
    for (var i = 0; i < digitKeys.length && i < _currentPageKeys.length; i++) {
      final idx = i;
      bindings[SingleActivator(digitKeys[i], control: true)] =
          () => _navigateToSubPage(idx);
    }

    return CallbackShortcuts(
      bindings: bindings,
      child: Focus(
        autofocus: true,
        child: Scaffold(
          appBar: AppBar(
            title: Text(l.configTitle),
            actions: [
              Consumer<ConfigProvider>(
                builder: (context, cfg, _) => IconButton(
                  icon: const Icon(Icons.refresh, size: 20),
                  tooltip: AppLocalizations.of(context).reloadFromBackend,
                  onPressed: cfg.loading ? null : () => cfg.loadAll(),
                ),
              ),
            ],
          ),
          body: Consumer<ConfigProvider>(
            builder: (context, cfg, _) {
              if (cfg.loading && cfg.cfg.isEmpty) {
                return const Center(child: CircularProgressIndicator());
              }

              return Column(
                children: [
                  // Error banner
                  if (cfg.error != null)
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 8),
                      color: JarvisTheme.red.withValues(alpha: 0.15),
                      child: Row(
                        children: [
                          Icon(Icons.warning_amber,
                              size: 16, color: JarvisTheme.orange),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              cfg.error!,
                              style: TextStyle(
                                  color: JarvisTheme.orange, fontSize: 12),
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          TextButton(
                            onPressed: () => cfg.loadAll(),
                            child: Text(l.retry,
                                style: const TextStyle(fontSize: 12)),
                          ),
                        ],
                      ),
                    ),
                  // Category tabs
                  _buildCategoryTabs(context),
                  // Content area
                  Expanded(
                    child: isWide
                        ? Row(
                            children: [
                              _buildSubPageSidebar(context, l),
                              VerticalDivider(
                                width: 1,
                                thickness: 1,
                                color: Theme.of(context).dividerColor,
                              ),
                              Expanded(
                                child: _pageRegistry[_currentPageKey]!
                                    .builder(),
                              ),
                            ],
                          )
                        : _pageRegistry[_currentPageKey]!.builder(),
                  ),
                  // Save bar
                  _buildSaveBar(context, cfg, l),
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  Widget _buildCategoryTabs(BuildContext context) {
    const tint = JarvisTheme.sectionAdmin;

    return GlassPanel(
      tint: tint,
      borderRadius: 0,
      blur: 12,
      padding: EdgeInsets.zero,
      child: TabBar(
        controller: _tabController,
        isScrollable: true,
        tabAlignment: TabAlignment.start,
        indicatorColor: tint,
        indicatorWeight: 3,
        indicatorSize: TabBarIndicatorSize.tab,
        indicator: const UnderlineTabIndicator(
          borderSide: BorderSide(color: tint, width: 4),
          borderRadius: BorderRadius.vertical(top: Radius.circular(4)),
        ),
        labelColor: tint,
        unselectedLabelColor: JarvisTheme.textSecondary,
        labelStyle: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
        unselectedLabelStyle:
            const TextStyle(fontSize: 13, fontWeight: FontWeight.normal),
        overlayColor: WidgetStatePropertyAll(tint.withValues(alpha: 0.08)),
        dividerColor: Colors.transparent,
        padding: const EdgeInsets.symmetric(horizontal: 8),
        tabs: _categories.map((cat) {
          final l = AppLocalizations.of(context);
          final isActive = _categories.indexOf(cat) == _selectedCategory;
          return Tab(
            height: 48,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(cat.icon, size: 18),
                const SizedBox(width: 8),
                Text(cat.labelKey(l)),
                if (isActive) ...[
                  const SizedBox(width: 6),
                  Container(
                    width: 6,
                    height: 6,
                    decoration: BoxDecoration(
                      color: tint,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: tint.withValues(alpha: 0.8),
                          blurRadius: 10,
                        ),
                      ],
                    ),
                  ),
                ],
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildSubPageSidebar(BuildContext context, AppLocalizations l) {
    final keys = _currentPageKeys;
    const tint = JarvisTheme.sectionAdmin;

    return SizedBox(
      width: 220,
      child: ListView.builder(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 6),
        itemCount: keys.length,
        itemBuilder: (context, i) {
          final key = keys[i];
          final def = _pageRegistry[key]!;
          final selected = i == _selectedSubPage;
          final shortcutLabel = i < 9
              ? '^${i + 1}'
              : i == 9
                  ? '^0'
                  : null;

          return Padding(
            padding: const EdgeInsets.only(bottom: 2),
            child: GlassPanel(
              tint: selected ? tint : tint.withValues(alpha: 0.3),
              borderRadius: 8,
              blur: 8,
              padding: EdgeInsets.zero,
              glowOnHover: true,
              onTap: () => setState(() => _selectedSubPage = i),
              child: ListTile(
                dense: true,
                visualDensity: VisualDensity.compact,
                leading: Icon(def.icon,
                    size: 18,
                    color: selected ? tint : JarvisTheme.textSecondary),
                title: Text(
                  def.labelKey(l),
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight:
                        selected ? FontWeight.w600 : FontWeight.normal,
                    color: selected
                        ? tint
                        : Theme.of(context).textTheme.bodyMedium?.color,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                trailing: shortcutLabel != null
                    ? Text(
                        shortcutLabel,
                        style: TextStyle(
                          fontSize: 10,
                          color: JarvisTheme.textTertiary,
                          fontFamily: 'monospace',
                        ),
                      )
                    : null,
                selected: selected,
                selectedTileColor: tint.withValues(alpha: 0.08),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8)),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildSaveBar(
      BuildContext context, ConfigProvider cfg, AppLocalizations l) {
    if (!cfg.hasChanges && !cfg.saving) return const SizedBox.shrink();

    const tint = JarvisTheme.sectionAdmin;

    return _NeonPulseWrapper(
      active: cfg.hasChanges,
      color: tint,
      child: GlassPanel(
        tint: tint,
        borderRadius: 0,
        blur: 12,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          children: [
            Icon(Icons.edit, size: 16, color: JarvisTheme.orange),
            const SizedBox(width: 8),
            Text(l.unsavedChanges),
            const Spacer(),
            TextButton(
              onPressed: cfg.saving ? null : () => cfg.discard(),
              child: Text(l.discard),
            ),
            const SizedBox(width: 8),
            ElevatedButton.icon(
              onPressed: cfg.saving ? null : _save,
              icon: cfg.saving
                  ? const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.save, size: 16),
              label: Text(cfg.saving ? l.saving : l.saveCtrlS),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Neon pulse animation wrapper for the save bar ────────────────────────────

class _NeonPulseWrapper extends StatefulWidget {
  const _NeonPulseWrapper({
    required this.active,
    required this.color,
    required this.child,
  });

  final bool active;
  final Color color;
  final Widget child;

  @override
  State<_NeonPulseWrapper> createState() => _NeonPulseWrapperState();
}

class _NeonPulseWrapperState extends State<_NeonPulseWrapper>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  late final Animation<double> _opacity;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    );
    _opacity = Tween<double>(begin: 0.25, end: 0.55).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
    if (widget.active) _controller.repeat(reverse: true);
  }

  @override
  void didUpdateWidget(covariant _NeonPulseWrapper old) {
    super.didUpdateWidget(old);
    if (widget.active && !_controller.isAnimating) {
      _controller.repeat(reverse: true);
    } else if (!widget.active && _controller.isAnimating) {
      _controller.stop();
      _controller.value = 0;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.active) return widget.child;

    return AnimatedBuilder(
      animation: _opacity,
      builder: (context, child) => Container(
        decoration: BoxDecoration(
          boxShadow: [
            BoxShadow(
              color: widget.color.withValues(alpha: _opacity.value),
              blurRadius: 22,
              spreadRadius: -2,
            ),
          ],
        ),
        child: child,
      ),
      child: widget.child,
    );
  }
}
