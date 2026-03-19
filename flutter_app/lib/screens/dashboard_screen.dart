import 'dart:async';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/providers/pip_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/glass_panel.dart';
import 'package:jarvis_ui/widgets/jarvis_empty_state.dart';
import 'package:jarvis_ui/widgets/radial_gauge.dart';
import 'package:jarvis_ui/widgets/robot_office/robot_office_widget.dart';
import 'package:jarvis_ui/widgets/robot_office/glass_reflection_painter.dart';
import 'package:jarvis_ui/widgets/shimmer_loading.dart';

// ---------------------------------------------------------------------------
// Dashboard Screen — Command Center
// ---------------------------------------------------------------------------

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  Map<String, dynamic>? _dashboard;
  List<dynamic>? _events;
  Map<String, dynamic>? _models;
  Map<String, dynamic>? _status;
  bool _loading = true;
  String? _error;
  Timer? _refreshTimer;

  // Robot Office state
  String _robotCurrentTask = 'Warte auf Aufgabe...';
  int _robotTaskCount = 0;

  @override
  void initState() {
    super.initState();
    _loadData();
    _refreshTimer = Timer.periodic(
      const Duration(seconds: 15),
      (_) => _loadData(),
    );
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    super.dispose();
  }

  Future<void> _loadData() async {
    try {
      final api = context.read<ConnectionProvider>().api;
      final results = await Future.wait([
        api.getMonitoringDashboard(),
        api.getMonitoringEvents(n: 10),
        api.getModelStats(),
        api.getSystemStatus(),
      ]);

      if (!mounted) return;

      final dashboard = results[0];
      final eventsResult = results[1];
      final modelsResult = results[2];
      final statusResult = results[3];

      if (dashboard.containsKey('error')) {
        setState(() {
          _error = dashboard['error'] as String;
          _loading = false;
        });
        return;
      }

      setState(() {
        _dashboard = dashboard;
        _events = eventsResult['events'] as List<dynamic>? ?? [];
        _models = modelsResult;
        _status = statusResult;
        _loading = false;
        _error = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  double _toDouble(dynamic raw, [double fallback = 0]) {
    if (raw == null) return fallback;
    if (raw is num) return raw.toDouble();
    return double.tryParse(raw.toString()) ?? fallback;
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    if (_loading) {
      return const _DashboardLoadingState();
    }

    if (_error != null && _dashboard == null) {
      return _DashboardErrorState(
        error: _error!,
        onRetry: _loadData,
      );
    }

    final cpuValue = _toDouble(_dashboard?['cpu_usage']);
    final memValue = _toDouble(_dashboard?['memory_usage']);
    final rtValue = _toDouble(_dashboard?['response_time_ms']);
    final tokenValue = _toDouble(_dashboard?['total_tokens'] ?? _dashboard?['tool_executions']);

    // Normalize for gauge (0-1 range)
    final cpuNorm = (cpuValue / 100).clamp(0.0, 1.0);
    final memNorm = (memValue / 100).clamp(0.0, 1.0);
    final rtNorm = (rtValue / 2000).clamp(0.0, 1.0); // 2000ms = max
    final tokenNorm = (tokenValue / 10000).clamp(0.0, 1.0); // 10k = max

    // System load = average of CPU and memory
    final systemLoad = ((cpuNorm + memNorm) / 2).clamp(0.0, 1.0);

    return RefreshIndicator(
      onRefresh: _loadData,
      color: JarvisTheme.sectionDashboard,
      child: ListView(
        padding: const EdgeInsets.all(JarvisTheme.spacing),
        children: [
          // ── 1. Robot Office Hero (50% viewport height) ──────────
          Consumer<PipProvider>(
            builder: (context, pip, _) {
              if (pip.visible) {
                return _RobotOfficePipNotice(
                  onShowFullscreen: () => pip.exitFullscreen(),
                );
              }
              return SizedBox(
                height: MediaQuery.of(context).size.height * 0.5,
                child: GlassPanel(
                  tint: JarvisTheme.sectionDashboard,
                  padding: EdgeInsets.zero,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: Stack(
                      children: [
                        RobotOfficeWidget(
                          isRunning: true,
                          cpuUsage: cpuNorm,
                          memoryUsage: memNorm,
                          activePhase: _activePhaseFromStatus(),
                          systemLoad: systemLoad,
                          onStateChanged: (task, count) {
                            setState(() {
                              _robotCurrentTask = task;
                              _robotTaskCount = count;
                            });
                          },
                        ),
                        // Glass reflection overlay
                        Positioned.fill(
                          child: IgnorePointer(
                            child: CustomPaint(
                              painter: GlassReflectionPainter(),
                            ),
                          ),
                        ),
                        // Status overlay at bottom
                        Positioned(
                          bottom: 0,
                          left: 0,
                          right: 0,
                          child: _RobotStatusOverlay(
                            currentTask: _robotCurrentTask,
                            taskCount: _robotTaskCount,
                          ),
                        ),
                        // PiP mode button
                        Positioned(
                          top: 8,
                          right: 8,
                          child: _PipModeButton(
                            onTap: () => pip.show(),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              );
            },
          ),
          const SizedBox(height: JarvisTheme.spacingLg),

          // ── 2. Radial Gauge Row ─────────────────────────────────
          GlassPanel(
            tint: JarvisTheme.sectionDashboard,
            child: Wrap(
              spacing: JarvisTheme.spacing,
              runSpacing: JarvisTheme.spacing,
              alignment: WrapAlignment.spaceEvenly,
              children: [
                RadialGauge(
                  value: cpuNorm,
                  label: l.cpuUsage,
                  color: JarvisTheme.sectionDashboard,
                  valueText: '${cpuValue.round()}%',
                ),
                RadialGauge(
                  value: memNorm,
                  label: l.memoryUsage,
                  color: JarvisTheme.orange,
                  valueText: '${memValue.round()}%',
                ),
                RadialGauge(
                  value: tokenNorm,
                  label: l.toolExecutions,
                  color: JarvisTheme.accent,
                  valueText: '${tokenValue.round()}',
                ),
                RadialGauge(
                  value: rtNorm,
                  label: l.responseTime,
                  color: JarvisTheme.info,
                  valueText: '${rtValue.round()}ms',
                ),
              ],
            ),
          ),
          const SizedBox(height: JarvisTheme.spacingLg),

          // ── 3. Event Ticker ─────────────────────────────────────
          GlassPanel(
            tint: JarvisTheme.sectionDashboard,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            child: _EventTicker(events: _events),
          ),
        ],
      ),
    );
  }

  /// Maps system status to a pipeline phase index (0-4).
  int _activePhaseFromStatus() {
    final phase = _status?['phase']?.toString().toLowerCase() ?? '';
    return switch (phase) {
      'plan' || 'planning' => 0,
      'gate' || 'gatekeeper' => 1,
      'execute' || 'executing' => 2,
      'replan' || 'replanning' => 3,
      'complete' || 'done' => 4,
      _ => 0,
    };
  }
}

// ---------------------------------------------------------------------------
// Loading State
// ---------------------------------------------------------------------------

class _DashboardLoadingState extends StatelessWidget {
  const _DashboardLoadingState();

  @override
  Widget build(BuildContext context) {
    return const Padding(
      padding: EdgeInsets.all(JarvisTheme.spacing),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ShimmerLoading(count: 1, height: 200),
          SizedBox(height: JarvisTheme.spacingLg),
          ShimmerLoading(count: 1, height: 140),
          SizedBox(height: JarvisTheme.spacingLg),
          ShimmerLoading(count: 1, height: 50),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Error State
// ---------------------------------------------------------------------------

class _DashboardErrorState extends StatelessWidget {
  const _DashboardErrorState({
    required this.error,
    required this.onRetry,
  });

  final String error;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return JarvisEmptyState(
      icon: Icons.dashboard_outlined,
      title: l.noData,
      subtitle: error,
      action: ElevatedButton.icon(
        onPressed: onRetry,
        icon: const Icon(Icons.refresh),
        label: Text(l.retry),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Event Ticker — horizontal scrolling row of severity-colored chips
// ---------------------------------------------------------------------------

class _EventTicker extends StatelessWidget {
  const _EventTicker({required this.events});

  final List<dynamic>? events;

  Color _severityColor(String severity) {
    return switch (severity.toUpperCase()) {
      'ERROR' || 'CRITICAL' => JarvisTheme.red,
      'WARNING' || 'WARN' => JarvisTheme.orange,
      'INFO' => JarvisTheme.accent,
      _ => JarvisTheme.green,
    };
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final l = AppLocalizations.of(context);

    if (events == null || events!.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(JarvisTheme.spacingSm),
          child: Text(
            l.noEvents,
            style: theme.textTheme.bodySmall,
          ),
        ),
      );
    }

    return SizedBox(
      height: 36,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: events!.length,
        separatorBuilder: (_, _) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          final e = events![index] as Map<String, dynamic>;
          final severity = e['severity']?.toString() ?? 'INFO';
          final message = e['message']?.toString() ?? '';
          final color = _severityColor(severity);

          return Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(JarvisTheme.chipRadius),
              border: Border.all(
                color: color.withValues(alpha: 0.3),
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 6,
                  height: 6,
                  decoration: BoxDecoration(
                    color: color,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 6),
                ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 200),
                  child: Text(
                    message,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: color,
                      fontWeight: FontWeight.w500,
                    ),
                    overflow: TextOverflow.ellipsis,
                    maxLines: 1,
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Robot Status Overlay — glassmorphism bar at bottom of the office scene
// ---------------------------------------------------------------------------

class _RobotStatusOverlay extends StatelessWidget {
  const _RobotStatusOverlay({
    required this.currentTask,
    required this.taskCount,
  });

  final String currentTask;
  final int taskCount;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return ClipRRect(
      borderRadius: const BorderRadius.only(
        bottomLeft: Radius.circular(16),
        bottomRight: Radius.circular(16),
      ),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 12, sigmaY: 12),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            color: (isDark ? Colors.black : Colors.white)
                .withValues(alpha: isDark ? 0.45 : 0.55),
            border: Border(
              top: BorderSide(
                color: Colors.white.withValues(alpha: isDark ? 0.06 : 0.2),
              ),
            ),
          ),
          child: Row(
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: const BoxDecoration(
                  color: JarvisTheme.sectionDashboard,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  currentTask,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: isDark ? Colors.white70 : Colors.black87,
                    fontWeight: FontWeight.w500,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 12),
              Text(
                '$taskCount Tasks',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: JarvisTheme.sectionDashboard,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Robot Office PiP Notice — shown when PiP overlay is active
// ---------------------------------------------------------------------------

class _RobotOfficePipNotice extends StatelessWidget {
  const _RobotOfficePipNotice({required this.onShowFullscreen});

  final VoidCallback onShowFullscreen;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return GlassPanel(
      tint: JarvisTheme.sectionDashboard,
      padding: const EdgeInsets.symmetric(
        horizontal: JarvisTheme.spacing,
        vertical: 14,
      ),
      child: Row(
        children: [
          const Icon(
            Icons.picture_in_picture_alt,
            size: 20,
            color: JarvisTheme.sectionDashboard,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              'Robot Office is in Picture-in-Picture mode',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: isDark ? Colors.white70 : Colors.black87,
              ),
            ),
          ),
          TextButton.icon(
            onPressed: onShowFullscreen,
            icon: const Icon(Icons.fullscreen, size: 18),
            label: const Text('Fullscreen'),
            style: TextButton.styleFrom(
              foregroundColor: JarvisTheme.sectionDashboard,
              visualDensity: VisualDensity.compact,
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// PiP Mode Button — small button to switch from inline to PiP
// ---------------------------------------------------------------------------

class _PipModeButton extends StatelessWidget {
  const _PipModeButton({required this.onTap});

  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.black.withValues(alpha: 0.5),
      borderRadius: BorderRadius.circular(8),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(8),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.picture_in_picture_alt,
                size: 14,
                color: Colors.white.withValues(alpha: 0.8),
              ),
              const SizedBox(width: 4),
              Text(
                'PiP',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: Colors.white.withValues(alpha: 0.8),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
