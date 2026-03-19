import 'dart:async';
import 'dart:ui';

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/providers/pip_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/glass_card.dart';
import 'package:jarvis_ui/widgets/jarvis_card.dart';
import 'package:jarvis_ui/widgets/jarvis_empty_state.dart';
import 'package:jarvis_ui/widgets/animated_counter.dart';
import 'package:jarvis_ui/widgets/jarvis_section.dart';
import 'package:jarvis_ui/widgets/jarvis_status_badge.dart';
import 'package:jarvis_ui/widgets/robot_office/robot_office_widget.dart';
import 'package:jarvis_ui/widgets/shimmer_loading.dart';
import 'package:jarvis_ui/widgets/staggered_list.dart';

// ---------------------------------------------------------------------------
// Dashboard Screen
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

    return RefreshIndicator(
      onRefresh: _loadData,
      color: JarvisTheme.accent,
      child: ListView(
        padding: const EdgeInsets.all(JarvisTheme.spacing),
        children: [
          // ── Robot Office — hero visualization or PiP notice ────
          Consumer<PipProvider>(
            builder: (context, pip, _) {
              if (pip.visible) {
                // PiP is active — show a compact notice instead of the
                // full inline office.
                return _RobotOfficePipNotice(
                  onShowFullscreen: () => pip.exitFullscreen(),
                );
              }
              // PiP is hidden — show the Robot Office inline at full width.
              return SizedBox(
                height: 300,
                child: Stack(
                  children: [
                    RobotOfficeWidget(
                      isRunning: true,
                      onStateChanged: (task, count) {
                        setState(() {
                          _robotCurrentTask = task;
                          _robotTaskCount = count;
                        });
                      },
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
                    // Button to switch to PiP mode
                    Positioned(
                      top: 8,
                      right: 8,
                      child: _PipModeButton(
                        onTap: () => pip.show(),
                      ),
                    ),
                  ],
                ),
              );
            },
          ),
          const SizedBox(height: JarvisTheme.spacingSm),

          // ── Agent Cards ────────────────────────────────────────
          const _AgentCardRow(),
          const SizedBox(height: JarvisTheme.spacingLg),

          StaggeredList(
            children: [
              // -- System Status --
              JarvisSection(
                title: l.systemStatus,
                trailing: Text(
                  l.dashboardRefreshing,
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
              SystemStatusCard(
                status: _status,
                dashboard: _dashboard,
                backendVersion:
                    context.watch<ConnectionProvider>().backendVersion,
              ),
              const SizedBox(height: JarvisTheme.spacingLg),

              // -- Performance Metrics --
              JarvisSection(title: l.performance),
              PerformanceGrid(dashboard: _dashboard!),
              const SizedBox(height: JarvisTheme.spacingLg),

              // -- Model Info --
              JarvisSection(title: l.modelInfo),
              ModelInfoCard(models: _models),
              const SizedBox(height: JarvisTheme.spacingLg),

              // -- Recent Events --
              JarvisSection(
                title: l.recentEvents,
                trailing: TextButton(
                  onPressed: () {
                    // TODO: Navigate to full events view
                  },
                  child: Text(l.viewAll),
                ),
              ),
              RecentEventsCard(events: _events),
              const SizedBox(height: JarvisTheme.spacingLg),

              // -- Activity Chart --
              JarvisSection(title: l.activityChart),
              ActivityChart(dashboard: _dashboard),
            ],
          ),
        ],
      ),
    );
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
          ShimmerLoading(count: 1, height: 100),
          SizedBox(height: JarvisTheme.spacingLg),
          ShimmerLoading(count: 2, height: 80),
          SizedBox(height: JarvisTheme.spacingLg),
          ShimmerLoading(count: 1, height: 60),
          SizedBox(height: JarvisTheme.spacingLg),
          ShimmerLoading(count: 4, height: 40),
          SizedBox(height: JarvisTheme.spacingLg),
          ShimmerLoading(count: 1, height: 160),
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
// System Status Card
// ---------------------------------------------------------------------------

class SystemStatusCard extends StatelessWidget {
  const SystemStatusCard({
    super.key,
    required this.status,
    required this.dashboard,
    this.backendVersion,
  });

  final Map<String, dynamic>? status;
  final Map<String, dynamic>? dashboard;
  final String? backendVersion;

  String _formatUptime(dynamic raw) {
    if (raw == null) return '-';
    final seconds = raw is num ? raw.toInt() : int.tryParse(raw.toString());
    if (seconds == null) return raw.toString();

    final days = seconds ~/ 86400;
    final hours = (seconds % 86400) ~/ 3600;
    final minutes = (seconds % 3600) ~/ 60;

    final parts = <String>[];
    if (days > 0) parts.add('${days}d');
    if (hours > 0) parts.add('${hours}h');
    parts.add('${minutes}m');
    return parts.join(' ');
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    final isRunning = status?['status']?.toString().toLowerCase() == 'running' ||
        (status != null && !status!.containsKey('error'));
    final uptime = status?['uptime'] ?? dashboard?['uptime'];
    final version = backendVersion ??
        status?['version']?.toString() ??
        'Unknown';

    return JarvisCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              JarvisStatusBadge(
                label: isRunning ? l.running : l.stopped,
                color: isRunning ? JarvisTheme.green : JarvisTheme.red,
                icon: isRunning
                    ? Icons.check_circle_outline
                    : Icons.cancel_outlined,
              ),
              const Spacer(),
              Icon(
                Icons.circle,
                size: 10,
                color: isRunning ? JarvisTheme.green : JarvisTheme.red,
              ),
            ],
          ),
          const SizedBox(height: JarvisTheme.spacing),
          _infoRow(theme, l.uptime, _formatUptime(uptime)),
          const Divider(height: 20),
          _infoRow(theme, l.backendVersion, version),
        ],
      ),
    );
  }

  Widget _infoRow(ThemeData theme, String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: theme.textTheme.bodySmall),
        Flexible(
          child: Text(
            value,
            style: theme.textTheme.bodyMedium,
            textAlign: TextAlign.end,
            overflow: TextOverflow.ellipsis,
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Performance Grid
// ---------------------------------------------------------------------------

class PerformanceGrid extends StatelessWidget {
  const PerformanceGrid({
    super.key,
    required this.dashboard,
  });

  final Map<String, dynamic> dashboard;

  double _toDouble(dynamic raw, [double fallback = 0]) {
    if (raw == null) return fallback;
    if (raw is num) return raw.toDouble();
    return double.tryParse(raw.toString()) ?? fallback;
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    final cpuValue = _toDouble(dashboard['cpu_usage']);
    final memValue = _toDouble(dashboard['memory_usage']);
    final rtValue = _toDouble(dashboard['response_time_ms']);
    final toolValue = _toDouble(dashboard['tool_executions']);

    return LayoutBuilder(
      builder: (context, constraints) {
        final cardWidth =
            (constraints.maxWidth - JarvisTheme.spacingSm) / 2;
        return Wrap(
          spacing: JarvisTheme.spacingSm,
          runSpacing: JarvisTheme.spacingSm,
          children: [
            SizedBox(
              width: cardWidth,
              child: _AnimatedMetricCard(
                title: l.cpuUsage,
                numericValue: cpuValue,
                suffix: '%',
                icon: Icons.memory,
                color: JarvisTheme.accent,
              ),
            ),
            SizedBox(
              width: cardWidth,
              child: _AnimatedMetricCard(
                title: l.memoryUsage,
                numericValue: memValue,
                suffix: '%',
                icon: Icons.storage,
                color: JarvisTheme.orange,
              ),
            ),
            SizedBox(
              width: cardWidth,
              child: _AnimatedMetricCard(
                title: l.responseTime,
                numericValue: rtValue,
                suffix: 'ms',
                icon: Icons.speed,
                color: JarvisTheme.info,
              ),
            ),
            SizedBox(
              width: cardWidth,
              child: _AnimatedMetricCard(
                title: l.toolExecutions,
                numericValue: toolValue,
                icon: Icons.build,
                color: JarvisTheme.green,
              ),
            ),
          ],
        );
      },
    );
  }
}

// ---------------------------------------------------------------------------
// Animated Metric Card — wraps JarvisMetricCard layout with AnimatedCounter
// ---------------------------------------------------------------------------

class _AnimatedMetricCard extends StatelessWidget {
  const _AnimatedMetricCard({
    required this.title,
    required this.numericValue,
    this.suffix = '',
    this.icon,
    this.color,
  });

  final String title;
  final double numericValue;
  final String suffix;
  final IconData? icon;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final effectiveColor = color ?? JarvisTheme.accent;

    return JarvisCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              if (icon != null) ...[
                Icon(icon, size: JarvisTheme.iconSizeMd, color: effectiveColor),
                const SizedBox(width: JarvisTheme.spacingSm),
              ],
              Expanded(
                child: Text(title, style: theme.textTheme.bodySmall),
              ),
            ],
          ),
          const SizedBox(height: JarvisTheme.spacingSm),
          AnimatedCounter(
            value: numericValue,
            suffix: suffix,
            style: theme.textTheme.titleLarge?.copyWith(
              color: effectiveColor,
              fontSize: 28,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Model Info Card
// ---------------------------------------------------------------------------

class ModelInfoCard extends StatelessWidget {
  const ModelInfoCard({
    super.key,
    required this.models,
  });

  final Map<String, dynamic>? models;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    final planner = models?['planner']?.toString() ?? '-';
    final executor = models?['executor']?.toString() ?? '-';
    final coder = models?['coder']?.toString() ?? '-';

    return JarvisCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _modelRow(theme, l.plannerModel, planner, Icons.psychology,
              JarvisTheme.accent),
          const Divider(height: 20),
          _modelRow(theme, l.executorModel, executor, Icons.play_arrow,
              JarvisTheme.green),
          const Divider(height: 20),
          _modelRow(
              theme, l.coderModel, coder, Icons.code, JarvisTheme.purple),
        ],
      ),
    );
  }

  Widget _modelRow(
    ThemeData theme,
    String label,
    String value,
    IconData icon,
    Color color,
  ) {
    return Row(
      children: [
        Icon(icon, size: JarvisTheme.iconSizeMd, color: color),
        const SizedBox(width: JarvisTheme.spacingSm),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: theme.textTheme.bodySmall,
              ),
              const SizedBox(height: 2),
              Text(
                value,
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Recent Events Card
// ---------------------------------------------------------------------------

class RecentEventsCard extends StatelessWidget {
  const RecentEventsCard({
    super.key,
    required this.events,
  });

  final List<dynamic>? events;

  Color _severityColor(String severity) {
    return switch (severity.toUpperCase()) {
      'ERROR' || 'CRITICAL' => JarvisTheme.red,
      'WARNING' || 'WARN' => JarvisTheme.orange,
      'INFO' => JarvisTheme.accent,
      _ => JarvisTheme.green,
    };
  }

  IconData _severityIcon(String severity) {
    return switch (severity.toUpperCase()) {
      'ERROR' || 'CRITICAL' => Icons.error_outline,
      'WARNING' || 'WARN' => Icons.warning_amber_outlined,
      'INFO' => Icons.info_outline,
      _ => Icons.check_circle_outline,
    };
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    if (events == null || events!.isEmpty) {
      return JarvisCard(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(JarvisTheme.spacingLg),
            child: Text(
              l.noEvents,
              style: theme.textTheme.bodySmall,
            ),
          ),
        ),
      );
    }

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: events!.map<Widget>((event) {
        final e = event as Map<String, dynamic>;
        final severity = e['severity']?.toString() ?? 'INFO';
        final message = e['message']?.toString() ?? '';
        final timestamp = e['timestamp']?.toString() ?? '';

        return JarvisCard(
          padding: const EdgeInsets.symmetric(
            horizontal: 12,
            vertical: 10,
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(
                _severityIcon(severity),
                size: 20,
                color: _severityColor(severity),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        JarvisStatusBadge(
                          label: severity,
                          color: _severityColor(severity),
                        ),
                        const SizedBox(width: 8),
                        if (timestamp.isNotEmpty)
                          Expanded(
                            child: Text(
                              timestamp,
                              style: theme.textTheme.bodySmall,
                              textAlign: TextAlign.end,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      message,
                      style: theme.textTheme.bodyMedium,
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }
}

// ---------------------------------------------------------------------------
// Activity Chart
// ---------------------------------------------------------------------------

class ActivityChart extends StatelessWidget {
  const ActivityChart({
    super.key,
    required this.dashboard,
  });

  final Map<String, dynamic>? dashboard;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final theme = Theme.of(context);

    // Extract time-series data from dashboard; backend may provide
    // request_history as a list of {timestamp, count} maps.
    final rawHistory =
        (dashboard?['request_history'] as List<dynamic>?) ?? [];

    if (rawHistory.isEmpty) {
      return JarvisCard(
        child: SizedBox(
          height: 180,
          child: Center(
            child: Text(
              l.noData,
              style: theme.textTheme.bodySmall,
            ),
          ),
        ),
      );
    }

    final spots = <FlSpot>[];
    for (var i = 0; i < rawHistory.length; i++) {
      final entry = rawHistory[i];
      final y = entry is num
          ? entry.toDouble()
          : (entry is Map
              ? ((entry['count'] ?? entry['value'] ?? 0) as num).toDouble()
              : 0.0);
      spots.add(FlSpot(i.toDouble(), y));
    }

    return JarvisCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            l.requestsOverTime,
            style: theme.textTheme.bodySmall?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: JarvisTheme.spacing),
          SizedBox(
            height: 180,
            child: LineChart(
              LineChartData(
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: false,
                  horizontalInterval: 1,
                  getDrawingHorizontalLine: (_) => FlLine(
                    color: theme.dividerColor,
                    strokeWidth: 0.5,
                  ),
                ),
                titlesData: const FlTitlesData(
                  leftTitles: AxisTitles(
                    sideTitles:
                        SideTitles(showTitles: true, reservedSize: 32),
                  ),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                  topTitles: AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                  rightTitles: AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                ),
                borderData: FlBorderData(show: false),
                lineTouchData: LineTouchData(
                  touchTooltipData: LineTouchTooltipData(
                    getTooltipColor: (_) =>
                        theme.cardColor.withValues(alpha: 0.9),
                    getTooltipItems: (spots) => spots.map((spot) {
                      return LineTooltipItem(
                        spot.y.toStringAsFixed(0),
                        TextStyle(
                          color: JarvisTheme.accent,
                          fontWeight: FontWeight.w600,
                          fontSize: 13,
                        ),
                      );
                    }).toList(),
                  ),
                ),
                lineBarsData: [
                  LineChartBarData(
                    spots: spots,
                    isCurved: true,
                    color: JarvisTheme.accent,
                    barWidth: 2.5,
                    dotData: const FlDotData(show: false),
                    belowBarData: BarAreaData(
                      show: true,
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          JarvisTheme.accent.withValues(alpha: 0.2),
                          JarvisTheme.accent.withValues(alpha: 0.0),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
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
        bottomLeft: Radius.circular(12),
        bottomRight: Radius.circular(12),
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
                  color: Color(0xFF00e676),
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
                  color: JarvisTheme.accent,
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
// Agent Card Row
// ---------------------------------------------------------------------------

class _AgentCardRow extends StatelessWidget {
  const _AgentCardRow();

  @override
  Widget build(BuildContext context) {
    return const Row(
      children: [
        Expanded(
          child: _AgentCard(
            name: 'Planner',
            role: 'Strategie',
            color: Color(0xFF6366f1),
          ),
        ),
        SizedBox(width: JarvisTheme.spacingSm),
        Expanded(
          child: _AgentCard(
            name: 'Executor',
            role: 'Ausführung',
            color: Color(0xFF10b981),
          ),
        ),
        SizedBox(width: JarvisTheme.spacingSm),
        Expanded(
          child: _AgentCard(
            name: 'Researcher',
            role: 'Recherche',
            color: Color(0xFFf59e0b),
          ),
        ),
        SizedBox(width: JarvisTheme.spacingSm),
        Expanded(
          child: _AgentCard(
            name: 'Gatekeeper',
            role: 'Sicherheit',
            color: Color(0xFFef4444),
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Agent Card — small glassmorphism card with color accent
// ---------------------------------------------------------------------------

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

    return JarvisCard(
      padding: const EdgeInsets.symmetric(
        horizontal: JarvisTheme.spacing,
        vertical: 14,
      ),
      child: Row(
        children: [
          Icon(
            Icons.picture_in_picture_alt,
            size: 20,
            color: JarvisTheme.accent,
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
              foregroundColor: JarvisTheme.accent,
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

// ---------------------------------------------------------------------------
// Agent Card — small glassmorphism card with color accent
// ---------------------------------------------------------------------------

class _AgentCard extends StatelessWidget {
  const _AgentCard({
    required this.name,
    required this.role,
    required this.color,
  });

  final String name;
  final String role;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return GlassCard(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: color.withValues(alpha: 0.4),
                  blurRadius: 6,
                ),
              ],
            ),
          ),
          const SizedBox(height: 6),
          Text(
            name,
            style: theme.textTheme.bodySmall?.copyWith(
              fontWeight: FontWeight.w600,
              fontSize: 11,
            ),
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 2),
          Text(
            role,
            style: theme.textTheme.bodySmall?.copyWith(
              fontSize: 10,
              color: color.withValues(alpha: 0.8),
            ),
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}
