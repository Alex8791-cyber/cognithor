import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class BudgetPage extends StatefulWidget {
  const BudgetPage({super.key});

  @override
  State<BudgetPage> createState() => _BudgetPageState();
}

class _BudgetPageState extends State<BudgetPage> {
  Map<String, dynamic>? _budgetData;
  Map<String, dynamic>? _resourceData;
  Map<String, dynamic>? _evolutionData;
  bool _loading = true;
  String? _error;
  Timer? _refreshTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadAll());
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    super.dispose();
  }

  Future<void> _loadAll() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final results = await Future.wait([
        api.get('budget/agents'),
        api.get('system/resources'),
        api.get('evolution/stats'),
      ]);
      if (!mounted) return;
      setState(() {
        _budgetData = results[0];
        _resourceData = results[1];
        _evolutionData = results[2];
        _loading = false;
      });
      _refreshTimer?.cancel();
      _refreshTimer = Timer.periodic(
        const Duration(seconds: 10),
        (_) => _refreshResources(),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = e.toString();
      });
    }
  }

  Future<void> _refreshResources() async {
    try {
      final api = context.read<ConnectionProvider>().api;
      final data = await api.get('system/resources');
      if (!mounted) return;
      setState(() {
        _resourceData = data;
      });
    } catch (_) {}
  }

  // ── Helpers ──────────────────────────────────────────────────────────────

  String _fmtCost(dynamic v) {
    final d = (v is num) ? v.toDouble() : 0.0;
    return '\$${d.toStringAsFixed(2)}';
  }

  double _sumCosts(Map<String, dynamic>? agents) {
    if (agents == null) return 0;
    return agents.values.fold<double>(
      0,
      (sum, v) => sum + ((v is num) ? v.toDouble() : 0),
    );
  }

  Color _usageColor(double pct) {
    if (pct > 80) return CognithorTheme.red;
    if (pct > 60) return CognithorTheme.orange;
    return CognithorTheme.green;
  }

  // ── Build ────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_error != null && _budgetData == null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.error_outline, size: 48, color: CognithorTheme.red),
            const SizedBox(height: 16),
            Text(
              _error!,
              style: TextStyle(color: CognithorTheme.textSecondary),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _loadAll,
              icon: const Icon(Icons.refresh, size: 16),
              label: Text(l.retry),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadAll,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildCostSummaryCard(),
          const SizedBox(height: 16),
          _buildAgentTable(),
          const SizedBox(height: 16),
          _buildResourceBars(),
          const SizedBox(height: 16),
          _buildEvolutionStatus(),
          const SizedBox(height: 16),
        ],
      ),
    );
  }

  // ── Cost Summary Card ────────────────────────────────────────────────────

  Widget _buildCostSummaryCard() {
    final today = _sumCosts(
      _budgetData?['agents_today'] as Map<String, dynamic>?,
    );
    final week = _sumCosts(
      _budgetData?['agents_week'] as Map<String, dynamic>?,
    );
    final month = _sumCosts(
      _budgetData?['agents_month'] as Map<String, dynamic>?,
    );

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.account_balance_wallet,
                  size: 20,
                  color: CognithorTheme.accent,
                ),
                const SizedBox(width: 8),
                const Text(
                  'Cost Overview',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                _costTile('Today', today, CognithorTheme.green),
                _costTile('This Week', week, CognithorTheme.orange),
                _costTile('This Month', month, CognithorTheme.accent),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _costTile(String label, double value, Color color) {
    return Expanded(
      child: Column(
        children: [
          Text(
            label,
            style: TextStyle(fontSize: 12, color: CognithorTheme.textSecondary),
          ),
          const SizedBox(height: 6),
          Text(
            _fmtCost(value),
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w700,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  // ── Agent Table ──────────────────────────────────────────────────────────

  Widget _buildAgentTable() {
    final agentsToday =
        (_budgetData?['agents_today'] as Map<String, dynamic>?) ?? {};
    final agentsWeek =
        (_budgetData?['agents_week'] as Map<String, dynamic>?) ?? {};
    final agentsMonth =
        (_budgetData?['agents_month'] as Map<String, dynamic>?) ?? {};
    final budgets = (_budgetData?['budgets'] as Map<String, dynamic>?) ?? {};

    // Collect all agent names
    final names = <String>{
      ...agentsToday.keys,
      ...agentsWeek.keys,
      ...agentsMonth.keys,
      ...budgets.keys,
    }.toList()..sort();

    if (names.isEmpty) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Center(
            child: Text(
              'No agent cost data yet',
              style: TextStyle(color: CognithorTheme.textSecondary),
            ),
          ),
        ),
      );
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: DataTable(
            columnSpacing: 24,
            headingTextStyle: TextStyle(
              fontWeight: FontWeight.w600,
              fontSize: 12,
              color: CognithorTheme.textSecondary,
            ),
            dataTextStyle: const TextStyle(fontSize: 13),
            columns: const [
              DataColumn(label: Text('Agent')),
              DataColumn(label: Text('Today'), numeric: true),
              DataColumn(label: Text('Week'), numeric: true),
              DataColumn(label: Text('Month'), numeric: true),
              DataColumn(label: Text('Limit'), numeric: true),
              DataColumn(label: Text('Status')),
            ],
            rows: names.map((name) {
              final b = budgets[name] as Map<String, dynamic>?;
              final limit = b?['daily_limit_usd'];
              final ok = b?['ok'] as bool? ?? true;
              final warn = b?['warning'] as bool? ?? false;

              Color statusColor;
              IconData statusIcon;
              if (!ok) {
                statusColor = CognithorTheme.red;
                statusIcon = Icons.error;
              } else if (warn) {
                statusColor = CognithorTheme.orange;
                statusIcon = Icons.warning;
              } else {
                statusColor = CognithorTheme.green;
                statusIcon = Icons.check_circle;
              }

              return DataRow(
                cells: [
                  DataCell(
                    Text(
                      name,
                      style: const TextStyle(fontWeight: FontWeight.w500),
                    ),
                  ),
                  DataCell(Text(_fmtCost(agentsToday[name]))),
                  DataCell(Text(_fmtCost(agentsWeek[name]))),
                  DataCell(Text(_fmtCost(agentsMonth[name]))),
                  DataCell(
                    Text(
                      limit != null ? _fmtCost(limit) : '--',
                      style: TextStyle(color: CognithorTheme.textSecondary),
                    ),
                  ),
                  DataCell(Icon(statusIcon, color: statusColor, size: 18)),
                ],
              );
            }).toList(),
          ),
        ),
      ),
    );
  }

  // ── Resource Bars ────────────────────────────────────────────────────────

  Widget _buildResourceBars() {
    final cpu = (_resourceData?['cpu_percent'] as num?)?.toDouble() ?? 0;
    final ramUsed = (_resourceData?['ram_used_gb'] as num?)?.toDouble() ?? 0;
    final ramTotal = (_resourceData?['ram_total_gb'] as num?)?.toDouble() ?? 1;
    final ramPct = (_resourceData?['ram_percent'] as num?)?.toDouble() ?? 0;
    final gpuUtil = (_resourceData?['gpu_util_percent'] as num?)?.toDouble();
    final gpuVramUsed = (_resourceData?['gpu_vram_used_gb'] as num?)
        ?.toDouble();
    final gpuVramTotal = (_resourceData?['gpu_vram_total_gb'] as num?)
        ?.toDouble();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.monitor_heart,
                  size: 20,
                  color: CognithorTheme.accent,
                ),
                const SizedBox(width: 8),
                const Text(
                  'System Resources',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _resourceBar(
              'CPU',
              cpu,
              '${cpu.toStringAsFixed(1)}%',
              Icons.developer_board,
            ),
            const SizedBox(height: 12),
            _resourceBar(
              'RAM',
              ramPct,
              '${ramUsed.toStringAsFixed(1)} / ${ramTotal.toStringAsFixed(1)} GB',
              Icons.memory,
            ),
            if (gpuUtil != null) ...[
              const SizedBox(height: 12),
              _resourceBar(
                'GPU',
                gpuUtil,
                gpuVramUsed != null && gpuVramTotal != null
                    ? '${gpuVramUsed.toStringAsFixed(1)} / ${gpuVramTotal.toStringAsFixed(1)} GB'
                    : '${gpuUtil.toStringAsFixed(1)}%',
                Icons.videogame_asset,
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _resourceBar(
    String label,
    double percent,
    String detail,
    IconData icon,
  ) {
    final clamped = percent.clamp(0.0, 100.0);
    final color = _usageColor(clamped);

    return Row(
      children: [
        Icon(icon, size: 18, color: CognithorTheme.textSecondary),
        const SizedBox(width: 8),
        SizedBox(
          width: 36,
          child: Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: CognithorTheme.textSecondary,
            ),
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: clamped / 100,
              minHeight: 10,
              backgroundColor: color.withValues(alpha: 0.15),
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
        ),
        const SizedBox(width: 12),
        SizedBox(
          width: 110,
          child: Text(
            detail,
            textAlign: TextAlign.right,
            style: TextStyle(
              fontSize: 12,
              color: color,
              fontWeight: FontWeight.w500,
              fontFamily: 'monospace',
            ),
          ),
        ),
      ],
    );
  }

  // ── Evolution Status ─────────────────────────────────────────────────────

  Widget _buildEvolutionStatus() {
    final running = _evolutionData?['running'] as bool? ?? false;
    final isIdle = _evolutionData?['is_idle'] as bool? ?? true;
    final cyclesToday = _evolutionData?['cycles_today'] as int? ?? 0;
    final totalCycles = _evolutionData?['total_cycles'] as int? ?? 0;
    final skillsCreated = _evolutionData?['total_skills_created'] as int? ?? 0;
    final recent = (_evolutionData?['recent_results'] as List<dynamic>?) ?? [];

    final statusColor = running
        ? (isIdle ? CognithorTheme.orange : CognithorTheme.green)
        : CognithorTheme.textSecondary;
    final statusLabel = running ? (isIdle ? 'Idle' : 'Running') : 'Stopped';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.auto_awesome,
                  size: 20,
                  color: CognithorTheme.accent,
                ),
                const SizedBox(width: 8),
                const Text(
                  'Evolution Engine',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: statusColor.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: statusColor.withValues(alpha: 0.4),
                    ),
                  ),
                  child: Text(
                    statusLabel,
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: statusColor,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                _statChip('Cycles Today', '$cyclesToday'),
                _statChip('Total Cycles', '$totalCycles'),
                _statChip('Skills Created', '$skillsCreated'),
              ],
            ),
            if (recent.isNotEmpty) ...[
              const SizedBox(height: 16),
              Text(
                'Recent Activity',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: CognithorTheme.textSecondary,
                ),
              ),
              const SizedBox(height: 8),
              ...recent.take(5).map((r) {
                final entry = r is Map<String, dynamic>
                    ? r
                    : <String, dynamic>{};
                final name = entry['skill_name'] ?? entry['name'] ?? '?';
                final success = entry['success'] as bool? ?? false;
                return Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Row(
                    children: [
                      Icon(
                        success ? Icons.check_circle : Icons.cancel,
                        size: 14,
                        color: success
                            ? CognithorTheme.green
                            : CognithorTheme.red,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          '$name',
                          style: const TextStyle(fontSize: 12),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                );
              }),
            ],
          ],
        ),
      ),
    );
  }

  Widget _statChip(String label, String value) {
    return Expanded(
      child: Column(
        children: [
          Text(
            value,
            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: TextStyle(fontSize: 11, color: CognithorTheme.textSecondary),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
