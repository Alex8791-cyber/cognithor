import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/security_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
import 'package:cognithor_ui/widgets/neon_glow.dart';
import 'package:cognithor_ui/widgets/cognithor_chip.dart';
import 'package:cognithor_ui/widgets/cognithor_empty_state.dart';
import 'package:cognithor_ui/widgets/cognithor_list_tile.dart';
import 'package:cognithor_ui/widgets/cognithor_progress_bar.dart';
import 'package:cognithor_ui/widgets/cognithor_section.dart';
import 'package:cognithor_ui/widgets/cognithor_stat.dart';
import 'package:cognithor_ui/widgets/cognithor_status_badge.dart';
import 'package:cognithor_ui/widgets/cognithor_tab_bar.dart';

class SecurityScreen extends StatefulWidget {
  const SecurityScreen({super.key});

  @override
  State<SecurityScreen> createState() => _SecurityScreenState();
}

class _SecurityScreenState extends State<SecurityScreen> {
  int _tabIndex = 0;
  String? _severityFilter;
  String? _actionFilter;
  bool _initialLoaded = false;
  bool _initialized = false;

  static int _toInt(dynamic v) {
    if (v is int) return v;
    if (v is num) return v.toInt();
    if (v is String) return int.tryParse(v) ?? 0;
    return 0;
  }

  static double _toDouble(dynamic v) {
    if (v is double) return v;
    if (v is num) return v.toDouble();
    if (v is String) return double.tryParse(v) ?? 0.0;
    return 0.0;
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      _initialized = true;
      context.read<SecurityProvider>().setApi(
        context.read<ConnectionProvider>().api,
      );
      _loadAll();
    }
  }

  Future<void> _loadAll() async {
    final sec = context.read<SecurityProvider>();
    // Load each independently - don't fail all if one endpoint is missing
    await Future.wait([
      sec.loadComplianceStats().catchError((_) {}),
      sec.loadRemediations().catchError((_) {}),
      sec.loadComplianceReport().catchError((_) {}),
      sec.loadRoles().catchError((_) {}),
      sec.loadAuthStats().catchError((_) {}),
      sec.loadRedteamStatus().catchError((_) {}),
      sec.loadAudit().catchError((_) {}),
    ]);
    if (mounted) setState(() => _initialLoaded = true);
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final sec = context.watch<SecurityProvider>();

    if (!_initialLoaded && sec.isLoading) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 16),
            Text(l.loading),
          ],
        ),
      );
    }

    if (sec.error != null && !_initialLoaded) {
      return CognithorEmptyState(
        icon: Icons.security,
        title: l.securityTitle,
        subtitle: sec.error,
        action: ElevatedButton.icon(
          onPressed: _loadAll,
          icon: const Icon(Icons.refresh),
          label: Text(l.retry),
        ),
      );
    }

    final tabs = [l.complianceTitle, l.rolesAccess, l.redTeam, l.auditLog];
    final icons = [
      Icons.verified_user,
      Icons.people,
      Icons.bug_report,
      Icons.list_alt,
    ];

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: CognithorTabBar(
            tabs: tabs,
            icons: icons,
            selectedIndex: _tabIndex,
            onChanged: (i) => setState(() => _tabIndex = i),
          ),
        ),
        Expanded(
          child: IndexedStack(
            index: _tabIndex,
            children: [
              _buildComplianceTab(l, sec),
              _buildRolesTab(l, sec),
              _buildRedTeamTab(l, sec),
              _buildAuditTab(l, sec),
            ],
          ),
        ),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Tab 1: Compliance
  // ---------------------------------------------------------------------------

  Widget _buildComplianceTab(AppLocalizations l, SecurityProvider sec) {
    final stats = sec.complianceStats ?? {};
    final totalDecisions = stats['total_decisions']?.toString() ?? '0';
    final flagged = stats['flagged']?.toString() ?? '0';
    final approvalRate = _toDouble(stats['approval_rate']);
    final avgConfidence = stats['avg_confidence']?.toString() ?? '-';

    final rems = sec.remediations ?? {};
    final open = _toInt(rems['open']);
    final inProgress = _toInt(rems['in_progress']);
    final resolved = _toInt(rems['resolved']);
    final overdue = _toInt(rems['overdue']);

    final report = sec.complianceReport ?? {};
    final euAiAct = report['eu_ai_act'] == true;
    final dsgvo = report['dsgvo'] == true;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Stats
        Wrap(
          spacing: 10,
          runSpacing: 10,
          children: [
            CognithorStat(
              label: l.decisionsTitle,
              value: totalDecisions,
              icon: Icons.gavel,
              color: CognithorTheme.accent,
            ),
            CognithorStat(
              label: l.flaggedCount,
              value: flagged,
              icon: Icons.flag,
              color: CognithorTheme.orange,
            ),
            CognithorStat(
              label: l.approvalRate,
              value: '${(approvalRate * 100).toStringAsFixed(1)}%',
              icon: Icons.check_circle,
              color: CognithorTheme.green,
            ),
            CognithorStat(
              label: l.confidence,
              value: avgConfidence,
              icon: Icons.speed,
              color: CognithorTheme.accent,
            ),
          ],
        ),
        const SizedBox(height: 16),

        // Approval rate bar
        NeonCard(
          tint: CognithorTheme.sectionAdmin,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(
                    Icons.bar_chart,
                    size: 18,
                    color: CognithorTheme.sectionAdmin,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    l.approvalRate,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ],
              ),
              const SizedBox(height: 8),
              CognithorProgressBar(
                value: approvalRate,
                label: '${(approvalRate * 100).toStringAsFixed(1)}%',
                color: CognithorTheme.green,
              ),
            ],
          ),
        ),

        // Remediations
        CognithorSection(title: l.remediations),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            CognithorStatusBadge(
              label: '${l.openStatus}: $open',
              color: CognithorTheme.accent,
              icon: Icons.pending,
            ),
            CognithorStatusBadge(
              label: '${l.inProgressStatus}: $inProgress',
              color: CognithorTheme.orange,
              icon: Icons.autorenew,
            ),
            CognithorStatusBadge(
              label: '${l.resolvedStatus}: $resolved',
              color: CognithorTheme.green,
              icon: Icons.check,
            ),
            CognithorStatusBadge(
              label: '${l.overdueStatus}: $overdue',
              color: CognithorTheme.red,
              icon: Icons.warning,
            ),
          ],
        ),
        const SizedBox(height: 16),

        // Compliance badges
        CognithorSection(title: l.complianceReport),
        Row(
          children: [
            CognithorStatusBadge(
              label: l.euAiAct,
              color: euAiAct ? CognithorTheme.green : CognithorTheme.red,
              icon: euAiAct ? Icons.check_circle : Icons.cancel,
            ),
            const SizedBox(width: 8),
            CognithorStatusBadge(
              label: l.dsgvo,
              color: dsgvo ? CognithorTheme.green : CognithorTheme.red,
              icon: dsgvo ? Icons.check_circle : Icons.cancel,
            ),
          ],
        ),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Tab 2: Roles & Access
  // ---------------------------------------------------------------------------

  Widget _buildRolesTab(AppLocalizations l, SecurityProvider sec) {
    final rolesData = sec.roles ?? {};
    final rolesList = rolesData['roles'] is List
        ? rolesData['roles'] as List
        : [];
    final authData = sec.authStats ?? {};

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Auth stats summary
        if (authData.isNotEmpty) ...[
          CognithorSection(title: l.statusLabel),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: [
              CognithorStat(
                label: l.activeSessions,
                value: authData['active_sessions']?.toString() ?? '0',
                icon: Icons.people,
                color: CognithorTheme.green,
              ),
              CognithorStat(
                label: l.total,
                value: authData['total_auth']?.toString() ?? '0',
                icon: Icons.key,
                color: CognithorTheme.accent,
              ),
            ],
          ),
          const SizedBox(height: 16),
        ],

        CognithorSection(title: l.rolesTitle),
        if (rolesList.isEmpty)
          CognithorEmptyState(icon: Icons.people_outline, title: l.noData),
        ...rolesList.whereType<Map<String, dynamic>>().map<Widget>((r) {
          final name = r['name']?.toString() ?? '';
          final perms = r['permissions'] is List
              ? r['permissions'] as List
              : [];
          return _RoleCard(name: name, permissions: perms);
        }),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Tab 3: Red Team
  // ---------------------------------------------------------------------------

  Widget _buildRedTeamTab(AppLocalizations l, SecurityProvider sec) {
    final status = sec.redteamStatus ?? {};
    final available = status['available'] == true;
    final lastScan = status['last_scan']?.toString();
    final rawResults = status['results'];
    final results = rawResults is Map<String, dynamic> ? rawResults : null;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        CognithorSection(title: l.scanStatus),
        NeonCard(
          tint: CognithorTheme.sectionAdmin,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  CognithorStatusBadge(
                    label: available ? l.enabled : l.scanNotAvailable,
                    color: available
                        ? CognithorTheme.green
                        : CognithorTheme.red,
                    icon: available ? Icons.check_circle : Icons.cancel,
                  ),
                ],
              ),
              if (lastScan != null) ...[
                const SizedBox(height: 8),
                Text(
                  '${l.lastScan}: $lastScan',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ],
          ),
        ),
        const SizedBox(height: 8),
        NeonGlow(
          color: CognithorTheme.sectionAdmin,
          intensity: 0.2,
          blurRadius: 8,
          child: ElevatedButton.icon(
            onPressed: available
                ? () => context.read<SecurityProvider>().runRedteamScan({
                    'scope': 'full',
                  })
                : null,
            icon: const Icon(Icons.play_arrow),
            label: Text(l.runScan),
          ),
        ),
        if (results != null) ...[
          const SizedBox(height: 16),
          CognithorSection(title: l.scanResults),
          NeonCard(
            tint: CognithorTheme.sectionAdmin,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: results.entries.map<Widget>((e) {
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Row(
                    children: [
                      SizedBox(
                        width: 140,
                        child: Text(
                          e.key,
                          style: Theme.of(context).textTheme.bodySmall
                              ?.copyWith(fontWeight: FontWeight.w500),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          e.value.toString(),
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                      ),
                    ],
                  ),
                );
              }).toList(),
            ),
          ),
        ],
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Tab 4: Audit
  // ---------------------------------------------------------------------------

  Widget _buildAuditTab(AppLocalizations l, SecurityProvider sec) {
    final entries = sec.auditEntries;

    return Column(
      children: [
        // Filter bar
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
          child: Row(
            children: [
              Expanded(
                child: DropdownButtonFormField<String>(
                  initialValue: _severityFilter,
                  decoration: InputDecoration(
                    labelText: l.severityLabel,
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                  ),
                  items: [
                    DropdownMenuItem(value: null, child: Text(l.allSeverities)),
                    DropdownMenuItem(
                      value: 'critical',
                      child: Text(l.critical),
                    ),
                    DropdownMenuItem(value: 'error', child: Text(l.errorLabel)),
                    DropdownMenuItem(
                      value: 'warning',
                      child: Text(l.warningLabel),
                    ),
                    DropdownMenuItem(value: 'info', child: Text(l.infoLabel)),
                  ],
                  onChanged: (v) {
                    setState(() => _severityFilter = v);
                    context.read<SecurityProvider>().loadAudit(
                      severity: _severityFilter,
                      action: _actionFilter,
                    );
                  },
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: TextField(
                  decoration: InputDecoration(
                    labelText: l.filter,
                    hintText: l.allActions,
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                  ),
                  onSubmitted: (v) {
                    setState(() => _actionFilter = v.isEmpty ? null : v);
                    context.read<SecurityProvider>().loadAudit(
                      severity: _severityFilter,
                      action: _actionFilter,
                    );
                  },
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),

        // Entries list
        Expanded(
          child: entries.isEmpty
              ? CognithorEmptyState(
                  icon: Icons.list_alt,
                  title: l.noAuditEntries,
                )
              : ListView.builder(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  itemCount: entries.length,
                  itemBuilder: (context, index) {
                    final entry = entries[index] is Map<String, dynamic>
                        ? entries[index] as Map<String, dynamic>
                        : <String, dynamic>{};
                    final action = entry['action']?.toString() ?? '';
                    final actor = entry['actor']?.toString() ?? '';
                    final ts = entry['timestamp']?.toString() ?? '';
                    final severity = entry['severity']?.toString() ?? 'info';

                    return CognithorListTile(
                      title: action,
                      subtitle: '${l.actor}: $actor  |  $ts',
                      leading: Icon(
                        _severityIcon(severity),
                        color: _severityColor(severity),
                        size: 20,
                      ),
                      trailing: CognithorStatusBadge(
                        label: severity,
                        color: _severityColor(severity),
                      ),
                    );
                  },
                ),
        ),
      ],
    );
  }

  Color _severityColor(String severity) {
    return switch (severity) {
      'critical' => CognithorTheme.red,
      'error' => CognithorTheme.red,
      'warning' => CognithorTheme.orange,
      'info' => CognithorTheme.info,
      _ => CognithorTheme.textSecondary,
    };
  }

  IconData _severityIcon(String severity) {
    return switch (severity) {
      'critical' => Icons.error,
      'error' => Icons.error_outline,
      'warning' => Icons.warning_amber,
      'info' => Icons.info_outline,
      _ => Icons.circle,
    };
  }
}

// ---------------------------------------------------------------------------
// Expandable role card
// ---------------------------------------------------------------------------

class _RoleCard extends StatefulWidget {
  const _RoleCard({required this.name, required this.permissions});

  final String name;
  final List<dynamic> permissions;

  @override
  State<_RoleCard> createState() => _RoleCardState();
}

class _RoleCardState extends State<_RoleCard> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: NeonCard(
        tint: CognithorTheme.sectionAdmin,
        glowOnHover: true,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(
                  Icons.shield,
                  size: 18,
                  color: CognithorTheme.sectionAdmin,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    widget.name,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
                CognithorChip(
                  label: '${widget.permissions.length} ${l.permissions}',
                  color: CognithorTheme.accent,
                ),
                IconButton(
                  icon: Icon(
                    _expanded ? Icons.expand_less : Icons.expand_more,
                    size: 20,
                  ),
                  onPressed: () => setState(() => _expanded = !_expanded),
                ),
              ],
            ),
            if (_expanded) ...[
              const SizedBox(height: 8),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: widget.permissions
                    .map<Widget>((p) => CognithorChip(label: p.toString()))
                    .toList(),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
