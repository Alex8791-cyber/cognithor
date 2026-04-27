import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/observe/agent_log_panel.dart';
import 'package:cognithor_ui/widgets/observe/kanban_panel.dart';
import 'package:cognithor_ui/widgets/observe/live_dag_panel.dart';
import 'package:cognithor_ui/widgets/plan_detail_panel.dart';

class ObservePanel extends StatefulWidget {
  const ObservePanel({
    super.key,
    required this.agentLog,
    required this.planDetail,
    required this.pipelineState,
    required this.onClose,
  });

  final List<Map<String, dynamic>> agentLog;
  final Map<String, dynamic>? planDetail;
  final List<Map<String, dynamic>> pipelineState;
  final VoidCallback onClose;

  @override
  State<ObservePanel> createState() => _ObservePanelState();
}

class _ObservePanelState extends State<ObservePanel>
    with SingleTickerProviderStateMixin {
  late final TabController _tabCtrl;

  @override
  void initState() {
    super.initState();
    _tabCtrl = TabController(length: 4, vsync: this);
  }

  @override
  void dispose() {
    _tabCtrl.dispose();
    super.dispose();
  }

  /// Derive the current pipeline status from the latest log entry and
  /// pipeline state.
  _PipelineStatus get _currentStatus {
    // Check pipeline state for running entries.
    for (final entry in widget.pipelineState) {
      final status = (entry['status'] ?? '').toString().toLowerCase();
      if (status == 'running' || status == 'active') {
        final phase = (entry['phase'] ?? '').toString();
        final tool = (entry['tool'] ?? '').toString();
        if (phase == 'execute' && tool.isNotEmpty) {
          return _PipelineStatus('Executing $tool...', CognithorTheme.green);
        }
        if (phase == 'plan') {
          return _PipelineStatus('Planning...', CognithorTheme.accent);
        }
        if (phase == 'gate') {
          return _PipelineStatus('Gatekeeper review...', CognithorTheme.orange);
        }
        if (phase == 'replan') {
          return _PipelineStatus('Re-planning...', CognithorTheme.warning);
        }
        return _PipelineStatus('Processing...', CognithorTheme.accent);
      }
    }

    // Check if everything is done.
    final hasDone = widget.pipelineState.any((e) {
      final s = (e['status'] ?? '').toString().toLowerCase();
      return s == 'complete' || s == 'done';
    });
    if (hasDone && widget.pipelineState.isNotEmpty) {
      return _PipelineStatus('Complete', CognithorTheme.green);
    }

    // Fallback: derive from latest log entry.
    if (widget.agentLog.isNotEmpty) {
      final last = widget.agentLog.last;
      final phase = (last['phase'] ?? '').toString();
      if (phase == 'plan') {
        return _PipelineStatus('Planning...', CognithorTheme.accent);
      }
      if (phase == 'execute') {
        final tool = (last['tool'] ?? '').toString();
        return _PipelineStatus(
          tool.isNotEmpty ? 'Executing $tool...' : 'Executing...',
          CognithorTheme.green,
        );
      }
      if (phase == 'gate') {
        return _PipelineStatus('Gatekeeper review...', CognithorTheme.orange);
      }
      if (phase == 'replan') {
        return _PipelineStatus('Re-planning...', CognithorTheme.warning);
      }
    }

    return _PipelineStatus('Idle', CognithorTheme.textSecondary);
  }

  @override
  Widget build(BuildContext context) {
    final status = _currentStatus;
    final totalKanban = widget.pipelineState.length;

    return Container(
      width: 360,
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        border: Border(left: BorderSide(color: Theme.of(context).dividerColor)),
      ),
      child: Column(
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Row(
              children: [
                Text(
                  AppLocalizations.of(context).observe,
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(fontSize: 16),
                ),
                const Spacer(),
                IconButton(
                  icon: const Icon(Icons.close, size: 18),
                  onPressed: widget.onClose,
                ),
              ],
            ),
          ),
          // Pipeline status indicator
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            margin: const EdgeInsets.symmetric(horizontal: 12),
            decoration: BoxDecoration(
              color: status.color.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(6),
              border: Border.all(color: status.color.withValues(alpha: 0.25)),
            ),
            child: Row(
              children: [
                SizedBox(
                  width: 12,
                  height: 12,
                  child: status.label == 'Idle' || status.label == 'Complete'
                      ? Icon(
                          status.label == 'Complete'
                              ? Icons.check_circle
                              : Icons.circle_outlined,
                          size: 12,
                          color: status.color,
                        )
                      : SizedBox(
                          width: 12,
                          height: 12,
                          child: CircularProgressIndicator(
                            strokeWidth: 1.5,
                            color: status.color,
                          ),
                        ),
                ),
                const SizedBox(width: 8),
                Text(
                  status.label,
                  style: TextStyle(
                    color: status.color,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 4),
          // Tabs
          TabBar(
            controller: _tabCtrl,
            isScrollable: true,
            tabAlignment: TabAlignment.start,
            labelColor: CognithorTheme.accent,
            indicatorColor: CognithorTheme.accent,
            tabs: [
              Tab(
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(AppLocalizations.of(context).log),
                    if (widget.agentLog.isNotEmpty) ...[
                      const SizedBox(width: 4),
                      _CountBadge(count: widget.agentLog.length),
                    ],
                  ],
                ),
              ),
              Tab(
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(AppLocalizations.of(context).kanban),
                    if (totalKanban > 0) ...[
                      const SizedBox(width: 4),
                      _CountBadge(count: totalKanban),
                    ],
                  ],
                ),
              ),
              Tab(text: AppLocalizations.of(context).dag),
              Tab(text: AppLocalizations.of(context).plan),
            ],
          ),
          Expanded(
            child: TabBarView(
              controller: _tabCtrl,
              children: [
                AgentLogPanel(entries: widget.agentLog),
                KanbanPanel(entries: widget.pipelineState),
                LiveDagPanel(entries: widget.pipelineState),
                widget.planDetail != null
                    ? PlanDetailPanel(plan: widget.planDetail!, onClose: () {})
                    : Center(
                        child: Text(AppLocalizations.of(context).noPlanData),
                      ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _PipelineStatus {
  const _PipelineStatus(this.label, this.color);
  final String label;
  final Color color;
}

class _CountBadge extends StatelessWidget {
  const _CountBadge({required this.count});
  final int count;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
      decoration: BoxDecoration(
        color: CognithorTheme.accent.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        '$count',
        style: TextStyle(
          color: CognithorTheme.accent,
          fontSize: 10,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}
