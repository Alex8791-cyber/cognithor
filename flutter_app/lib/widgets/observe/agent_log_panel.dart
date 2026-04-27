import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class AgentLogPanel extends StatefulWidget {
  const AgentLogPanel({super.key, required this.entries});

  final List<Map<String, dynamic>> entries;

  @override
  State<AgentLogPanel> createState() => _AgentLogPanelState();
}

class _AgentLogPanelState extends State<AgentLogPanel> {
  final _scrollCtrl = ScrollController();

  @override
  void didUpdateWidget(AgentLogPanel old) {
    super.didUpdateWidget(old);
    if (widget.entries.length > old.entries.length) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scrollCtrl.hasClients) {
          _scrollCtrl.animateTo(
            _scrollCtrl.position.maxScrollExtent,
            duration: const Duration(milliseconds: 200),
            curve: Curves.easeOut,
          );
        }
      });
    }
  }

  @override
  void dispose() {
    _scrollCtrl.dispose();
    super.dispose();
  }

  Color _phaseColor(String phase) {
    return switch (phase) {
      'plan' => CognithorTheme.accent,
      'gate' => CognithorTheme.orange,
      'execute' => CognithorTheme.green,
      'replan' => CognithorTheme.warning,
      _ => CognithorTheme.textSecondary,
    };
  }

  IconData _phaseIcon(String phase) {
    return switch (phase) {
      'plan' => Icons.psychology, // brain
      'gate' => Icons.shield, // shield
      'execute' => Icons.play_arrow, // play
      'replan' => Icons.refresh, // refresh
      _ => Icons.info_outline,
    };
  }

  /// Format elapsed milliseconds into a human-readable string.
  String? _formatElapsed(Map<String, dynamic> entry) {
    final raw = entry['elapsed_ms'] ?? entry['elapsed'] ?? entry['duration_ms'];
    if (raw == null) return null;
    final ms = raw is num ? raw.toInt() : int.tryParse(raw.toString());
    if (ms == null) return null;
    if (ms < 1000) return '${ms}ms';
    final secs = (ms / 1000).toStringAsFixed(1);
    return '${secs}s';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    if (widget.entries.isEmpty) {
      return Center(child: Text(AppLocalizations.of(context).noLogEntries));
    }

    return ListView.builder(
      controller: _scrollCtrl,
      padding: const EdgeInsets.all(8),
      itemCount: widget.entries.length,
      itemBuilder: (context, i) {
        final e = widget.entries[i];
        final phase = (e['phase'] ?? '').toString();
        final tool = (e['tool'] ?? '').toString();
        final message = (e['message'] ?? e['text'] ?? '').toString();
        final ts = (e['timestamp'] ?? '').toString();
        final elapsed = _formatElapsed(e);

        return Padding(
          padding: const EdgeInsets.only(bottom: 6),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Phase icon
              Icon(_phaseIcon(phase), size: 14, color: _phaseColor(phase)),
              const SizedBox(width: 4),
              if (ts.isNotEmpty)
                Text(
                  ts.length > 8 ? ts.substring(ts.length - 8) : ts,
                  style: theme.textTheme.bodySmall?.copyWith(
                    fontFamily: 'monospace',
                    fontSize: 10,
                  ),
                ),
              const SizedBox(width: 6),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                decoration: BoxDecoration(
                  color: _phaseColor(phase).withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(3),
                ),
                child: Text(
                  phase.isNotEmpty ? phase : 'info',
                  style: TextStyle(
                    color: _phaseColor(phase),
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              if (tool.isNotEmpty) ...[
                const SizedBox(width: 4),
                Text(
                  tool,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: CognithorTheme.accent,
                    fontSize: 11,
                  ),
                ),
              ],
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  message,
                  style: theme.textTheme.bodySmall?.copyWith(fontSize: 11),
                  maxLines: 3,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              // Elapsed time badge
              if (elapsed != null) ...[
                const SizedBox(width: 4),
                Text(
                  elapsed,
                  style: theme.textTheme.bodySmall?.copyWith(
                    fontSize: 9,
                    color: CognithorTheme.textSecondary,
                    fontFamily: 'monospace',
                  ),
                ),
              ],
            ],
          ),
        );
      },
    );
  }
}
