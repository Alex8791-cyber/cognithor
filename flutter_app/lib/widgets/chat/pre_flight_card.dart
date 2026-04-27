import 'dart:async';
import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

/// Non-blocking plan preview card with auto-execute countdown.
class PreFlightCard extends StatefulWidget {
  const PreFlightCard({
    super.key,
    required this.goal,
    required this.steps,
    required this.timeoutSeconds,
    this.onCancel,
    this.onModify,
  });

  final String goal;
  final List<Map<String, dynamic>> steps;
  final int timeoutSeconds;
  final VoidCallback? onCancel;
  final void Function(String)? onModify;

  @override
  State<PreFlightCard> createState() => _PreFlightCardState();
}

class _PreFlightCardState extends State<PreFlightCard> {
  late int _remaining;
  Timer? _timer;
  bool _expanded = false;
  bool _executed = false;

  @override
  void initState() {
    super.initState();
    _remaining = widget.timeoutSeconds;
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (_remaining <= 1) {
        _timer?.cancel();
        setState(() => _executed = true);
      } else {
        setState(() => _remaining--);
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_executed) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          children: [
            Icon(Icons.play_arrow, size: 14, color: CognithorTheme.green),
            const SizedBox(width: 6),
            Text(
              'Plan gestartet: ${widget.goal}',
              style: TextStyle(
                fontSize: 12,
                color: CognithorTheme.textSecondary,
              ),
            ),
          ],
        ),
      );
    }

    final stepsSummary = widget.steps
        .map((s) => s['tool'] ?? '?')
        .join(' \u2192 ');

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 6),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: CognithorTheme.accent.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: CognithorTheme.accent.withValues(alpha: 0.25),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.route, size: 16, color: CognithorTheme.accent),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  '${widget.steps.length} Schritte: $stepsSummary',
                  style: TextStyle(
                    fontSize: 12,
                    color: CognithorTheme.textSecondary,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              Text(
                '${_remaining}s',
                style: TextStyle(
                  fontSize: 11,
                  color: CognithorTheme.accent,
                  fontFamily: 'monospace',
                ),
              ),
            ],
          ),
          if (_expanded) ...[
            const SizedBox(height: 6),
            ...widget.steps.map(
              (s) => Padding(
                padding: const EdgeInsets.only(left: 22, bottom: 2),
                child: Text(
                  '${s['tool']}: ${s['rationale'] ?? ''}',
                  style: TextStyle(
                    fontSize: 11,
                    color: CognithorTheme.textTertiary,
                  ),
                ),
              ),
            ),
          ],
          const SizedBox(height: 6),
          Row(
            children: [
              InkWell(
                onTap: () => setState(() => _expanded = !_expanded),
                child: Text(
                  _expanded ? 'Weniger' : 'Details',
                  style: TextStyle(fontSize: 11, color: CognithorTheme.accent),
                ),
              ),
              const Spacer(),
              TextButton(
                onPressed: () {
                  _timer?.cancel();
                  widget.onCancel?.call();
                },
                style: TextButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 2,
                  ),
                  minimumSize: Size.zero,
                ),
                child: Text(
                  'Abbrechen',
                  style: TextStyle(fontSize: 11, color: CognithorTheme.red),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
