import 'package:flutter/material.dart';

import 'package:cognithor_ui/models/crew_trace.dart';

/// One row in the timeline-log. Layout per spec §8.4 / mockup:
///   [TS] [icon] [agent] [description] [badge]
class EventRow extends StatelessWidget {
  final CrewEvent event;
  final String? traceStartedAt;

  const EventRow({
    super.key,
    required this.event,
    required this.traceStartedAt,
  });

  String _relativeTimestamp() {
    if (event.timestamp == null || traceStartedAt == null) return '—';
    try {
      final t0 = DateTime.parse(traceStartedAt!.replaceAll('Z', '+00:00'));
      final t1 = DateTime.parse(event.timestamp!.replaceAll('Z', '+00:00'));
      final secs = t1.difference(t0).inMilliseconds / 1000.0;
      return '${secs.toStringAsFixed(2)}s';
    } catch (_) {
      return event.timestamp ?? '—';
    }
  }

  ({IconData icon, Color color, String label}) _styleFor(String type) {
    switch (type) {
      case 'crew_kickoff_started':
      case 'crew_task_started':
        return (
          icon: Icons.play_arrow,
          color: const Color(0xFF8B5CF6),
          label: 'start',
        );
      case 'crew_task_completed':
      case 'crew_kickoff_completed':
        return (
          icon: Icons.check,
          color: const Color(0xFF00E676),
          label: 'done',
        );
      case 'crew_task_failed':
      case 'crew_kickoff_failed':
        return (
          icon: Icons.error_outline,
          color: const Color(0xFFFF5252),
          label: 'fail',
        );
      case 'crew_guardrail_check':
        return (
          icon: Icons.shield,
          color: const Color(0xFFFFAB40),
          label: 'guard',
        );
      default:
        return (icon: Icons.circle, color: Colors.grey, label: type);
    }
  }

  Widget _badge(String text, {Color? color}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: const Color(0xFF131A30),
        border: Border.all(
          color: (color ?? const Color(0xFF2A3550)).withValues(alpha: 0.4),
        ),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Text(
        text,
        style: TextStyle(
          fontSize: 10,
          color: color ?? const Color(0xFF6B7A99),
          fontFamily: 'JetBrainsMono',
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final s = _styleFor(event.eventType);
    final agent = event.agentRole ?? 'crew';
    final tokens = event.tokens;
    final verdict = event.verdict;

    Widget? rightBadge;
    if (tokens != null) {
      rightBadge = _badge('$tokens tok');
    } else if (verdict != null) {
      rightBadge = _badge(
        verdict,
        color: verdict == 'pass'
            ? const Color(0xFF00E676)
            : const Color(0xFFFF5252),
      );
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(
            width: 80,
            child: Text(
              _relativeTimestamp(),
              style: const TextStyle(
                color: Color(0xFF6B7A99),
                fontFamily: 'JetBrainsMono',
                fontSize: 12,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Container(
            width: 18,
            height: 18,
            decoration: BoxDecoration(
              color: s.color.withValues(alpha: 0.2),
              shape: BoxShape.circle,
            ),
            child: Icon(s.icon, size: 12, color: s.color),
          ),
          const SizedBox(width: 12),
          SizedBox(
            width: 100,
            child: Text(
              agent,
              style: const TextStyle(
                color: Color(0xFF8B5CF6),
                fontFamily: 'JetBrainsMono',
                fontSize: 11,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              event.eventType,
              style: const TextStyle(
                color: Color(0xFFE8ECF4),
                fontFamily: 'JetBrainsMono',
                fontSize: 12,
              ),
            ),
          ),
          if (rightBadge != null) rightBadge,
        ],
      ),
    );
  }
}
