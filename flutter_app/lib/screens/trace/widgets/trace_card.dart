import 'package:flutter/material.dart';

import 'package:cognithor_ui/models/crew_trace.dart';

class TraceCard extends StatelessWidget {
  final CrewTraceMeta meta;
  final VoidCallback onTap;

  const TraceCard({super.key, required this.meta, required this.onTap});

  Color _statusColor() {
    switch (meta.status) {
      case TraceStatus.running:
        return const Color(0xFF00E676);
      case TraceStatus.completed:
        return const Color(0xFF8B5CF6);
      case TraceStatus.failed:
        return const Color(0xFFFF5252);
    }
  }

  String _statusLabel() {
    switch (meta.status) {
      case TraceStatus.running:
        return 'RUNNING';
      case TraceStatus.completed:
        return 'COMPLETED';
      case TraceStatus.failed:
        return 'FAILED';
    }
  }

  @override
  Widget build(BuildContext context) {
    final color = _statusColor();
    return Card(
      color: const Color(0xFF131A30),
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Container(
                width: 8,
                height: 40,
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(2),
                  boxShadow: meta.status == TraceStatus.running
                      ? [BoxShadow(color: color, blurRadius: 6)]
                      : null,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      meta.traceId.length > 24
                          ? '${meta.traceId.substring(0, 24)}…'
                          : meta.traceId,
                      style: const TextStyle(
                        color: Color(0xFFE8ECF4),
                        fontFamily: 'JetBrainsMono',
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${_statusLabel()} · ${meta.nTasks} tasks · ${meta.agentCount} agents',
                      style: TextStyle(
                        color: color,
                        fontSize: 11,
                        letterSpacing: 0.8,
                      ),
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    '${meta.totalTokens} tok',
                    style: const TextStyle(
                      color: Color(0xFFE8ECF4),
                      fontFamily: 'JetBrainsMono',
                      fontSize: 12,
                    ),
                  ),
                  if (meta.nFailedGuardrails > 0)
                    Padding(
                      padding: const EdgeInsets.only(top: 4),
                      child: Text(
                        '${meta.nFailedGuardrails} retry',
                        style: const TextStyle(
                          color: Color(0xFFFFAB40),
                          fontSize: 10,
                        ),
                      ),
                    ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
