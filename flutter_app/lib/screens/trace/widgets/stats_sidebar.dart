import 'package:flutter/material.dart';

import 'package:cognithor_ui/models/crew_trace.dart';

/// Right-pane stats: Elapsed / Tokens / Guardrails / Per-Agent breakdown.
class StatsSidebar extends StatelessWidget {
  final CrewTraceStats? stats;
  final CrewTraceMeta? meta;

  const StatsSidebar({super.key, required this.stats, required this.meta});

  Widget _statBlock(String label, String value, {String? sub}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label.toUpperCase(),
            style: const TextStyle(
              fontSize: 10,
              color: Color(0xFF6B7A99),
              letterSpacing: 1.2,
              fontFamily: 'JetBrainsMono',
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: const TextStyle(
              fontSize: 18,
              color: Color(0xFFE8ECF4),
              fontWeight: FontWeight.w600,
              fontFamily: 'JetBrainsMono',
            ),
          ),
          if (sub != null) ...[
            const SizedBox(height: 2),
            Text(
              sub,
              style: const TextStyle(
                fontSize: 11,
                color: Color(0xFF8B5CF6),
                fontFamily: 'JetBrainsMono',
              ),
            ),
          ],
        ],
      ),
    );
  }

  String _formatElapsed(double? ms) {
    if (ms == null) return '—';
    final s = ms / 1000.0;
    final m = (s / 60).floor();
    final sr = s - (m * 60);
    return '${m.toString().padLeft(2, '0')}:${sr.toStringAsFixed(2).padLeft(5, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    final s = stats;
    final m = meta;
    return Container(
      width: 280,
      padding: const EdgeInsets.all(16),
      color: const Color(0xFF0E1426),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _statBlock(
            'Elapsed',
            _formatElapsed(s?.totalDurationMs ?? m?.durationMs),
            sub: m == null ? null : '${m.nTasks} tasks',
          ),
          _statBlock(
            'Token usage',
            (s?.totalTokens ?? m?.totalTokens ?? 0).toString(),
          ),
          _statBlock(
            'Guardrails',
            ((s?.guardrailPass ?? 0) + (s?.guardrailFail ?? 0)).toString(),
            sub: s == null
                ? null
                : '${s.guardrailPass} pass · ${s.guardrailFail} fail · ${s.guardrailRetries} retries',
          ),
          if (s != null && s.agentBreakdown.isNotEmpty) ...[
            const Text(
              'PER AGENT',
              style: TextStyle(
                fontSize: 10,
                color: Color(0xFF6B7A99),
                letterSpacing: 1.2,
                fontFamily: 'JetBrainsMono',
              ),
            ),
            const SizedBox(height: 4),
            ...s.agentBreakdown.entries.map(
              (e) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      e.key,
                      style: const TextStyle(
                        color: Color(0xFF8B5CF6),
                        fontFamily: 'JetBrainsMono',
                        fontSize: 11,
                      ),
                    ),
                    Text(
                      e.value.toString(),
                      style: const TextStyle(
                        color: Color(0xFFE8ECF4),
                        fontFamily: 'JetBrainsMono',
                        fontSize: 11,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
