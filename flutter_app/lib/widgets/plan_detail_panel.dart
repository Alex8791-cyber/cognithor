import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/cognithor_chip.dart';
import 'package:cognithor_ui/widgets/cognithor_progress_bar.dart';

/// Shows plan details when plan_detail WS message arrives.
class PlanDetailPanel extends StatelessWidget {
  const PlanDetailPanel({super.key, required this.plan, required this.onClose});

  final Map<String, dynamic> plan;
  final VoidCallback onClose;

  @override
  Widget build(BuildContext context) {
    final goal = plan['goal']?.toString() ?? '';
    final reasoning = plan['reasoning']?.toString() ?? '';
    final confidence = (plan['confidence'] as num?)?.toDouble() ?? 0.0;
    final steps = (plan['steps'] as List?) ?? [];
    final iteration = plan['iteration'] ?? 1;
    final confColor = confidence > 0.7
        ? CognithorTheme.success
        : CognithorTheme.warning;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: CognithorTheme.accent.withAlpha(15),
        border: Border(
          top: BorderSide(color: CognithorTheme.accent.withAlpha(50)),
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.route, color: CognithorTheme.accent, size: 18),
              const SizedBox(width: 8),
              Text(
                'Plan #$iteration',
                style: const TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 13,
                ),
              ),
              const Spacer(),
              CognithorChip(
                label: '${(confidence * 100).toStringAsFixed(0)}%',
                color: confColor,
              ),
              const SizedBox(width: 8),
              GestureDetector(
                onTap: onClose,
                child: const Icon(Icons.expand_more, size: 20),
              ),
            ],
          ),
          if (goal.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              goal,
              style: TextStyle(
                color: CognithorTheme.textPrimary,
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
          if (reasoning.isNotEmpty) ...[
            const SizedBox(height: 4),
            Text(
              reasoning,
              style: TextStyle(
                color: CognithorTheme.textSecondary,
                fontSize: 12,
              ),
            ),
          ],
          if (steps.isNotEmpty) ...[
            const SizedBox(height: 8),
            ...steps.asMap().entries.map((e) {
              final step = e.value.toString();
              return Padding(
                padding: const EdgeInsets.only(bottom: 2),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '${e.key + 1}. ',
                      style: TextStyle(
                        color: CognithorTheme.accent,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    Expanded(
                      child: Text(
                        step,
                        style: TextStyle(
                          color: CognithorTheme.textPrimary,
                          fontSize: 12,
                        ),
                      ),
                    ),
                  ],
                ),
              );
            }),
          ],
          const SizedBox(height: 6),
          CognithorProgressBar(value: confidence, color: confColor, height: 4),
        ],
      ),
    );
  }
}
