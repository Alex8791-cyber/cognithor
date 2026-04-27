import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';

class MemoryPage extends StatelessWidget {
  const MemoryPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final mem = cfg.cfg['memory'] as Map<String, dynamic>? ?? {};
        final wVector = (mem['weight_vector'] as num?)?.toDouble() ?? 0.5;
        final wBm25 = (mem['weight_bm25'] as num?)?.toDouble() ?? 0.3;
        final wGraph = (mem['weight_graph'] as num?)?.toDouble() ?? 0.2;
        final sum = wVector + wBm25 + wGraph;
        final sumColor = (sum - 1.0).abs() < 0.02
            ? CognithorTheme.green
            : sum > 1.01
            ? CognithorTheme.red
            : CognithorTheme.orange;

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            CognithorNumberField(
              label: 'Chunk Size (tokens)',
              value: (mem['chunk_size_tokens'] as num?) ?? 400,
              onChanged: (v) => cfg.set('memory.chunk_size_tokens', v),
              min: 64,
            ),
            CognithorNumberField(
              label: 'Chunk Overlap (tokens)',
              value: (mem['chunk_overlap_tokens'] as num?) ?? 80,
              onChanged: (v) => cfg.set('memory.chunk_overlap_tokens', v),
              min: 0,
            ),
            CognithorNumberField(
              label: 'Search Top K',
              value: (mem['search_top_k'] as num?) ?? 6,
              onChanged: (v) => cfg.set('memory.search_top_k', v),
              min: 1,
              max: 50,
            ),
            const Divider(height: 32),
            Row(
              children: [
                Text(
                  AppLocalizations.of(context).searchWeights,
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(fontSize: 16),
                ),
                const Spacer(),
                Text(
                  'Sum: ${sum.toStringAsFixed(2)}',
                  style: TextStyle(
                    color: sumColor,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            CognithorSliderField(
              label: 'Vector Weight',
              value: wVector,
              onChanged: (v) => cfg.set('memory.weight_vector', v),
            ),
            CognithorSliderField(
              label: 'BM25 Weight',
              value: wBm25,
              onChanged: (v) => cfg.set('memory.weight_bm25', v),
            ),
            CognithorSliderField(
              label: 'Graph Weight',
              value: wGraph,
              onChanged: (v) => cfg.set('memory.weight_graph', v),
            ),
            const Divider(height: 32),
            CognithorNumberField(
              label: 'Recency Half-Life (days)',
              value: (mem['recency_half_life_days'] as num?) ?? 30,
              onChanged: (v) => cfg.set('memory.recency_half_life_days', v),
              min: 1,
            ),
            CognithorSliderField(
              label: 'Compaction Threshold',
              value: (mem['compaction_threshold'] as num?)?.toDouble() ?? 0.8,
              onChanged: (v) => cfg.set('memory.compaction_threshold', v),
              min: 0.5,
              max: 0.95,
              step: 0.05,
            ),
            CognithorNumberField(
              label: 'Compaction Keep Last N',
              value: (mem['compaction_keep_last_n'] as num?) ?? 8,
              onChanged: (v) => cfg.set('memory.compaction_keep_last_n', v),
              min: 2,
            ),
            CognithorNumberField(
              label: 'Episodic Retention (days)',
              value: (mem['episodic_retention_days'] as num?) ?? 365,
              onChanged: (v) => cfg.set('memory.episodic_retention_days', v),
              min: 1,
            ),
            CognithorToggleField(
              label: 'Dynamic Weighting',
              value: mem['dynamic_weighting'] == true,
              onChanged: (v) => cfg.set('memory.dynamic_weighting', v),
              description: 'Automatically adjust weights based on query type',
            ),
          ],
        );
      },
    );
  }
}
