import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/research_provider.dart';
import 'package:cognithor_ui/widgets/research/hop_progress_indicator.dart';
import 'package:cognithor_ui/widgets/research/research_history_list.dart';
import 'package:cognithor_ui/widgets/research/research_report_view.dart';

class ResearchScreen extends StatefulWidget {
  const ResearchScreen({super.key});

  @override
  State<ResearchScreen> createState() => _ResearchScreenState();
}

class _ResearchScreenState extends State<ResearchScreen> {
  final _queryController = TextEditingController();
  bool _historyOpen = false;
  bool _initialized = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      _initialized = true;
      final conn = context.read<ConnectionProvider>();
      if (conn.state == CognithorConnectionState.connected) {
        final provider = context.read<ResearchProvider>();
        provider.setApi(conn.api);
        provider.loadHistory();
      }
    }
  }

  @override
  void dispose() {
    _queryController.dispose();
    super.dispose();
  }

  Future<void> _startResearch() async {
    final query = _queryController.text.trim();
    if (query.isEmpty) return;
    _queryController.clear();
    await context.read<ResearchProvider>().startResearch(query);
  }

  Future<void> _exportResult(String id, String format) async {
    final messenger = ScaffoldMessenger.of(context);
    final path = await context.read<ResearchProvider>().exportResearch(
      id,
      format,
    );
    if (!mounted) return;
    messenger.showSnackBar(
      SnackBar(
        content: Text(path != null ? 'Exported to: $path' : 'Export failed.'),
        duration: const Duration(seconds: 4),
      ),
    );
  }

  Future<void> _deleteResult(String id) async {
    final provider = context.read<ResearchProvider>();
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete research?'),
        content: const Text(
          'This will permanently remove this research report.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            style: FilledButton.styleFrom(
              backgroundColor: const Color(0xFFFF5252),
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await provider.deleteResearch(id);
    }
  }

  Widget _buildQueryBar(ResearchProvider provider) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        border: Border(
          bottom: BorderSide(color: theme.colorScheme.outlineVariant),
        ),
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _queryController,
              onSubmitted: (_) => _startResearch(),
              enabled: !provider.loading,
              decoration: InputDecoration(
                hintText: 'Enter a research question\u2026',
                prefixIcon: const Icon(Icons.search, size: 20),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide(
                    color: theme.colorScheme.outlineVariant,
                  ),
                ),
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 12,
                ),
              ),
            ),
          ),
          const SizedBox(width: 10),
          FilledButton.icon(
            onPressed: provider.loading ? null : _startResearch,
            icon: provider.loading
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.biotech, size: 18),
            label: Text(provider.loading ? 'Researching\u2026' : 'Research'),
            style: FilledButton.styleFrom(
              backgroundColor: const Color(0xFF00BCD4),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
            ),
          ),
          const SizedBox(width: 8),
          IconButton(
            icon: Icon(
              _historyOpen ? Icons.history_toggle_off : Icons.history,
              color: _historyOpen
                  ? const Color(0xFF00BCD4)
                  : theme.colorScheme.onSurfaceVariant,
            ),
            tooltip: 'Research history',
            onPressed: () => setState(() => _historyOpen = !_historyOpen),
          ),
        ],
      ),
    );
  }

  Widget _buildResultActions(ResearchResult result) {
    return Wrap(
      spacing: 8,
      children: [
        OutlinedButton.icon(
          onPressed: () => _exportResult(result.id, 'md'),
          icon: const Icon(Icons.description_outlined, size: 16),
          label: const Text('Export MD'),
          style: OutlinedButton.styleFrom(
            foregroundColor: const Color(0xFF00BCD4),
            side: const BorderSide(color: Color(0xFF00BCD4)),
          ),
        ),
        OutlinedButton.icon(
          onPressed: () => _exportResult(result.id, 'pdf'),
          icon: const Icon(Icons.picture_as_pdf_outlined, size: 16),
          label: const Text('Export PDF'),
          style: OutlinedButton.styleFrom(
            foregroundColor: const Color(0xFF00BCD4),
            side: const BorderSide(color: Color(0xFF00BCD4)),
          ),
        ),
        OutlinedButton.icon(
          onPressed: () => _deleteResult(result.id),
          icon: const Icon(Icons.delete_outline, size: 16),
          label: const Text('Delete'),
          style: OutlinedButton.styleFrom(
            foregroundColor: const Color(0xFFFF5252),
            side: const BorderSide(color: Color(0xFFFF5252)),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<ResearchProvider>();
    final theme = Theme.of(context);

    return Scaffold(
      body: Column(
        children: [
          _buildQueryBar(provider),
          Expanded(
            child: Row(
              children: [
                // History drawer
                if (_historyOpen)
                  Container(
                    width: 300,
                    decoration: BoxDecoration(
                      border: Border(
                        right: BorderSide(
                          color: theme.colorScheme.outlineVariant,
                        ),
                      ),
                    ),
                    child: Column(
                      children: [
                        Padding(
                          padding: const EdgeInsets.fromLTRB(12, 12, 8, 8),
                          child: Row(
                            children: [
                              const Icon(Icons.history, size: 16),
                              const SizedBox(width: 6),
                              Text(
                                'History',
                                style: theme.textTheme.titleSmall?.copyWith(
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const Spacer(),
                              IconButton(
                                icon: const Icon(Icons.close, size: 16),
                                onPressed: () =>
                                    setState(() => _historyOpen = false),
                              ),
                            ],
                          ),
                        ),
                        const Divider(height: 1),
                        Expanded(
                          child: ResearchHistoryList(
                            history: provider.history,
                            onSelect: (id) {
                              provider.loadResult(id);
                              setState(() => _historyOpen = false);
                            },
                            onDelete: (id) => provider.deleteResearch(id),
                          ),
                        ),
                      ],
                    ),
                  ),

                // Main content
                Expanded(child: _buildMainContent(provider, theme)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMainContent(ResearchProvider provider, ThemeData theme) {
    // Loading state
    if (provider.loading) {
      return const HopProgressIndicator();
    }

    // Error state
    if (provider.error != null && provider.activeResult == null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, size: 48, color: Color(0xFFFF5252)),
            const SizedBox(height: 12),
            Text(
              provider.error!,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: const Color(0xFFFF5252),
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    // Active result
    if (provider.activeResult != null) {
      final result = provider.activeResult!;
      return SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ResearchReportView(result: result),
            const SizedBox(height: 16),
            _buildResultActions(result),
          ],
        ),
      );
    }

    // Empty / welcome state
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.biotech,
            size: 64,
            color: const Color(0xFF00BCD4).withValues(alpha: 0.4),
          ),
          const SizedBox(height: 16),
          Text(
            'Deep Research Analyst',
            style: theme.textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
              color: const Color(0xFF00BCD4),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Enter a research question above to start a multi-hop\nweb research session with citations.',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 24),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            alignment: WrapAlignment.center,
            children:
                [
                      'What are the latest LLM benchmarks?',
                      'How does RAG compare to fine-tuning?',
                      'Best practices for Python async code?',
                    ]
                    .map(
                      (suggestion) => ActionChip(
                        label: Text(
                          suggestion,
                          style: const TextStyle(fontSize: 12),
                        ),
                        onPressed: () {
                          _queryController.text = suggestion;
                        },
                      ),
                    )
                    .toList(),
          ),
        ],
      ),
    );
  }
}
