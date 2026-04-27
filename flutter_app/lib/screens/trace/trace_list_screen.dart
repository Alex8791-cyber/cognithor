import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/trace_provider.dart';
import 'package:cognithor_ui/screens/trace/trace_detail_screen.dart';
import 'package:cognithor_ui/screens/trace/widgets/trace_card.dart';

class TraceListScreen extends StatefulWidget {
  const TraceListScreen({super.key});

  @override
  State<TraceListScreen> createState() => _TraceListScreenState();
}

class _TraceListScreenState extends State<TraceListScreen> {
  String? _statusFilter;
  late TraceProvider _provider;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _provider = context.read<TraceProvider>();
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      await _provider.loadTraces();
      _provider.subscribeToLifecycle();
    });
  }

  @override
  void dispose() {
    _provider.unsubscribeFromLifecycle();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<TraceProvider>();
    return Scaffold(
      backgroundColor: const Color(0xFF0A0F24),
      appBar: AppBar(
        title: const Text('Traces'),
        backgroundColor: const Color(0xFF131A30),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                _filterChip('All', null),
                const SizedBox(width: 8),
                _filterChip('Running', 'running'),
                const SizedBox(width: 8),
                _filterChip('Completed', 'completed'),
                const SizedBox(width: 8),
                _filterChip('Failed', 'failed'),
              ],
            ),
          ),
          if (provider.errorMessage != null)
            Padding(
              padding: const EdgeInsets.all(12),
              child: Text(
                provider.errorMessage!,
                style: const TextStyle(color: Color(0xFFFF5252)),
              ),
            ),
          Expanded(
            child: provider.isLoading
                ? const Center(child: CircularProgressIndicator())
                : provider.traces.isEmpty
                ? const Center(
                    child: Text(
                      'No traces yet',
                      style: TextStyle(color: Color(0xFF6B7A99)),
                    ),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    itemCount: provider.traces.length,
                    itemBuilder: (ctx, i) {
                      final meta = provider.traces[i];
                      return TraceCard(
                        meta: meta,
                        onTap: () {
                          Navigator.of(ctx).push(
                            MaterialPageRoute(
                              builder: (_) =>
                                  TraceDetailScreen(traceId: meta.traceId),
                            ),
                          );
                        },
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Widget _filterChip(String label, String? value) {
    final selected = _statusFilter == value;
    return ChoiceChip(
      label: Text(label),
      selected: selected,
      onSelected: (s) async {
        setState(() => _statusFilter = value);
        await context.read<TraceProvider>().loadTraces(status: value);
      },
    );
  }
}
