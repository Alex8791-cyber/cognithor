import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/models/crew_trace.dart';
import 'package:cognithor_ui/providers/trace_provider.dart';
import 'package:cognithor_ui/screens/trace/widgets/event_row.dart';
import 'package:cognithor_ui/screens/trace/widgets/stats_sidebar.dart';

class TraceDetailScreen extends StatefulWidget {
  final String traceId;

  const TraceDetailScreen({super.key, required this.traceId});

  @override
  State<TraceDetailScreen> createState() => _TraceDetailScreenState();
}

class _TraceDetailScreenState extends State<TraceDetailScreen> {
  final _scrollController = ScrollController();
  late TraceProvider _provider;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _provider = context.read<TraceProvider>();
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _provider.pinTrace(widget.traceId);
    });
  }

  @override
  void dispose() {
    _provider.unpinTrace();
    _scrollController.dispose();
    super.dispose();
  }

  String? _traceStartedAt(List<CrewEvent> events) {
    for (final e in events) {
      if (e.eventType == 'crew_kickoff_started') return e.timestamp;
    }
    return null;
  }

  CrewTraceMeta? _findMeta(List<CrewTraceMeta> traces) {
    for (final t in traces) {
      if (t.traceId == widget.traceId) return t;
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<TraceProvider>();
    final events = provider.pinnedEvents;
    final stats = provider.pinnedStats;
    final meta = _findMeta(provider.traces);
    final startedAt = _traceStartedAt(events);

    // Auto-scroll to bottom on new event.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
      }
    });

    return Scaffold(
      backgroundColor: const Color(0xFF0A0F24),
      appBar: AppBar(
        title: Text(
          'Trace ${widget.traceId.length > 16 ? "${widget.traceId.substring(0, 16)}…" : widget.traceId}',
          style: const TextStyle(fontFamily: 'JetBrainsMono', fontSize: 14),
        ),
        backgroundColor: const Color(0xFF131A30),
      ),
      body: provider.errorMessage != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  provider.errorMessage!.contains('trace_not_found')
                      ? 'Trace ${widget.traceId} nicht gefunden — möglicherweise rotiert.'
                      : provider.errorMessage!,
                  style: const TextStyle(color: Color(0xFFFF5252)),
                  textAlign: TextAlign.center,
                ),
              ),
            )
          : Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Expanded(
                  child: events.isEmpty && provider.isLoading
                      ? const Center(child: CircularProgressIndicator())
                      : ListView.builder(
                          controller: _scrollController,
                          padding: const EdgeInsets.symmetric(horizontal: 16),
                          itemCount: events.length,
                          itemBuilder: (ctx, i) => EventRow(
                            event: events[i],
                            traceStartedAt: startedAt,
                          ),
                        ),
                ),
                Container(width: 1, color: const Color(0xFF1F2942)),
                StatsSidebar(stats: stats, meta: meta),
              ],
            ),
    );
  }
}
