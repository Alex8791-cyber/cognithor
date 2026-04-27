import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

/// Live-streaming log viewer that polls monitoring events.
class LiveLogsTab extends StatefulWidget {
  const LiveLogsTab({super.key});

  @override
  State<LiveLogsTab> createState() => _LiveLogsTabState();
}

class _LiveLogsTabState extends State<LiveLogsTab> {
  static const _maxEvents = 500;
  static const _pollInterval = Duration(seconds: 5);
  static const _initialFetchCount = 100;
  static const _pollFetchCount = 20;

  final List<Map<String, dynamic>> _events = [];
  final Set<String> _seenIds = {};
  final ScrollController _scrollController = ScrollController();

  Timer? _pollTimer;
  String _filter = 'ALL';
  bool _isAtBottom = true;
  int _newEventCount = 0;
  bool _initialLoading = true;

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onScroll);
    _fetchEvents(_initialFetchCount, initial: true);
    _pollTimer = Timer.periodic(
      _pollInterval,
      (_) => _fetchEvents(_pollFetchCount),
    );
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _scrollController.removeListener(_onScroll);
    _scrollController.dispose();
    super.dispose();
  }

  void _onScroll() {
    final atBottom =
        _scrollController.hasClients &&
        _scrollController.offset >=
            _scrollController.position.maxScrollExtent - 40;
    if (atBottom != _isAtBottom) {
      setState(() {
        _isAtBottom = atBottom;
        if (_isAtBottom) _newEventCount = 0;
      });
    }
  }

  Future<void> _fetchEvents(int count, {bool initial = false}) async {
    final conn = context.read<ConnectionProvider>();
    if (conn.state != CognithorConnectionState.connected) return;

    try {
      final result = await conn.api.getMonitoringEvents(n: count);
      final incoming = (result['events'] as List<dynamic>?) ?? [];

      if (!mounted) return;

      int added = 0;
      for (final raw in incoming) {
        final event = raw as Map<String, dynamic>;
        final id = _eventId(event);
        if (_seenIds.contains(id)) continue;
        _seenIds.add(id);
        _events.add(event);
        added++;
      }

      // Sort by timestamp ascending so newest is at the bottom.
      _events.sort(
        (a, b) => (a['timestamp']?.toString() ?? '').compareTo(
          b['timestamp']?.toString() ?? '',
        ),
      );

      // Trim oldest events if over cap.
      while (_events.length > _maxEvents) {
        final removed = _events.removeAt(0);
        _seenIds.remove(_eventId(removed));
      }

      if (added > 0 || initial) {
        setState(() {
          _initialLoading = false;
          if (!_isAtBottom && !initial) {
            _newEventCount += added;
          }
        });
        if (_isAtBottom) {
          _scrollToBottom();
        }
      }
    } catch (_) {
      if (initial && mounted) {
        setState(() => _initialLoading = false);
      }
    }
  }

  String _eventId(Map<String, dynamic> event) {
    if (event.containsKey('id') && event['id'] != null) {
      return event['id'].toString();
    }
    // Fallback: composite of timestamp + message.
    return '${event['timestamp'] ?? ''}_${event['message'] ?? ''}_${event['severity'] ?? ''}';
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _jumpToBottom() {
    setState(() {
      _newEventCount = 0;
      _isAtBottom = true;
    });
    _scrollToBottom();
  }

  void _clearLogs() {
    setState(() {
      _events.clear();
      _seenIds.clear();
      _newEventCount = 0;
    });
  }

  List<Map<String, dynamic>> get _filteredEvents {
    if (_filter == 'ALL') return _events;
    return _events.where((e) {
      final s = (e['severity']?.toString() ?? 'INFO').toUpperCase();
      return switch (_filter) {
        'INFO' => s == 'INFO',
        'WARNING' => s == 'WARNING' || s == 'WARN',
        'ERROR' => s == 'ERROR' || s == 'CRITICAL',
        _ => true,
      };
    }).toList();
  }

  Color _severityColor(String severity) {
    return switch (severity.toUpperCase()) {
      'ERROR' || 'CRITICAL' => CognithorTheme.red,
      'WARNING' || 'WARN' => CognithorTheme.orange,
      'INFO' => CognithorTheme.blue,
      _ => CognithorTheme.green,
    };
  }

  @override
  Widget build(BuildContext context) {
    if (_initialLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    final filtered = _filteredEvents;

    return Column(
      children: [
        // Filter bar
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
          child: Row(
            children: [
              _FilterChip(
                label: 'All',
                active: _filter == 'ALL',
                onTap: () => setState(() => _filter = 'ALL'),
              ),
              const SizedBox(width: 8),
              _FilterChip(
                label: 'Info',
                active: _filter == 'INFO',
                color: CognithorTheme.blue,
                onTap: () => setState(() => _filter = 'INFO'),
              ),
              const SizedBox(width: 8),
              _FilterChip(
                label: 'Warning',
                active: _filter == 'WARNING',
                color: CognithorTheme.orange,
                onTap: () => setState(() => _filter = 'WARNING'),
              ),
              const SizedBox(width: 8),
              _FilterChip(
                label: 'Error',
                active: _filter == 'ERROR',
                color: CognithorTheme.red,
                onTap: () => setState(() => _filter = 'ERROR'),
              ),
              const Spacer(),
              IconButton(
                icon: const Icon(Icons.delete_outline, size: 20),
                tooltip: 'Clear',
                onPressed: _clearLogs,
                style: IconButton.styleFrom(
                  foregroundColor: CognithorTheme.textSecondary,
                ),
              ),
            ],
          ),
        ),

        // Log entries
        Expanded(
          child: Stack(
            children: [
              filtered.isEmpty
                  ? Center(
                      child: Text(
                        'No log entries',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: CognithorTheme.textSecondary,
                        ),
                      ),
                    )
                  : ListView.builder(
                      controller: _scrollController,
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      itemCount: filtered.length,
                      itemBuilder: (context, index) {
                        final event = filtered[index];
                        final severity =
                            event['severity']?.toString() ?? 'INFO';
                        final message = event['message']?.toString() ?? '';
                        final name = event['name']?.toString() ?? '';
                        final timestamp = event['timestamp']?.toString() ?? '';

                        // Extract HH:MM:SS from timestamp.
                        final timeStr = _formatTime(timestamp);

                        return Padding(
                          padding: const EdgeInsets.only(bottom: 2),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              // Timestamp
                              SizedBox(
                                width: 72,
                                child: Text(
                                  timeStr,
                                  style: const TextStyle(
                                    fontFamily: 'monospace',
                                    fontSize: 12,
                                    color: Color(0xFF9CA3AF),
                                  ),
                                ),
                              ),
                              // Severity badge
                              Container(
                                width: 56,
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 6,
                                  vertical: 2,
                                ),
                                margin: const EdgeInsets.only(right: 8),
                                decoration: BoxDecoration(
                                  color: _severityColor(
                                    severity,
                                  ).withValues(alpha: 0.15),
                                  borderRadius: BorderRadius.circular(4),
                                ),
                                child: Text(
                                  severity.length > 5
                                      ? severity.substring(0, 5)
                                      : severity,
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                    fontSize: 10,
                                    fontWeight: FontWeight.w600,
                                    color: _severityColor(severity),
                                  ),
                                ),
                              ),
                              // Name + message
                              Expanded(
                                child: Text.rich(
                                  TextSpan(
                                    children: [
                                      if (name.isNotEmpty) ...[
                                        TextSpan(
                                          text: '$name  ',
                                          style: const TextStyle(
                                            fontSize: 12,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ],
                                      TextSpan(
                                        text: message,
                                        style: TextStyle(
                                          fontSize: 12,
                                          color: CognithorTheme.textSecondary,
                                        ),
                                      ),
                                    ],
                                  ),
                                  maxLines: 2,
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),

              // "New events" floating button
              if (_newEventCount > 0 && !_isAtBottom)
                Positioned(
                  bottom: 16,
                  left: 0,
                  right: 0,
                  child: Center(
                    child: FilledButton.icon(
                      onPressed: _jumpToBottom,
                      icon: const Icon(Icons.arrow_downward, size: 16),
                      label: Text('Neue Events ($_newEventCount)'),
                      style: FilledButton.styleFrom(
                        backgroundColor: CognithorTheme.accent,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ],
    );
  }

  String _formatTime(String timestamp) {
    if (timestamp.isEmpty) return '--:--:--';
    try {
      final dt = DateTime.parse(timestamp);
      return '${dt.hour.toString().padLeft(2, '0')}:'
          '${dt.minute.toString().padLeft(2, '0')}:'
          '${dt.second.toString().padLeft(2, '0')}';
    } catch (_) {
      // Try extracting HH:MM:SS from string.
      final match = RegExp(r'(\d{2}:\d{2}:\d{2})').firstMatch(timestamp);
      return match?.group(1) ?? timestamp;
    }
  }
}

class _FilterChip extends StatelessWidget {
  const _FilterChip({
    required this.label,
    required this.active,
    required this.onTap,
    this.color,
  });

  final String label;
  final bool active;
  final VoidCallback onTap;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final chipColor = color ?? CognithorTheme.accent;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: active ? chipColor.withValues(alpha: 0.2) : Colors.transparent,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: active
                ? chipColor
                : CognithorTheme.textSecondary.withValues(alpha: 0.3),
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: active ? FontWeight.w600 : FontWeight.normal,
            color: active ? chipColor : CognithorTheme.textSecondary,
          ),
        ),
      ),
    );
  }
}
