import 'dart:math';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';


class KnowledgeGraphScreen extends StatefulWidget {
  const KnowledgeGraphScreen({super.key});

  @override
  State<KnowledgeGraphScreen> createState() => _KnowledgeGraphScreenState();
}

class _KnowledgeGraphScreenState extends State<KnowledgeGraphScreen> {
  List<Map<String, dynamic>> _entities = [];
  List<Map<String, dynamic>> _relations = [];
  bool _loading = true;
  String? _error;
  String _searchQuery = '';
  String _typeFilter = 'all';
  Map<String, dynamic>? _selectedEntity;
  List<Map<String, dynamic>> _selectedRelations = [];

  /// Pre-computed node positions shared between painter and hit-testing.
  Map<String, Offset> _nodePositions = {};

  /// The size used for the last layout computation.
  Size _lastLayoutSize = Size.zero;

  bool _initialized = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      _initialized = true;
      _loadGraph();
    }
  }

  Future<void> _loadGraph() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final result = await api.getMemoryGraphEntities();
      setState(() {
        _entities = (result['entities'] as List?)
                ?.map((e) => e as Map<String, dynamic>)
                .toList() ??
            [];
        _relations = (result['relations'] as List?)
                ?.map((e) => e as Map<String, dynamic>)
                .toList() ??
            [];
        _loading = false;
        _nodePositions = {};
        _lastLayoutSize = Size.zero;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  /// Compute force-directed layout positions for the current filtered entities.
  /// Results are cached until entities, relations, or canvas size change.
  void _computeLayout(Size size) {
    if (size == _lastLayoutSize && _nodePositions.isNotEmpty) return;

    final entities = _filteredEntities;
    if (entities.isEmpty) {
      _nodePositions = {};
      _lastLayoutSize = size;
      return;
    }

    final positions = <String, Offset>{};
    final rng = Random(42);

    // Initial circular placement
    for (var i = 0; i < entities.length; i++) {
      final id = (entities[i]['id'] ?? '$i').toString();
      final angle = 2 * pi * i / entities.length;
      final r = min(size.width, size.height) * 0.35;
      positions[id] = Offset(
        size.width / 2 + r * cos(angle) + (rng.nextDouble() - 0.5) * 20,
        size.height / 2 + r * sin(angle) + (rng.nextDouble() - 0.5) * 20,
      );
    }

    // Force-directed iterations
    for (var iter = 0; iter < 50; iter++) {
      final forces = <String, Offset>{};
      for (final e in entities) {
        forces[(e['id'] ?? '').toString()] = Offset.zero;
      }

      // Repulsion
      for (var i = 0; i < entities.length; i++) {
        for (var j = i + 1; j < entities.length; j++) {
          final idA = (entities[i]['id'] ?? '$i').toString();
          final idB = (entities[j]['id'] ?? '$j').toString();
          final delta = positions[idA]! - positions[idB]!;
          final dist = max(delta.distance, 1.0);
          final force = delta / dist * (5000 / (dist * dist));
          forces[idA] = forces[idA]! + force;
          forces[idB] = forces[idB]! - force;
        }
      }

      // Attraction (edges)
      for (final rel in _relations) {
        final src = (rel['source_id'] ?? rel['from'] ?? '').toString();
        final tgt = (rel['target_id'] ?? rel['to'] ?? '').toString();
        if (positions.containsKey(src) && positions.containsKey(tgt)) {
          final delta = positions[tgt]! - positions[src]!;
          final force = delta * 0.005;
          forces[src] = (forces[src] ?? Offset.zero) + force;
          forces[tgt] = (forces[tgt] ?? Offset.zero) - force;
        }
      }

      // Center gravity
      for (final e in entities) {
        final id = (e['id'] ?? '').toString();
        final center = Offset(size.width / 2, size.height / 2);
        forces[id] = (forces[id] ?? Offset.zero) +
            (center - positions[id]!) * 0.01;
      }

      // Apply with damping
      const damping = 0.9;
      for (final e in entities) {
        final id = (e['id'] ?? '').toString();
        positions[id] = positions[id]! + (forces[id] ?? Offset.zero) * damping;
        // Clamp to bounds
        positions[id] = Offset(
          positions[id]!.dx.clamp(20, size.width - 20),
          positions[id]!.dy.clamp(20, size.height - 20),
        );
      }
    }

    _nodePositions = positions;
    _lastLayoutSize = size;
  }

  /// Invalidate cached layout so it is recomputed on next paint.
  void _invalidateLayout() {
    _nodePositions = {};
    _lastLayoutSize = Size.zero;
  }

  Future<void> _selectEntity(Map<String, dynamic> entity) async {
    setState(() => _selectedEntity = entity);
    try {
      final api = context.read<ConnectionProvider>().api;
      final id = entity['id']?.toString() ?? '';
      final result = await api.getEntityRelations(id);
      setState(() {
        _selectedRelations = (result['relations'] as List?)
                ?.map((e) => e as Map<String, dynamic>)
                .toList() ??
            [];
      });
    } catch (_) {}
  }

  List<Map<String, dynamic>> get _filteredEntities {
    var list = _entities;
    if (_typeFilter != 'all') {
      list = list.where((e) => e['type'] == _typeFilter).toList();
    }
    if (_searchQuery.isNotEmpty) {
      final q = _searchQuery.toLowerCase();
      list = list
          .where((e) =>
              (e['name'] ?? e['label'] ?? '').toString().toLowerCase().contains(q))
          .toList();
    }
    return list;
  }

  static const _typeColors = JarvisTheme.entityColors;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: const Text('Knowledge Graph')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(_error!, style: TextStyle(color: JarvisTheme.red)),
                      const SizedBox(height: 12),
                      ElevatedButton(
                          onPressed: _loadGraph, child: const Text('Retry')),
                    ],
                  ),
                )
              : Row(
                  children: [
                    // Graph area
                    Expanded(
                      flex: 3,
                      child: Column(
                        children: [
                          // Toolbar
                          Padding(
                            padding: const EdgeInsets.all(12),
                            child: Row(
                              children: [
                                Expanded(
                                  child: TextField(
                                    decoration: const InputDecoration(
                                      hintText: 'Search entities...',
                                      prefixIcon: Icon(Icons.search),
                                      isDense: true,
                                      contentPadding: EdgeInsets.symmetric(
                                          horizontal: 12, vertical: 8),
                                    ),
                                    onChanged: (v) {
                                      setState(() => _searchQuery = v);
                                      _invalidateLayout();
                                    },
                                  ),
                                ),
                                const SizedBox(width: 8),
                                DropdownButton<String>(
                                  value: _typeFilter,
                                  items: [
                                    const DropdownMenuItem(
                                        value: 'all', child: Text('All Types')),
                                    ..._typeColors.keys.map((t) =>
                                        DropdownMenuItem(
                                            value: t,
                                            child: Text(t))),
                                  ],
                                  onChanged: (v) {
                                    setState(() => _typeFilter = v ?? 'all');
                                    _invalidateLayout();
                                  },
                                ),
                              ],
                            ),
                          ),
                          // Legend
                          Padding(
                            padding:
                                const EdgeInsets.symmetric(horizontal: 12),
                            child: Wrap(
                              spacing: 12,
                              children: _typeColors.entries
                                  .map((e) => Row(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          Container(
                                            width: 10,
                                            height: 10,
                                            decoration: BoxDecoration(
                                              color: e.value,
                                              shape: BoxShape.circle,
                                            ),
                                          ),
                                          const SizedBox(width: 4),
                                          Text(e.key,
                                              style: theme.textTheme.bodySmall),
                                        ],
                                      ))
                                  .toList(),
                            ),
                          ),
                          // Canvas
                          Expanded(
                            child: LayoutBuilder(
                              builder: (context, constraints) {
                                final canvasSize = Size(
                                  constraints.maxWidth,
                                  constraints.maxHeight,
                                );
                                _computeLayout(canvasSize);
                                return GestureDetector(
                                  onTapDown: (details) {
                                    _onTapGraph(details.localPosition);
                                  },
                                  child: CustomPaint(
                                    size: canvasSize,
                                    painter: _ForceGraphPainter(
                                      entities: _filteredEntities,
                                      relations: _relations,
                                      positions: _nodePositions,
                                      typeColors: _typeColors,
                                      brightness: theme.brightness,
                                    ),
                                  ),
                                );
                              },
                            ),
                          ),
                        ],
                      ),
                    ),
                    // Detail panel
                    if (_selectedEntity != null)
                      Container(
                        width: 280,
                        decoration: BoxDecoration(
                          color: theme.cardColor,
                          border: Border(
                              left: BorderSide(color: theme.dividerColor)),
                        ),
                        child: _buildDetailPanel(theme),
                      ),
                  ],
                ),
    );
  }

  void _onTapGraph(Offset position) {
    if (_nodePositions.isEmpty) return;

    final entities = _filteredEntities;
    Map<String, dynamic>? closest;
    var minDist = double.infinity;

    for (final entity in entities) {
      final id = (entity['id'] ?? '').toString();
      final nodePos = _nodePositions[id];
      if (nodePos == null) continue;

      final dist = (nodePos - position).distance;
      if (dist < 20 && dist < minDist) {
        minDist = dist;
        closest = entity;
      }
    }

    if (closest != null) _selectEntity(closest);
  }

  Widget _buildDetailPanel(ThemeData theme) {
    final e = _selectedEntity!;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                (e['name'] ?? e['label'] ?? 'Entity').toString(),
                style: theme.textTheme.titleLarge?.copyWith(fontSize: 16),
              ),
            ),
            IconButton(
              icon: const Icon(Icons.close, size: 18),
              onPressed: () => setState(() => _selectedEntity = null),
            ),
          ],
        ),
        const SizedBox(height: 8),
        _detailRow(theme, 'ID', (e['id'] ?? '-').toString()),
        _detailRow(theme, 'Type', (e['type'] ?? 'unknown').toString()),
        _detailRow(
            theme, 'Confidence', (e['confidence'] ?? '-').toString()),
        if (e['attributes'] is Map) ...[
          const SizedBox(height: 12),
          Text('Attributes', style: theme.textTheme.bodyMedium?.copyWith(
              fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          ...(e['attributes'] as Map).entries.map((kv) =>
              _detailRow(theme, kv.key.toString(), kv.value.toString())),
        ],
        if (_selectedRelations.isNotEmpty) ...[
          const SizedBox(height: 12),
          Text('Relations (${_selectedRelations.length})',
              style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          ..._selectedRelations.map((r) => Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: theme.scaffoldBackgroundColor,
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    '${r['type'] ?? 'relates'} -> ${r['target_name'] ?? r['target_id'] ?? '?'}',
                    style: theme.textTheme.bodySmall,
                  ),
                ),
              )),
        ],
      ],
    );
  }

  Widget _detailRow(ThemeData theme, String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
              width: 80,
              child: Text(label, style: theme.textTheme.bodySmall)),
          Expanded(child: Text(value, style: theme.textTheme.bodyMedium)),
        ],
      ),
    );
  }
}

class _ForceGraphPainter extends CustomPainter {
  _ForceGraphPainter({
    required this.entities,
    required this.relations,
    required this.positions,
    required this.typeColors,
    this.brightness = Brightness.dark,
  });

  final List<Map<String, dynamic>> entities;
  final List<Map<String, dynamic>> relations;
  final Map<String, Offset> positions;
  final Map<String, Color> typeColors;
  final Brightness brightness;

  @override
  void paint(Canvas canvas, Size size) {
    if (entities.isEmpty || positions.isEmpty) return;

    // Draw edges
    final edgePaint = Paint()
      ..color = (brightness == Brightness.dark ? JarvisTheme.textPrimary : JarvisTheme.textTertiary)
          .withValues(alpha: 0.1)
      ..strokeWidth = 1;

    for (final rel in relations) {
      final src = (rel['source_id'] ?? rel['from'] ?? '').toString();
      final tgt = (rel['target_id'] ?? rel['to'] ?? '').toString();
      if (positions.containsKey(src) && positions.containsKey(tgt)) {
        canvas.drawLine(positions[src]!, positions[tgt]!, edgePaint);
      }
    }

    // Draw nodes
    for (final e in entities) {
      final id = (e['id'] ?? '').toString();
      final pos = positions[id];
      if (pos == null) continue;
      final type = (e['type'] ?? 'unknown').toString();
      final color = typeColors[type] ?? JarvisTheme.entityColors['unknown']!;
      final name = (e['name'] ?? e['label'] ?? '').toString();

      // Node circle
      canvas.drawCircle(pos, 8, Paint()..color = color.withValues(alpha: 0.3));
      canvas.drawCircle(
          pos,
          8,
          Paint()
            ..color = color
            ..style = PaintingStyle.stroke
            ..strokeWidth = 2);

      // Label
      final tp = TextPainter(
        text: TextSpan(
          text: name.length > 12 ? '${name.substring(0, 12)}...' : name,
          style: TextStyle(
            color: brightness == Brightness.dark ? JarvisTheme.textSecondary : JarvisTheme.textTertiary,
            fontSize: 9,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: 80);
      tp.paint(canvas, Offset(pos.dx - tp.width / 2, pos.dy + 12));
    }
  }

  @override
  bool shouldRepaint(_ForceGraphPainter old) => true;
}
