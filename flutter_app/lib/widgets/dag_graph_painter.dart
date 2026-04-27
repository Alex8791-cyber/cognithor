import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

enum DagNodeStatus { pending, running, complete, error }

class DagNode {
  DagNode({
    required this.id,
    required this.label,
    this.status = DagNodeStatus.pending,
  });
  final String id;
  final String label;
  final DagNodeStatus status;
  double x = 0;
  double y = 0;
}

class DagEdge {
  const DagEdge({required this.from, required this.to});
  final String from;
  final String to;
}

/// Interactive DAG graph widget with pan, zoom, and node tap support.
class InteractiveDagGraph extends StatefulWidget {
  const InteractiveDagGraph({
    super.key,
    required this.nodes,
    required this.edges,
    this.onNodeTap,
    this.selectedNodeId,
    this.brightness = Brightness.dark,
  });

  final List<DagNode> nodes;
  final List<DagEdge> edges;
  final void Function(DagNode node)? onNodeTap;
  final String? selectedNodeId;
  final Brightness brightness;

  @override
  State<InteractiveDagGraph> createState() => _InteractiveDagGraphState();
}

class _InteractiveDagGraphState extends State<InteractiveDagGraph> {
  Offset _offset = Offset.zero;
  double _scale = 1.0;
  double _scaleStart = 1.0;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onScaleStart: (details) {
        _scaleStart = _scale;
      },
      onScaleUpdate: (details) {
        setState(() {
          _scale = (_scaleStart * details.scale).clamp(0.3, 3.0);
          _offset += details.focalPointDelta;
        });
      },
      onTapUp: (details) {
        final localPos = (details.localPosition - _offset) / _scale;
        for (final node in widget.nodes) {
          final dx = node.x - localPos.dx;
          final dy = node.y - localPos.dy;
          if (dx * dx + dy * dy < 900) {
            // 30px radius hit area
            widget.onNodeTap?.call(node);
            return;
          }
        }
      },
      child: ClipRect(
        child: CustomPaint(
          size: Size.infinite,
          painter: DagGraphPainter(
            nodes: widget.nodes,
            edges: widget.edges,
            brightness: widget.brightness,
            offset: _offset,
            scale: _scale,
            selectedNodeId: widget.selectedNodeId,
          ),
        ),
      ),
    );
  }
}

class DagGraphPainter extends CustomPainter {
  DagGraphPainter({
    required this.nodes,
    required this.edges,
    this.brightness = Brightness.dark,
    this.offset = Offset.zero,
    this.scale = 1.0,
    this.selectedNodeId,
  });

  final List<DagNode> nodes;
  final List<DagEdge> edges;
  final Brightness brightness;
  final Offset offset;
  final double scale;
  final String? selectedNodeId;

  @override
  void paint(Canvas canvas, Size size) {
    if (nodes.isEmpty) return;

    // Topological layout: Kahn's algorithm for layer assignment
    final layers = _assignLayers();
    _positionNodes(layers, size);

    canvas.save();
    canvas.translate(offset.dx, offset.dy);
    canvas.scale(scale);

    // Draw edges
    final nodeMap = {for (final n in nodes) n.id: n};
    final edgePaint = Paint()
      ..color = CognithorTheme.textSecondary.withValues(alpha: 0.5)
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    for (final edge in edges) {
      final from = nodeMap[edge.from];
      final to = nodeMap[edge.to];
      if (from == null || to == null) continue;

      final path = Path()
        ..moveTo(from.x, from.y + 15)
        ..cubicTo(from.x, from.y + 30, to.x, to.y - 30, to.x, to.y - 15);
      canvas.drawPath(path, edgePaint);

      // Arrow head
      _drawArrow(canvas, to.x, to.y - 15, edgePaint);
    }

    // Draw nodes
    for (final node in nodes) {
      _drawNode(canvas, node);
    }

    canvas.restore();
  }

  List<List<DagNode>> _assignLayers() {
    if (nodes.isEmpty) return [];

    final inDegree = <String, int>{};
    final adj = <String, List<String>>{};
    for (final n in nodes) {
      inDegree[n.id] = 0;
      adj[n.id] = [];
    }
    for (final e in edges) {
      adj[e.from]?.add(e.to);
      inDegree[e.to] = (inDegree[e.to] ?? 0) + 1;
    }

    final queue = <String>[
      for (final n in nodes)
        if (inDegree[n.id] == 0) n.id,
    ];
    final layers = <List<DagNode>>[];
    final nodeMap = {for (final n in nodes) n.id: n};

    while (queue.isNotEmpty) {
      final layer = <DagNode>[];
      final nextQueue = <String>[];
      for (final id in queue) {
        layer.add(nodeMap[id]!);
        for (final child in adj[id]!) {
          inDegree[child] = (inDegree[child] ?? 1) - 1;
          if (inDegree[child] == 0) nextQueue.add(child);
        }
      }
      layers.add(layer);
      queue
        ..clear()
        ..addAll(nextQueue);
    }

    return layers;
  }

  void _positionNodes(List<List<DagNode>> layers, Size size) {
    if (layers.isEmpty) return;
    final layerHeight = size.height / (layers.length + 1);

    for (var l = 0; l < layers.length; l++) {
      final layer = layers[l];
      final nodeWidth = size.width / (layer.length + 1);
      for (var n = 0; n < layer.length; n++) {
        layer[n].x = nodeWidth * (n + 1);
        layer[n].y = layerHeight * (l + 1);
      }
    }
  }

  void _drawNode(Canvas canvas, DagNode node) {
    final color = switch (node.status) {
      DagNodeStatus.running => CognithorTheme.accent,
      DagNodeStatus.complete => CognithorTheme.green,
      DagNodeStatus.error => CognithorTheme.red,
      DagNodeStatus.pending => CognithorTheme.textSecondary,
    };

    final isSelected = node.id == selectedNodeId;

    // Selection glow
    if (isSelected) {
      canvas.drawCircle(
        Offset(node.x, node.y),
        20,
        Paint()
          ..color = color.withValues(alpha: 0.3)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6),
      );
      canvas.drawCircle(
        Offset(node.x, node.y),
        17,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.5,
      );
    }

    // Circle fill
    canvas.drawCircle(
      Offset(node.x, node.y),
      14,
      Paint()..color = color.withValues(alpha: isSelected ? 0.35 : 0.2),
    );
    // Circle stroke
    canvas.drawCircle(
      Offset(node.x, node.y),
      14,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );

    // Pulsing effect for running nodes
    if (node.status == DagNodeStatus.running) {
      canvas.drawCircle(Offset(node.x, node.y), 6, Paint()..color = color);
    }

    // Label
    final tp = TextPainter(
      text: TextSpan(
        text: node.label,
        style: TextStyle(
          color: brightness == Brightness.dark
              ? CognithorTheme.textPrimary
              : const Color(0xFF1A1A2E),
          fontSize: 10,
          fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: 80);
    tp.paint(canvas, Offset(node.x - tp.width / 2, node.y + 18));
  }

  void _drawArrow(Canvas canvas, double x, double y, Paint paint) {
    final path = Path()
      ..moveTo(x - 4, y - 6)
      ..lineTo(x, y)
      ..lineTo(x + 4, y - 6);
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(DagGraphPainter old) => true;
}
