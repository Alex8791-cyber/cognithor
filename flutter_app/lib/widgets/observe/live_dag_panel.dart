import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/widgets/dag_graph_painter.dart';

class LiveDagPanel extends StatelessWidget {
  const LiveDagPanel({super.key, required this.entries});

  final List<Map<String, dynamic>> entries;

  @override
  Widget build(BuildContext context) {
    if (entries.isEmpty) {
      return Center(child: Text(AppLocalizations.of(context).noDagData));
    }

    // Build nodes and edges from pipeline entries
    final nodes = <DagNode>[];
    final edges = <DagEdge>[];

    for (var i = 0; i < entries.length; i++) {
      final e = entries[i];
      final status = (e['status'] ?? '').toString().toLowerCase();
      nodes.add(
        DagNode(
          id: 'n$i',
          label: (e['phase'] ?? e['name'] ?? 'Step $i').toString(),
          status: status == 'running'
              ? DagNodeStatus.running
              : status == 'complete' || status == 'done'
              ? DagNodeStatus.complete
              : status == 'error'
              ? DagNodeStatus.error
              : DagNodeStatus.pending,
        ),
      );
      if (i > 0) {
        edges.add(DagEdge(from: 'n${i - 1}', to: 'n$i'));
      }
    }

    return Padding(
      padding: const EdgeInsets.all(8),
      child: CustomPaint(
        painter: DagGraphPainter(
          nodes: nodes,
          edges: edges,
          brightness: Theme.of(context).brightness,
        ),
        size: Size.infinite,
      ),
    );
  }
}
