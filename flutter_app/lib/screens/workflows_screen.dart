import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/providers/workflow_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/dag_graph_painter.dart';
import 'package:cognithor_ui/widgets/dag_node_detail.dart';
import 'package:cognithor_ui/widgets/neon_card.dart';
import 'package:cognithor_ui/widgets/neon_glow.dart';
import 'package:cognithor_ui/widgets/cognithor_chip.dart';
import 'package:cognithor_ui/widgets/cognithor_empty_state.dart';
import 'package:cognithor_ui/widgets/cognithor_loading_skeleton.dart';
import 'package:cognithor_ui/widgets/cognithor_section.dart';

class WorkflowsScreen extends StatefulWidget {
  const WorkflowsScreen({super.key});

  @override
  State<WorkflowsScreen> createState() => _WorkflowsScreenState();
}

class _WorkflowsScreenState extends State<WorkflowsScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabCtrl;
  List<Map<String, dynamic>> _instances = [];
  List<Map<String, dynamic>> _dagRuns = [];
  bool _instancesLoading = false;
  bool _dagLoading = false;
  String? _instancesError;
  String? _dagError;
  String? _selectedDagNodeId;
  Map<String, dynamic>? _selectedNodeData;

  bool _initialized = false;

  @override
  void initState() {
    super.initState();
    _tabCtrl = TabController(length: 3, vsync: this);
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      _initialized = true;
      _loadData();
    }
  }

  void _loadData() {
    final provider = context.read<WorkflowProvider>();
    provider.setApi(context.read<ConnectionProvider>().api);
    provider.loadCategories();
    _loadInstances();
    _loadDagRuns();
  }

  @override
  void dispose() {
    _tabCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadInstances() async {
    setState(() {
      _instancesLoading = true;
      _instancesError = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final result = await api.getWorkflowInstances();
      if (result.containsKey('error')) {
        setState(() {
          _instancesError = result['error'].toString();
          _instancesLoading = false;
        });
        return;
      }
      final list = result['instances'] as List? ?? [];
      setState(() {
        _instances = list.map((e) => e as Map<String, dynamic>).toList();
        _instancesLoading = false;
      });
    } catch (_) {
      setState(() {
        _instancesError = 'Instances endpoint not reachable';
        _instancesLoading = false;
      });
    }
  }

  Future<void> _loadDagRuns() async {
    setState(() {
      _dagLoading = true;
      _dagError = null;
    });
    try {
      final api = context.read<ConnectionProvider>().api;
      final result = await api.getWorkflowDagRuns();
      if (result.containsKey('error')) {
        setState(() {
          _dagError = result['error'].toString();
          _dagLoading = false;
        });
        return;
      }
      final list = result['runs'] as List? ?? [];
      setState(() {
        _dagRuns = list.map((e) => e as Map<String, dynamic>).toList();
        _dagLoading = false;
      });
    } catch (_) {
      setState(() {
        _dagError = 'DAG runs endpoint not reachable';
        _dagLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(l.workflowsTitle),
        bottom: TabBar(
          controller: _tabCtrl,
          labelColor: CognithorTheme.accent,
          indicatorColor: CognithorTheme.accent,
          tabs: [
            Tab(text: l.templates),
            Tab(text: l.instances),
            Tab(text: l.dagRuns),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabCtrl,
        children: [_buildTemplatesTab(l), _buildInstancesTab(), _buildDagTab()],
      ),
    );
  }

  Widget _buildTemplatesTab(AppLocalizations l) {
    return Consumer<WorkflowProvider>(
      builder: (context, provider, _) {
        if (provider.isLoading && provider.categories.isEmpty) {
          return const Padding(
            padding: EdgeInsets.all(CognithorTheme.spacing),
            child: CognithorLoadingSkeleton(count: 5, height: 20),
          );
        }

        if (provider.error != null && provider.categories.isEmpty) {
          return CognithorEmptyState(
            icon: Icons.error_outline,
            title: l.errorLabel,
            subtitle: provider.error,
            action: ElevatedButton.icon(
              onPressed: () => provider.loadCategories(),
              icon: const Icon(Icons.refresh),
              label: Text(l.retry),
            ),
          );
        }

        if (provider.categories.isEmpty) {
          return CognithorEmptyState(
            icon: Icons.account_tree,
            title: l.noWorkflows,
            subtitle: l.comingSoon,
          );
        }

        final categories = provider.categories;
        return RefreshIndicator(
          onRefresh: () => provider.loadCategories(),
          child: CustomScrollView(
            slivers: [
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(
                  CognithorTheme.spacing,
                  CognithorTheme.spacing,
                  CognithorTheme.spacing,
                  CognithorTheme.spacingSm,
                ),
                sliver: SliverToBoxAdapter(
                  child: CognithorSection(title: l.categories),
                ),
              ),
              SliverPadding(
                padding: const EdgeInsets.symmetric(
                  horizontal: CognithorTheme.spacing,
                ),
                sliver: SliverList.builder(
                  itemCount: categories.length,
                  itemBuilder: (context, i) =>
                      _buildCategoryCard(categories[i]),
                ),
              ),
              const SliverPadding(
                padding: EdgeInsets.only(bottom: CognithorTheme.spacing),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildCategoryCard(dynamic category) {
    final l = AppLocalizations.of(context);
    final name = (category is Map ? category['name'] : category).toString();
    final description = category is Map
        ? category['description']?.toString() ?? ''
        : '';
    final templates = category is Map
        ? (category['templates'] as List?)?.length ?? 0
        : 0;

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: NeonCard(
        tint: CognithorTheme.sectionAdmin,
        glowOnHover: true,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(
                  Icons.category,
                  size: 18,
                  color: CognithorTheme.sectionAdmin,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    name,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
              ],
            ),
            if (description.isNotEmpty) ...[
              const SizedBox(height: 4),
              Text(
                description,
                style: Theme.of(context).textTheme.bodySmall,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ],
            const SizedBox(height: 8),
            Row(
              children: [
                CognithorChip(
                  label: '$templates ${l.templates}',
                  icon: Icons.description,
                  color: CognithorTheme.accent,
                ),
                const Spacer(),
                if (category is Map &&
                    (category['templates'] as List?)?.isNotEmpty == true)
                  NeonGlow(
                    color: CognithorTheme.sectionAdmin,
                    intensity: 0.2,
                    blurRadius: 8,
                    child: ElevatedButton.icon(
                      onPressed: () =>
                          _startWorkflow(Map<String, dynamic>.from(category)),
                      icon: const Icon(Icons.play_arrow, size: 18),
                      label: Text(l.startComponent),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 8,
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInstancesTab() {
    final l = AppLocalizations.of(context);
    if (_instancesLoading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_instancesError != null) {
      return RefreshIndicator(
        onRefresh: _loadInstances,
        child: ListView(
          children: [
            SizedBox(
              height: MediaQuery.of(context).size.height * 0.6,
              child: CognithorEmptyState(
                icon: Icons.cloud_off,
                title: l.notAvailable,
                subtitle: _instancesError,
              ),
            ),
          ],
        ),
      );
    }
    if (_instances.isEmpty) {
      return RefreshIndicator(
        onRefresh: _loadInstances,
        child: ListView(
          children: [
            SizedBox(
              height: MediaQuery.of(context).size.height * 0.6,
              child: CognithorEmptyState(
                icon: Icons.history,
                title: l.noInstances,
                subtitle: l.startWorkflow,
              ),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadInstances,
      child: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: _instances.length,
        itemBuilder: (context, i) {
          final inst = _instances[i];
          final status = (inst['status'] ?? 'unknown').toString();
          final progress = (inst['progress'] as num?)?.toDouble() ?? 0;
          final duration = (inst['duration'] ?? '').toString();

          return Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: NeonCard(
              tint: CognithorTheme.sectionAdmin,
              glowOnHover: true,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(
                        Icons.play_circle_outline,
                        size: 18,
                        color: CognithorTheme.sectionAdmin,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          (inst['template_id'] ?? inst['id'] ?? 'Instance $i')
                              .toString(),
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                      ),
                      CognithorChip(
                        label: status,
                        color: status == 'running'
                            ? CognithorTheme.accent
                            : status == 'complete'
                            ? CognithorTheme.green
                            : CognithorTheme.orange,
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  LinearProgressIndicator(
                    value: progress,
                    color: CognithorTheme.accent,
                    backgroundColor: Theme.of(context).dividerColor,
                  ),
                  const SizedBox(height: 4),
                  if (duration.isNotEmpty)
                    Text(
                      '${l.duration}: $duration',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildDagTab() {
    final l = AppLocalizations.of(context);
    if (_dagLoading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_dagError != null) {
      return RefreshIndicator(
        onRefresh: _loadDagRuns,
        child: ListView(
          children: [
            SizedBox(
              height: MediaQuery.of(context).size.height * 0.6,
              child: CognithorEmptyState(
                icon: Icons.cloud_off,
                title: l.notAvailable,
                subtitle: _dagError,
              ),
            ),
          ],
        ),
      );
    }
    if (_dagRuns.isEmpty) {
      return RefreshIndicator(
        onRefresh: _loadDagRuns,
        child: ListView(
          children: [
            SizedBox(
              height: MediaQuery.of(context).size.height * 0.6,
              child: CognithorEmptyState(
                icon: Icons.account_tree,
                title: l.noDagRuns,
                subtitle: l.comingSoon,
              ),
            ),
          ],
        ),
      );
    }

    // Show the most recent DAG run as a graph
    final run = _dagRuns.first;
    final steps =
        (run['steps'] as List?)
            ?.map((e) => e as Map<String, dynamic>)
            .toList() ??
        [];
    final nodeResults = (run['node_results'] as Map<String, dynamic>?) ?? {};

    final nodes = <DagNode>[];
    final edges = <DagEdge>[];
    for (var i = 0; i < steps.length; i++) {
      final s = steps[i];
      final nodeId = (s['id'] ?? 'step$i').toString();
      final status = (s['status'] ?? '').toString().toLowerCase();
      nodes.add(
        DagNode(
          id: nodeId,
          label: (s['name'] ?? 'Step $i').toString(),
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
        final prevId = (steps[i - 1]['id'] ?? 'step${i - 1}').toString();
        edges.add(DagEdge(from: prevId, to: nodeId));
      }
    }

    return RefreshIndicator(
      onRefresh: _loadDagRuns,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          SizedBox(
            height: 400,
            child: Row(
              children: [
                Expanded(
                  child: InteractiveDagGraph(
                    nodes: nodes,
                    edges: edges,
                    brightness: Theme.of(context).brightness,
                    selectedNodeId: _selectedDagNodeId,
                    onNodeTap: (node) {
                      setState(() {
                        if (_selectedDagNodeId == node.id) {
                          // Deselect on second tap
                          _selectedDagNodeId = null;
                          _selectedNodeData = null;
                        } else {
                          _selectedDagNodeId = node.id;
                          // Look up data from node_results or steps
                          final result =
                              nodeResults[node.id] as Map<String, dynamic>?;
                          final step = steps
                              .cast<Map<String, dynamic>?>()
                              .firstWhere(
                                (s) => (s?['id'] ?? '').toString() == node.id,
                                orElse: () => null,
                              );
                          _selectedNodeData = <String, dynamic>{
                            'id': node.id,
                            'name': node.label,
                            if (step != null) ...step,
                            if (result != null) ...result,
                          };
                        }
                      });
                    },
                  ),
                ),
                if (_selectedNodeData != null) ...[
                  const SizedBox(width: 12),
                  DagNodeDetail(
                    nodeData: _selectedNodeData!,
                    onClose: () {
                      setState(() {
                        _selectedDagNodeId = null;
                        _selectedNodeData = null;
                      });
                    },
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _startWorkflow(Map<String, dynamic> category) async {
    final l = AppLocalizations.of(context);
    final templates = (category['templates'] as List?) ?? [];
    if (templates.isEmpty) return;

    final templateId =
        (templates.first is Map ? templates.first['id'] : templates.first)
            .toString();

    final provider = context.read<WorkflowProvider>();
    await provider.startWorkflow(templateId.toString());

    if (mounted && provider.error == null) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text(l.workflowStarted)));
      _loadInstances();
    }
  }
}
