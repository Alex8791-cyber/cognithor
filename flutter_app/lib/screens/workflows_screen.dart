import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/providers/workflow_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/dag_graph_painter.dart';
import 'package:jarvis_ui/widgets/jarvis_card.dart';
import 'package:jarvis_ui/widgets/jarvis_chip.dart';
import 'package:jarvis_ui/widgets/jarvis_empty_state.dart';
import 'package:jarvis_ui/widgets/jarvis_loading_skeleton.dart';
import 'package:jarvis_ui/widgets/jarvis_section.dart';

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
        _instancesError = 'Could not reach the instances endpoint';
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
        _dagError = 'Could not reach the DAG runs endpoint';
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
          labelColor: JarvisTheme.accent,
          indicatorColor: JarvisTheme.accent,
          tabs: const [
            Tab(text: 'Templates'),
            Tab(text: 'Instances'),
            Tab(text: 'DAG Runs'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabCtrl,
        children: [
          _buildTemplatesTab(l),
          _buildInstancesTab(),
          _buildDagTab(),
        ],
      ),
    );
  }

  Widget _buildTemplatesTab(AppLocalizations l) {
    return Consumer<WorkflowProvider>(
      builder: (context, provider, _) {
        if (provider.isLoading && provider.categories.isEmpty) {
          return const Padding(
            padding: EdgeInsets.all(JarvisTheme.spacing),
            child: JarvisLoadingSkeleton(count: 5, height: 20),
          );
        }

        if (provider.error != null && provider.categories.isEmpty) {
          return JarvisEmptyState(
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
          return JarvisEmptyState(
            icon: Icons.account_tree,
            title: l.noWorkflows,
            subtitle: l.comingSoon,
          );
        }

        return RefreshIndicator(
          onRefresh: () => provider.loadCategories(),
          child: ListView(
            padding: const EdgeInsets.all(JarvisTheme.spacing),
            children: [
              JarvisSection(title: l.categories),
              const SizedBox(height: JarvisTheme.spacingSm),
              ...provider.categories.map(_buildCategoryCard),
            ],
          ),
        );
      },
    );
  }

  Widget _buildCategoryCard(dynamic category) {
    final l = AppLocalizations.of(context);
    final name = (category is Map ? category['name'] : category).toString();
    final description =
        category is Map ? category['description']?.toString() ?? '' : '';
    final templates = category is Map
        ? (category['templates'] as List?)?.length ?? 0
        : 0;

    return JarvisCard(
      title: name,
      icon: Icons.category,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (description.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(bottom: JarvisTheme.spacingSm),
              child: Text(
                description,
                style: Theme.of(context).textTheme.bodySmall,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          Row(
            children: [
              JarvisChip(
                label: '$templates ${l.templates}',
                icon: Icons.description,
                color: JarvisTheme.accent,
              ),
              const Spacer(),
              if (category is Map &&
                  (category['templates'] as List?)?.isNotEmpty == true)
                ElevatedButton.icon(
                  onPressed: () =>
                      _startWorkflow(Map<String, dynamic>.from(category)),
                  icon: const Icon(Icons.play_arrow, size: 18),
                  label: Text(l.startComponent),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 8),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildInstancesTab() {
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
              child: JarvisEmptyState(
                icon: Icons.cloud_off,
                title: 'Instances not available',
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
              child: const JarvisEmptyState(
                icon: Icons.history,
                title: 'No Instances',
                subtitle: 'Start a workflow to see instances here',
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

          return JarvisCard(
            title: (inst['template_id'] ?? inst['id'] ?? 'Instance $i')
                .toString(),
            icon: Icons.play_circle_outline,
            trailing: JarvisChip(
              label: status,
              color: status == 'running'
                  ? JarvisTheme.accent
                  : status == 'complete'
                      ? JarvisTheme.green
                      : JarvisTheme.orange,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                LinearProgressIndicator(
                    value: progress,
                    color: JarvisTheme.accent,
                    backgroundColor: Theme.of(context).dividerColor),
                const SizedBox(height: 4),
                if (duration.isNotEmpty)
                  Text('Duration: $duration',
                      style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildDagTab() {
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
              child: JarvisEmptyState(
                icon: Icons.cloud_off,
                title: 'DAG Runs not available',
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
              child: const JarvisEmptyState(
                icon: Icons.account_tree,
                title: 'No DAG Runs',
                subtitle: 'DAG execution history will appear here',
              ),
            ),
          ],
        ),
      );
    }

    // Show the most recent DAG run as a graph
    final run = _dagRuns.first;
    final steps = (run['steps'] as List?)
            ?.map((e) => e as Map<String, dynamic>)
            .toList() ??
        [];

    final nodes = <DagNode>[];
    final edges = <DagEdge>[];
    for (var i = 0; i < steps.length; i++) {
      final s = steps[i];
      final status = (s['status'] ?? '').toString().toLowerCase();
      nodes.add(DagNode(
        id: 'step$i',
        label: (s['name'] ?? 'Step $i').toString(),
        status: status == 'running'
            ? DagNodeStatus.running
            : status == 'complete' || status == 'done'
                ? DagNodeStatus.complete
                : status == 'error'
                    ? DagNodeStatus.error
                    : DagNodeStatus.pending,
      ));
      if (i > 0) edges.add(DagEdge(from: 'step${i - 1}', to: 'step$i'));
    }

    return RefreshIndicator(
      onRefresh: _loadDagRuns,
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          SizedBox(
            height: 400,
            child: CustomPaint(
              painter: DagGraphPainter(
                nodes: nodes,
                edges: edges,
                brightness: Theme.of(context).brightness,
              ),
              size: Size.infinite,
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
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(l.workflowStarted)),
      );
      _loadInstances();
    }
  }
}
