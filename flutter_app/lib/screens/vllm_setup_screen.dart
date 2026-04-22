import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class VllmSetupScreen extends StatefulWidget {
  const VllmSetupScreen({super.key});

  @override
  State<VllmSetupScreen> createState() => _VllmSetupScreenState();
}

class _VllmSetupScreenState extends State<VllmSetupScreen> {
  LlmBackendProvider? _provider;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      _provider = context.read<LlmBackendProvider>();
      _provider!.startPolling();
    });
  }

  @override
  void dispose() {
    _provider?.stopPolling();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final p = context.watch<LlmBackendProvider>();
    final s = p.vllmStatus;

    return Scaffold(
      appBar: AppBar(title: const Text('Configure vLLM')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _HardwareCard(status: s),
          const SizedBox(height: 12),
          _DockerCard(status: s),
          const SizedBox(height: 12),
          _ImageCard(status: s),
          const SizedBox(height: 12),
          _ModelCard(status: s),
        ],
      ),
    );
  }
}

enum _CardState { ok, todo, pending, error }

Color _colorFor(_CardState st, BuildContext ctx) {
  switch (st) {
    case _CardState.ok:
      return Colors.green;
    case _CardState.todo:
      return Colors.orange;
    case _CardState.pending:
      return Theme.of(ctx).colorScheme.outline;
    case _CardState.error:
      return Colors.red;
  }
}

IconData _iconFor(_CardState st) {
  switch (st) {
    case _CardState.ok:
      return Icons.check_circle;
    case _CardState.todo:
      return Icons.radio_button_unchecked;
    case _CardState.pending:
      return Icons.more_horiz;
    case _CardState.error:
      return Icons.error;
  }
}

class _StatusCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final _CardState state;
  final Widget? action;
  final ValueKey<String> cardKey;

  const _StatusCard({
    required this.title,
    required this.subtitle,
    required this.state,
    required this.cardKey,
    this.action,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      key: cardKey,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: BorderSide(color: _colorFor(state, context), width: 1),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            Icon(_iconFor(state), color: _colorFor(state, context)),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title,
                      style: const TextStyle(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    style:
                        const TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                ],
              ),
            ),
            if (action != null) action!,
          ],
        ),
      ),
    );
  }
}

class _HardwareCard extends StatelessWidget {
  final VLLMStatus? status;
  const _HardwareCard({required this.status});

  @override
  Widget build(BuildContext context) {
    final hw = status?.hardwareInfo;
    final ok = status?.hardwareOk ?? false;
    return _StatusCard(
      cardKey: const ValueKey('card-hardware'),
      title: 'NVIDIA GPU',
      subtitle: hw == null
          ? 'Not detected'
          : '${hw.gpuName}, ${hw.vramGb} GB, SM ${hw.computeCapability}',
      state: ok ? _CardState.ok : _CardState.error,
    );
  }
}

class _DockerCard extends StatelessWidget {
  final VLLMStatus? status;
  const _DockerCard({required this.status});

  @override
  Widget build(BuildContext context) {
    final ok = status?.dockerOk ?? false;
    return _StatusCard(
      cardKey: const ValueKey('card-docker'),
      title: 'Docker Desktop',
      subtitle: ok ? 'Running' : 'Not running — start Docker Desktop',
      state: ok ? _CardState.ok : _CardState.todo,
    );
  }
}

class _ImageCard extends StatefulWidget {
  final VLLMStatus? status;
  const _ImageCard({required this.status});

  @override
  State<_ImageCard> createState() => _ImageCardState();
}

class _ImageCardState extends State<_ImageCard> {
  double? _progress;
  String? _layer;
  bool _pulling = false;

  Future<void> _pull() async {
    setState(() {
      _pulling = true;
      _progress = null;
      _layer = null;
    });
    try {
      final p = context.read<LlmBackendProvider>();
      await for (final ev in p.pullImage()) {
        final detail = ev['progressDetail'] as Map<String, dynamic>?;
        final current = detail?['current'] as int?;
        final total = detail?['total'] as int?;
        if (current != null && total != null && total > 0) {
          setState(() {
            _progress = current / total;
            _layer = ev['id'] as String?;
          });
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Pull failed: $e')));
      }
    } finally {
      if (mounted) setState(() => _pulling = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final pulled = widget.status?.imagePulled ?? false;
    return _StatusCard(
      cardKey: const ValueKey('card-image'),
      title: 'vLLM Docker image',
      subtitle: pulled
          ? 'vllm/vllm-openai:v0.19.1 ready'
          : _pulling
              ? 'Downloading${_layer != null ? " layer ${_layer!.length >= 12 ? _layer!.substring(0, 12) : _layer!}..." : "…"}'
              : 'Pull required (~10 GB, one-time)',
      state: pulled
          ? _CardState.ok
          : _pulling
              ? _CardState.pending
              : _CardState.todo,
      action: pulled
          ? null
          : _pulling
              ? SizedBox(
                  width: 100,
                  child: LinearProgressIndicator(value: _progress),
                )
              : FilledButton.tonal(
                  onPressed: _pull,
                  child: const Text('Pull image'),
                ),
    );
  }
}

class _ModelCard extends StatefulWidget {
  final VLLMStatus? status;
  const _ModelCard({required this.status});

  @override
  State<_ModelCard> createState() => _ModelCardState();
}

class _ModelCardState extends State<_ModelCard> {
  String? _selected;
  bool _starting = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) context.read<LlmBackendProvider>().fetchAvailableModels();
    });
  }

  @override
  Widget build(BuildContext context) {
    final p = context.watch<LlmBackendProvider>();
    final running = widget.status?.containerRunning ?? false;
    final pulled = widget.status?.imagePulled ?? false;

    if (running) {
      return _StatusCard(
        cardKey: const ValueKey('card-model'),
        title: 'Model',
        subtitle: 'Running: ${widget.status?.currentModel ?? "unknown"}',
        state: _CardState.ok,
      );
    }

    if (!pulled) {
      return const _StatusCard(
        cardKey: ValueKey('card-model'),
        title: 'Model',
        subtitle: 'Available after image pull',
        state: _CardState.pending,
      );
    }

    final models = p.availableModels;
    final recommendedId = p.recommendedModelId;
    _selected ??=
        recommendedId ?? (models.isNotEmpty ? models[0]['id'] as String : null);

    return Card(
      key: const ValueKey('card-model'),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Model',
              style: TextStyle(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            DropdownButton<String>(
              value: _selected,
              isExpanded: true,
              onChanged: (v) => setState(() => _selected = v),
              items: [
                for (final m in models)
                  DropdownMenuItem<String>(
                    value: m['id'] as String,
                    enabled: m['fits'] as bool,
                    child: Row(
                      children: [
                        if (m['id'] == recommendedId)
                          const Padding(
                            padding: EdgeInsets.only(right: 6),
                            child: Icon(
                              Icons.star,
                              size: 14,
                              color: Colors.amber,
                            ),
                          ),
                        Expanded(child: Text(m['display_name'] as String)),
                        Text(
                          '${m['vram_gb_min']} GB',
                          style: TextStyle(
                            fontSize: 11,
                            color: (m['fits'] as bool)
                                ? Colors.grey
                                : Colors.red,
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 8),
            FilledButton(
              onPressed: _starting || _selected == null
                  ? null
                  : () async {
                      setState(() => _starting = true);
                      final messenger = ScaffoldMessenger.of(context);
                      try {
                        await p.startContainer(_selected!);
                      } catch (e) {
                        if (mounted) {
                          messenger.showSnackBar(
                            SnackBar(content: Text('Start failed: $e')),
                          );
                        }
                      } finally {
                        if (mounted) setState(() => _starting = false);
                      }
                    },
              child: Text(_starting ? 'Starting\u2026' : 'Start vLLM'),
            ),
          ],
        ),
      ),
    );
  }
}
