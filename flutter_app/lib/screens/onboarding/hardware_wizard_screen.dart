// Hardware-Aware First-Run Wizard — Flutter Edition.
//
// Maps 1:1 to the CLI flow in cognithor.system.wizard.cli but with cards
// and a Stepper instead of ANSI text. Driven by OnboardingProvider which
// wraps /api/system/* endpoints (system_api.py).

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/onboarding_provider.dart';

class HardwareWizardScreen extends StatefulWidget {
  const HardwareWizardScreen({super.key, this.onCompleted});

  final VoidCallback? onCompleted;

  @override
  State<HardwareWizardScreen> createState() => _HardwareWizardScreenState();
}

class _HardwareWizardScreenState extends State<HardwareWizardScreen> {
  int _step = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final p = context.read<OnboardingProvider>();
      if (p.stage == WizardStage.idle) {
        p.detect();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Cognithor · Hardware-Setup')),
      body: Consumer<OnboardingProvider>(
        builder: (context, p, _) {
          // Auto-advance step based on stage
          if (p.stage == WizardStage.detected && _step == 0) {
            _step = 1;
          } else if (p.stage == WizardStage.awaitingChoice && _step < 2) {
            _step = 2;
          } else if (p.stage == WizardStage.applied && _step < 3) {
            _step = 3;
            WidgetsBinding.instance.addPostFrameCallback((_) {
              widget.onCompleted?.call();
            });
          }

          return Stepper(
            type: StepperType.vertical,
            currentStep: _step,
            controlsBuilder: (ctx, details) => const SizedBox.shrink(),
            steps: [
              Step(
                title: const Text('Hardware-Erkennung'),
                content: _DetectionStep(p: p),
                isActive: _step >= 0,
                state: _stepStateFor(0, p),
              ),
              Step(
                title: const Text('Was ist dir wichtig?'),
                content: _ObjectiveStep(p: p),
                isActive: _step >= 1,
                state: _stepStateFor(1, p),
              ),
              Step(
                title: const Text('Konfiguration wählen'),
                content: _SolutionsStep(p: p),
                isActive: _step >= 2,
                state: _stepStateFor(2, p),
              ),
              Step(
                title: const Text('Fertig'),
                content: _DoneStep(p: p, onCompleted: widget.onCompleted),
                isActive: _step >= 3,
                state: _stepStateFor(3, p),
              ),
            ],
          );
        },
      ),
    );
  }

  StepState _stepStateFor(int idx, OnboardingProvider p) {
    if (idx < _step) return StepState.complete;
    if (idx == _step && p.stage == WizardStage.failed) return StepState.error;
    return idx == _step ? StepState.editing : StepState.indexed;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Step 1 — Detection
// ─────────────────────────────────────────────────────────────────────────

class _DetectionStep extends StatelessWidget {
  const _DetectionStep({required this.p});
  final OnboardingProvider p;

  @override
  Widget build(BuildContext context) {
    if (p.stage == WizardStage.detecting) {
      return const Padding(
        padding: EdgeInsets.symmetric(vertical: 24),
        child: Row(
          children: [
            SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            SizedBox(width: 12),
            Text('Erkenne Hardware (≤12s) …'),
          ],
        ),
      );
    }
    if (p.stage == WizardStage.failed && p.profile == null) {
      return _ErrorBlock(
        message: p.errorMessage ?? 'Detection fehlgeschlagen.',
        onRetry: p.detect,
      );
    }
    final pr = p.profile;
    final caps = p.capabilities;
    if (pr == null || caps == null) return const SizedBox.shrink();

    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Components
        Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: pr.components.entries.map((e) {
                final value = e.value['value']?.toString() ?? '';
                final status = e.value['status']?.toString() ?? 'ok';
                final icon = status == 'ok'
                    ? Icons.check_circle_outline
                    : status == 'warn'
                    ? Icons.error_outline
                    : Icons.cancel_outlined;
                final color = status == 'ok'
                    ? Colors.green
                    : status == 'warn'
                    ? Colors.orange
                    : Colors.red;
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Row(
                    children: [
                      Icon(icon, size: 16, color: color),
                      const SizedBox(width: 8),
                      SizedBox(
                        width: 110,
                        child: Text(e.key, style: theme.textTheme.bodySmall),
                      ),
                      Expanded(child: Text(value)),
                    ],
                  ),
                );
              }).toList(),
            ),
          ),
        ),

        // Sanity warnings
        if (pr.sanityWarnings.isNotEmpty) ...[
          const SizedBox(height: 8),
          for (final w in pr.sanityWarnings)
            Container(
              margin: const EdgeInsets.symmetric(vertical: 2),
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.orange.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: Colors.orange.withValues(alpha: 0.4)),
              ),
              child: Row(
                children: [
                  const Icon(
                    Icons.error_outline,
                    size: 16,
                    color: Colors.orange,
                  ),
                  const SizedBox(width: 8),
                  Expanded(child: Text(w['message'] ?? '')),
                ],
              ),
            ),
        ],

        // Capability badges
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 4,
          children: [
            _CapChip('NVFP4', caps.canRunNvfp4),
            _CapChip('FP8', caps.canRunFp8Marlin),
            _CapChip('GGUF-CUDA', caps.canRunGgufCuda),
            _CapChip('GGUF-Metal', caps.canRunGgufMetal),
            _CapChip('vLLM-Container', caps.canRunVllmContainer),
            _CapChip('Ollama', caps.canRunOllamaNative),
          ],
        ),
      ],
    );
  }
}

class _CapChip extends StatelessWidget {
  const _CapChip(this.label, this.enabled);
  final String label;
  final bool enabled;
  @override
  Widget build(BuildContext context) {
    return Chip(
      avatar: Icon(
        enabled ? Icons.check : Icons.close,
        size: 14,
        color: enabled ? Colors.green : Colors.grey,
      ),
      label: Text(label, style: const TextStyle(fontSize: 12)),
      backgroundColor: enabled
          ? Colors.green.withValues(alpha: 0.1)
          : Colors.grey.withValues(alpha: 0.05),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Step 2 — Objective Preset
// ─────────────────────────────────────────────────────────────────────────

class _ObjectiveStep extends StatelessWidget {
  const _ObjectiveStep({required this.p});
  final OnboardingProvider p;

  static const _presets = [
    (
      'balanced',
      'Ausgewogen',
      'Standard — gute Mischung aus Qualität, Speed, Cost und Privacy.',
    ),
    ('quality', 'Beste Qualität', 'Maximale Antwort-Qualität, ggf. langsamer.'),
    (
      'speed',
      'Schnellste Antworten',
      'Minimale Latenz — bevorzugt kleinere Modelle.',
    ),
    (
      'privacy',
      'Maximale Privacy',
      'Nur lokale Inferenz, keine Cloud-Anfragen.',
    ),
    (
      'cost',
      'Geringste Kosten',
      'Lokale Inferenz oder günstigster Cloud-Anbieter.',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        for (final (key, title, desc) in _presets)
          Card(
            elevation: p.objectivePreset == key ? 4 : 1,
            color: p.objectivePreset == key
                ? Theme.of(context).colorScheme.primaryContainer
                : null,
            child: ListTile(
              title: Text(
                title,
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
              subtitle: Text(desc),
              trailing: p.objectivePreset == key
                  ? const Icon(Icons.check_circle, color: Colors.blue)
                  : const Icon(Icons.radio_button_unchecked),
              onTap: () => p.setObjective(key),
            ),
          ),
        if (p.stage == WizardStage.loadingRecommendations) ...[
          const SizedBox(height: 12),
          const Row(
            children: [
              SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
              SizedBox(width: 12),
              Text('Berechne Pareto-optimale Konfigurationen …'),
            ],
          ),
        ],
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Step 3 — Solution selection
// ─────────────────────────────────────────────────────────────────────────

class _SolutionsStep extends StatelessWidget {
  const _SolutionsStep({required this.p});
  final OnboardingProvider p;

  @override
  Widget build(BuildContext context) {
    if (p.solutions.isEmpty) {
      if (p.stage == WizardStage.failed) {
        return _ErrorBlock(
          message: p.errorMessage ?? 'Keine Lösungen erhalten.',
          onRetry: () => p.setObjective(p.objectivePreset),
        );
      }
      return const Text('Wähle zuerst ein Profil oben.');
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        for (var i = 0; i < p.solutions.length; i++)
          _SolutionCard(
            solution: p.solutions[i],
            recommended: i == 0,
            onApply: p.solutions[i].isImmediatelyRunnable
                ? () => p.apply(p.solutions[i].tierId)
                : null,
          ),
      ],
    );
  }
}

class _SolutionCard extends StatelessWidget {
  const _SolutionCard({
    required this.solution,
    required this.recommended,
    required this.onApply,
  });

  final HardwareSolution solution;
  final bool recommended;
  final VoidCallback? onApply;

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: recommended ? 4 : 1,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                if (recommended)
                  const Icon(Icons.star, color: Colors.amber, size: 18),
                if (recommended) const SizedBox(width: 4),
                Expanded(
                  child: Text(
                    solution.displayName,
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
                  ),
                ),
                if (!solution.isImmediatelyRunnable)
                  const Chip(
                    label: Text('blocked', style: TextStyle(fontSize: 11)),
                    backgroundColor: Color(0xFFFFCDD2),
                  ),
              ],
            ),
            const SizedBox(height: 8),
            _ScoreBars(breakdown: solution.scoreBreakdown),
            const SizedBox(height: 8),
            Text(
              'Setup ~${solution.estimatedSetupMinutes}min · '
              'Disk ${solution.estimatedDiskGb.toStringAsFixed(0)} GB · '
              '${solution.estimatedCostEurPerMonth == 0 ? "€0/Monat lokal" : "~€${solution.estimatedCostEurPerMonth.toStringAsFixed(0)}/Monat (Cloud)"}',
              style: TextStyle(color: Colors.grey[600], fontSize: 12),
            ),
            const SizedBox(height: 6),
            Text(solution.rationaleDe, style: const TextStyle(fontSize: 12)),
            const SizedBox(height: 6),
            Text(
              'Modelle: planner=${solution.modelSet["planner"]} · executor=${solution.modelSet["executor"]}',
              style: TextStyle(color: Colors.grey[600], fontSize: 11),
            ),
            if (solution.blockers.isNotEmpty) ...[
              const SizedBox(height: 6),
              Text(
                '⚠ blockiert durch: ${solution.blockers.join(", ")}',
                style: const TextStyle(color: Colors.orange, fontSize: 12),
              ),
            ],
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: FilledButton(
                onPressed: onApply,
                child: Text(onApply == null ? 'Nicht verfügbar' : 'Anwenden'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ScoreBars extends StatelessWidget {
  const _ScoreBars({required this.breakdown});
  final Map<String, double> breakdown;

  @override
  Widget build(BuildContext context) {
    final entries = [
      ('Q', breakdown['quality'] ?? 0, Colors.indigo),
      ('S', breakdown['speed'] ?? 0, Colors.teal),
      ('C', breakdown['cost'] ?? 0, Colors.green),
      ('P', breakdown['privacy'] ?? 0, Colors.purple),
    ];
    return Row(
      children: [
        for (final (label, value, color) in entries) ...[
          SizedBox(
            width: 14,
            child: Text(label, style: const TextStyle(fontSize: 11)),
          ),
          Expanded(
            child: Container(
              height: 8,
              decoration: BoxDecoration(
                color: Colors.grey.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(4),
              ),
              child: FractionallySizedBox(
                alignment: Alignment.centerLeft,
                widthFactor: value.clamp(0.0, 1.0),
                child: Container(
                  decoration: BoxDecoration(
                    color: color,
                    borderRadius: BorderRadius.circular(4),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 4),
          SizedBox(
            width: 26,
            child: Text(
              '${(value * 100).round()}',
              style: const TextStyle(fontSize: 11),
            ),
          ),
          const SizedBox(width: 6),
        ],
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Step 4 — Done / Apply Progress
// ─────────────────────────────────────────────────────────────────────────

class _DoneStep extends StatelessWidget {
  const _DoneStep({required this.p, required this.onCompleted});
  final OnboardingProvider p;
  final VoidCallback? onCompleted;

  @override
  Widget build(BuildContext context) {
    if (p.stage == WizardStage.applying) {
      return const Padding(
        padding: EdgeInsets.symmetric(vertical: 24),
        child: Row(
          children: [
            SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            SizedBox(width: 12),
            Text('Schreibe Konfiguration …'),
          ],
        ),
      );
    }
    if (p.stage == WizardStage.applied) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.check_circle, color: Colors.green, size: 32),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Konfiguration angewendet: ${p.appliedTierId}',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          const Text(
            'Cognithor läuft jetzt mit der für deine Hardware optimalen '
            'Konfiguration. Du kannst sie jederzeit über Settings → '
            '"Hardware neu konfigurieren" ändern.',
          ),
          const SizedBox(height: 12),
          if (onCompleted != null)
            Align(
              alignment: Alignment.centerRight,
              child: FilledButton(
                onPressed: onCompleted,
                child: const Text('Weiter'),
              ),
            ),
        ],
      );
    }
    if (p.stage == WizardStage.failed) {
      return _ErrorBlock(
        message: p.errorMessage ?? 'Apply fehlgeschlagen.',
        onRetry: p.reset,
      );
    }
    return const SizedBox.shrink();
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Shared error block
// ─────────────────────────────────────────────────────────────────────────

class _ErrorBlock extends StatelessWidget {
  const _ErrorBlock({required this.message, required this.onRetry});
  final String message;
  final VoidCallback onRetry;
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.red.withValues(alpha: 0.1),
        border: Border.all(color: Colors.red.withValues(alpha: 0.4)),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.error_outline, color: Colors.red),
              SizedBox(width: 8),
              Text('Fehler', style: TextStyle(fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 6),
          Text(message),
          const SizedBox(height: 8),
          OutlinedButton(
            onPressed: onRetry,
            child: const Text('Erneut versuchen'),
          ),
        ],
      ),
    );
  }
}
