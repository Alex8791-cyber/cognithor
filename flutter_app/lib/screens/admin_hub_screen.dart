import 'package:flutter/material.dart';
import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/staggered_list.dart';

import 'package:jarvis_ui/screens/agents_screen.dart';
import 'package:jarvis_ui/screens/config_screen.dart';
import 'package:jarvis_ui/screens/memory_screen.dart';
import 'package:jarvis_ui/screens/models_screen.dart';
import 'package:jarvis_ui/screens/security_screen.dart';
import 'package:jarvis_ui/screens/system_screen.dart';
import 'package:jarvis_ui/screens/credentials_screen.dart';
import 'package:jarvis_ui/screens/knowledge_graph_screen.dart';
import 'package:jarvis_ui/screens/vault_screen.dart';
import 'package:jarvis_ui/screens/learning_screen.dart';
import 'package:jarvis_ui/screens/teach_screen.dart';
import 'package:jarvis_ui/screens/workflows_screen.dart';

class AdminHubScreen extends StatefulWidget {
  const AdminHubScreen({super.key});

  @override
  State<AdminHubScreen> createState() => _AdminHubScreenState();
}

class _AdminHubScreenState extends State<AdminHubScreen> {
  int _selectedIndex = 0;

  List<_AdminSection> _buildSections(AppLocalizations l) => [
        _AdminSection(
          icon: Icons.tune,
          title: l.config,
          subtitle: l.adminConfigSubtitle,
          builder: (_) => const ConfigScreen(),
        ),
        _AdminSection(
          icon: Icons.smart_toy,
          title: l.agentsTitle,
          subtitle: l.adminAgentsSubtitle,
          builder: (_) => const AgentsScreen(),
        ),
        _AdminSection(
          icon: Icons.model_training,
          title: l.modelsTitle,
          subtitle: l.adminModelsSubtitle,
          builder: (_) => const ModelsScreen(),
        ),
        _AdminSection(
          icon: Icons.shield,
          title: l.securityTitle,
          subtitle: l.adminSecuritySubtitle,
          builder: (_) => const SecurityScreen(),
        ),
        _AdminSection(
          icon: Icons.account_tree,
          title: l.workflowsTitle,
          subtitle: l.adminWorkflowsSubtitle,
          builder: (_) => const WorkflowsScreen(),
        ),
        _AdminSection(
          icon: Icons.hub,
          title: l.memoryTitle,
          subtitle: l.adminMemorySubtitle,
          builder: (_) => const MemoryScreen(),
        ),
        _AdminSection(
          icon: Icons.lock,
          title: l.vaultTitle,
          subtitle: l.adminVaultSubtitle,
          builder: (_) => const VaultScreen(),
        ),
        _AdminSection(
          icon: Icons.dns,
          title: l.systemTitle,
          subtitle: l.adminSystemSubtitle,
          builder: (_) => const SystemScreen(),
        ),
        _AdminSection(
          icon: Icons.scatter_plot,
          title: l.knowledgeGraph,
          subtitle: 'Entity visualization',
          builder: (_) => const KnowledgeGraphScreen(),
        ),
        _AdminSection(
          icon: Icons.vpn_key,
          title: l.credentialsTitle,
          subtitle: 'Manage secrets',
          builder: (_) => const CredentialsScreen(),
        ),
        _AdminSection(
          icon: Icons.school,
          title: l.learningTitle,
          subtitle: l.adminLearningSubtitle,
          builder: (_) => const LearningScreen(),
        ),
        _AdminSection(
          icon: Icons.auto_stories,
          title: l.teachCognithor,
          subtitle: l.adminTeachSubtitle,
          builder: (_) => const TeachScreen(),
        ),
      ];

  Widget _buildList(
    BuildContext context, {
    required List<_AdminSection> sections,
    required bool isWide,
  }) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    final tiles = List.generate(sections.length, (index) {
      final section = sections[index];
      final selected = isWide && index == _selectedIndex;

      return ListTile(
        leading: Icon(
          section.icon,
          color: selected ? colorScheme.primary : JarvisTheme.textSecondary,
        ),
        title: Text(
          section.title,
          style: theme.textTheme.bodyLarge?.copyWith(
            fontWeight: selected ? FontWeight.w600 : FontWeight.normal,
            color: selected
                ? colorScheme.primary
                : theme.textTheme.bodyLarge?.color,
          ),
        ),
        subtitle: Text(
          section.subtitle,
          style: theme.textTheme.bodySmall,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
        selected: selected,
        selectedTileColor: colorScheme.primary.withValues(alpha: 0.08),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(JarvisTheme.buttonRadius),
        ),
        onTap: () {
          if (isWide) {
            setState(() => _selectedIndex = index);
          } else {
            Navigator.of(context).push(
              MaterialPageRoute<void>(builder: section.builder),
            );
          }
        },
      );
    });

    return ListView(
      padding: const EdgeInsets.symmetric(vertical: JarvisTheme.spacingSm),
      children: [
        StaggeredList(children: tiles),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    final sections = _buildSections(l);
    final isWide = MediaQuery.sizeOf(context).width > 700;

    // Clamp index in case sections list changed.
    if (_selectedIndex >= sections.length) {
      _selectedIndex = 0;
    }

    if (isWide) {
      return Scaffold(
        appBar: AppBar(title: Text(l.adminTitle)),
        body: Row(
          children: [
            SizedBox(
              width: 260,
              child: _buildList(context, sections: sections, isWide: true),
            ),
            const VerticalDivider(width: 1),
            Expanded(
              child: sections[_selectedIndex].builder(context),
            ),
          ],
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(title: Text(l.adminTitle)),
      body: _buildList(context, sections: sections, isWide: false),
    );
  }
}

class _AdminSection {
  const _AdminSection({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.builder,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final WidgetBuilder builder;
}
