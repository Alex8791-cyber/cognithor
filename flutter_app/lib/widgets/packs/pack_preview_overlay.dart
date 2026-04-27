import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:cognithor_ui/data/known_packs.dart';

class PackPreviewOverlay extends StatelessWidget {
  final KnownPack pack;
  final Widget child;

  const PackPreviewOverlay({
    super.key,
    required this.pack,
    required this.child,
  });

  Future<void> _openDetail() async {
    final uri = Uri.parse(pack.packDetailUrl);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Stack(
      children: [
        IgnorePointer(child: Opacity(opacity: 0.35, child: child)),
        Positioned.fill(
          child: Center(
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 24),
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: theme.colorScheme.surface.withValues(alpha: 0.95),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: pack.accentColor.withValues(alpha: 0.4),
                ),
                boxShadow: [
                  BoxShadow(
                    color: pack.accentColor.withValues(alpha: 0.1),
                    blurRadius: 24,
                    spreadRadius: 4,
                  ),
                ],
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: pack.accentColor.withValues(alpha: 0.1),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      Icons.lock_outline,
                      color: pack.accentColor,
                      size: 32,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    pack.displayName,
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Install this pack to unlock these settings',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  if (pack.listPriceBadge != null)
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          pack.listPriceBadge!,
                          style: theme.textTheme.bodySmall?.copyWith(
                            decoration: TextDecoration.lineThrough,
                            color: theme.colorScheme.outline,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          pack.priceBadge,
                          style: theme.textTheme.titleMedium?.copyWith(
                            color: pack.accentColor,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'monospace',
                          ),
                        ),
                      ],
                    )
                  else
                    Text(
                      pack.priceBadge,
                      style: theme.textTheme.titleMedium?.copyWith(
                        color: pack.accentColor,
                        fontWeight: FontWeight.bold,
                        fontFamily: 'monospace',
                      ),
                    ),
                  const SizedBox(height: 20),
                  FilledButton.icon(
                    onPressed: _openDetail,
                    icon: const Icon(Icons.open_in_new, size: 16),
                    label: const Text('Get this Pack'),
                    style: FilledButton.styleFrom(
                      backgroundColor: pack.accentColor,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 24,
                        vertical: 14,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}
