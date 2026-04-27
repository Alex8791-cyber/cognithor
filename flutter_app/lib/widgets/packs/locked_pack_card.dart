import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:cognithor_ui/data/known_packs.dart';

class LockedPackCard extends StatelessWidget {
  final KnownPack pack;

  const LockedPackCard({super.key, required this.pack});

  Future<void> _openDetail() async {
    final uri = Uri.parse(pack.packDetailUrl);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        side: BorderSide(color: pack.accentColor.withValues(alpha: 0.3)),
        borderRadius: BorderRadius.circular(4),
      ),
      color: theme.colorScheme.surface,
      child: InkWell(
        onTap: _openDetail,
        borderRadius: BorderRadius.circular(4),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: pack.accentColor.withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Icon(pack.icon, color: pack.accentColor, size: 24),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      pack.displayName,
                      style: theme.textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  Icon(
                    Icons.lock_outline,
                    size: 18,
                    color: theme.colorScheme.outline,
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                pack.tagline,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
                maxLines: 3,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 16),
              ...pack.featureBullets.map(
                (b) => Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '> ',
                        style: TextStyle(
                          color: pack.accentColor,
                          fontFamily: 'monospace',
                          fontSize: 12,
                        ),
                      ),
                      Expanded(
                        child: Text(
                          b,
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  if (pack.listPriceBadge != null) ...[
                    Text(
                      pack.listPriceBadge!,
                      style: theme.textTheme.bodySmall?.copyWith(
                        decoration: TextDecoration.lineThrough,
                        color: theme.colorScheme.outline,
                      ),
                    ),
                    const SizedBox(width: 8),
                  ],
                  Text(
                    pack.priceBadge,
                    style: theme.textTheme.titleMedium?.copyWith(
                      color: pack.accentColor,
                      fontWeight: FontWeight.bold,
                      fontFamily: 'monospace',
                    ),
                  ),
                  const Spacer(),
                  FilledButton.icon(
                    onPressed: _openDetail,
                    icon: const Icon(Icons.open_in_new, size: 16),
                    label: const Text('Get Pack'),
                    style: FilledButton.styleFrom(
                      backgroundColor: pack.accentColor,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
