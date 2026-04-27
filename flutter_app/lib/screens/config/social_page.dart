import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/data/known_packs.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/providers/sources_provider.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';
import 'package:cognithor_ui/widgets/packs/pack_preview_overlay.dart';

class SocialPage extends StatelessWidget {
  const SocialPage({super.key});

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final social = cfg.cfg['social'] as Map<String, dynamic>? ?? {};
        final productName = (social['reddit_product_name'] ?? '').toString();
        final hasSubs =
            (social['reddit_subreddits'] as List<dynamic>?)?.isNotEmpty ??
            false;
        final isConfigured = productName.isNotEmpty && hasSubs;

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Consumer<SourcesProvider>(
              builder: (context, sources, _) {
                final redditInstalled = sources.hasSource('reddit');
                final redditPack = findKnownPackBySourceId('reddit');

                Widget redditConfig = Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (!isConfigured)
                      Container(
                        margin: const EdgeInsets.only(bottom: 16),
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.orange.withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(
                            color: Colors.orange.withValues(alpha: 0.3),
                          ),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              Icons.warning_amber,
                              color: Colors.orange[300],
                              size: 20,
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                l.socialSetupRequired,
                                style: Theme.of(context).textTheme.bodySmall
                                    ?.copyWith(color: Colors.orange[300]),
                              ),
                            ),
                          ],
                        ),
                      ),
                    CognithorToggleField(
                      label: l.autoScan,
                      value: social['reddit_scan_enabled'] == true,
                      onChanged: (v) =>
                          cfg.set('social.reddit_scan_enabled', v),
                    ),
                    CognithorTextField(
                      label: l.productName,
                      value: (social['reddit_product_name'] ?? '').toString(),
                      placeholder: 'e.g. Cognithor',
                      onChanged: (v) =>
                          cfg.set('social.reddit_product_name', v),
                    ),
                    CognithorTextField(
                      label: l.productDescription,
                      value: (social['reddit_product_description'] ?? '')
                          .toString(),
                      placeholder: 'One-sentence description for AI scoring',
                      onChanged: (v) =>
                          cfg.set('social.reddit_product_description', v),
                    ),
                    CognithorTextField(
                      label: l.replyTone,
                      value: (social['reddit_reply_tone'] ?? '').toString(),
                      placeholder:
                          'helpful, technically credible, no sales pitch',
                      onChanged: (v) => cfg.set('social.reddit_reply_tone', v),
                    ),
                    CognithorTextField(
                      label: l.subreddits,
                      value:
                          (social['reddit_subreddits'] as List<dynamic>?)?.join(
                            ', ',
                          ) ??
                          '',
                      description: l.subredditsHint,
                      placeholder: 'LocalLLaMA, SaaS, Python',
                      onChanged: (v) => cfg.set(
                        'social.reddit_subreddits',
                        v
                            .split(',')
                            .map((s) => s.trim())
                            .where((s) => s.isNotEmpty)
                            .toList(),
                      ),
                    ),
                    CognithorNumberField(
                      label: l.minIntentScore,
                      value: (social['reddit_min_score'] as num?) ?? 60,
                      min: 0,
                      max: 100,
                      onChanged: (v) => cfg.set('social.reddit_min_score', v),
                    ),
                    CognithorNumberField(
                      label: l.scanInterval,
                      value:
                          (social['reddit_scan_interval_minutes'] as num?) ??
                          30,
                      min: 5,
                      max: 1440,
                      onChanged: (v) =>
                          cfg.set('social.reddit_scan_interval_minutes', v),
                    ),
                    CognithorToggleField(
                      label: l.autoPost,
                      value: social['reddit_auto_post'] == true,
                      description: l.autoPostHint,
                      onChanged: (v) => cfg.set('social.reddit_auto_post', v),
                    ),
                    CognithorTextField(
                      label: 'Auto-Post Whitelist',
                      value:
                          (social['reddit_auto_post_whitelist']
                                  as List<dynamic>?)
                              ?.join(', ') ??
                          '',
                      placeholder: 'ollama, selfhosted',
                      description:
                          'Subreddits where auto-posting is allowed (comma-separated, '
                          'without r/). Leave empty to disable auto-posting globally. '
                          'Non-whitelisted subs always fall back to clipboard review.',
                      onChanged: (v) => cfg.set(
                        'social.reddit_auto_post_whitelist',
                        v
                            .split(',')
                            .map((s) => s.trim())
                            .where((s) => s.isNotEmpty)
                            .toList(),
                      ),
                    ),
                    CognithorNumberField(
                      label: 'Min Auto-Post Score',
                      value: (social['reddit_min_auto_score'] as num?) ?? 85,
                      min: 0,
                      max: 100,
                      description:
                          'Minimum intent score required to auto-post. Leads below '
                          'this threshold always fall back to clipboard review, '
                          'even on whitelisted subs.',
                      onChanged: (v) =>
                          cfg.set('social.reddit_min_auto_score', v),
                    ),
                  ],
                );

                if (!redditInstalled && redditPack != null) {
                  redditConfig = SizedBox(
                    height: 400,
                    child: PackPreviewOverlay(
                      pack: redditPack,
                      child: redditConfig,
                    ),
                  );
                }

                return redditConfig;
              },
            ),

            // ── Hacker News ─────────────────────────────────
            const Divider(height: 32),
            Text('Hacker News', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            CognithorToggleField(
              label: 'HN Scanning',
              value: social['hn_enabled'] == true,
              onChanged: (v) => cfg.set('social.hn_enabled', v),
            ),
            CognithorTextField(
              label: 'HN Categories',
              value:
                  (social['hn_categories'] as List<dynamic>?)?.join(', ') ??
                  'top, new',
              placeholder: 'top, new, best, ask, show',
              onChanged: (v) => cfg.set(
                'social.hn_categories',
                v
                    .split(',')
                    .map((s) => s.trim())
                    .where((s) => s.isNotEmpty)
                    .toList(),
              ),
            ),
            CognithorNumberField(
              label: 'HN Min Score',
              value: (social['hn_min_score'] as num?) ?? 60,
              min: 0,
              max: 100,
              onChanged: (v) => cfg.set('social.hn_min_score', v),
            ),
            CognithorNumberField(
              label: 'HN Scan Interval (min)',
              value: (social['hn_scan_interval_minutes'] as num?) ?? 60,
              min: 10,
              max: 1440,
              onChanged: (v) => cfg.set('social.hn_scan_interval_minutes', v),
            ),

            // ── Discord ─────────────────────────────────────
            const Divider(height: 32),
            Text('Discord', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            CognithorToggleField(
              label: 'Discord Scanning',
              value: social['discord_scanner_enabled'] == true,
              description: 'Requires COGNITHOR_DISCORD_TOKEN env var',
              onChanged: (v) => cfg.set('social.discord_scanner_enabled', v),
            ),
            CognithorTextField(
              label: 'Discord Channel IDs',
              value:
                  (social['discord_scan_channels'] as List<dynamic>?)?.join(
                    ', ',
                  ) ??
                  '',
              placeholder: '123456789, 987654321',
              description: 'Comma-separated Discord channel IDs to monitor',
              onChanged: (v) => cfg.set(
                'social.discord_scan_channels',
                v
                    .split(',')
                    .map((s) => s.trim())
                    .where((s) => s.isNotEmpty)
                    .toList(),
              ),
            ),
            CognithorNumberField(
              label: 'Discord Min Score',
              value: (social['discord_min_score'] as num?) ?? 60,
              min: 0,
              max: 100,
              onChanged: (v) => cfg.set('social.discord_min_score', v),
            ),
            CognithorNumberField(
              label: 'Discord Scan Interval (min)',
              value: (social['discord_scan_interval_minutes'] as num?) ?? 30,
              min: 5,
              max: 1440,
              onChanged: (v) =>
                  cfg.set('social.discord_scan_interval_minutes', v),
            ),

            // ── RSS / Atom ──────────────────────────────────
            const Divider(height: 32),
            Text(
              'RSS / Atom Feeds',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            CognithorToggleField(
              label: 'RSS Scanning',
              value: social['rss_enabled'] == true,
              description:
                  'Scan any RSS or Atom feed (news sites, blogs, forums) for leads',
              onChanged: (v) => cfg.set('social.rss_enabled', v),
            ),
            CognithorTextField(
              label: 'RSS Feed URLs',
              value: (social['rss_feeds'] as List<dynamic>?)?.join(', ') ?? '',
              placeholder:
                  'https://example.com/feed.xml, https://blog.example.com/rss',
              description:
                  'Comma-separated full feed URLs (RSS 2.0 or Atom). Each entry is scored by the LLM.',
              onChanged: (v) => cfg.set(
                'social.rss_feeds',
                v
                    .split(',')
                    .map((s) => s.trim())
                    .where((s) => s.isNotEmpty)
                    .toList(),
              ),
            ),
            CognithorNumberField(
              label: 'RSS Min Score',
              value: (social['rss_min_score'] as num?) ?? 60,
              min: 0,
              max: 100,
              onChanged: (v) => cfg.set('social.rss_min_score', v),
            ),
            CognithorNumberField(
              label: 'RSS Scan Interval (min)',
              value: (social['rss_scan_interval_minutes'] as num?) ?? 60,
              min: 5,
              max: 1440,
              onChanged: (v) => cfg.set('social.rss_scan_interval_minutes', v),
            ),
          ],
        );
      },
    );
  }
}
