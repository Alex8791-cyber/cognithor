import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';

class WebPage extends StatelessWidget {
  const WebPage({super.key});

  static List<String> _toStringList(dynamic v) {
    if (v is List) return v.map((e) => e.toString()).toList();
    return [];
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final web = cfg.cfg['web'] as Map<String, dynamic>? ?? {};

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            CognithorCollapsibleCard(
              title: 'Search Backends',
              icon: Icons.search,
              initiallyExpanded: true,
              children: [
                CognithorTextField(
                  label: 'SearXNG URL',
                  value: (web['searxng_url'] ?? '').toString(),
                  onChanged: (v) => cfg.set('web.searxng_url', v),
                ),
                CognithorTextField(
                  label: 'Brave API Key',
                  value: (web['brave_api_key'] ?? '').toString(),
                  onChanged: (v) => cfg.set('web.brave_api_key', v),
                  isPassword: true,
                  isSecret: true,
                ),
                CognithorTextField(
                  label: 'Google CSE API Key',
                  value: (web['google_cse_api_key'] ?? '').toString(),
                  onChanged: (v) => cfg.set('web.google_cse_api_key', v),
                  isPassword: true,
                  isSecret: true,
                ),
                CognithorTextField(
                  label: 'Google CSE CX',
                  value: (web['google_cse_cx'] ?? '').toString(),
                  onChanged: (v) => cfg.set('web.google_cse_cx', v),
                ),
                CognithorTextField(
                  label: 'Jina API Key',
                  value: (web['jina_api_key'] ?? '').toString(),
                  onChanged: (v) => cfg.set('web.jina_api_key', v),
                  isPassword: true,
                  isSecret: true,
                ),
                CognithorToggleField(
                  label: 'DuckDuckGo Enabled',
                  value: web['duckduckgo_enabled'] != false,
                  onChanged: (v) => cfg.set('web.duckduckgo_enabled', v),
                ),
              ],
            ),
            CognithorCollapsibleCard(
              title: 'Domain Filters',
              icon: Icons.filter_list,
              children: [
                CognithorDomainListField(
                  label: 'Blocklist',
                  value: _toStringList(web['domain_blocklist']),
                  onChanged: (v) => cfg.set('web.domain_blocklist', v),
                ),
                CognithorDomainListField(
                  label: 'Allowlist',
                  value: _toStringList(web['domain_allowlist']),
                  onChanged: (v) => cfg.set('web.domain_allowlist', v),
                ),
              ],
            ),
            CognithorCollapsibleCard(
              title: 'Fetch Limits',
              icon: Icons.speed,
              children: [
                CognithorNumberField(
                  label: 'Max Fetch Bytes',
                  value: (web['max_fetch_bytes'] as num?) ?? 500000,
                  onChanged: (v) => cfg.set('web.max_fetch_bytes', v),
                  min: 10000,
                ),
                CognithorNumberField(
                  label: 'Max Text Chars',
                  value: (web['max_text_chars'] as num?) ?? 20000,
                  onChanged: (v) => cfg.set('web.max_text_chars', v),
                  min: 1000,
                ),
                CognithorNumberField(
                  label: 'Fetch Timeout (s)',
                  value: (web['fetch_timeout_seconds'] as num?) ?? 15,
                  onChanged: (v) => cfg.set('web.fetch_timeout_seconds', v),
                  min: 5,
                ),
                CognithorNumberField(
                  label: 'Search and Read Max Chars',
                  value: (web['search_and_read_max_chars'] as num?) ?? 5000,
                  onChanged: (v) => cfg.set('web.search_and_read_max_chars', v),
                  min: 1000,
                ),
              ],
            ),
            CognithorCollapsibleCard(
              title: 'Search Limits',
              icon: Icons.manage_search,
              children: [
                CognithorNumberField(
                  label: 'Search Timeout (s)',
                  value: (web['search_timeout_seconds'] as num?) ?? 10,
                  onChanged: (v) => cfg.set('web.search_timeout_seconds', v),
                  min: 5,
                ),
                CognithorNumberField(
                  label: 'Max Search Results',
                  value: (web['max_search_results'] as num?) ?? 10,
                  onChanged: (v) => cfg.set('web.max_search_results', v),
                  min: 1,
                  max: 50,
                ),
              ],
            ),
            CognithorCollapsibleCard(
              title: 'DuckDuckGo Rate Limiting',
              icon: Icons.timer,
              children: [
                CognithorNumberField(
                  label: 'Min Delay (s)',
                  value: (web['ddg_min_delay_seconds'] as num?) ?? 2,
                  onChanged: (v) => cfg.set('web.ddg_min_delay_seconds', v),
                  min: 0,
                  decimal: true,
                ),
                CognithorNumberField(
                  label: 'Rate Limit Wait (s)',
                  value: (web['ddg_ratelimit_wait_seconds'] as num?) ?? 30,
                  onChanged: (v) =>
                      cfg.set('web.ddg_ratelimit_wait_seconds', v),
                  min: 0,
                  decimal: true,
                ),
                CognithorNumberField(
                  label: 'Cache TTL (s)',
                  value: (web['ddg_cache_ttl_seconds'] as num?) ?? 3600,
                  onChanged: (v) => cfg.set('web.ddg_cache_ttl_seconds', v),
                  min: 0,
                ),
              ],
            ),
            CognithorCollapsibleCard(
              title: 'HTTP Request Limits',
              icon: Icons.http,
              children: [
                CognithorNumberField(
                  label: 'Max Body Bytes',
                  value:
                      (web['http_request_max_body_bytes'] as num?) ?? 1048576,
                  onChanged: (v) =>
                      cfg.set('web.http_request_max_body_bytes', v),
                  min: 1000,
                ),
                CognithorNumberField(
                  label: 'Timeout (s)',
                  value: (web['http_request_timeout_seconds'] as num?) ?? 30,
                  onChanged: (v) =>
                      cfg.set('web.http_request_timeout_seconds', v),
                  min: 1,
                ),
                CognithorNumberField(
                  label: 'Rate Limit (s)',
                  value: (web['http_request_rate_limit_seconds'] as num?) ?? 1,
                  onChanged: (v) =>
                      cfg.set('web.http_request_rate_limit_seconds', v),
                  min: 0,
                  decimal: true,
                ),
              ],
            ),
          ],
        );
      },
    );
  }
}
