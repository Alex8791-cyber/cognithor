import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class AuditPage extends StatefulWidget {
  const AuditPage({super.key});

  @override
  State<AuditPage> createState() => _AuditPageState();
}

class _AuditPageState extends State<AuditPage> {
  Map<String, dynamic>? _verifyResult;
  Map<String, dynamic>? _timestampsResult;
  bool _verifying = false;
  bool _loadingTimestamps = false;
  final _channelController = TextEditingController();

  Future<void> _verifyChain() async {
    setState(() => _verifying = true);
    try {
      final api = context.read<ConnectionProvider>().api;
      final result = await api.get('audit/verify');
      setState(() => _verifyResult = result);
    } catch (e) {
      setState(
        () => _verifyResult = {'status': 'error', 'message': e.toString()},
      );
    } finally {
      setState(() => _verifying = false);
    }
  }

  Future<void> _loadTimestamps() async {
    setState(() => _loadingTimestamps = true);
    try {
      final api = context.read<ConnectionProvider>().api;
      final result = await api.get('audit/timestamps');
      setState(() => _timestampsResult = result);
    } catch (e) {
      setState(() => _timestampsResult = {'error': e.toString()});
    } finally {
      setState(() => _loadingTimestamps = false);
    }
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadTimestamps());
  }

  @override
  void dispose() {
    _channelController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // ── Section 1: Chain Integrity ──────────────────────────
        _sectionHeader(l.auditChainIntegrity),
        const SizedBox(height: 8),
        Row(
          children: [
            ElevatedButton.icon(
              onPressed: _verifying ? null : _verifyChain,
              icon: _verifying
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.security, size: 18),
              label: Text(l.auditVerifyChain),
            ),
            const SizedBox(width: 16),
            if (_verifyResult != null) ...[
              Icon(
                _verifyResult!['status'] == 'intact'
                    ? Icons.check_circle
                    : Icons.warning,
                color: _verifyResult!['status'] == 'intact'
                    ? Colors.green
                    : CognithorTheme.orange,
                size: 20,
              ),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  '${_verifyResult!['status']}'
                  ' \u2014 ${_verifyResult!['total_entries'] ?? 0} entries,'
                  ' ${_verifyResult!['valid_entries'] ?? 0} valid',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
            ],
          ],
        ),
        if (_verifyResult != null && _verifyResult!['broken_at_line'] != null)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(
              'Chain broken at line ${_verifyResult!['broken_at_line']}',
              style: TextStyle(color: CognithorTheme.red, fontSize: 12),
            ),
          ),
        const SizedBox(height: 24),

        // ── Section 2: TSA Timestamps ──────────────────────────
        _sectionHeader(l.auditTimestamps),
        const SizedBox(height: 8),
        if (_loadingTimestamps)
          const Center(child: CircularProgressIndicator())
        else if (_timestampsResult != null) ...[
          Text(
            'TSA: ${_timestampsResult!['tsa_enabled'] == true ? 'Enabled' : 'Disabled'}'
            ' \u2014 ${_timestampsResult!['count'] ?? 0} timestamps',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: CognithorTheme.textSecondary,
            ),
          ),
          const SizedBox(height: 8),
          if ((_timestampsResult!['timestamps'] as List?)?.isNotEmpty == true)
            ...(_timestampsResult!['timestamps'] as List).map(
              (ts) => ListTile(
                dense: true,
                leading: const Icon(
                  Icons.verified,
                  size: 18,
                  color: Colors.green,
                ),
                title: Text(
                  ts['date']?.toString() ?? '',
                  style: const TextStyle(fontSize: 13),
                ),
                trailing: Text(
                  '${((ts['size_bytes'] ?? 0) / 1024).toStringAsFixed(1)} KB',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
            )
          else
            Text(
              'No timestamps yet',
              style: Theme.of(context).textTheme.bodySmall,
            ),
        ],
        const SizedBox(height: 24),

        // ── Section 3: GDPR Export ─────────────────────────────
        _sectionHeader(l.auditGdprExport),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _channelController,
                decoration: InputDecoration(
                  hintText: 'Channel filter (optional)',
                  isDense: true,
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 10,
                  ),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                style: const TextStyle(fontSize: 13),
              ),
            ),
            const SizedBox(width: 12),
            ElevatedButton(
              onPressed: () async {
                final api = context.read<ConnectionProvider>().api;
                final messenger = ScaffoldMessenger.of(context);
                final channel = _channelController.text.trim();
                final query = channel.isNotEmpty ? '?channel=$channel' : '';
                try {
                  final result = await api.get('user/audit-data$query');
                  if (mounted) {
                    messenger.showSnackBar(
                      SnackBar(
                        content: Text(
                          'Exported ${result['count'] ?? 0} entries',
                        ),
                      ),
                    );
                  }
                } catch (e) {
                  if (mounted) {
                    messenger.showSnackBar(
                      SnackBar(content: Text('Export failed: $e')),
                    );
                  }
                }
              },
              child: Text(l.auditExport),
            ),
          ],
        ),
      ],
    );
  }

  Widget _sectionHeader(String title) {
    return Text(
      title,
      style: Theme.of(context).textTheme.titleSmall?.copyWith(
        color: CognithorTheme.accent,
        fontWeight: FontWeight.w600,
      ),
    );
  }
}
