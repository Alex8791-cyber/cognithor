import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class ApprovalDialog extends StatefulWidget {
  const ApprovalDialog({
    super.key,
    required this.request,
    required this.onRespond,
  });

  final ApprovalRequest request;
  final void Function(bool approved) onRespond;

  @override
  State<ApprovalDialog> createState() => _ApprovalDialogState();
}

class _ApprovalDialogState extends State<ApprovalDialog> {
  bool _busy = false;
  String? _localError;
  String? _lastClickStatus;

  Future<void> _handle(bool approved) async {
    if (_busy) return;
    final reqId = widget.request.requestId;
    // ignore: avoid_print
    debugPrint('[APPROVAL] clicked approved=$approved id=$reqId');
    setState(() {
      _busy = true;
      _localError = null;
      _lastClickStatus = approved
          ? AppLocalizations.of(context).sendingApproval
          : AppLocalizations.of(context).sendingRejection;
    });

    try {
      // Call REST endpoint directly via the connection provider's API client.
      final api = context.read<ConnectionProvider>().api;
      debugPrint(
        '[APPROVAL] posting REST request_id=$reqId approved=$approved',
      );
      final resp = await api.post('approval_response', {
        'request_id': reqId,
        'approved': approved,
      });
      debugPrint('[APPROVAL] REST response: $resp');

      if (!mounted) return;
      if (resp['ok'] == true) {
        // Clear the pending approval in the chat provider so the dialog
        // disappears.
        setState(() {
          _lastClickStatus = approved
              ? AppLocalizations.of(context).actionApproved
              : AppLocalizations.of(context).actionRejected;
        });
        if (!mounted) return;
        context.read<ChatProvider>().clearPendingApproval();
      } else {
        final err = (resp['error'] as String?) ?? 'unbekannt';
        debugPrint('[APPROVAL] REST failed: $err');
        setState(() {
          _busy = false;
          _localError = AppLocalizations.of(context).errorWithDetail(err);
        });
      }
    } catch (e, st) {
      debugPrint('[APPROVAL] REST exception: $e\n$st');
      if (!mounted) return;
      setState(() {
        _busy = false;
        _localError = AppLocalizations.of(context).errorWithDetail('$e');
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    return Container(
      margin: const EdgeInsets.all(12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: CognithorTheme.orange.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: CognithorTheme.orange.withValues(alpha: 0.4)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Icon(Icons.shield, color: CognithorTheme.orange, size: 20),
              const SizedBox(width: 8),
              Text(
                l.approvalTitle,
                style: TextStyle(
                  color: CognithorTheme.orange,
                  fontWeight: FontWeight.w600,
                  fontSize: 15,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            l.approvalBody(widget.request.tool),
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          const SizedBox(height: 8),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Theme.of(context).cardColor,
              borderRadius: BorderRadius.circular(8),
            ),
            child: SelectableText(
              widget.request.params.toString(),
              style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
            ),
          ),
          if (widget.request.reason.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              l.approvalReason(widget.request.reason),
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
          if (_lastClickStatus != null) ...[
            const SizedBox(height: 8),
            Text(
              _lastClickStatus!,
              style: TextStyle(
                color: CognithorTheme.accent,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
          if (_localError != null) ...[
            const SizedBox(height: 8),
            Text(
              _localError!,
              style: TextStyle(
                color: CognithorTheme.red,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              OutlinedButton(
                onPressed: _busy ? null : () => _handle(false),
                style: OutlinedButton.styleFrom(
                  foregroundColor: CognithorTheme.red,
                  side: BorderSide(color: CognithorTheme.red),
                ),
                child: Text(l.reject),
              ),
              const SizedBox(width: 8),
              ElevatedButton(
                onPressed: _busy ? null : () => _handle(true),
                style: ElevatedButton.styleFrom(
                  backgroundColor: CognithorTheme.green,
                ),
                child: Text(l.approve),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
