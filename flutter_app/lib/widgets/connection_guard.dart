import 'package:flutter/material.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/providers/connection_provider.dart';

/// Wraps child with a connection-lost overlay when backend is unreachable.
///
/// Only activates after the initial connection has been established
/// (i.e. [ConnectionProvider] was connected at least once). This avoids
/// conflicting with the SplashScreen which handles the first connect.
class ConnectionGuard extends StatelessWidget {
  final Widget child;
  const ConnectionGuard({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    return Consumer<ConnectionProvider>(
      builder: (context, conn, _) {
        // Always block on version mismatch, even before first connect.
        final showOverlay =
            conn.versionMismatch ||
            (conn.wasConnected &&
                (conn.state == CognithorConnectionState.error ||
                    conn.state == CognithorConnectionState.disconnected));
        return Stack(
          children: [
            child,
            if (showOverlay)
              _ConnectionLostOverlay(
                state: conn.state,
                errorMessage: conn.errorMessage,
                onRetry: conn.connect,
                versionMismatch: conn.versionMismatch,
                frontendVersion: conn.frontendVersion,
                backendVersion: conn.backendVersion,
              ),
          ],
        );
      },
    );
  }
}

class _ConnectionLostOverlay extends StatelessWidget {
  final CognithorConnectionState state;
  final String? errorMessage;
  final VoidCallback onRetry;
  final bool versionMismatch;
  final String? frontendVersion;
  final String? backendVersion;

  const _ConnectionLostOverlay({
    required this.state,
    this.errorMessage,
    required this.onRetry,
    this.versionMismatch = false,
    this.frontendVersion,
    this.backendVersion,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      color: Colors.black.withValues(alpha: 0.85),
      child: Center(
        child: Card(
          elevation: 8,
          child: Padding(
            padding: const EdgeInsets.all(32),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  versionMismatch
                      ? Icons.system_update_alt
                      : Icons.cloud_off_rounded,
                  size: 64,
                  color: theme.colorScheme.error,
                ),
                const SizedBox(height: 16),
                Text(
                  versionMismatch
                      ? 'Version Mismatch'
                      : 'Backend nicht erreichbar',
                  style: theme.textTheme.titleLarge,
                ),
                const SizedBox(height: 12),
                if (versionMismatch) ...[
                  Text(
                    'Frontend version: ${frontendVersion ?? "unknown"}',
                    style: theme.textTheme.bodyMedium,
                  ),
                  Text(
                    'Backend version: ${backendVersion ?? "unknown"}',
                    style: theme.textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'Update Cognithor via the EXE installer or run:\n'
                    'pip install --upgrade cognithor',
                    textAlign: TextAlign.center,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurface.withValues(alpha: 0.7),
                    ),
                  ),
                  const SizedBox(height: 16),
                  OutlinedButton.icon(
                    onPressed: onRetry,
                    icon: const Icon(Icons.refresh),
                    label: Text(AppLocalizations.of(context).recheck),
                  ),
                ] else ...[
                  Text(
                    errorMessage ?? AppLocalizations.of(context).connectionLost,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: theme.colorScheme.onSurface.withValues(alpha: 0.7),
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    AppLocalizations.of(context).connectionRestoring,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
                    ),
                  ),
                  const SizedBox(height: 16),
                  OutlinedButton.icon(
                    onPressed: onRetry,
                    icon: const Icon(Icons.refresh),
                    label: Text(AppLocalizations.of(context).connectNow),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}
