/// Test fakes shared across screen smoke tests.
///
/// `FakeConnectionProvider` lets tests render screens that depend on a
/// post-connect `ConnectionProvider.api` without spinning up a real
/// HTTP/WS lifecycle. The fake just hands back a caller-supplied
/// (typically mocktail-mocked) ApiClient + WebSocketService.
library;

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';
import 'package:cognithor_ui/services/websocket_service.dart';

class FakeConnectionProvider extends ConnectionProvider {
  FakeConnectionProvider({
    required ApiClient apiClient,
    WebSocketService? wsService,
    CognithorConnectionState initialState = CognithorConnectionState.connected,
    String? backendVersion,
  }) : _api = apiClient,
       _ws = wsService {
    state = initialState;
    if (backendVersion != null) {
      this.backendVersion = backendVersion;
    }
  }

  final ApiClient _api;
  final WebSocketService? _ws;

  @override
  ApiClient get api => _api;

  @override
  WebSocketService get ws =>
      _ws ?? (throw StateError('FakeConnectionProvider has no WS configured'));
}
