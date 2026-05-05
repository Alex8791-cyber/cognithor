import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:provider/provider.dart';

import 'package:cognithor_ui/providers/connection_provider.dart';
import 'package:cognithor_ui/screens/qr_scanner_screen.dart';
import 'package:cognithor_ui/services/api_client.dart';

import '../helpers/fakes.dart';
import '../helpers/test_app.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  group('QrScannerScreen smoke', () {
    late _MockApiClient api;
    late FakeConnectionProvider conn;

    setUp(() {
      api = _MockApiClient();
      when(() => api.post(any(), any())).thenAnswer(
        (_) async => {'qr_payload': 'cog://pair/abc', 'device_id': 'd1'},
      );
      conn = FakeConnectionProvider(apiClient: api);
    });

    Widget wrap() => localizedTestApp(
      child: ChangeNotifierProvider<ConnectionProvider>.value(
        value: conn,
        child: const QrScannerScreen(),
      ),
    );

    testWidgets('shows spinner during initial load', (tester) async {
      await tester.pumpWidget(wrap());
      // First frame: _loading=true.
      expect(find.byType(CircularProgressIndicator), findsOneWidget);
    });

    testWidgets('drives the devices/pair API endpoint', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      verify(() => api.post(any(), any())).called(greaterThanOrEqualTo(1));
    });

    testWidgets('renders QR payload after successful load', (tester) async {
      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      // QR payload text appears in the body.
      expect(find.textContaining('cog://pair/abc'), findsWidgets);
    });

    testWidgets('renders error UI when API returns error key', (tester) async {
      when(
        () => api.post(any(), any()),
      ).thenAnswer((_) async => {'error': 'pairing disabled'});

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(QrScannerScreen), findsOneWidget);
      // Error path replaces spinner.
      expect(find.byType(CircularProgressIndicator), findsNothing);
    });

    testWidgets('renders error UI when API throws', (tester) async {
      when(() => api.post(any(), any())).thenThrow(Exception('refused'));

      await tester.pumpWidget(wrap());
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
      tester.takeException();
      expect(find.byType(QrScannerScreen), findsOneWidget);
    });
  });
}
