import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import 'package:cognithor_ui/providers/security_provider.dart';
import 'package:cognithor_ui/services/api_client.dart';

class _MockApiClient extends Mock implements ApiClient {}

void main() {
  setUpAll(() {
    registerFallbackValue(<String, dynamic>{});
  });

  group('SecurityProvider', () {
    late _MockApiClient api;
    late SecurityProvider provider;

    setUp(() {
      api = _MockApiClient();
      provider = SecurityProvider()..setApi(api);
    });

    test('initial state is empty', () {
      expect(provider.roles, isNull);
      expect(provider.complianceReport, isNull);
      expect(provider.complianceStats, isNull);
      expect(provider.decisions, isNull);
      expect(provider.remediations, isNull);
      expect(provider.redteamStatus, isNull);
      expect(provider.authStats, isNull);
      expect(provider.auditEntries, isEmpty);
      expect(provider.isLoading, false);
      expect(provider.error, isNull);
      expect(provider.partialError, isNull);
    });

    test('loadRoles stores result', () async {
      when(() => api.getRbacRoles()).thenAnswer(
        (_) async => {
          'roles': ['admin', 'user'],
        },
      );

      await provider.loadRoles();

      expect(provider.roles, isNotNull);
      expect(provider.error, isNull);
      expect(provider.isLoading, false);
    });

    test('loadComplianceReport sets error on failure', () async {
      when(
        () => api.getComplianceReport(),
      ).thenThrow(Exception('compliance-down'));

      await provider.loadComplianceReport();

      expect(provider.complianceReport, isNull);
      expect(provider.error, contains('compliance-down'));
    });

    test('loadDecisions stores result', () async {
      when(
        () => api.getComplianceDecisions(),
      ).thenAnswer((_) async => {'decisions': []});

      await provider.loadDecisions();

      expect(provider.decisions, isNotNull);
    });

    test('loadAudit normalizes non-list "entries" to empty', () async {
      when(
        () => api.getMonitoringAudit(
          action: any(named: 'action'),
          severity: any(named: 'severity'),
        ),
      ).thenAnswer((_) async => {'entries': 'not-a-list'});

      await provider.loadAudit();

      expect(provider.auditEntries, isEmpty);
      expect(provider.error, isNull);
    });

    test('loadAudit forwards action + severity filters', () async {
      when(
        () => api.getMonitoringAudit(action: 'create', severity: 'high'),
      ).thenAnswer(
        (_) async => {
          'entries': [
            {'id': 'a1'},
          ],
        },
      );

      await provider.loadAudit(action: 'create', severity: 'high');

      expect(provider.auditEntries.length, 1);
      verify(
        () => api.getMonitoringAudit(action: 'create', severity: 'high'),
      ).called(1);
    });

    test('runRedteamScan triggers loadRedteamStatus on success', () async {
      when(() => api.runRedteamScan(any())).thenAnswer((_) async => {});
      when(
        () => api.getRedteamStatus(),
      ).thenAnswer((_) async => {'last_scan': '2026-04-29T12:00:00Z'});

      await provider.runRedteamScan({'policy': 'default'});

      expect(provider.redteamStatus, isNotNull);
      verify(() => api.runRedteamScan({'policy': 'default'})).called(1);
      verify(() => api.getRedteamStatus()).called(1);
    });

    test(
      'runRedteamScan sets error on failure (no follow-up status load)',
      () async {
        when(
          () => api.runRedteamScan(any()),
        ).thenThrow(Exception('scan-failed'));

        await provider.runRedteamScan({'policy': 'default'});

        expect(provider.error, contains('scan-failed'));
        verifyNever(() => api.getRedteamStatus());
      },
    );

    test(
      'loadAll fans out to every endpoint and clears partialError on success',
      () async {
        when(() => api.getComplianceStats()).thenAnswer((_) async => {'k': 1});
        when(() => api.getComplianceRemediations()).thenAnswer((_) async => {});
        when(() => api.getComplianceReport()).thenAnswer((_) async => {});
        when(() => api.getRbacRoles()).thenAnswer((_) async => {});
        when(() => api.getAuthStats()).thenAnswer((_) async => {});
        when(() => api.getRedteamStatus()).thenAnswer((_) async => {});
        when(
          () => api.getMonitoringAudit(
            action: any(named: 'action'),
            severity: any(named: 'severity'),
          ),
        ).thenAnswer((_) async => {'entries': []});

        await provider.loadAll();

        expect(provider.complianceStats, isNotNull);
        expect(provider.partialError, isNull);
        expect(provider.error, isNull);
        expect(provider.isLoading, false);
      },
    );

    test('loadAll keeps partialError when SOME endpoints fail', () async {
      when(() => api.getComplianceStats()).thenAnswer((_) async => {'k': 1});
      when(
        () => api.getComplianceRemediations(),
      ).thenThrow(Exception('rem-down'));
      when(() => api.getComplianceReport()).thenAnswer((_) async => {});
      when(() => api.getRbacRoles()).thenAnswer((_) async => {});
      when(() => api.getAuthStats()).thenAnswer((_) async => {});
      when(() => api.getRedteamStatus()).thenAnswer((_) async => {});
      when(
        () => api.getMonitoringAudit(
          action: any(named: 'action'),
          severity: any(named: 'severity'),
        ),
      ).thenAnswer((_) async => {'entries': []});

      await provider.loadAll();

      expect(provider.partialError, contains('remediations'));
      expect(provider.error, contains('remediations'));
      expect(provider.complianceStats, isNotNull);
    });

    test('loadAll surfaces single-error when ALL endpoints fail', () async {
      when(() => api.getComplianceStats()).thenThrow(Exception('1'));
      when(() => api.getComplianceRemediations()).thenThrow(Exception('2'));
      when(() => api.getComplianceReport()).thenThrow(Exception('3'));
      when(() => api.getRbacRoles()).thenThrow(Exception('4'));
      when(() => api.getAuthStats()).thenThrow(Exception('5'));
      when(() => api.getRedteamStatus()).thenThrow(Exception('6'));
      when(
        () => api.getMonitoringAudit(
          action: any(named: 'action'),
          severity: any(named: 'severity'),
        ),
      ).thenThrow(Exception('7'));

      await provider.loadAll();

      // All 7 failed → error is the first one (compliance/1).
      expect(provider.error, contains('compliance'));
      expect(provider.partialError, isNotNull);
    });

    test('loaders no-op when API is unset', () async {
      final unbound = SecurityProvider();
      await unbound.loadRoles();
      await unbound.loadComplianceStats();
      await unbound.loadAudit();
      await unbound.runRedteamScan({});
      await unbound.loadAll();
      expect(unbound.isLoading, false);
      verifyZeroInteractions(api);
    });
  });
}
