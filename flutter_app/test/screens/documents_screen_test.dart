import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:cognithor_ui/screens/documents_screen.dart';

import '../helpers/test_app.dart';

void main() {
  group('DocumentsScreen smoke', () {
    testWidgets('renders the placeholder title + icon', (tester) async {
      await tester.pumpWidget(
        localizedTestApp(child: const DocumentsScreen()),
      );
      expect(find.byType(DocumentsScreen), findsOneWidget);
      expect(find.byIcon(Icons.description_outlined), findsOneWidget);
      expect(find.text('Dokument-Vorlagen'), findsOneWidget);
    });

    testWidgets('lists the 6 built-in templates by name', (tester) async {
      await tester.pumpWidget(
        localizedTestApp(child: const DocumentsScreen()),
      );
      // The screen advertises 6 templates. Once they're wired up the
      // string changes — this guards the public name list.
      expect(
        find.textContaining('Brief, Rechnung, Bericht'),
        findsOneWidget,
      );
    });

    testWidgets('shows the chat-fallback hint', (tester) async {
      await tester.pumpWidget(
        localizedTestApp(child: const DocumentsScreen()),
      );
      // Hint pointing the user to chat-driven document creation.
      expect(
        find.textContaining('Erstelle einen Brief'),
        findsOneWidget,
      );
    });
  });
}
