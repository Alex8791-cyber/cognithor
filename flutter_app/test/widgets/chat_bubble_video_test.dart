import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:cognithor_ui/widgets/chat_bubble.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('video-kind metadata renders filename + duration + sampling', (
    tester,
  ) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: ChatBubble(
            role: MessageRole.user,
            text: 'Was siehst du?',
            metadata: {
              'kind': 'video',
              'filename': 'drone.mp4',
              'duration_sec': 42.5,
              'sampling': {'fps': 2.0},
              'thumb_url': null,
            },
          ),
        ),
      ),
    );

    expect(find.text('drone.mp4'), findsOneWidget);
    expect(find.textContaining('0:42'), findsOneWidget);
    expect(find.textContaining('fps=2'), findsOneWidget);
  });

  testWidgets('long-video banner appears when duration > 15 minutes', (
    tester,
  ) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: ChatBubble(
            role: MessageRole.user,
            text: 'Analyze',
            metadata: {
              'kind': 'video',
              'filename': 'lecture.mp4',
              'duration_sec': 2400.0,
              'sampling': {'num_frames': 32},
            },
          ),
        ),
      ),
    );
    expect(find.byKey(const ValueKey('video-long-banner')), findsOneWidget);
  });

  testWidgets('short video shows no banner', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: ChatBubble(
            role: MessageRole.user,
            text: 'Kurzes Video',
            metadata: {
              'kind': 'video',
              'filename': 'clip.mp4',
              'duration_sec': 30.0,
              'sampling': {'fps': 2.0},
            },
          ),
        ),
      ),
    );
    expect(find.byKey(const ValueKey('video-long-banner')), findsNothing);
  });
}
