import 'package:cognithor_ui/providers/chat_provider.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('handlePastedTextForVideoUrl', () {
    test('recognizes mp4 URL and populates pendingVideoAttachment', () {
      final p = ChatProvider();
      final ok = p.handlePastedTextForVideoUrl('https://example.com/clip.mp4');
      expect(ok, isTrue);
      expect(p.pendingVideoAttachment, isNotNull);
      expect(p.pendingVideoAttachment!['url'], 'https://example.com/clip.mp4');
      expect(p.pendingVideoAttachment!['filename'], 'clip.mp4');
    });

    test('recognizes webm / mov / mkv / avi', () {
      final p = ChatProvider();
      for (final ext in ['webm', 'mov', 'mkv', 'avi']) {
        p.clearPendingVideo();
        final ok = p.handlePastedTextForVideoUrl('https://x.com/a.$ext');
        expect(ok, isTrue, reason: ext);
        expect(p.pendingVideoAttachment, isNotNull, reason: ext);
      }
    });

    test('rejects non-video URL', () {
      final p = ChatProvider();
      expect(
        p.handlePastedTextForVideoUrl('https://example.com/page.html'),
        isFalse,
      );
      expect(p.pendingVideoAttachment, isNull);
    });

    test('rejects plain text', () {
      final p = ChatProvider();
      expect(p.handlePastedTextForVideoUrl('just typing a sentence'), isFalse);
    });

    test('case-insensitive extension match', () {
      final p = ChatProvider();
      expect(p.handlePastedTextForVideoUrl('https://x.com/clip.MP4'), isTrue);
    });

    test('strips query string from filename but preserves full URL', () {
      final p = ChatProvider();
      final ok = p.handlePastedTextForVideoUrl(
        'https://cdn.example.com/clip.mp4?token=abc&exp=12345',
      );
      expect(ok, isTrue);
      expect(
        p.pendingVideoAttachment!['url'],
        'https://cdn.example.com/clip.mp4?token=abc&exp=12345',
      );
      expect(p.pendingVideoAttachment!['filename'], 'clip.mp4');
    });

    test('strips fragment from filename', () {
      final p = ChatProvider();
      final ok = p.handlePastedTextForVideoUrl(
        'https://x.com/video.webm#t=10,20',
      );
      expect(ok, isTrue);
      expect(p.pendingVideoAttachment!['filename'], 'video.webm');
    });

    test('strips both query and fragment from filename', () {
      final p = ChatProvider();
      final ok = p.handlePastedTextForVideoUrl(
        'https://x.com/clip.mov?x=1#frag',
      );
      expect(ok, isTrue);
      expect(p.pendingVideoAttachment!['filename'], 'clip.mov');
      expect(
        p.pendingVideoAttachment!['url'],
        'https://x.com/clip.mov?x=1#frag',
      );
    });
  });
}
