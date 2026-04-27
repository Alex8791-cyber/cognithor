import 'dart:math';

import 'package:flutter/material.dart';

/// Matrix green color used for the rain effect.
const matrixGreen = Color(0xFF00FF41);

/// CustomPainter that draws Matrix-style falling characters.
///
/// Columns of random characters (katakana, digits, latin) fall down at
/// different speeds per column. The effect is drawn at very low opacity
/// (0.03-0.08) so it works as a subtle background layer.
///
/// The caller controls animation by passing a [time] value (seconds) that
/// advances each frame. The painter only draws when painted — no internal
/// timers or subscriptions.
class MatrixRainPainter extends CustomPainter {
  MatrixRainPainter({required this.time});

  /// Elapsed time in seconds, used to drive the animation.
  final double time;

  // Katakana (U+30A0..U+30FF) + digits + latin uppercase
  static const _chars =
      '\u30A2\u30A4\u30A6\u30A8\u30AA\u30AB\u30AD\u30AF\u30B1\u30B3'
      '\u30B5\u30B7\u30B9\u30BB\u30BD\u30BF\u30C1\u30C4\u30C6\u30C8'
      '\u30CA\u30CB\u30CC\u30CD\u30CE\u30CF\u30D2\u30D5\u30D8\u30DB'
      '\u30DE\u30DF\u30E0\u30E1\u30E2\u30E4\u30E6\u30E8\u30E9\u30EA'
      '\u30EB\u30EC\u30ED\u30EF\u30F2\u30F3'
      'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
      '0123456789';

  // Pre-generated column data (lazily initialized per column count).
  static int _cachedCols = 0;
  static List<double> _columnSpeeds = [];
  static List<double> _columnOffsets = [];

  static const _maxColumns = 40;

  @override
  void paint(Canvas canvas, Size size) {
    if (size.isEmpty) return;

    const charWidth = 14.0;
    const charHeight = 18.0;
    final rawCols = (size.width / charWidth).ceil();
    final cols = rawCols.clamp(1, _maxColumns);
    final rows = (size.height / charHeight).ceil();

    // (Re-)generate column data when column count changes.
    if (_cachedCols != cols) {
      final rng = Random(42);
      _columnSpeeds = List.generate(cols, (_) => 0.3 + rng.nextDouble() * 0.7);
      _columnOffsets = List.generate(cols, (_) => rng.nextDouble() * rows);
      _cachedCols = cols;
    }

    // Spread columns evenly when clamped below rawCols.
    final colSpacing = rawCols > _maxColumns ? size.width / cols : charWidth;

    final textPainter = TextPainter(textDirection: TextDirection.ltr);

    for (int col = 0; col < cols; col++) {
      final speed = _columnSpeeds[col];
      final offset = _columnOffsets[col];
      final currentRow = ((time * speed * 0.5 * rows + offset) % (rows + 12))
          .floor();

      for (int row = 0; row < rows; row++) {
        final distFromHead = currentRow - row;
        if (distFromHead < 0 || distFromHead > 14) continue;

        // Head character is brightest (0.35), tail fades to 0.05.
        final alpha = (1.0 - distFromHead / 14.0) * 0.35;
        if (alpha < 0.05) continue;

        final charIndex =
            ((col * 17 + row * 31 + (time * 8).floor()) % _chars.length).abs();
        final char = _chars[charIndex];

        textPainter.text = TextSpan(
          text: char,
          style: TextStyle(
            fontFamily: 'JetBrains Mono',
            fontSize: 12,
            color: matrixGreen.withValues(alpha: alpha),
          ),
        );
        textPainter.layout();
        textPainter.paint(canvas, Offset(col * colSpacing, row * charHeight));
      }
    }
  }

  @override
  bool shouldRepaint(MatrixRainPainter oldDelegate) => oldDelegate.time != time;
}
