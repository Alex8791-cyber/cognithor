/// Jarvis design system — centralized colors, spacing, typography.
library;

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

abstract final class JarvisTheme {
  // ── Core Palette ──────────────────────────────────────────
  static const _accent = Color(0xFF00E5FF);
  static const _accentLight = Color(0xFF4DF0FF);
  static const _accentDim = Color(0xFF00B8D4);

  // Status colors
  static const _green = Color(0xFF00e676);
  static const _red = Color(0xFFff5252);
  static const _orange = Color(0xFFffab40);
  static const _blue = Color(0xFF448AFF);
  static const _purple = Color(0xFFB388FF);

  // Dark theme surfaces
  static const _bg = Color(0xFF080B16);
  static const _surface = Color(0xFF0F1629);
  static const _surfaceHover = Color(0xFF161D34);
  static const _border = Color(0xFF1C2340);
  static const _borderHover = Color(0xFF2A3358);
  static const _text1 = Color(0xFFe0e0e8);
  static const _text2 = Color(0xFF8888a0);
  static const _text3 = Color(0xFF555568);

  // Light theme surfaces
  static const _lightBg = Color(0xFFF5F5F8);
  static const _lightSurface = Color(0xFFFFFFFF);
  // ignore: unused_field
  static const _lightSurfaceHover = Color(0xFFF0F0F4);
  static const _lightBorder = Color(0xFFE0E0E8);
  // ignore: unused_field
  static const _lightBorderHover = Color(0xFFD0D0DC);
  static const _lightText1 = Color(0xFF1A1A2E);
  static const _lightText2 = Color(0xFF6B6B80);
  // ignore: unused_field
  static const _lightText3 = Color(0xFF9999AA);
  static const _lightAccent = Color(0xFF0077CC);

  // ── Public Color Accessors ────────────────────────────────
  static Color get accent => _accent;
  static Color get accentLight => _accentLight;
  static Color get accentDim => _accentDim;
  static Color get green => _green;
  static Color get red => _red;
  static Color get orange => _orange;
  static Color get blue => _blue;
  static Color get purple => _purple;
  static Color get surface => _surface;
  static Color get surfaceHover => _surfaceHover;
  static Color get bg => _bg;
  static Color get border => _border;
  static Color get borderHover => _borderHover;
  static Color get textPrimary => _text1;
  static Color get textSecondary => _text2;
  static Color get textTertiary => _text3;

  // Semantic aliases
  static Color get success => _green;
  static Color get error => _red;
  static Color get warning => _orange;
  static const info = Color(0xFF448AFF);

  // ── Entity Type Colors (Knowledge Graph) ──────────────────
  static const entityColors = {
    'person': Color(0xFF448AFF),
    'organization': Color(0xFFB388FF),
    'location': Color(0xFF00e676),
    'product': Color(0xFFffab40),
    'concept': Color(0xFF8888a0),
    'unknown': Color(0xFF607D8B),
  };

  // ── Pipeline Phase Colors ─────────────────────────────────
  static const phaseColors = {
    'plan': Color(0xFF00d4ff),
    'gate': Color(0xFFffab40),
    'execute': Color(0xFF00e676),
    'replan': Color(0xFFffab40),
    'complete': Color(0xFF00e676),
    'error': Color(0xFFff5252),
  };

  // ── Code Block Colors ─────────────────────────────────────
  static const codeBlockBg = Color(0xFF12121E);
  static const codeBlockBorder = Color(0xFF2A2A40);

  // ── Component-Specific ────────────────────────────────────
  /// Semi-transparent accent for button backgrounds (Hermes-style)
  static Color get accentSurface => _accent.withValues(alpha: 0.12);

  /// Accent border (subtle)
  static Color get accentBorder => _accent.withValues(alpha: 0.15);

  /// Accent border hover
  static Color get accentBorderHover => _accent.withValues(alpha: 0.3);

  // ── Spacing ───────────────────────────────────────────────
  static const spacing = 16.0;
  static const spacingSm = 8.0;
  static const spacingXs = 4.0;
  static const spacingLg = 24.0;
  static const spacingXl = 32.0;

  // ── Border Radii ──────────────────────────────────────────
  static const cardRadius = 12.0;
  static const buttonRadius = 8.0;
  static const chipRadius = 20.0;
  static const inputRadius = 12.0;

  // ── Icon Sizes ────────────────────────────────────────────
  static const iconSizeSm = 16.0;
  static const iconSizeMd = 24.0;
  static const iconSizeLg = 32.0;
  static const iconSizeXl = 48.0;

  // ── Animations ────────────────────────────────────────────
  static const animDuration = Duration(milliseconds: 250);
  static const animDurationFast = Duration(milliseconds: 150);
  static const animDurationSlow = Duration(milliseconds: 400);
  static const animCurve = Curves.easeOutQuart;

  // ── ThemeData ─────────────────────────────────────────────
  static ThemeData get light {
    final baseText = GoogleFonts.interTextTheme(ThemeData.light().textTheme);
    return ThemeData(
      brightness: Brightness.light,
      scaffoldBackgroundColor: _lightBg,
      colorScheme: const ColorScheme.light(
        primary: _lightAccent,
        secondary: _green,
        surface: _lightSurface,
        error: _red,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onSurface: _lightText1,
        onError: Colors.white,
      ),
      cardColor: _lightSurface,
      dividerColor: _lightBorder,
      appBarTheme: const AppBarTheme(
        backgroundColor: _lightSurface,
        foregroundColor: _lightText1,
        elevation: 0,
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: _lightSurface,
        hintStyle: const TextStyle(color: _lightText2),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(inputRadius),
          borderSide: const BorderSide(color: _lightBorder),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(inputRadius),
          borderSide: const BorderSide(color: _lightBorder),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(inputRadius),
          borderSide: const BorderSide(color: _lightAccent),
        ),
      ),
      textTheme: baseText.copyWith(
        bodyLarge: baseText.bodyLarge?.copyWith(
          color: _lightText1,
          fontSize: 15,
        ),
        bodyMedium: baseText.bodyMedium?.copyWith(
          color: _lightText1,
          fontSize: 14,
        ),
        bodySmall: baseText.bodySmall?.copyWith(
          color: _lightText2,
          fontSize: 12,
        ),
        titleLarge: baseText.titleLarge?.copyWith(
          color: _lightText1,
          fontSize: 20,
          fontWeight: FontWeight.w600,
          letterSpacing: -0.3,
        ),
      ),
      iconTheme: const IconThemeData(color: _lightText2),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: _lightAccent,
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(buttonRadius),
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: _lightAccent,
          side: BorderSide(color: _lightAccent.withValues(alpha: 0.3)),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(buttonRadius),
          ),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(foregroundColor: _lightAccent),
      ),
    );
  }

  static ThemeData get dark {
    final baseText = GoogleFonts.interTextTheme(ThemeData.dark().textTheme);
    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: _bg,
      colorScheme: const ColorScheme.dark(
        primary: _accent,
        secondary: _green,
        surface: _surface,
        error: _red,
        onPrimary: _bg,
        onSecondary: _bg,
        onSurface: _text1,
        onError: Colors.white,
      ),
      cardColor: _surface,
      dividerColor: _border,
      appBarTheme: const AppBarTheme(
        backgroundColor: _surface,
        foregroundColor: _text1,
        elevation: 0,
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: _surface,
        hintStyle: const TextStyle(color: _text2),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(inputRadius),
          borderSide: const BorderSide(color: _border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(inputRadius),
          borderSide: const BorderSide(color: _border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(inputRadius),
          borderSide: const BorderSide(color: _accent),
        ),
      ),
      textTheme: baseText.copyWith(
        bodyLarge: baseText.bodyLarge?.copyWith(
          color: _text1,
          fontSize: 15,
        ),
        bodyMedium: baseText.bodyMedium?.copyWith(
          color: _text1,
          fontSize: 14,
        ),
        bodySmall: baseText.bodySmall?.copyWith(
          color: _text2,
          fontSize: 12,
        ),
        titleLarge: baseText.titleLarge?.copyWith(
          color: _text1,
          fontSize: 20,
          fontWeight: FontWeight.w600,
          letterSpacing: -0.3,
        ),
      ),
      iconTheme: const IconThemeData(color: _text2),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: _accent,
          foregroundColor: _bg,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(buttonRadius),
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: _accent,
          side: BorderSide(color: _accent.withValues(alpha: 0.3)),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(buttonRadius),
          ),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(foregroundColor: _accent),
      ),
    );
  }
}
