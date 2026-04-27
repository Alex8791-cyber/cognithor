/// Cognithor design system — centralized colors, spacing, typography.
/// Sci-Fi Command Center aesthetic with Cyberpunk-Neon palette.
library;

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

abstract final class CognithorTheme {
  // ── Primary Accents ─────────────────────────────────────
  static const _violet = Color(0xFF8B5CF6); // Neon Violett (main)
  static const _violetLight = Color(0xFFA78BFA);
  static const _gold = Color(0xFFFFD700); // Gold (branding)
  // ignore: unused_field
  static const _goldLight = Color(0xFFFFE44D);
  static const _matrix = Color(0xFF00FF41); // Matrix Green (code/data)

  // ── Section Colors — each screen has its own neon identity ──
  static const sectionChat = Color(0xFF00E5FF); // Electric Cyan
  static const sectionDashboard = Color(0xFF00FF41); // Neon Green
  static const sectionAdmin = Color(0xFF8B5CF6); // Neon Violett
  static const sectionIdentity = Color(0xFFFFD700); // Gold
  static const sectionSkills = Color(0xFFFF1493); // Neon Pink

  // ── Status Colors ───────────────────────────────────────
  static const _green = Color(0xFF00e676);
  static const _red = Color(0xFFff5252);
  static const _orange = Color(0xFFffab40);
  static const _blue = Color(0xFF448AFF);
  static const _purple = Color(0xFFB388FF);

  /// Updated by ThemeProvider — allows static getters to return theme-aware colors.
  static bool _isDark = true;

  /// Call when theme changes.
  static void setDarkMode(bool dark) {
    _isDark = dark;
  }

  // ── Dark Theme Surfaces ─────────────────────────────────
  static const _bg = Color(0xFF050510); // Deep space black
  static const _surface = Color(0xFF0A0F24); // Dark navy
  static const _surfaceHover = Color(0xFF101833); // Lighter navy
  static const _border = Color(0xFF1A2044); // Subtle blue border
  static const _borderHover = Color(0xFF2A3366);

  // ── Text ────────────────────────────────────────────────
  static const _text1 = Color(0xFFE8ECF4); // Primary (bright white-blue)
  static const _text2 = Color(0xFF8892B0); // Secondary
  static const _text3 = Color(0xFF4A5580); // Tertiary

  // ── Light Theme Surfaces ────────────────────────────────
  // ignore: unused_field
  static const _lightBg = Color(0xFFF5F5F8);
  static const _lightSurface = Color(0xFFFFFFFF);
  // ignore: unused_field
  static const _lightSurfaceHover = Color(0xFFF0F0F4);
  // ignore: unused_field
  static const _lightBorder = Color(0xFFE0E0E8);
  // ignore: unused_field
  static const _lightBorderHover = Color(0xFFD0D0DC);
  static const _lightText1 = Color(0xFF1A1A2E);
  static const _lightText2 = Color(0xFF6B6B80);
  // ignore: unused_field
  static const _lightText3 = Color(0xFF9999AA);
  static const _lightAccent = Color(0xFF8B5CF6); // Violet instead of blue

  // ── Public Color Accessors ────────────────────────────────
  static Color get accent => _violet;
  static Color get accentLight => _violetLight;
  static Color get accentDim => _violet.withValues(alpha: 0.7);
  static Color get green => _green;
  static Color get red => _red;
  static Color get orange => _orange;
  static Color get blue => _blue;
  static Color get purple => _purple;
  static Color get gold => _gold;
  static Color get matrix => _matrix;
  static Color get surface => _surface;
  static Color get surfaceHover => _surfaceHover;
  static Color get bg => _bg;
  static Color get border => _border;
  static Color get borderHover => _borderHover;
  static Color get textPrimary => _isDark ? _text1 : _lightText1;
  static Color get textSecondary => _isDark ? _text2 : _lightText2;
  static Color get textTertiary => _isDark ? _text3 : const Color(0xFF9090A0);

  // ── Semantic Aliases ────────────────────────────────────
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
    'concept': Color(0xFF8892B0),
    'unknown': Color(0xFF607D8B),
  };

  // ── Pipeline Phase Colors ─────────────────────────────────
  static const phaseColors = {
    'plan': Color(0xFF8B5CF6),
    'gate': Color(0xFFFFD700),
    'execute': Color(0xFF00FF41),
    'replan': Color(0xFFffab40),
    'complete': Color(0xFF00e676),
    'error': Color(0xFFff5252),
  };

  // ── Code Block Colors ─────────────────────────────────────
  static Color get codeBlockBg =>
      _isDark ? const Color(0xFF0A0F24) : const Color(0xFFF0F0F5);
  static Color get codeBlockBorder =>
      _isDark ? const Color(0xFF1A2044) : const Color(0xFFD8D8E4);

  // ── Component-Specific ────────────────────────────────────
  /// Semi-transparent accent for button backgrounds
  static Color get accentSurface => _violet.withValues(alpha: 0.18);

  /// Accent border (subtle)
  static Color get accentBorder => _violet.withValues(alpha: 0.25);

  /// Accent border hover
  static Color get accentBorderHover => _violet.withValues(alpha: 0.45);

  // ── Section Color Helpers ─────────────────────────────────
  /// Returns the neon color for a navigation tab index.
  static Color sectionColorFor(int tabIndex) {
    return switch (tabIndex) {
      0 => sectionChat,
      1 => sectionDashboard,
      2 => sectionSkills,
      3 => sectionAdmin,
      4 => sectionIdentity,
      _ => _violet,
    };
  }

  /// Section color names for display.
  static String sectionNameFor(int tabIndex) {
    return switch (tabIndex) {
      0 => 'Chat',
      1 => 'Dashboard',
      2 => 'Skills',
      3 => 'Admin',
      4 => 'Identity',
      _ => 'System',
    };
  }

  // ── Glass Decoration Factory ──────────────────────────────
  /// Returns a BoxDecoration suitable for glassmorphism panels.
  /// Use with ClipRRect + BackdropFilter for full glass effect.
  static BoxDecoration glassDecoration({
    Color? tint,
    double borderRadius = 16,
    bool glowBorder = false,
  }) {
    final color = tint ?? _violet;
    return BoxDecoration(
      color: color.withValues(alpha: 0.07),
      borderRadius: BorderRadius.circular(borderRadius),
      border: Border.all(
        color: color.withValues(alpha: glowBorder ? 0.3 : 0.12),
      ),
    );
  }

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

  // ── Mono Text Theme ───────────────────────────────────────
  /// Monospaced text theme for code and data displays.
  static TextTheme get monoTextTheme =>
      GoogleFonts.jetBrainsMonoTextTheme(ThemeData.dark().textTheme);

  // ── ThemeData ─────────────────────────────────────────────
  static ThemeData get light {
    final baseText = GoogleFonts.interTextTheme(ThemeData.light().textTheme);
    return ThemeData(
      brightness: Brightness.light,
      scaffoldBackgroundColor: const Color(0xFFF0F0F5),
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
      dividerColor: const Color(0xFFCCCCD8),
      cardTheme: CardThemeData(
        color: _lightSurface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(cardRadius),
          side: const BorderSide(color: Color(0xFFD8D8E4)),
        ),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: _lightSurface,
        foregroundColor: _lightText1,
        elevation: 0,
        scrolledUnderElevation: 1,
        shadowColor: Color(0x22000000),
        surfaceTintColor: Colors.transparent,
        shape: Border(bottom: BorderSide(color: Color(0xFFD8D8E4))),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: _lightSurface,
        hintStyle: const TextStyle(color: _lightText2),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(inputRadius),
          borderSide: const BorderSide(color: Color(0xFFBBBBCC)),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(inputRadius),
          borderSide: const BorderSide(color: Color(0xFFBBBBCC)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(inputRadius),
          borderSide: const BorderSide(color: _lightAccent, width: 2),
        ),
      ),
      listTileTheme: const ListTileThemeData(
        textColor: _lightText1,
        iconColor: _lightText2,
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
        titleMedium: baseText.titleMedium?.copyWith(
          color: _lightText1,
          fontWeight: FontWeight.w500,
        ),
        labelLarge: baseText.labelLarge?.copyWith(color: _lightText1),
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
          side: BorderSide(color: _lightAccent.withValues(alpha: 0.5)),
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
        primary: _violet,
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
          borderSide: const BorderSide(color: _violet),
        ),
      ),
      textTheme: baseText.copyWith(
        bodyLarge: baseText.bodyLarge?.copyWith(color: _text1, fontSize: 15),
        bodyMedium: baseText.bodyMedium?.copyWith(color: _text1, fontSize: 14),
        bodySmall: baseText.bodySmall?.copyWith(color: _text2, fontSize: 12),
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
          backgroundColor: _violet,
          foregroundColor: _bg,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(buttonRadius),
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: _violet,
          side: BorderSide(color: _violet.withValues(alpha: 0.3)),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(buttonRadius),
          ),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(foregroundColor: _violet),
      ),
    );
  }
}
