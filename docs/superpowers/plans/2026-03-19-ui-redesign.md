# Cognithor UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Flutter UI from functional-but-dull into a Sci-Fi Command Center with Cyberpunk-Neon aesthetics, Glassmorphism, and per-section neon color identities.

**Architecture:** Rewrite the theme system with section-aware colors, rebuild the navigation as a morphing sidebar with top command bar, redesign chat with immersive bubbles + context panel + hacker mode, turn the dashboard into a command center with real data in the Robot Office, and reorganize 18 config pages into 5 categories.

**Tech Stack:** Flutter 3.41, Provider, CustomPainter, BackdropFilter, AnimationController, Google Fonts (Inter + JetBrains Mono)

---

## Phase 1: Color System + Glassmorphism Foundation

### Task 1: Rewrite Theme System

**Files:**
- Modify: `lib/theme/jarvis_theme.dart`

- [ ] **Step 1: Replace color palette**

Replace the entire color system with the new Sci-Fi palette:

```dart
// Primary
static const _violet = Color(0xFF8B5CF6);    // Neon Violett
static const _gold = Color(0xFFFFD700);       // Gold
static const _matrix = Color(0xFF00FF41);     // Matrix Green

// Section colors
static const sectionChat = Color(0xFF00E5FF);       // Electric Cyan
static const sectionDashboard = Color(0xFF00FF41);   // Neon Green
static const sectionAdmin = Color(0xFF8B5CF6);       // Neon Violett
static const sectionIdentity = Color(0xFFFFD700);    // Gold
static const sectionSkills = Color(0xFFFF1493);      // Neon Pink

// Surfaces
static const _bg = Color(0xFF050510);         // Deep space black
static const _surface = Color(0xFF0A0F24);    // Dark navy
```

Add `sectionColorFor(int tabIndex)` method that returns the appropriate neon color for each main navigation tab.

Add `glassDecoration({Color? tint, double blur = 16})` factory that returns a BoxDecoration with BackdropFilter-compatible styling.

- [ ] **Step 2: Add JetBrains Mono font**

In `pubspec.yaml`, add `google_fonts` is already present. In the theme, add a `monoTextTheme` using `GoogleFonts.jetBrainsMonoTextTheme()`.

- [ ] **Step 3: Run flutter analyze**

Run: `flutter analyze`
Expected: No new issues

- [ ] **Step 4: Commit**

```bash
git add lib/theme/jarvis_theme.dart pubspec.yaml
git commit -m "feat(theme): sci-fi color system with section colors + glassmorphism"
```

### Task 2: GlassPanel Widget

**Files:**
- Create: `lib/widgets/glass_panel.dart`

- [ ] **Step 1: Create reusable glassmorphism container**

```dart
class GlassPanel extends StatelessWidget {
  const GlassPanel({
    super.key,
    required this.child,
    this.tint,           // section color tint
    this.borderRadius = 16,
    this.blur = 16,
    this.padding,
    this.glowOnHover = false,
  });
  // Uses ClipRRect + BackdropFilter + Container with semi-transparent bg
  // Border: tint color at 15% opacity
  // Background: tint color at 4% opacity
}
```

- [ ] **Step 2: Run flutter analyze**
- [ ] **Step 3: Commit**

### Task 3: NeonGlow Widget

**Files:**
- Create: `lib/widgets/neon_glow.dart`

- [ ] **Step 1: Create neon glow effect wrapper**

A widget that adds a colored glow/shadow around its child, with optional pulse animation:

```dart
class NeonGlow extends StatefulWidget {
  const NeonGlow({
    super.key,
    required this.child,
    required this.color,
    this.intensity = 0.3,
    this.pulse = false,     // breathing animation
    this.blurRadius = 12,
  });
}
```

- [ ] **Step 2: Commit**

---

## Phase 2: Morphing Sidebar + Command Bar

### Task 4: Section-Aware Navigation Provider

**Files:**
- Create: `lib/providers/navigation_provider.dart`

- [ ] **Step 1: Create navigation state provider**

```dart
class NavigationProvider extends ChangeNotifier {
  int _currentTab = 0;

  int get currentTab => _currentTab;
  Color get sectionColor => JarvisTheme.sectionColorFor(_currentTab);

  // Sidebar width varies by tab
  double get sidebarWidth => switch (_currentTab) {
    0 => 64,   // Chat — minimal
    1 => 48,   // Dashboard — hidden
    2 => 180,  // Skills
    3 => 260,  // Admin — expanded with sub-nav
    4 => 180,  // Identity
    _ => 180,
  };

  void setTab(int index) { _currentTab = index; notifyListeners(); }
}
```

- [ ] **Step 2: Register in main.dart MultiProvider**
- [ ] **Step 3: Commit**

### Task 5: Command Bar Widget

**Files:**
- Create: `lib/widgets/command_bar.dart`

- [ ] **Step 1: Create the top command bar (40px)**

Left: Screen name + section-colored icon
Center: Search trigger (Ctrl+K visual)
Right: Status dot + model name + token badge

Uses `GlassPanel` with section color tint. All text high-contrast white.

- [ ] **Step 2: Commit**

### Task 6: Rewrite Morphing Sidebar

**Files:**
- Modify: `lib/widgets/responsive_scaffold.dart`

- [ ] **Step 1: Replace current sidebar with morphing version**

The sidebar reads `NavigationProvider.sidebarWidth` and animates between widths (300ms easeOutQuart). Each nav item uses the section color for its glow state. Active item has neon pill + vertical accent line.

Logo at top has subtle breathing animation (scale 0.98-1.02, 4s cycle).

- [ ] **Step 2: Add command bar to scaffold**

Insert `CommandBar` above the content area in the scaffold layout.

- [ ] **Step 3: Update main_shell.dart**

Replace current NavigationBar-based shell with the new scaffold that uses `NavigationProvider`.

- [ ] **Step 4: Run flutter analyze**
- [ ] **Step 5: Commit**

---

## Phase 3: Immersive Chat Screen

### Task 7: Chat Bubble Redesign

**Files:**
- Modify: `lib/widgets/chat_bubble.dart`

- [ ] **Step 1: Redesign user bubbles**

Right-aligned, gradient from `sectionChat` to darker shade, subtle glow border, rounded corners. Slide-in animation on first appearance.

- [ ] **Step 2: Redesign assistant bubbles**

Left-aligned, `GlassPanel` with blur, left accent bar in `sectionChat` color. Markdown content with neon-bordered code blocks.

- [ ] **Step 3: Commit**

### Task 8: Waveform Typing Indicator

**Files:**
- Modify: `lib/widgets/typing_indicator.dart`

- [ ] **Step 1: Replace bouncing dots with waveform**

3 animated sine waves using `CustomPainter`, colored with `sectionChat`. Smooth oscillation at different frequencies.

- [ ] **Step 2: Commit**

### Task 9: Context Side Panel

**Files:**
- Create: `lib/widgets/chat/context_panel.dart`

- [ ] **Step 1: Create the context-aware side panel (350px)**

`GlassPanel` that appears on the right when `ChatProvider` has active tools, search results, or code output. Shows different content based on `activeTool`:
- `web_search` / `search_and_read`: Source cards with snippets
- `run_python` / `analyze_code`: Code output preview
- Other tools: Animated progress with tool name

- [ ] **Step 2: Integrate into chat_screen.dart**

Show/hide based on `ChatProvider.activeTool` state.

- [ ] **Step 3: Commit**

### Task 10: Hacker Mode

**Files:**
- Create: `lib/providers/hacker_mode_provider.dart`
- Create: `lib/widgets/chat/hacker_chat_view.dart`

- [ ] **Step 1: Create hacker mode provider**

Simple toggle: `bool hackerMode = false` with `SharedPreferences` persistence.

- [ ] **Step 2: Create hacker chat view**

Monospace JetBrains Mono font, green (`_matrix`) on black. Messages rendered as:
```
[07:44:16] USER > message text
[07:44:18] TOOL web_search {"query": "..."}
[07:44:20] ASST > response text
```

Subtle Matrix-style falling characters as background using `CustomPainter`.

- [ ] **Step 3: Add toggle button to chat AppBar**

Terminal icon that switches between normal and hacker view.

- [ ] **Step 4: Commit**

---

## Phase 4: Command Center Dashboard

### Task 11: Radial Gauge Widget

**Files:**
- Create: `lib/widgets/radial_gauge.dart`

- [ ] **Step 1: Create animated radial gauge**

`CustomPainter` that draws a circular arc with neon glow. Props: value (0-1), color, label, size. Animated value changes with `AnimatedCounter` logic.

- [ ] **Step 2: Commit**

### Task 12: Robot Office Real Data Integration

**Files:**
- Modify: `lib/widgets/robot_office/office_painter.dart`
- Modify: `lib/widgets/robot_office/robot_office_widget.dart`

- [ ] **Step 1: Add real data props to OfficePainter**

```dart
class OfficePainter extends CustomPainter {
  // ... existing props ...
  final double cpuUsage;      // 0-1, maps to server rack LED intensity
  final double memoryUsage;   // 0-1, maps to server rack color
  final int activePhase;      // 0-4, maps to kanban board highlight
  final double systemLoad;    // 0-1, maps to ceiling light brightness
}
```

- [ ] **Step 2: Wire real data from dashboard API**

In `RobotOfficeWidget`, accept optional monitoring data and pass to painter.

- [ ] **Step 3: Commit**

### Task 13: Dashboard Layout Redesign

**Files:**
- Modify: `lib/screens/dashboard_screen.dart`

- [ ] **Step 1: Rebuild as Command Center**

- Robot Office hero (50-60% height) with monitoring data overlay
- Below: 4 `RadialGauge` widgets in a row (CPU, Memory, Tokens, Response Time)
- Bottom: Scrolling event ticker with severity-colored entries
- All metric cards use `GlassPanel` with `sectionDashboard` tint

- [ ] **Step 2: Commit**

---

## Phase 5: Config Reorganization

### Task 14: Category-Based Config Screen

**Files:**
- Modify: `lib/screens/config_screen.dart`

- [ ] **Step 1: Replace 18-page flat list with 5 category tabs**

```dart
const _categories = [
  _Category('AI Engine', Icons.psychology, [
    'providers', 'models', 'planner', 'executor', 'prompts',
  ]),
  _Category('Channels', Icons.cell_tower, [
    'channels',
  ]),
  _Category('Knowledge', Icons.storage, [
    'memory', 'agents', 'bindings', 'web',
  ]),
  _Category('Security', Icons.shield, [
    'security', 'database',
  ]),
  _Category('System', Icons.settings, [
    'general', 'language', 'logging', 'cron', 'mcp', 'system',
  ]),
];
```

Top: 5 category tabs with section-violet underline.
Content: Current sidebar sub-page list within each category.
Save bar: `GlassPanel` floating at bottom with neon glow when dirty.

- [ ] **Step 2: Commit**

---

## Phase 6: Hacker Mode Toggle

Already covered in Task 10. This phase is for polish:

### Task 15: Matrix Rain Background Effect

**Files:**
- Create: `lib/widgets/chat/matrix_rain_painter.dart`

- [ ] **Step 1: Create Matrix-style falling characters**

`CustomPainter` that draws columns of random characters falling down, in green, at very low opacity (0.03-0.08). Only active when hacker mode is on.

- [ ] **Step 2: Commit**

---

## Phase 7: First-Run Wizard

### Task 16: Setup Wizard

**Files:**
- Create: `lib/screens/setup_wizard_screen.dart`

- [ ] **Step 1: Create 3-step wizard**

Step 1: Select LLM Provider (card selection from providers list)
Step 2: Configure Model (auto-detect available models from Ollama/API)
Step 3: Test Connection (send test message, show success/failure)

Full-screen overlay with animated step indicator. `GlassPanel` cards. Confetti on success.

- [ ] **Step 2: Trigger on first launch**

In `splash_screen.dart`, check for a `first_run` flag in SharedPreferences. If true, show wizard before MainShell.

- [ ] **Step 3: Commit**

---

## Phase 8: Polish + Particle Effects

### Task 17: Background Particles

**Files:**
- Modify: `lib/widgets/gradient_background.dart`

- [ ] **Step 1: Add floating particles**

Very subtle, low-density floating dots that drift slowly. Use section color at 5% opacity. Max 30 particles.

- [ ] **Step 2: Commit**

### Task 18: Final Sweep

**Files:**
- All screens

- [ ] **Step 1: Apply GlassPanel to all remaining cards**

Replace `JarvisCard` usage across screens with `GlassPanel` where appropriate. Ensure section color tints are applied.

- [ ] **Step 2: Apply NeonGlow to all interactive elements**

Buttons, toggles, active states — all get subtle neon glow in section color.

- [ ] **Step 3: Run full flutter analyze + build**

```bash
flutter analyze
flutter build web --release
```

- [ ] **Step 4: Final commit + version bump**

```bash
git add -A
git commit -m "feat: v0.47.0 — Sci-Fi Command Center UI redesign"
```
