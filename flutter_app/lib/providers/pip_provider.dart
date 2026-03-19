import 'package:flutter/foundation.dart';

/// Controls the visibility and state of the Robot Office Picture-in-Picture
/// overlay from anywhere in the widget tree.
class PipProvider extends ChangeNotifier {
  bool _visible = true;
  bool _fullscreenOnDashboard = false;

  /// Whether the PiP overlay is visible.
  bool get visible => _visible;

  /// Whether the dashboard should show the Robot Office inline (fullscreen)
  /// instead of using the PiP overlay.
  bool get fullscreenOnDashboard => _fullscreenOnDashboard;

  void show() {
    _visible = true;
    notifyListeners();
  }

  void hide() {
    _visible = false;
    notifyListeners();
  }

  void toggle() {
    _visible = !_visible;
    notifyListeners();
  }

  /// Switch to fullscreen mode on the dashboard (hides PiP, shows inline).
  void enterFullscreen() {
    _fullscreenOnDashboard = true;
    _visible = false;
    notifyListeners();
  }

  /// Switch back to PiP mode (hides inline, shows PiP).
  void exitFullscreen() {
    _fullscreenOnDashboard = false;
    _visible = true;
    notifyListeners();
  }
}
