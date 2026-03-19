import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Manages the hacker mode toggle state with SharedPreferences persistence.
class HackerModeProvider extends ChangeNotifier {
  static const _prefKey = 'hacker_mode_enabled';

  bool _enabled = false;
  bool get enabled => _enabled;

  HackerModeProvider() {
    _load();
  }

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    _enabled = prefs.getBool(_prefKey) ?? false;
    notifyListeners();
  }

  Future<void> toggle() async {
    _enabled = !_enabled;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_prefKey, _enabled);
  }
}
