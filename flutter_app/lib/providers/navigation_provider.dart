import 'package:flutter/material.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class NavigationProvider extends ChangeNotifier {
  int _currentTab = 0;

  int get currentTab => _currentTab;
  Color get sectionColor => CognithorTheme.sectionColorFor(_currentTab);
  String get sectionName => CognithorTheme.sectionNameFor(_currentTab);

  double get sidebarWidth => switch (_currentTab) {
    3 => 220, // Admin — slightly wider for sub-navigation
    _ => 180, // All tabs: consistent expanded sidebar
  };

  void setTab(int index) {
    if (index != _currentTab) {
      _currentTab = index;
      notifyListeners();
    }
  }
}
