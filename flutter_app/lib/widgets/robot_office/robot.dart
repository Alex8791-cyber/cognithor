import 'package:flutter/material.dart';

/// Mutable data class representing one robot agent in the office scene.
class Robot {
  Robot({
    required this.id,
    required this.name,
    required this.color,
    required this.eyeColor,
    required this.role,
    this.hasAntenna = false,
    required this.x,
    required this.y,
    double? targetX,
    double? targetY,
    this.state = RobotState.idle,
    this.stateTimer = 0,
    this.bobPhase = 0,
    this.blinkTimer = 0,
    this.typing = false,
    this.carrying = false,
    this.facing = 1,
    this.taskMsg = '',
    this.msgTimer = 0,
    this.emoji = '',
    this.emojiTimer = 0,
  })  : targetX = targetX ?? x,
        targetY = targetY ?? y;

  final String id;
  final String name;
  final Color color;
  final Color eyeColor;
  final String role;
  final bool hasAntenna;

  double x;
  double y;
  double targetX;
  double targetY;

  RobotState state;
  double stateTimer;
  double bobPhase;
  double blinkTimer;

  bool typing;
  bool carrying;
  int facing; // 1 = right, -1 = left

  String taskMsg;
  double msgTimer;

  String emoji;
  double emojiTimer;
}

enum RobotState { idle, walking, working }
