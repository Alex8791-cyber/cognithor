import 'package:flutter/material.dart';
import 'package:cognithor_ui/services/voice_service.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';

class VoiceIndicator extends StatefulWidget {
  const VoiceIndicator({super.key, required this.state});

  final VoiceState state;

  @override
  State<VoiceIndicator> createState() => _VoiceIndicatorState();
}

class _VoiceIndicatorState extends State<VoiceIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulseCtrl;
  late final Animation<double> _pulse;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    );
    _pulse = Tween<double>(
      begin: 1.0,
      end: 1.4,
    ).animate(CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut));
    _updateAnimation();
  }

  @override
  void didUpdateWidget(VoiceIndicator old) {
    super.didUpdateWidget(old);
    if (old.state != widget.state) _updateAnimation();
  }

  void _updateAnimation() {
    if (widget.state == VoiceState.listening ||
        widget.state == VoiceState.conversation) {
      _pulseCtrl.repeat(reverse: true);
    } else {
      _pulseCtrl.stop();
      _pulseCtrl.value = 0;
    }
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  Color get _color => switch (widget.state) {
    VoiceState.off => CognithorTheme.textSecondary,
    VoiceState.listening => CognithorTheme.orange,
    VoiceState.conversation => CognithorTheme.green,
    VoiceState.processing => CognithorTheme.accent,
    VoiceState.speaking => CognithorTheme.accent,
  };

  String get _label => switch (widget.state) {
    VoiceState.off => 'Off',
    VoiceState.listening => 'Listening...',
    VoiceState.conversation => 'Speak now',
    VoiceState.processing => 'Processing...',
    VoiceState.speaking => 'Speaking...',
  };

  IconData get _icon => switch (widget.state) {
    VoiceState.off => Icons.mic_off,
    VoiceState.listening => Icons.hearing,
    VoiceState.conversation => Icons.mic,
    VoiceState.processing => Icons.hourglass_top,
    VoiceState.speaking => Icons.volume_up,
  };

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _pulse,
      builder: (context, child) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: _color.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: _color.withValues(alpha: 0.3)),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Transform.scale(
                scale: _pulse.value,
                child: Icon(_icon, size: 16, color: _color),
              ),
              const SizedBox(width: 6),
              Text(_label, style: TextStyle(color: _color, fontSize: 12)),
            ],
          ),
        );
      },
    );
  }
}
