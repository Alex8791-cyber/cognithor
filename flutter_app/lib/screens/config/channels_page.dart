import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cognithor_ui/l10n/generated/app_localizations.dart';
import 'package:cognithor_ui/providers/config_provider.dart';
import 'package:cognithor_ui/theme/cognithor_theme.dart';
import 'package:cognithor_ui/widgets/form/form_widgets.dart';

class ChannelsPage extends StatelessWidget {
  const ChannelsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ConfigProvider>(
      builder: (context, cfg, _) {
        final ch = cfg.cfg['channels'] as Map<String, dynamic>? ?? {};

        // Quick-toggle grid for all channels
        final channelDefs = [
          ('cli', 'CLI', Icons.terminal),
          ('webui', 'Web UI', Icons.web),
          ('telegram', 'Telegram', Icons.telegram),
          ('slack', 'Slack', Icons.tag),
          ('discord', 'Discord', Icons.discord),
          ('whatsapp', 'WhatsApp', Icons.chat),
          ('signal', 'Signal', Icons.lock),
          ('matrix', 'Matrix', Icons.grid_view),
          ('teams', 'Teams', Icons.groups),
          ('imessage', 'iMessage', Icons.message),
          ('google_chat', 'Google Chat', Icons.chat_bubble),
          ('mattermost', 'Mattermost', Icons.forum),
          ('feishu', 'Feishu', Icons.business),
          ('irc', 'IRC', Icons.tag),
          ('twitch', 'Twitch', Icons.live_tv),
        ];

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // Compact toggle grid
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Text(
                AppLocalizations.of(context).channelToggles,
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: channelDefs.map((def) {
                final (key, label, icon) = def;
                final enabled = ch['${key}_enabled'] == true;
                return _CompactChannelToggle(
                  label: label,
                  icon: icon,
                  enabled: enabled,
                  onChanged: (v) => cfg.set('channels.${key}_enabled', v),
                );
              }).toList(),
            ),
            const SizedBox(height: 24),
            // Detailed config per channel (collapsible)
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Text(
                AppLocalizations.of(context).channelSettings,
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ),
            _channelCard(
              cfg,
              ch,
              'webui',
              'Web UI',
              Icons.web,
              extra: [
                CognithorNumberField(
                  label: 'Port',
                  value: (ch['webui_port'] as num?) ?? 8741,
                  onChanged: (v) => cfg.set('channels.webui_port', v),
                  min: 1024,
                  max: 65535,
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'telegram',
              'Telegram',
              Icons.telegram,
              extra: [
                CognithorListField(
                  label: 'Whitelist',
                  value: _toStringList(ch['telegram_whitelist']),
                  onChanged: (v) => cfg.set('channels.telegram_whitelist', v),
                  placeholder: 'User ID',
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'slack',
              'Slack',
              Icons.tag,
              extra: [
                CognithorTextField(
                  label: 'Default Channel',
                  value: (ch['slack_default_channel'] ?? '').toString(),
                  onChanged: (v) =>
                      cfg.set('channels.slack_default_channel', v),
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'discord',
              'Discord',
              Icons.discord,
              extra: [
                CognithorTextField(
                  label: 'Channel ID',
                  value: (ch['discord_channel_id'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.discord_channel_id', v),
                  description: 'Stored as string to prevent precision loss',
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'whatsapp',
              'WhatsApp',
              Icons.chat,
              extra: [
                CognithorTextField(
                  label: 'Default Chat',
                  value: (ch['whatsapp_default_chat'] ?? '').toString(),
                  onChanged: (v) =>
                      cfg.set('channels.whatsapp_default_chat', v),
                ),
                CognithorTextField(
                  label: 'Phone Number ID',
                  value: (ch['whatsapp_phone_number_id'] ?? '').toString(),
                  onChanged: (v) =>
                      cfg.set('channels.whatsapp_phone_number_id', v),
                ),
                CognithorNumberField(
                  label: 'Webhook Port',
                  value: (ch['whatsapp_webhook_port'] as num?) ?? 8742,
                  onChanged: (v) =>
                      cfg.set('channels.whatsapp_webhook_port', v),
                  min: 1024,
                ),
                CognithorTextField(
                  label: 'Verify Token',
                  value: (ch['whatsapp_verify_token'] ?? '').toString(),
                  onChanged: (v) =>
                      cfg.set('channels.whatsapp_verify_token', v),
                  isPassword: true,
                ),
                CognithorListField(
                  label: 'Allowed Numbers',
                  value: _toStringList(ch['whatsapp_allowed_numbers']),
                  onChanged: (v) =>
                      cfg.set('channels.whatsapp_allowed_numbers', v),
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'signal',
              'Signal',
              Icons.lock,
              extra: [
                CognithorTextField(
                  label: 'Default User',
                  value: (ch['signal_default_user'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.signal_default_user', v),
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'matrix',
              'Matrix',
              Icons.grid_view,
              extra: [
                CognithorTextField(
                  label: 'Homeserver',
                  value: (ch['matrix_homeserver'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.matrix_homeserver', v),
                ),
                CognithorTextField(
                  label: 'User ID',
                  value: (ch['matrix_user_id'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.matrix_user_id', v),
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'teams',
              'Teams',
              Icons.groups,
              extra: [
                CognithorTextField(
                  label: 'Default Channel',
                  value: (ch['teams_default_channel'] ?? '').toString(),
                  onChanged: (v) =>
                      cfg.set('channels.teams_default_channel', v),
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'imessage',
              'iMessage',
              Icons.message,
              extra: [
                CognithorTextField(
                  label: 'Device ID',
                  value: (ch['imessage_device_id'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.imessage_device_id', v),
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'google_chat',
              'Google Chat',
              Icons.chat_bubble,
              extra: [
                CognithorTextField(
                  label: 'Credentials Path',
                  value: (ch['google_chat_credentials_path'] ?? '').toString(),
                  onChanged: (v) =>
                      cfg.set('channels.google_chat_credentials_path', v),
                ),
                CognithorListField(
                  label: 'Allowed Spaces',
                  value: _toStringList(ch['google_chat_allowed_spaces']),
                  onChanged: (v) =>
                      cfg.set('channels.google_chat_allowed_spaces', v),
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'mattermost',
              'Mattermost',
              Icons.forum,
              extra: [
                CognithorTextField(
                  label: 'URL',
                  value: (ch['mattermost_url'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.mattermost_url', v),
                ),
                CognithorTextField(
                  label: 'Token',
                  value: (ch['mattermost_token'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.mattermost_token', v),
                  isPassword: true,
                ),
                CognithorTextField(
                  label: 'Channel',
                  value: (ch['mattermost_channel'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.mattermost_channel', v),
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'feishu',
              'Feishu',
              Icons.business,
              extra: [
                CognithorTextField(
                  label: 'App ID',
                  value: (ch['feishu_app_id'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.feishu_app_id', v),
                ),
                CognithorTextField(
                  label: 'App Secret',
                  value: (ch['feishu_app_secret'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.feishu_app_secret', v),
                  isPassword: true,
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'irc',
              'IRC',
              Icons.tag,
              extra: [
                CognithorTextField(
                  label: 'Server',
                  value: (ch['irc_server'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.irc_server', v),
                ),
                CognithorNumberField(
                  label: 'Port',
                  value: (ch['irc_port'] as num?) ?? 6667,
                  onChanged: (v) => cfg.set('channels.irc_port', v),
                ),
                CognithorTextField(
                  label: 'Nick',
                  value: (ch['irc_nick'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.irc_nick', v),
                ),
                CognithorListField(
                  label: 'Channels',
                  value: _toStringList(ch['irc_channels']),
                  onChanged: (v) => cfg.set('channels.irc_channels', v),
                ),
              ],
            ),
            _channelCard(
              cfg,
              ch,
              'twitch',
              'Twitch',
              Icons.live_tv,
              extra: [
                CognithorTextField(
                  label: 'Token',
                  value: (ch['twitch_token'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.twitch_token', v),
                  isPassword: true,
                ),
                CognithorTextField(
                  label: 'Channel',
                  value: (ch['twitch_channel'] ?? '').toString(),
                  onChanged: (v) => cfg.set('channels.twitch_channel', v),
                ),
                CognithorListField(
                  label: 'Allowed Users',
                  value: _toStringList(ch['twitch_allowed_users']),
                  onChanged: (v) => cfg.set('channels.twitch_allowed_users', v),
                ),
              ],
            ),
            const Divider(height: 32),
            // Voice config
            CognithorCollapsibleCard(
              title: 'Voice',
              icon: Icons.mic,
              children: [
                CognithorToggleField(
                  label: 'Voice Enabled',
                  value: ch['voice_enabled'] == true,
                  onChanged: (v) => cfg.set('channels.voice_enabled', v),
                ),
                ..._buildVoiceConfig(cfg, ch),
              ],
            ),
          ],
        );
      },
    );
  }

  Widget _channelCard(
    ConfigProvider cfg,
    Map<String, dynamic> ch,
    String key,
    String label,
    IconData icon, {
    List<Widget> extra = const [],
  }) {
    final enabledKey = '${key}_enabled';
    return CognithorCollapsibleCard(
      title: label,
      icon: icon,
      badge: ch[enabledKey] == true ? 'ON' : null,
      children: [
        CognithorToggleField(
          label: 'Enabled',
          value: ch[enabledKey] == true,
          onChanged: (v) => cfg.set('channels.$enabledKey', v),
        ),
        ...extra,
      ],
    );
  }

  List<Widget> _buildVoiceConfig(ConfigProvider cfg, Map<String, dynamic> ch) {
    final vc = ch['voice_config'] as Map<String, dynamic>? ?? {};
    return [
      CognithorSelectField.fromStrings(
        label: 'TTS Backend',
        value: (vc['tts_backend'] ?? 'piper').toString(),
        options: const ['piper', 'espeak', 'elevenlabs'],
        onChanged: (v) => cfg.set('channels.voice_config.tts_backend', v),
      ),
      CognithorTextField(
        label: 'Piper Voice',
        value: (vc['piper_voice'] ?? 'de_DE-pavoque-low').toString(),
        onChanged: (v) => cfg.set('channels.voice_config.piper_voice', v),
      ),
      CognithorSliderField(
        label: 'Piper Length Scale',
        value: (vc['piper_length_scale'] as num?)?.toDouble() ?? 1.0,
        onChanged: (v) =>
            cfg.set('channels.voice_config.piper_length_scale', v),
        min: 0.5,
        max: 2.0,
        step: 0.1,
      ),
      CognithorTextField(
        label: 'ElevenLabs API Key',
        value: (vc['elevenlabs_api_key'] ?? '').toString(),
        onChanged: (v) =>
            cfg.set('channels.voice_config.elevenlabs_api_key', v),
        isPassword: true,
        isSecret: true,
      ),
      CognithorTextField(
        label: 'ElevenLabs Voice ID',
        value: (vc['elevenlabs_voice_id'] ?? '').toString(),
        onChanged: (v) =>
            cfg.set('channels.voice_config.elevenlabs_voice_id', v),
      ),
      CognithorToggleField(
        label: 'Wake Word Enabled',
        value: vc['wake_word_enabled'] == true,
        onChanged: (v) => cfg.set('channels.voice_config.wake_word_enabled', v),
      ),
      CognithorTextField(
        label: 'Wake Word',
        value: (vc['wake_word'] ?? 'jarvis').toString(),
        onChanged: (v) => cfg.set('channels.voice_config.wake_word', v),
      ),
      CognithorSelectField.fromStrings(
        label: 'Wake Word Backend',
        value: (vc['wake_word_backend'] ?? 'browser').toString(),
        options: const ['browser', 'vosk', 'porcupine'],
        onChanged: (v) => cfg.set('channels.voice_config.wake_word_backend', v),
      ),
      CognithorToggleField(
        label: 'Talk Mode',
        value: vc['talk_mode_enabled'] == true,
        onChanged: (v) => cfg.set('channels.voice_config.talk_mode_enabled', v),
      ),
      CognithorToggleField(
        label: 'Auto-Listen',
        value: vc['talk_mode_auto_listen'] == true,
        onChanged: (v) =>
            cfg.set('channels.voice_config.talk_mode_auto_listen', v),
      ),
    ];
  }

  static List<String> _toStringList(dynamic v) {
    if (v is List) return v.map((e) => e.toString()).toList();
    return [];
  }
}

/// Compact channel toggle chip: icon + name + switch in a single row.
class _CompactChannelToggle extends StatelessWidget {
  const _CompactChannelToggle({
    required this.label,
    required this.icon,
    required this.enabled,
    required this.onChanged,
  });

  final String label;
  final IconData icon;
  final bool enabled;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    final accentColor = enabled
        ? CognithorTheme.accent
        : (isDark ? CognithorTheme.textSecondary : const Color(0xFF9999AA));

    return Container(
      width: 180,
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: enabled
            ? CognithorTheme.accent.withValues(alpha: isDark ? 0.12 : 0.06)
            : theme.cardColor,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: enabled
              ? CognithorTheme.accent.withValues(alpha: 0.3)
              : theme.dividerColor,
          width: 1.0,
        ),
      ),
      child: Row(
        children: [
          Icon(icon, size: 16, color: accentColor),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              label,
              style: TextStyle(
                fontSize: 13,
                fontWeight: enabled ? FontWeight.w600 : FontWeight.normal,
                color: enabled
                    ? (isDark
                          ? CognithorTheme.textPrimary
                          : const Color(0xFF1A1A2E))
                    : (isDark
                          ? CognithorTheme.textSecondary
                          : const Color(0xFF6B6B80)),
              ),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ),
          SizedBox(
            height: 24,
            child: FittedBox(
              child: Switch.adaptive(
                value: enabled,
                onChanged: onChanged,
                activeTrackColor: CognithorTheme.accent,
                activeThumbColor: Colors.white,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
