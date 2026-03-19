import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/providers/chat_provider.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/providers/voice_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/approval_dialog.dart';
import 'package:jarvis_ui/widgets/canvas_panel.dart';
import 'package:jarvis_ui/widgets/chat_bubble.dart';
import 'package:jarvis_ui/widgets/chat_input.dart';
import 'package:jarvis_ui/widgets/jarvis_empty_state.dart';
import 'package:jarvis_ui/widgets/message_actions.dart';
import 'package:jarvis_ui/widgets/observe/observe_panel.dart';
import 'package:jarvis_ui/widgets/pipeline_indicator.dart';
import 'package:jarvis_ui/widgets/plan_detail_panel.dart';
import 'package:jarvis_ui/widgets/tool_indicator.dart';
import 'package:jarvis_ui/widgets/typing_indicator.dart';
import 'package:jarvis_ui/widgets/voice_indicator.dart';
import 'package:jarvis_ui/screens/teach_screen.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  late ChatProvider _chat;
  final _scrollController = ScrollController();
  bool _initialized = false;
  bool _showObserve = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initialized) {
      final conn = context.read<ConnectionProvider>();
      final sessionId = 'flutter_${DateTime.now().millisecondsSinceEpoch}';
      _chat = ChatProvider(ws: conn.ws);
      conn.ws.connect(sessionId);
      _initialized = true;
    }
  }

  @override
  void dispose() {
    final conn = context.read<ConnectionProvider>();
    conn.ws.disconnect();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.of(context);

    return ChangeNotifierProvider.value(
      value: _chat,
      child: Scaffold(
        appBar: _buildAppBar(l),
        body: Row(
          children: [
            // Main chat area
            Expanded(
              child: Column(
                children: [
                  // Pipeline indicator
                  Consumer<ChatProvider>(
                    builder: (context, chat, _) {
                      if (chat.pipeline.isEmpty) {
                        return const SizedBox.shrink();
                      }
                      return PipelineIndicator(phases: chat.pipeline);
                    },
                  ),

                  // Voice indicator
                  Consumer<VoiceProvider>(
                    builder: (context, voice, _) {
                      if (!voice.isActive) return const SizedBox.shrink();
                      return Padding(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16, vertical: 4),
                        child: VoiceIndicator(state: voice.state),
                      );
                    },
                  ),

                  // Plan detail panel
                  Consumer<ChatProvider>(
                    builder: (context, chat, _) {
                      if (chat.planDetail == null) {
                        return const SizedBox.shrink();
                      }
                      return PlanDetailPanel(
                        plan: chat.planDetail!,
                        onClose: chat.dismissPlan,
                      );
                    },
                  ),

                  // Messages list
                  Expanded(
                    child: Consumer<ChatProvider>(
                      builder: (context, chat, _) {
                        _scrollToBottom();

                        if (chat.messages.isEmpty && !chat.isStreaming) {
                          return JarvisEmptyState(
                            icon: Icons.chat_bubble_outline,
                            title: l.startConversation,
                            subtitle: l.typeMessage,
                          );
                        }

                        final itemCount = chat.messages.length +
                            (chat.isStreaming ? 1 : 0) +
                            (_showTyping(chat) ? 1 : 0);

                        return ListView.builder(
                          controller: _scrollController,
                          padding: const EdgeInsets.symmetric(
                            horizontal: JarvisTheme.spacing,
                            vertical: JarvisTheme.spacingSm,
                          ),
                          itemCount: itemCount,
                          itemBuilder: (context, index) {
                            if (_showTyping(chat) && index == itemCount - 1) {
                              return const TypingIndicator();
                            }

                            if (chat.isStreaming &&
                                index == chat.messages.length) {
                              return ChatBubble(
                                role: MessageRole.assistant,
                                text: chat.streamingText,
                                isStreaming: true,
                              );
                            }

                            final msg = chat.messages[index];
                            return MessageActions(
                              text: msg.text,
                              child: ChatBubble(
                                role: msg.role,
                                text: msg.text,
                              ),
                            );
                          },
                        );
                      },
                    ),
                  ),

                  // Canvas panel
                  Consumer<ChatProvider>(
                    builder: (context, chat, _) {
                      if (chat.canvasHtml == null) {
                        return const SizedBox.shrink();
                      }
                      return CanvasPanel(
                        html: chat.canvasHtml!,
                        title: chat.canvasTitle,
                        onClose: chat.dismissCanvas,
                      );
                    },
                  ),

                  // Tool indicator
                  Consumer<ChatProvider>(
                    builder: (context, chat, _) {
                      if (chat.activeTool == null &&
                          chat.statusText.isEmpty) {
                        return const SizedBox.shrink();
                      }
                      return ToolIndicator(
                        tool: chat.activeTool,
                        status: chat.statusText,
                      );
                    },
                  ),

                  // Approval banner
                  Consumer<ChatProvider>(
                    builder: (context, chat, _) {
                      if (chat.pendingApproval == null) {
                        return const SizedBox.shrink();
                      }
                      return ApprovalDialog(
                        request: chat.pendingApproval!,
                        onRespond: chat.respondApproval,
                      );
                    },
                  ),

                  // Input bar
                  Consumer<ChatProvider>(
                    builder: (context, chat, _) {
                      return ChatInput(
                        onSend: (text) {
                          chat.sendMessage(text);
                          // Also forward voice transcripts
                          final voice = context.read<VoiceProvider>();
                          if (voice.isActive) {
                            voice.stop();
                          }
                        },
                        onCancel: chat.cancelOperation,
                        onFile: chat.sendFile,
                        isProcessing:
                            chat.isStreaming || chat.activeTool != null,
                      );
                    },
                  ),
                ],
              ),
            ),

            // Observe panel
            if (_showObserve)
              Consumer<ChatProvider>(
                builder: (context, chat, _) {
                  return ObservePanel(
                    agentLog: chat.agentLog,
                    planDetail: chat.planDetail,
                    pipelineState: chat.pipeline
                        .map((p) => {
                              'phase': p.phase,
                              'status': p.status,
                              'elapsed_ms': p.elapsedMs,
                            })
                        .toList(),
                    onClose: () => setState(() => _showObserve = false),
                  );
                },
              ),
          ],
        ),
      ),
    );
  }

  bool _showTyping(ChatProvider chat) {
    return chat.statusText.isNotEmpty &&
        !chat.isStreaming &&
        chat.activeTool == null;
  }

  PreferredSizeWidget _buildAppBar(AppLocalizations l) {
    return AppBar(
      title: Text(l.appTitle),
      actions: [
        // Voice toggle
        Consumer<VoiceProvider>(
          builder: (context, voice, _) {
            return IconButton(
              icon: Icon(
                voice.isActive ? Icons.mic : Icons.mic_none,
                color: voice.isActive ? JarvisTheme.green : null,
              ),
              tooltip: l.voiceMode,
              onPressed: () => voice.toggle(),
            );
          },
        ),
        // Teach Cognithor
        IconButton(
          icon: const Icon(Icons.auto_stories),
          tooltip: l.teachCognithor,
          onPressed: () {
            Navigator.of(context).push(
              MaterialPageRoute<void>(
                builder: (_) => const TeachScreen(),
              ),
            );
          },
        ),
        // Observe panel toggle
        IconButton(
          icon: Icon(
            Icons.analytics_outlined,
            color: _showObserve ? JarvisTheme.accent : null,
          ),
          tooltip: 'Observe',
          onPressed: () => setState(() => _showObserve = !_showObserve),
        ),
        // Canvas toggle
        Consumer<ChatProvider>(
          builder: (context, chat, _) {
            if (chat.canvasHtml == null) return const SizedBox.shrink();
            return IconButton(
              icon: Icon(Icons.web_asset, color: JarvisTheme.accent),
              tooltip: l.canvasLabel,
              onPressed: () {},
            );
          },
        ),
        // Clear chat
        IconButton(
          icon: const Icon(Icons.delete_outline),
          tooltip: l.clearChat,
          onPressed: () => _chat.clearChat(),
        ),
      ],
    );
  }
}
