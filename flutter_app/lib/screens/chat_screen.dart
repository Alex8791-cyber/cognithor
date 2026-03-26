import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:jarvis_ui/l10n/generated/app_localizations.dart';
import 'package:jarvis_ui/providers/chat_provider.dart';
import 'package:jarvis_ui/providers/connection_provider.dart';
import 'package:jarvis_ui/providers/hacker_mode_provider.dart';
import 'package:jarvis_ui/providers/pip_provider.dart';
import 'package:jarvis_ui/providers/sessions_provider.dart';
import 'package:jarvis_ui/providers/voice_provider.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';
import 'package:jarvis_ui/widgets/approval_dialog.dart';
import 'package:jarvis_ui/widgets/canvas_panel.dart';
import 'package:jarvis_ui/widgets/chat_bubble.dart';
import 'package:jarvis_ui/widgets/chat_input.dart';
import 'package:jarvis_ui/widgets/chat/context_panel.dart';
import 'package:jarvis_ui/widgets/chat/hacker_chat_view.dart';
import 'package:jarvis_ui/widgets/jarvis_empty_state.dart';
import 'package:jarvis_ui/widgets/message_actions.dart';
import 'package:jarvis_ui/widgets/chat/chat_history_drawer.dart';
import 'package:jarvis_ui/widgets/chat/feedback_buttons.dart';
import 'package:jarvis_ui/widgets/chat/message_actions.dart';
import 'package:jarvis_ui/widgets/chat/version_navigator.dart';
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
  final _scrollController = ScrollController();
  final _scaffoldKey = GlobalKey<ScaffoldState>();
  final _inputController = TextEditingController();
  final _inputFocusNode = FocusNode();
  bool _showObserve = false;
  bool _pipListenerAttached = false;
  bool _sessionsInitialized = false;

  bool get _isIncognito {
    final sessions = context.read<SessionsProvider>();
    final activeId = sessions.activeSessionId;
    if (activeId == null) return false;
    final match = sessions.sessions.where((s) => s['id'] == activeId);
    if (match.isEmpty) return false;
    return match.first['incognito'] == true;
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();

    // Initialize sessions provider once
    if (!_sessionsInitialized) {
      final conn = context.read<ConnectionProvider>();
      if (conn.state == JarvisConnectionState.connected) {
        final sessions = context.read<SessionsProvider>();
        sessions.setApi(conn.api);
        sessions.loadSessions();
        sessions.loadFolders();
      }
      _sessionsInitialized = true;
    }

    // Wire VoiceProvider sendToChat callback
    final voice = context.read<VoiceProvider>();
    voice.sendToChat = (text) {
      context.read<ChatProvider>().sendMessage(text);
    };

    // Attach PipProvider idle callback once
    if (!_pipListenerAttached) {
      final chat = context.read<ChatProvider>();
      chat.addListener(() {
        if (!chat.isStreaming && chat.activeTool == null && chat.statusText.isEmpty) {
          final pip = context.read<PipProvider>();
          if (pip.busy) {
            Future.delayed(const Duration(seconds: 3), () {
              if (mounted) pip.setBusy(false);
            });
          }
        }
      });
      _pipListenerAttached = true;
    }
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _inputController.dispose();
    _inputFocusNode.dispose();
    super.dispose();
  }

  /// Claude-style edit: rewind conversation to this message, put text in
  /// input field for editing. User submits → editAndResend removes all
  /// messages from this point and sends the new version.
  int? _editingIndex;

  void _startEdit(int index, String text) {
    setState(() => _editingIndex = index);
    _inputController.text = text;
    _inputController.selection = TextSelection.collapsed(offset: text.length);
    _inputFocusNode.requestFocus();
  }

  void _submitEdit(String text) {
    final idx = _editingIndex;
    setState(() => _editingIndex = null);
    if (idx != null) {
      context.read<ChatProvider>().editAndResend(idx, text);
    }
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

    return Scaffold(
        key: _scaffoldKey,
        appBar: _buildAppBar(l),
        drawer: Consumer<SessionsProvider>(
          builder: (context, sessions, _) {
            return ChatHistoryDrawer(
              sessions: sessions.sessions,
              folders: sessions.folders,
              activeSessionId: sessions.activeSessionId,
              onSelectSession: _onSelectSession,
              onNewChat: _onNewChat,
              onNewIncognitoChat: _onNewIncognitoChat,
              onDeleteSession: _onDeleteSession,
              onRenameSession: _onRenameSession,
              onMoveToFolder: _onMoveToFolder,
              searchResults: sessions.searchResults,
              onSearchChanged: sessions.searchChats,
              sessionsByProject: sessions.sessionsByProject,
            );
          },
        ),
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

                  // Messages list — switches between normal and hacker view
                  Expanded(
                    child: Consumer2<ChatProvider, HackerModeProvider>(
                      builder: (context, chat, hackerMode, _) {
                        debugPrint('[Chat] Consumer2 rebuild: messages=${chat.messages.length} streaming=${chat.isStreaming} id=${identityHashCode(chat)}');
                        _scrollToBottom();

                        if (hackerMode.enabled) {
                          return HackerChatView(
                            messages: chat.messages,
                            streamingText: chat.streamingText,
                            isStreaming: chat.isStreaming,
                            activeTool: chat.activeTool,
                            scrollController: _scrollController,
                          );
                        }

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
                            final bubble = MessageActions(
                              text: msg.text,
                              child: ChatBubble(
                                role: msg.role,
                                text: msg.text,
                              ),
                            );
                            if (msg.role == MessageRole.assistant) {
                              return Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  bubble,
                                  Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      MessageActionButtons(
                                        text: msg.text,
                                        isUser: false,
                                        onRetry: () => chat.retryLastResponse(),
                                        showRetry: index == chat.messages.length - 1,
                                      ),
                                      const SizedBox(width: 4),
                                      FeedbackButtons(
                                        messageId: msg.id,
                                        onFeedback: (rating, msgId) {
                                          chat.sendFeedback(
                                              rating, msgId, msg.text);
                                        },
                                      ),
                                    ],
                                  ),
                                ],
                              );
                            }
                            return Column(
                              crossAxisAlignment: CrossAxisAlignment.end,
                              children: [
                                bubble,
                                Row(
                                  mainAxisSize: MainAxisSize.min,
                                  mainAxisAlignment: MainAxisAlignment.end,
                                  children: [
                                    if (msg.hasVersions)
                                      VersionNavigator(
                                        currentVersion: msg.activeVersion,
                                        totalVersions: msg.versionCount,
                                        onPrevious: () =>
                                            chat.switchVersion(index, msg.activeVersion - 1),
                                        onNext: () =>
                                            chat.switchVersion(index, msg.activeVersion + 1),
                                      ),
                                    if (msg.hasVersions) const SizedBox(width: 8),
                                    MessageActionButtons(
                                      text: msg.text,
                                      isUser: true,
                                      onEdit: () => _startEdit(index, msg.text),
                                    ),
                                  ],
                                ),
                              ],
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

                  // Feedback follow-up dialog
                  Consumer<ChatProvider>(
                    builder: (context, chat, _) {
                      if (chat.pendingFeedbackFollowup == null) {
                        return const SizedBox.shrink();
                      }
                      final followup = chat.pendingFeedbackFollowup!;
                      return _FeedbackFollowupBanner(
                        feedbackId: followup['feedback_id'] ?? '',
                        question: followup['question'] ?? '',
                        onSubmit: (comment) {
                          chat.sendFeedbackComment(
                            followup['feedback_id'] ?? '',
                            comment,
                          );
                          chat.dismissFeedbackFollowup();
                        },
                        onDismiss: chat.dismissFeedbackFollowup,
                      );
                    },
                  ),

                  // Input bar
                  Consumer<ChatProvider>(
                    builder: (context, chat, _) {
                      return ChatInput(
                        controller: _inputController,
                        focusNode: _inputFocusNode,
                        onSend: (text) {
                          if (_editingIndex != null) {
                            // Claude-style: rewind + resend
                            _submitEdit(text);
                          } else {
                            chat.sendMessage(text);
                          }
                          // Wake up the robots!
                          context.read<PipProvider>().setBusy(true);
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

            // Context side panel (when tool is active)
            Consumer<ChatProvider>(
              builder: (context, chat, _) {
                if (chat.activeTool == null) {
                  return const SizedBox.shrink();
                }
                return ContextPanel(
                  activeTool: chat.activeTool,
                  statusText: chat.statusText,
                );
              },
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
    );
  }

  bool _showTyping(ChatProvider chat) {
    // Show typing when there's a status text but no streaming/tool activity
    if (chat.statusText.isNotEmpty &&
        !chat.isStreaming &&
        chat.activeTool == null) {
      return true;
    }
    // Also show typing when streaming started but no tokens arrived yet
    if (chat.isStreaming && chat.streamingText.isEmpty) {
      return true;
    }
    return false;
  }

  void _onSelectSession(String sessionId) async {
    final sessions = context.read<SessionsProvider>();
    final chat = context.read<ChatProvider>();
    final conn = context.read<ConnectionProvider>();

    final history = await sessions.loadHistory(sessionId);
    if (history != null) {
      chat.loadFromHistory(history);
    }
    if (conn.state == JarvisConnectionState.connected) {
      await conn.ws.switchSession(sessionId);
      chat.attach(conn.ws);
    }
  }

  void _onNewChat() async {
    final sessions = context.read<SessionsProvider>();
    final chat = context.read<ChatProvider>();
    final conn = context.read<ConnectionProvider>();

    final sessionId = await sessions.createNewSession();
    if (sessionId != null) {
      chat.clearForNewSession();
      if (conn.state == JarvisConnectionState.connected) {
        await conn.ws.switchSession(sessionId);
        chat.attach(conn.ws);
      }
    }
    if (mounted) Navigator.of(context).pop(); // close drawer
  }

  void _onNewIncognitoChat() async {
    final sessions = context.read<SessionsProvider>();
    final chat = context.read<ChatProvider>();
    final conn = context.read<ConnectionProvider>();

    final sessionId = await sessions.createIncognitoSession();
    if (sessionId != null) {
      chat.clearForNewSession();
      if (conn.state == JarvisConnectionState.connected) {
        await conn.ws.switchSession(sessionId);
        chat.attach(conn.ws);
      }
    }
    if (mounted) Navigator.of(context).pop(); // close drawer
  }

  void _onDeleteSession(String sessionId) {
    context.read<SessionsProvider>().deleteSession(sessionId);
  }

  void _onRenameSession(String sessionId, String newTitle) {
    context.read<SessionsProvider>().renameSession(sessionId, newTitle);
  }

  void _onMoveToFolder(String sessionId, String folder) {
    context.read<SessionsProvider>().moveToFolder(sessionId, folder);
  }

  PreferredSizeWidget _buildAppBar(AppLocalizations l) {
    return AppBar(
      leading: IconButton(
        icon: const Icon(Icons.history, color: JarvisTheme.sectionChat),
        tooltip: l.chatHistory,
        onPressed: () => _scaffoldKey.currentState?.openDrawer(),
      ),
      title: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(l.appTitle),
          if (_isIncognito) ...[
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: Colors.purple.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.purple.withValues(alpha: 0.4)),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.visibility_off, size: 12, color: Colors.purple),
                  SizedBox(width: 3),
                  Text('Inkognito', style: TextStyle(fontSize: 10, color: Colors.purple)),
                ],
              ),
            ),
          ],
        ],
      ),
      actions: _buildAppBarActions(l),
    );
  }

  List<Widget> _buildAppBarActions(AppLocalizations l) {
    final isLargePhone = MediaQuery.of(context).size.width > 400;
    final spacer = isLargePhone ? const SizedBox(width: 4) : const SizedBox.shrink();

    return [
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
      spacer,
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
      spacer,
      // Hacker mode toggle
      Consumer<HackerModeProvider>(
        builder: (context, hackerMode, _) {
          return IconButton(
            icon: Icon(
              Icons.terminal,
              color: hackerMode.enabled
                  ? const Color(0xFF00FF41)
                  : null,
            ),
            tooltip: l.hackerMode,
            onPressed: () => hackerMode.toggle(),
          );
        },
      ),
      spacer,
      // Observe panel toggle
      IconButton(
        icon: Icon(
          Icons.analytics_outlined,
          color: _showObserve ? JarvisTheme.accent : null,
        ),
        tooltip: l.observe,
        onPressed: () => setState(() => _showObserve = !_showObserve),
      ),
      spacer,
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
        onPressed: () => context.read<ChatProvider>().clearChat(),
      ),
    ];
  }
}

// ---------------------------------------------------------------------------
// Feedback Follow-up Banner
// ---------------------------------------------------------------------------

class _FeedbackFollowupBanner extends StatefulWidget {
  const _FeedbackFollowupBanner({
    required this.feedbackId,
    required this.question,
    required this.onSubmit,
    required this.onDismiss,
  });

  final String feedbackId;
  final String question;
  final void Function(String comment) onSubmit;
  final VoidCallback onDismiss;

  @override
  State<_FeedbackFollowupBanner> createState() =>
      _FeedbackFollowupBannerState();
}

class _FeedbackFollowupBannerState extends State<_FeedbackFollowupBanner> {
  final _controller = TextEditingController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: JarvisTheme.orange.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: JarvisTheme.orange.withValues(alpha: 0.30)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            widget.question,
            style: TextStyle(
              fontSize: 13,
              color: JarvisTheme.textSecondary,
            ),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _controller,
            maxLines: 2,
            decoration: InputDecoration(
              hintText: 'Was war nicht ideal?',
              hintStyle: TextStyle(color: JarvisTheme.textTertiary),
              filled: true,
              fillColor: Theme.of(context).scaffoldBackgroundColor,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(8),
                borderSide: BorderSide(
                    color: JarvisTheme.orange.withValues(alpha: 0.30)),
              ),
              contentPadding:
                  const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
            ),
            style: const TextStyle(fontSize: 13),
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              TextButton(
                onPressed: widget.onDismiss,
                child: Text(
                  'Ueberspringen',
                  style: TextStyle(color: JarvisTheme.textTertiary),
                ),
              ),
              const SizedBox(width: 8),
              ElevatedButton(
                onPressed: () {
                  final comment = _controller.text.trim();
                  if (comment.isNotEmpty) {
                    widget.onSubmit(comment);
                  } else {
                    widget.onDismiss();
                  }
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: JarvisTheme.orange,
                  foregroundColor: Colors.white,
                ),
                child: const Text('Senden'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
