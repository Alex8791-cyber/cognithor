/// Data models for the conversation tree (chat branching).
library;

class ChatNode {
  ChatNode({
    required this.id,
    required this.conversationId,
    this.parentId,
    required this.role,
    required this.text,
    this.branchIndex = 0,
    this.agentName = 'jarvis',
    this.modelUsed = '',
    this.durationMs = 0,
    DateTime? createdAt,
  }) : createdAt = createdAt ?? DateTime.now();

  final String id;
  final String conversationId;
  final String? parentId;
  final String role; // 'user' | 'assistant' | 'system'
  String text;
  final int branchIndex;
  final String agentName;
  final String modelUsed;
  final int durationMs;
  final DateTime createdAt;

  bool get isUser => role == 'user';
  bool get isAssistant => role == 'assistant';

  factory ChatNode.fromJson(Map<String, dynamic> json) {
    return ChatNode(
      id: json['id'] as String,
      conversationId: json['conversation_id'] as String? ?? '',
      parentId: json['parent_id'] as String?,
      role: json['role'] as String,
      text: json['text'] as String,
      branchIndex: json['branch_index'] as int? ?? 0,
      agentName: json['agent_name'] as String? ?? 'jarvis',
      modelUsed: json['model_used'] as String? ?? '',
      durationMs: json['duration_ms'] as int? ?? 0,
    );
  }

  Map<String, dynamic> toJson() => {
    'id': id,
    'conversation_id': conversationId,
    'parent_id': parentId,
    'role': role,
    'text': text,
    'branch_index': branchIndex,
    'agent_name': agentName,
  };
}

class ForkPoint {
  const ForkPoint({
    required this.nodeId,
    required this.childCount,
    this.activeChildIndex = 0,
  });

  final String nodeId;
  final int childCount;
  final int activeChildIndex;
}
