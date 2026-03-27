# Chat Branching — Full Conversation Tree

**Date:** 2026-03-27
**Author:** Alexander Soellner + Claude Opus 4.6
**Status:** Approved

---

## Problem

Linear chat forces users to edit+rewind when they want to explore alternatives.
There is no way to:
- Fork a conversation at any point and explore two directions
- Roll back 5 messages and try a different approach while keeping the old one
- Compare the same question answered by different agents/approaches

## Solution: Full Conversation Tree

Every message has a `parentId` instead of living in a flat list. A message can
have multiple children (branches). The user navigates branches via inline
`< 1/3 >` controls at each fork point, with an optional tree sidebar for
full overview.

## Data Model

### Message Node

```
ChatMessageNode {
  id: string               // Unique message ID
  parentId: string | null   // null = root message
  role: user | assistant | system
  text: string
  timestamp: datetime
  branchIndex: int          // Which child am I? (0-based)
  childIds: list[string]    // My children message IDs
  metadata: {
    agentName: string       // Which agent answered
    model: string           // Which model was used
    durationMs: int         // How long the response took
  }
}
```

### Conversation Tree

```
ConversationTree {
  id: string                // Session/conversation ID
  rootId: string            // First message node ID
  activePathIds: list[string]  // Currently displayed path from root to leaf
  nodes: map[string, ChatMessageNode]  // All nodes by ID
}
```

### Active Path

The "active path" is the sequence of message IDs currently displayed in the
chat view. When the user switches branches, the active path changes.

Example tree:
```
User: "Hallo" (root, id=m1)
  └─ Asst: "Hi!" (id=m2, branchIndex=0)
       ├─ User: "Recherchiere X" (id=m3, branchIndex=0)
       │    └─ Asst: "Ergebnis X..." (id=m4)
       └─ User: "Programmiere Y" (id=m5, branchIndex=1)
            └─ Asst: "Code Y..." (id=m6)
```

Active path when viewing branch 0: [m1, m2, m3, m4]
Active path when viewing branch 1: [m1, m2, m5, m6]
Inline navigator at m2: `< 1/2 >` (because m2 has 2 children sets)

## Backend Architecture

### Hybrid Memory Model

- **Active branch**: Full WorkingMemory in RAM (fast, immediate responses)
- **Inactive branches**: Only message history stored in SQLite
- **Branch switch**: Replay message history into fresh WorkingMemory (1-2s)

### Session Branching

When the user forks (edits a message or explicitly branches):
1. Store current branch state (messages + agent context) in SQLite
2. Create new branch with cloned message history up to fork point
3. New branch becomes active (gets the WorkingMemory)
4. Old branch becomes inactive (history-only)

### SQLite Schema

```sql
CREATE TABLE chat_nodes (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    parent_id TEXT,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    branch_index INTEGER DEFAULT 0,
    agent_name TEXT DEFAULT 'jarvis',
    model_used TEXT DEFAULT '',
    duration_ms INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX idx_nodes_conv ON chat_nodes(conversation_id);
CREATE INDEX idx_nodes_parent ON chat_nodes(parent_id);

CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    title TEXT DEFAULT '',
    active_leaf_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL
);
```

### WebSocket Protocol

New message types:
- `branch_switch` (client→server): `{type: "branch_switch", nodeId: "m2", branchIndex: 1}`
  Server replays history for that branch into WorkingMemory.
- `branch_info` (server→client): `{type: "branch_info", nodeId: "m2", childCount: 2, activeIndex: 0}`
  Tells the UI which nodes have forks and which branch is active.

### API Endpoints

- `GET /api/v1/chat/tree/{conversation_id}` — Full tree structure
- `POST /api/v1/chat/branch` — Create explicit branch at a node
- `GET /api/v1/chat/path/{conversation_id}/{leaf_id}` — Get active path

## Frontend Architecture

### ConversationTree Provider

New provider (or extension of ChatProvider) that manages the tree:
- `nodes: Map<String, ChatMessageNode>`
- `activePath: List<String>`
- `switchBranch(nodeId, branchIndex)` — updates activePath + sends WS message
- `forkAtNode(nodeId, newText)` — creates a new child branch
- `getChildCount(nodeId)` — how many branches at this node
- `getActiveChildIndex(nodeId)` — which branch is displayed

### Inline Branch Navigator

At each message that has sibling branches, show `< 1/3 >`:
- Appears between the parent message and the current child
- Clicking arrows switches the displayed branch (and all descendants)
- Only visible at fork points (nodes where parent has multiple children)

### Tree Sidebar (Optional, Toggle)

A collapsible panel showing the full tree:
- Nodes as indented list with role icons (user/assistant)
- Active path highlighted
- Click any node to navigate there
- Fork points marked with branch icon
- Toggle via toolbar button

### Edit = Fork

When the user edits a message (pencil icon):
1. The old message + its subtree remain as branch 0
2. New text creates branch 1 at the same parent
3. Inline navigator appears: `< 1/2 >`
4. Active path switches to the new branch
5. Cognithor responds in the new branch context

## Migration from Current System

The current `ChatMessage` with `versions: List<MessageVersion>` is replaced
by the tree model. The version navigator `< 1/2 >` stays but now controls
tree branches instead of inline version arrays.

## Config

No new config fields needed — branching is a core chat feature, always available.

## Files

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/jarvis/core/conversation_tree.py` | ConversationTree, ChatNode, SQLite persistence |
| Modify | `src/jarvis/gateway/gateway.py` | Branch-aware session handling, WM replay |
| Modify | `src/jarvis/__main__.py` | WebSocket branch_switch handler |
| Modify | `src/jarvis/channels/config_routes.py` | Tree API endpoints |
| Create | `flutter_app/lib/providers/tree_provider.dart` | ConversationTree state management |
| Create | `flutter_app/lib/widgets/chat/branch_navigator.dart` | Inline < 1/3 > controls |
| Create | `flutter_app/lib/widgets/chat/tree_sidebar.dart` | Optional tree overview panel |
| Modify | `flutter_app/lib/screens/chat_screen.dart` | Tree-aware message rendering |
| Modify | `flutter_app/lib/providers/chat_provider.dart` | Integrate with tree provider |
| Create | `tests/unit/test_conversation_tree.py` | Tests |

## Not in Scope

- Cross-branch merging (too complex, no clear UX)
- Branch comparison view (side-by-side) — future feature
- Branch naming/tagging — future feature
- Automatic branch pruning — manual only
