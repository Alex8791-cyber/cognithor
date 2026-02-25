"""Jarvis Human-in-the-Loop v20 -- Graph-Level HITL-Workflows.

Professionelle HITL-Integration auf Graph-Ebene:
  - Approval/Reject/Delegate an beliebigen Knoten
  - Multi-Approval, Eskalation, Timeout-Handling
  - Gate-Nodes mit bedingtem Human-Review
  - Multi-Channel-Notifications (In-App, Webhook, Callback)
  - Selection/Edit-Nodes für strukturierte menschliche Eingabe

Usage:
    from jarvis.hitl import ApprovalManager, create_approval_node, HITLConfig
    from jarvis.graph import GraphBuilder, END, NodeType

    manager = ApprovalManager()
    graph = (
        GraphBuilder("review_flow")
        .add_node("process", process_handler)
        .add_node("review", create_approval_node(manager, config=HITLConfig(
            title="Review Results",
            assignees=["supervisor"],
        )), node_type=NodeType.HITL)
        .chain("process", "review", END)
        .build()
    )
"""

from jarvis.hitl.types import (
    ApprovalStatus,
    ReviewPriority,
    EscalationAction,
    NotificationType,
    HITLNodeKind,
    NotificationChannel,
    EscalationPolicy,
    HITLConfig,
    ApprovalRequest,
    ApprovalResponse,
    ReviewTask,
)
from jarvis.hitl.notifier import HITLNotifier, NotificationRecord
from jarvis.hitl.manager import ApprovalManager
from jarvis.hitl.nodes import (
    create_approval_node,
    create_review_node,
    create_input_node,
    create_gate_node,
    create_selection_node,
    create_edit_node,
)

__all__ = [
    # Enums
    "ApprovalStatus", "ReviewPriority", "EscalationAction",
    "NotificationType", "HITLNodeKind",
    # Config Types
    "NotificationChannel", "EscalationPolicy", "HITLConfig",
    # Request/Response
    "ApprovalRequest", "ApprovalResponse", "ReviewTask",
    # Notifier
    "HITLNotifier", "NotificationRecord",
    # Manager
    "ApprovalManager",
    # Node Factories
    "create_approval_node", "create_review_node",
    "create_input_node", "create_gate_node",
    "create_selection_node", "create_edit_node",
]
