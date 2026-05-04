"""Sprint-27 streaming sinks.

PR-A ships the :class:`Sink` ABC + bounded-buffer + critical-event
bypass machinery. Concrete sinks (:mod:`jsonl_sink`,
:mod:`ws_sink`) land in PR-B and PR-C.
"""

from __future__ import annotations

from cognithor.streaming.sinks.base import Sink, SinkBufferConfig
from cognithor.streaming.sinks.jsonl_sink import JsonlSink
from cognithor.streaming.sinks.ws_sink import WebSocketSink, encode_event_frame

__all__ = [
    "JsonlSink",
    "Sink",
    "SinkBufferConfig",
    "WebSocketSink",
    "encode_event_frame",
]
