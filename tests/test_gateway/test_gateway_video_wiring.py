from __future__ import annotations


class TestClassifyAttachments:
    def test_video_and_image_classified_correctly(self):
        from cognithor.gateway.gateway import _classify_attachments

        images, video, rejected = _classify_attachments(
            [
                "/tmp/pic.png",
                "/tmp/clip.mp4",
            ]
        )
        assert images == ["/tmp/pic.png"]
        assert video == "/tmp/clip.mp4"
        assert rejected == []

    def test_second_video_is_rejected(self):
        from cognithor.gateway.gateway import _classify_attachments

        images, video, rejected = _classify_attachments(
            [
                "/tmp/clip1.mp4",
                "/tmp/clip2.mp4",
                "/tmp/clip3.webm",
            ]
        )
        assert images == []
        assert video == "/tmp/clip1.mp4"  # first wins
        assert rejected == ["/tmp/clip2.mp4", "/tmp/clip3.webm"]

    def test_no_videos_returns_none(self):
        from cognithor.gateway.gateway import _classify_attachments

        images, video, rejected = _classify_attachments(
            [
                "/tmp/pic.png",
                "/tmp/doc.pdf",
            ]
        )
        assert video is None
        assert rejected == []
        assert images == ["/tmp/pic.png"]

    def test_empty_input_returns_empty(self):
        from cognithor.gateway.gateway import _classify_attachments

        images, video, rejected = _classify_attachments([])
        assert images == []
        assert video is None
        assert rejected == []


class TestBuildVideoAttachmentIsNonBlocking:
    def test_gateway_wraps_build_video_attachment_in_to_thread(self):
        """Regression for Bug I3: the per-turn handler must offload the
        blocking _build_video_attachment call (which runs ffprobe via
        subprocess.run) to a thread pool, otherwise a slow remote URL
        ties up the async event loop for up to 30 s.

        Source-level assertion. The handler used to live in `gateway.py`
        but was extracted into `gateway/message_handler.py` as part of
        the staged gateway split (PR #192 onwards). Scan both modules.
        """
        import inspect

        from cognithor.gateway import gateway as gw_mod

        src = inspect.getsource(gw_mod)
        try:
            from cognithor.gateway import message_handler

            src += "\n" + inspect.getsource(message_handler)
        except ImportError:
            pass

        # The blocking call site must be wrapped; look for the exact pattern.
        # Accept the wrapped forms; reject any bare unwrapped form.
        has_wrapped = "asyncio.to_thread(_build_video_attachment" in src or (
            "asyncio.to_thread(\n" in src and "_build_video_attachment" in src
        )
        assert has_wrapped, (
            "Expected _build_video_attachment to be called via "
            "asyncio.to_thread (either gateway.py or message_handler.py) — see Bug I3."
        )

        # And the bare unwrapped form must NOT appear in any active code path.
        # (A commented-out example is fine; a genuine bare call is not.)
        lines = [
            line
            for line in src.splitlines()
            if "_build_video_attachment(" in line and not line.lstrip().startswith("#")
        ]
        bare_calls = [
            line
            for line in lines
            if "asyncio.to_thread" not in line
            and "def _build_video_attachment" not in line
            and "await" not in line.split("_build_video_attachment", 1)[0][-20:]
        ]
        # Only the definition and the wrapped call should remain; no bare
        # unwrapped invocations. The function definition itself counts; filter
        # it out.
        bare_invocations = [line for line in bare_calls if "def " not in line]
        assert not bare_invocations, (
            f"Found unwrapped bare call(s) to _build_video_attachment: {bare_invocations}"
        )
