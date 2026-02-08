"""Token streaming handler for the TUI.

This module provides a simple token streaming handler that writes
streamed LLM tokens to the TUI's visualizer widget.
"""

from collections.abc import Callable

from openhands.sdk.llm.streaming import LLMStreamChunk
from openhands_cli.tui.widgets.richlog_visualizer import ConversationVisualizer


class TUITokenStreamer:
    """Handles token streaming for the TUI visualizer.

    This class receives streaming token chunks from the LLM and writes
    them to the TUI's visualizer widget for real-time display.
    """

    def __init__(
        self,
        visualizer: ConversationVisualizer,
        write_callback: Callable[[str], None] | None = None,
    ):
        """Initialize the token streamer.

        Args:
            visualizer: The TUI visualizer to write tokens to.
            write_callback: Optional callback to write text to the UI.
                           If not provided, uses visualizer's write method.
        """
        self.visualizer = visualizer
        self._write_callback = write_callback
        self._current_content = ""
        self._reasoning_header_emitted = False

    def reset(self) -> None:
        """Reset state for a new streaming response."""
        self._current_content = ""
        self._reasoning_header_emitted = False

    def on_token(self, chunk: LLMStreamChunk) -> None:
        """Handle a streaming token chunk.

        This callback is invoked for each token delta from the LLM.

        Args:
            chunk: The streaming chunk containing token deltas.
        """
        try:
            for choice in chunk.choices:
                delta = getattr(choice, "delta", None)
                if not delta:
                    continue

                choice_index = getattr(choice, "index", 0) or 0

                # Only process content from choice.index == 0
                # to avoid interleaving from multiple choices
                if choice_index != 0:
                    continue

                # Handle reasoning content (thinking/CoT)
                reasoning = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning, str) and reasoning.strip():
                    if not self._reasoning_header_emitted:
                        self._reasoning_header_emitted = True
                        reasoning = "💭 Thinking:\n" + reasoning
                    self._write_text(reasoning)

                # Handle regular content
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    self._write_text(content)

        except Exception:
            # Silently ignore streaming errors to avoid disrupting the UI
            pass

    def _write_text(self, text: str) -> None:
        """Write text to the UI.

        Args:
            text: The text to write.
        """
        if self._write_callback:
            self._write_callback(text)
        # Note: The visualizer doesn't have a direct write method for streaming,
        # so we rely on the write_callback for now. The visualizer will receive
        # complete events through the normal event callback mechanism.
