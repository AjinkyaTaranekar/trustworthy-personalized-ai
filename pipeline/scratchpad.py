"""
Session-scoped working memory for the inference pipeline.

The model reads from and writes to the scratchpad during inference to
track First Principles decomposition, 5W+H state, tasks, and intermediate notes.
No disk persistence — all state is in-memory and session-scoped.

Sections
--------
5wh_state  Known/unknown 5W+H dimensions for this conversation. Update after
           user_memory_read or whenever a new dimension is revealed. Format:
             Known:   WHO=<...>, WHAT=<...>
             Unknown: WHY, WHEN (ask about WHY next — most critical)
tasks      Numbered checklist. Use [ ] / [x] markers.
notes      Intermediate results: calculations, web snippets, hypotheses.

The old 'context' section is aliased to 'notes' for backward compatibility.
The constitution TLDR was removed — it was stale (referenced v2 principle
numbers) and consumed ~200 tokens per scratchpad_read() with no benefit,
since the system prompt already carries the full instruction set.
"""

import uuid
from typing import Dict

_WRITABLE_SECTIONS = frozenset({"5wh_state", "tasks", "notes"})
# Alias: old training data wrote to 'context' — map it to 'notes'
_SECTION_ALIASES = {"context": "notes"}


class ScratchpadStore:
    """In-memory session-scoped scratchpad store. One pad per session_id."""

    def __init__(self) -> None:
        self._pads: Dict[str, Dict[str, str]] = {}

    def _init_pad(self, session_id: str) -> None:
        self._pads[session_id] = {
            "5wh_state": "(empty — update after reading user memory)",
            "tasks":     "(empty)",
            "notes":     "(empty)",
        }

    def new_session_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def sections(self, session_id: str) -> str:
        """Return a compact list of writable section names and their current state."""
        if session_id not in self._pads:
            self._init_pad(session_id)
        p = self._pads[session_id]
        lines = []
        for k, v in p.items():
            status = "(empty)" if v.startswith("(empty") else f"({len(v)} chars)"
            lines.append(f"  {k:<12} {status}")
        return (
            "Scratchpad sections — call scratchpad_update(section=<key>, content=<value>):\n"
            + "\n".join(lines)
        )

    def read(self, session_id: str) -> str:
        if session_id not in self._pads:
            self._init_pad(session_id)
        p = self._pads[session_id]
        return (
            f"=== SCRATCHPAD (session: {session_id}) ===\n\n"
            f"[5W+H STATE]\n{p['5wh_state']}\n\n"
            f"[TASKS]\n{p['tasks']}\n\n"
            f"[NOTES]\n{p['notes']}"
        )

    def update(self, session_id: str, section: str, content: str) -> str:
        section = _SECTION_ALIASES.get(section, section)
        if section not in _WRITABLE_SECTIONS:
            return (
                f"Error: '{section}' is not a valid section. "
                f"Writable sections: {sorted(_WRITABLE_SECTIONS)}. "
                f"Tip: use '5wh_state' to record known/unknown dimensions."
            )
        if session_id not in self._pads:
            self._init_pad(session_id)
        self._pads[session_id][section] = content
        return f"✓ {section} updated"

    def get_section(self, session_id: str, section: str) -> str:
        section = _SECTION_ALIASES.get(section, section)
        return self._pads.get(session_id, {}).get(section, "")

    def get_task_status(self, session_id: str) -> str:
        """Compact status string auto-injected after tool results.

        Shows task list + 5W+H next-to-ask so the model always knows its
        position without having to call scratchpad_read() again.
        """
        p = self._pads.get(session_id, {})

        # Task summary
        tasks = p.get("tasks", "(empty)")
        task_lines = [
            ln.strip() for ln in tasks.splitlines()
            if ln.strip() and ln.strip()[0].isdigit()
        ]
        task_summary = " | ".join(ln[:45] for ln in task_lines[:4]) if task_lines else ""

        # 5W+H next unknown — extract the "Unknown:" line if present
        wh_state = p.get("5wh_state", "")
        wh_unknown = ""
        for ln in wh_state.splitlines():
            if ln.strip().lower().startswith("unknown"):
                wh_unknown = ln.strip()
                break

        parts = []
        if task_summary:
            parts.append(f"TASKS: {task_summary}")
        if wh_unknown:
            parts.append(wh_unknown)
        return f"[{' | '.join(parts)}]" if parts else ""

    def destroy(self, session_id: str) -> None:
        self._pads.pop(session_id, None)
