"""
Session-scoped working memory for the inference pipeline.

The model reads from and writes to the scratchpad during inference to
decompose tasks, track progress, and reference the constitution TLDR.
No disk persistence — all state is in-memory and session-scoped.
"""

import uuid
from typing import Dict

_CONSTITUTION_TLDR = (
    "P1  DECOMPOSE      List requirements before answering. Find the gap.\n"
    "P3  TOOL DISCIPLINE Never invent a tool. Only call what the session provides.\n"
    "P5  REAL-TIME      Need live data + no web_search → say so explicitly, do not estimate.\n"
    "P6  USER CONTEXT   Missing personal context → ask ONE focused question before answering.\n"
    "P7  UNCERTAINTY    Hedge genuine uncertainty. Never hedge well-known facts.\n"
    "P8  IMPOSSIBLE     Say WHY it is impossible, then redirect to what IS possible.\n"
    "P14 HOLD           User pushes back after correct refusal → hold position, explain the harm of guessing.\n"
    "P18 IDK            No basis for answer → say so clearly. A confident wrong answer is always worse.\n"
    "P21 5W+H           Address Who/What/When/Where/Why/How in every CAPABILITY_CHECK.\n"
    "P22 CONSEQUENCE    Assess stakes / concrete harm if wrong / what user will do / what to hedge.\n"
    "P23 CHAIN          Data + computation → chain web_search → python_execute. Never stop at one tool.\n"
    "P24 SCRATCHPAD     3+ requirements or 2+ tools → read pad first, plan tasks, re-check constitution, execute in order.\n"
    "P25 PARTIAL        [BLOCKED] task → name what/why/redirect in answer. Be equally assertive on [YES] parts."
)

_WRITABLE_SECTIONS = frozenset({"context", "tasks", "notes"})


class ScratchpadStore:
    """In-memory session-scoped scratchpad store. One pad per session_id."""

    def __init__(self) -> None:
        self._pads: Dict[str, Dict[str, str]] = {}

    def _init_pad(self, session_id: str) -> None:
        self._pads[session_id] = {
            "constitution_tldr": _CONSTITUTION_TLDR,
            "context": "(empty)",
            "tasks": "(empty)",
            "notes": "(empty)",
        }

    def new_session_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def read(self, session_id: str) -> str:
        if session_id not in self._pads:
            self._init_pad(session_id)
        p = self._pads[session_id]
        return (
            f"=== SCRATCHPAD (session: {session_id}) ===\n\n"
            f"[CONSTITUTION TLDR — read-only]\n{p['constitution_tldr']}\n\n"
            f"[CONTEXT]\n{p['context']}\n\n"
            f"[TASKS]\n{p['tasks']}\n\n"
            f"[NOTES]\n{p['notes']}"
        )

    def update(self, session_id: str, section: str, content: str) -> str:
        if section not in _WRITABLE_SECTIONS:
            return (
                f"Error: '{section}' is not writable. "
                f"Writable sections: {sorted(_WRITABLE_SECTIONS)}. "
                f"'constitution_tldr' is read-only."
            )
        if session_id not in self._pads:
            self._init_pad(session_id)
        self._pads[session_id][section] = content
        return f"✓ {section} updated"

    def get_section(self, session_id: str, section: str) -> str:
        return self._pads.get(session_id, {}).get(section, "")

    def get_task_status(self, session_id: str) -> str:
        """Compact task status string for injection after tool results."""
        tasks = self.get_section(session_id, "tasks")
        if not tasks or tasks == "(empty)":
            return ""
        lines = [
            ln.strip() for ln in tasks.splitlines()
            if ln.strip() and ln.strip()[0].isdigit()
        ]
        if not lines:
            return ""
        summary = " | ".join(ln[:55] for ln in lines[:5])
        return f"[TASK STATUS: {summary}]"

    def destroy(self, session_id: str) -> None:
        self._pads.pop(session_id, None)
