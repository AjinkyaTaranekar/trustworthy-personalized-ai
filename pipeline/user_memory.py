"""
Per-user persistent memory with 5W+H ontology sections.

Stores one JSON file per user at data/user_memory/<user_id>.json.
The UserMemoryStore interface is designed to be wire-compatible with the
GraphRAG integration planned for a later milestone — read/update stay the same.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional

_SECTIONS: Dict[str, str] = {
    "who":         "Role, expertise, background, identity",
    "what":        "Current goals, projects, active topics",
    "where":       "Geographic/domain context, jurisdiction",
    "why":         "Underlying motivations, constraints, reasons",
    "how":         "Preferred formats, communication style, tool preferences",
    "facts":       "Confirmed factual assertions about the user",
    "constraints": "Hard limits: budget, time, access restrictions",
}

_SECTION_ALIASES: Dict[str, str] = {
    "background":  "who",
    "identity":    "who",
    "role":        "who",
    "expertise":   "who",
    "goals":       "what",
    "projects":    "what",
    "topics":      "what",
    "location":    "where",
    "domain":      "where",
    "motivation":  "why",
    "reason":      "why",
    "style":       "how",
    "preferences": "how",
    "format":      "how",
    "fact":        "facts",
    "assertion":   "facts",
    "constraint":  "constraints",
    "limit":       "constraints",
    "limits":      "constraints",
}

_STOP_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "be", "it", "i", "my", "me", "we", "you", "with",
    "that", "this", "from", "by", "as", "do", "not", "no",
})

_DEFAULT_STORE_DIR = Path(__file__).parent / "data" / "user_memory"


def _relevance_score(content: str, prompt: str) -> float:
    """Keyword overlap fraction: |prompt_words ∩ content_words| / |prompt_words|."""
    if not content or "(empty" in content:
        return 0.0
    prompt_words = set(re.findall(r'\w+', prompt.lower())) - _STOP_WORDS
    if not prompt_words:
        return 0.5
    content_words = set(re.findall(r'\w+', content.lower()))
    return len(prompt_words & content_words) / len(prompt_words)


class UserMemoryStore:
    """JSON-on-disk per-user memory. In-memory cache; saves on every update."""

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self._dir = Path(store_dir) if store_dir else _DEFAULT_STORE_DIR
        self._cache: Dict[str, Dict[str, str]] = {}

    def _path(self, user_id: str) -> Path:
        return self._dir / f"{user_id}.json"

    def _empty_record(self) -> Dict[str, str]:
        return {k: f"(empty — {v})" for k, v in _SECTIONS.items()}

    def _load(self, user_id: str) -> Dict[str, str]:
        if user_id in self._cache:
            return self._cache[user_id]
        p = self._path(user_id)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for k, v in _SECTIONS.items():
                    data.setdefault(k, f"(empty — {v})")
                self._cache[user_id] = data
                return data
            except (json.JSONDecodeError, OSError):
                pass
        data = self._empty_record()
        self._cache[user_id] = data
        return data

    def _save(self, user_id: str, data: Dict[str, str]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path(user_id).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def read(self, user_id: str, prompt: str = "") -> str:
        """Return memory sections. Without prompt: all sections. With prompt: top-3 by relevance."""
        data = self._load(user_id)
        if not prompt:
            lines = [f"[{k.upper()}]\n{v}" for k, v in data.items()]
            return f"=== USER MEMORY (id: {user_id}) ===\n\n" + "\n\n".join(lines)

        scored = []
        for section, content in data.items():
            score = _relevance_score(content, prompt)
            if section == "who":
                score = max(score, 0.1)  # always surface identity as baseline context
            scored.append((score, section, content))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [(s, c) for _, s, c in scored[:3]]
        lines = [f"[{s.upper()}]\n{c}" for s, c in top]
        return f"=== USER MEMORY (id: {user_id}) ===\n\n" + "\n\n".join(lines)

    def update(self, user_id: str, section: str, content: str) -> str:
        section = _SECTION_ALIASES.get(section, section)
        if section not in _SECTIONS:
            return (
                f"Error: '{section}' is not a valid section. "
                f"Valid sections: {sorted(_SECTIONS.keys())}."
            )
        data = self._load(user_id)
        data[section] = content
        self._cache[user_id] = data
        self._save(user_id, data)
        return f"✓ user_memory[{section}] updated"
