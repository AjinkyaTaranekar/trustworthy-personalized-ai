"""
User Modelling Module
=====================
FalkorDB-backed 5W+H user profile graph with a Mem0g-pattern write pipeline
and a scrutability layer (inspect / contest / correct).

Modules
-------
GraphClient          — typed Cypher wrapper over the falkordb Python package
write_pipeline()     — 4-stage Mem0g write triggered after every user turn:
                         entity_extractor → relation_generator
                         → conflict_detector → conditional_write
retrieve_for_query() — anti-over-personalisation gate: only fetch the
                       5W+H subgraph when a schema slot is relevant to
                       the current query
inspect_memory()     — FastAPI handler: NL summary of the user graph
contest_belief()     — FastAPI handler: mark a node as contested
correct_belief()     — FastAPI handler: apply a user correction

Wiring (Phase 4 — 3_infererence.py)
-------------------------------------
    from user_modelling import GraphClient, write_pipeline, retrieve_for_query
    from user_modelling import inspect_memory, contest_belief, correct_belief

    _graph_client = GraphClient(cfg)   # once at server startup, inside lifespan

    # In the chat_completions handler, after reading the user turn:
    if cfg.ENABLE_USER_MODELLING:
        write_pipeline(turn_text, session_id, _graph_client, llm_fn)
    if cfg.ENABLE_PERSONALISATION:
        user_ctx = retrieve_for_query(query_text, session_id, _graph_client, llm_fn)

The llm_fn parameter is a plain callable (str) -> str — pass the server's own
_raw_generate() so the write pipeline reuses the already-loaded model with no
HTTP round-trip.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5W+H schema constants
# ---------------------------------------------------------------------------

# Node labels — one per 5W+H slot
class Label:
    USER    = "User"
    GOAL    = "Goal"      # WHY
    TASK    = "Task"      # WHAT
    CONTEXT = "Context"   # WHERE
    TIME    = "Time"      # WHEN
    STYLE   = "Style"     # HOW
    SKILL   = "Skill"     # WHO / HOW

# Typed edge labels
class Rel:
    PURSUES          = "PURSUES"
    HAS_SKILL        = "HAS_SKILL"
    PREFERS          = "PREFERS"
    IS_WORKING_ON    = "IS_WORKING_ON"
    WORKS_IN         = "WORKS_IN"
    DEPRECATED_BY    = "DEPRECATED_BY"
    USER_CORRECTED   = "USER_CORRECTED"
    CONFLICTS_WITH   = "CONFLICTS_WITH"

# Slot name → node label
SLOT_TO_LABEL: Dict[str, str] = {
    "WHO":   Label.SKILL,
    "WHAT":  Label.TASK,
    "WHEN":  Label.TIME,
    "WHERE": Label.CONTEXT,
    "WHY":   Label.GOAL,
    "HOW":   Label.STYLE,
}

# Node label → relationship from User node
LABEL_TO_REL: Dict[str, str] = {
    Label.GOAL:    Rel.PURSUES,
    Label.TASK:    Rel.IS_WORKING_ON,
    Label.CONTEXT: Rel.WORKS_IN,
    Label.STYLE:   Rel.PREFERS,
    Label.SKILL:   Rel.HAS_SKILL,
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TypedEntity:
    slot:        str            # WHO | WHAT | WHEN | WHERE | WHY | HOW
    label:       str            # FalkorDB node label
    name:        str            # short identifier
    description: str            # longer free-text
    node_id:     str = field(default_factory=lambda: str(uuid.uuid4()))
    extra:       Dict[str, Any] = field(default_factory=dict)  # level, domain, etc.


@dataclass
class TypedEdge:
    from_id:   str
    to_id:     str
    rel_type:  str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conflict:
    new_entity:      TypedEntity
    existing_node_id: str
    existing_props:  Dict[str, Any]
    reason:          str


@dataclass
class WriteResult:
    entities_written: int = 0
    edges_written:    int = 0
    conflicts:        List[Conflict] = field(default_factory=list)
    skipped:          bool = False   # True when GraphClient is unavailable


@dataclass
class UserContext:
    """Structured user context returned by retrieve_for_query()."""
    session_id:    str
    relevant_slots: List[str]      # which 5W+H slots were matched
    goals:         List[Dict]
    tasks:         List[Dict]
    skills:        List[Dict]
    styles:        List[Dict]
    contexts:      List[Dict]
    raw_nodes:     int = 0         # total nodes fetched before MAX_NODES cap

    def is_empty(self) -> bool:
        return not any([self.goals, self.tasks, self.skills, self.styles, self.contexts])

    def to_prompt_block(self) -> str:
        """Format for injection into the system prompt."""
        if self.is_empty():
            return ""
        lines = ["<user_context>"]
        if self.goals:
            lines.append("  Goals: " + "; ".join(g["description"] for g in self.goals))
        if self.skills:
            lines.append("  Skills: " + "; ".join(
                f"{s['name']} ({s.get('level', 'unknown')})" for s in self.skills
            ))
        if self.tasks:
            lines.append("  Current tasks: " + "; ".join(t["description"] for t in self.tasks))
        if self.styles:
            lines.append("  Preferences: " + "; ".join(s["description"] for s in self.styles))
        if self.contexts:
            lines.append("  Context: " + "; ".join(c["description"] for c in self.contexts))
        lines.append("</user_context>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GraphClient
# ---------------------------------------------------------------------------

class GraphClient:
    """
    Typed Cypher wrapper over the falkordb Python package.

    All FalkorDB interaction goes through this class. When FalkorDB is not
    reachable (cfg flag is off, Docker not running), available=False and every
    method returns an empty/no-op result — the server never crashes.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.available = False
        self._graph = None

        if not cfg.ENABLE_USER_MODELLING:
            return

        try:
            import falkordb as _fdb
            kwargs = {"host": cfg.FALKORDB_HOST, "port": cfg.FALKORDB_PORT}
            if cfg.FALKORDB_USER:
                kwargs["username"] = cfg.FALKORDB_USER
            if cfg.FALKORDB_PASSWORD:
                kwargs["password"] = cfg.FALKORDB_PASSWORD
            db = _fdb.FalkorDB(**kwargs)
            self._graph = db.select_graph(cfg.FALKORDB_GRAPH_NAME)
            self._ensure_indexes()
            self.available = True
            logger.info(
                "UserModelling: connected to FalkorDB at %s:%s graph=%s",
                cfg.FALKORDB_HOST, cfg.FALKORDB_PORT, cfg.FALKORDB_GRAPH_NAME,
            )
        except ImportError:
            logger.warning("UserModelling: falkordb package not installed — pip install falkordb")
        except Exception as exc:
            logger.warning("UserModelling: FalkorDB not reachable — %s", exc)

    # ------------------------------------------------------------------
    # Schema bootstrapping
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create indexes on session_id for fast per-session queries."""
        for label in (Label.USER, Label.GOAL, Label.TASK, Label.CONTEXT,
                      Label.STYLE, Label.SKILL, Label.TIME):
            try:
                self._graph.query(
                    f"CREATE INDEX ON :{label}(session_id)"
                )
            except Exception:
                pass  # index already exists

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def ensure_user_node(self, session_id: str) -> None:
        """Create the User anchor node for this session if it does not exist."""
        if not self.available:
            return
        self._graph.query(
            "MERGE (:User {session_id: $sid})",
            {"sid": session_id},
        )

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def create_entity_node(self, entity: TypedEntity, session_id: str) -> None:
        """Create a node for the entity and link it to the User node."""
        if not self.available:
            return
        props = {
            "node_id":     entity.node_id,
            "session_id":  session_id,
            "name":        entity.name,
            "description": entity.description,
            "created_at":  _now(),
            "contested":   False,
            **entity.extra,
        }
        cypher = (
            f"MATCH (u:User {{session_id: $sid}}) "
            f"CREATE (n:{entity.label} $props) "
            f"CREATE (u)-[:{LABEL_TO_REL.get(entity.label, 'RELATED_TO')}]->(n)"
        )
        self._graph.query(cypher, {"sid": session_id, "props": props})

    def create_edge(self, edge: TypedEdge) -> None:
        """Create a relationship between two nodes identified by their node_id property."""
        if not self.available:
            return
        cypher = (
            "MATCH (a {node_id: $from_id}), (b {node_id: $to_id}) "
            f"CREATE (a)-[:{edge.rel_type} $props]->(b)"
        )
        self._graph.query(cypher, {
            "from_id": edge.from_id,
            "to_id":   edge.to_id,
            "props":   edge.properties or {},
        })

    def deprecate_node(self, old_node_id: str, new_node_id: str, reason: str) -> None:
        """Mark old_node as deprecated, pointing to new_node. Never deletes."""
        if not self.available:
            return
        self._graph.query(
            "MATCH (old {node_id: $oid}), (new {node_id: $nid}) "
            "SET old.deprecated = true, old.deprecated_at = $ts, old.deprecated_reason = $reason "
            "CREATE (old)-[:DEPRECATED_BY {reason: $reason, ts: $ts}]->(new)",
            {"oid": old_node_id, "nid": new_node_id, "ts": _now(), "reason": reason},
        )

    def mark_contested(self, node_id: str) -> None:
        if not self.available:
            return
        self._graph.query(
            "MATCH (n {node_id: $nid}) SET n.contested = true, n.contested_at = $ts",
            {"nid": node_id, "ts": _now()},
        )

    def mark_user_corrected(self, old_node_id: str, new_node_id: str) -> None:
        if not self.available:
            return
        self._graph.query(
            "MATCH (old {node_id: $oid}), (new {node_id: $nid}) "
            "SET old.user_corrected = true "
            "CREATE (old)-[:USER_CORRECTED {ts: $ts}]->(new)",
            {"oid": old_node_id, "nid": new_node_id, "ts": _now()},
        )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def query_nodes_by_label(
        self, session_id: str, label: str, exclude_deprecated: bool = True
    ) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        where = "WHERE NOT n.deprecated = true AND NOT n.contested = true" if exclude_deprecated else ""
        result = self._graph.query(
            f"MATCH (n:{label} {{session_id: $sid}}) {where} RETURN n",
            {"sid": session_id},
        )
        return [_node_to_dict(r[0]) for r in (result.result_set or [])]

    def find_conflicting_nodes(
        self, entity: TypedEntity, session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Find existing non-deprecated nodes of the same label and session with
        the same name but different key attributes — a signal of contradiction.
        """
        if not self.available:
            return []
        result = self._graph.query(
            f"MATCH (n:{entity.label} {{session_id: $sid, name: $name}}) "
            "WHERE NOT n.deprecated = true AND n.node_id <> $nid "
            "RETURN n",
            {"sid": session_id, "name": entity.name, "nid": entity.node_id},
        )
        return [_node_to_dict(r[0]) for r in (result.result_set or [])]

    def get_all_nodes(self, session_id: str) -> Dict[str, List[Dict]]:
        """Full graph dump for a session, grouped by label."""
        if not self.available:
            return {}
        result: Dict[str, List[Dict]] = {}
        for label in (Label.GOAL, Label.TASK, Label.CONTEXT, Label.STYLE, Label.SKILL):
            result[label.lower() + "s"] = self.query_nodes_by_label(session_id, label)
        return result


# ---------------------------------------------------------------------------
# Stage 1 — Entity extractor
# ---------------------------------------------------------------------------

_ENTITY_PROMPT = """\
Extract structured user information from the following conversational turn.
Identify any WHO (skills, expertise, role), WHAT (current task or question),
WHERE (situation, environment, constraints), WHY (goals, objectives, motivations),
and HOW (communication preferences, style) entities the user has revealed.

Return ONLY a JSON object with this structure (no prose, no markdown fence):
{{
  "entities": [
    {{
      "slot": "WHY",
      "label": "Goal",
      "name": "short identifier (3-6 words)",
      "description": "one sentence",
      "extra": {{}}
    }}
  ]
}}

Valid slots: WHO, WHAT, WHERE, WHY, HOW
Valid labels: Goal (WHY), Task (WHAT), Context (WHERE), Style (HOW), Skill (WHO/HOW)
If no user information is revealed, return {{"entities": []}}

User turn:
{turn_text}"""


def entity_extractor(
    turn_text: str, llm_fn: Callable[[str], str]
) -> List[TypedEntity]:
    """Stage 1: extract typed 5W+H entities from one user turn."""
    prompt = _ENTITY_PROMPT.format(turn_text=turn_text.strip())
    raw = _llm_json_call(prompt, llm_fn)
    if raw is None:
        return []
    entities = []
    for item in raw.get("entities", []):
        try:
            slot  = str(item.get("slot", "")).upper()
            label = item.get("label", SLOT_TO_LABEL.get(slot, Label.TASK))
            name  = str(item.get("name", "")).strip()
            desc  = str(item.get("description", "")).strip()
            if not name or not desc:
                continue
            entities.append(TypedEntity(
                slot=slot,
                label=label,
                name=name,
                description=desc,
                extra=item.get("extra", {}),
            ))
        except Exception:
            continue
    return entities


# ---------------------------------------------------------------------------
# Stage 2 — Relation generator
# ---------------------------------------------------------------------------

_RELATION_PROMPT = """\
Given these newly extracted entities and the user's existing graph context,
infer any meaningful relationships between the NEW entities only.
Do not create relationships to existing nodes — that is handled separately.

New entities:
{entities_json}

Return ONLY a JSON object (no prose, no markdown fence):
{{
  "relations": [
    {{
      "from_id": "<node_id of source>",
      "to_id":   "<node_id of target>",
      "type":    "CONFLICTS_WITH | SUPPORTS | REQUIRES"
    }}
  ]
}}

If no cross-entity relations are evident, return {{"relations": []}}"""


def relation_generator(
    entities: List[TypedEntity], llm_fn: Callable[[str], str]
) -> List[TypedEdge]:
    """Stage 2: infer typed edges between the newly extracted entities."""
    if len(entities) < 2:
        return []
    entities_json = json.dumps(
        [{"node_id": e.node_id, "label": e.label, "name": e.name, "description": e.description}
         for e in entities],
        indent=2,
    )
    prompt = _RELATION_PROMPT.format(entities_json=entities_json)
    raw = _llm_json_call(prompt, llm_fn)
    if raw is None:
        return []
    valid_from_ids = {e.node_id for e in entities}
    edges = []
    for item in raw.get("relations", []):
        try:
            from_id  = str(item.get("from_id", ""))
            to_id    = str(item.get("to_id", ""))
            rel_type = str(item.get("type", "")).upper()
            if from_id not in valid_from_ids or to_id not in valid_from_ids:
                continue
            if from_id == to_id:
                continue
            edges.append(TypedEdge(from_id=from_id, to_id=to_id, rel_type=rel_type))
        except Exception:
            continue
    return edges


# ---------------------------------------------------------------------------
# Stage 3 — Conflict detector
# ---------------------------------------------------------------------------

def conflict_detector(
    entities: List[TypedEntity], session_id: str, graph_client: GraphClient
) -> List[Conflict]:
    """
    Stage 3: for each new entity, check whether a non-deprecated node of the
    same label and name already exists for this session with different attributes.
    Returns a list of conflicts — does NOT write anything.
    """
    conflicts = []
    for entity in entities:
        existing = graph_client.find_conflicting_nodes(entity, session_id)
        for node in existing:
            # Check for a meaningful attribute difference
            conflict_reason = _detect_attribute_conflict(entity, node)
            if conflict_reason:
                conflicts.append(Conflict(
                    new_entity=entity,
                    existing_node_id=node.get("node_id", ""),
                    existing_props=node,
                    reason=conflict_reason,
                ))
    return conflicts


def _detect_attribute_conflict(entity: TypedEntity, existing: Dict[str, Any]) -> Optional[str]:
    """Return a human-readable conflict reason, or None if no meaningful conflict."""
    # Level conflict (skills)
    new_level = entity.extra.get("level", "")
    old_level = existing.get("level", "")
    if new_level and old_level and new_level.lower() != old_level.lower():
        return (
            f"Skill level changed: was '{old_level}', now '{new_level}'. "
            f"Update '{entity.name}' profile?"
        )
    # Description substantially different (>50% word change as a quick heuristic)
    new_words = set(entity.description.lower().split())
    old_words = set(str(existing.get("description", "")).lower().split())
    if new_words and old_words:
        overlap = len(new_words & old_words) / max(len(new_words | old_words), 1)
        if overlap < 0.4:
            return (
                f"'{entity.name}' description has changed significantly. "
                f"Old: \"{existing.get('description', '')[:80]}\""
            )
    return None


# ---------------------------------------------------------------------------
# Stage 4 — Conditional write
# ---------------------------------------------------------------------------

def conditional_write(
    entities:     List[TypedEntity],
    edges:        List[TypedEdge],
    conflicts:    List[Conflict],
    session_id:   str,
    graph_client: GraphClient,
) -> WriteResult:
    """
    Stage 4: write entities and edges, handling conflicts with DEPRECATED_BY edges.
    Conflicts are surfaced via WriteResult.conflicts — never silently resolved.
    Never deletes any node or edge.
    """
    if not graph_client.available:
        return WriteResult(skipped=True)

    conflicted_ids = {c.new_entity.node_id for c in conflicts}
    nodes_written = 0
    edges_written = 0

    graph_client.ensure_user_node(session_id)

    for entity in entities:
        graph_client.create_entity_node(entity, session_id)
        nodes_written += 1

        # If this entity has a conflict with an existing node, mark the old
        # node deprecated (pointing to the new one) — but surface the conflict
        # in the result so the caller can notify the user.
        if entity.node_id in conflicted_ids:
            conflict = next(c for c in conflicts if c.new_entity.node_id == entity.node_id)
            if conflict.existing_node_id:
                graph_client.deprecate_node(
                    old_node_id=conflict.existing_node_id,
                    new_node_id=entity.node_id,
                    reason=conflict.reason,
                )

    for edge in edges:
        try:
            graph_client.create_edge(edge)
            edges_written += 1
        except Exception as exc:
            logger.debug("Edge write failed (non-fatal): %s", exc)

    return WriteResult(
        entities_written=nodes_written,
        edges_written=edges_written,
        conflicts=conflicts,
    )


# ---------------------------------------------------------------------------
# Public write pipeline entrypoint
# ---------------------------------------------------------------------------

def write_pipeline(
    turn_text:    str,
    session_id:   str,
    graph_client: GraphClient,
    llm_fn:       Callable[[str], str],
) -> WriteResult:
    """
    Run the full 4-stage Mem0g write pipeline for one user turn.
    Safe to call even when GraphClient.available is False — returns skipped=True.
    """
    if not graph_client.available:
        return WriteResult(skipped=True)

    entities  = entity_extractor(turn_text, llm_fn)
    if not entities:
        return WriteResult()

    edges     = relation_generator(entities, llm_fn)
    conflicts = conflict_detector(entities, session_id, graph_client)
    result    = conditional_write(entities, edges, conflicts, session_id, graph_client)

    if conflicts:
        logger.info(
            "UserModelling [%s]: %d conflict(s) detected — surfacing to user",
            session_id, len(conflicts),
        )

    return result


# ---------------------------------------------------------------------------
# Retrieval gating
# ---------------------------------------------------------------------------

_SLOT_RELEVANCE_PROMPT = """\
Given the following user query, identify which 5W+H user profile slots are
relevant to answering it well. Only select slots where the user's personal
context would materially change the response.

Slots: WHO (skills/expertise), WHAT (current task), WHERE (situation/constraints),
WHY (goals/objectives), HOW (communication preferences)

Return ONLY a JSON object (no prose, no markdown fence):
{{
  "relevant_slots": ["WHY", "WHO"]
}}

If no personal context is needed (e.g. factual/math questions), return:
{{
  "relevant_slots": []
}}

Query: {query_text}"""


def retrieve_for_query(
    query_text:   str,
    session_id:   str,
    graph_client: GraphClient,
    llm_fn:       Callable[[str], str],
    max_nodes:    int = 10,
    subgraph_depth: int = 2,
) -> UserContext:
    """
    Anti-over-personalisation gate. Only fetches user context when relevant.

    Returns an empty UserContext when no slots are relevant — the inference
    server then generates without any user-profile injection.
    """
    empty = UserContext(session_id=session_id, relevant_slots=[],
                        goals=[], tasks=[], skills=[], styles=[], contexts=[])

    if not graph_client.available:
        return empty

    # Classify which slots are relevant
    prompt = _SLOT_RELEVANCE_PROMPT.format(query_text=query_text.strip())
    raw = _llm_json_call(prompt, llm_fn)
    relevant_slots: List[str] = []
    if raw:
        relevant_slots = [s.upper() for s in raw.get("relevant_slots", [])]

    if not relevant_slots:
        return empty

    # Fetch subgraph for relevant slots only
    label_map = {
        "WHY":  (Label.GOAL,    "goals"),
        "WHAT": (Label.TASK,    "tasks"),
        "WHO":  (Label.SKILL,   "skills"),
        "HOW":  (Label.STYLE,   "styles"),
        "WHERE":(Label.CONTEXT, "contexts"),
    }

    ctx_dict: Dict[str, List[Dict]] = {
        "goals": [], "tasks": [], "skills": [], "styles": [], "contexts": []
    }
    total_nodes = 0

    for slot in relevant_slots:
        if slot not in label_map:
            continue
        label, key = label_map[slot]
        nodes = graph_client.query_nodes_by_label(session_id, label)
        for node in nodes:
            if total_nodes >= max_nodes:
                break
            ctx_dict[key].append(node)
            total_nodes += 1

    return UserContext(
        session_id=session_id,
        relevant_slots=relevant_slots,
        raw_nodes=total_nodes,
        **ctx_dict,
    )


# ---------------------------------------------------------------------------
# Scrutability handlers (registered on the FastAPI server in Phase 4)
# ---------------------------------------------------------------------------

def inspect_memory(session_id: str, graph_client: GraphClient) -> Dict[str, Any]:
    """
    GET /memory/inspect/{session_id}
    Returns a human-readable NL summary of all non-deprecated beliefs plus
    the raw structured graph for developers.
    """
    if not graph_client.available:
        return {"session_id": session_id, "available": False,
                "summary": "User modelling is not enabled.", "graph": {}}

    graph_data = graph_client.get_all_nodes(session_id)
    summary_lines = [f"Here is what I currently know about session {session_id[:8]}…:"]

    label_titles = {
        "goals": "Goals", "tasks": "Current tasks", "skills": "Skills",
        "styles": "Preferences", "contexts": "Context / environment",
    }
    for key, title in label_titles.items():
        nodes = graph_data.get(key, [])
        if not nodes:
            continue
        items = []
        for n in nodes:
            text = n.get("description", n.get("name", ""))
            if n.get("contested"):
                text += " [contested — not used until corrected]"
            elif n.get("user_corrected"):
                text += " [user-corrected]"
            items.append(f"  • {text}")
        summary_lines.append(f"{title}:\n" + "\n".join(items))

    if len(summary_lines) == 1:
        summary_lines.append("  Nothing recorded yet.")

    return {
        "session_id": session_id,
        "available":  True,
        "summary":    "\n\n".join(summary_lines),
        "graph":      graph_data,
    }


def contest_belief(
    session_id:   str,
    node_id:      str,
    graph_client: GraphClient,
) -> Dict[str, Any]:
    """
    POST /memory/contest
    Marks a belief node as contested. The retrieval gate will not inject
    contested nodes until the user resolves them via correct_belief().
    """
    if not graph_client.available:
        return {"ok": False, "reason": "User modelling is not enabled."}

    graph_client.mark_contested(node_id)
    return {
        "ok":     True,
        "node_id": node_id,
        "message": (
            "This belief has been flagged as contested and will not be used "
            "to personalise responses until you correct it."
        ),
    }


def correct_belief(
    session_id:   str,
    old_node_id:  str,
    correction:   str,
    label:        str,
    graph_client: GraphClient,
    llm_fn:       Callable[[str], str],
) -> Dict[str, Any]:
    """
    POST /memory/correct
    Applies a user-supplied correction: creates a new node from the correction
    text (single entity extraction pass), marks the old node as USER_CORRECTED,
    and points to the new node.
    """
    if not graph_client.available:
        return {"ok": False, "reason": "User modelling is not enabled."}

    # Extract a single entity from the correction text
    new_entities = entity_extractor(correction, llm_fn)
    if not new_entities:
        # Fall back to a simple stub entity from the correction text
        new_entity = TypedEntity(
            slot="",
            label=label,
            name=f"corrected_{old_node_id[:8]}",
            description=correction.strip(),
        )
    else:
        new_entity = new_entities[0]
        new_entity.label = label  # honour the caller's label

    graph_client.create_entity_node(new_entity, session_id)
    graph_client.mark_user_corrected(old_node_id, new_entity.node_id)

    return {
        "ok":          True,
        "old_node_id": old_node_id,
        "new_node_id": new_entity.node_id,
        "message":     (
            f"Correction applied. The old belief is now archived with a "
            f"USER_CORRECTED link to the new one, so the full history is preserved."
        ),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _llm_json_call(
    prompt: str, llm_fn: Callable[[str], str]
) -> Optional[Dict[str, Any]]:
    """
    Call llm_fn with prompt, extract the first JSON object from the response.
    Returns None on any failure — callers must handle gracefully.
    """
    try:
        raw = llm_fn(prompt)
    except Exception as exc:
        logger.debug("llm_fn call failed: %s", exc)
        return None

    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"```", "", raw)

    # Find the first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logger.debug("No JSON object found in LLM response")
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        logger.debug("JSON parse failed: %s", exc)
        return None


def _node_to_dict(node: Any) -> Dict[str, Any]:
    """Convert a FalkorDB Node object to a plain dict."""
    try:
        return dict(node.properties)
    except AttributeError:
        # Already a dict (e.g. in tests)
        return dict(node) if node else {}
