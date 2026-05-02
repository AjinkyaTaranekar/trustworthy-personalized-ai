"""
Pipeline Configuration
======================
Single source of truth for all feature flags and module settings.
Every pipeline script imports the module-level `cfg` singleton.

Usage
-----
    from config import cfg

    if cfg.ENABLE_USER_MODELLING:
        import user_modelling
        ...

Override at runtime via environment variables (prefix PIPELINE_):
    PIPELINE_ENABLE_GRPO=true python 2_model_trainer.py --mode grpo

Override via YAML config file (keys match field names exactly):
    python 3_infererence.py --config my_config.yaml

Dependency rules (enforced by cfg.validate()):
    ENABLE_GRPO            requires ENABLE_SFT to have produced checkpoint_sft
    ENABLE_PERSONALISATION requires ENABLE_USER_MODELLING
    ENABLE_EMPATHY         requires data/appraisal_labels.jsonl to exist
    ENABLE_ONTOLOGY_VERIF  requires ONTOLOGY_PATH to point to a valid OWL file
"""

import os
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class PipelineConfig:
    # ── Training flags ─────────────────────────────────────────────────────────
    # ENABLE_SFT: Phase 1 supervised fine-tuning. Always on — the constitutional
    # baseline. Produces models/checkpoint_sft/ which is the reference policy for GRPO.
    ENABLE_SFT: bool = True

    # ENABLE_GRPO: Phase 2 GRPO/DAPO RL training loop.
    # Hard dependency: checkpoint_sft must exist before this runs.
    ENABLE_GRPO: bool = False

    # ── Runtime module flags ───────────────────────────────────────────────────
    # ENABLE_USER_MODELLING: FalkorDB 5W+H graph, Mem0g write pipeline, retrieval
    # gating, and scrutability endpoints (/memory/inspect|contest|correct).
    # Requires FalkorDB on FALKORDB_HOST:FALKORDB_PORT (docker compose up -d).
    ENABLE_USER_MODELLING: bool = False

    # ENABLE_EMPATHY: Appraisal-conditioned generation. Qwen fine-tuned on
    # AppraisePLM-labelled EmpatheticDialogues data produces an <appraisal> block
    # in its <think> chain. No external model at inference time.
    # Requires: data/appraisal_labels.jsonl (run appraisal_labeller.py first).
    ENABLE_EMPATHY: bool = False

    # ENABLE_PERSONALISATION: Per-query 5W+H slot relevance gate. Only injects
    # user context when a schema slot is relevant to the current query, preventing
    # the context-inflation degradation documented by OP-Bench (26-61% task degradation).
    # Hard dependency: ENABLE_USER_MODELLING must be True.
    ENABLE_PERSONALISATION: bool = False

    # ENABLE_ONTOLOGY_VERIF: Post-hoc SPARQL claim scorer. After each assistant
    # response, extracts factual claims and verifies them against a loaded OWL
    # ontology. Implements Experiment 6 Approach B (post-hoc verifier).
    # Requires: ONTOLOGY_PATH points to a valid OWL file.
    ENABLE_ONTOLOGY_VERIF: bool = False

    # ── FalkorDB connection ────────────────────────────────────────────────────
    FALKORDB_HOST: str = "localhost"
    FALKORDB_PORT: int = 6379
    FALKORDB_GRAPH_NAME: str = "user_model"

    # ── Empathy settings ───────────────────────────────────────────────────────
    APPRAISAL_LABELS_PATH: str = "data/appraisal_labels.jsonl"
    APPRAISE_PLM_REPO: str = "appraise-PLM"
    # Top-N appraisal dimensions included in the <appraisal> block
    APPRAISAL_TOP_N: int = 3
    # Thresholds for selecting emotionally significant examples from EmpatheticDialogues:
    #   include turn if |valence - 0.5| > VALENCE_THRESHOLD
    #                 or coping_potential < COPING_THRESHOLD
    #                 or goal_relevance > GOAL_THRESHOLD
    APPRAISAL_VALENCE_THRESHOLD: float = 0.3
    APPRAISAL_COPING_THRESHOLD: float = 0.4
    APPRAISAL_GOAL_THRESHOLD: float = 0.7

    # ── Ontology verifier settings ─────────────────────────────────────────────
    ONTOLOGY_PATH: str = "data/ontology.owl"
    # Optional remote SPARQL endpoint (e.g. "https://dbpedia.org/sparql").
    # When set, OntologyGraph queries this endpoint instead of the local OWL file.
    # Requires: pip install SPARQLWrapper
    ONTOLOGY_SPARQL_ENDPOINT: str = ""
    ONTOLOGY_SPARQL_TIMEOUT: int = 5  # seconds per SPARQL query
    # Max factual claims extracted per response (prevents excessive SPARQL calls)
    ONTOLOGY_MAX_CLAIMS: int = 8

    # ── Retrieval gating ───────────────────────────────────────────────────────
    # Depth of the subgraph fetched from FalkorDB for relevant 5W+H slots
    RETRIEVAL_SUBGRAPH_DEPTH: int = 2
    # Hard cap on injected graph nodes per query (anti-context-inflation guard)
    RETRIEVAL_MAX_NODES: int = 10

    # ── Paths ──────────────────────────────────────────────────────────────────
    SFT_CHECKPOINT_DIR: str = "models/checkpoint_sft"
    GRPO_CHECKPOINT_DIR: str = "models/checkpoint_grpo_d"
    DATA_DIR: str = "data"
    REPORTS_DIR: str = "reports"

    # --------------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Check dependency rules. Returns violation messages; empty list = valid."""
        issues: list[str] = []

        if self.ENABLE_GRPO and not self.ENABLE_SFT:
            issues.append(
                "ENABLE_GRPO=True requires ENABLE_SFT=True "
                "(GRPO uses checkpoint_sft as its reference policy)"
            )
        if self.ENABLE_PERSONALISATION and not self.ENABLE_USER_MODELLING:
            issues.append(
                "ENABLE_PERSONALISATION=True requires ENABLE_USER_MODELLING=True"
            )
        if self.ENABLE_EMPATHY and not Path(self.APPRAISAL_LABELS_PATH).exists():
            issues.append(
                f"ENABLE_EMPATHY=True requires {self.APPRAISAL_LABELS_PATH} — "
                "run: python pipeline/appraisal_labeller.py"
            )
        if self.ENABLE_ONTOLOGY_VERIF and not Path(self.ONTOLOGY_PATH).exists():
            issues.append(
                f"ENABLE_ONTOLOGY_VERIF=True requires {self.ONTOLOGY_PATH} — "
                "place a valid OWL file at this path"
            )
        if self.ENABLE_USER_MODELLING:
            # Warn if FalkorDB is not reachable (soft check — just a warning)
            try:
                import socket
                s = socket.create_connection((self.FALKORDB_HOST, self.FALKORDB_PORT), timeout=2)
                s.close()
            except OSError:
                issues.append(
                    f"ENABLE_USER_MODELLING=True but FalkorDB is not reachable at "
                    f"{self.FALKORDB_HOST}:{self.FALKORDB_PORT} — run: docker compose up -d"
                )

        return issues

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Build config from defaults, then override with PIPELINE_<NAME> env vars."""
        instance = cls()
        _bool  = {f.name for f in fields(cls) if f.type in ("bool",  "bool")}
        _str   = {f.name for f in fields(cls) if f.type == "str"}
        _int   = {f.name for f in fields(cls) if f.type == "int"}
        _float = {f.name for f in fields(cls) if f.type == "float"}

        for name in _bool:
            val = os.environ.get(f"PIPELINE_{name}")
            if val is not None:
                setattr(instance, name, val.strip().lower() in ("1", "true", "yes"))
        for name in _str:
            val = os.environ.get(f"PIPELINE_{name}")
            if val is not None:
                setattr(instance, name, val)
        for name in _int:
            val = os.environ.get(f"PIPELINE_{name}")
            if val is not None:
                setattr(instance, name, int(val))
        for name in _float:
            val = os.environ.get(f"PIPELINE_{name}")
            if val is not None:
                setattr(instance, name, float(val))

        return instance

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load config from a YAML file. Keys must match field names exactly."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml  to use YAML config files")

        instance = cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}

        type_map: dict[str, set[str]] = {
            "bool":  {f.name for f in fields(cls) if f.type == "bool"},
            "str":   {f.name for f in fields(cls) if f.type == "str"},
            "int":   {f.name for f in fields(cls) if f.type == "int"},
            "float": {f.name for f in fields(cls) if f.type == "float"},
        }
        converters = {"bool": bool, "str": str, "int": int, "float": float}

        for key, val in raw.items():
            for typ, names in type_map.items():
                if key in names:
                    setattr(instance, key, converters[typ](val))
                    break

        return instance

    def summary(self) -> str:
        """One-line flag summary for preflight and startup logs."""
        bool_fields = [f.name for f in fields(self) if f.type == "bool"]
        on  = [n for n in bool_fields if getattr(self, n)]
        off = [n for n in bool_fields if not getattr(self, n)]
        return (
            f"  ENABLED : {', '.join(on)  or 'none'}\n"
            f"  DISABLED: {', '.join(off) or 'none'}"
        )


# ---------------------------------------------------------------------------
# Module-level singleton — the only instance scripts should import
# ---------------------------------------------------------------------------

cfg = PipelineConfig.from_env()
