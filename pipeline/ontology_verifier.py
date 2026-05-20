"""
Ontology Verifier (Post-hoc)
============================
Implements Experiment 6 Approach B: after every assistant response, extract
factual claims and verify them against a loaded knowledge base.

Two knowledge-base backends are supported:
  1. Local OWL/RDF file — loaded with rdflib, queried via SPARQL
  2. Remote SPARQL endpoint — queried via SPARQLWrapper (e.g. DBpedia, Wikidata)

The backend is chosen from config:
  ONTOLOGY_PATH              → rdflib loads this OWL file
  ONTOLOGY_SPARQL_ENDPOINT   → if non-empty, overrides local file

Setup (local OWL):
    Place any valid OWL/RDF file at data/ontology.owl
    pip install rdflib

Setup (DBpedia endpoint):
    pip install SPARQLWrapper
    Set PIPELINE_ONTOLOGY_SPARQL_ENDPOINT=https://dbpedia.org/sparql

Wiring (Phase 4 — 3_infererence.py):
    from ontology_verifier import OntologyGraph, score_response

    _onto_graph = OntologyGraph(cfg)   # once at server startup, inside lifespan

    if cfg.ENABLE_ONTOLOGY_VERIF:
        onto_score = score_response(response_text, _onto_graph, llm_fn)
        # onto_score.ontology_score is added to the response metadata

Public API:
    OntologyGraph(cfg)
    extract_claims(response_text, llm_fn) -> list[str]
    verify_claim(claim, graph, llm_fn)    -> VerificationResult
    score_response(response_text, graph, llm_fn) -> OntologyScore
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common RDF/SPARQL namespace prefixes injected into every generated query
# ---------------------------------------------------------------------------

COMMON_PREFIXES = """\
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:    <http://www.w3.org/2002/07/owl#>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
PREFIX schema: <https://schema.org/>
PREFIX dbo:    <http://dbpedia.org/ontology/>
PREFIX dbr:    <http://dbpedia.org/resource/>
PREFIX dbp:    <http://dbpedia.org/property/>
PREFIX wd:     <http://www.wikidata.org/entity/>
PREFIX wdt:    <http://www.wikidata.org/prop/direct/>
"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    claim:      str
    verified:   bool
    confidence: float       # 0.0 – 1.0
    evidence:   Optional[str] = None
    # method explains HOW the result was obtained
    method:     str = "unknown"
    # raw_sparql is kept for explainability (logged, not sent to user)
    raw_sparql: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OntologyScore:
    ontology_score:    float               # mean confidence across all claims
    total_claims:      int
    verified_count:    int
    unverified_claims: List[str]           = field(default_factory=list)
    results:           List[VerificationResult] = field(default_factory=list)
    skipped:           bool = False        # True when ENABLE_ONTOLOGY_VERIF=False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def stub(cls) -> "OntologyScore":
        """Neutral stub returned when the flag is off."""
        return cls(ontology_score=1.0, total_claims=0, verified_count=0, skipped=True)


# ---------------------------------------------------------------------------
# OntologyGraph — knowledge-base wrapper
# ---------------------------------------------------------------------------

class OntologyGraph:
    """
    Wraps either a local rdflib graph or a remote SPARQL endpoint.

    available=False when neither is configured or reachable — all query
    methods return empty results rather than crashing the server.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg       = cfg
        self.available = False
        self._graph    = None          # rdflib.Graph or None
        self._endpoint = None          # SPARQLWrapper or None
        self._prefixes = COMMON_PREFIXES
        self._extra_prefixes: str = "" # ontology-specific prefixes extracted at load time

        if not cfg.ENABLE_ONTOLOGY_VERIF:
            return

        endpoint = getattr(cfg, "ONTOLOGY_SPARQL_ENDPOINT", "").strip()
        if endpoint:
            self._init_remote(endpoint)
        else:
            self._init_local(cfg.ONTOLOGY_PATH)

    # ------------------------------------------------------------------
    def _init_local(self, path: str) -> None:
        try:
            import rdflib  # type: ignore
        except ImportError:
            logger.warning("OntologyVerifier: pip install rdflib  to use local OWL files")
            return

        from pathlib import Path
        owl = Path(path)
        if not owl.exists():
            logger.warning(
                "OntologyVerifier: file not found at '%s' — "
                "place a valid OWL/RDF file there or set ONTOLOGY_SPARQL_ENDPOINT", path
            )
            return

        try:
            g = rdflib.ConjunctiveGraph()
            g.parse(str(owl))
            self._graph = g
            self._extra_prefixes = _extract_prefixes(g)
            self.available = True
            logger.info(
                "OntologyVerifier: loaded %d triples from %s", len(g), path
            )
        except Exception as exc:
            logger.warning("OntologyVerifier: failed to parse OWL file — %s", exc)

    def _init_remote(self, endpoint: str) -> None:
        try:
            from SPARQLWrapper import SPARQLWrapper, JSON  # type: ignore
            w = SPARQLWrapper(endpoint)
            w.setReturnFormat(JSON)
            w.setTimeout(self.cfg.ONTOLOGY_SPARQL_TIMEOUT)
            # Probe with a trivial query
            w.setQuery("SELECT * WHERE { ?s ?p ?o } LIMIT 1")
            w.query()
            self._endpoint = w
            self.available = True
            logger.info("OntologyVerifier: connected to remote endpoint %s", endpoint)
        except ImportError:
            logger.warning("OntologyVerifier: pip install SPARQLWrapper  for remote endpoints")
        except Exception as exc:
            logger.warning("OntologyVerifier: remote endpoint not reachable — %s", exc)

    # ------------------------------------------------------------------
    @property
    def full_prefixes(self) -> str:
        return self._prefixes + self._extra_prefixes

    def query(self, sparql: str) -> List[Dict[str, str]]:
        """Execute a SPARQL SELECT. Returns list of row dicts, empty list on error."""
        if not self.available:
            return []
        # Prepend prefixes unless the query already declares them
        if "PREFIX" not in sparql.upper():
            sparql = self.full_prefixes + sparql

        try:
            if self._endpoint is not None:
                return _run_remote(self._endpoint, sparql, self.cfg.ONTOLOGY_SPARQL_TIMEOUT)
            if self._graph is not None:
                return _run_local(self._graph, sparql, self.cfg.ONTOLOGY_SPARQL_TIMEOUT)
        except Exception as exc:
            logger.debug("SPARQL query failed: %s", exc)
        return []

    def ask(self, sparql: str) -> bool:
        """Execute a SPARQL ASK. Returns False on error."""
        if not self.available:
            return False
        if "PREFIX" not in sparql.upper():
            sparql = self.full_prefixes + sparql
        try:
            if self._endpoint is not None:
                return _ask_remote(self._endpoint, sparql)
            if self._graph is not None:
                return _ask_local(self._graph, sparql)
        except Exception as exc:
            logger.debug("SPARQL ASK failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
Extract all atomic, verifiable factual claims from the text below.
An atomic factual claim is a single sentence that states something that can be
true or false independently — e.g. "Paris is the capital of France",
"Python supports multiple inheritance", "Water freezes at 0°C at 1 atm".

Exclude: opinions, hedged statements ("might", "could"), questions,
instructions, and meta-commentary about the response itself.

Return ONLY a JSON object (no prose, no markdown):
{{"claims": ["claim 1", "claim 2"]}}

If no verifiable claims are present: {{"claims": []}}

Text:
{text}"""


def extract_claims(
    response_text: str,
    llm_fn: Callable[[str], str],
    max_claims: int = 8,
) -> List[str]:
    """
    Extract atomic factual claims from an assistant response using the LLM.
    Falls back to empty list on any failure.
    """
    # Strip XML tags first — we want only the answer content
    text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    text = re.sub(r"<answer>(.*?)</answer>", r"\1", text, flags=re.DOTALL).strip()
    if not text:
        return []

    prompt = _EXTRACT_PROMPT.format(text=text[:3000])
    raw = _llm_json_call(prompt, llm_fn)
    if raw is None:
        return []

    claims = [str(c).strip() for c in raw.get("claims", []) if str(c).strip()]
    return claims[:max_claims]


# ---------------------------------------------------------------------------
# SPARQL generation
# ---------------------------------------------------------------------------

_SPARQL_GEN_PROMPT = """\
Generate a SPARQL SELECT query to verify the following factual claim against
an RDF knowledge base. The query should return at least one row if the claim
is true, zero rows if false.

Claim: {claim}

Available namespace prefixes:
{prefixes}

Rules:
- Use FILTER, OPTIONAL, or VALUES to match the claim as precisely as possible.
- Prefer LABEL-based lookups (rdfs:label) when querying by name.
- Keep the query simple — one SELECT with at most 2-3 triple patterns.
- Return ONLY a JSON object (no prose, no markdown):
  {{"sparql": "SELECT ?x WHERE {{ ... }}", "explanation": "what this checks"}}
- If the claim cannot be expressed as SPARQL (opinion, procedural, hypothetical):
  {{"sparql": null, "explanation": "reason"}}"""


def _generate_sparql(
    claim: str,
    graph: OntologyGraph,
    llm_fn: Callable[[str], str],
) -> Optional[str]:
    """Ask the LLM to write a SPARQL query for this claim. Returns None if not expressible."""
    prompt = _SPARQL_GEN_PROMPT.format(
        claim=claim,
        prefixes=graph.full_prefixes.strip(),
    )
    raw = _llm_json_call(prompt, llm_fn)
    if raw is None:
        return None
    sparql = raw.get("sparql")
    if not sparql:
        return None
    return str(sparql).strip()


# ---------------------------------------------------------------------------
# Claim verification
# ---------------------------------------------------------------------------

def verify_claim(
    claim:   str,
    graph:   OntologyGraph,
    llm_fn:  Callable[[str], str],
    timeout: int = 5,
) -> VerificationResult:
    """
    Verify a single factual claim against the knowledge base.

    Strategy:
      1. Generate SPARQL via LLM.
      2. Execute as SELECT — if rows returned → verified (confidence = 0.9).
      3. Execute as ASK (reformulated) — if True → verified (confidence = 0.8).
      4. Zero results → not verified (confidence = 0.2).
      5. SPARQL not generatable / graph not available → unverifiable (confidence = 0.5).
    """
    if not graph.available:
        return VerificationResult(
            claim=claim, verified=False, confidence=0.5,
            method="graph_unavailable",
            evidence="Knowledge base not loaded — cannot verify.",
        )

    sparql = _generate_sparql(claim, graph, llm_fn)

    if sparql is None:
        return VerificationResult(
            claim=claim, verified=False, confidence=0.5,
            method="unverifiable",
            evidence="This claim type cannot be expressed as a SPARQL query.",
        )

    # Try SELECT first
    try:
        rows = graph.query(sparql)
        if rows:
            evidence = _format_evidence(rows[:3])
            return VerificationResult(
                claim=claim, verified=True, confidence=0.9,
                evidence=evidence, method="sparql_select_match",
                raw_sparql=sparql,
            )
    except Exception:
        pass

    # Try rewriting as ASK
    ask_sparql = _select_to_ask(sparql)
    if ask_sparql:
        try:
            result = graph.ask(ask_sparql)
            if result:
                return VerificationResult(
                    claim=claim, verified=True, confidence=0.8,
                    evidence="ASK query returned true.", method="sparql_ask_match",
                    raw_sparql=ask_sparql,
                )
        except Exception:
            pass

    # Query ran but returned nothing
    return VerificationResult(
        claim=claim, verified=False, confidence=0.2,
        evidence="No matching triples found in the knowledge base.",
        method="sparql_no_match", raw_sparql=sparql,
    )


# ---------------------------------------------------------------------------
# Response scoring
# ---------------------------------------------------------------------------

def score_response(
    response_text: str,
    graph:         OntologyGraph,
    llm_fn:        Callable[[str], str],
    max_claims:    int = 8,
) -> OntologyScore:
    """
    Extract all factual claims from a response and verify each against the
    knowledge base.  Returns an OntologyScore with per-claim breakdown.

    When ENABLE_ONTOLOGY_VERIF=False the caller should not call this function,
    but OntologyScore.stub() is returned defensively if graph is unavailable.
    """
    if not graph.available:
        return OntologyScore.stub()

    claims = extract_claims(response_text, llm_fn, max_claims=max_claims)
    if not claims:
        return OntologyScore(
            ontology_score=1.0, total_claims=0, verified_count=0,
        )

    results: List[VerificationResult] = []
    for claim in claims:
        r = verify_claim(claim, graph, llm_fn)
        results.append(r)
        logger.debug(
            "Claim [%s=%.2f | %s]: %s",
            "✓" if r.verified else "✗", r.confidence, r.method, claim[:60],
        )

    verified_count    = sum(1 for r in results if r.verified)
    unverified_claims = [r.claim for r in results if not r.verified]
    mean_confidence   = round(sum(r.confidence for r in results) / len(results), 4)

    logger.info(
        "OntologyVerifier: %d/%d claims verified | score=%.3f",
        verified_count, len(claims), mean_confidence,
    )

    return OntologyScore(
        ontology_score=mean_confidence,
        total_claims=len(claims),
        verified_count=verified_count,
        unverified_claims=unverified_claims,
        results=results,
    )


# ---------------------------------------------------------------------------
# SPARQL execution helpers
# ---------------------------------------------------------------------------

def _run_local(graph: Any, sparql: str, timeout: int) -> List[Dict[str, str]]:
    """Execute SELECT on a rdflib graph. Returns list of row dicts."""
    import signal  # noqa: PLC0415

    def _handler(signum, frame):
        raise TimeoutError("SPARQL timeout")

    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)
    except (AttributeError, OSError):
        pass  # SIGALRM not available on Windows

    try:
        result = graph.query(sparql)
        rows = []
        for row in result:
            rows.append({str(var): str(val) for var, val in zip(result.vars, row)})
        return rows
    finally:
        try:
            signal.alarm(0)
        except (AttributeError, OSError):
            pass


def _ask_local(graph: Any, sparql: str) -> bool:
    result = graph.query(sparql)
    return bool(result.askAnswer)


def _run_remote(wrapper: Any, sparql: str, timeout: int) -> List[Dict[str, str]]:
    """Execute SELECT on a remote SPARQLWrapper endpoint."""
    wrapper.setQuery(sparql)
    wrapper.setTimeout(timeout)
    results = wrapper.query().convert()
    rows = []
    for binding in results.get("results", {}).get("bindings", []):
        row = {k: v.get("value", "") for k, v in binding.items()}
        rows.append(row)
    return rows


def _ask_remote(wrapper: Any, sparql: str) -> bool:
    from SPARQLWrapper import JSON  # type: ignore  # noqa: PLC0415
    wrapper.setQuery(sparql)
    wrapper.setReturnFormat(JSON)
    result = wrapper.query().convert()
    return bool(result.get("boolean", False))


def _select_to_ask(sparql: str) -> Optional[str]:
    """Best-effort rewrite of a SELECT query to ASK form."""
    s = sparql.strip()
    if re.match(r"^\s*ASK\b", s, re.IGNORECASE):
        return s
    # Match SELECT ... WHERE { ... }
    m = re.search(r"WHERE\s*\{(.+)\}", s, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    where_body = m.group(1)
    prefixes = re.findall(r"PREFIX\s+\S+\s+<[^>]+>", s, re.IGNORECASE)
    prefix_block = "\n".join(prefixes) + "\n" if prefixes else ""
    return f"{prefix_block}ASK {{ {where_body} }}"


def _format_evidence(rows: List[Dict[str, str]]) -> str:
    """Format the first few result rows as a readable evidence string."""
    parts = []
    for row in rows:
        parts.append(", ".join(f"{k}={v[:60]}" for k, v in row.items()))
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Ontology-specific prefix extraction
# ---------------------------------------------------------------------------

def _extract_prefixes(graph: Any) -> str:
    """Extract namespace bindings from a loaded rdflib graph as PREFIX declarations."""
    lines = []
    try:
        for prefix, ns in graph.namespaces():
            if prefix and str(prefix) not in ("rdf", "rdfs", "owl", "xsd"):
                lines.append(f"PREFIX {prefix}: <{ns}>")
    except Exception:
        pass
    return "\n".join(lines) + "\n" if lines else ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _llm_json_call(
    prompt: str, llm_fn: Callable[[str], str]
) -> Optional[Dict[str, Any]]:
    """Call llm_fn, extract first JSON object. Returns None on failure."""
    try:
        raw = llm_fn(prompt)
    except Exception as exc:
        logger.debug("llm_fn call failed: %s", exc)
        return None

    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"```", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None
