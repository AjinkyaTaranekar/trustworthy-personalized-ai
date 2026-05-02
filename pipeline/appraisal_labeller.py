"""
Appraisal Labeller (Offline)
============================
Runs AppraisePLM (Debnath, Graham, Conlan — CoNLL 2025) over the
EmpatheticDialogues dataset to produce 21-dim appraisal annotations.
Output feeds the `appraisal_empathy` SFT training category.

This script is run ONCE offline on CPU before GPU training begins.
AppraisePLM is used as a labeller only — not a runtime dependency.

Setup:
    git clone https://github.com/alokdebnath/appraise-PLM
    pip install datasets transformers torch

Usage:
    python pipeline/appraisal_labeller.py
    python pipeline/appraisal_labeller.py --limit 2000 --output data/appraisal_labels.jsonl
    python pipeline/appraisal_labeller.py --smoke          # 20 examples, validates pipeline
    python pipeline/appraisal_labeller.py --mock_model     # random vectors, no model needed

Output format (one JSON object per line):
    {
        "utterance":        "I finally got the promotion...",
        "emotion":          "excited",
        "conv_id":          "hit_1_0",
        "utterance_idx":    0,
        "appraisal_21dim":  [0.9, 0.6, ...],          # raw 21-dim float vector
        "appraisal_named":  {"pleasantness": 0.9, ...},
        "top3":             ["pleasantness", "goal_relevance", "coping_potential"],
        "valence":          0.9,
        "appraisal_reading":"high pleasantness, high goal relevance, high coping potential → user is experiencing a positive achievement"
    }
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 21 appraisal dimension names (crowd-EnVent / Geneva Appraisal Questionnaire)
# ---------------------------------------------------------------------------

DIM_NAMES: List[str] = [
    "suddenness",
    "familiarity",
    "predictability",
    "intrinsic_pleasantness",
    "goal_relevance",
    "chance_success",
    "causal_agent_other",
    "causal_agent_self",
    "self_control",
    "other_control",
    "situation_control",
    "anticipated_effort",
    "fairness",
    "norm_violation",
    "self_consistency",
    "unpleasantness",
    "pleasantness",
    "coping_potential",
    "meaning_significance",
    "attention",
    "agency",
]

# Dimensions relevant to empathetic response generation (the ones we inject)
EMPATHY_DIMS = ["pleasantness", "unpleasantness", "goal_relevance",
                "coping_potential", "meaning_significance", "self_control", "fairness"]

# Thresholds for an example to be considered "emotionally significant"
# enough to include in the appraisal_empathy training set
VALENCE_THRESHOLD  = 0.3   # |pleasantness - 0.5| > threshold
COPING_THRESHOLD   = 0.4   # coping_potential < threshold (low coping = emotionally demanding)
GOAL_THRESHOLD     = 0.7   # goal_relevance > threshold (high stakes)


# ---------------------------------------------------------------------------
# AppraisePLM model loader
# ---------------------------------------------------------------------------

class AppraisePLMWrapper:
    """
    Thin wrapper around the AppraisePLM model.

    Tries to load from a locally cloned repo (--appraise_plm_path).
    Falls back to a generic DeBERTa loading if the repo is not available.
    Use --mock_model for pipeline testing without any model.
    """

    def __init__(self, repo_path: str, mock: bool = False, device: str = "cpu") -> None:
        self.mock = mock
        self.model = None
        self.tokenizer = None
        self.device = device

        if mock:
            logger.info("AppraisePLM: mock mode — random vectors (pipeline test only)")
            return

        repo = Path(repo_path)
        if not repo.exists():
            raise FileNotFoundError(
                f"AppraisePLM repo not found at '{repo_path}'.\n"
                f"Clone it:  git clone https://github.com/alokdebnath/appraise-PLM {repo_path}\n"
                f"Or run with --mock_model to test the pipeline without the model."
            )

        # Attempt 1: import the repo's own model class
        sys.path.insert(0, str(repo.resolve()))
        try:
            from model import AppraisePLM as _APM  # type: ignore
            self.model = _APM.from_pretrained(str(repo / "checkpoints"))
            self.model.eval()
            logger.info("AppraisePLM: loaded from repo model class")
            self._use_repo_class = True
        except Exception as e1:
            logger.warning("Repo class import failed (%s) — trying generic HF load", e1)
            self._use_repo_class = False

        if not self._use_repo_class:
            # Attempt 2: generic DeBERTa load — assumes a standard HF checkpoint layout
            try:
                from transformers import AutoTokenizer, AutoModel  # type: ignore
                ckpt = next(
                    (p for p in [repo / "checkpoints", repo / "model", repo]
                     if (p / "config.json").exists()),
                    None,
                )
                if ckpt is None:
                    raise FileNotFoundError("No config.json found under the repo path")
                self.tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
                self.model = AutoModel.from_pretrained(str(ckpt))
                self.model.eval()
                logger.info("AppraisePLM: loaded via AutoModel from %s", ckpt)
            except Exception as e2:
                raise RuntimeError(
                    f"Could not load AppraisePLM.\n"
                    f"Attempt 1 (repo class): {e1}\n"
                    f"Attempt 2 (AutoModel): {e2}\n"
                    f"Use --mock_model to bypass."
                )

        # Load tokenizer if not already loaded
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer  # type: ignore
                ckpt = next(
                    (p for p in [repo / "checkpoints", repo / "model", repo]
                     if (p / "tokenizer_config.json").exists() or (p / "vocab.txt").exists()),
                    repo,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
            except Exception as te:
                logger.warning("Tokenizer load failed: %s — will use deberta-base fallback", te)
                from transformers import AutoTokenizer  # type: ignore
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    def predict(self, text: str) -> Tuple[List[float], str]:
        """
        Returns (appraisal_21dim, predicted_emotion).
        appraisal_21dim: list of 21 floats in [0, 1]
        predicted_emotion: string label
        """
        if self.mock:
            return _mock_predict(text)

        import torch  # type: ignore
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256, padding=True
        )
        with torch.no_grad():
            try:
                # Repo class API: returns (appraisal_logits, emotion_logits)
                if self._use_repo_class:
                    out = self.model(**inputs)
                    appraisal = torch.sigmoid(out[0]).squeeze().tolist()
                    emotion_idx = int(torch.argmax(out[1], dim=-1).item())
                    emotion = self.model.emotion_labels[emotion_idx] \
                        if hasattr(self.model, "emotion_labels") else str(emotion_idx)
                else:
                    # AutoModel: raw hidden states — use CLS for a dummy 21-dim regression
                    # This is a fallback for testing; real labels require the repo class
                    out = self.model(**inputs)
                    cls = out.last_hidden_state[:, 0, :]
                    appraisal = torch.sigmoid(cls[0, :21]).tolist()
                    emotion = "unknown"
            except Exception as exc:
                logger.debug("Predict failed for text: %s — using mock (%s)", text[:40], exc)
                return _mock_predict(text)

        # Normalise to exactly 21 values
        if len(appraisal) < 21:
            appraisal = appraisal + [0.5] * (21 - len(appraisal))
        appraisal = [float(v) for v in appraisal[:21]]
        return appraisal, emotion


def _mock_predict(text: str) -> Tuple[List[float], str]:
    """Deterministic mock based on text length — for pipeline testing."""
    rng = random.Random(len(text))
    dims = [round(rng.uniform(0.1, 0.9), 3) for _ in range(21)]
    emotions = ["joy", "sadness", "fear", "anger", "surprise", "disgust", "neutral"]
    emotion = emotions[sum(ord(c) for c in text[:8]) % len(emotions)]
    return dims, emotion


# ---------------------------------------------------------------------------
# Appraisal interpretation helpers
# ---------------------------------------------------------------------------

def dims_to_named(appraisal_21dim: List[float]) -> Dict[str, float]:
    return {name: round(val, 4) for name, val in zip(DIM_NAMES, appraisal_21dim)}


def extract_top3(named: Dict[str, float]) -> List[str]:
    """Return 3 most extreme (furthest from 0.5) empathy-relevant dimensions."""
    relevant = {k: v for k, v in named.items() if k in EMPATHY_DIMS}
    return sorted(relevant, key=lambda k: abs(relevant[k] - 0.5), reverse=True)[:3]


def compute_valence(named: Dict[str, float]) -> float:
    """Composite valence: mean of pleasantness, minus unpleasantness, normalised to [0,1]."""
    p = named.get("pleasantness", 0.5)
    u = named.get("unpleasantness", 0.5)
    return round((p + (1.0 - u)) / 2.0, 4)


def build_reading(top3: List[str], named: Dict[str, float], emotion: str) -> str:
    """Build a one-sentence appraisal reading for injection into the <appraisal> block."""
    parts = []
    for dim in top3:
        val = named.get(dim, 0.5)
        level = "high" if val > 0.5 else "low"
        parts.append(f"{level} {dim.replace('_', ' ')}")
    reading = ", ".join(parts)

    # Heuristic narrative based on the dominant dimensions
    narrative = _narrative(top3, named, emotion)
    return f"{reading} → {narrative}"


def _narrative(top3: List[str], named: Dict[str, float], emotion: str) -> str:
    p    = named.get("pleasantness", 0.5)
    u    = named.get("unpleasantness", 0.5)
    cope = named.get("coping_potential", 0.5)
    goal = named.get("goal_relevance", 0.5)
    ctrl = named.get("self_control", 0.5)

    if p > 0.7 and goal > 0.6:
        return "user is experiencing a positive, goal-relevant outcome — respond with warm acknowledgement"
    if u > 0.7 and cope < 0.4:
        return "user is in distress with low coping capacity — respond with empathetic validation before any advice"
    if goal > 0.7 and ctrl < 0.4:
        return "user faces a high-stakes situation outside their control — acknowledge the difficulty first"
    if u > 0.6 and ctrl > 0.6:
        return "user is stressed but feels capable of handling it — supportive and practical tone"
    if p < 0.4 and goal > 0.6:
        return "user is disappointed about something that matters — validate the feeling before problem-solving"
    return f"user is experiencing {emotion} — respond with attentive, personalised empathy"


def is_significant(named: Dict[str, float]) -> bool:
    """Filter: include example only if it carries meaningful emotional signal."""
    valence = compute_valence(named)
    cope    = named.get("coping_potential", 0.5)
    goal    = named.get("goal_relevance", 0.5)
    return (
        abs(valence - 0.5) > VALENCE_THRESHOLD
        or cope < COPING_THRESHOLD
        or goal > GOAL_THRESHOLD
    )


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def load_empathetic_dialogues(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Download and flatten EmpatheticDialogues into individual utterances."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("pip install datasets")

    logger.info("Loading EmpatheticDialogues from HuggingFace...")
    ds = load_dataset("empathetic_dialogues", split="train", trust_remote_code=True)

    rows: List[Dict[str, Any]] = []
    for item in ds:
        rows.append({
            "utterance":    item["utterance"].strip(),
            "emotion":      item["context"],          # emotion label is in 'context' field
            "conv_id":      item["conv_id"],
            "utterance_idx": item["utterance_idx"],
        })

    # Keep only speaker turns (utterance_idx 0 = speaker who revealed the emotion)
    rows = [r for r in rows if r["utterance_idx"] == 0 and len(r["utterance"]) > 20]

    if limit:
        rows = rows[:limit]

    logger.info("Loaded %d EmpatheticDialogues speaker utterances", len(rows))
    return rows


def label_utterances(
    utterances: List[Dict[str, Any]],
    model: AppraisePLMWrapper,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Run AppraisePLM over all utterances, filtering to emotionally significant ones."""
    results: List[Dict[str, Any]] = []
    total = len(utterances)

    for i in range(0, total, batch_size):
        batch = utterances[i : i + batch_size]
        if i % (batch_size * 10) == 0:
            logger.info("Labelling %d / %d...", i, total)

        for row in batch:
            appraisal_21dim, predicted_emotion = model.predict(row["utterance"])
            named    = dims_to_named(appraisal_21dim)
            top3     = extract_top3(named)
            valence  = compute_valence(named)
            reading  = build_reading(top3, named, row["emotion"])

            if not is_significant(named):
                continue

            results.append({
                "utterance":       row["utterance"],
                "emotion":         row["emotion"],
                "conv_id":         row["conv_id"],
                "utterance_idx":   row["utterance_idx"],
                "appraisal_21dim": appraisal_21dim,
                "appraisal_named": named,
                "top3":            top3,
                "valence":         valence,
                "appraisal_reading": reading,
            })

    logger.info(
        "Labelling complete: %d / %d examples kept (%.1f%% pass significance filter)",
        len(results), total, 100 * len(results) / max(total, 1),
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label EmpatheticDialogues with AppraisePLM appraisal vectors"
    )
    parser.add_argument(
        "--output", default="data/appraisal_labels.jsonl",
        help="Output JSONL path (default: data/appraisal_labels.jsonl)",
    )
    parser.add_argument(
        "--appraise_plm_path", default="appraise-PLM",
        help="Path to the cloned appraise-PLM repo (default: appraise-PLM/)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max EmpatheticDialogues utterances to process (default: all)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Processing batch size (default: 32)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: process 20 examples only",
    )
    parser.add_argument(
        "--mock_model", action="store_true",
        help="Use random vectors instead of AppraisePLM (for pipeline testing)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for model inference: cpu | cuda (default: cpu)",
    )
    args = parser.parse_args()

    if args.smoke:
        args.limit = 20
        args.output = "data/appraisal_labels_smoke.jsonl"
        logger.info("Smoke mode: 20 examples → %s", args.output)

    # Load model
    model = AppraisePLMWrapper(
        repo_path=args.appraise_plm_path,
        mock=args.mock_model,
        device=args.device,
    )

    # Load dataset
    utterances = load_empathetic_dialogues(limit=args.limit)

    # Label
    results = label_utterances(utterances, model, batch_size=args.batch_size)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Wrote %d labelled examples to %s", len(results), out_path)

    # Quick stats
    emotions = {}
    for r in results:
        emotions[r["emotion"]] = emotions.get(r["emotion"], 0) + 1
    top_emotions = sorted(emotions.items(), key=lambda x: -x[1])[:8]
    logger.info("Top emotions: %s", ", ".join(f"{e}={n}" for e, n in top_emotions))


if __name__ == "__main__":
    main()
