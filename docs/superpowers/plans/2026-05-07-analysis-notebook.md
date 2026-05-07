# Analysis Dashboard Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace three dark-mode HTML viewers and a server script with a single Jupyter notebook (`pipeline/analysis.ipynb`) that provides publication-quality, exportable figures for dissertation use, interactive data exploration, and cross-model performance comparison.

**Architecture:** One self-contained notebook, 8 sections. Section 1 loads all JSONL/JSON into two globals — `df` (pandas DataFrame of metadata) and `ALL_RECORDS` (list of full records) — that all later sections use. Charts are saved to `pipeline/exports/` as SVG + high-res PNG via plotly/kaleido. An IPython HTML renderer provides a light-mode conversation viewer. ipywidgets power an interactive sample browser.

**Tech Stack:** Python 3, Jupyter, pandas, plotly, kaleido, ipywidgets

---

## File Map

| Action | Path |
|--------|------|
| Create | `pipeline/analysis.ipynb` |
| Create | `pipeline/exports/` (auto-created by notebook cell) |
| Delete | `pipeline/dataset_viewer.html` |
| Delete | `pipeline/reports/view_benchmark.html` |
| Delete | `pipeline/reports/view_reports.html` |
| Delete | `pipeline/reports/server.py` |

---

### Task 1: Create notebook skeleton — Section 0 (header) and Section 1 (data loading)

**Files:**
- Create: `pipeline/analysis.ipynb`

- [ ] **Step 1: Install requirements**

Run from `pipeline/` directory:
```
pip install plotly kaleido ipywidgets nbformat pandas
```

- [ ] **Step 2: Create `pipeline/analysis.ipynb` with Sections 0 and 1**

Create the file as a valid `.ipynb` JSON. It must have `"nbformat": 4` at the top level. Add the following cells in order.

**Cell 0 — markdown:**
```markdown
# Trustworthy Personalised AI — Analysis Dashboard

Research notebook for analysing training data quality, model benchmark results, and conversation samples. All charts are saved to `exports/` as SVG (vector, dissertation-ready) and PNG (high-resolution raster) via plotly + kaleido. Run cells top-to-bottom on first use; individual sections can be re-run independently after that.
```

**Cell 1 — code (imports and config):**
```python
import json
import re
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from IPython.display import HTML, display

pio.templates.default = "plotly_white"
PALETTE = px.colors.qualitative.Set2

DATA_DIR    = Path("data")
REPORTS_DIR = Path("reports")
EXPORTS_DIR = Path("exports")
EXPORTS_DIR.mkdir(exist_ok=True)

def save_fig(fig, name):
    """Save dissertation-ready SVG + high-res PNG and show inline."""
    fig.write_image(str(EXPORTS_DIR / f"{name}.svg"))
    fig.write_image(str(EXPORTS_DIR / f"{name}.png"), scale=3)
    print(f"✓ exports/{name}.svg + .png")
    fig.show()
```

**Cell 2 — markdown:**
```markdown
## Section 1 — Data Loading
```

**Cell 3 — code (data loading):**
```python
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def extract_tag_len(content, tag):
    """Return character length of first <tag>…</tag> block, else 0."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
    return len(m.group(1).strip()) if m else 0

SPLITS = {
    "train":             DATA_DIR / "train_sft_v2.jsonl",
    "eval":              DATA_DIR / "eval_sft_v2.jsonl",
    "train_interleaved": DATA_DIR / "train_interleaved.jsonl",
    "train_partB":       DATA_DIR / "train_partB.jsonl",
}

ALL_RECORDS = []   # full records — used by conversation renderer (Section 7)
rows = []

for split_name, path in SPLITS.items():
    if not path.exists():
        continue
    for rec in load_jsonl(path):
        meta = rec.get("metadata", {}).copy()
        msgs = rec.get("messages", [])
        asst = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
        rows.append({
            **meta,
            "split":          split_name,
            "num_messages":   len(msgs),
            "response_chars": len(asst),
            "think_chars":    extract_tag_len(asst, "think"),
            "answer_chars":   extract_tag_len(asst, "answer"),
            "_idx":           len(ALL_RECORDS),
        })
        ALL_RECORDS.append(rec)

df = pd.DataFrame(rows)
print(f"Loaded {len(df):,} records | {df['split'].nunique()} splits")
print(df.groupby("split").size().to_string())
```

- [ ] **Step 3: Run Sections 0 and 1, verify output**

Expected (numbers will vary with your data):
```
Loaded 236 records | 4 splits
split
eval                    10
train                   92
train_interleaved       92
train_partB             42
```
No errors. `EXPORTS_DIR` directory exists.

- [ ] **Step 4: Commit**
```bash
git add pipeline/analysis.ipynb
git commit -m "feat: add analysis notebook — setup and data loading"
```

---

### Task 2: Add Section 2 — Dataset Overview

**Files:**
- Modify: `pipeline/analysis.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## Section 2 — Dataset Overview
```

- [ ] **Step 2: Add summary stats card cell**

```python
# --- 2a: Summary metric cards ---
total      = len(df)
avg_score  = df["constitution_score"].mean() if "constitution_score" in df else 0
avg_len    = df["response_chars"].mean()
rev_pct    = df["revised"].mean() * 100 if "revised" in df.columns else 100.0

metrics = [
    (f"{total:,}",            "Total Records"),
    (f"{df['split'].nunique()}", "Splits"),
    (f"{df['category'].nunique()}", "Categories"),
    (f"{avg_score:.3f}",      "Avg Constitution Score"),
    (f"{avg_len:,.0f}",       "Avg Response Length (chars)"),
    (f"{rev_pct:.0f}%",       "Revised"),
]
cards = "".join(f"""
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
              padding:16px 24px;min-width:140px;text-align:center">
    <div style="font-size:26px;font-weight:700;color:#1e293b">{v}</div>
    <div style="font-size:12px;color:#64748b;margin-top:4px">{label}</div>
  </div>""" for v, label in metrics)
display(HTML(f"""
<div style="display:flex;gap:16px;flex-wrap:wrap;font-family:ui-sans-serif,sans-serif;
            margin:12px 0">{cards}</div>"""))
```

- [ ] **Step 3: Add category distribution cell**

```python
# --- 2b: Category distribution donut ---
cat_counts = df["category"].value_counts().reset_index()
cat_counts.columns = ["category", "count"]
fig = px.pie(
    cat_counts, names="category", values="count",
    title="Training Data — Category Distribution",
    hole=0.45, color_discrete_sequence=PALETTE
)
fig.update_traces(textposition="outside", textinfo="label+percent")
fig.update_layout(showlegend=False, margin=dict(t=50, b=20))
save_fig(fig, "01_category_distribution")
```

- [ ] **Step 4: Add split × category breakdown cell**

```python
# --- 2c: Records per category per split (grouped bar) ---
split_cat = df.groupby(["split", "category"]).size().reset_index(name="count")
fig = px.bar(
    split_cat, x="category", y="count", color="split",
    barmode="group",
    title="Records per Category by Split",
    color_discrete_sequence=PALETTE,
    labels={"count": "Count", "category": "Category", "split": "Split"}
)
fig.update_xaxes(tickangle=-30)
save_fig(fig, "02_split_category_counts")
```

- [ ] **Step 5: Run Section 2 and verify**

Expected: stat cards visible with numeric values, two plotly charts rendered.

- [ ] **Step 6: Commit**
```bash
git add pipeline/analysis.ipynb
git commit -m "feat: add dataset overview section to analysis notebook"
```

---

### Task 3: Add Section 3 — Constitution Quality Analysis

**Files:**
- Modify: `pipeline/analysis.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## Section 3 — Constitution Quality Analysis
```

- [ ] **Step 2: Add score distribution cell**

```python
# --- 3a: Score distribution histogram ---
fig = px.histogram(
    df, x="constitution_score", nbins=20,
    title="Constitution Score Distribution",
    labels={"constitution_score": "Score (0–1)", "count": "Records"},
    color_discrete_sequence=[PALETTE[0]]
)
fig.update_layout(bargap=0.05)
save_fig(fig, "03_score_distribution")
```

- [ ] **Step 3: Add score by category box plot cell**

```python
# --- 3b: Score by category ---
fig = px.box(
    df, x="category", y="constitution_score",
    color="category", color_discrete_sequence=PALETTE,
    title="Constitution Score by Category",
    labels={"constitution_score": "Score", "category": "Category"},
    points="all"
)
fig.update_layout(showlegend=False)
fig.update_xaxes(tickangle=-30)
save_fig(fig, "04_score_by_category")
```

- [ ] **Step 4: Add violations histogram cell**

```python
# --- 3c: Draft violations histogram ---
fig = px.histogram(
    df, x="constitution_violations_in_draft",
    title="Constitution Violations in Draft",
    labels={"constitution_violations_in_draft": "Violations", "count": "Records"},
    color_discrete_sequence=[PALETTE[1]]
)
fig.update_layout(bargap=0.1)
save_fig(fig, "05_violations_histogram")
```

- [ ] **Step 5: Add score vs violations scatter cell**

```python
# --- 3d: Score vs violations scatter ---
fig = px.scatter(
    df, x="constitution_violations_in_draft", y="constitution_score",
    color="category",
    title="Constitution Score vs. Violations in Draft",
    labels={"constitution_violations_in_draft": "Violations in Draft",
            "constitution_score": "Final Score"},
    color_discrete_sequence=PALETTE,
    hover_data=["category", "tool_profile"]
)
save_fig(fig, "06_score_vs_violations")
```

- [ ] **Step 6: Run Section 3 and verify**

Expected: 4 charts. Scores cluster near 0.89–0.95. Violations typically 1–2.

- [ ] **Step 7: Commit**
```bash
git add pipeline/analysis.ipynb
git commit -m "feat: add constitution quality analysis section"
```

---

### Task 4: Add Section 4 — Tool Profile × Category Analysis

**Files:**
- Modify: `pipeline/analysis.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## Section 4 — Tool Profile & Category Analysis
```

- [ ] **Step 2: Add tool profile donut cell**

```python
# --- 4a: Tool profile distribution ---
tp_counts = df["tool_profile"].value_counts().reset_index()
tp_counts.columns = ["tool_profile", "count"]
fig = px.pie(
    tp_counts, names="tool_profile", values="count", hole=0.45,
    title="Tool Profile Distribution",
    color_discrete_sequence=PALETTE
)
fig.update_traces(textposition="outside", textinfo="label+percent")
fig.update_layout(showlegend=False)
save_fig(fig, "07_tool_profile_distribution")
```

- [ ] **Step 3: Add category × tool profile heatmap cell**

```python
# --- 4b: Category × tool_profile count heatmap ---
pivot = (
    df.groupby(["category", "tool_profile"]).size()
    .reset_index(name="count")
    .pivot(index="category", columns="tool_profile", values="count")
    .fillna(0)
    .astype(int)
)
fig = go.Figure(go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    colorscale="Blues",
    text=pivot.values,
    texttemplate="%{text}",
    showscale=True,
))
fig.update_layout(
    title="Category × Tool Profile Count",
    xaxis_title="Tool Profile",
    yaxis_title="Category",
    margin=dict(l=160)
)
save_fig(fig, "08_category_toolprofile_heatmap")
```

- [ ] **Step 4: Add avg score by tool profile cell**

```python
# --- 4c: Average constitution score by tool profile ---
avg_tp = df.groupby("tool_profile")["constitution_score"].mean().reset_index()
fig = px.bar(
    avg_tp, x="tool_profile", y="constitution_score",
    title="Average Constitution Score by Tool Profile",
    color="tool_profile", color_discrete_sequence=PALETTE,
    labels={"constitution_score": "Avg Score", "tool_profile": "Tool Profile"},
    text_auto=".3f"
)
fig.update_layout(showlegend=False, yaxis_range=[0.85, 1.0])
save_fig(fig, "09_avg_score_by_tool_profile")
```

- [ ] **Step 5: Run Section 4 and verify**

Expected: donut, heatmap grid, bar chart all render without error.

- [ ] **Step 6: Commit**
```bash
git add pipeline/analysis.ipynb
git commit -m "feat: add tool profile analysis section"
```

---

### Task 5: Add Section 5 — Response Quality Deep-Dive

**Files:**
- Modify: `pipeline/analysis.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## Section 5 — Response Quality Deep-Dive
```

- [ ] **Step 2: Add think vs answer length scatter cell**

```python
# --- 5a: Think-tag vs answer-tag length scatter ---
has_think = df[df["think_chars"] > 0].copy()
fig = px.scatter(
    has_think,
    x="think_chars", y="answer_chars",
    color="category", size="constitution_score", size_max=14,
    title="Think-Tag Length vs Answer-Tag Length",
    labels={"think_chars": "<think> block (chars)", "answer_chars": "<answer> block (chars)"},
    color_discrete_sequence=PALETTE,
    hover_data=["tool_profile", "constitution_score", "constitution_violations_in_draft"]
)
save_fig(fig, "10_think_vs_answer_scatter")
```

- [ ] **Step 3: Add response length violin cell**

```python
# --- 5b: Response length distribution by category ---
fig = px.violin(
    df, x="category", y="response_chars",
    color="category", box=True, points="outliers",
    title="Response Length Distribution by Category",
    labels={"response_chars": "Response Length (chars)", "category": "Category"},
    color_discrete_sequence=PALETTE
)
fig.update_layout(showlegend=False)
fig.update_xaxes(tickangle=-30)
save_fig(fig, "11_response_length_by_category")
```

- [ ] **Step 4: Add response length by tool profile cell**

```python
# --- 5c: Response length by tool profile ---
fig = px.box(
    df, x="tool_profile", y="response_chars",
    color="tool_profile", points="all",
    title="Response Length by Tool Profile",
    labels={"response_chars": "Response Length (chars)", "tool_profile": "Tool Profile"},
    color_discrete_sequence=PALETTE
)
fig.update_layout(showlegend=False)
save_fig(fig, "12_response_length_by_tool_profile")
```

- [ ] **Step 5: Add score vs response length cell**

```python
# --- 5d: Score vs response length with OLS trendline ---
fig = px.scatter(
    df, x="response_chars", y="constitution_score",
    color="category", trendline="ols",
    title="Constitution Score vs. Response Length",
    labels={"response_chars": "Response Length (chars)", "constitution_score": "Score"},
    color_discrete_sequence=PALETTE,
    hover_data=["tool_profile"]
)
save_fig(fig, "13_score_vs_response_length")
```

- [ ] **Step 6: Run Section 5 and verify**

Expected: 4 charts. Violin shows variance across categories. OLS trendlines visible on scatter.

- [ ] **Step 7: Commit**
```bash
git add pipeline/analysis.ipynb
git commit -m "feat: add response quality analysis section"
```

---

### Task 6: Add Section 6 — Cross-Model Performance Comparison

This section answers: *how do different model variants (base/custom × with/without tools) perform on the same tasks?*

**Files:**
- Modify: `pipeline/analysis.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## Section 6 — Cross-Model Performance Comparison

Compares base model vs custom (SFT-trained) model, with and without tools, on the same prompts.
Metrics extracted: number of turns, tool calls made, response length, reasoning depth (think-block length), and answer clarity (presence of `<answer>` tag).
```

- [ ] **Step 2: Add data loading and feature extraction cell**

```python
# --- 6a: Load reports and extract per-run metrics ---
def load_json_reports(pattern):
    return [json.load(open(p, encoding="utf-8"))
            for p in sorted(REPORTS_DIR.glob(pattern))]

benchmarks  = load_json_reports("benchmark_*.json")
comparisons = load_json_reports("comparison_*.json")
print(f"Benchmarks: {len(benchmarks)} | Comparisons: {len(comparisons)}")

def run_metrics(conversation):
    """Extract scalar metrics from a single model run conversation."""
    asst_msgs = [m for m in conversation if m["role"] == "assistant"]
    tool_calls = sum(1 for m in asst_msgs if "<tool>" in m.get("content", ""))
    has_answer = sum(1 for m in asst_msgs if "<answer>" in m.get("content", ""))
    avg_think  = (sum(extract_tag_len(m["content"], "think") for m in asst_msgs)
                  / len(asst_msgs) if asst_msgs else 0)
    avg_len    = (sum(len(m["content"]) for m in asst_msgs)
                  / len(asst_msgs) if asst_msgs else 0)
    return {
        "turns":       len(asst_msgs),
        "tool_calls":  tool_calls,
        "has_answer":  has_answer,
        "avg_think":   avg_think,
        "avg_len":     avg_len,
    }
```

- [ ] **Step 3: Add comparison turn-count chart cell**

```python
# --- 6b: Turn count — base vs custom across all comparison prompts ---
comp_rows = []
for c in comparisons:
    prompt = c.get("prompt", "")
    short  = (prompt[:55] + "…") if len(prompt) > 55 else prompt
    bm     = run_metrics(c.get("base_model_no_tools", {}).get("conversation", []))
    cm     = run_metrics(c.get("custom_model_output", {}).get("conversation", []))
    comp_rows.append({
        "prompt":            short,
        "Base (no tools)":   bm["turns"],
        "Custom (w/ tools)": cm["turns"],
    })
cdf = pd.DataFrame(comp_rows)

fig = go.Figure([
    go.Bar(name="Base (no tools)",   x=cdf["prompt"], y=cdf["Base (no tools)"],   marker_color=PALETTE[0]),
    go.Bar(name="Custom (w/ tools)", x=cdf["prompt"], y=cdf["Custom (w/ tools)"], marker_color=PALETTE[1]),
])
fig.update_layout(
    barmode="group",
    title="Response Turn Count: Base vs Custom Model",
    xaxis_title="Prompt", yaxis_title="Turns",
    xaxis_tickangle=-40, legend_title="Model", height=420
)
save_fig(fig, "14_turn_count_comparison")
```

- [ ] **Step 4: Add multi-metric radar chart cell**

```python
# --- 6c: Multi-metric radar — aggregate comparison across all comparisons ---
def avg_metrics(key):
    vals = []
    for c in comparisons:
        conv = c.get(key, {}).get("conversation", [])
        if conv:
            vals.append(run_metrics(conv))
    if not vals:
        return {k: 0 for k in ["turns","tool_calls","has_answer","avg_think","avg_len"]}
    df_v = pd.DataFrame(vals)
    return df_v.mean().to_dict()

base_avg   = avg_metrics("base_model_no_tools")
custom_avg = avg_metrics("custom_model_output")

# Normalise each metric to 0-1 across both models for radar display
metrics_keys   = ["turns", "tool_calls", "has_answer", "avg_think", "avg_len"]
metrics_labels = ["Turns", "Tool Calls", "Answer Tags", "Avg Think (chars)", "Avg Resp (chars)"]

def normalise(vals):
    max_v = max(abs(v) for v in vals) or 1
    return [v / max_v for v in vals]

base_vals   = [base_avg[k]   for k in metrics_keys]
custom_vals = [custom_avg[k] for k in metrics_keys]
base_n      = normalise(base_vals)
custom_n    = normalise(custom_vals)

fig = go.Figure()
for label, vals, color in [
    ("Base (no tools)",   base_n,   PALETTE[0]),
    ("Custom (w/ tools)", custom_n, PALETTE[1]),
]:
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=metrics_labels + [metrics_labels[0]],
        fill="toself", name=label, line_color=color,
        fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)") if "rgb" in color else color,
    ))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Model Capability Radar — Normalised Metrics",
    showlegend=True,
)
save_fig(fig, "15_model_radar")
```

- [ ] **Step 5: Add per-prompt metric comparison table cell**

```python
# --- 6d: Side-by-side metric table for each comparison prompt ---
table_rows = []
for c in comparisons:
    prompt = c.get("prompt", "")
    short  = (prompt[:60] + "…") if len(prompt) > 60 else prompt
    bm     = run_metrics(c.get("base_model_no_tools", {}).get("conversation", []))
    cm     = run_metrics(c.get("custom_model_output", {}).get("conversation", []))
    table_rows.append({
        "Prompt":              short,
        "Base Turns":          bm["turns"],
        "Custom Turns":        cm["turns"],
        "Custom Tool Calls":   cm["tool_calls"],
        "Custom Answer Tags":  cm["has_answer"],
        "Base Avg Len":        f"{bm['avg_len']:.0f}",
        "Custom Avg Len":      f"{cm['avg_len']:.0f}",
    })
pd.DataFrame(table_rows).style.background_gradient(
    subset=["Base Turns","Custom Turns","Custom Tool Calls"], cmap="Blues"
)
```

- [ ] **Step 6: Add benchmark multi-run comparison cell (uses benchmark_*.json)**

```python
# --- 6e: Benchmark run comparison for each benchmark file ---
for bm in benchmarks:
    run_rows = []
    for run_name, run_data in bm.get("runs", {}).items():
        conv = run_data.get("conversation", [])
        m = run_metrics(conv)
        run_rows.append({"Run": run_name, **m})
    if not run_rows:
        continue
    bdf = pd.DataFrame(run_rows)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Turns", "Tool Calls", "Avg Response Length (chars)")
    )
    colors = PALETTE[:len(bdf)]
    fig.add_trace(go.Bar(x=bdf["Run"], y=bdf["turns"],      marker_color=colors, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=bdf["Run"], y=bdf["tool_calls"],  marker_color=colors, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=bdf["Run"], y=bdf["avg_len"],     marker_color=colors, showlegend=False), row=1, col=3)
    ts = bm.get("timestamp", "")
    fig.update_layout(title=f"Benchmark Multi-Run Comparison — {ts}", height=380)
    save_fig(fig, f"16_benchmark_{ts}")
```

- [ ] **Step 7: Run Section 6 and verify**

Expected: grouped bar chart, radar chart, styled table, and one subplot panel per benchmark file. Custom model should show more turns and tool calls than base.

- [ ] **Step 8: Commit**
```bash
git add pipeline/analysis.ipynb
git commit -m "feat: add cross-model performance comparison section"
```

---

### Task 7: Add Section 7 — Conversation Renderer and Interactive Sample Browser

**Files:**
- Modify: `pipeline/analysis.ipynb`

- [ ] **Step 1: Add markdown cell**

```markdown
## Section 7 — Conversation Viewer

Light-mode, dissertation-ready conversation renderer with syntax highlighting for `<think>`, `<answer>`, and `<tool>` tags. Use the interactive browser to explore training samples by category and tool profile.
```

- [ ] **Step 2: Add renderer helper cell**

```python
# --- 7a: Conversation renderer ---
_ROLE_STYLE = {
    "system":    ("System",    "#eff6ff", "#1d4ed8"),
    "user":      ("User",      "#f0fdf4", "#16a34a"),
    "assistant": ("Assistant", "#faf5ff", "#7c3aed"),
}

def _highlight_tags(text):
    text = re.sub(
        r"<think>(.*?)</think>",
        lambda m: (
            "<details open style='margin:4px 0'>"
            "<summary style='color:#6366f1;font-weight:600;cursor:pointer'>&#x1f4ad; Reasoning</summary>"
            "<pre style='white-space:pre-wrap;background:#f8f7ff;padding:10px;border-radius:6px;"
            "font-size:13px;color:#3730a3;margin:4px 0'>"
            + m.group(1) + "</pre></details>"
        ),
        text, flags=re.DOTALL
    )
    text = re.sub(
        r"<answer>(.*?)</answer>",
        lambda m: (
            "<div style='background:#f0fdf4;border-left:3px solid #22c55e;"
            "padding:8px 12px;margin:6px 0;border-radius:0 6px 6px 0'>"
            "<span style='font-weight:600;color:#16a34a'>Answer: </span>"
            + m.group(1) + "</div>"
        ),
        text, flags=re.DOTALL
    )
    text = re.sub(
        r"<tool>(.*?)</tool>",
        lambda m: (
            "<code style='background:#fff7ed;border:1px solid #fed7aa;"
            "padding:3px 8px;border-radius:4px;font-size:13px'>"
            "&#x1f527; " + m.group(1) + "</code>"
        ),
        text, flags=re.DOTALL
    )
    return text

def render_conversation(messages, title="Conversation", score=None, category=None):
    meta = ""
    if category or score is not None:
        parts = []
        if category:        parts.append(f"<strong>Category:</strong> {category}")
        if score is not None: parts.append(f"<strong>Score:</strong> {score:.3f}")
        meta = f"<p style='font-size:12px;color:#64748b;margin:4px 0 10px'>{' · '.join(parts)}</p>"

    bubbles = ""
    for msg in messages:
        role            = msg.get("role", "assistant")
        label, bg, acc  = _ROLE_STYLE.get(role, ("?", "#fff", "#000"))
        content         = _highlight_tags(msg.get("content", ""))
        bubbles += (
            f"<div style='background:{bg};border:1px solid #e2e8f0;border-radius:8px;"
            f"padding:10px 14px'>"
            f"<div style='font-size:11px;font-weight:700;color:{acc};"
            f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px'>{label}</div>"
            f"<div style='font-size:14px;color:#1e293b;line-height:1.6'>{content}</div>"
            f"</div>"
        )

    html = (
        f"<div style='font-family:ui-sans-serif,system-ui,sans-serif;max-width:860px;"
        f"border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;"
        f"box-shadow:0 1px 4px rgba(0,0,0,0.06)'>"
        f"<div style='background:#f8fafc;padding:12px 18px;border-bottom:1px solid #e2e8f0'>"
        f"<h3 style='margin:0;font-size:16px;color:#1e293b'>{title}</h3>{meta}</div>"
        f"<div style='padding:14px 16px;display:flex;flex-direction:column;gap:10px'>"
        f"{bubbles}</div></div>"
    )
    return HTML(html)
```

- [ ] **Step 3: Add interactive sample browser cell**

```python
# --- 7b: Interactive training sample browser ---
from ipywidgets import interact, Dropdown, IntSlider

_CATEGORIES    = sorted(df["category"].unique().tolist())
_TOOL_PROFILES = ["(any)"] + sorted(df["tool_profile"].unique().tolist())

def browse_samples(category=_CATEGORIES[0], tool_profile="(any)", seed=42):
    subset = df[df["category"] == category]
    if tool_profile != "(any)":
        subset = subset[subset["tool_profile"] == tool_profile]
    if subset.empty:
        display(HTML("<p style='color:#ef4444;font-family:sans-serif'>No matching records.</p>"))
        return
    row = subset.sample(1, random_state=seed).iloc[0]
    rec = ALL_RECORDS[row["_idx"]]
    display(render_conversation(
        rec["messages"],
        title=f"{row['category']} · {row['tool_profile']}",
        score=row.get("constitution_score"),
        category=row["category"],
    ))

interact(
    browse_samples,
    category=Dropdown(options=_CATEGORIES, description="Category:"),
    tool_profile=Dropdown(options=_TOOL_PROFILES, description="Tool Profile:"),
    seed=IntSlider(min=0, max=100, value=42, description="Seed:"),
)
```

- [ ] **Step 4: Add benchmark side-by-side renderer cell**

```python
# --- 7c: Side-by-side comparison viewer ---
from ipywidgets import interact, Dropdown

_COMP_LABELS = [
    ((c.get("prompt","")[:60]+"…") if len(c.get("prompt",""))>60 else c.get("prompt",""))
    for c in comparisons
]

def show_comparison(prompt_label=_COMP_LABELS[0] if _COMP_LABELS else ""):
    idx = _COMP_LABELS.index(prompt_label) if prompt_label in _COMP_LABELS else 0
    c   = comparisons[idx]
    display(HTML(
        f"<h3 style='font-family:sans-serif;margin:12px 0 4px'>"
        f"Prompt: <em style='font-weight:400'>{c.get('prompt','')}</em></h3>"
    ))
    display(render_conversation(
        c["base_model_no_tools"]["conversation"], "Base Model (No Tools)"))
    display(HTML("<br>"))
    display(render_conversation(
        c["custom_model_output"]["conversation"], "Custom Model (With Tools)"))

if _COMP_LABELS:
    interact(show_comparison, prompt_label=Dropdown(options=_COMP_LABELS, description="Prompt:"))
else:
    display(HTML("<p style='color:#64748b'>No comparison reports found in reports/</p>"))
```

- [ ] **Step 5: Run Section 7 and verify**

Expected: ipywidgets dropdowns render. Selecting a category shows a styled conversation. Comparison viewer shows base and custom side by side.

- [ ] **Step 6: Commit**
```bash
git add pipeline/analysis.ipynb
git commit -m "feat: add conversation renderer and interactive sample browser"
```

---

### Task 8: Delete old HTML viewers and server script

**Files:**
- Delete: `pipeline/dataset_viewer.html`
- Delete: `pipeline/reports/view_benchmark.html`
- Delete: `pipeline/reports/view_reports.html`
- Delete: `pipeline/reports/server.py`

- [ ] **Step 1: Verify the notebook covers all old functionality**

| Old file | Replaced by |
|----------|-------------|
| `dataset_viewer.html` | Section 7 — interactive sample browser |
| `reports/view_benchmark.html` | Section 6 — benchmark multi-run comparison |
| `reports/view_reports.html` | Section 6 comparison charts + Section 7 side-by-side viewer |
| `reports/server.py` | Not needed — notebook runs locally in Jupyter |

- [ ] **Step 2: Delete files**
```bash
git rm pipeline/dataset_viewer.html
git rm pipeline/reports/view_benchmark.html
git rm pipeline/reports/view_reports.html
git rm pipeline/reports/server.py
```

- [ ] **Step 3: Commit**
```bash
git commit -m "chore: remove old HTML viewers and server — replaced by analysis.ipynb"
```

---

## Self-Review

**Spec coverage:**
- Dataset overview ✓ (Task 2)
- Constitution quality ✓ (Task 3)
- Tool profile × category ✓ (Task 4)
- Response quality ✓ (Task 5)
- Cross-model comparison / same-task comparison ✓ (Task 6 — radar chart + benchmark multi-run)
- Conversation viewer ✓ (Task 7)
- Interactive sample browser ✓ (Task 7)
- Light mode throughout ✓ (all HTML uses light palette)
- Export as SVG + PNG ✓ (`save_fig` in every chart cell)
- Delete old files ✓ (Task 8)

**Placeholder scan:** No TBDs. All cells contain complete, runnable Python.

**Type consistency:**
- `ALL_RECORDS` — defined Task 1, used Task 7 ✓
- `df` — defined Task 1, used Tasks 2–7 ✓
- `save_fig` — defined Task 1, used Tasks 2–6 ✓
- `PALETTE` — defined Task 1, used Tasks 2–6 ✓
- `comparisons` — defined Task 6, used Task 7 ✓
- `run_metrics` — defined Task 6, used Task 6 only ✓
- `render_conversation` — defined Task 7, used Task 7 ✓
- `extract_tag_len` — defined Task 1, used Tasks 1 and 6 ✓
