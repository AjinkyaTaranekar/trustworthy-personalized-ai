# TCD Report-Guideline and Marking-Sheet Compliance Audit

Audit date: 2026-07-25. Scope: every `.tex` file under `docs/TCD_Dissertation/`, checked
end-to-end against (a) the SCSS Report Guidelines and (b) the Dissertation Evaluation Form
and Marking Sheet (`projects.scss.tcd.ie/.../Dissertation-Marking-Sheet-Sample.pdf`).

Every fix named here is also written into the `.tex` files as a `% [GUIDELINE 2026-07-25 ...]`
comment at the point it applies. Nothing in the live prose was changed.

---

## 0. Marking sheet: weights and what the band descriptors actually ask for

| Criterion | Weight | 70+ band requires | 50–60 band (what to avoid) |
|---|---|---|---|
| Problem statement, motivation, analysis | 10% | deep critical analysis; objectives clearly stated *and critically analysed* | motivation lacking; objectives unclear |
| Background research & literature review | 15% | **critical and analytical** perspective; exceptional ability to relate theory to project work | **"more descriptive than critical/analytical"** |
| Technical content and project execution | 50% | methodology **executed rigorously**; limitations recognised; ambitious project executed well; publishable at the top end | "justification and limitations not fully explored; methodology followed but lacking rigour" |
| Testing, evaluation, critical analysis & conclusions | 15% | clear link with **all** chapters; **well supported** analysis; limitations/future work clearly defined | unclear link to research questions; lacking a critical approach |
| Report presentation and writing | 10% | internally consistent; adheres **throughout** to academic conventions; **excellent** use of figures/tables | "**discrepancies in language and academic convention usage**"; "some unsupported assertions" |

Plus two free-text boxes the examiner must fill before scoring: **Project Description** and
**Challenges** ("what were the major difficulties encountered (technical or other)?").

There is no 90% band on this form — 70+ is the top band. The practical reading: to land at the
top of the 70+ band rather than the bottom of it, every band descriptor above has to be
*visibly* satisfiable from the document without the examiner having to give benefit of the doubt.

---

## 1. Current measured shape of the document

| Chapter | Live words | Citations | Figures | Tables |
|---|---:|---:|---:|---:|
| Introduction | 2,942 | 21 | **0** | **0** |
| Background & Literature Review | **8,995** | 68 | 2 | 3 |
| State of the Art | **1,427** | 15 | 1 | 2 |
| Methodology | 4,002 | 17 | 2 | 3 |
| Experiments and Results | 4,259 | 2 | 18 | 8 |
| Discussion | 3,948 | 9 | 1 | 1 |
| Conclusion | 2,442 | 5 | **0** | **0** |
| Appendix | 2,690 | 1 | 0 | 7 |
| **Total** | **31,788** | — | 24 | 24 |

Roughly 71 pages of prose before floats, front matter and appendix listings — expect ~110–130
typeset pages. Also present: ~23,800 words of `%`-commented drafts, polish and audit notes that
do not render.

Mechanical checks that **passed**: no broken `\ref` (155 labels, 0 dangling), no `\cite` key
missing from `ref.bib`, both central results tables now `\input` correctly, `\listoffigures` and
`\listoftables` present, Nomenclature present, GenAI use acknowledged in front matter.

---

## 2. Findings, ranked by marks at risk

### BLOCKER-1 — ~60 identified grammar fixes are still unapplied in the live text (10% + 15%)

The chapters carry roughly sixty `% [POLISH ... | DROP-IN]` comments, each naming an exact
live-text error and its replacement. **None have been applied to the live prose.** The rendered
PDF therefore still contains subject–verb disagreements, comma splices and sentence fragments —
for example `Language models varies in different sizes`, `Humans reads and understand text`,
`all three experiments shares one pipeline`, `The two questions which were put to this experiment.`

This is precisely the 50–60 band descriptor *"displays some discrepancies in language and
academic convention usage"*, and it is the single cheapest set of marks in the whole document.

**Fix:** apply every `[POLISH ... DROP-IN]` block, then delete the comment. Track them with
`grep -c "POLISH.*DROP-IN" **/*.tex` until it reaches zero.

### BLOCKER-2 — a known citation misattribution is still live in four places (15%)

`background.tex:440`, `introduction.tex:308`, `state-of-the-art.tex:45` and
`conclusion.tex:167` all state that **Rainone et al. found GRPO unstable at 0.6B and attribute
this to model scale**. The existing `% [AUDIT-FIX 2026-07-24]` block at `introduction.tex:287`
already establishes this is wrong: Rainone et al.'s smallest model is Llama-1B, and their
finding is the *opposite* — a training-format choice (Chain-of-Edits) rescues reasoning at 1–3B,
with the advantage reversing at 8B.

A second reader who opens the cited paper finds a claim reversed. Under *"systematic and complete
reference to sources used"* this is the most damaging single item in the document.

**Fix (already drafted at `introduction.tex:293`):** re-cite the 0.5B RL-collapse evidence to
RAGEN — `\cite{wang2025ragenunderstandingselfevolutionllm}`, already in `ref.bib` and currently
uncited — plus Beyond ReAct, and keep Rainone only for small-model reasoning brittleness. Apply
at all four sites.

### BLOCKER-3 — no uncertainty is reported on any headline number (50% + 15%)

The central claim of Section~\ref{subsec:h3-prompt-engineering} is a *parity* result:
0.589 against 0.583, described as *"within the noise of a single-judge protocol"*. No interval,
no test, no n-per-cell is given. The same holds for every ladder delta.

`Section~\ref{subsec:evaluation-limitations}` concedes *"a modest number of test items yields
wide uncertainty"* — but concedes it in words rather than reporting the number. For *"evidence of
critical and well supported analysis"*, a stated-but-unquantified uncertainty reads as an
unsupported assertion.

**Fix:** the 151 questions are paired across conditions and every verdict is already persisted.
A paired bootstrap over questions gives a CI on each H2H score and on each ladder delta at zero
GPU cost, from saved reports. Add the interval to `tab_rank_overall` and
`tab_rank_lineage`, and replace "within the noise" with the measured interval.

### BLOCKER-4 — an experiment that was run is never reported (50%)

`appendix.tex:206` states that the absolute judge was re-run under two further exposures
(*bare*: principle name only; *none*: no principle at all) as a self-enhancement control. **The
results of that ablation appear nowhere in the body.** This is a completed piece of validation
work — and it is exactly the evidence that answers "is the judge rewarding constitution-shaped
text rather than compliance?" — sitting unreported.

Likewise `Section~\ref{subsec:mr-judge-validity}` says each item *"is judged several times and
averaged"*, but the spread across those repeats is never given.

**Fix:** report both in Methodology or Experiments — the exposure ablation as a small table, the
repeat variance as a single sentence with a number. Both are recoverable from saved reports.

### BLOCKER-5 — the judge model is not named (50%)

The measurement instrument is described only as *"a more capable language model"* and
*"frontier models served through NVIDIA NIM"*. A study whose headline scores come from an
unnamed grader is not reproducible. `methodology-results.tex:386` already flags this and records
the actual judges (comparative lens: ZAI GLM-5.1 via crusoe — **not** NIM, so the current
sentence is also inaccurate; absolute lens: minimaxai/minimax-m3).

**Fix:** name both judges with version, decoding settings and serving path; state that the judge
is held identical across all five conditions; correct the NIM claim.

### MAJOR-6 — the literature review is out of balance for the 15% criterion

8,995 words of Background against 1,427 words of State of the Art. The bulk of the Background is
standard textbook material (tokenisation, embeddings, attention, the human-vs-machine
philosophical discussion), which the marking sheet describes as *"more descriptive than
critical/analytical"* — the 50–60 band. The genuinely critical, comparative work that the 70+
band rewards lives in State of the Art, and it is the shortest chapter in the dissertation.

**Fix:** the existing `[TRIM-CANDIDATE]`, `[COMPRESS]` and `[CUT]` comments already identify
~4–5 pages of descriptive material to retire (human-vs-machine four-gap walk, both tokeniser
screenshots, the duplicated making-models-small preview, the LoRA/quantisation derivations).
Retire them and spend the space in State of the Art: the Menke & Tan Qwen-family self-critique
finding (already drafted at `state-of-the-art.tex:55`) and the two-sided FLAN reading (drafted at
`background.tex:458`) are both critical-analysis content that is currently missing.

### MAJOR-7 — no "problems encountered" anywhere in the document (15% + examiner's box)

The guideline is explicit: *"The conclusions should be a critique of your work… Any problems
encountered in the course of the project should be mentioned here."* The marking sheet gives the
examiner a **Challenges** box asking what the major technical difficulties were and whether the
solutions offered were appropriate.

The Discussion has *Limitations* (what the work cannot claim). That is a different thing from
*challenges* (what went wrong during the work and how it was handled). Nothing in the document
tells the examiner that, for example, the reasoning traces were lost during trajectory
construction and had to be diagnosed after the fact, that the RL route was closed off, that the
work ran on a single rented GPU, or that the harness retry layer was never enabled.

**Fix:** add `\section{Challenges Encountered}` to the Conclusion, before Future Work. This
section is free marks — the examiner has a box they must fill and currently no source for it.

### MAJOR-8 — the Introduction and the Conclusion contain no figure or table (10%)

The 70+ presentation band asks for *"excellent use of figures, tables, and diagrams where
appropriate"*. Both bookend chapters have none, while the Experiments chapter has 18 figures.
Two assets that would fix this are **already drafted and commented out**:

- `introduction.tex:223` — `fig:architecture`, the whole-system diagram (the comment records
  that the supervisor liked it). It is referenced by seven other drafted openers.
- `conclusion.tex:39` — `tab:hypothesis-verdicts`, the hypothesis-by-hypothesis verdict table.
  This is the classic "objectives revisited" device, and the 15% criterion explicitly rewards a
  *"clear link with all chapters in keeping with research questions/objectives"*.

**Fix:** uncomment both, then add the pointer sentences the comments supply.

### MODERATE-9 — first person and contractions violate two explicit guideline rules (10%)

The guideline states *"Avoid use of contractions"* and *"Use third person as much as possible"*.
Live-text hits (comments and quoted model output excluded):

- **First person**, 15 live instances, almost all in Background: `we will discuss`, `our work`,
  `our brain`, `we cannot simply paste`, `the models we use`, `the way we edit`, `we see today`,
  `known to us`, `the only stage we ourselves touch`, `tell us the capital`, `the trade-off we
  keep returning to`, `the very capability we are trying to measure`, `the failure we call
  hallucination`, `we treat trustworthiness`, plus `The society we live in` (Introduction).
- **Contractions**, 14 live instances: `it's` ×8, `that's` ×2, `doesn't` ×2, `didn't` ×2.

Both are listed with exact replacements in the per-file comment blocks. Note these are separate
from, and additional to, the `[POLISH]` backlog of BLOCKER-1.

### MODERATE-10 — internal inconsistency visible in the appendix (10%)

`appendix/constitution.md` is reproduced verbatim and its first line reads *"This document
defines the **19 principles** that govern every response"*, while the body of the dissertation
says twenty-five throughout and the same file defines 25 (`grep -c "^### P"` = 25). An examiner
reading Appendix A2 sees the contradiction directly. `appendix.tex:8` records it as a private
note; the reader never sees that note.

**Fix:** either correct the header in `pipeline/constitution.md` and re-copy, or add a visible
footnote in Appendix A2 explaining the count.

### MODERATE-11 — orphaned float and orphaned appendix

- `tab:cai-at-scale` (State of the Art) is never `\ref`'d from any prose. A table no sentence
  points to reads as decoration.
- `chap:ethics-statement` is never referenced from the body. The reference existed at
  `conclusion.tex:157` but was swept into a comment block; `conclusion.tex:144` already flags this.

### MINOR-12 — front-matter and preamble items to resolve before submission

| Item | Where | Action |
|---|---|---|
| `I consent / do not consent` — both still present | `main.tex:151` | delete one |
| Bold template instruction *"Please consult with your supervisor… and delete if you do not consent"* still live | `main.tex:154` | delete the bold sentence |
| `Signed:` rule line | `main.tex:157` | guideline says signatures must not appear in the submitted PDF; confirm with supervisor whether the blank rule stays |
| `\\` inside `\thesistitle`, passed to `pdftitle=` | `main.tex:24,125` | line break in PDF metadata; define a separate single-line `\thesistitleshort` for `hypersetup` |
| `\frontmatter` placed *before* `\begin{document}` | `main.tex:136–137` | belongs after `\begin{document}`; verify in Overleaf |
| `\appendix` and `\renewcommand{\thechapter}` issued twice | `main.tex:287` and `appendix.tex:1` | harmless but redundant; keep one |
| 74 entries in `ref.bib` are never cited | `bibs/ref.bib` | not rendered by IEEEtran, so no marking impact — but prune before release |

---

## 3. Guideline-by-guideline checklist

| Guideline requirement | Status |
|---|---|
| Ch.1 = brief description + reader's guide chapter by chapter | **Pass** — `sec:dissertation-structure` |
| Ch.2 = background, motivation, state of the art (split allowed) | **Pass** — split across two chapters |
| Main body = what you did and how, illustrated with diagrams and examples | **Pass** — strong; 24 figures, worked case studies |
| Final chapter split into conclusions **and** future work | **Partial** — future work is thorough; the *critique of own work / problems encountered* half is missing (MAJOR-7) |
| Conclusions relate achievement back to the initial aim | **Pass** — `sec:conclusion-answer` |
| Negative results discussed, disadvantages not hidden | **Strong pass** — H5 reported as unsupported; this is done well |
| Sources systematically and completely referenced | **Fail on one point** — BLOCKER-2 misattribution |
| Journal titles not abbreviated | **Pass** — all 12 `journal` fields are full titles |
| No verbatim quoting of large chunks | **Pass** — verbatim material is the author's own artefacts, properly labelled |
| Impersonal style; objective and quantitative | **Partial** — MODERATE-9; and BLOCKER-3 leaves the key claim qualitative |
| Avoid contractions | **Fail** — 14 live instances |
| Use third person | **Fail** — 15 live instances |
| GenAI use acknowledged and cited | **Pass** — front-matter statement is thorough and specific |
| No student ID in the PDF | **Pass** — `authorid` deliberately undefined |
| Signatures absent from the PDF | **Check** — blank `Signed:` rule present (MINOR-12) |
| Spell-check and proof-read | **Fail** — BLOCKER-1; plus `recognizable`, `typicall`, `saftey`, `focues`, `prefered`, `constituionally`, `assment`, `studing`, `inorder` in live text |

---

## 4. Suggested order of work

> **Status 2026-07-25: step 1 is DONE.** Tier 0 was applied to the live prose — 155 directives
> applied, 11 subsumed by broader rewrites, 0 failures. All 97 `[POLISH … DROP-IN]` blocks are
> consumed and deleted. Contractions fell 22 → 0 in authored prose (5 remaining are verbatim model
> output and quoted principle names); first person 29 → 0 in authored prose (remaining hits are the
> Acknowledgements, where it is the convention, plus quoted material); `So …` openers 6 → 3 (the 3
> are legitimate "So far"). Spelling fixed: `saftey`, `recognizable`, `typicall`, `focues`,
> `prefered`, `constituionally`, `assment`, `studing`, `inorder`. Braces and math verified balanced
> in all nine files; 155 labels still resolve; 0 dangling refs. Backup of the pre-apply state is in
> the session scratchpad at `backup_pre_tier0/`. **Not yet verified in Overleaf — build before
> trusting.** 87 proposal blocks remain for steps 2–10 below.
>
> One fix was completed beyond a pure find-and-replace: `pdftitle=\thesistitle` →
> `pdftitle=\thesistitleshort`, with `\newcommand{\thesistitleshort}{…}` added live in `main.tex`,
> since the replacement alone would have left an undefined control sequence.
>
> Two items deliberately NOT auto-applied because they are your decisions, not mechanical fixes:
> the Declaration consent wording (`I consent / do not consent` — both still printed), and the
> bootstrap-interval sentence in the parity claim (its drop-in contains `[X, Y]` placeholders that
> must be filled from a real resampling run).

1. ~~Apply all `[POLISH … DROP-IN]` blocks (BLOCKER-1)~~ — **done 2026-07-25.**
2. Fix the Rainone misattribution at all four sites (BLOCKER-2).
3. Uncomment `fig:architecture` and `tab:hypothesis-verdicts` (MAJOR-8) — two floats, high value.
4. Write `\section{Challenges Encountered}` in the Conclusion (MAJOR-7).
5. Name the judge models and correct the NIM claim (BLOCKER-5).
6. Add bootstrap intervals to the two rank tables and replace "within the noise" (BLOCKER-3).
7. Report the exposure ablation and judge repeat variance (BLOCKER-4).
8. Sweep first person and contractions (MODERATE-9).
9. Execute the existing `[TRIM]`/`[COMPRESS]` cuts and reinvest in State of the Art (MAJOR-6).
10. Front-matter and preamble cleanup (MINOR-12); resolve MODERATE-10 and MODERATE-11.

Items 1–4 are the ones that move bands. Items 5–7 are what separate the top of the 70+ band from
the bottom of it on the 50%-weighted criterion.

---

## 5. Beyond compliance: making the reader understand and connect the dots

Everything above is about not losing marks. This section is about the different goal — a reader who
finishes the dissertation holding the whole argument. Six devices, each drafted in the `.tex` files.

### The core diagnostic

**The Background chapter has 26 sections and subsections and exactly one forward reference into the
Results or Discussion chapters.** That single number explains why the chapter reads as descriptive
despite being full of load-bearing material: the material *is* load-bearing, but the load is never
shown. A reader cannot connect dots that were never drawn as dots.

The same pattern repeats at document level. The dissertation contains a tight causal argument —
privacy forces the model onto the device → the device forces it below 1B → that size breaks the
self-critique step canonical CAI depends on → so the constitution must be taught by a teacher → which
buys compliance at the cost of reasoning → which reveals a fixed capacity budget → which motivates
splitting it across two models. Every link is established somewhere. **The chain is never written
down as a chain.**

### The six devices

| # | Device | Where drafted | What it fixes |
|---|---|---|---|
| 1 | **"The Argument in Brief"** — the eleven-step chain, numbered, each step citing the section that establishes it | `introduction.tex`, before §Dissertation Structure | The reader meets every later chapter as a step they are already expecting. §Dissertation Structure says what each chapter *contains*; this says what the dissertation *argues*. Different jobs; only one was being done. |
| 2 | **Foundation-payoff table** — foundation → property established → design decision forced → where tested (14 rows) | end of `background.tex` | Retro-justifies all nine thousand words in one page. After it, no examiner asks "why was the tokenisation section here?" A row that cannot be filled is a section that should be cut. |
| 3 | **Forward-debt clauses** — name the debt where the foundation is incurred | convention note in `background.tex` | Converts the chapter from a tour of topics into a chain, prospectively. Two or three per section suffices. |
| 4 | **Prediction before result** — state C3AI's typology as a *prediction*, confirm it by name in the results | `methodology-results.tex` §constitution | The strongest single analytical move available and currently wasted: C3AI predicts prohibitions instil more easily than generative instructions, and your results show exactly that — but it is introduced as a grouping rationale and only cashed in post-hoc in the Discussion. A prediction made in advance and confirmed is what "critical and analytical" means. Same treatment available for LIMA and for Zhang's collapse finding. |
| 5 | **"Why one measurement lens is not enough"** — promote the lens disagreement to its own section | `discussion.tex`, before §Limitations | Your most interesting *methodological* discovery is currently filed as the verdict of the third hypothesis. Since the stated contribution is a measuring framework, the finding that a single lens would have given the wrong answer **is** that contribution demonstrating itself. |
| 6 | **Validity structure** — reorganise Limitations under construct / internal / external / conclusion validity | `discussion.tex`, before §Limitations | Almost no new writing; the existing subsections remap onto three of the four headings. Exposes the empty fourth (conclusion validity) as exactly where the missing intervals belong — and surfaces an unclaimed strength: the question set spans 16 situation categories and 20 region/culture/demographic axes, a deliberate defence against Western-default evaluation that is currently invisible in the live text. |

### Presentation devices (drafted in `main.tex` house style guide)

- **Captions state the finding, not the contents.** The cheapest high-impact change in the document.
  You already do this correctly once (`fig:think-distribution`) and descriptively everywhere else.
  A float whose caption cannot be given a finding is probably confirmatory — and a cut candidate.
- **No float without an introducing sentence.** `tab:cai-at-scale` currently has none.
- **One term per concept, fixed at first use.** "visible reasoning" not three synonyms; "condition"
  not "variant"/"arm"; "the constitutional model" and "the dual model" fixed once.
- **Progressive architecture figure** (optional, strongest through-line device): draw
  `fig:architecture` once, then reuse the same diagram at the head of Methodology, Experiments and
  Discussion with everything greyed except the part that chapter concerns. One TikZ definition plus
  three opacity settings. The reader is never lost because they are always looking at the same
  picture with a different part lit — and it directly answers the supervisor's point that an examiner
  without a language-model background must be able to follow every chapter.

### If only three things get done

1. **Device 1** (the argument chain) — the whole document becomes navigable.
2. **Device 2** (the foundation-payoff table) — the 15% Background problem is solved in one page.
3. **Finding-first captions** — every skim-reader learns the results without reading the prose.

---

## 6. Keeping the reader to the end

The instinct when a reader might get lost is to explain more. That is the wrong move — more
explanation is more to get lost in. What holds a reader is (a) always knowing where they are,
(b) always having an unanswered question pulling them forward, and (c) being able to read at
whatever depth they have time for without falling off.

### The length budget, so this does not silently grow the document

| | Words |
|---|---:|
| Argument chain (Device 1) | +350 |
| Foundation-payoff table (Device 2) | +1 page |
| Chapter-closing bridges × 5 | +420 |
| Section takeaway lines × ~20 | +500 |
| Running example | ~0 (replaces existing examples) |
| **Added** | **~1,300 words + 1 page** |
| **Available from existing `[TRIM]`/`[COMPRESS]`/`[CUT]` blocks** | **~10 pages** |

Execute the trims **first**, then add these. The dissertation ends up shorter than it is now and
considerably easier to follow. A page that orients a reader is worth more than a page that informs
one, and that is the trade to make when space is tight.

### The five retention devices

| Device | Where drafted | Why it works |
|---|---|---|
| **A. Chapter-closing bridges** | end of `introduction`, `background`, `state-of-the-art`, `methodology-results`, `experiments-results`, `discussion` — full text drafted | Three sentences: what to carry forward, the question now open, the chapter that answers it. An **open loop** is the actual reason a reader turns the page. Currently every chapter simply stops. |
| **B. One running example** | `introduction.tex`, with all five insertion points and exact text | The document uses five different throwaway examples that never accumulate. One request — *"Can you recommend a good programming book for me?"* — threaded from Chapter 1 as a puzzle to Chapter 5 as data gives the reader something concrete to hold while the abstractions change. It is already Case 2, needs no technical background, and all five conditions answer it differently. |
| **C. Layered reading** | house style guide, `main.tex` | Three depths: openers + argument chain + captions; then takeaway lines + summary tables; then full prose. An examiner under time pressure reads depth 1 or 2 — design for that reader deliberately rather than hoping they read everything. |
| **D. Section takeaway lines** | `background.tex`, exact text for all 11 sections | One italicised sentence per section: *"To carry forward: …"*. What makes depth-2 reading work. Also a cutting test — a section whose takeaway is hard to write is usually a section to cut. |
| **E. Plain gloss on first use** | house style guide | A technical term gets its plain meaning in the *same* sentence it first appears, never a later one. Three ungloseed terms in a row and a reader stops; the second reader is from linguistics. |

### What not to do

Do not add a recap section at the head of each chapter. It feels helpful and it is exactly how
documents become long and repetitive — this audit already found the same claim stated five times
across chapters. The bridge at the **end** of a chapter does the same job in a fifth of the words
and does not tempt restatement.

### Measured terminology drift (drop-in replacements in `main.tex`)

One concept, one name. Current counts:

- **visible reasoning** (22) — replace *externalised reasoning* (6), *reasoning trace* (5),
  *written-out reasoning* (2), *visible working* (2), *think trace* (1). Keep `<think>` as the name
  of the mechanism and "visible reasoning" as the name of what it contains. Metric names
  (empty-`<think>` rate, externalisation ratio) are fixed and must not be reworded.
- **condition** (99) — replace *variant* (11, except the prompt and corpus variants, which are
  correct) and *arm* (1).
- **staged design** for the design, **ladder** (33) for the figure that plots it. Replace
  *experimental sequence* (1).
