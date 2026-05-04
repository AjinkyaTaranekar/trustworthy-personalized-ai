---
title: Psychological Grounding of the Constitution
type: topic
tags: [constitution, psychology, trust, reasoning, empathy, evaluation]
sources:
  - pipeline/constitution.md
updated: 2026-05-03
status: current
---

# Psychological Grounding of the Constitution

**The 19-principle constitution is not self-authored speculation — each principle maps to a validated construct from psychology, human-computer interaction, or epistemology, giving the training data external validity.**

Without this grounding, the constitution is merely a prompt and the training pipeline is circular distillation. With it, the constitution becomes a codified, literature-backed specification of trustworthy behaviour that can be evaluated against independent human judgment. This page documents that mapping explicitly so the thesis contribution is defensible.

---

## Why grounding matters for the thesis

The core epistemological problem with LLM-generated training data is that the teacher model's biases get compressed into the student. The constitution breaks this circularity in two ways: (1) the principles themselves are externally grounded in human values research, not derived from the teacher model, and (2) evaluation against those principles can be performed by humans using the [[experiments/human-evaluation-rubric]], providing a ground truth that is independent of both models. See [[decisions/2026-05-03-research-question-reframe]] for the broader framing.

---

## Three organising frameworks

The 19 principles cluster into three bodies of literature.

### Framework A: Trust theory (Mayer, Davis & Schoorman 1995)

The most-cited model in organisational trust research. Trust has three dimensions:

- **Ability** — the trustee has the skills and competence to perform the task.
- **Benevolence** — the trustee is motivated by the beneficiary's interests, not only their own.
- **Integrity** — the trustee adheres to a set of principles that the beneficiary finds acceptable.

This framework was extended to human-computer trust by Lee & See (2004), who showed that the same three dimensions govern whether operators trust automated systems. It is the theoretical anchor for the rubric in [[experiments/human-evaluation-rubric]].

### Framework B: Epistemic virtue and metacognition

Zagzebski (1996) defines epistemic virtues as stable cognitive dispositions that reliably produce true belief: intellectual humility, open-mindedness, and calibration. Kahneman (2011) operationalises the distinction between fast intuitive System 1 reasoning and deliberate System 2 reasoning — the capability-check pattern in the constitution is a forced System 2 intervention. Metacognition (Flavell 1979; Nelson & Narens 1990) is the capacity to monitor and regulate one's own cognitive processes — tool inventory and self-correction in the constitution are direct operationalisations of metacognitive monitoring.

### Framework C: Conversational theory and distributed cognition

Clark & Brennan (1991) establish conversational grounding: both parties in a dialogue must reach a shared understanding, and the cost of repair increases with conversational distance. This is the theoretical basis for one-question-at-a-time clarification. Grice (1975) provides the four maxims (quantity, quality, relation, manner) that govern cooperative conversation — each maps to a constitution principle. Hutchins (1995) and Clark & Chalmers (1998) articulate distributed and extended cognition: intelligent behaviour is not confined to the skull but is produced by the system of agent + environment + tools. The constitution's tool-use principles operationalise this for AI systems.

---

## Principle-by-principle mapping

| # | Principle | Framework | Key Citation | Mechanism |
|---|-----------|-----------|-------------|-----------|
| 1 | DECOMPOSE FIRST | Epistemic virtue | Kahneman (2011), *Thinking Fast and Slow* | Forces System 2 engagement before commitment; prevents availability heuristic errors |
| 2 | TOOL INVENTORY | Metacognition | Endsley (1995), situational awareness; Flavell (1979) | Metacognitive monitoring of available cognitive resources |
| 3 | TOOL DISCIPLINE | Distributed cognition | Parasuraman & Riley (1997), appropriate automation | Misuse of automation causes automation bias; discipline prevents false confidence |
| 4 | MATH = CODE | Extended mind | Clark & Chalmers (1998); Risko & Gilbert (2016), cognitive offloading | Arithmetic is an extended-mind task; the calculator is part of the cognitive system |
| 5 | REAL-TIME HONESTY | Epistemic virtue / Integrity | Zagzebski (1996); Mayer et al. (1995) — Integrity | Grice's Maxim of Quality: do not assert what you believe to be false or lack evidence for |
| 6 | USER CONTEXT GATE | Person-centred communication | Rogers (1951), *Client-Centred Therapy*; Nissenbaum (2004), contextual integrity | Imposing demographic assumptions violates contextual integrity and undermines benevolence |
| 7 | UNCERTAINTY QUANTIFICATION | Calibration | Fischhoff, Slovic & Lichtenstein (1977); Kahneman et al. (1982), *Judgment under Uncertainty* | Overconfidence is the most documented calibration failure; explicit uncertainty reduces it |
| 8 | IMPOSSIBILITY ACKNOWLEDGMENT | Epistemic humility | Whitcomb et al. (2017), "Intellectual Humility: Owning Our Limitations" | Epistemic courage to state limits — prevents confabulation |
| 9 | TRADEOFF PRESENTATION | Decision support | Simon (1956), satisficing; Turban (1990), decision support systems | Presenting tradeoffs respects user agency and avoids anchoring bias |
| 10 | CORRECT TOOL USE | Distributed cognition | Hutchins (1995), *Cognition in the Wild* | Artefacts must be used correctly for distributed cognition to produce reliable results |
| 11 | TOOL AVOIDANCE | Cognitive load | Sweller (1988), cognitive load theory; Parasuraman & Riley (1997) | Unnecessary tool calls increase cognitive load without epistemic gain; automation bias risk |
| 12 | TOOL FAILURE HANDLING | Resilience | Hollnagel (2006), resilience engineering | Graceful degradation — system maintains function at reduced capacity rather than failing silently |
| 13 | NO TOOL FAKING | Integrity | Mayer et al. (1995) — Integrity; Bok (1978), *Lying* | Deception via false procedural authority violates the integrity dimension of trust irreparably |
| 14 | HOLD UNDER PRESSURE | Social influence resistance | Cialdini (1984), *Influence*; Asch (1956), conformity | Social pressure is the most reliable way to elicit confabulation — resistance is integrity operationalised |
| 15 | EXPLICIT SELF-CORRECTION | Metacognitive control | Nelson & Narens (1990); Falkenstein et al. (2000), error-related negativity | Transparent error correction preserves the ability audit; silent correction destroys it |
| 16 | KNOWLEDGE CUTOFF AWARENESS | Temporal epistemics | Metcalfe & Shimamura (1994), *Metacognition*; Tulving (1983), episodic memory | Failure to distinguish "when I learned this" from "when this was true" is a systematic memory error |
| 17 | MULTI-STEP CLARIFICATION | Conversational grounding | Clark & Brennan (1991); Grice (1975) — Quantity maxim | Grounding is incremental; dumping all questions violates quantity and burdens the user's working memory |
| 18 | EXPLICIT I DON'T KNOW | Epistemic integrity | Zagzebski (1996); Socratic principle | The most durable epistemic virtue: confessing ignorance rather than constructing a plausible fiction |
| 19 | SEARCH FOR FACTS ABOUT ENTITIES | Information foraging | Pirolli & Card (1999), information foraging theory; Tulving (1983) | Entity facts are semantically decaying; information foraging predicts seeking fresh sources over recalled stale data |

---

## Implication for evaluation

Because each principle is grounded in an independent theoretical construct, human evaluators can assess compliance without knowing the constitution. The [[experiments/human-evaluation-rubric]] operationalises the Mayer et al. trust dimensions (Ability items 1–3, Integrity items 4–6, Benevolence items 7–8) and the Davis (1983) empathy index (Empathy items 9–10). This gives the comparison study an external validity baseline that is independent of the LLM-generated training data.

---

## Related

- [[experiments/human-evaluation-rubric]]
- [[experiments/frontier-model-comparison]]
- [[decisions/2026-05-03-research-question-reframe]]
- [[entities/constitution]]
- [[topics/reasoning]]
- [[topics/empathy]]

## Sources

- Mayer, R. C., Davis, J. H., & Schoorman, F. D. (1995). An integrative model of organizational trust. *Academy of Management Review*, 20(3), 709–734.
- Lee, J. D., & See, K. A. (2004). Trust in automation. *Human Factors*, 46(1), 50–80.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Zagzebski, L. (1996). *Virtues of the Mind*. Cambridge University Press.
- Flavell, J. H. (1979). Metacognition and cognitive monitoring. *American Psychologist*, 34(10), 906–911.
- Nelson, T. O., & Narens, L. (1990). Metamemory. *The Psychology of Learning and Motivation*, 26, 125–173.
- Clark, H. H., & Brennan, S. E. (1991). Grounding in communication. *Perspectives on Socially Shared Cognition*, 13, 127–149.
- Grice, H. P. (1975). Logic and conversation. *Syntax and Semantics*, 3, 41–58.
- Hutchins, E. (1995). *Cognition in the Wild*. MIT Press.
- Clark, A., & Chalmers, D. (1998). The extended mind. *Analysis*, 58(1), 7–19.
- Risko, E. F., & Gilbert, S. J. (2016). Cognitive offloading. *Trends in Cognitive Sciences*, 20(9), 676–688.
- Parasuraman, R., & Riley, V. (1997). Humans and automation. *Human Factors*, 39(2), 230–253.
- Sweller, J. (1988). Cognitive load during problem solving. *Cognitive Science*, 12(2), 257–285.
- Fischhoff, B., Slovic, P., & Lichtenstein, S. (1977). Knowing with certainty. *Journal of Experimental Psychology: Human Perception and Performance*, 3(4), 552–564.
- Nissenbaum, H. (2004). Privacy as contextual integrity. *Washington Law Review*, 79(1), 119–158.
- Rogers, C. R. (1951). *Client-Centred Therapy*. Houghton Mifflin.
- Cialdini, R. B. (1984). *Influence: The Psychology of Persuasion*. Harper Collins.
- Pirolli, P., & Card, S. (1999). Information foraging. *Psychological Review*, 106(4), 643–675.
- Whitcomb, D. et al. (2017). Intellectual humility. *Philosophy and Phenomenological Research*, 94(3), 509–539.
- Davis, M. H. (1983). Measuring individual differences in empathy. *Journal of Personality and Social Psychology*, 44(1), 113–126.
- Simon, H. A. (1956). Rational choice and the structure of the environment. *Psychological Review*, 63(2), 129–138.
- Hollnagel, E. (2006). *Resilience Engineering*. Ashgate.
