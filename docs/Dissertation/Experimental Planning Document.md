# Experimental Planning Document
## Trustworthy and Empathetic AI Research

**Purpose:** Planning experimental approaches for the Master's dissertation on building trustworthy conversational LLMs with explainability, accuracy, privacy-preservation, and genuine empathy

**Last Updated:** November 10, 2025  
**Status:** Post-meeting revision - focus shifted to ontology-LLM integration

---

## Meeting Summary (November 10, 2025)

### Key Discussion Points

#### 1. Fundamental LLM Limitations Identified
- **Token Understanding Problem:** Transformers cannot understand the meaning of tokens; they only see them as IDs
- **Architectural Constraint:** LLMs cannot explain token generation process due to inherent limitations
- **Two Main Reasons for Lack of Explainability:**
  1. Transformers lack semantic understanding of token meanings
  2. Output complexity is influenced by multiple parameters simultaneously, making attribution impossible
- **Current Industry Approach:** Use MCP servers and external tools for fact-based reasoning, but this doesn't address the core explainability problem

#### 2. Research Direction: Focus on Practical, Measurable Components
- **Advisor's Guidance:** Move away from purely theoretical concepts toward concrete experimental basis
- **Core Challenge:** Address the "sociopath yapper" problem of post-hoc rationalization in LLMs
- **Proposed Solution:** Use ontological spaces to control and ratify answers before presentation

#### 3. New Primary Research Focus: Ontology-LLM Integration
- **Core Idea:** Integrate structured ontologies with LLMs to enhance reasoning and explanation capabilities
- **Two Complementary Approaches:**
  1. **Ontology as Knowledge Base:** Use ontology as the core reasoning component, with LLM as an intelligent linguistic interface
  2. **Ontology as Verifier:** LLM generates answers, ontology validates/ratifies them for accuracy and logical consistency

#### 4. Differentiation from Current Industry Approaches
- **Current Method:** Delegate tasks to external resources (MCP servers, tools, databases)
- **Proposed Method:** Use post-hoc verification processes where ontology assesses LLM responses for coherence and factual accuracy
- **Key Distinction:** Not just retrieval, but logical reasoning and validation through structured knowledge representation

#### 5. Potential Experimental Domains
- **Suggested Focus:** Areas where LLMs struggle, particularly:
  - **Politics and Geopolitical Understanding:** Test with ontologies representing different global perspectives (e.g., Western vs. Eastern political frameworks)
  - **Complex Reasoning Tasks:** Logic, causality, multi-step inference
  - **Knowledge Verification:** Fact-checking, consistency checking across claims

#### 6. Three Complementary AI Approaches Discussed
- **Human Brain Frequency Modes:** Hierarchical reasoning inspired by brain architecture
- **Neurosymbolic AI:** Combining neural networks with symbolic reasoning systems
- **New Quen Model Approach:** Handling different tasks (puzzles vs. text generation) with specialized architectures

#### 7. Next Steps & Timeline
- **Immediate Task:** Outline concrete experimental design for ontology-LLM integration
- **By Early Next Year:** Develop strong understanding of key literature areas and technological basis
- **Next Meeting:** Tuesday at 10am (in person) to discuss refined concrete research ideas
- **Emphasis:** Write down concrete ideas even if they might be rejected - learning from rejection helps explore alternatives

---

## 1. Proposed Experimental Ideas (Bullet-Point Descriptions)

### **NEW PRIMARY FOCUS: Experiment 6 - Ontology-LLM Integration for Trustworthy Reasoning**

**Objective:** Investigate whether integrating structured ontologies with LLMs can address the "sociopath yapper" problem by providing verifiable, explainable reasoning that goes beyond post-hoc rationalization.

#### Approach A: Ontology as Core Knowledge Base
**Design:**
- **Knowledge Representation:** Use existing ontology (e.g., political/geopolitical domain ontology, or domain-specific reasoning ontology)
- **LLM Role:** Acts as intelligent linguistic interface between user and ontology
  - Translates natural language queries into ontology queries
  - Formats ontology outputs into natural language responses
  - Determines routing: "Should this be answered via ontology or via my own generation?"
- **Query Classification Module:**
  - Factual/logical questions → Route to ontology
  - Opinion/creative questions → LLM generation with explicit disclaimer
  - Hybrid questions → Combine ontology facts with LLM interpretation
- **Test Domains:**
  - **Political Reasoning:** Compare Western vs. Eastern political ontologies
  - **Geopolitical Analysis:** Test consistency in analyzing same events through different cultural/political lenses
  - **Logical Puzzles:** Structured reasoning tasks where ontology provides inference rules

**Implementation Steps:**
1. Select/acquire extant ontology (OWL, RDF format)
2. Build query translation layer (NL → SPARQL or similar)
3. Implement routing logic (classification model or rule-based)
4. Develop response formatting layer (ontology output → NL)
5. Create test dataset with ground truth from ontology

**Evaluation Metrics:**
- **Accuracy:** Correctness of answers vs. ontology ground truth
- **Routing Precision:** Correctly identifying which questions should use ontology
- **Explainability:** Can system trace answer back to ontology concepts/relations?
- **Consistency:** Same question asked differently yields same answer
- **Transparency:** User can see which knowledge source was used
- **Cross-Cultural Validation:** How do different ontologies answer same political/geopolitical questions?

**Expected Outcomes:**
- Higher factual accuracy on ontology-domain questions
- Verifiable reasoning chains (can trace through ontology structure)
- Clear distinction between "knowledge-based" vs. "generated" answers
- Reduced hallucination on factual queries

---

#### Approach B: Ontology as Post-Hoc Verifier
**Design:**
- **LLM Generation First:** LLM generates answer to user query as normal
- **Ontology Validation:** Before presenting answer, pass through ontology-based verification
  - Extract factual claims from LLM response
  - Check claims against ontology knowledge base
  - Flag inconsistencies, logical errors, or unsupported claims
  - Optionally: Request LLM to regenerate with corrections
- **Output Options:**
  - **Verified Response:** LLM answer + "Verified by knowledge base" badge
  - **Corrected Response:** LLM answer modified based on ontology feedback
  - **Flagged Response:** LLM answer + warnings about unverified/contradictory claims
  - **Rejected Response:** Ontology generates answer when LLM fails verification

**Implementation Steps:**
1. Claim extraction from LLM outputs (NER, relation extraction)
2. Claim-to-ontology mapping algorithm
3. Consistency checking logic (logical inference, contradiction detection)
4. Feedback integration (how to communicate corrections to LLM)
5. User interface showing verification status

**Evaluation Metrics:**
- **Verification Accuracy:** Correctly identifying true/false claims
- **Hallucination Reduction:** Decrease in factually incorrect statements
- **User Trust:** Does verification badge increase user confidence?
- **Explanation Quality:** Can system explain *why* a claim was flagged?
- **Performance Trade-off:** Latency vs. accuracy improvement

**Expected Outcomes:**
- Catch and correct LLM hallucinations before user sees them
- Provide evidence trail for factual claims
- Build user trust through transparent verification process
- Address "sociopath yapper" by showing work, not just generating plausible text

---

#### Comparison: Ontology-LLM vs. Current Industry Approaches

| Aspect | Current MCP/Tool Approach | Proposed Ontology Approach |
|--------|---------------------------|----------------------------|
| **Knowledge Source** | External APIs, databases, calculators | Structured ontology with inference capabilities |
| **LLM Role** | Task delegator | Interface (Approach A) or Generator-to-be-verified (Approach B) |
| **Reasoning** | Delegated to tools | Logical inference within ontology |
| **Explainability** | "I called tool X" | "Based on ontology concept Y, relation Z..." |
| **Verification** | Tool output assumed correct | Ontology validates logical consistency |
| **Scope** | Task-specific (math, search, etc.) | Domain knowledge and reasoning patterns |
| **Novelty** | Established practice | Post-hoc verification and routing intelligence |

---

### PREVIOUS EXPERIMENTS (Now Secondary Priority)

### Experiment 1: Reasoning Process Reward System
Objective: Test whether rewarding correct thought processes (rather than final answers) improves reasoning capability in LLMs.

Design:
- Setup: Fine-tune a smaller LLM (e.g., Llama 3.1 8B or similar) using reinforcement learning
- Reward Structure: 
  - Process rewards: Award points for logical step-by-step breakdown, proper use of 5W+H framework
  - Outcome penalties: Penalize incorrect final answers but analyze whether correct process was followed
  - Mixed rewards: Combine both to compare effectiveness
- Test Tasks: 
  - Mathematical word problems with multiple steps
  - Logic puzzles (e.g., modified Sudoku, constraint satisfaction problems)
  - Novel problems not seen in training (to test generalization of thought process)
- Evaluation Metrics:
  - Accuracy on final answers
  - Quality of intermediate reasoning steps (human-evaluated rubric)
  - Robustness to problem variations (e.g., Apple's "illusion thinking" style tests)
  - Transfer learning to new domains
- Control Group: Standard outcome-only reward LLM
- Expected Outcome: Process-rewarded model should show better reasoning on novel problems even if training accuracy is similar

---

### Experiment 2: Empathetic Response Generation Using Appraisal Theory
Objective: Evaluate whether LLMs can generate genuinely empathetic responses based on appraisal detection from user conversations.

Design:
- Dataset: Use Crowd-event dataset (6600 events with 21 appraisals) + collect new conversational data
- Phase 1 - Appraisal Detection:
  - Fine-tune LLM to detect appraisal dimensions from user statements
  - Test on zero-shot conversations with emotional content
  - Validate against human annotators
- Phase 2 - Response Generation:
  - Train model to generate responses conditioned on detected appraisals
  - Compare three approaches:
    1. Standard LLM responses (baseline)
    2. Appraisal-conditioned responses (proposed)
    3. Human-written empathetic responses (gold standard)
- Evaluation Metrics:
  - Human evaluation: Empathy ratings (1-5 scale), perceived understanding, helpfulness
  - Appraisal detection accuracy (precision, recall, F1)
  - User satisfaction in real conversations
  - Contextual appropriateness scoring
- Critical Questions to Address:
  - Clarity of 21 appraisal dimensions to average users
  - Temporal effects (time gap between event and recording)
  - Pre-existing mood influence on appraisals
  - Demographic variations in appraisal interpretation
- Expected Outcome: Appraisal-aware model should score significantly higher on empathy and appropriateness than baseline

---

### Experiment 3: Proactive Questioning vs. Assumption-Based Responses
Objective: Test whether AI that asks clarifying questions before responding is perceived as more empathetic and helpful than AI that makes assumptions.

Design:
- Scenario-Based Study:
  - Create 20-30 ambiguous user queries (e.g., "I reached late to class on the first day and got scolded. What to do?")
  - Develop two AI response strategies:
    - Assumption-Based (Control): Provide generic advice immediately
    - Inquiry-Based (Experimental): Ask 1-2 targeted clarifying questions using 5W+H framework before responding
- Participant Study (N=50-100):
  - Within-subject design: Each participant interacts with both AI types across different scenarios
  - Randomize order and scenario assignment
- Evaluation Metrics:
  - Perceived empathy (validated empathy scale)
  - Usefulness of advice (Likert scale)
  - User satisfaction and trust
  - Preferred interaction style (forced choice)
  - Qualitative feedback on experience
- Conversation Analysis:
  - Number of turns needed to reach resolution
  - User engagement level (response length, detail shared)
  - Actual relevance of final advice to user's real situation
- Expected Outcome: Inquiry-based approach should score higher on empathy and usefulness despite requiring more interaction turns

---

### Experiment 4: Hybrid Reasoning Architecture (Latent + Explicit)
Objective: Build and evaluate a hybrid system combining latent space reasoning (HRM-inspired) with explicit natural language explanation generation.

Design:
- Architecture:
  - Module 1: Latent Reasoner - Processes problems in vector space with recurrent depth
  - Module 2: Language Generator - Translates latent reasoning states into natural language steps
  - Module 3: Tool Augmentation - Calls external calculators/interpreters (PAL approach)
- Implementation:
  - Use CoCoNut-style continuous latent reasoning
  - Implement interleaved thinking pattern (MiniMax M2/Kimi 2 inspired)
  - Add RAG module for source citation
- Benchmark Tasks:
  - Complex mathematics (multi-step arithmetic, algebra)
  - Logic puzzles (ToT benchmark tasks)
  - Real-world reasoning (medical diagnosis scenarios, legal case analysis)
- Evaluation Metrics:
  - Task accuracy vs. pure LLM baseline
  - Explainability quality (human rating of generated explanations)
  - Token efficiency (tokens per solution vs. CoT)
  - Computation transparency (can users verify steps?)
- Scrutability Assessment:
  - Source attribution accuracy
  - Tool usage transparency
  - Alignment between latent reasoning and natural language explanation
- Expected Outcome: Hybrid system should achieve higher accuracy with better explainability than either pure LLM or unexplained latent models

---

### Experiment 5: Real-Time User Modeling with Dynamic Adaptation
Objective: Develop and test a system that builds and updates user models during conversation to personalize responses.

Design:
- User Model Components:
  - Preference tracking (explicitly stated and inferred)
  - Emotional state detection (from text/voice features)
  - Conversation pace adaptation (response length, complexity)
  - Context memory (recent topics, goals, concerns)
  - Interaction style matching (formal/casual, detailed/brief)
- Implementation:
  - Cold start: Initial profiling with metadata and first few interactions
  - Dynamic update: Continuously refine model during conversation
  - Memory management: Short-term (session) vs. long-term (cross-session) retention
- Test Scenarios:
  - Multi-turn conversations on personal topics (career advice, relationship issues, etc.)
  - Returning users (test long-term memory)
  - Different interaction styles per user
- Evaluation:
  - User satisfaction across conversation turns (does it improve?)
  - Personalization accuracy (predicted preferences vs. actual)
  - Adaptation responsiveness (how quickly does it adjust?)
  - Privacy concerns (user comfort with model persistence)
- Control Conditions:
  - No user modeling (standard LLM)
  - Static user profile (set at start, no updates)
  - Dynamic user modeling (proposed)
- Expected Outcome: Dynamic modeling should show increasing satisfaction over conversation turns and higher overall ratings

---

## 2. Related Research Documentation

### Core Reasoning & Thinking Architectures

#### Chain of Thought (CoT) & Extensions
- Foundational: *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (2201.11903v6)
  - Establishes that explicit step-by-step reasoning improves LLM performance
  - Mechanism: Expands computational scratchpad through additional tokens
  - Limitation: Linear, no backtracking possible
  - Relevant to: Experiments 1, 4 (baseline comparison)

- Automatic CoT: *Automatic Chain of Thought Prompting in Large Language Models* (2210.03493v1)
  - Template-based fine-tuning for structured reasoning
  - Relevant to: Experiment 1 (process reward design)

- Tree of Thoughts (ToT): *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (2305.10601v2)
  - Explores multiple reasoning branches with evaluation/pruning
  - Hierarchical problem-solving approach
  - Relevant to: Experiment 4 (architecture inspiration)

#### Advanced Reasoning Models

- DeepSeek R1: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2501.12948v1)
  - Shows "more CoT = better quality" but with latency trade-off
  - RL approach to incentivize reasoning
  - Relevant to: Experiment 1 (RL for reasoning processes)

- Hierarchical Reasoning Model (HRM): *Hierarchical Reasoning Model* (2506.21734v3)
  - Specialist reasoning engine, not general LLM
  - Recurrent, stateful computation in latent space
  - Excels at logic puzzles where LLMs fail
  - Limitation: Not inherently explainable in natural language
  - Relevant to: Experiment 4 (latent reasoning component)

- Latent Space Reasoning: *Training Large Language Models to Reason in a Continuous Latent Space* (CoCoNut, 2412.06769v3)
  - Addresses token-heavy CoT inefficiency
  - Uses latent vectors for reasoning
  - Relevant to: Experiment 4 (efficiency improvements)

- Interleaved Thinking: *Reasoning with Latent Thoughts: On the Power of Looped Transformers* (2502.17416v1)
  - Parallel reasoning and answer generation (MiniMax M2, Kimi 2)
  - Dynamic compute allocation based on problem complexity
  - Mimics human iterative thinking
  - Relevant to: Experiment 4 (architecture design)

#### Reasoning Limitations & Evaluation

- Hidden Reasoning: *Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding* (2411.04282v2)
  - RL for unlocking latent reasoning
  - Addresses thought process vs. outcome rewards
  - Relevant to: Experiment 1 (core motivation)

- Memorization vs. Reasoning: *None of the Others: a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks* (2502.12896v5)
  - Critical problem: LLMs fail when problems are modified (illusion thinking)
  - Tests true reasoning vs. pattern matching
  - Relevant to: Experiment 1 (evaluation design)

- Decreasing CoT Value: *Prompting Science Report 2: The Decreasing Value of Chain of Thought in Prompting* (2506.07142v1)
  - Questions universal benefit of CoT
  - Context-dependent effectiveness
  - Relevant to: Experiments 1, 4 (when to use what approach)

### Tool-Augmented & Hybrid Systems

- PAL: *PAL: Program-aided Language Models* (2211.10435v2)
  - LLM as reasoner, external tools as executors
  - Improves scrutability and accuracy for computation
  - Honest explainability: "I used a calculator"
  - Relevant to: Experiment 4 (tool augmentation module)

- ReAct: *ReAct: Synergizing Reasoning and Acting in Language Models* (2210.03629v3)
  - Combines reasoning with action-taking
  - Interleaved thought-action-observation pattern
  - Relevant to: Experiment 4 (architecture patterns)

### Empathy, Sentiment & Appraisal

- Appraisal-Based Empathy: *Explainable Sentiment Analysis with DeepSeek-R1: Performance, Efficiency, and Few-Shot Learning* (2503.11655v4)
  - Addresses "sociopath yapper" problem
  - Post-hoc rationalization vs. genuine reasoning
  - Relevant to: Experiments 2, 3 (empathy framework)

- Sentiment & Reasoning: *Dual-Head Reasoning Distillation: Improving Classifier Accuracy with Train-Time-Only Reasoning* (2509.21487v2)
  - Distillation approach for reasoning
  - Relevant to: Experiment 1, 2 (training methodologies)

### Foundational Architecture & Representations

- Transformer Architecture: *Attention Is All You Need* (1706.03762v7)
  - Foundation for modern LLMs
  - Self-attention mechanism
  - Limitation: Feed-forward, no backtracking
  - Relevant to: Understanding architectural constraints across all experiments

- Word Embeddings: *Efficient Estimation of Word Representations in Vector Space* (1301.3781v3)
  - Understanding tokenization fundamentals
  - Relevant to: Experiment 4 (latent space reasoning)

- Subword Tokenization: *Neural Machine Translation of Rare Words with Subword Units* (1508.07909v5)
  - BPE algorithm
  - Explains why LLMs fail at computation (number fragmentation)
  - Relevant to: Experiment 4 (tool delegation justification)

- Word Significance: *Measuring Word Significance using Distributed Representations of Words* (1508.02297v1)
  - Distributional semantics
  - Relevant to: User modeling approaches (Experiment 5)

- BERT Pre-training: *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (1810.04805v2)
  - Bidirectional context understanding
  - Relevant to: Contextual understanding for empathy (Experiments 2, 3, 5)

### Reasoning Diffusion & Alternative Approaches

- Diffusion of Thoughts: *Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models* (2402.07754v3)
  - Alternative paradigm to autoregressive reasoning
  - Relevant to: Experiment 4 (exploring alternative architectures)

- LaDiR: *LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning* (2510.04573v3)
  - Latent diffusion for reasoning enhancement
  - Relevant to: Experiment 4 (latent reasoning methods)

### Reinforcement Learning & Training

- RL for Reasoning: *Interleaved Reasoning for Large Language Models via Reinforcement Learning* (2505.19640v1)
  - RL techniques for reasoning capability
  - Relevant to: Experiment 1 (training methodology)

- Self-Enhanced Reasoning: *Self-Enhanced Reasoning Training: Activating Latent Reasoning in Small Models for Enhanced Reasoning Distillation* (2502.12744v1)
  - Training small models to "think" rather than just "answer"
  - Distillation from larger reasoning models
  - Relevant to: Experiment 1 (thought process learning)

- Seed1.5-Thinking: *Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning* (2504.13914v3)
  - Reward correct thought processes, not just outcomes
  - Relevant to: Experiment 1 (core methodology)

### Multi-Agent & Advanced Systems

- Model Context Protocol (MCP): *Advancing Multi-Agent Systems Through Model Context Protocol: Architecture, Implementation, and Applications* (2504.21030v1)
  - Framework for agent communication and coordination
  - Relevant to: Experiment 4 (modular architecture design)

---

## 3. Meeting Discussion Preparation

### Priority Ranking for Experiments (Updated Post-Meeting)

**HIGHEST PRIORITY (Primary Focus):**
1. **Experiment 6 (Ontology-LLM Integration)** - Directly addresses advisor's concern about post-hoc rationalization, has clear experimental basis, measurable outcomes, and practical implementation path
   - Start with Approach A (Ontology as Knowledge Base) for clearer methodology
   - Approach B (Post-hoc Verification) as extension or alternative track

**High Priority (Complementary to Main Focus):**
2. **Experiment 3 (Proactive Questioning)** - Still relevant for empathy aspects, can be integrated with ontology routing logic
3. **Literature Review on Ontologies & Neurosymbolic AI** - Critical for understanding existing work and positioning contribution

**Medium Priority (Potential Integration):**
4. **Experiment 4 (Hybrid Architecture)** - Could incorporate ontology component as one module
5. **Experiment 2 (Appraisal-Based Empathy)** - May be combined with Exp 3 or pursued separately

**Lower Priority (Future Work):**
6. **Experiment 1 (Process Rewards)** - Interesting but resource-intensive, defer unless clear connection to ontology work emerges
7. **Experiment 5 (User Modeling)** - Important but outside core trustworthiness focus for now

---

### Key Questions for Next Meeting (Tuesday 10am)

#### Ontology-LLM Integration Specifics
1. **Ontology Selection:**
   - Which specific ontology should we use for initial experiments?
   - Should we focus on political/geopolitical domain as suggested, or explore alternatives?
   - Existing ontologies to consider: DBpedia, Wikidata, domain-specific (medical, legal, political)?

2. **Approach Selection:**
   - Should we pursue Approach A (Ontology as KB) or Approach B (Post-hoc Verifier) first?
   - Or develop both in parallel with smaller scope for each?
   - Which approach has stronger contribution potential?

3. **Technical Architecture:**
   - What query language should we use? (SPARQL, Cypher, custom)?
   - How to handle ontology inference? (OWL reasoning, rule-based, neural approximation?)
   - Integration point: API calls to ontology server vs. embedded reasoning engine?

#### Methodology Concerns for Ontology Experiments

4. **Query Routing Logic:**
   - How to determine which questions should go to ontology vs. LLM generation?
   - Machine learning classifier, rule-based system, or hybrid?
   - What training data exists for this task?

5. **Ground Truth & Evaluation:**
   - How to create test datasets with verifiable ground truth?
   - For political/geopolitical questions: who defines "correct" answers?
   - Should we test across multiple ontologies (different perspectives) simultaneously?

6. **Explainability vs. Complexity:**
   - How detailed should ontology-based explanations be?
   - Trade-off: Full reasoning trace vs. user-friendly summary?
   - Can users understand/verify ontology reasoning chains?

7. **Handling Ontology Limitations:**
   - What happens when query is outside ontology scope?
   - How to communicate uncertainty or knowledge gaps?
   - Fallback strategies when ontology doesn't have answer?

#### Evaluation Challenges

8. **Measuring Trustworthiness:**
   - How to quantify improvement in trustworthiness vs. baseline LLM?
   - Metrics: Factual accuracy, consistency, user trust ratings, expert verification?
   - Should we measure both objective correctness and perceived trustworthiness?

9. **Cross-Cultural/Perspective Testing:**
   - For political ontologies: How to fairly evaluate different worldviews?
   - Is the goal to show consistency within an ontology, or correctness by some standard?
   - Ethical implications of encoding particular political perspectives?

10. **Baseline Comparisons:**
   - Compare against: Pure LLM, RAG system, Tool-augmented LLM, other neurosymbolic approaches?
   - How to ensure fair comparison (same domains, same knowledge coverage)?

#### Technical Feasibility

11. **Ontology Engineering Requirements:**
   - Do we need ontology expertise, or can we use existing ontologies as-is?
   - If modifications needed, what tools/frameworks? (Protégé, custom scripts?)
   - Ontology size/complexity constraints for real-time querying?

12. **Integration Complexity:**
   - How much engineering effort for LLM-ontology integration?
   - Existing frameworks to build on? (LangChain, LlamaIndex with ontology plugins?)
   - Development timeline realistic for thesis scope?

13. **Computational Resources:**
   - Ontology reasoning can be computationally expensive - what are limits?
   - Do we need specialized hardware/servers for ontology queries?
   - Latency constraints: How fast must responses be for user acceptability?

#### Ethical Considerations

14. **Ontology Bias & Representation:**
    - Every ontology encodes particular worldviews/biases - how to address this?
    - Is presenting multiple ontological perspectives (Western/Eastern) sufficient?
    - Risk: System appears objective while actually encoding specific ideology

15. **Transparency Requirements:**
    - Should users always know when ontology vs. LLM is answering?
    - How to communicate confidence levels in ontology-based answers?
    - What if ontology answer is factually correct but politically sensitive?

16. **IRB & User Studies:**
    - If testing with political/geopolitical content, any special IRB concerns?
    - How to handle potential emotional responses to political questions?

#### Publication Strategy

17. **Contribution Positioning:**
    - What's the novel contribution? (Novel architecture? New evaluation framework? Empirical findings?)
    - How does this differ from existing neurosymbolic AI work?
    - Connection to trustworthy AI literature vs. explainable AI vs. knowledge representation?

18. **Target Venues:**
    - Neurosymbolic AI workshops/conferences?
    - Knowledge representation conferences (KR, ISWC)?
    - NLP conferences with neurosymbolic track?
    - HCI if focusing on user trust and interaction?

#### Literature Review Priorities

19. **Immediate Reading Needed:**
    - Neurosymbolic AI survey papers - what's state of the art?
    - Ontology-based reasoning in NLP - existing approaches?
    - Trustworthy AI metrics and evaluation frameworks
    - Political/geopolitical ontologies - what exists, how were they built?

20. **Related Work Positioning:**
    - How does this differ from knowledge graphs + LLM (e.g., RAG on Wikidata)?
    - Comparison to other verification approaches (fact-checking systems, multi-model voting)?
    - Connection to symbolic AI revival in recent years?

---

### Revised Timeline (Post-Meeting)

**Phase 1: Foundation & Planning (Nov-Dec 2025)**
- Week 1-2: Deep dive into neurosymbolic AI and ontology-based reasoning literature
- Week 3-4: Survey existing ontologies (political, geopolitical, general knowledge)
- Week 5-6: Select specific ontology and finalize Approach A vs. B decision
- Week 7-8: Design detailed experimental protocol and evaluation framework
- **Deliverable:** Concrete research proposal with chosen ontology and approach

**Phase 2: System Development (Jan-Feb 2026)**
- Month 1: Build query translation layer (NL → ontology queries)
- Month 1: Implement routing logic (if Approach A) or claim extraction (if Approach B)
- Month 2: Develop response generation and formatting
- Month 2: Create test dataset with ground truth from ontology
- **Deliverable:** Working prototype system

**Phase 3: Baseline & Initial Evaluation (Mar 2026)**
- Week 1-2: Establish baseline comparisons (pure LLM, RAG, tool-augmented)
- Week 3-4: Run initial experiments on test dataset
- **Deliverable:** Preliminary results showing feasibility

**Phase 4: Refinement & Extended Evaluation (Apr-May 2026)**
- Month 1: Refine system based on initial results
- Month 1: Expand test coverage (more domains, edge cases)
- Month 2: User studies if applicable (trustworthiness ratings)
- Month 2: Cross-ontology testing (different perspectives)
- **Deliverable:** Comprehensive evaluation results

**Phase 5: Analysis & Write-up (Jun-Jul 2026)**
- Month 1: Data analysis, statistical testing, qualitative insights
- Month 2: Thesis writing (related work, methodology, results, discussion)
- **Deliverable:** Complete thesis draft

**Phase 6: Revision & Defense Prep (Aug 2026)**
- Incorporate advisor feedback
- Prepare defense presentation
- Final revisions
- **Deliverable:** Final thesis submission

---

### Resources Needed (Updated)

**Ontology Resources:**
- Access to existing ontologies (DBpedia, Wikidata, or domain-specific)
- Ontology editing tools if modifications needed (Protégé, OWL API)
- SPARQL endpoint hosting (local server or cloud-based)
- Ontology reasoning engine (Pellet, HermiT, or custom)

**Computational:**
- API access for baseline LLM comparisons (OpenAI, Anthropic, or open-source models)
- Cloud hosting for ontology query server
- Moderate compute for LLM fine-tuning if needed (routing classifier, claim extraction)

**Data & Evaluation:**
- Test dataset creation (domain experts for ground truth?)
- Human evaluation budget for trustworthiness ratings
- Annotation budget for claim verification tasks

**Technical Infrastructure:**
- Integration frameworks (LangChain, LlamaIndex, or custom)
- Query translation tools (NL → SPARQL)
- Evaluation frameworks for fact-checking, consistency measurement

Human:
- Advisor time for iterative feedback
- Potential collaboration with psychology/HCI experts for empathy evaluation
- Domain experts for reasoning task validation (Exp 1, 4)

---

### Contingency Plans

**If selected ontology proves inadequate:**
- Switch to alternative ontology in same domain
- Build minimal custom ontology for proof-of-concept
- Use knowledge graph (less formal structure) as fallback

**If integration complexity exceeds timeline:**
- Focus on Approach A (simpler query routing) and defer Approach B
- Reduce scope to single domain rather than cross-cultural comparison
- Demonstrate concept with wizard-of-oz study (manual ontology queries)

**If ontology reasoning is too slow:**
- Pre-compute common queries and cache results
- Use approximate reasoning instead of full logical inference
- Limit ontology complexity (smaller subset, simpler relations)

**If baseline comparisons are unfair:**
- Provide both systems with same knowledge scope (restricted LLM training data)
- Focus on qualitative differences (explainability, consistency) rather than pure accuracy
- Use multiple baselines to show trade-off space

---

## 4. Next Steps

### Before Next Meeting (Tuesday 10am):
- [x] Document meeting discussion and revise experimental plan ✓
- [ ] Begin literature review on neurosymbolic AI and ontology-based reasoning
- [ ] Survey existing ontologies (political, geopolitical, general knowledge)
- [ ] Draft concrete experimental protocol for Experiment 6 (Approach A and/or B)
- [ ] Identify 2-3 candidate ontologies with access information
- [ ] Prepare specific questions about ontology selection and approach

### During Next Meeting:
- [ ] Present refined Experiment 6 design with concrete implementation plan
- [ ] Discuss ontology candidates and get guidance on selection
- [ ] Decide between Approach A, Approach B, or both
- [ ] Clarify technical feasibility questions (tools, frameworks, timeline)
- [ ] Confirm evaluation methodology and success criteria
- [ ] Align on immediate next steps and timeline

### After Next Meeting:
- [ ] Finalize ontology selection and acquire/access it
- [ ] Set up development environment (ontology tools, query engines)
- [ ] Begin implementation of core components
- [ ] Create initial test dataset from ontology ground truth
- [ ] Establish baseline system for comparison

---

## Appendix A: Open Research Questions

### Core Questions from Meeting Discussion

1. **Why can't LLMs explain their reasoning?**
   - Transformers only see tokens as IDs, not meanings
   - Multiple parameters influence each output simultaneously
   - No single causal path from input to output
   - **Research Direction:** Can ontology-based verification provide the missing explanation layer?

2. **What is the difference between retrieval and reasoning?**
   - Current tools (MCP, databases) provide retrieval
   - Ontologies provide logical inference and consistency checking
   - **Research Question:** Does ontology-based inference constitute "real reasoning"?

3. **How to address the "sociopath yapper" problem?**
   - LLMs generate plausible post-hoc rationalizations
   - Ontology can ratify answers before presentation
   - **Research Question:** Does pre-verification reduce hallucination perception?

4. **Neurosymbolic AI vs. Hybrid Architectures:**
   - Three approaches: Brain-inspired hierarchical, neurosymbolic, specialized models
   - **Research Question:** Which architecture best balances reasoning capability and explanation quality?

### Previous Research Questions (Still Relevant)

5. **What is "true reasoning" philosophically and computationally?**
   - Connects to ontology-based logical inference
   - May need philosophy of mind and epistemology literature
   - Symbolic reasoning vs. statistical pattern matching

6. **How do different reasoning architectures compare on trustworthiness?**
   - Not just accuracy but explainability, verifiability, consistency
   - Multi-dimensional evaluation framework needed
   - Now includes: Pure LLM, RAG, Tool-augmented, Ontology-based

7. **Cross-cultural perspectives in knowledge representation:**
   - How do different ontologies encode worldviews?
   - Can system present multiple valid perspectives without bias?
   - Ethical implications of encoding particular political/cultural frameworks

8. **What's the relationship between explanation detail and user trust?**
   - Does ontology reasoning trace increase trust?
   - Or is there a cognitive load limit?
   - How technical can explanations be for lay users?

### Literature Gaps to Investigate

9. **Existing work on ontology + LLM integration:**
   - What's already been done in neurosymbolic AI community?
   - How does knowledge graph QA differ from ontology-based verification?
   - Where is the novel contribution space?

10. **Evaluation frameworks for trustworthy AI:**
    - What metrics exist beyond accuracy?
    - How to measure "trustworthiness" objectively?
    - Standards for explainability quality assessment?

---

## Appendix B: Meeting Action Items

**Ajinkya's Responsibilities:**
1. Outline potential experiment design for ontology-LLM integration
2. Share notes on research collection methodology
3. Deep dive into neurosymbolic AI literature
4. Prepare concrete implementation plan for Tuesday meeting

**Topics for Tuesday Meeting:**
1. Review concrete experimental proposal
2. Discuss ontology selection
3. Technical architecture decisions
4. Timeline and resource availability
5. Next immediate steps

**Key Takeaway from Meeting:**
> "Focus on practical, measurable components with rigorous experimental basis rather than purely theoretical concepts. Write down concrete ideas even if they might be rejected - learning from rejection helps explore alternatives."

---

**Document Status:** Updated post-meeting (November 10, 2025)  
**Next Update:** After Tuesday in-person meeting  
**Primary Focus:** Experiment 6 - Ontology-LLM Integration
