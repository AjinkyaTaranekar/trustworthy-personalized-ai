
### Central Research Question

> *How can we build a trustworthy Conversational LLM that is explainable, accurate, privacy-preserving, and capable of genuine empathy through user modeling and contextual understanding?*

### Secondary Research Question
> *What is true reasoning?*
 
> *I think I opened a Pandora's Box to Super Intelligence*
---
#### 1. What Makes Humans & AI Different?

> Fundamental Question: What truly distinguishes human intelligence from AI?

- Human Characteristics:
    - Understanding - Deep comprehension beyond pattern matching
    - Asking Right Questions - Proactive inquiry and curiosity
    - Genuine Interest - Intrinsic motivation to help
    - Discovery - Joy in learning and connecting ideas
    - Narrative Journey - Creating coherent stories from experiences
    - Sense Making - Finding meaning in chaos
    - Relevance Judgment - Understanding what matters in context
    - Empathy - Genuine emotional connection and understanding

- AI Current State:
    - Pattern Matching - Not true understanding
    - Reactive - Responds but doesn't proactively inquire meaningfully
    - Simulated Interest - No intrinsic motivation
    - Retrieval - Not true discovery
    - Token Generation - Not narrative consciousness
    - Statistical Correlation - Not meaning-making
    - Context-Limited - Struggles with nuanced relevance
    - Simulated Empathy - Post-hoc rationalization, not genuine feeling

- Critical Missing Elements in AI:
    - Consciousness & Qualia: 
        - Humans experience "what it's like" to understand something (phenomenal consciousness).
        - LLMs process information but have no subjective experience, no "feeling" of understanding.
        - Question: If an LLM perfectly simulates empathy, does the lack of qualia matter to the user?
    
    - Embodied Cognition:
        - Human understanding is grounded in physical experience (we know "hot" because we've felt heat).
        - LLMs are disembodied - they learn from text describing experiences, not from direct sensory input.
        - This is why they struggle with spatial reasoning, physical intuition, and common-sense physics.
        - Potential solution: Multimodal models (vision + language) partially address this, but still no true embodiment.
    
    - Causal Understanding vs Correlation:
        - Humans build causal models of the world ("rain makes the ground wet").
        - LLMs learn correlations ("rain" and "wet" appear together) without understanding causality.
        - This is why they can be easily fooled by adversarial examples or counterfactual scenarios.
        - Example: "If I hadn't gone to the store, would I still have milk?" requires causal reasoning, not pattern matching.
    
    - Meta-Cognition:
        - Humans can think about their own thinking ("I'm confused about X").
        - LLMs can generate text that looks like meta-cognition, but it's still just next-token prediction.
        - They can't genuinely assess their own uncertainty or knowledge gaps.
        - Note: This connects to the "Sociopath Yapper" problem - they can't honestly say "I don't know."
#### 2. Foundational Mechanics: How LLMs Actually Work

> *How does an LLM see words and numbers?*
> *Why does LLM need tokens in a vocabulary to be as minimum as possible, why not n-grams or sentences as tokens?*

- Tokenization (General): [[Efficient Estimation of Word Representations in Vector Space (1301.3781v3)]]
    - An LLM does not see words or characters, but "tokens". Or say IDs of the token.
    - The Vocabulary is the fixed dictionary of all unique tokens a model understands.
    - At its core, an LLM is a transformer-based statistical model that predicts the next token in a sequence.
    - GPT-2 architecture performed tokenization at character level, a 256 bits dataset.
    - Modern Tokenisation is done using a standard old algorithm of merging characters into sub-words based on number of occurences. -> BPE
    
- Subword Tokenization (BPE):  [[Neural Machine Translation of Rare Words with Subword Units (1508.07909v5)]]
    - Modern LLMs use subword tokenization, like Byte Pair Encoding (BPE), to manage vocabulary size.
    - This method breaks rare words into common sub-word pieces (e.g., "unhappiness" becomes `["un", "happi", "ness"]`).
    - Critical Limitation: This is the technical root of why LLMs fail at computation. A number like `183491` is not a single concept; it's tokenized into meaningless fragments (e.g., `["183", "491"]`), destroying its mathematical properties.
    
- Attention & Transformers: [[Attention Is All You Need (1706.03762v7)]]
    - The Transformer architecture is the foundation of modern LLMs.
    - Its core mechanism is self-attention, which allows the model to weigh the importance of different tokens in the input sequence when generating the next token.
    - Critical Limitation: The standard autoregressive decoding process is feed-forward. It generates one token at a time and cannot go back to "fix" a mistake made earlier in the sequence. This lack of stateful backtracking is a major hurdle for complex, multi-step reasoning.
    
- Contextual Understanding - BERT vs GPT: [[BERT Pre-training of Deep Bidirectional Transformers for Language Understanding (1810.04805v2)]]
    - Critical Distinction: Not all transformers are autoregressive!
    - BERT uses bidirectional attention - it can look at tokens both before AND after the current position.
    - GPT uses causal (left-to-right) attention - it can only look at previous tokens.
    - Why this matters:
        - BERT is better at understanding context (fill-in-the-blank tasks, classification).
        - GPT is better at generation (writing, conversation).
        - Question: Could a hybrid approach (BERT-like understanding + GPT-like generation) improve reasoning?
    
- Token Embeddings vs Contextualized Representations:
    - Static Embeddings (Word2Vec era): Each word gets one fixed vector.
        - Problem: "bank" means the same thing in "river bank" and "bank account".
    - Contextualized Embeddings (Transformer era): Each token gets a different vector based on surrounding context.
        - BERT/GPT compute these dynamically using attention across all layers.
        - This is why modern LLMs handle polysemy (multiple meanings) better.
    - Deep Insight: [[Measuring Word Significance using Distributed Representations of Words (1508.02297v1)]]
        - Not all tokens are equally important in a sequence.
        - Attention mechanism learns which tokens matter most for prediction.
        - Connection to explainability: Can we visualize attention weights to show "why" a model focused on certain words?
    
- RAG (Retrieval-Augmented Generation):
    - Grounds LLM responses in retrievable, verifiable sources.
    - Vector Stores like Milvus
    - Provides explicit citations, making the system more scrutable.
    
- Tool-Augmented LLMs:
    - This approach addresses the computation problem by delegating tasks. [[PAL Program-aided Language Models (2211.10435v2)]]
    - The LLM's role changes from "solver" to "reasoner." It determines what needs to be done, then calls an external tool (e.g., a Python interpreter) to perform the actual calculation.
    - This is a more scrutinizable and honest form of explainability. The LLM can truthfully state, "I used a calculator to get this number".
    - Example: Qwen 2.5 Math Model thinks in terms of Python code to solve complex puzzles.
    
- MCP Servers (Model Context Protocol): [[Advancing Multi-Agent Systems Through Model Context Protocol Architecture, Implementation, and Applications (2504.21030v1)]]
    - A standardized protocol for connecting LLMs to external data sources and tools.
	- Think of it as "USB for AI" - a universal interface for tools, databases, APIs.
	- Developed by Anthropic as an open standard.
	- Scrutability: Each tool call is explicit and logged. User can see "LLM called Calculator with input X".
	- Privacy: Sensitive data can stay in local MCP servers, never sent to cloud LLMs.
	- Accuracy: Delegate tasks to specialized tools rather than relying on LLM's flawed internal knowledge.    
    - MCP Architecture:
        - MCP Client: The LLM application (e.g., Claude, custom chatbot).
        - MCP Server: Exposes tools/resources to the client (e.g., file system, database, calculator).
        - Transport Layer: How they communicate (stdio, HTTP, WebSocket).
    - Critical Advantage Over Function Calling:
        - Function calling is model-specific (OpenAI, Anthropic have different formats).
        - MCP is a universal standard - write once, use with any MCP-compatible model.
        - Enables modular, composable AI systems (swap out reasoning engines without changing tools).
#### 3. Why Current Architecture Fails?

##### 3.1 Reasoning is Not Trustworthy

> Core Problem: The "Strawberry issue" - LLMs struggle with simple tasks that require actual reasoning rather than pattern matching.

- The "Sociopath Yapper" Problem:
    
    - This is the critical difference between introspection (a human reporting on their actual thought process) and post-hoc rationalization (an LLM generating a new, plausible-sounding story about why it gave an answer).
    - An LLM's explanation is a "sycophant yapper". It's not showing its work; it's generating a new essay that looks like an explanation, which is fundamentally untrustworthy. [[Explainable Sentiment Analysis with DeepSeek-R1 Performance, Efficiency, and Few-Shot Learning (2503.11655v4)]]
    - Metaphor: Current LLMs = "Sociopath Yappers" - highly fluent and convincing but lack true understanding, empathy, or honest introspection.
    
    - Why do humans respond to someone's question?
        - Building relationships?
        - Some ulterior motive?
        - Is it for some kind of reward?
        - Family or close friends connection
        - Respect
        - Karma / social reciprocity
        
        > Insight: Can we use the "yapping" of LLMs to predict what question the user might ask next? This could enable proactive, empathetic responses.
        
    - Reinforcement Learning for Thought Processes:
        - Discussion with professor: Can an LLM know that giving this answer will reward it?
        - Key idea: Instead of rewarding final answers, reward the correct thought process itself. [[Seed1.5-Thinking Advancing Superb Reasoning Models with Reinforcement Learning (2504.13914v3)]]
        - Challenge: Traditional RL rewards are based on outcomes, but data and correct answers are ever-changing.
        - Proposed direction: What if LLM RL can be done only for the thought process, not specific answers?
        - This could help LLMs learn "how to think" rather than "what to answer". [[Self-Enhanced Reasoning Training Activating Latent Reasoning in Small Models for Enhanced Reasoning Distillation (2502.12744v1)]]
        
        - Seed1.5-Thinking Model Insights:
            - Uses RL to train reasoning capabilities, not just final answers.
            - Reward signal based on:
                - Process correctness: Are the intermediate steps logically sound?
                - Efficiency: Did it reach the answer via the shortest valid path?
                - Verifiability: Can each step be checked independently?
            - This is closer to "teaching how to fish" vs "giving the fish."
        
        - Critical Advantage Over Outcome-Only RL:
            - Outcome-only RL: Model learns to game the reward (might guess or use shortcuts).
            - Process-based RL: Model must demonstrate understanding at each step.
            - More generalizable: Good reasoning process works on novel problems.
        
        - Connection to AbsoluteZero paper: Can we fix the thought process framework and dynamically pour in data?
        - Philosophical angle: Isn't everything situation + reaction? We need to fix the thought process and pour in data.
        - Open question: Without training on specific outcomes, how will LLMs connect the dots? Does it need both process rewards and outcome feedback?
            - Likely Answer: Need both. Process rewards for "how to think," outcome rewards for "is this useful?"
            - Analogy: Like teaching a student - praise good reasoning even if answer is wrong (process), but also correct final answer (outcome).
    
    - Emergence vs Programmed Capabilities:
        - Critical debate: Are LLM capabilities "emergent" (spontaneously arising from scale) or "programmed" (present in training data)?
        - Example: GPT-3 could do basic arithmetic. GPT-4 is much better. Is this emergence or just more arithmetic examples in training?
        - [[Language Models are Hidden Reasoners Unlocking Latent Reasoning Capabilities via Self-Rewarding (2411.04282v2)]]
        - Research Question: If reasoning is emergent, can we "unlock" it without task-specific training?
        - Counter-argument: Apple's research shows LLMs fail when problems are slightly rephrased, suggesting memorization, not true reasoning.
        - Implication: We can't rely on "more data, bigger models" alone. Need architectural changes (ToT, HRM, neuro-symbolic).
    
    - Catastrophic Forgetting:
        - When LLMs are fine-tuned on new data, they often "forget" previous knowledge.
        - This is a fundamental problem for continual learning and user personalization.
        - Example: Fine-tune on User A's preferences → loses general knowledge or User B's preferences.
        - Potential solutions:
            - Separate user models from base model (retrieval-based personalization).
            - Elastic Weight Consolidation (EWC) - protect important parameters from changing.
            - Modular architecture - user-specific adapters + frozen base model.
        - Connection to research: If we want lifelong learning AI that adapts to each user, we MUST solve this.
        
- Computation vs. Pattern Matching:
    
    - LLMs fail at math because they are pattern-matching engines, not calculators.
    - They can solve `24 + 45 = 69` because that exact sequence of tokens likely appeared thousands of times in the training data.
    - They fail at `183491 + 923456` because (a) the numbers are tokenized into fragments and (b) that specific problem has never been seen before, so there is no pattern to retrieve.
        
##### 3.2 User Personalization & Contextual Understanding

> Core Problem: LLMs lack true user modeling and contextual awareness. They assume rather than inquire.

- The Late-to-Class Example:
    - User: "I reached late to class on the first day and got scolded. What to do to reach early?"
    - Existing LLM response: Throws away generic advice on "how to not get late" without asking WHY you got late. Assumes you might not have packed your bag or woke up late.
    - Ideal Response: "Oh! How did it happen? Why did you get late?"
    - Principle: LLMs should be listeners, not interrogators.

- The Frappuccino Scenario:
    - Let's say an LLM is trained on Open Internet Forums where people talked about having a Frappuccino on a "sunny day."
    - What about temperature? Location?
        - A "sunny" day feels different at 20°C in Dublin versus 35°C in Delhi.
        - Why recommend only this coffee for the same feeling with vastly different temperatures?
    - Missing Context: The LLM doesn't understand the nuanced relationship between weather, location, culture, and personal preferences.

- The Cold Start Problem & Solutions:
    - Problem: How do you personalize when you have zero user history?
    - Current Approaches:
        - Ask explicit questions upfront (user onboarding survey).
        - Use demographic proxies (location, age) - but this risks stereotyping.
        - Leverage metadata (time of day, device, previous query topics).
    - Proposed Solution for This Research:
        - Start with the 5W+H framework - systematically gather context in first conversation.
        - Don't ask "What do you want?" Ask "What are you trying to achieve? Why?"
        - Build initial mental model from this structured inquiry.
        - Use few-shot learning: "Users like you (similar context) typically prefer X."
    
    - Privacy Paradox:
        - Users want personalization but fear surveillance.
        - Question: Can we build user models that are private by design?
        - Potential approaches:
            - Local-only storage: User model never leaves device (MCP server on localhost).
            - Federated learning: Learn patterns across users without centralizing data.
            - Differential privacy: Add noise to aggregated data to protect individuals.
        - Research direction: How much personalization can we achieve with privacy guarantees?

- Technical Questions to Explore:
    - Can SHAP be used on LLMs? How will we access the internal layers?
        - Short answer: Very difficult. SHAP assumes feature independence; tokens in LLMs are highly dependent.
        - Alternative: Use attention weight visualization or gradient-based attribution methods.
    - What are sociopaths in psychological terms? → Manipulation, deceit, lacks empathy. (Parallel to current LLM behavior)
    - Can a "System Prompt" fix this? Or is there more architectural work needed?
        - System prompts can guide behavior but can't give the LLM genuine understanding or memory.
        - Need architectural changes: persistent user models, explicit reasoning modules, tool integration.

##### 3.3 Lack of Explainability

> Q: Can an LLM scrutinize itself?

- How LLMs Work: Transformer Layer → Predicts next token based on statistical patterns.
- The Problem: We can't tell the user how each specific token is predicted. It's a black box of weighted parameters.
- Questions:
    - Is it based on weighted parameters? If so, how? Which models contributed?
    - How did they generalize from training data to this specific output?

- Q: What are the other ways to achieve scrutability then?
    - State the sources (like RAG systems do).
    - User preference tracking - "You mentioned you prefer X."
    - Use interpretable models - Like Decision Trees from classical ML for certain subtasks.
    
- Mechanistic Interpretability:
    - A new field focused on reverse-engineering what neural networks learn.
    - Goal: Identify specific "circuits" (groups of neurons/attention heads) that perform specific tasks.
    - Example findings:
        - Certain attention heads in GPT-2 specifically detect "indirect object identification."
        - Some neurons activate for specific concepts (the "Golden Gate Bridge neuron").
    - Relevance to research:
        - If we can identify empathy-related circuits, can we strengthen them?
        - Can we detect when a model is hallucinating by observing internal activations?
    - Challenge: Most interpretability research is on small models. Do findings generalize to 70B+ parameter models?
    
- Attention Visualization & Limitations:
    - Popular method: Show which input tokens the model "paid attention to" when generating an output.
    - Problem: Attention is NOT explanation!
        - High attention doesn't mean causation (token X could have high attention but not influence the output).
        - Attention patterns are just one of many factors (residual connections, MLP layers also matter).
    - Research finding: Adversarial attacks can manipulate attention without changing outputs.
    - Conclusion: Attention visualization is useful for intuition but not rigorous explainability.
    
- Model Calibration & Uncertainty:
    - A well-calibrated model's confidence matches its accuracy.
        - If it says 80% confident, it should be correct 80% of the time.
    - Problem: LLMs are often overconfident (say "I'm certain" even when wrong).
    - Why this matters for trustworthy AI:
        - Users need to know when the AI is uncertain.
        - A trustworthy system says "I don't know" instead of hallucinating.
    - Potential solutions:
        - Train models to output calibrated probabilities.
        - Use ensembles (multiple model predictions) to estimate uncertainty.
        - Teach models to say "Let me check" and query tools/RAG when uncertain.
#### 4. Reasoning in LLMs: Current State & Limitations

##### 4.1 What is Reasoning, Actually?

> Q: What is reasoning in LLM?
> Q: So how do you reason a reasoning? What is reasoning actually? How does a human reason?

- Reasoning breakdown using First Principles:
    - Human Reasoning: The process of making connections between different pieces of information (logical and emotional) to draw conclusions, often involving systematic information gathering.
        - Nature vs. Nurture: Some people are skeptical or in "detective mode"; some people reason based on past experiences or emotions.
        - Involves systematic information gathering, understanding context, and drawing inferences.
        - Humans can backtrack, question assumptions, and revise their thinking mid-process.
    
    - LLM "Reasoning": Currently, this is not a conscious, logical process. It is an emergent capability derived from statistical pattern matching. The model predicts the next token in a sequence that looks like a logical argument, based on the vast amount of human-generated text it was trained on.
        - An LLM predicts the next possible token/concept based on patterns seen in training data.
        - Currently, reasoning is just expansion and self-realization/reinforcement to get more accurate steps, based on data fed with steps (generally for Maths, Physics, or core subjects).
        - Critical Problem: If given a puzzle, it fails when there are changes (e.g., Apple's Illusion Thinking paper, Strawberry counting problem). [[None of the Others a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks (2502.12896v5)]]
        
- Chain of Thought (CoT): [[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2201.11903v6)]]
    
    - A prompting technique that forces the LLM to "think step-by-step" by generating its intermediate reasoning as natural language text.
    - Why it works: It is not necessarily "reasoning" better. It works by expanding the available "computational scratchpad". By generating more tokens, the model has more steps to attend to its own previous outputs, guiding it along a more accurate statistical path.
    
    - System 1 vs System 2 Thinking (Kahneman's Framework):
        - System 1: Fast, automatic, intuitive (pattern recognition).
        - System 2: Slow, deliberate, logical (analytical reasoning).
        - Parallel in LLMs:
            - Direct answers (no CoT) = System 1: Quick pattern matching from training data.
            - CoT/ToT = System 2: Deliberate step-by-step processing.
        - Critical insight: CoT doesn't give LLMs true System 2 reasoning, but it approximates it by slowing down generation and adding intermediate steps.
        - Research question: Can we architect LLMs to explicitly have dual-process cognition (fast path + slow path)?
    
    - Few-Shot Learning & In-Context Learning:
        - LLMs can learn new tasks from just a few examples in the prompt (no parameter updates!).
        - Example: Show 3 math problems with solutions → model solves 4th problem.
        - Why this is profound:
            - Suggests LLMs build internal "task representations" dynamically.
            - They're not just memorizing; they're identifying patterns in the prompt itself.
        - Connection to CoT: Few-shot CoT (showing examples with reasoning steps) works better than zero-shot.
        - Limitation: In-context learning has a token limit (context window). Can't learn from 10,000 examples.
        - Research direction: Can we combine in-context learning with persistent user models?
    
- Deepseek R1: [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.12948v1)]]
	- Demonstrates that "too much CoT" (extended chain-of-thought generation) achieved better output quality.
	- Shows the trade-off: more tokens = more reasoning steps = better accuracy, but slower response times.
	
- Tree of Thoughts (ToT): [[Tree of Thoughts Deliberate Problem Solving with Large Language Models (2305.10601v2)]]
    - An advanced architecture that addresses the key flaw of CoT's linear, "no-backtracking" nature.
    - It allows the model to explore multiple (e.g., 3-5) possible reasoning paths (branches) at each step.
    - It then uses the LLM itself to evaluate which branches are most promising, pruning bad paths and pursuing good ones. This is a truer, more hierarchical form of problem-solving.
    - Connection to Proposed Work: This aligns with the idea of using mental models and 5W+H framework to break down tasks systematically.
    
- ReAct (Reasoning + Acting): [[ReAct Synergizing Reasoning and Acting in Language Models (2210.03629v3)]]
    - The Framework: Interleave reasoning steps with action steps.
    - Example:
        - Thought: "I need to find the current weather in Paris."
        - Action: Call weather API for Paris.
        - Observation: Temperature is 15°C, cloudy.
        - Thought: "User asked about outdoor activities, cold weather suggests indoor alternatives."
        - Action: Search for indoor activities in Paris.
    - Why This is Powerful:
        - Grounds reasoning in real-world actions (like humans do).
        - Enables iterative refinement (observe result → adjust reasoning → new action).
        - More scrutable: Users see both thoughts AND actions.
    - Connection to Proposed Work:
        - Perfect framework for tool-augmented empathetic AI.
        - Can reason about user emotions → query user model → adjust empathetic response.
        - Aligns with MCP architecture (actions = MCP server calls).

- Self-Enhanced Reasoning: [[Self-Enhanced Reasoning Training Activating Latent Reasoning in Small Models for Enhanced Reasoning Distillation (2502.12744v1)]]
    - The Idea: Small models can learn reasoning from their own outputs (self-distillation).
    - Process:
        1. Model generates multiple reasoning paths for a problem.
        2. Evaluate which paths lead to correct answers.
        3. Fine-tune on the successful reasoning paths.
    - Advantage: Don't need large teacher models or human-annotated reasoning.
    - Challenge: Risk of reinforcing biases if initial model is systematically wrong.
    - Research Question: Can this be combined with user feedback?
        - User corrects reasoning → system learns from correction → improves for future.

- Template-Based Fine Tuning: [[Automatic Chain of Thought Prompting in Large Language Models (2210.03493v1)]]
    - Training LLMs on structured reasoning templates to improve step-by-step problem decomposition.
    - Key Insight: Don't need human-written CoT for every problem.
    - Auto-CoT Approach:
        - Cluster problems by similarity.
        - For each cluster, automatically generate CoT examples.
        - Use diverse examples to cover different reasoning patterns.
    - Relevance: Could automate creation of 5W+H reasoning templates for different conversation types.
    
- Reinforcement Learning in LLM: [[Language Models are Hidden Reasoners Unlocking Latent Reasoning Capabilities via Self-Rewarding (2411.04282v2)]]
    - Key Question: Can LLM know that giving this answer will reward it?
    - Challenge: Data is ever-changing. What if LLM RL can be done only for thought process, not specific answers?
    - Insight from AbsoluteZero paper: Can we fix the thought process and pour in data dynamically?
    - Philosophical Angle: Isn't everything situation + reaction? We need to just fix the thought process and pour in data.
    
- Hierarchical Reasoning Model (HRM): [[Hierarchical Reasoning Model (2506.21734v3)]]
    
    - This is a specialist reasoning engine, not a general-purpose LLM.
    - Its architecture (inspired by high/low frequencies in the brain) is designed for deep, recurrent, and stateful computation, allowing it to solve complex logic puzzles (like Sudoku) where LLMs fail.
    - Limitation: It is not a text generator. It reasons in its own internal vector space (latent space) and is not inherently explainable in natural language.
        
- CoCoNut (Chain of Continuous Thought): [[Training Large Language Models to Reason in a Continuous Latent Space (2412.06769v3)]]
    - Addresses the inefficiency of token-heavy CoT by using latent vectors for reasoning.
    - Key Innovation: Instead of reasoning in discrete natural language tokens, reason in continuous vector space.
    - Advantages:
        - More efficient (fewer tokens needed for complex reasoning).
        - Potentially more powerful (vectors can represent nuanced, non-linguistic concepts).
        - Less "yapping" (no need to verbalize every micro-step).
    - Challenge: How do we make latent reasoning explainable?
        - If reasoning happens in vector space, users can't see the process.
        - Potential solution: Train a "translator" model to convert latent steps to natural language explanations post-hoc.
    - Connection to HRM: Both use internal representations for reasoning, but HRM is fully latent while CoCoNut is hybrid (latent reasoning + text generation).
    
- Diffusion-Based Reasoning: [[LaDiR Latent Diffusion Enhances LLMs for Text Reasoning (2510.04573v3)]] & [[Diffusion of Thoughts Chain-of-Thought Reasoning in Diffusion Language Models (2402.07754v3)]]
    - Completely Different Paradigm: Reasoning as iterative refinement, not sequential generation.
    - How it works:
        - Start with a random/noisy "reasoning plan."
        - Iteratively denoise/refine it until you reach a coherent solution.
        - Think of it like solving a puzzle by gradually filling in pieces, not writing left-to-right.
    - Why this matters:
        - Autoregressive models can't go back and fix mistakes.
        - Diffusion models can globally optimize the entire reasoning path.
        - They can explore multiple solutions simultaneously (similar to ToT but more fluid).
    - Challenge: Much slower than autoregressive generation (needs many denoising steps).
    - Research potential: Could diffusion be used for the "Reasoner" module in a hybrid system?
        - User asks question → Diffusion model generates optimal reasoning plan → Autoregressive model executes it and generates text.
	    
- Neuro Symbolic AI:
    - The Core Idea: Combine neural networks (sub-symbolic, pattern-based) with symbolic reasoning (logic, rules, knowledge graphs).
    - Why We Need This:
        - Neural networks (LLMs) are great at pattern matching, handling ambiguity, learning from data.
        - Symbolic systems are great at logical reasoning, guarantees, explainability.
        - Neither alone is sufficient for trustworthy reasoning.
    
    - Key Approaches:
        - Knowledge Graph Integration:
            - Store factual knowledge in explicit graphs (entities, relations).
            - LLM queries the graph for verified facts rather than hallucinating.
            - Example: "Who is the CEO of Microsoft?" → Query knowledge graph, not rely on training data.
        
        - Rule-Based Constraints:
            - Define hard logical rules the system must follow.
            - Example: "If patient is allergic to X, never recommend medication containing X."
            - LLM generates candidate responses, symbolic checker filters invalid ones.
        
        - Program Synthesis:
            - LLM generates executable code (Python, SQL) to perform reasoning.
            - Code runs in a sandboxed environment with deterministic output.
            - This is what PAL does - LLM becomes the programmer, not the calculator.
    
    - Connection to Appraisal Theory:
        - Emotions can be modeled as a symbolic ontology (21 appraisal dimensions).
        - Neural network detects emotional cues from text.
        - Symbolic reasoner maps detected appraisals to appropriate empathetic responses.
        - This hybrid approach gives both flexibility (handling natural language) and transparency (explicit emotion model).
    
    - Challenge - The Symbol Grounding Problem:
        - How do symbols in a formal system connect to real-world meanings?
        - A knowledge graph says "Paris is the capital of France," but does the LLM truly understand "capital" or just pattern-match the words?
        - Potential solution: Multimodal grounding (images, videos) + interactive learning (user feedback).
    
    - Research Direction:
        - Can we build a neuro-symbolic user model?
        - Neural: Learns patterns from user interactions.
        - Symbolic: Stores explicit preferences, rules, constraints.
        - Hybrid: Generates personalized, explainable responses.
        
- Dual-Head Reasoning: [[Dual-Head Reasoning Distillation Improving Classifier Accuracy with Train-Time-Only Reasoning (2509.21487v2)]]
    - The Concept: Train a model with two "heads" - one for reasoning, one for answering.
    - During training:
        - Reasoning head generates explanations.
        - Answer head generates final response.
        - Both are trained, but reasoning head guides learning.
    - During inference:
        - Only use the answer head (no reasoning overhead!).
        - The reasoning capability has been "distilled" into the answer head.
    - Advantages:
        - Fast inference (no CoT generation at runtime).
        - Still benefits from reasoning during training.
    - Challenge: Does the answer head truly internalize reasoning, or just memorize patterns from the reasoning head?
    - Research direction: Can we combine this with interleaved thinking for best of both worlds?

- Interleaved Thinking: [[Reasoning with Latent Thoughts On the Power of Looped Transformers (2502.17416v1)]] & [[Interleaved Reasoning for Large Language Models via Reinforcement Learning (2505.19640v1)]]
    
    - A novel approach where reasoning and answer generation happen in parallel, alternating between "thinking tokens" and "answer tokens" rather than separating them into distinct phases.
    - MiniMax M2 Model:
        - Implements interleaved thinking by mixing reasoning steps directly with answer generation.
        - The model can think, provide partial answers, think more, and refine—creating a more natural, human-like problem-solving flow.
        - Advantages: More efficient token usage than pure CoT, faster time-to-first-token, and ability to provide incremental answers while still reasoning.
        - Challenge: Requires careful training to balance when to think vs. when to answer, avoiding either over-reasoning (wasting tokens) or under-reasoning (reducing accuracy).
    - Kimi 2 Thinking:
        - Demonstrates interleaved reasoning where the model continuously weaves between analytical thinking and response formulation.
        - Uses dynamic allocation of compute: spends more reasoning tokens on complex parts and fewer on straightforward sections.
        - Key Innovation: The model learns to identify which parts of a problem require deep reasoning and which can be answered directly from pattern matching.
        - Connection to Human Cognition: Mimics how humans think - we don't fully formulate our entire thought process before speaking; we think, speak, adjust, and continue iteratively.
    - Connection to Proposed Work:
        - Aligns with the hierarchical reasoning model where different cognitive "frequencies" (fast pattern matching vs slow deliberate reasoning) can operate in parallel.
        - Could be integrated with the proposed 5W+H framework: the model could answer "What" quickly while still reasoning about "Why" and "How."
        - For empathetic AI: Could provide immediate acknowledgment ("I hear you") while still processing deeper emotional context.

- Prompting Science & Diminishing Returns: [[Prompting Science Report 2 The Decreasing Value of Chain of Thought in Prompting (2506.07142v1)]]
    - Critical Finding: As models get larger and more capable, CoT provides less benefit.
    - Why this matters:
        - Suggests advanced models may internalize reasoning patterns.
        - OR: They're just better at pattern-matching without needing explicit steps.
    - Implication for Research:
        - Can't rely solely on prompting for trustworthy reasoning.
        - Need architectural solutions (ToT, HRM, diffusion) for guaranteed improvements.
    - Counter-evidence: DeepSeek-R1 still benefits massively from extended CoT.
        - Suggests the "sweet spot" of model size vs reasoning approach is still unclear.
    
    - Open Question: 
        - Is there a theoretical limit to how much reasoning can be extracted via prompting?
        - Do we need fundamentally different architectures for true reasoning?

#### 5. Core Research Themes

##### 5.1 Scrutability & Explainability (XAI)
    
- Definition: The quality of a system being open to scrutiny or able to be understood. The act of carefully examining something, especially in a critical way.

- Problem: How can an LLM be scrutinized if its "reasoning" is just statistics from Transformer layers?

- Solution Path: Scrutability is not achieved by asking the LLM to explain itself (post-hoc rationalization). It is achieved through:
    
    1. Stating sources (RAG) - "I found this information from [source X]."
    2. Honest tool-use reporting (PAL) - "I used a Python calculator to compute this."
    3. Translating the internal states of a reasoning module (HRM) into an explanation.

##### 5.2 User Modeling & Empathy

> Primary Question: Can an LLM also have empathy?

- Problem: Current LLMs are "sociopath yappers" that give generic, non-empathetic advice. They don't listen; they assume.
- Ideal AI Characteristics: Should act as a "good listener". It must adapt to the user during the conversation.
    - Be a sense maker
    - Show relevance to the challenge the user is facing
    - Engage in a narrative journey with the user
    - Show genuine interest and facilitate discovery

- Proposed Method: Instead of "assuming," the AI should ask why ("Oh! How did it happen? Why did you get late?").

- Core Concept of User Modeling:
    - User modeling is not just knowing the user but adapting to the user during the conversation.
    - Example: If a user is speaking fast in a voice conversation, the model could adaptively ask them to slow down.

- Empathetic AI Requirements:
    - Mental Models of the user
    - Good Listener capability
    - Adaptively Ask Right Questions
        - For each dialogue, apply 5W+H framework
    - User Models
        - "Ask why behind it" - probe deeper into motivations
    - Dynamic Updating - learn and adapt in real-time
    - Plans & Persuasion - help users achieve their goals

    Side Note: Ask simple questions (Yes/No) instead of open-ended ones when clarity is needed.

- Conversation Design & Pragmatics:
    - Turn-Taking: In human conversation, there are implicit rules for when to speak and when to listen.
        - Current LLMs always wait for user input (passive).
        - Proposed: AI should know when to interject with clarifying questions.
    - Grounding: Establishing mutual understanding in conversation.
        - Humans use "uh-huh," "I see," "wait, you mean X?" to ground understanding.
        - LLMs rarely do this unless explicitly prompted.
        - Research direction: Can we train models to actively ground understanding?
    - Conversational Implicature (Grice's Maxims):
        - Humans infer meaning beyond literal words.
        - Example: "Can you pass the salt?" is not a yes/no question about ability.
        - LLMs sometimes fail at implicature, taking things too literally.
        - Connection to empathy: Understanding subtext is crucial for emotional support.

##### 5.3 Appraisal Theory & Emotion Detection

> Primary Question: Can LLM, given how a user is feeling (appraisals) and event descriptions, generate an empathetic response?

- Appraisal Theory Foundation:
    - For any event, the emotion an experiencer feels is mostly because of sub-events happening.
    - The model was trained on the Crowd-event dataset, which contains 6600 events.
    - Appraisal changes based on context, and this affects the outcome of the next event.

- Key Capabilities:
    - Detect Appraisal points from user conversation.
    - Act as a good listener / gossip companion.
    - Help connect the dots between events and emotions.
    - Using knowledge base graphs, personalize responses for the user.

- Critical Questions to Investigate:
    - How were the 21 appraisals set for each event?
    - Does the experiencer know and understand all 21 appraisal words clearly?
    - What was the user's mood before the event?
    - What are the demographics of the dataset?
    - How much time passed after the event when it got recorded?
        - What if someone else consoled them, which changed their thought process later?
    
    - Appraisal Dimensions (Examples from common frameworks):
        - Novelty: Is this event new/unexpected?
        - Valence: Is it pleasant or unpleasant?
        - Goal conduciveness: Does it help or hinder my goals?
        - Coping potential: Can I handle this?
        - Agency: Who caused this? (Self, others, circumstances)
        - Fairness: Is this just or unjust?
    
    - Cultural Variation in Appraisal:
        - Critical oversight: Emotion appraisal varies across cultures.
        - Example: In some cultures, expressing anger at injustice is appropriate; in others, it's shameful.
        - The Crowd-event dataset - what cultures are represented?
        - Research challenge: Can we build culturally-aware appraisal models?
    
    - Temporal Dynamics of Emotion:
        - Emotions change over time (anger → acceptance, grief → healing).
        - A static appraisal misses this trajectory.
        - Proposed: Track appraisal changes across multiple conversations.
        - This connects to user modeling - the AI should remember how you felt yesterday and ask "How are you feeling about that situation now?"

- Testing Method:
    - Zero-shot testing on new conversations
    - Need to add: Few-shot testing with user-specific examples.
    - Need to add: Longitudinal evaluation (does empathy improve over multiple sessions?).
    - Need to add: Cross-cultural validation (does it work for diverse user populations?).
 ---
### Experiments

#### Proposed System Architecture:
- We will experiment with interleaved thinking, CoCoNut, and tool-augmented approaches (MCP) together, along with Appraisal Theory, combining it with 5W+H concept and Mental Models.
- Core Transformation: From "sociopath yapper" to "trustworthy listener" who:
    - Can reason based on intermediate steps (transparent process).
    - Provides proof of work for each step (citations, tool calls, reasoning traces).
    - Allows users to inspect, correct, and provide feedback on the reasoning process.
    - Learns from corrections and adds that knowledge back to persistent memory.

---

### Evaluation Framework & Metrics

> Critical Question: How do we measure "trustworthiness" and "empathy" in AI?

---

### Open Research Questions

1. Can we quantify "genuine" vs "simulated" empathy in a way that matters to users?
   - If users can't tell the difference, does the philosophical distinction matter?

2. What is the minimum viable user model?
   - How much data is needed before personalization outweforms generic responses?

3. Can reasoning be learned end-to-end or must it be architected?
   - Will sufficiently large models spontaneously develop reasoning, or do we need explicit modules?

4. How do we balance proactive inquiry with not being annoying?
   - When should the AI ask questions vs just answer?

5. Can we detect when an LLM is hallucinating in real-time?
   - Are there internal activation patterns that signal unreliability?

6. What is the role of embodiment in empathy?
   - Do multimodal models (voice, video) convey empathy better than text?

7. Can we build user models that transfer across platforms?
   - Portable user model standard (like MCP but for user data)?

8. How do we evaluate long-term impact?
   - Does repeated use of empathetic AI improve or harm users' wellbeing?

---

### Glossary of Key Terms

| Term / Concept                       | Definition                                                                                                                                                                                                                      |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Appraisal Theory                     | The idea that emotion is determined by an individual's "appraisal" (evaluation) of sub-events. This can be a model for detecting user empathy points. Core to understanding how users emotionally respond to conversational AI. |
| 5W+H Framework                       | Who, What, When, Where, Why + How. A systematic questioning framework for breaking down complex problems and ensuring comprehensive understanding.                                                                              |
| Autoregressive                       | A model that generates output one piece at a time, where each new piece is conditioned on all the previous pieces (e.g., standard LLM decoding).                                                                                |
| BPE (Byte Pair Encoding)             | A common subword tokenization algorithm. The foundational paper that explains why LLMs break up words and numbers.                                                                                                              |
| Chain of Thought (CoT)               | A prompting method that forces an LLM to output its intermediate steps as natural language text to improve reasoning accuracy.                                                                                                  |
| CoCoNut                              | Chain of Continuous Thought - A model that performs reasoning in latent space rather than natural language, improving efficiency and coherence.                                                                                 |
| Cold Start Problem                   | The challenge of providing personalized recommendations or responses when there's no prior user history. Solved with metadata and initial context gathering.                                                                    |
| Dual-Process Theory                  | Kahneman's framework: System 1 (fast, intuitive) vs System 2 (slow, deliberate). LLMs approximate System 2 through CoT/ToT but lack true dual-process architecture.                                                            |
| Embodied Cognition                   | The theory that human intelligence is grounded in physical, sensory experience. LLMs lack embodiment, limiting their understanding of physical concepts and common sense.                                                      |
| Few-Shot Learning                    | The ability to learn new tasks from just a few examples (2-10), without parameter updates. LLMs do this through in-context learning.                                                                                           |
| Hallucination                        | When an LLM generates plausible-sounding but factually incorrect information. A critical trustworthiness issue.                                                                                                                |
| HRM (Hierarchical Reasoning Model)   | A specialized, recurrent neural architecture (not a Transformer) designed for deep, stateful reasoning, outperforming LLMs on logic tasks.                                                                                      |
| In-Context Learning                  | LLM's ability to learn from examples provided in the prompt itself, without changing model weights. Enables rapid task adaptation within token limits.                                                                         |
| Mechanistic Interpretability         | Research field focused on reverse-engineering neural networks to find specific "circuits" that implement particular capabilities. Critical for deep explainability.                                                            |
| MCP (Model Context Protocol)         | Anthropic's open standard for connecting LLMs to external tools and data sources. "USB for AI" - enables modular, composable AI systems with explicit tool-use transparency.                                                   |
| Hybrid / Modular AI                  | An architecture combining specialized modules (e.g., a "Reasoner" and a "Generator") rather than a single monolithic model.                                                                                                     |
| Latent Reasoning                     | The act of performing reasoning steps in the model's internal vector space (latent space) rather than in external, natural language text.                                                                                       |
| Latent Space                         | The high-dimensional vector space where a model represents the meaning and relationships of concepts (tokens, words, images).                                                                                                   |
| Mental Models                        | Internal representations of how users think, behave, and make decisions. Critical for empathetic AI to understand and predict user needs.                                                                                       |
| PAL (Program-aided LM)               | A method where an LLM generates code (e.g., Python) as its reasoning step and delegates the execution to an interpreter for perfect accuracy.                                                                                   |
| Model Calibration                    | The degree to which a model's confidence matches its accuracy. Well-calibrated models say "I'm 80% sure" and are actually correct 80% of the time. LLMs are often poorly calibrated.                                           |
| Neuro-Symbolic AI                    | Hybrid approach combining neural networks (pattern matching, learning) with symbolic systems (logic, rules, knowledge graphs). Aims for both flexibility and explainability.                                                   |
| Post-Hoc Rationalization             | The act of generating a plausible-sounding explanation for a decision after the decision has been made, without access to the actual causal reasons.                                                                            |
| Qualia                               | The subjective, phenomenal aspect of experience ("what it's like" to see red, feel pain). LLMs lack qualia - they process information but have no subjective experience.                                                       |
| RAG (Retrieval-Augmented Generation) | A method that grounds LLM responses in retrievable, verifiable sources, providing citations and improving trustworthiness.                                                                                                      |
| Scrutability                         | A core research theme: The quality of a system being open to examination and understood by its users.                                                                                                                           |
| Symbol Grounding Problem             | The philosophical challenge of how abstract symbols in a formal system connect to real-world meanings. Critical issue for neuro-symbolic AI.                                                                                    |
| SHAP (SHapley Additive exPlanations) | An explainable AI method from game theory. Question: Can this be applied to LLMs to explain token-level predictions?                                                                                                            |
| Sociopath Yapper                     | Your term for a current LLM: a system that is highly fluent and convincing but lacks true understanding, empathy, or a stable model of the user.                                                                                |
| Process-Based RL                     | Reinforcement learning that rewards correct reasoning steps, not just final answers. Teaches "how to think" rather than "what to answer." Key for generalizable reasoning.                                                     |
| ReAct                                | Reasoning + Acting - A framework that interleaves reasoning steps with tool-use actions, grounding abstract thought in concrete observations. Critical for tool-augmented AI.                                                  |
| Token / Tokenization                 | The process of converting raw text into a sequence of numerical IDs (tokens) from a fixed vocabulary, which the model can process.                                                                                              |
| ToT (Tree of Thoughts)               | An advanced reasoning framework where an LLM explores and evaluates multiple "branches" of a thought process, allowing for backtracking.                                                                                        |
| Transformer                          | The foundational neural network architecture (based on self-attention) for virtually all modern LLMs.                                                                                                                           |
| User Modeling                        | The process of building and maintaining a dynamic representation of user preferences, context, emotional state, and interaction patterns.                                                                                       |
| Zero-Shot / Few-Shot / Many-Shot     | Learning paradigms: Zero-shot (no examples), Few-shot (2-10 examples), Many-shot (10+ examples). LLMs excel at few-shot learning through in-context prompting.                                                                 |

