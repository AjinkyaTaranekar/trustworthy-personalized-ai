### Appraisal Theory

- Appraisal Theory: For any event, the emotion an experiencer feels is mostly because of subevents happening.
    - The model was trained on the Crowd-event dataset , which contains 6600 events.
    - Appraisal changes and for which it gets appraisal.
    - Appraisal changes the outcome of the next event.
- **Questions:**
    - How were the 21 appraisals set for each event?
    - Does the experiencer know and understand all 21 appraisal words clearly?
    - What was the mood before the event?
    - What are the demographics of this dataset?
    - How much time passed after the event when it got recorded?
        - What if someone else has consoled them, which changed their thought process later?
- **Testing Method:**
    - Zeroshot testing
- **Dataset Comparison:**

|   |   |   |   |
|---|---|---|---|
|Empathetic Dialogue|Daily Dialogue|Epitome|Emowoz|
|101 empathetic convos. (Best Results)|multiturn dialogues daily "chit-chat"|mental health support platform Reddit & Talklife|task oriented dialogue|

- **Model & Metrics:**

|   |   |
|---|---|
|Abbreviation|Full Term|
|T|Text|
|A|Appraisal|
|E|Emotion class|
|T+A -> E|Text + Appraisal predicts Emotion|
|TA|Text -> Appraisal (prediction)|
|TE|Text -> Emotion|
|P|Precision|
|R|Recall|
|F1|Harmonic mean of P & R|

|   |
|---|
|Model/Method|
|DeBERTA-Large|
|Adam optimizer|

**User Modeling**

- **Empathetic AI Needs to Be:**
    - Mental Models
    - Good Listener
    - Adaptively Ask Right Questions
        - For each dialogue, there needs to be 5W+H
    - User Models
        - "Ask why behind it"
    - Dynamic Updating
    - Plans & Persuasion

_Side Note:_ Ask simple questions (Yes/No) instead of open-ended ones.

- **Core Concept:**
    - User modeling is not just knowing the user but adapting to the user during the conversation.
    - **Example:** If a user is going fast in a speech convo, the model could ask them to slow down.
- **Fundamental Question:**
    - What makes Humans & AI different?

---

### **Literature: What makes Humans & AI different**

- **Process Flow:**
    - Understanding
    - Ask Right Questions
    - Interest
    - Discovery
    - narrative journey
- **Key Concepts:**
    - be a sense maker
    - Relevance to the challenge
    - "Scrutability"
        - why it's doing what it's doing
        - How user [understands]
    - Expressive Literature Searching
    - A good dissertation needs a solid foundation and research question. It should have a clear Title, Abstract, and Conclusion.
    - Not a Fixed Topic
- **Inspiration:**
    - "Her" movie
- **Meeting Details:**
    - **Next Meeting:** 1st Oct. 11-12pm
    - **Location:** F29 - O'Reilly Building
- **Core Idea:** User Empathy with AI interaction
    - LLM lacks explainability
        - Is it based on weighted parameters? If so, how? Which models?
        - How did they generalize?
- **Scenario: Contextual Understanding**
    - Let's say it's trained on Open Internet Forums where people talked about having a Frappuccino on a sunny day.
        - What about temperature? Location?
        - A "sunny" day feels different at 20°C in Dublin versus 35°C in Delhi. Why only this coffee for the same feeling with different temperatures?
- **Proposed Solution:**
    - What if AI asks the right questions to itself and the (multiple) knowledge graph(s)?
        - How will AI ask questions?
        - Mental models can be built using user dialogue.
        - Based on location, let's build the initial knowledge, maybe from user's metadata too, to solve the cold start problem.
- **Technical Questions:**
    - Can SHAP be used on LLMs? How will we get to the internal layer?
    - What are sociopaths? → manipulation, deceit, lacks empathy.
    - Can a "System prompt" fix this? Or is there more to it?

### **Four-Phase Process:**

- **Problem Statement:**
    - How can we build a trustworthy Conversational LLM?
        - Explainable, Accurate, Privacy
    - It should adapt to user preferences and know the user in and out.
        - **Information Sources:**
            - From metadata: Location, Time zone, Current Time, Device, Battery?, Apps on the device?
            - From previous chat history, basic intro settings (what user does, what they are looking for, other ice breakers).
            - From connected social profiles.
    - **Data-Backed Answers:**
        - via RAG
        - via Datasource
        - via Web Search
    - **User Aligned:**
        - via a model trained on a public dataset
- **Primary Question:** Can it also have empathy?
    - Detect Appraisal points from the user convo.
    - Act as a good listener / gossip companion.
    - Can LLM, given how a user is feeling (appraisals) and event descriptions, generate an empathetical response?
    - Can it help connect the dots?
    - Using knowledge base graphs, can it personalize for the user?
- **Q: What is reasoning in LLM?**
    - An LLM predicts the next possible token/concept.
    - Currently, reasoning is just either expansion and self-realization/reinforcement to get more accurate steps, based on data fed with steps (generally for Maths, Physics, or core subjects).
    - So if given a puzzle, it fails when there are changes (e.g., Apple's Illusion Thinking paper).
- **Q: So how do you reason a reasoning?** What is reasoning actually? How does a human reason?
    - Make connections between different info & draw conclusions.
    - From a logical and emotional setting.
    - Systematic information gathering.
    - Nature vs. Nurture: Some people are skeptical or in detective mode; some people reason based on past experiences or emotions.
- **Q: Can LLM build knowledge graphs while doing reasoning?**
    - **Proposed Flow:**
        1. Given an LLM and a question.
        2. Break down into steps using mental models and 5W+H.
        3. Gather data for each step.
        4. Connect with the user's own graph / area graph / belief graph.
        5. Now, form a Knowledge Graph of its own.
        6. The articulation point must be the answer , OR the breakdown of each step using 5W+H is the answer.
- **Example:**
    - **User:** "I reached late to class on the first day and got scolded. What to do to reach early?"
    - **Existing LLM response:** Would throw away its knowledge on how to not get late. It won't ask you why you got late in the first place. It assumed directly I might have not packed my bag or woke up late—general stuff.
    - **Conclusion:** So current LLMs are just "sociopath yappers" XD.
    - **Ideal response:** "Oh! How did it happen? Why did you get late?"
    - **Principle:** LLMs should be listeners but not interrogating.

**System Prompts**

- Can reasoning be done via just task breakdown using a great mental model and 5W+H framework?
    - **Proposed Architecture:**
        1. Tree of thoughts solution for each step.
        2. Backtracks if incorrect, else go with the next step in the branch.
        3. Feedback for RL of each step.
        4. Context accumulation via self-attention.
        5. Generate response.
        6. Check relevance.
- **Trade-off: Speed v/s Accuracy**
    - If the system is a chatbot / customer service / live voice agent, speed and accuracy both matter significantly.
    - If the system is a research agent, accuracy matters more.
- **11th October Meeting Agenda**
    - What is scrutability for LLM?
    - Which concepts can or cannot be explained?
    - Blanchard's Hot Chocolate*
    - Find out if a response can be explained to a general user and why the LLM generated it.
    - Know about user.
    - LLM = "Sociopath Yapper"

**Scrutability**

- **Definition:**
    - The quality of being open to scrutiny or able to be understood.
    - The act of carefully examining something, especially in a critical way.
- **Q: Can an LLM scrutinize?**
    - **How LLMs work:** Transformer Layer → Predicts next token based on statistics.
    - **The problem:** We can't tell the user how each token is predicted.
- **Q: What are the other ways to scrutinize then?**
    - State the sources.
    - Maybe the user asked/told about their liking.
    - Maybe use some other models like a Decision Tree from Machine Learning.
- **Final Thoughts:**
    - Can we use yapping of LLMs to predict what question user might ask next.
    - Why do Humans respond to someone's question?
        - Building Relation?
        - Some Ulterior motive?
        - Is it for some kind of reward?
            - Can LLM knows giving this answer will reward it? Reinforcement Learning.
            - But data is every changing.
            - What if LLM RL can be done only for thought process.
            - AbsoluteZero paper
            - But without training how will it connect dots?
            - Wait isn't everything situation reaction? We need to just fix thought process and pour in data.
        - Family or Close friends
        - Respect
        - Karma?

  

> [!info] Oopsie 👀 | Yashar Ahmadov | 95 comments  
> Oopsie 👀 | 95 comments on LinkedIn  
> [https://www.linkedin.com/posts/yashar-ahmadov_oopsie-activity-7382823679182655489-nEjq?utm_source=share&utm_medium=member_desktop&rcm=ACoAACfjSv8BzwFqChRt0hviMOFJ4hayX-iyeFM](https://www.linkedin.com/posts/yashar-ahmadov_oopsie-activity-7382823679182655489-nEjq?utm_source=share&utm_medium=member_desktop&rcm=ACoAACfjSv8BzwFqChRt0hviMOFJ4hayX-iyeFM)