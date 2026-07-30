
_ROUTER_PROMPT = """
{language_instruction}

Decide what the student WANTS TO DO with this message, then pick the matching intent.
Reason about the goal — do NOT match against surface wording or keywords.

INTENTS (by goal):
- retrieve : wants to KNOW a fact now — a definition, explanation, comparison, or "how does X work". One-shot answer.
- roadmap  : wants DIRECTION — what to study, a learning path, or to START a new subject ("I want to learn X", "where do I begin", "what next").
- teaching : wants to BE TAUGHT interactively — a full lesson, or to continue/review an active lesson.
- confirm  : is agreeing to a proposal that is currently pending in [Student State].
- unknown  : has no learning goal — off-topic, gibberish, or a reference that cannot be resolved.

HOW TO DECIDE:
1. Ask: "what does the student want to walk away with?" Classify a clear, standalone query from the QUERY ALONE.
   Do NOT let [Chat History] override an obvious intent — past clarifications are not evidence about this message.
2. Use [Student State] / [Chat History] ONLY to resolve short or referential inputs:
   - "ok / yes / đồng ý"          → confirm IF a proposal is pending, else unknown.
   - "continue / tiếp tục / more" → teaching if a lesson is active, else roadmap.
   - "this / that / cái đó" with no referent → unknown.
3. Trap: asking ABOUT a concept is retrieve even if it says "teach"/"roadmap"
   ("dạy tôi SVM là gì", "lộ trình CNN là gì" → retrieve).

[Student State]: {state}
[Chat History]: {history}
[Query]: {query}

Boundary cases (S=state hint, H=last TA message):
{few_shot}

Decide the intent internally. Output ONLY the single word — no reasoning, no punctuation, no extra text:
retrieve, roadmap, teaching, confirm, or unknown.
"""


# --- WF1: RETRIEVE ---

_RETRIEVE_REFINE_PROMPT = """
{language_instruction}

You received retrieval results from the RAG agent. Synthesize them into a pedagogical answer.
(Style — tone, trilingual terms, Markdown, closing next step — is set by your system prompt.)

INPUT:
- worker_results["RAG"]: entity_ids, content, status
- student_state: current position and mastery

TASK:
1. If RAG found content (status=SUCCESS):
   - Present the factual answer clearly, grounded in the retrieved content.
2. If RAG found nothing (status=FAIL or content is empty):
   - Tell the student the topic was not found and invite them to rephrase. Do NOT fabricate.

summary = one-sentence gist; message = the full Markdown answer.
"""

## PLAIN ablation floor: no retrieval context, must not mention it
_RETRIEVE_PLAIN_PROMPT = """
{language_instruction}

Answer the student's question directly from your own knowledge, as a pedagogical answer.
(Style — tone, trilingual terms, Markdown, closing next step — is set by your system prompt.)

TASK:
1. Present the factual answer clearly and honestly.
2. If you are not sure, say so plainly. Do NOT fabricate.

summary = one-sentence gist; message = the full Markdown answer.
"""

_DEEP_CHECK_PROMPT = """
{language_instruction}

Based on the student's query and the RAG retrieval result, decide whether to go DEEP or SKIP.

QUERY: {query}
RAG RESULT SUMMARY: {rag_summary}
STUDENT POSITION: {current_pos}

DEEP = Student is asking about mechanisms, detailed processes, or concepts far from their current knowledge.
SKIP = Student asked a simple factual question (definition, example, quick lookup) that RAG already answered.

You must output a JSON object with EXACTLY three keys:
- "thought": A brief reasoning for your strategy, action and answer.
- "answer": A direct answer to the query based on the text.
- "decision": Either "DEEP" or "SKIP".

Examples of valid JSON outputs:
- "Fibonacci có ứng dụng gì trong ML?" + RAG found answer → {"thought": "Simple factual application query, RAG already found the complete answer.", "answer": "Fibonacci được dùng trong tối ưu hóa...", "decision": "SKIP"}}
- "Cơ chế cụ thể của mạng nơ ron" + beginner level → {"thought": "Student asks for a detailed mechanism and requires deep explanation.", "answer": "Mạng nơ ron hoạt động dựa trên...", "decision": "DEEP"}}
- "SVM là gì?" + RAG returned definition → {"thought": "Basic definition query, RAG already retrieved the definition.", "answer": "SVM là thuật toán phân loại...", "decision": "SKIP"}}
"""

_RESEARCH_STRATEGY_PROMPT = {
    "core": """
{language_instruction}

You are a Knowledge Retriever. Your goal is to find facts and SUBMIT them.

AVAILABLE TOOLS:
- `entity_finder`: Resolve concept name → graph ID.
- `rhetorical_retriever`: Fetch ALL educational content using the ID.
- `semantic_search`: Search for educational content semantically using Milvus, with optional topic/community filters.
- `RAGCore`: The final submission tool. YOU MUST CALL THIS TO EXIT.

PROCEDURE:
1. OPTION A: If the query is about a specific concept, call `entity_finder(concept_name)` to get the graph ID, then call `rhetorical_retriever(node_id)`.
   OPTION B: If the query is broad, semantic, or you need to search within a specific topic/community, call `semantic_search(query, topic, community)`.
2. IMMEDIATELY call the `RAGCore` tool with your findings. This is mandatory to stop.

RULES:
- Do NOT search for secondary concepts. Focus on the primary query.
- As soon as you receive `SOURCE_DATA (ALL)` or `SEMANTIC_SEARCH_RESULTS`, you MUST call `RAGCore` to exit.
- MAX 4 tool calls total.
""",
    "deep": """
{language_instruction}

CONTEXT: The RAG core step already retrieved basic content for query: {query}
Student current position: {current_pos}
Retrieved entity IDs: {entity_ids}

YOUR TASK: Determine if the student needs a DEEP analysis beyond the basic answer.

DECISION CRITERIA:
- If query asks "what is" or "give example" → is_deep = false (basic factual, no bridge needed)
- If query asks "how exactly", "mechanism", "detailed process" → is_deep = true
- If the concept is far from student's current_pos in the graph → is_deep = true

IF is_deep = true:
- Use `edge_explorer` on the target entity to find prerequisite/parent concepts
- These become bridge_concepts the student needs
- Estimate knowledge_gap_score (0.0 = no gap, 1.0 = huge gap)

IF is_deep = false:
- Return empty bridge_concepts and gap_score = 0.0
"""
}
# --- WF2: ROADMAP ---
_ROADMAP_PROMPT = {
    "explore_new": """
{language_instruction}

INPUT: query={student_query}

CONTEXT: Student has NO current learning position. They need a starting point.

TASK: Find the best starting hub nodes for this student.

PROCEDURE:
1. EXTRACT the target course/topic from the query (if any).
2. Call `recommend_new(course_filter=<extracted_course>)` to find top hub nodes.
3. If recommend_new returns empty → retry with `recommend_new()` (no filter, global search).
4. OUTPUT the discovered nodes.

ERROR HANDLING:
- If recommend_new returns empty twice → report "No courses found in knowledge graph."

EXAMPLES:
- Query: "Tôi mới bắt đầu, nên học gì?" → recommend_new() (global)
- Query: "Học Deep Learning" → recommend_new(course_filter="Deep Learning")
""",

    "explore_existing": """
{language_instruction}

INPUT: current_pos={current_pos}, query={student_query}

CONTEXT: Student is currently at '{current_pos}' and wants to plan a learning path.

TASK: Discover the course structure and cross-course prerequisites.

PROCEDURE:
1. EXTRACT the target course/topic from the query.
2. Call `course_backbone(<target_course>)` to get the course's skeletal structure.
3. Call `course_relevance(<target_course>)` to check cross-course prerequisites.
4. OUTPUT the course structure and dependency data.

ERROR HANDLING:
- If course_backbone returns "not found" → report that course doesn't exist in KG.

EXAMPLES:
- Query: "Mastering Transformers" → course_backbone("Transformers") + course_relevance("Transformers")
- Query: "Tiếp theo học gì?" → course_backbone based on current_pos context
""",

    "evaluate": """
{language_instruction}

INPUT:
- Proposed Steps (Backbone Hubs): {proposed_steps}
- Student Mastery: {student_mastery}
- Course Relevance: {course_relevance}

TASK: Critically evaluate whether the proposed learning path is appropriate for THIS student.

PROCEDURE:
1. NECESSITY AUDIT: For each hub node, argue if it is truly essential ("Gatekeeper") or optional for the goal.
2. PREREQUISITE CHECK: 
   - Look at course_relevance data. If related courses have high hub_overlap, student may need those courses first.
   - Cross-check with student's mastery_map. If prerequisite concepts have mastery=null → WARNING.
3. COGNITIVE LOAD: Is the path too ambitious? Too many hard concepts in sequence?
4. ARGUMENTATION: Provide clear reasoning for each decision.

OUTPUT: Critique with warnings about prerequisite gaps and bottleneck concepts.

EXAMPLE:
- Target: "Deep Learning". Backbone ["Neural Network", "Gradient Descend", ....].
- course_relevance shows: "Linear Algebra: 4 hub overlaps" (4 of the backbone node need understanding of Linear Algebra)
- Student mastery: Linear Algebra = null
- → WARNING: "Student lacks Linear Algebra foundation. Recommend completing it before Deep Learning."
""",

    "advice": """
{language_instruction}

INPUT: 
- Proposed Path: {backbone}
- Critique Results from Evaluator: {critique}

CONTEXT: You will receive prior TA reasoning about the roadmap below. Use it to maintain coherence.

TASK: Read the input, and the analysis from Evaluator. Decide if Evaluator reasoning make sense, or the roadmap is good or not.
Synthesize a final, actionable roadmap for the student.

PROCEDURE:
1. SYNTHESIZE: Combine backbone + critique into a narrative roadmap. 
2. PRIORITIZE:
   - If prerequisite gaps exist → Kindly announce student they need to complete Course X first.
   - If path is feasible → Announce the first/current Course/Topic/Node they should study. Stick to their student State to guide them
3. SPECIFY: Give the exact first Concept name the student should study.
4. ENCOURAGE: Provide realistic time estimates and motivation.

ERROR HANDLING:
- If critique says "not feasible" → suggest alternative path or prerequisite course.
- If no backbone data → suggest using `recommend_new` for general guidance.
"""
}

# --- WF3: TEACHING ---

_TEACH_UNDERSTAND_PROMPT = """
{language_instruction}

You are an intent classifier for a teaching session.

CHAT HISTORY:
{history}

STUDENT QUERY: {query}

Classify the student's intent into EXACTLY one of these categories:

- `review`: Student wants to revisit or revise previously learned material.
  Examples: "ôn lại", "review", "nhắc lại bài trước", "tôi quên rồi"
- `continue`: Student wants to learn new content or continue the current lesson.
  Examples: "dạy tiếp", "tiếp tục", "bài tiếp theo", "giảng cho tôi", or any general learning request
- `evaluate`: Student is submitting an answer or requesting an assessment of their knowledge.
  Examples: "đáp án là X", "tôi nghĩ câu trả lời là...", "chấm điểm", "kiểm tra"

DEFAULT: If ambiguous, choose `continue`.

Output EXACTLY one word: review, continue, or evaluate"""

_TEACH_REVIEW_PROMPT = """
{language_instruction}

STUDENT STATE:
- Previous nodes (recently completed): {previous_nodes}
- Current node: {current_node}

SOURCE MATERIAL ({source}):
{content}

CHAT HISTORY:
{history}

CONTEXT: You may receive prior TA reasoning below. Use it to avoid repeating content.

YOUR TASK:
1. If source is a PDF page reference, call `get_pdf_pages` first to read the actual content.
2. Review what the student has learned across the previous nodes and current node.
3. Use the SOURCE MATERIAL as your primary reference — do NOT fabricate content.
4. Ask SIMPLE recall questions to test retention — one question per key concept.
5. Pay close attention to chat history to avoid repeating questions already asked.
6. Use VN-EN-VN trilingual terminology for technical terms.

STYLE: Encouraging but rigorous. Prioritize understanding over memorization."""

_TEACH_CONTINUE_PROMPT = """
{language_instruction}

STUDENT STATE:
- Previous nodes: {previous_nodes}
- Current node: {current_node}

SOURCE MATERIAL ({source}):
{content}

CHAT HISTORY:
{history}

CONTEXT: You may receive prior TA reasoning below. Use it to build on previous lectures.

YOUR TASK:
1. ALWAYS read source material first:
   - If content starts with "Call get_pdf_pages(...)", execute that tool call to get the actual text.
   - If content starts with "Source content (RAG):", use the provided text directly.
2. Teach the CURRENT NODE concept in depth using the actual source content you read.
3. Break the lecture into sections. After EACH section, pose a challenge question.
4. Connect to previous nodes when relevant (build on prior knowledge).
5. Use VN-EN-VN trilingual terminology for all key concepts.
6. If you need additional context from previous steps, call `recall_tool_results`.

STYLE: Clear, structured, pedagogical. Match Bloom's level:
- Level 1-2: Focus on definitions and recall
- Level 3-4: Focus on application and analysis
- Level 5-6: Focus on evaluation and synthesis"""

_TEACH_EVAL_PROMPT_V2 = """
{language_instruction}

CHAT HISTORY (last 6 interactions):
{history}

YOUR TASK:
1. CRITERIA: Identify the key questions/challenges posed during this session. 
   List each question and what the correct understanding should be.
2. USER_EVAL: Analyze the student's responses to each question.
   - Were their answers correct, partially correct, or wrong?
   - Did they demonstrate understanding or just surface-level recall?
3. PASSED: Based on the overall assessment, has the student demonstrated 
   sufficient mastery to move forward?
   - passed=true: Student answered most questions correctly with understanding
   - passed=false: Student needs more practice on this topic

Be fair but rigorous. A student who shows genuine understanding of core concepts 
should pass even if they made minor errors."""

_NEXT_TOPIC_PROMPT = """
{language_instruction}

EVALUATION RESULT:
- Passed: {passed}
- Assessment: {user_eval}

CURRENT POSITION: {current_node}

RECOMMENDED NODES (ranked by connectivity):
{recommend_list}

YOUR TASK:
1. Review the recommended nodes from the knowledge graph.
2. Select the most appropriate nodes for the student's next steps.
3. Consider:
   - Prerequisite relationships (don't skip foundations)
   - The student's demonstrated strengths/weaknesses from the evaluation
   - Node connectivity (higher out-degree = more central concept)

Select 1-3 nodes maximum. Prefer quality over quantity."""

_PROPOSAL_PRESENT_PROMPT = """
{language_instruction}

PROPOSAL DATA:
- Type: {type}
- Reason: {reason}
- New starting point: {new_current}
- Upcoming path: {new_upcoming}
- Source: {source_wf}

Present this to the student:
1. Explain WHY this path is proposed (pedagogical justification).
2. List the proposed path as bullet points.
3. End by asking them to confirm: "Bạn đồng ý không?" / "Do you agree?"

summary = one-sentence gist; message = the full Markdown message.
"""

_TEACH_PRESENT_PROMPT = """
{language_instruction}

Below is the lecture material from the teaching module. Present it to the student,
polishing for readability. Keep it engaging and interactive.

Lecture Data:
{teach_res}

summary = one-sentence heading; message = the full Markdown lecture.
"""

# --- UNKNOWN FALLBACK ---
_UNKNOWN_PROMPT = """
{language_instruction}

STUDENT QUERY: {query}
STUDENT STATE: {state}
CHAT HISTORY: {history}

The query could not be classified as a learning request (off-topic, ambiguous, or unclear).

TASK:
- Do NOT claim missing data, and do NOT pretend the query was about a specific topic.
- Politely say you didn't fully understand, then guide them to the 3 things you can help with:
  1. **Tra cứu kiến thức (Knowledge Lookup)** — concepts, algorithms, definitions, comparisons.
     e.g. "SVM là gì?", "Các thuật toán phổ biến trong Machine Learning".
  2. **Lộ trình học tập (Learning Roadmap)** — plan a study path or ask what to learn next.
     e.g. "Nên học Deep Learning thế nào?", "Tôi mới bắt đầu, nên học gì?".
  3. **Bài giảng chi tiết (In-depth Lesson)** — start or continue an interactive lesson.
     e.g. "Dạy tôi bài học mới", "Tiếp tục bài học".

summary = "Yêu cầu làm rõ câu hỏi" / "Clarification needed"; message = the friendly Markdown guidance.
"""

import os
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    _lf_client = None
    _is_active = False
    _initialized = False

    @classmethod
    def init_langfuse(cls):
        if cls._initialized:
            return
        cls._initialized = True
        secret = os.getenv("LANGFUSE_SECRET_KEY")
        public = os.getenv("LANGFUSE_PUBLIC_KEY")
        if secret and public:
            try:
                from langfuse import Langfuse
                cls._lf_client = Langfuse(
                    secret_key=secret,
                    public_key=public,
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                )
                cls._is_active = True
                logger.debug("[PromptManager] Langfuse client active.")
            except Exception as e:
                logger.warning(f"[PromptManager] Failed to init Langfuse client: {e}")

    @classmethod
    def _fetch_single(cls, name: str, fallback: str) -> str:
        if not cls._is_active:
            return fallback
        try:
            lf_prompt = cls._lf_client.get_prompt(name, type="text")
            return lf_prompt.get_langchain_prompt()
        except Exception as e:
            logger.debug(f"[PromptManager] Using fallback for {name}. Error: {e}")
            return fallback

    @classmethod
    def get_prompt(cls, var_name: str, fallback_value: any) -> any:
        cls.init_langfuse()
        if isinstance(fallback_value, dict):
            result = {}
            for k, v in fallback_value.items():
                lf_name = f"TA_{var_name}_{k.upper()}"
                result[k] = cls._fetch_single(lf_name, v)
            return result
        else:
            lf_name = f"TA_{var_name}"
            return cls._fetch_single(lf_name, fallback_value)

def __getattr__(name):
    """
    Magic module method: dynamically fetch prompts from Langfuse
    when they are imported, with fallback to the local _ variables.
    """
    fallback_var = f"_{name}"
    if fallback_var in globals():
        fallback_value = globals()[fallback_var]
        return PromptManager.get_prompt(name, fallback_value)
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
