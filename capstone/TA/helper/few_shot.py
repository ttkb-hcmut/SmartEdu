"""These shots are synthesis by Sonnet 4.5
These shots are the hard boundary cases only — no obvious anchor spam"""
# TA/helper/few_shot.py
## -- Few-shot examples for TA_Router intent classification
## -- Design principles:
##    1. Hard boundary cases ONLY — no obvious anchor spam
##    2. Include state_hint + history_hint for context-dependent cases
##    3. WHY annotations are for developer docs only, NOT injected into the prompt
##    4. Compact output format to respect the 4096 token context window budget
##
## -- Context window budget estimate (TA profile, num_ctx=4096):
##    student_state  : ~200-400 tokens
##    chat_history   : ~300-600 tokens
##    prompt base    : ~150 tokens
##    few_shot output: target ≤ 500 tokens  ← THIS FILE controls this
##    query          : ~20-100 tokens
##    safety margin  : ~400 tokens
##    ────────────────────────────────────
##    total target   : ≤ 2000 tokens (leaves room for num_ctx=4096)

# ─────────────────────────────────────────────────
# Schema of each example:
#   query       : the student's raw input
#   state_hint  : (optional) relevant part of student_state that matters for this decision
#   history_hint: (optional) last TA message that affects classification
#   intent      : ground-truth label
#   why         : developer-only note, NEVER injected into prompt
# ─────────────────────────────────────────────────

## -- Boundary cases ONLY. The intent definitions in _ROUTER_PROMPT cover the easy
##    cases; these examples exist solely to disambiguate inputs that CANNOT be
##    classified from the query alone (need state/history) or that fight keyword bias.
_VN_EXAMPLES = [
    # affirmation: confirm vs unknown depends on pending_proposal
    {
        "query": "Ok",
        "state_hint": "pending_proposal: roadmap đến CNN",
        "history_hint": "TA: 'Bạn có muốn áp dụng lộ trình này không?'",
        "intent": "confirm",
        "why": "[CTX] Affirmation + pending proposal → confirm.",
    },
    {
        "query": "Tôi hiểu rồi",
        "state_hint": "pending_proposal: null",
        "history_hint": "TA: 'SVM phân loại dữ liệu bằng hyperplane.'",
        "intent": "unknown",
        "why": "[CTX][TRAP] Same kind of words, but no proposal pending → just reacting, not confirming.",
    },
    # "tiếp tục": teaching vs roadmap depends on whether a lesson is active
    {
        "query": "Tiếp tục đi",
        "state_hint": "current_pos: Neural Network",
        "history_hint": "TA: 'Phần 1 về Forward Pass hoàn thành. Bạn có câu hỏi không?'",
        "intent": "teaching",
        "why": "[CTX] Active lesson in history → resume lesson, teaching.",
    },
    {
        "query": "Tiếp tục đi",
        "state_hint": "current_pos: null",
        "history_hint": "",
        "intent": "roadmap",
        "why": "[CTX] No lesson active → 'where do I go next?', roadmap.",
    },
    # keyword bias: "dạy"/"lộ trình" wording but the goal is a factual lookup
    {
        "query": "Dạy tôi biết SVM là gì",
        "state_hint": "",
        "history_hint": "",
        "intent": "retrieve",
        "why": "[TRAP] 'Dạy tôi' but purely definitional → retrieve.",
    },
    {
        "query": "Lộ trình học CNN là gì?",
        "state_hint": "",
        "history_hint": "",
        "intent": "retrieve",
        "why": "[TRAP] 'Lộ trình' keyword but asking ABOUT the concept, not requesting a personal plan.",
    },
]

_ENG_EXAMPLES = [
    # affirmation: confirm vs unknown depends on pending_proposal
    {
        "query": "Sure",
        "state_hint": "pending_proposal: roadmap to CNN",
        "history_hint": "TA: 'Do you want to apply this learning path?'",
        "intent": "confirm",
        "why": "[CTX] Affirmative + pending proposal → confirm.",
    },
    {
        "query": "I see",
        "state_hint": "pending_proposal: null",
        "history_hint": "TA: 'SVM classifies data using a hyperplane.'",
        "intent": "unknown",
        "why": "[CTX][TRAP] Reaction with no pending proposal → unknown.",
    },
    # "continue": teaching vs roadmap depends on whether a lesson is active
    {
        "query": "Continue",
        "state_hint": "current_pos: Neural Network",
        "history_hint": "TA: 'Part 1 on Forward Pass is done. Any questions?'",
        "intent": "teaching",
        "why": "[CTX] Active lesson → resume lesson, teaching.",
    },
    {
        "query": "Continue",
        "state_hint": "current_pos: null",
        "history_hint": "",
        "intent": "roadmap",
        "why": "[CTX] No lesson context → where do I go next? roadmap.",
    },
    # keyword bias: "teach"/"roadmap" wording but the goal is a factual lookup
    {
        "query": "Teach me what SVM is",
        "state_hint": "",
        "history_hint": "",
        "intent": "retrieve",
        "why": "[TRAP] 'Teach me' + purely definitional → retrieve.",
    },
    {
        "query": "What does a Deep Learning roadmap look like?",
        "state_hint": "",
        "history_hint": "",
        "intent": "retrieve",
        "why": "[TRAP] Asking ABOUT a roadmap concept = retrieve, not requesting a personal plan.",
    },
]

FEW_SHOT_EXAMPLES = {
    "vn": _VN_EXAMPLES,
    "eng": _ENG_EXAMPLES,
}


def format_few_shot(language: str = "vn") -> str:
    """
    Format few-shot examples for injection into the Router prompt.

    Compact format to respect context window budget (target ≤ 500 tokens):
      [S: state_hint] [H: history_hint] "query" → intent

    WHY annotations are intentionally EXCLUDED from output — they are
    developer docs only. The compact format is what the LLM sees.
    """
    examples = FEW_SHOT_EXAMPLES.get(language, FEW_SHOT_EXAMPLES["vn"])
    lines = []
    for e in examples:
        parts = []
        if e.get("state_hint"):
            parts.append(f'[S: {e["state_hint"]}]')
        if e.get("history_hint"):
            parts.append(f'[H: {e["history_hint"]}]')
        parts.append(f'"{e["query"]}" → {e["intent"]}')
        lines.append("  " + " ".join(parts))
    return "\n".join(lines)


LANGUAGE_INSTRUCTIONS = {
    "vn": "Bạn PHẢI trả lời bằng Tiếng Việt trong mọi trường hợp.",
    "eng": "You MUST respond in English in all cases.",
}


def get_language_instruction(language: str = "vn") -> str:
    return LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["vn"])
