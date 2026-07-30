
import logging
import typing
import types as _builtin_types
from typing import Dict, Any, Type, TypeVar

from pydantic import BaseModel

_T = TypeVar("_T", bound=BaseModel)

logger = logging.getLogger(__name__)


# ── JSON Repair utilities ────────────────────────────────────────────────────

# Required str fields whose value should be filled with the raw LLM text on fallback
_CONTENT_FIELD_NAMES = {"message", "content", "lecture", "ai_message", "answer", "response", "text"}


def extract_llm_raw_text(exc: Exception) -> str:
    """Standardize raw LLM text extraction from various LangChain exception formats."""
    if hasattr(exc, "ai_message") and exc.ai_message:
        return exc.ai_message.content
    if hasattr(exc, "llm_output") and exc.llm_output:
        return str(exc.llm_output)
    if "Invalid json output:" in str(exc):
        return str(exc).split("Invalid json output:")[-1].strip()
    return ""


def _strip_markdown_block(text: str) -> str:
    """Remove ```json ... ``` wrapper if present."""
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def safe_parse_structured(raw_text: str, schema_class: Type[_T]) -> _T:
    """
    Robustly parse raw LLM text into a Pydantic schema.

    Steps:
      1. Strip markdown code block (``` ... ```)
      2. Use json_repair to fix any malformed JSON (unterminated strings, trailing
         commas, unescaped newlines, single quotes, truncation, etc.)
      3. model_validate() into schema_class
      4. If all above fails → auto_default_schema() based on field name + type
    """
    from json_repair import repair_json

    cleaned = _strip_markdown_block(raw_text.strip()) if raw_text else ""

    if cleaned:
        try:
            repaired = repair_json(cleaned, return_objects=True)
            if isinstance(repaired, dict):
                return schema_class.model_validate(repaired)
        except Exception as e:
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                fields = [
                    f"{'.'.join(str(l) for l in err['loc'])}: {err['msg']}"
                    for err in e.errors()
                ]
                logger.warning(
                    f"[safe_parse_structured] {schema_class.__name__} validation failed "
                    f"— fields: {fields} | raw[:200]: {cleaned[:200]!r}"
                )
            else:
                logger.debug(f"[safe_parse_structured] {schema_class.__name__}: {e}")

    logger.warning(
        f"[safe_parse_structured] All parse attempts failed for "
        f"{schema_class.__name__}. Using auto_default."
    )
    return auto_default_schema(raw_text or "", schema_class)


def auto_default_schema(raw_text: str, schema_class: Type[_T]) -> _T:
    """
    Auto-construct a safe default Pydantic instance using field name + type heuristics.

    Rules (applied only to required fields without defaults):
      - str with content-like name (message, lecture, ai_message, ...): raw_text
      - str other names (thought, criteria, ...): ""
      - bool: False
      - float: 0.0
      - int: 0
      - List / list: []
      - Dict / dict: {}
      - Literal[...]: first literal value
      - Optional[X] / Union[X, None] / X | None: None
    """
    from pydantic.fields import PydanticUndefined

    kwargs = {}
    for name, field_info in schema_class.model_fields.items():
        has_default = (
            field_info.default is not PydanticUndefined
            or field_info.default_factory is not None
        )
        if has_default:
            continue
        kwargs[name] = _resolve_field_default(name, field_info.annotation, raw_text)

    if kwargs:
        logger.warning(
            f"[auto_default_schema] {schema_class.__name__}: auto-filled {list(kwargs.keys())}"
        )
    return schema_class(**kwargs)


def _resolve_field_default(name: str, annotation, raw_text: str):
    """Return a type-safe default value for a required Pydantic field."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Optional[X] / Union[X, None] (typing) or X | None (Python 3.10+)
    is_union = origin is typing.Union
    if not is_union and hasattr(_builtin_types, "UnionType"):
        is_union = isinstance(annotation, _builtin_types.UnionType)
    if is_union and type(None) in args:
        return None

    # Literal["A", "B", ...] → first value
    if origin is typing.Literal:
        return args[0]

    # Generic containers
    if origin is list:
        return []
    if origin is dict:
        return {}

    # Primitive scalars
    if annotation is str:
        return raw_text if name in _CONTENT_FIELD_NAMES else ""
    if annotation is bool:
        return False
    if annotation is float:
        return 0.0
    if annotation is int:
        return 0

    return None




def extract_agent_result(result: dict, schema_class: Type[_T], node_name: str) -> _T:
    """Extract structured_response from agent result dict.
    Falls back to last message content
    """
    if "structured_response" in result:
        return result["structured_response"]
    msgs = result.get("messages", [])
    raw = msgs[-1].content if msgs and hasattr(msgs[-1], "content") else ""
    logger.warning(f"[{node_name}] 'structured_response' missing — agent didn't submit schema tool. Attempting json_repair.")
    return safe_parse_structured(raw, schema_class)


def parse_agent_steps(messages: list) -> list:
    """ Extract tool call history from agent messages for tracing"""
    parsed = []
    
    for i, msg in enumerate(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name")
                tool_input = tool_call.get("args")
                
                observation = "No output found"
                if i + 1 < len(messages) and messages[i+1].type == "tool":
                    if messages[i+1].tool_call_id == tool_call.get("id"):
                        observation = messages[i+1].content
                
                parsed.append({
                    "agent_action": {
                        "tool": tool_name,
                        "input": tool_input,
                        "log": f"Calling tool {tool_name} with args {tool_input}"
                    },
                    "observation": str(observation)[:500] + "..." if len(str(observation)) > 500 else observation
                })
                
    return parsed


def trilingual_format(vn: str, en: str) -> str:
    return f"**{vn}** ({en}) - **{vn}**"


def compact_explore(raw_tool_output: str) -> str:
    """ Truncate raw explorer output for context window"""
    if not raw_tool_output:
        return ""
    return raw_tool_output[:2500] + "\n...(truncated)" if len(raw_tool_output) > 2500 else raw_tool_output


# KG tools whose observations carry the prerequisite/structure signal the Evaluator needs.
# Ordered by priority so the most decision-relevant output survives the context cap.
_KG_CONTEXT_TOOLS = ("course_relevance", "course_backbone", "recommend_new")


def extract_kg_context(messages: list, cap: int = 8096) -> str:
    """Pull KG tool observations from a RAG agent trace into a single string for the
    Roadmap_Evaluator prompt.

    Without this the tool output dies in the agent's intermediate messages and the
    Evaluator gets an empty ``course_relevance`` — the prereq check then runs blind.

    ``cap`` bounds total length (default 8096 chars) so a large backbone dump can't
    blow the downstream context window; course_relevance is emitted first so the
    prereq signal is the last thing dropped on truncation.
    """
    if not messages:
        return ""

    collected = {name: [] for name in _KG_CONTEXT_TOOLS}
    for i, msg in enumerate(messages):
        for tool_call in getattr(msg, "tool_calls", None) or []:
            name = tool_call.get("name")
            if name not in collected:
                continue
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            if nxt is not None and getattr(nxt, "type", "") == "tool" \
               and getattr(nxt, "tool_call_id", None) == tool_call.get("id"):
                collected[name].append(str(nxt.content))

    chunks = []
    for name in _KG_CONTEXT_TOOLS:
        for obs in collected[name]:
            chunks.append(f"[{name}]\n{obs}")

    joined = "\n\n".join(chunks)
    if len(joined) > cap:
        return joined[:cap] + "\n...(truncated)"
    return joined


def filter_mastery(roadmap_data: dict, mastery_map: dict) -> dict:
    """ Filter mastery_map to only backbone-relevant entries"""
    backbone_names = {n.get("name") for n in roadmap_data.get("steps", []) if isinstance(n, dict)}
    
    relevant_mastery = {}
    if mastery_map:
        for k, v in mastery_map.items():
            if k in backbone_names:
                relevant_mastery[k] = v
            elif any(k.lower() in name.lower() or name.lower() in k.lower() for name in backbone_names):
                relevant_mastery[k] = v
                
    return relevant_mastery


def parse_student_state(state: dict) -> str:
    """ Serialize student state to compact string for prompt injection"""
    parsed_state = {}
    
    current = state.get("current_pos")
    if current:
        parsed_state["current_target"] = {
            "name": current.name,
            "type": current.type,
            "description": current.description,
            "node_mastery": current.mastery
        }
        
        mastery_map = state.get("mastery_map", {})
        if current.name in mastery_map:
            parsed_state["current_target"]["global_mastery_score"] = mastery_map[current.name]

    parsed_state["path"] = {
        "recently_completed": [n.name for n in state.get("previous_nodes", [])[:3]],
        "upcoming": [n.name for n in state.get("upcoming_nodes", [])[:3]]
    }

    if state.get("summary"):
        parsed_state["student_profile"] = state.get("summary")

    if state.get("active_resource"):
        parsed_state["active_resource"] = state.get("active_resource")

    return str(parsed_state)