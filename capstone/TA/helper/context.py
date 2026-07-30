from core.schema.wf_state import AgentState

def extract_ta_context(state: AgentState, max_msgs: int = 2) -> str:
    """ 
    Extract last N assistant messages from state for TA context injection.
    Only extracts standard UI-facing assistant messages, effectively skipping intermediate agent steps
    which are now logged directly to the DB's memo array.
    """
    messages = state.get("messages", [])
    ta_msgs = []
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            ta_msgs.append(msg.content)
        elif isinstance(msg, dict) and msg.get("role") == "assistant":
            ta_msgs.append(msg["content"])
        if len(ta_msgs) >= max_msgs:
            break
    return "\n---\n".join(reversed(ta_msgs))
