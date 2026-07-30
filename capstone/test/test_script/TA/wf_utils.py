from TA.helper.utils import safe_parse_structured, auto_default_schema, extract_llm_raw_text
from TA.helper.schema import RAGCore, RAGDeep, DeepDecision
from core.schema.wf_state import TAOutput


def test_safe_parse_structured():
    # Test 1: trailing comma (json_repair)
    r1 = safe_parse_structured('{"thought": "ok", "entity_ids": [], "content": "", "status": "FAIL",}', RAGCore)
    print('Test1 (trailing comma):', r1.status)
    assert r1.status == "FAIL"

    # Test 2: markdown block strip + valid JSON
    r2 = safe_parse_structured('```json\n{"thought": "ok", "is_deep": false}\n```', RAGDeep)
    print('Test2 (markdown + valid JSON):', r2.is_deep)
    assert r2.is_deep == False

    # Test 3: auto_default on garbage -> RAGCore (status has default "SUCCESS")
    r3 = safe_parse_structured('this is not json at all', RAGCore)
    print('Test3 (auto_default RAGCore thought):', repr(r3.thought), '| status:', r3.status)
    assert isinstance(r3, RAGCore)

    # Test 4: auto_default TAOutput - message is in CONTENT_FIELD_NAMES -> gets raw_text
    r4 = safe_parse_structured('fallback text for user', TAOutput)
    print('Test4 (TAOutput.message):', r4.message)
    assert r4.message == 'fallback text for user'

    # Test 5: Literal auto_default -> first literal arg
    r5 = auto_default_schema('', DeepDecision)
    print('Test5 (Literal DeepDecision.decision):', r5.decision)
    assert r5.decision in ("DEEP", "SKIP")

    # Test 6: extract_llm_raw_text from "Invalid json output:" exception
    class FakeExc(Exception):
        pass
    exc = FakeExc('Invalid json output: {"thought": "x"}')
    raw = extract_llm_raw_text(exc)
    print('Test6 (extract_llm_raw_text):', raw)
    assert raw == '{"thought": "x"}'

    print('\nAll 6 tests passed!')


if __name__ == "__main__":
    test_safe_parse_structured()
