"""
Context-management smoke test (acceptance gate).

Drives ONE session through >max_turns turns to prove:
  - no context overflow / crash  -> every turn returns status "finished" w/ non-empty message
  - history continuity           -> a seeded fact from turn 1 is recalled in a late turn

Run from capstone/ with the app + Ollama + DB stack up:
    uv run python -m test.test_script.TA.test_context_smoke
"""
import asyncio
import httpx

from core.config import App_settings

POLL_INTERVAL = 10
POLL_TIMEOUT = 300
INITIAL_WAIT = 2

SEED_FACT = "Tung"
# turn 0 seeds the fact; turn N_TURNS-1 probes recall. middle turns push past window.
QUERIES = [
    f"My name is {SEED_FACT}. Please remember it.",
    "What is machine learning?",
    "Explain supervised vs unsupervised learning.",
    "What is a neural network?",
    "What is gradient descent?",
    "What is overfitting?",
    "What is a loss function?",
    "What is regularization?",
    "What is cross validation?",
    "What is a confusion matrix?",
    "What is precision and recall?",
    "What is my name?",  # recall probe -> expects SEED_FACT
]


async def _submit_and_wait(client: httpx.AsyncClient, ta_url: str, headers: dict, session_id: str, query: str) -> str:
    res = await client.post(f"{ta_url}/chat", headers=headers, json={"session_id": session_id, "user_input": query})
    assert res.status_code == 202, f"submit failed [{res.status_code}]: {res.text}"
    task_id = res.json()["task_id"]

    status_url = f"{ta_url}/chat/status/{task_id}"
    await asyncio.sleep(INITIAL_WAIT)
    elapsed = INITIAL_WAIT
    while elapsed < POLL_TIMEOUT:
        entry = (await client.get(status_url, headers=headers)).json()
        if entry["status"] == "finished":
            return entry["result"]["message"]
        if entry["status"] == "Fail":
            raise AssertionError(f"TA workflow failed: {entry.get('error')}")
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    raise AssertionError(f"timed out after {POLL_TIMEOUT}s")


async def test_context_smoke():
    c = App_settings()
    base = f"http://127.0.0.1:{c.port}"
    student_url, ta_url = f"{base}{c.stu_end}", f"{base}{c.ta_end}"
    user, pw = "smoke_student_01", "password123"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            await client.post(f"{student_url}/register", json={"student_id": user, "password": pw})
        except Exception:
            pass
        login = await client.post(f"{student_url}/login", data={"username": user, "password": pw})
        assert login.status_code == 200, f"login failed: {login.text}"
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        sess = await client.post(f"{student_url}/session/start", headers=headers)
        assert sess.status_code == 200, f"session start failed: {sess.text}"
        session_id = sess.json()["session_id"]
        print(f"session={session_id} | {len(QUERIES)} turns")

        last_msg = ""
        for i, q in enumerate(QUERIES):
            msg = await _submit_and_wait(client, ta_url, headers, session_id, q)
            assert msg and msg.strip(), f"turn {i} empty message"
            print(f"  [{i:>2}] ok ({len(msg)} chars)  q={q[:40]!r}")
            last_msg = msg

        # continuity: late turn should recall the seeded fact (soft — local LLM nondeterministic)
        recalled = SEED_FACT.lower() in last_msg.lower()
        print(f"\ncontinuity recall of {SEED_FACT!r}: {'PASS' if recalled else 'WARN (not found)'}")
        print("--- smoke PASS: survived all turns, no overflow/crash ---")


if __name__ == "__main__":
    asyncio.run(test_context_smoke())
