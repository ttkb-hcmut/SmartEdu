import httpx
import asyncio
import datetime
import json
import os

from core.config import App_settings

POLL_INTERVAL = 15      # seconds between status polls after initial wait
POLL_TIMEOUT  = 300    # max seconds to wait for TA response
INITIAL_WAIT  = 2      # wait time before first poll

async def test_ta_logic(query: str, name="TA_v0", path="test/TA", filename=None):
    c = App_settings()
    base_url = f"http://127.0.0.1:{c.port}"
    student_url = f"{base_url}{c.stu_end}"
    ta_url = f"{base_url}{c.ta_end}"

    test_user = "test_student_01"
    test_pass = "password123"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Try to register (might fail if already exists, that's fine)
        try:
            await client.post(f"{student_url}/register", json={
                "student_id": test_user,
                "password": test_pass
            })
        except Exception:
            pass
        print("Create account successfully")

        # 2. Login
        login_res = await client.post(f"{student_url}/login", data={
            "username": test_user,
            "password": test_pass
        })
        if login_res.status_code != 200:
            print(f"Login failed: {login_res.text}")
            return
        else:
            print("Login Successfull")
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 3. Start Session
        session_res = await client.post(f"{student_url}/session/start", headers=headers)
        if session_res.status_code != 200:
            print(f"Session start failed: {session_res.text}")
            return

        session_id = session_res.json()["session_id"]

        # 4. Fire-and-forget POST /chat → receive task_id immediately
        print(f"Submitting query to {ta_url}/chat...")
        chat_res = await client.post(f"{ta_url}/chat", headers=headers, json={
            "session_id": session_id,
            "user_input": query
        })

        if chat_res.status_code != 202:
            print(f"Chat submission failed [{chat_res.status_code}]: {chat_res.text}")
            return

        accepted = chat_res.json()
        task_id = accepted["task_id"]
        print(f"Task accepted — id={task_id}, status={accepted['status']}, agent={accepted['agent']}")

        # 5. Poll GET /chat/status/{task_id} until done or timeout
        status_url = f"{ta_url}/chat/status/{task_id}"
        elapsed = INITIAL_WAIT
        result = None
        
        print(f"Waiting {INITIAL_WAIT} seconds before first status check...")
        await asyncio.sleep(INITIAL_WAIT)
        
        while elapsed < POLL_TIMEOUT:
            status_res = await client.get(status_url, headers=headers)
            if status_res.status_code != 200:
                print(f"Status poll error [{status_res.status_code}]: {status_res.text}")
                return

            entry = status_res.json()
            status = entry["status"]
            agent_name = entry.get("agent_name", "N/A")
            intent = entry.get("intent", "N/A")
            print(f"  [{elapsed:>3}s] status={status} | agent={agent_name} | intent={intent}")

            if status == "finished":
                result = entry["result"]
                break
            if status == "Fail":
                print(f"TA workflow error: {entry.get('error')}")
                return

            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

        if result is None:
            print(f"Timed out after {POLL_TIMEOUT}s waiting for TA response.")
            return

        response = result["message"]
        ui_action = result.get("ui_action")

        # Prepare output data
        output_data = {
            "query": query,
            "message": response,
            "ui_action": ui_action,
            "timestamp": datetime.datetime.now().isoformat()
        }

        if not filename:
            now = datetime.datetime.now()
            filename = f"{name}_{now.strftime('%H%M%S')}_{now.strftime('%d%m')}.json"
            
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        
        existing_data = []
        if os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        existing_data = json.loads(content)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
            except Exception:
                existing_data = []
                
        existing_data.append(output_data)
        
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

        print(f"\n--- TA Response ---\n{response}")
        if ui_action:
            print(f"UI Action: {json.dumps(ui_action, indent=2, ensure_ascii=False)}")
        print(f"\nResult saved to: {full_path}")

if __name__ == "__main__":
    asyncio.run(test_ta_logic(query="What is Machine Learning?"))
