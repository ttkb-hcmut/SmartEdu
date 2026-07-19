import httpx
import asyncio
import json

from core.config import App_settings

async def run_ingestion_test(course_name, slide_files, textbook_files):

    c = App_settings()
    # Ensure your uvicorn server is running on this port
    url = f"http://127.0.0.1:{c.port}{c.kg_end}/ingest-course"
    
    # This payload matches your CourseIngestionRequest model
    payload = {
        "course_name": course_name,
        "slide_files": slide_files,
        "textbook_files":textbook_files,
        "reset": True
            }

    print(f"Sending request to {url}...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                print("Status: SUCCESS")
                print("Response Data:")
                print(json.dumps(response.json(), indent=4))
                print("\nVerification Note:")
                print("The task is running in the background. Check your server logs")
                print("to see the Graph construction progress and final node count.")
            else:
                print(f"Status: FAILED (Code {response.status_code})")
                print(f"Error Detail: {response.text}")
                
        except httpx.ConnectError:
            print("Connection Error: Is your FastAPI server running?")

if __name__ == "__main__":
    asyncio.run(run_ingestion_test())