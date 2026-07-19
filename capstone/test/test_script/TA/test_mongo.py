from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv("core/.env")

user = os.getenv("MONGO_USER", "admin")
passw = os.getenv("MONGO_PASS", "password123")
host = os.getenv("MONGO_HOST", "localhost:27017")

print(f"Loaded credentials: user='{user}', passw='{passw}', host='{host}'")

uri = f"mongodb://{user}:{passw}@{host}/?authSource=admin"
print(f"Connecting to: {uri}")

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=2000)
    info = client.server_info()
    print("Successfully connected to MongoDB!")
    print(info)
except Exception as e:
    print(f"Connection failed: {e}")
