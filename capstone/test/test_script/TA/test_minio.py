import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.repo.storage.minio_repo import MinioDB
from core.config import Minio_conf

def test_minio_insertion():
    print("Initializing MinioDB...")
    try:
        conf = Minio_conf()
        print(f"Connecting to Minio at {conf.endpoint}...")
        storage = MinioDB(config=conf)
        
        test_content = b"This is a test file for Minio insertion."
        test_filename = "test_file.pdf"
        course_name = "test_course"
        
        print(f"Attempting to upload {test_filename} to bucket '{storage.bucket_name}'...")
        uri = storage.upload_slide(test_filename, course_name, test_content)
        
        if uri:
            print(f"SUCCESS! File uploaded. URI: {uri}")
        else:
            print("FAILED! Upload returned None.")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    test_minio_insertion()
