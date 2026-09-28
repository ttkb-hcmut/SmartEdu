import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add capstone to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from main import app
from core.config import App_settings
from core.repo.nosql.mongo_db import Mongo_DB

def test_reset_student_state():
    c = App_settings()
    with TestClient(app) as client:
        # Setup a dummy student in MongoDB
        student_id = "test_reset_student"
        tracker = app.state.student_tracker
        mongo = tracker.mongodb
        
        # Make sure we have a fresh state
        mongo.students.delete_one({"_id": student_id})
        mongo.create_student(student_id)
        
        # Modify the state to simulate progress
        mongo.update_student_field(student_id, "state.summary", "Some progress here")
        
        # Verify it was modified
        state_before = mongo.get_student_state(student_id)
        assert state_before.get("summary") == "Some progress here"
        
        # Hit the reset endpoint
        response = client.post(f"{c.stu_end}/admin/reset", json={"student_id": student_id})
        assert response.status_code == 200
        assert "reset successfully" in response.json()["detail"]
        
        # Verify it was reset to default
        state_after = mongo.get_student_state(student_id)
        assert state_after.get("summary") in ("Student just started", "")
        
        # Cleanup
        mongo.students.delete_one({"_id": student_id})
    
if __name__ == "__main__":
    pytest.main(["-v", __file__])
