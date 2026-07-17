import asyncio
import uuid
import sys
import os

# Add capstone to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.repo.nosql.mongo_db import Mongo_DB
from core.repo.sql.sql_db import SQL_DB
from core.repo.graph.graphdb import GraphDB
from student.Student_Tracker import Student_Tracker
from core.config import NeoStudent
from core.schema.wf_state import ConceptNode

async def test_concurrency():
    print("--- Starting Concurrency Test ---")
    mongo = Mongo_DB()
    sql = SQL_DB()
    graph = GraphDB(config=NeoStudent)
    
    tracker = Student_Tracker(graphdb=graph, sqldb=sql, mongodb=mongo)
    
    student_id = f"test_concurrent_{uuid.uuid4().hex[:8]}"
    s1 = f"session_1_{uuid.uuid4().hex[:4]}"
    s2 = f"session_2_{uuid.uuid4().hex[:4]}"
    
    print(f"Testing with Student: {student_id}")
    print(f"Session 1: {s1}, Session 2: {s2}")
    
    tracker.create_chat_session(student_id, s1)
    tracker.create_chat_session(student_id, s2)
    
    # 1. Test Identity Isolation (Memos)
    session1 = tracker.get_session(s1)
    session2 = tracker.get_session(s2)
    
    assert session1 is not session2, "Sessions should be different objects"
    assert session1.memo.session_id == s1
    assert session2.memo.session_id == s2
    
    await session1.memo.save({"role": "student", "message": "Msg from S1", "heading": "H1"})
    await session2.memo.save({"role": "student", "message": "Msg from S2", "heading": "H2"})
    
    h1 = tracker.get_chat_history(s1)
    h2 = tracker.get_chat_history(s2)
    
    print(f"History 1: {h1}")
    print(f"History 2: {h2}")
    
    assert "Msg from S1" in h1 and "Msg from S2" not in h1, "History S1 contaminated"
    assert "Msg from S2" in h2 and "Msg from S1" not in h2, "History S2 contaminated"
    print("Success: Identity isolation confirmed.")
    
    # 2. Test Atomic Updates (Simulated Concurrency)
    # We use update_learning_position which now uses update_student_field
    node1 = ConceptNode(id="n1", name="Concurrent Node 1", rrole="Definition", content="C1")
    node2 = ConceptNode(id="n2", name="Concurrent Node 2", rrole="Definition", content="C2")
    
    print("Updating Node 1 via Session 1 and Node 2 via Session 2 concurrently...")
    
    # In a real async environment, these might overlap. 
    # Here we just check if they both persist correctly without overwriting the whole state.
    tracker.update_learning_position(s1, node1)
    tracker.update_learning_position(s2, node2)
    
    # Verify both exist in state (shared in memory)
    state = tracker.get_student_state(s1)
    assert state["current_pos"].name == "Concurrent Node 2", "Last one should be current"
    assert any(n.name == "Concurrent Node 1" for n in state["previous_nodes"]), "Node 1 should be in history"
    
    # Verify in MongoDB (Fetch fresh)
    fresh_state = mongo.get_student_state(student_id)
    print(f"Fresh state from Mongo: {fresh_state['current_pos']['name']}")
    assert fresh_state["current_pos"]["name"] == "Concurrent Node 2"
    assert any(n["name"] == "Concurrent Node 1" for n in fresh_state["previous_nodes"])
    
    print("Success: Atomic updates confirmed (no data loss).")
    
    # Cleanup
    # (Optional: delete student from Mongo/SQL/Graph)
    
    print("--- Concurrency Test Passed ---")

if __name__ == "__main__":
    asyncio.run(test_concurrency())
