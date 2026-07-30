from unittest.mock import Mock

from student.Student_Tracker import Student_Tracker


def test_delete_student_cleans_memory_and_all_repositories():
    tracker = Student_Tracker.__new__(Student_Tracker)
    tracker.graphdb = Mock()
    tracker.sqldb = Mock()
    tracker.mongodb = Mock()
    tracker._student_states = {"bench": {"summary": "x"}, "other": {}}
    tracker._session_map = {"bench-1": "bench", "other-1": "other"}
    tracker._sessions = {"bench-1": object(), "other-1": object()}

    tracker.delete_student("bench")

    assert "bench" not in tracker._student_states
    assert "bench-1" not in tracker._session_map
    assert "bench-1" not in tracker._sessions
    tracker.graphdb.delete_student.assert_called_once_with("bench")
    tracker.mongodb.delete_student.assert_called_once_with("bench")
    tracker.sqldb.delete_student.assert_called_once_with("bench")
