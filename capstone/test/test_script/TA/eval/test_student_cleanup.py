import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from student.Student_Tracker import Student_Tracker
from student import api as student_api
from core.repo.nosql.mongo_db import Mongo_DB


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


def test_session_list_filters_orders_and_paginates():
    mongo = Mongo_DB.__new__(Mongo_DB)
    mongo.students = Mock()
    mongo.students.find_one.return_value = {"memo": [
        {"id": "old", "name": "New Session", "chats": [{"id": "c1", "invoke": "Vectors", "messages": [{"role": "student", "timestamp": "2025-01-01"}]}]},
        {"id": "new", "name": "Practice", "chats": [{"id": "c2", "invoke": "Matrices", "messages": [{"role": "TA", "timestamp": "2025-02-01"}]}]},
    ]}
    assert [item["id"] for item in mongo.list_sessions("alice")["items"]] == ["new", "old"]
    assert mongo.list_sessions("alice", query="vectors")["items"][0]["name"] == "Vectors"
    assert mongo.list_sessions("alice", limit=1, offset=1)["items"][0]["id"] == "old"
    mongo.students.find_one.assert_called_with({"_id": "alice"}, {"memo": 1})


def test_session_read_and_resume_enforce_owner_and_hide_traces():
    session = {"id": "mine", "chats": [{"id": "c", "messages": [
        {"role": "student", "message": "Question"},
        {"role": "tool", "message": "secret trace"},
        {"role": "TA", "message": "Answer"},
    ], "ui_action": {"citations": [{"uri": "u", "document": None, "page": None}]}}]}
    mongo = Mock()
    mongo.get_session_data.side_effect = lambda owner, sid: session if (owner, sid) == ("alice", "mine") else {}
    tracker = Mock(mongodb=mongo)
    tracker.get_student_id_by_session.return_value = None
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(student_tracker=tracker)))
    user = SimpleNamespace(id="alice")

    data = asyncio.run(student_api.read_session("mine", request, user))
    assert [m["role"] for m in data["messages"]] == ["user", "ta"]
    assert data["messages"][-1]["ui_action"]["citations"][0]["uri"] == "u"
    asyncio.run(student_api.resume_session("mine", request, user))
    tracker.create_chat_session.assert_called_once_with("alice", "mine")
    with pytest.raises(HTTPException) as error:
        asyncio.run(student_api.read_session("mine", request, SimpleNamespace(id="bob")))
    assert error.value.status_code == 404


def test_session_read_collapses_replayed_ta_completion_and_keeps_citation():
    citation = {"uri": "passage-12", "document": "Machine Learning/_raw/ch1.pdf", "page": 12}
    session = {"id": "mine", "chats": [{"id": "c", "ui_action": {"citations": [citation]}, "messages": [
        {"role": "student", "message": "Question"},
        {"role": "TA", "message": "Answer"},
        {"role": "TA", "message": "Answer"},
    ]}]}
    mongo = Mock()
    mongo.get_session_data.return_value = session
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(student_tracker=SimpleNamespace(mongodb=mongo))))

    data = asyncio.run(student_api.read_session("mine", request, SimpleNamespace(id="alice")))

    assert [(message["role"], message["content"]) for message in data["messages"]] == [
        ("user", "Question"), ("ta", "Answer"),
    ]
    assert data["messages"][-1]["ui_action"]["citations"] == [citation]


def test_chat_message_append_is_atomic_and_role_idempotent():
    mongo = Mongo_DB.__new__(Mongo_DB)
    mongo.students = Mock()

    mongo.push_chat_message("alice", "session", "chat", {"role": "TA", "message": "Answer"})

    query, update = mongo.students.update_one.call_args.args
    chat_match = query["memo"]["$elemMatch"]["chats"]["$elemMatch"]
    assert chat_match["messages"] == {"$not": {"$elemMatch": {"role": "TA"}}}
    assert update["$push"]["memo.$[s].chats.$[c].messages"]["message"] == "Answer"
    assert mongo.students.update_one.call_args.kwargs["array_filters"] == [{"s.id": "session"}, {"c.id": "chat"}]


def test_pending_chat_is_exposed_for_status_resume():
    mongo = Mock()
    mongo.get_session_data.return_value = {"id": "s", "chats": [{"id": "task", "messages": [{"role": "student", "message": "Q"}]}]}
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(student_tracker=SimpleNamespace(mongodb=mongo))))
    data = asyncio.run(student_api.read_session("s", request, SimpleNamespace(id="alice")))
    assert data["pending_chat_id"] == "task"


def test_ui_action_is_written_to_exact_chat():
    mongo = Mongo_DB.__new__(Mongo_DB)
    mongo.students = Mock()
    action = {"citations": [{"uri": "source", "document": None, "page": None}]}
    mongo.set_chat_ui_action("alice", "session", "chat", action)
    _, update = mongo.students.update_one.call_args.args
    assert update == {"$set": {"memo.$[s].chats.$[c].ui_action": action}}
    assert mongo.students.update_one.call_args.kwargs["array_filters"] == [{"s.id": "session"}, {"c.id": "chat"}]
