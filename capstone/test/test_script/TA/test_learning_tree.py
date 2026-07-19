from core.repo.nosql.mongo_db import Mongo_DB

db = Mongo_DB()
db.learning_trees.delete_one({"_id": "t1::TestCourse"})
db.seed_learning_tree("t1", "TestCourse", [{"name": "A", "topic": "T"}, {"name": "B", "topic": "T"}])
db.update_node_mastery("t1", "TestCourse", "A", 5)
db.add_learning_node("t1", "TestCourse", {"name": "C"})
db.seed_learning_tree("t1", "TestCourse", [{"name": "A"}, {"name": "B"}])
db.append_attempt("t1", "TestCourse", {"concept": "A", "score": 5, "exercise": "q1"})
doc = db.get_learning_tree("t1", "TestCourse")
by_name = {n["name"]: n for n in doc["nodes"]}
assert by_name["A"]["mastery"] == 5, "re-seed must keep mastery"
assert by_name["C"]["discovered"] is True, "discovered node must survive re-seed"
assert len(doc["attempt_log"]) == 1
db.learning_trees.delete_one({"_id": "t1::TestCourse"})
print("learning_tree OK")
