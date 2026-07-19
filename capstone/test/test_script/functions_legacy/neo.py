
from core.repo.graph.test.doc import DOC
from core.repo.graph.graphdb import GraphDB

def test_query():
    from TA.tools.neo import EntityFinder, RhetoricalRetriever, EdgeExplorer
    db = GraphDB()

    finder = EntityFinder(engine=db)
    retriever = RhetoricalRetriever(engine=db)
    explorer = EdgeExplorer(engine=db)

    test_cases = [
        {"input": "Machine Learning", "role": "Definition"},
        {"input": "Supervised Learning", "role": "Definition"},
        {"input": "Bias Variance Tradeoff", "role": "Statement"}
    ]

    for case in test_cases:
        print(f"\n{'='*20} Testing: {case['input']} {'='*20}")
        
        # 1. Định danh thực thể (Wiki Bridge / Fallback)
        find_res = finder._run(case['input'])
        print(f"[Step 1 - Finder]: {find_res}")

        if isinstance(find_res, dict) and find_res.get("status") == "SUCCESS":
            node_id = find_res["data"]["id"]
            
            # 2. Lấy nội dung chi tiết (Rhetorical Roles)
            content_res = retriever._run(node_id=node_id, role=case['role'])
            print(f"[Step 2 - Retriever]: {content_res}")

            # 3. Khám phá các cạnh (Edges/Relationships)
            print(f"[Step 3 - Explorer]: Đang tìm các khái niệm liên quan cho ID {node_id}...")
            edge_res = explorer._run(node_id=node_id)
            print(f"{edge_res}")
            
        else:
            print("[Step 1] Failed to resolve entity.")

    db.close()