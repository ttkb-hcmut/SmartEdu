import asyncio
from typing import Any
from TA.tools.neo.retriever import EntityFinder, RhetoricalRetriever, EdgeExplorer
from core.repo.graph.graphdb import GraphDB
from core.config import Neo

def test_finder(g_db: GraphDB):
    print("\n=== Testing EntityFinder ===")
    finder = EntityFinder(engine=g_db)
    res_wiki = finder.run("Reinforcement Learning")
    print(f"Result (Standard): {res_wiki}")
    
    res_local = finder.run("ML")
    print(f"Result (Local/Fallback): {res_local}")
    
    res_fail = finder.run("Supervised Learning")
    print(f"Result: {res_fail}")

def test_rhetorical(g_db: GraphDB, test_id: str):
    print("\n=== Testing RhetoricalRetriever ===")
    retriever = RhetoricalRetriever(engine=g_db)
    res_def = retriever.run({"node_id": test_id, "role": "definition"})
    print(f"Definition: {res_def}")
    res_ex = retriever.run({"node_id": test_id, "role": "example"})
    print(f"Example: {res_ex}")

def test_edges(g_db: GraphDB, test_id: str):
    print("\n=== Testing EdgeExplorer ===")
    explorer = EdgeExplorer(engine=g_db)
    res_edges = explorer.run(test_id)
    print(f"Relationships for {test_id}:\n{res_edges}")

def full():
    g_db = GraphDB(config=Neo()) 
    try:
        test_finder(g_db)
        
        q = "MATCH (n:Entity) WHERE n.name = $name RETURN n.id AS id LIMIT 1"
        sample = g_db.run_query(g_db.db_name, q, {"name": "Machine Learning"})
        
        if sample:
            real_id = sample[0]['id']
            print(f"\nUsing Real ID for further testing: {real_id}")
            test_rhetorical(g_db, real_id)
            test_edges(g_db, real_id)
        else:
            print("Graph is empty, cannot test Retriever and Explorer.")
    finally:
        g_db.close()

if __name__ == "__main__":
    full()