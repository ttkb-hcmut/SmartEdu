CYPHER_optimal_path = """
MATCH (start:Entity {name: $start_node}), (end:Entity {name: $end_node})
CALL apoc.algo.dijkstra(start, end, '', 'weight') YIELD path, weight
UNWIND nodes(path) AS n
RETURN n.name AS name, n.typeNode AS type, n.content AS content
"""

CYPHER_optimal_path_fallback = """
MATCH (start:Entity {name: $start_node}), (end:Entity {name: $end_node}),
      path = shortestPath((start)-[*..10]-(end))
WHERE ALL(r IN relationships(path) WHERE type(r) <> 'CONTENT')
UNWIND nodes(path) AS n
WHERE n.rrole IS NULL
RETURN n.name AS name, n.typeNode AS type, n.content AS content
"""
