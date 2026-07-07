CYPHER_entity_finder = """
MATCH (n:Entity)
WHERE n.id = $wiki_id
   OR toLower(n.name) = toLower($query)
   OR n.id = $query
RETURN n.id AS id, n.name AS name LIMIT 1
"""

CYPHER_rhetorical_retriever = """
MATCH (n:Entity {id: $id})-[:CONTENT]->(c:Entity)
RETURN c.rrole AS role, c.content AS content LIMIT $limit
"""

CYPHER_rhetorical_retriever_role = """
MATCH (n:Entity {id: $id})-[:CONTENT]->(c:Entity)
WHERE toLower(c.rrole) = toLower($role)
RETURN c.rrole AS role, c.content AS content LIMIT $limit
"""

CYPHER_edge_explorer = """
MATCH (n:Entity {id: $id})
OPTIONAL MATCH path = (n)-[r*1..2]-(m:Entity)
WHERE ALL(rel IN relationships(path) WHERE type(rel) <> 'CONTENT')
  AND m.id <> $id AND m.rrole IS NULL
WITH n, m, path
LIMIT 20
RETURN
    m.name as name,
    m.id as id,
    type(relationships(path)[0]) as rel_type,
    CASE WHEN startNode(relationships(path)[0]) = n THEN 'OUTGOING' ELSE 'INCOMING' END as direction,
    length(path) as distance
"""
