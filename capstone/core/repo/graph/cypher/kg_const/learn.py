from core.util.cypher import concept_pred


def CYPHER_get_learning_graph() -> str:
    ## fn not const: body interpolates concept_pred, keep it single source of truth
    return f"""
    MATCH (n:Entity)
    WHERE {concept_pred('n')}
    OPTIONAL MATCH (n)-[r]->(m:Entity)
    WHERE type(r) <> 'CONTENT'
    WITH n, count(r) AS out_degree
    OPTIONAL MATCH (n)-[:BELONGS_TO|PART_OF*1..2]->(c:Entity)
    WHERE c.typeNode = 'Community'
    OPTIONAL MATCH (s:Student {{id: $sid}})-[mas:MASTERY]->(n)
    RETURN n.name AS name,
           n.typeNode AS type,
           c.name AS course_name,
           out_degree,
           substring(coalesce(n.content, ''), 0, 50) AS description,
           coalesce(mas.level, 0) AS mastery
    """


CYPHER_update_learn_mastery = """
MERGE (s:Student {id: $sid})
MERGE (n:Entity {name: $new_name})
MERGE (s)-[r:MASTERY]->(n)
SET r.level = CASE WHEN coalesce(r.level, 0) < 6 THEN coalesce(r.level, 0) + 1 ELSE 6 END,
    r.last_visited = datetime()
"""

CYPHER_update_learn_transition = """
MERGE (s:Student {id: $sid})
WITH s
MATCH (curr:Entity {name: $curr_name}), (next:Entity {name: $new_name})
MERGE (s)-[r:LEARNED_PATH]->(next)
SET r.from_node = $curr_name,
    r.count = coalesce(r.count, 0) + 1,
    r.last_visited = datetime()
"""

CYPHER_get_mastery = """
MATCH (s:Student {id: $sid})-[r:MASTERY]->(n:Entity {name: $name})
RETURN coalesce(r.level, 0) AS mastery
"""

CYPHER_delete_student = """
MATCH (s:Student {id: $sid})
DETACH DELETE s
"""
