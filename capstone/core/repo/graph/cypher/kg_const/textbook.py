CYPHER_write_sections = """
UNWIND $sections AS s
MERGE (n:Section {id: s.id})
SET n.title=s.title, n.level=s.level, n.p_lo=s.p_num[0], n.p_hi=s.p_num[1], n.order=s.order
WITH n, s WHERE s.parent_id IS NOT NULL
MATCH (p:Section {id: s.parent_id})
MERGE (p)-[:CONTAINS]->(n)
"""

CYPHER_write_passages = """
UNWIND $passages AS p
MERGE (n:Passage {id: p.id})
SET n.p_lo=p.p_num[0], n.p_hi=p.p_num[1], n.text=p.text, n.emb=p.emb,
    n.uri=coalesce(p.uri, $uri)
WITH n, p
MATCH (s:Section {id: p.section_id})
MERGE (s)-[:HAS_PASSAGE]->(n)
"""

CYPHER_get_toc_scoped = """
MATCH (s:Section)-[:HAS_PASSAGE]->(p:Passage)
WHERE p.uri STARTS WITH $hint
WITH DISTINCT s
OPTIONAL MATCH (parent:Section)-[:CONTAINS]->(s)
RETURN s.id AS id, s.title AS title, s.level AS level,
       s.p_lo AS p_lo, s.p_hi AS p_hi, s.order AS order, parent.id AS parent_id
ORDER BY s.level, s.order
"""

CYPHER_get_toc = """
MATCH (s:Section)
OPTIONAL MATCH (parent:Section)-[:CONTAINS]->(s)
RETURN s.id AS id, s.title AS title, s.level AS level,
       s.p_lo AS p_lo, s.p_hi AS p_hi, s.order AS order, parent.id AS parent_id
ORDER BY s.level, s.order
"""

CYPHER_list_passage_uris = """
MATCH (p:Passage)
WHERE p.uri STARTS WITH $prefix
RETURN p.uri AS uri
"""
