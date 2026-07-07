CYPHER_recommend_new_from_node = """
MATCH (src:Entity {name: $from_node})-[r]->(m:Entity)
WHERE type(r) <> 'CONTENT' AND m.rrole IS NULL
OPTIONAL MATCH (m)-[r2]->(other:Entity)
WHERE type(r2) <> 'CONTENT' AND other.rrole IS NULL
WITH m, sum(CASE WHEN type(r2) = 'PREREQUISITE' THEN $prereq_weight ELSE 1.0 END) AS semantic_out_degree
ORDER BY semantic_out_degree DESC
LIMIT $max_results
OPTIONAL MATCH (m)-[:BELONGS_TO|PART_OF*1..2]->(c:Entity)
WHERE c.typeNode = 'Community'
RETURN m.name AS name, m.content AS content,
       m.typeNode AS type, semantic_out_degree AS out_degree,
       c.name AS course_name
"""

CYPHER_recommend_new_course = """
MATCH (course:Entity {name: $course_filter})
MATCH (n:Entity)-[:BELONGS_TO|PART_OF*1..2]->(course)
WHERE n.rrole IS NULL AND n.id <> course.id
OPTIONAL MATCH (n)-[r]->(m:Entity)
WHERE type(r) <> 'CONTENT' AND m.rrole IS NULL
WITH n, course, sum(CASE WHEN type(r) = 'PREREQUISITE' THEN $prereq_weight ELSE 1.0 END) AS semantic_out_degree
ORDER BY semantic_out_degree DESC
LIMIT $max_results
RETURN n.name AS name, n.content AS content,
       n.typeNode AS type, semantic_out_degree AS out_degree,
       course.name AS course_name
"""

CYPHER_recommend_new = """
MATCH (n:Entity)
WHERE n.rrole IS NULL
OPTIONAL MATCH (n)-[r]->(m:Entity)
WHERE type(r) <> 'CONTENT' AND m.rrole IS NULL
WITH n, sum(CASE WHEN type(r) = 'PREREQUISITE' THEN $prereq_weight ELSE 1.0 END) AS semantic_out_degree
ORDER BY semantic_out_degree DESC
LIMIT $max_results
OPTIONAL MATCH (n)-[:BELONGS_TO|PART_OF*1..2]->(c:Entity)
WHERE c.typeNode = 'Community'
RETURN n.name AS name, n.content AS content,
       n.typeNode AS type, semantic_out_degree AS out_degree,
       c.name AS course_name
"""

CYPHER_course_backbone_hub = """
MATCH (course:Entity {name: $course_name})
MATCH (n:Entity)-[:BELONGS_TO|PART_OF*1..2]->(course)
WHERE n.rrole IS NULL AND n.id <> course.id
OPTIONAL MATCH (n)-[r]->(m:Entity)
WHERE type(r) <> 'CONTENT' AND m.rrole IS NULL
WITH n, course, sum(CASE WHEN type(r) = 'PREREQUISITE' THEN $prereq_weight ELSE 1.0 END) AS out_degree
ORDER BY out_degree DESC
LIMIT $max_hubs
RETURN n.id AS id, n.name AS name, n.content AS content,
       n.typeNode AS type, out_degree, course.name AS course_name
"""

CYPHER_course_backbone_rel = """
UNWIND $hub_ids AS h1_id
UNWIND $hub_ids AS h2_id
WITH h1_id, h2_id WHERE h1_id < h2_id
MATCH (h1:Entity {id: h1_id})-[r]-(h2:Entity {id: h2_id})
WHERE type(r) <> 'CONTENT'
RETURN h1.name AS from_node, h2.name AS to_node, type(r) AS relationship,
       CASE WHEN startNode(r) = h1 THEN 'FORWARD' ELSE 'REVERSE' END AS direction
"""

CYPHER_course_relevance = """
MATCH (target_course:Entity {name: $target_course})
MATCH (inner:Entity)-[:BELONGS_TO|PART_OF*1..2]->(target_course)
WHERE inner.rrole IS NULL

OPTIONAL MATCH (inner)-[ri]->(mi:Entity)
WHERE type(ri) <> 'CONTENT' AND mi.rrole IS NULL
WITH target_course, inner, count(ri) AS inner_degree
WHERE inner_degree >= $min_degree

MATCH (outer:Entity)-[r]->(inner)
WHERE outer.rrole IS NULL AND type(r) <> 'CONTENT'
  AND NOT (outer)-[:BELONGS_TO|PART_OF*1..2]->(target_course)

OPTIONAL MATCH (outer)-[ro]->(mo:Entity)
WHERE type(ro) <> 'CONTENT' AND mo.rrole IS NULL
WITH target_course, inner, outer, count(ro) AS outer_degree
WHERE outer_degree >= $min_degree

OPTIONAL MATCH (outer)-[:BELONGS_TO|PART_OF*1..2]->(other_course:Entity)
WHERE other_course.typeNode = 'Community'

RETURN other_course.name AS related_course,
       count(DISTINCT outer) AS hub_overlap,
       collect(DISTINCT outer.name)[..3] AS key_concepts
ORDER BY hub_overlap DESC
LIMIT 10
"""

CYPHER_course_tree_concept = """
MATCH (course:Entity {name: $course_name})
MATCH (n:Entity)-[:BELONGS_TO|PART_OF*1..2]->(course)
WHERE n.rrole IS NULL AND n.id <> course.id AND n.typeNode <> 'Topic'
OPTIONAL MATCH (n)-[r]->(m:Entity)
WHERE type(r) <> 'CONTENT' AND m.rrole IS NULL
WITH course, n,
     sum(CASE WHEN type(r) = 'PREREQUISITE' THEN $w ELSE 1.0 END) AS out_degree
OPTIONAL MATCH (n)-[:BELONGS_TO|PART_OF]->(t:Entity {typeNode: 'Topic'})
WHERE (t)-[:BELONGS_TO|PART_OF*0..1]->(course)
RETURN n.name AS name, n.typeNode AS type, n.content AS content,
       coalesce(n.closeness, out_degree) AS score, t.name AS topic
"""

CYPHER_course_tree_prereq = """
MATCH (a:Entity)-[:PREREQUISITE]->(b:Entity)
WHERE a.name IN $names AND b.name IN $names
RETURN a.name AS pre, b.name AS post
"""

CYPHER_get_recommendations = """
MATCH (n:Entity {name: $name})-[r]->(m:Entity)
WHERE type(r) <> 'CONTENT' AND m.rrole IS NULL
OPTIONAL MATCH (m)-[r2]->(other:Entity)
WHERE type(r2) <> 'CONTENT' AND other.rrole IS NULL
WITH m, sum(CASE WHEN type(r2) = 'PREREQUISITE' THEN $prereq_weight ELSE 1.0 END) AS out_degree
ORDER BY out_degree DESC
LIMIT 10
RETURN m.name AS name, m.content AS content,
       m.typeNode AS type, out_degree
"""
