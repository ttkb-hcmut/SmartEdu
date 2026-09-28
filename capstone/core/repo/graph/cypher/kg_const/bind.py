CYPHER_update_links = """
MERGE (c:TextbookChunk {id: $chunk_id})
SET c.heading = $heading, c.storage_uri = $storage_uri
WITH c
UNWIND $links AS link
MATCH (anchor {id: link.anchor_id})
MERGE (c)-[r:ELABORATES_ON]->(anchor)
SET r.justification = link.justification
"""

CYPHER_write_anchors = """
UNWIND $links AS l
MATCH (e:Entity {name: l.entity_name})
MATCH (p:Passage {id: l.passage_id})
MERGE (e)-[r:ANCHORED_IN]->(p)
SET r.score=l.score, r.justification=coalesce(l.justification, '')
"""

CYPHER_write_seg_anchors = """
UNWIND $links AS l
MATCH (e:Entity {name: l.entity_name})
MATCH (s:Segment {id: l.segment_id})
MERGE (e)-[r:ANCHORED_IN]->(s)
SET r.score=l.score
"""

CYPHER_get_concept_page = """
MATCH (e:Entity)-[r:ANCHORED_IN]->(p:Passage)
WHERE toLower(e.name) CONTAINS toLower($name)
RETURN p.uri AS uri, p.p_lo AS page, r.score AS score, e.name AS concept
ORDER BY r.score DESC LIMIT 1
"""

CYPHER_get_concept_anchors = """
MATCH (e:Entity)-[r:ANCHORED_IN]->(p:Passage)
WHERE toLower(e.name) CONTAINS toLower($name)
RETURN e.name AS concept, p.uri AS uri, p.p_lo AS p_lo, p.p_hi AS p_hi,
       r.score AS score, substring(p.text, 0, 160) AS preview
ORDER BY r.score DESC LIMIT 10
"""

CYPHER_anchor_search = """
CALL db.index.vector.queryNodes('passage_vec_index', $probe, $emb) YIELD node, score
WHERE $prefix IS NULL OR node.uri STARTS WITH $prefix
RETURN node.id AS passage_id, score
LIMIT $k
"""

CYPHER_anchor_search_scoped = """
MATCH (node:Passage)
WHERE node.uri STARTS WITH $prefix AND node.emb IS NOT NULL
WITH node, vector.similarity.cosine(node.emb, $emb) AS score
WHERE score IS NOT NULL
RETURN node.id AS passage_id, score
ORDER BY score DESC
LIMIT $k
"""

CYPHER_passage_search_vec = """
CALL db.index.vector.queryNodes('passage_vec_index', $probe, $emb) YIELD node, score
WHERE $prefix IS NULL OR node.uri STARTS WITH $prefix
RETURN node.id AS id, node.text AS text, node.uri AS uri,
       node.p_lo AS p_lo, node.p_hi AS p_hi, score
LIMIT $k
"""

CYPHER_passage_search_vec_scoped = """
MATCH (node:Passage)
WHERE node.uri STARTS WITH $prefix AND node.emb IS NOT NULL
WITH node, vector.similarity.cosine(node.emb, $emb) AS score
WHERE score IS NOT NULL
RETURN node.id AS id, node.text AS text, node.uri AS uri,
       node.p_lo AS p_lo, node.p_hi AS p_hi, score
ORDER BY score DESC
LIMIT $k
"""

CYPHER_passage_search_ft = """
CALL db.index.fulltext.queryNodes('passage_text_index', $q) YIELD node, score
WHERE $prefix IS NULL OR node.uri STARTS WITH $prefix
RETURN node.id AS id, node.text AS text, node.uri AS uri,
       node.p_lo AS p_lo, node.p_hi AS p_hi, score
LIMIT $k
"""

CYPHER_get_passage_context = """
MATCH (s:Section)-[:HAS_PASSAGE]->(target:Passage {id: $pid})
MATCH (s)-[:HAS_PASSAGE]->(p:Passage)
WITH p, target ORDER BY p.p_lo
RETURN p.id AS id, p.p_lo AS p_lo, p.p_hi AS p_hi, p.text AS text,
       (p.id = target.id) AS is_target
"""

CYPHER_graph_expand_passages = """
MATCH (p1:Passage)<-[:ANCHORED_IN]-(e:Entity)-[:ANCHORED_IN]->(p2:Passage)
WHERE p1.id IN $seed_uris AND NOT p2.id IN $seed_uris
WITH p2, e, count{(e)-[:ANCHORED_IN]->()} AS degree
WHERE degree <= $max_degree
WITH p2, sum(1.0 / (1.0 + log(degree))) AS score, count(e) AS shared_count, collect(e.name) AS shared_entities
RETURN p2.id AS id, p2.uri AS uri, p2.text AS text,
       p2.p_lo AS p_lo, p2.p_hi AS p_hi, score, shared_count, shared_entities
ORDER BY score DESC
LIMIT $k
"""

CYPHER_entity_passage_search = """
MATCH (e:Entity)-[r:ANCHORED_IN]->(p:Passage)
WHERE toLower(e.name) IN $entity_names
WITH p, e, r, count{(e)-[:ANCHORED_IN]->()} AS degree
WHERE degree <= $max_degree
WITH p, sum(coalesce(r.score, 1.0) / (1.0 + log(degree))) AS score, collect(e.name) AS matched_entities
RETURN p.id AS id, p.uri AS uri, p.text AS text,
       p.p_lo AS p_lo, p.p_hi AS p_hi, score, matched_entities
ORDER BY score DESC
LIMIT $k
"""

CYPHER_resolve_entity_sources = """
UNWIND $ids AS id
MATCH (e:Entity {id: id})
OPTIONAL MATCH (e)-[r:ANCHORED_IN]->(p:Passage)
WITH id, e, p, r ORDER BY r.score DESC
WITH id, e, collect(p)[0] AS passage
RETURN id, e.hard_ref AS hard_ref,
       passage.uri AS uri, passage.p_lo AS p_lo
"""
