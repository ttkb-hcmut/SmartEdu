CYPHER_insert_nodes = """
WITH {Rhetorical: 1, Concept: 2, Topic: 3, Community: 4,
      `Rhetorical Node`: 1, `Knowledge Concept`: 2, `Knowledge Topic`: 3, `Knowledge Community`: 4} AS rank
UNWIND $nodes AS node_item
OPTIONAL MATCH (by_id:Entity {id: node_item.id})
OPTIONAL MATCH (by_name:Entity {name: node_item.name})
WITH rank, node_item, coalesce(by_id, by_name) AS existing
CALL (node_item, existing) {
    WITH node_item, existing WHERE existing IS NOT NULL
    SET existing.id = coalesce(existing.id, node_item.id)
    RETURN existing AS n
    UNION
    WITH node_item, existing WHERE existing IS NULL
    MERGE (n:Entity {id: node_item.id})
    ON CREATE SET n = node_item
    RETURN n
}
SET n += apoc.map.removeKeys(node_item, ['id', 'name', 'typeNode'])
SET n.typeNode = CASE
        WHEN coalesce(rank[n.typeNode], 0) < coalesce(rank[node_item.typeNode], 0)
        THEN node_item.typeNode
        WHEN n.typeNode IS NULL THEN node_item.typeNode
        ELSE n.typeNode
    END
WITH n, node_item
CALL apoc.create.addLabels(n, [node_item.name]) YIELD node
RETURN node_item.id AS input_id, n.id AS id, n.name AS name, n.typeNode AS typeNode
"""

CYPHER_insert_edges = """
UNWIND $edges AS edge_item
MATCH (s:Entity)
WHERE (edge_item.source_id IS NOT NULL AND s.id = edge_item.source_id)
   OR (edge_item.source_id IS NULL AND s.name = edge_item.source_name)
MATCH (t:Entity)
WHERE (edge_item.target_id IS NOT NULL AND t.id = edge_item.target_id)
   OR (edge_item.target_id IS NULL AND t.name = edge_item.target_name)
WITH s, t, edge_item
CALL apoc.merge.relationship(s, edge_item.type, {}, edge_item.props, t) YIELD rel
RETURN count(*)
"""

CYPHER_insert_clusters = """
UNWIND $clusters AS c
MATCH (n:Entity)
WHERE (c.id IS NOT NULL AND n.id = c.id) OR (c.id IS NULL AND n.name = c.name)
SET n.cluster_id = c.cluster_id,
    n.cluster_level = c.level
"""

CYPHER_get_entity_by_id = "MATCH (n:Entity {id: $id}) RETURN n"
