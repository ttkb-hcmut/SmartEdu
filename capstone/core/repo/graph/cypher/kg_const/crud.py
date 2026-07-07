CYPHER_insert_nodes = """
WITH {`Rhetorical Node`: 1, `Knowledge Concept`: 2, `Knowledge Topic`: 3, `Knowledge Community`: 4} AS rank
UNWIND $nodes AS node_item
MERGE (n:Entity {name: node_item.name})
ON MATCH SET
    n.definition = coalesce(n.definition, node_item.definition),
    n.typeNode = CASE
        WHEN coalesce(rank[n.typeNode], 0) < coalesce(rank[node_item.typeNode], 0)
        THEN node_item.typeNode
        ELSE n.typeNode
    END
ON CREATE SET n = node_item
WITH n, node_item
CALL apoc.create.addLabels(n, [node_item.name]) YIELD node
RETURN count(*)
"""

CYPHER_insert_edges = """
UNWIND $edges AS edge_item
MATCH (s:Entity {name: edge_item.source_name})
MATCH (t:Entity {name: edge_item.target_name})
WITH s, t, edge_item
CALL apoc.merge.relationship(s, edge_item.type, {}, edge_item.props, t) YIELD rel
RETURN count(*)
"""

CYPHER_insert_clusters = """
UNWIND $clusters AS c
MATCH (n:Entity {name: c.name})
SET n.cluster_id = c.cluster_id,
    n.cluster_level = c.level
"""

CYPHER_get_entity_by_id = "MATCH (n:Entity {id: $id}) RETURN n"
