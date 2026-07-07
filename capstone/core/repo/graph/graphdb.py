import os
import logging
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
from time import time

from core.config import Neo
from core.util.cypher import concept_pred
class GraphDB:
    def __init__(self, config: Neo = Neo):
        self.driver = GraphDatabase.driver(config.uri, auth=config.auth)
        self.db_name = config.db_name
        self.setup_databases([self.db_name])

        self.config =config

    def close(self):
        self.driver.close()

    def setup_databases(self, db_names: List[str]):
        with self.driver.session(database="system") as session:
            for db in db_names:
                session.run(f"CREATE DATABASE {db} IF NOT EXISTS WAIT")
                self.create_constraints(db)

    def create_constraints(self, db_name: str ):
        id_const = "CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE"
        name_const = "CREATE CONSTRAINT node_name_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE"
        index_text = "CREATE FULLTEXT INDEX entity_text_index IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.id]"
        index_role = "CREATE INDEX entity_role_index IF NOT EXISTS FOR (n:Entity) ON (n.rrole)"
        
        with self.driver.session(database=db_name) as session:
            session.run(id_const)
            session.run(name_const)
            session.run(index_text)
            session.run(index_role)

    def reset(self, db_name: str):
        with self.driver.session(database="system") as session:
            session.run(f"CREATE OR REPLACE DATABASE {db_name} WAIT")
        self.create_constraints(db_name)

    def import_data(self, db_name: str =None, nodes: List[Dict] = [], edges: List[Dict]= [], clusters: List[Dict]= []):
        if db_name == None:
            db_name = self.db_name
        start = time()
        with self.driver.session(database=db_name) as session:
            if nodes:
                session.execute_write(self._insert_nodes, nodes)
            if edges:
                session.execute_write(self._insert_edges, edges)
            if clusters:
                session.execute_write(self._insert_clusters, clusters)
        logging.info(f"Graph inserted in {time() - start}")

    @staticmethod
    def _insert_nodes(tx, nodes: List[Dict]) -> None:
        nodes = [n for n in nodes if n.get('name') and str(n.get('name')).strip()]
        query = """
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
        tx.run(query, nodes=nodes)

    @staticmethod
    def _insert_edges(tx, edges: List[Dict]):
        edges = [e for e in edges if e.get('source_name') and e.get('target_name')]
        query = """
        UNWIND $edges AS edge_item  
        MATCH (s:Entity {name: edge_item.source_name})
        MATCH (t:Entity {name: edge_item.target_name})
        WITH s, t, edge_item
        CALL apoc.merge.relationship(s, edge_item.type, {}, edge_item.props, t) YIELD rel
        RETURN count(*)
        """
        tx.run(query, edges=edges)

    @staticmethod
    def _insert_clusters(tx, clusters: List[Dict]):
        
        query = """
        UNWIND $clusters AS c
        MATCH (n:Entity {name: c.name})
        SET n.cluster_id = c.cluster_id,
            n.cluster_level = c.level
        """
        tx.run(query, clusters=clusters)

    def run_query(self, db_name: str, query: str, params: Dict = None) -> List[Dict]:
        with self.driver.session(database=db_name) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
        
    def get_entity_by_id(self, db_name: str, node_id: str) -> Optional[Dict]:
        query = "MATCH (n:Entity {id: $id}) RETURN n"
        with self.driver.session(database=db_name) as session:
            result = session.run(query, id=node_id).single()
            return result["n"] if result else None
        
    def update_links(self,chunk_id, heading, storage_uri, links, db_name = "test"):
        with self.driver.session(database=db_name) as session:
                    session.run(
                        """
                        MERGE (c:TextbookChunk {id: $chunk_id})
                        SET c.heading = $heading, c.storage_uri = $storage_uri
                        WITH c
                        UNWIND $links AS link
                        MATCH (anchor {id: link.anchor_id})
                        MERGE (c)-[r:ELABORATES_ON]->(anchor)
                        SET r.justification = link.justification
                        """,
                        chunk_id=chunk_id, heading=heading, 
                        storage_uri=storage_uri, links=links
                    )
    def create_tb_indexes(self, db_name: str, dim: int = 768):
        ## fulltext (lexical) + vector index over :Passage for anchoring
        ft = "CREATE FULLTEXT INDEX passage_text_index IF NOT EXISTS FOR (n:Passage) ON EACH [n.text]"
        vec = (f"CREATE VECTOR INDEX passage_vec_index IF NOT EXISTS FOR (n:Passage) ON n.emb "
               f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}")
        with self.driver.session(database=db_name) as session:
            session.run(ft)
            session.run(vec)

    def write_textbook_tree(self, sections, passages, book_uri, db_name=None, dim: int = 768):
        ## deterministic write of :Section tree + :Passage units (textbook primitive)
        db_name = db_name or self.db_name
        self.create_tb_indexes(db_name, dim)
        with self.driver.session(database=db_name) as session:
            session.execute_write(self._write_sections, sections)
            session.execute_write(self._write_passages, passages, book_uri)

    @staticmethod
    def _write_sections(tx, sections: List[Dict]):
        q = """
        UNWIND $sections AS s
        MERGE (n:Section {id: s.id})
        SET n.title=s.title, n.level=s.level, n.p_lo=s.p_num[0], n.p_hi=s.p_num[1], n.order=s.order
        WITH n, s WHERE s.parent_id IS NOT NULL
        MATCH (p:Section {id: s.parent_id})
        MERGE (p)-[:CONTAINS]->(n)
        """
        tx.run(q, sections=sections)

    @staticmethod
    def _write_passages(tx, passages: List[Dict], book_uri: str):
        q = """
        UNWIND $passages AS p
        MERGE (n:Passage {id: p.id})
        SET n.p_lo=p.p_num[0], n.p_hi=p.p_num[1], n.text=p.text, n.emb=p.emb, n.uri=$uri
        WITH n, p
        MATCH (s:Section {id: p.section_id})
        MERGE (s)-[:HAS_PASSAGE]->(n)
        """
        tx.run(q, passages=passages, uri=book_uri)

    def anchor_search(self, emb: List[float], top_k: int = 5, db_name=None,
                      uri_prefix: Optional[str] = None) -> List[Dict]:
        ## top-k by vec (anchor target lookup)
        db_name = db_name or self.db_name
        q = """
        CALL db.index.vector.queryNodes('passage_vec_index', $probe, $emb) YIELD node, score
        WHERE $prefix IS NULL OR node.uri STARTS WITH $prefix
        RETURN node.id AS passage_id, score
        LIMIT $k
        """
        ## vec index cannot pre-filter, probe wide then trim
        probe = top_k * 4 if uri_prefix else top_k
        try:
            return self.run_query(db_name, q, {"probe": probe, "k": top_k,
                                               "emb": emb, "prefix": uri_prefix})
        except ClientError:
            ## index absent on slide-only db
            return []

    def write_anchors(self, links: List[Dict], db_name=None):
        ## batch concept->passage anchors in one UNWIND MERGE (idempotent)
        db_name = db_name or self.db_name
        q = """
        UNWIND $links AS l
        MATCH (e:Entity {name: l.entity_name})
        MATCH (p:Passage {id: l.passage_id})
        MERGE (e)-[r:ANCHORED_IN]->(p)
        SET r.score=l.score, r.justification=coalesce(l.justification, '')
        """
        with self.driver.session(database=db_name) as session:
            session.run(q, links=links)

    def get_concept_page(self, name: str, db_name=None) -> Optional[Dict]:
        ## concept -> best anchored passage -> (minio pdf uri, page)
        db_name = db_name or self.db_name
        q = """
        MATCH (e:Entity)-[r:ANCHORED_IN]->(p:Passage)
        WHERE toLower(e.name) CONTAINS toLower($name)
        RETURN p.uri AS uri, p.p_lo AS page, r.score AS score, e.name AS concept
        ORDER BY r.score DESC LIMIT 1
        """
        rows = self.run_query(db_name, q, {"name": name})
        return rows[0] if rows else None

    def passage_search(self, emb: List[float], query_text: str = "", top_k: int = 5,
                       db_name=None) -> List[Dict]:
        ## hybrid textbook retrieval: vector ANN + fulltext, RRF merge
        db_name = db_name or self.db_name
        vec_q = """
        CALL db.index.vector.queryNodes('passage_vec_index', $k, $emb) YIELD node, score
        RETURN node.id AS id, node.text AS text, node.uri AS uri,
               node.p_lo AS p_lo, node.p_hi AS p_hi, score
        """
        try:
            vec_rows = self.run_query(db_name, vec_q, {"k": top_k, "emb": emb})
        except ClientError:
            vec_rows = []
        ft_rows = []
        if query_text:
            ft_q = """
            CALL db.index.fulltext.queryNodes('passage_text_index', $q) YIELD node, score
            RETURN node.id AS id, node.text AS text, node.uri AS uri,
                   node.p_lo AS p_lo, node.p_hi AS p_hi, score
            LIMIT $k
            """
            try:
                ft_rows = self.run_query(db_name, ft_q, {"q": query_text, "k": top_k})
            except ClientError:
                ft_rows = []
        ## RRF, raw merge lets Lucene beat cosine
        merged: Dict[str, Dict] = {}
        for rows in (vec_rows, ft_rows):
            for rank, r in enumerate(rows, start=1):
                cur = merged.setdefault(r["id"], {**dict(r), "score": 0.0})
                cur["score"] += 1.0 / (60 + rank)
        out = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return out[:top_k]

    def get_concept_anchors(self, name: str, db_name=None) -> List[Dict]:
        ## all passages a concept is anchored in (primary + secondary citations)
        db_name = db_name or self.db_name
        q = """
        MATCH (e:Entity)-[r:ANCHORED_IN]->(p:Passage)
        WHERE toLower(e.name) CONTAINS toLower($name)
        RETURN e.name AS concept, p.uri AS uri, p.p_lo AS p_lo, p.p_hi AS p_hi,
               r.score AS score, substring(p.text, 0, 160) AS preview
        ORDER BY r.score DESC LIMIT 10
        """
        return self.run_query(db_name, q, {"name": name})

    def get_passage_context(self, passage_id: str, window: int = 1, db_name=None) -> List[Dict]:
        ## sibling passages in the same section, page-ordered, around the target
        db_name = db_name or self.db_name
        q = """
        MATCH (s:Section)-[:HAS_PASSAGE]->(target:Passage {id: $pid})
        MATCH (s)-[:HAS_PASSAGE]->(p:Passage)
        WITH p, target ORDER BY p.p_lo
        RETURN p.id AS id, p.p_lo AS p_lo, p.p_hi AS p_hi, p.text AS text,
               (p.id = target.id) AS is_target
        """
        rows = self.run_query(db_name, q, {"pid": passage_id})
        idx = next((i for i, r in enumerate(rows) if r["is_target"]), None)
        if idx is None:
            return rows
        lo, hi = max(0, idx - window), min(len(rows), idx + window + 1)
        return rows[lo:hi]

    def get_toc(self, course_hint: str = None, db_name=None) -> List[Dict]:
        ## authored :Section tree, optionally scoped to a course via passage uri
        db_name = db_name or self.db_name
        if course_hint:
            q = """
            MATCH (s:Section)-[:HAS_PASSAGE]->(p:Passage)
            WHERE p.uri STARTS WITH $hint
            WITH DISTINCT s
            OPTIONAL MATCH (parent:Section)-[:CONTAINS]->(s)
            RETURN s.id AS id, s.title AS title, s.level AS level,
                   s.p_lo AS p_lo, s.p_hi AS p_hi, s.order AS order, parent.id AS parent_id
            ORDER BY s.level, s.order
            """
            params = {"hint": f"{course_hint}/"}
        else:
            q = """
            MATCH (s:Section)
            OPTIONAL MATCH (parent:Section)-[:CONTAINS]->(s)
            RETURN s.id AS id, s.title AS title, s.level AS level,
                   s.p_lo AS p_lo, s.p_hi AS p_hi, s.order AS order, parent.id AS parent_id
            ORDER BY s.level, s.order
            """
            params = {}
        return self.run_query(db_name, q, params)

    def query(self, q = None, param = {}):
        if q == None or len(q) <=3:
            return
        db_name = self.config.db_name
        with self.driver.session(database=db_name) as session:
                    session.run(q,param)

    def get_learning_graph(self, student_id: Optional[str] = None) -> Dict:
        query = f"""
        MATCH (n:Entity)
        WHERE {concept_pred('n')}
        OPTIONAL MATCH (n)-[r]->(m:Entity)
        WHERE type(r) <> 'CONTENT'
        WITH n, count(r) AS out_degree
        OPTIONAL MATCH (n)-[:BELONGS_TO|PART_OF*1..2]->(c:Entity)
        WHERE c.typeNode = 'Community'
        OPTIONAL MATCH (s:Student {id: $sid})-[mas:MASTERY]->(n)
        RETURN n.name AS name, 
               n.typeNode AS type, 
               c.name AS course_name,
               out_degree,
               substring(coalesce(n.content, ''), 0, 50) AS description,
               coalesce(mas.level, 0) AS mastery
        """
        results = self.run_query(self.db_name, query, {"sid": student_id or ""})
        return {"nodes": results}

    def update_learn(self, student_id: str, current_pos, new_node) -> None:
        if not new_node or not student_id:
            return

        mastery_query = """
        MERGE (s:Student {id: $sid})
        MERGE (n:Entity {name: $new_name})
        MERGE (s)-[r:MASTERY]->(n)
        SET r.level = CASE WHEN coalesce(r.level, 0) < 6 THEN coalesce(r.level, 0) + 1 ELSE 6 END,
            r.last_visited = datetime()
        """
        self.query(mastery_query, {"sid": student_id, "new_name": new_node.name})

        if current_pos:
            transition_query = """
            MERGE (s:Student {id: $sid})
            WITH s
            MATCH (curr:Entity {name: $curr_name}), (next:Entity {name: $new_name})
            MERGE (s)-[r:LEARNED_PATH]->(next)
            SET r.from_node = $curr_name,
                r.count = coalesce(r.count, 0) + 1,
                r.last_visited = datetime()
            """
            self.query(transition_query, {
                "sid": student_id,
                "curr_name": current_pos.name,
                "new_name": new_node.name
            })

    def get_mastery(self, student_id: str, node_name: str) -> int:
        query = """
        MATCH (s:Student {id: $sid})-[r:MASTERY]->(n:Entity {name: $name})
        RETURN coalesce(r.level, 0) AS mastery
        """
        with self.driver.session(database=self.db_name) as session:
            result = session.run(query, sid=student_id, name=node_name).single()
            return result["mastery"] if result else 0