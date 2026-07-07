import os
import logging
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
from time import time

from core.config import Neo
from core.repo.graph.cypher.kg_const.crud import (
    CYPHER_insert_nodes, CYPHER_insert_edges, CYPHER_insert_clusters,
    CYPHER_get_entity_by_id,
)
from core.repo.graph.cypher.kg_const.textbook import (
    CYPHER_write_sections, CYPHER_write_passages,
    CYPHER_get_toc_scoped, CYPHER_get_toc,
)
from core.repo.graph.cypher.kg_const.bind import (
    CYPHER_update_links, CYPHER_write_anchors,
    CYPHER_get_concept_page, CYPHER_get_concept_anchors,
    CYPHER_anchor_search, CYPHER_passage_search_vec,
    CYPHER_passage_search_ft, CYPHER_get_passage_context,
)
from core.repo.graph.cypher.kg_const.learn import (
    CYPHER_get_learning_graph, CYPHER_update_learn_mastery,
    CYPHER_update_learn_transition, CYPHER_get_mastery,
)


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
        tx.run(CYPHER_insert_nodes, nodes=nodes)

    @staticmethod
    def _insert_edges(tx, edges: List[Dict]):
        edges = [e for e in edges if e.get('source_name') and e.get('target_name')]
        tx.run(CYPHER_insert_edges, edges=edges)

    @staticmethod
    def _insert_clusters(tx, clusters: List[Dict]):
        tx.run(CYPHER_insert_clusters, clusters=clusters)

    def run_query(self, db_name: str, query: str, params: Dict = None) -> List[Dict]:
        with self.driver.session(database=db_name) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
        
    def get_entity_by_id(self, db_name: str, node_id: str) -> Optional[Dict]:
        with self.driver.session(database=db_name) as session:
            result = session.run(CYPHER_get_entity_by_id, id=node_id).single()
            return result["n"] if result else None
        
    def update_links(self,chunk_id, heading, storage_uri, links, db_name = "test"):
        with self.driver.session(database=db_name) as session:
                    session.run(
                        CYPHER_update_links,
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
        tx.run(CYPHER_write_sections, sections=sections)

    @staticmethod
    def _write_passages(tx, passages: List[Dict], book_uri: str):
        tx.run(CYPHER_write_passages, passages=passages, uri=book_uri)

    def anchor_search(self, emb: List[float], top_k: int = 5, db_name=None,
                      uri_prefix: Optional[str] = None) -> List[Dict]:
        ## top-k by vec (anchor target lookup)
        db_name = db_name or self.db_name
        ## vec index cannot pre-filter, probe wide then trim
        probe = top_k * 4 if uri_prefix else top_k
        try:
            return self.run_query(db_name, CYPHER_anchor_search, {"probe": probe, "k": top_k,
                                               "emb": emb, "prefix": uri_prefix})
        except ClientError:
            ## index absent on slide-only db
            return []

    def write_anchors(self, links: List[Dict], db_name=None):
        ## batch concept->passage anchors in one UNWIND MERGE (idempotent)
        db_name = db_name or self.db_name
        with self.driver.session(database=db_name) as session:
            session.run(CYPHER_write_anchors, links=links)

    def get_concept_page(self, name: str, db_name=None) -> Optional[Dict]:
        ## concept -> best anchored passage -> (minio pdf uri, page)
        db_name = db_name or self.db_name
        rows = self.run_query(db_name, CYPHER_get_concept_page, {"name": name})
        return rows[0] if rows else None

    def passage_search(self, emb: List[float], query_text: str = "", top_k: int = 5,
                       db_name=None) -> List[Dict]:
        ## hybrid textbook retrieval: vector ANN + fulltext, RRF merge
        db_name = db_name or self.db_name
        try:
            vec_rows = self.run_query(db_name, CYPHER_passage_search_vec, {"k": top_k, "emb": emb})
        except ClientError:
            vec_rows = []
        ft_rows = []
        if query_text:
            try:
                ft_rows = self.run_query(db_name, CYPHER_passage_search_ft, {"q": query_text, "k": top_k})
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
        return self.run_query(db_name, CYPHER_get_concept_anchors, {"name": name})

    def get_passage_context(self, passage_id: str, window: int = 1, db_name=None) -> List[Dict]:
        ## sibling passages in the same section, page-ordered, around the target
        db_name = db_name or self.db_name
        rows = self.run_query(db_name, CYPHER_get_passage_context, {"pid": passage_id})
        idx = next((i for i, r in enumerate(rows) if r["is_target"]), None)
        if idx is None:
            return rows
        lo, hi = max(0, idx - window), min(len(rows), idx + window + 1)
        return rows[lo:hi]

    def get_toc(self, course_hint: str = None, db_name=None) -> List[Dict]:
        ## authored :Section tree, optionally scoped to a course via passage uri
        db_name = db_name or self.db_name
        if course_hint:
            q = CYPHER_get_toc_scoped
            params = {"hint": f"{course_hint}/"}
        else:
            q = CYPHER_get_toc
            params = {}
        return self.run_query(db_name, q, params)

    def query(self, q = None, param = {}):
        if q == None or len(q) <=3:
            return
        db_name = self.config.db_name
        with self.driver.session(database=db_name) as session:
                    session.run(q,param)

    def get_learning_graph(self, student_id: Optional[str] = None) -> Dict:
        results = self.run_query(self.db_name, CYPHER_get_learning_graph(), {"sid": student_id or ""})
        return {"nodes": results}

    def update_learn(self, student_id: str, current_pos, new_node) -> None:
        if not new_node or not student_id:
            return

        self.query(CYPHER_update_learn_mastery, {"sid": student_id, "new_name": new_node.name})

        if current_pos:
            self.query(CYPHER_update_learn_transition, {
                "sid": student_id,
                "curr_name": current_pos.name,
                "new_name": new_node.name
            })

    def get_mastery(self, student_id: str, node_name: str) -> int:
        with self.driver.session(database=self.db_name) as session:
            result = session.run(CYPHER_get_mastery, sid=student_id, name=node_name).single()
            return result["mastery"] if result else 0