import time
import logging
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List, Dict
from core.config import Mil_conf

class MilvusDB:
    def reset(self):
        pass
    def __init__(self, config: Mil_conf = Mil_conf()):
        uri_clean = config.uri.replace("http://", "").replace("https://", "")
        self.host = uri_clean.split(":")[0]
        self.port = uri_clean.split(":")[1] if ":" in uri_clean else "19530"
        
        self.collection_name = config.collection_name
        self.dim = config.dim
        self.retries = config.retries
        self.delay = config.delay

        self._connect()
        self.collection = self._init_collection()

    def _connect(self):
        for i in range(self.retries):
            try:
                connections.connect(
                    alias="default", 
                    host=self.host, 
                    port=self.port
                )
                logging.info(f"Milvus connected to {self.host}:{self.port}")
                return
            except Exception as e:
                logging.error(f"Attemps {i+1}: Milvus is not ready... {e}")
                time.sleep(self.delay)
        raise Exception("Milvus is cooked and so am I!")

    def _init_collection(self) -> Collection:
        if utility.has_collection(self.collection_name):
            col = Collection(self.collection_name)
            col.load()
            return col

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=65535),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="rrole", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="community", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
        collection = Collection(name=self.collection_name, schema=schema)
        
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.create_index(field_name="name", index_name="idx_name")
        collection.create_index(field_name="rrole", index_name="idx_rrole")
        collection.create_index(field_name="topic", index_name="idx_topic")
        collection.create_index(field_name="community", index_name="idx_community")
        collection.load()
        return collection

    def insert_data(self, nodes: List[Dict], embedder):
        insert_data = []
        for node in nodes:
            rrole = node.get("rrole")
            content = node.get("content")
            
            if not rrole or not content:
                continue

            name = node.get("name", "")
            node_id = str(node.get("id"))
            
            # Strip out ARTICLE context before embedding
            text_to_embed = f"{name}: {content}".split("\n**ARTICLE:")[0]
            
            topic = str(node.get("topic", ""))
            community = str(node.get("community", ""))
            
            vector = embedder.get_embedding(text_to_embed)
            
            insert_data.append({
                "id": node_id,
                "text": f"{name}: {content}",
                "name": name,
                "rrole": str(rrole),
                "topic": topic,
                "community": community,
                "embedding": vector
            })

        if insert_data:
            self.collection.insert(insert_data)
            self.collection.flush()

    def search(self, query: str, embedder, top_k: int = 5, expr: str = None) -> List[Dict]:
        query_vector = embedder.get_embedding(query)
        return self.search_vec(query_vector, top_k=top_k, expr=expr)

    def search_vec(self, vector: List[float], top_k: int = 5, expr: str = None) -> List[Dict]:
        ## search by precomputed embedding — segments carry their own vector, no re-embed
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }

        results = self.collection.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["id", "text", "name", "rrole", "topic", "community"]
        )

        output = []
        for hits in results:
            for hit in hits:
                output.append({
                    "id": hit.entity.get("id"),
                    "text": hit.entity.get("text"),
                    "name": hit.entity.get("name"),
                    "rrole": hit.entity.get("rrole"),
                    "topic": hit.entity.get("topic"),
                    "community": hit.entity.get("community"),
                    "score": hit.distance
                })
        return output

    def close(self):
        connections.disconnect("default")