import json
from typing import Tuple, List, Dict
from core.schema.graph import KG_Instance

def serialize_data(data: dict) -> dict:
        """
        Convert nested dictionaries and lists into JSON strings for Neo4j.
        """
        clean_data = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                clean_data[k] = json.dumps(v, ensure_ascii=False)
            else:
                clean_data[k] = v
        return clean_data

## Micro learning

def serialize_kg_to_dict(graph: KG_Instance) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    nodes_list = []
    id_to_name = {}

    for node_id, node_data in graph.nodes.items():
        node_props = serialize_data(node_data.model_dump())
        node_props["id"] = node_id
        id_to_name[node_id] = node_data.name
  
        nodes_list.append(node_props)

    edges_list = []
    for (src_id, tgt_id, rel_type), edge_data in graph.edges.items():
        src_name = id_to_name.get(src_id)
        tgt_name = id_to_name.get(tgt_id)
        
        relationship = edge_data.relationship
        if not src_name or not tgt_name:
            continue
            
        edge_props = serialize_data(edge_data.model_dump(exclude={"source", "target"}))
            
        edges_list.append({
            "source_id": src_id,
            "target_id": tgt_id,
            "source_name": src_name,
            "target_name": tgt_name,
            "type": rel_type,
            "relationship":relationship,
            "props": edge_props
        })

    cluster_list = []
    if graph.clusters:
        for cluster_id, cluster_data in graph.clusters.items():
            for node_id in cluster_data.member_ids:
                n_name = id_to_name.get(node_id)
                if n_name:
                    cluster_list.append({
                        "id": node_id,
                        "name": n_name,
                        "cluster_id": cluster_id,
                        "level": getattr(cluster_data, 'level', 0)
                    })

    return nodes_list, edges_list, cluster_list
