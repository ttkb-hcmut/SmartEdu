import json
from typing import Dict, List, Tuple

from core.config import Emb_conf, Ingest_param
from core.ingest.novelty import cliff_partition
from core.ingest.segment import group_passages


ANCHOR_VERSION = "cliff-v1"


def _to_items(vid_id: str, whisper_segs: List[Dict]) -> List[Dict]:
    ## whisper seg ---> valley item
    return [{
        "id": f"{vid_id}_s{i}",
        "text": seg["text"],
        "order": i,
        "section_id": vid_id,
        "p_num": (seg["t_lo"], seg["t_hi"]),
    } for i, seg in enumerate(whisper_segs)]


def build_vid_segs(embedder, milvus_db, vid_id: str, course: str,
                   whisper_segs: List[Dict],
                   cfg: Ingest_param) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if not whisper_segs:
        return [], [], []

    merged = group_passages(_to_items(vid_id, whisper_segs), embedder.get_embedding, cfg)
    expr = f'typeNode == "Concept" and course == {json.dumps(course, ensure_ascii=False)}'

    seg_nodes, anchor_links, novel_ents = [], [], []
    for order, p in enumerate(merged):
        hits = sorted(
            milvus_db.search_vec(p["emb"], top_k=cfg.segment_top_k, expr=expr),
            key=lambda hit: hit["score"],
            reverse=True,
        )
        scores = [h["score"] for h in hits]
        kept, novel = cliff_partition(scores, cfg.anchor_gradient_g, cfg.anchor_score_min)
        best_score = scores[0] if scores else 0.0

        seg_id = p["id"]
        seg_nodes.append({
            "id": seg_id,
            "p_num": p["p_num"],
            "text": p["text"],
            "emb": p["emb"],
            "order": order,
            "best_score": best_score,
            "candidate_ids": [h["id"] for h in hits],
            "candidate_names": [h["name"] for h in hits],
            "candidate_scores": scores,
            "anchor_model": Emb_conf().model_name,
            "anchor_version": ANCHOR_VERSION,
            "novel_candidate": novel,
        })

        for i in kept:
            if scores[i] >= cfg.anchor_score_min:
                anchor_links.append({"entity_name": hits[i]["name"],
                                     "segment_id": seg_id, "score": scores[i]})

        if novel:
            novel_ents.append({"t_lo": p["p_num"][0], "t_hi": p["p_num"][1],
                               "best_score": best_score})

    return seg_nodes, anchor_links, novel_ents
