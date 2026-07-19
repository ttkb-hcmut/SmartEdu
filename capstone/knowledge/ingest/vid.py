from typing import Dict, List, Tuple

from core.config import Ingest_param
from core.ingest.novelty import cliff_partition
from core.ingest.segment import group_passages


def _to_items(video_id: str, whisper_segments: List[Dict]) -> List[Dict]:
    ## whisper seg ---> valley item
    return [{
        "id": f"{video_id}_s{i}",
        "text": seg["text"],
        "order": i,
        "section_id": video_id,
        "p_num": (seg["t_lo"], seg["t_hi"]),
    } for i, seg in enumerate(whisper_segments)]


def build_video_segments(embedder, milvus_db, video_id: str, whisper_segments: List[Dict],
                         cfg: Ingest_param) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if not whisper_segments:
        return [], [], []

    merged = group_passages(_to_items(video_id, whisper_segments), embedder.get_embedding, cfg)

    seg_nodes, anchor_links, novel_entries = [], [], []
    for order, p in enumerate(merged):
        hits = milvus_db.search_vec(p["emb"], top_k=cfg.segment_top_k)
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
            "anchor_scores": sorted(scores, reverse=True),
            "novel_candidate": novel,
        })

        for i in kept:
            if scores[i] >= cfg.anchor_score_min:
                anchor_links.append({"entity_name": hits[i]["name"],
                                     "segment_id": seg_id, "score": scores[i]})

        if novel:
            novel_entries.append({"t_lo": p["p_num"][0], "t_hi": p["p_num"][1],
                                  "best_score": best_score})

    return seg_nodes, anchor_links, novel_entries
