from typing import Dict, List

from core.config import Ingest_param


def anchor_concepts(embedder, graph_db, concept_nodes: List[Dict], course_name: str,
                    db_name: str, cfg: Ingest_param) -> int:
    ## linked-after: concept -> passage by vector ANN, scoped to own course
    seen, names, texts = set(), [], []
    for c in concept_nodes:
        name = c.get("name")
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        names.append(name)
        texts.append(f"{name}. {c.get('content', '')}".strip())
    if not names:
        return 0

    embs = embedder.get_embeddings(texts)
    links = []
    for name, emb in zip(names, embs):
        hits = graph_db.anchor_search(emb, cfg.anchor_top_k, db_name, f"{course_name}/")
        for h in hits:
            if h.get("score", 0) >= cfg.anchor_score_min:
                links.append({"entity_name": name, "passage_id": h["passage_id"],
                              "score": h["score"], "justification": ""})
    if links:
        graph_db.write_anchors(links, db_name)
    return len(links)
