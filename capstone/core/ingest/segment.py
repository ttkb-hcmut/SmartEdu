from collections import OrderedDict
from typing import Callable, Dict, List

import numpy as np

from core.config import Ingest_param, MergeStrat


def _cos(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def _cuts_threshold(sims: List[float], cfg: Ingest_param) -> List[int]:
    return [i + 1 for i, s in enumerate(sims) if s < cfg.merge_threshold]


def _smooth(sims: List[float], w: int) -> List[float]:
    if w <= 1 or len(sims) < 3 * w:
        return list(sims)
    r, n, out = w // 2, len(sims), []
    for i in range(n):
        lo, hi = max(0, i - r), min(n, i + r + 1)
        out.append(sum(sims[lo:hi]) / (hi - lo))
    return out


def _depth(s: List[float], i: int, n: int) -> float:
    li = i
    while li > 0 and s[li - 1] >= s[li]:
        li -= 1
    ri = i
    while ri < n - 1 and s[ri + 1] >= s[ri]:
        ri += 1
    return (s[li] - s[i]) + (s[ri] - s[i])


def _cuts_valley(sims: List[float], cfg: Ingest_param) -> List[int]:
    s = _smooth(sims, max(1, cfg.valley_window))
    n = len(s)
    cuts = []
    for i in range(n):
        is_min = (i == 0 or s[i] <= s[i - 1]) and (i == n - 1 or s[i] <= s[i + 1])
        if is_min and _depth(s, i, n) >= cfg.valley_depth:
            cuts.append(i + 1)
    return cuts


def _split(items: List[dict], cuts: List[int]) -> List[List[dict]]:
    groups, start = [], 0
    for c in sorted(set(cuts)):
        if start < c < len(items):
            groups.append(items[start:c])
            start = c
    groups.append(items[start:])
    return [g for g in groups if g]


def group_passages(items: List[dict], embed: Callable[[str], List[float]],
                   cfg: Ingest_param = None) -> List[Dict]:
    ## merge adjacent items, never cross section
    cfg = cfg or Ingest_param()
    embs = {it["id"]: embed(it["text"]) for it in items}

    by_sec: "OrderedDict[str, list]" = OrderedDict()
    for it in sorted(items, key=lambda x: x["order"]):
        by_sec.setdefault(it["section_id"], []).append(it)

    passages, pc = [], 0
    for sec_id, sec_items in by_sec.items():
        if not cfg.semantic_merge:
            groups = [[it] for it in sec_items]
        else:
            vecs = [embs[it["id"]] for it in sec_items]
            sims = [_cos(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
            cuts = (_cuts_valley if cfg.merge_strategy == MergeStrat.VALLEY
                    else _cuts_threshold)(sims, cfg)
            groups = _split(sec_items, cuts)

        for g in groups:
            text = "\n".join(it["text"] for it in g)
            ## mean blur risk, re-embed merged text
            emb = embs[g[0]["id"]] if len(g) == 1 else embed(text)
            passages.append({
                "id": f"{sec_id}_p{pc}",
                "section_id": sec_id,
                "p_num": (min(it["p_num"][0] for it in g), max(it["p_num"][1] for it in g)),
                "text": text,
                "emb": list(emb),
                "member_ids": [it["id"] for it in g],
            })
            pc += 1
    return passages
