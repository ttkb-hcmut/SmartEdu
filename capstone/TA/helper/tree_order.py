import heapq
from typing import Dict, List, Tuple


def topo_order(names: List[str], edges: List[Tuple[str, str]], score: Dict[str, float]) -> List[str]:
    ## left->right learning order: Kahn over PREREQUISITE, score tiebreak, cycle-tolerant
    name_set = set(names)
    indeg = {n: 0 for n in names}
    adj = {n: [] for n in names}
    for pre, post in edges:
        if pre in name_set and post in name_set:
            adj[pre].append(post)
            indeg[post] += 1

    heap = [(-score.get(n, 0.0), n) for n in names if indeg[n] == 0]
    heapq.heapify(heap)
    out: List[str] = []
    while heap:
        _, n = heapq.heappop(heap)
        out.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(heap, (-score.get(m, 0.0), m))

    if len(out) < len(names):
        ## cycle remainder: append by score desc, never raise
        done = set(out)
        out.extend(sorted((n for n in names if n not in done),
                          key=lambda n: -score.get(n, 0.0)))
    return out
