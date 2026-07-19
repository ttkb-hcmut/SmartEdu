from typing import List, Tuple


def cliff_partition(scores: List[float], g: float, floor: float) -> Tuple[List[int], bool]:
    ## BookRAG gradient walk (arXiv:2512.03413 §4.3.2): relative cliff, not absolute threshold.
    ## scores DESC. keep idx 0, keep successor i while scores[i] > scores[i-1]/g (plateau=anchored-many).
    ## novel = no kept member >= floor (guards flat-because-junk speech).
    if not scores:
        return [], True

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept = [order[0]]
    prev = scores[order[0]]
    for idx in order[1:]:
        s = scores[idx]
        if prev > 0 and s > prev / g:
            kept.append(idx)
            prev = s
        else:
            break

    novel = not any(scores[i] >= floor for i in kept)
    return kept, novel
