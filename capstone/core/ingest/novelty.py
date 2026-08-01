from typing import List, Tuple


def cliff_partition(scores: List[float], g: float, floor: float) -> Tuple[List[int], bool]:
    ## DESC cliff walk selects anchor breadth; novelty reduces to top-score floor
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
