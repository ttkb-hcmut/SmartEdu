from typing import Dict, Optional

MASTERED_AT = 4


def _m(learning: Dict[str, dict], name: str) -> int:
    return int((learning.get(name) or {}).get("mastery", 0))


def render_frontier(tree: dict, learning: Dict[str, dict],
                    current_pos: Optional[str], char_budget: int = 1800) -> str:
    ## T1 current branch > T2 collapsed topics > T3 orphans; fill by tier, emit in doc order
    lines = []
    idx = 0

    def add(tier: int, text: str):
        nonlocal idx
        lines.append((tier, idx, text))
        idx += 1

    topics = tree.get("topics", [])
    n_mastered = sum(1 for t in topics
                     if t["concepts"] and all(_m(learning, c["name"]) >= MASTERED_AT
                                              for c in t["concepts"]))
    add(1, f"COURSE: {tree.get('course', '?')} ({len(topics)} topics, {n_mastered} mastered)")

    current_topic = None
    for t in topics:
        if any(c["name"] == current_pos for c in t["concepts"]):
            current_topic = t["name"]

    for t in topics:
        concepts = t["concepts"]
        all_mastered = bool(concepts) and all(_m(learning, c["name"]) >= MASTERED_AT for c in concepts)
        if t["name"] == current_topic:
            add(1, f"▸ {t['name']}  ← CURRENT TOPIC")
            for c in concepts:
                m = _m(learning, c["name"])
                mark = "✓" if m >= MASTERED_AT else ("●" if c["name"] == current_pos else "○")
                req = f" | requires: {', '.join(c['requires'])}" if c.get("requires") else ""
                cur = " ← current" if c["name"] == current_pos else ""
                add(1, f"   {mark} {c['name']} (mastery {m}){req}{cur}")
        elif all_mastered:
            add(2, f"✓ {t['name']} — {len(concepts)}/{len(concepts)} mastered")
        else:
            started = sum(1 for c in concepts if _m(learning, c["name"]) > 0)
            tail = f" ({started}/{len(concepts)} started)" if started else f" ({len(concepts)} concepts)"
            add(2, f"○ {t['name']}{tail}")

    for c in tree.get("orphan_concepts", []):
        add(3, f"○ {c['name']} (mastery {_m(learning, c['name'])})")

    chosen, used = [], 0
    for tier, i, text in sorted(lines, key=lambda x: (x[0], x[1])):
        cost = len(text) + 1
        if used + cost > char_budget:
            continue
        chosen.append((i, text))
        used += cost
    chosen.sort()
    return "\n".join(t for _, t in chosen)
