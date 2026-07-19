from typing import Dict, List, Tuple, TypedDict


class SlideChunk(TypedDict, total=False):
    chunk_id: str
    heading: str
    content: str
    page_num: Tuple[int, int]


class TreeItem(TypedDict):
    id: str
    text: str
    order: int
    section_id: str
    p_num: Tuple[int, int]


class Passage(TypedDict):
    id: str
    section_id: str
    p_num: Tuple[int, int]
    text: str
    emb: List[float]
    member_ids: List[str]


class AnchorLink(TypedDict):
    entity_name: str
    passage_id: str
    score: float
    justification: str


def new_report(course_name: str) -> Dict:
    return {"course": course_name, "textbooks": [], "slides": [],
            "anchors": 0, "errors": []}
