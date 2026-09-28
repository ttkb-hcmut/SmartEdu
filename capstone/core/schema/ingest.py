from datetime import datetime, timezone
from typing import Dict, List, Tuple, TypedDict


REPORT_RUNNING = "RUNNING"
REPORT_PARTIAL = "PARTIAL"
REPORT_FAILED = "FAILED"
REPORT_COMPLETED = "COMPLETED"
TERMINAL_REPORT_STATUSES = frozenset({
    REPORT_PARTIAL,
    REPORT_FAILED,
    REPORT_COMPLETED,
})


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


class ParsedSlideResult(TypedDict):
    chunks: List[SlideChunk]


class ParsedTextbookResult(TypedDict):
    sections: List[Dict]
    items: List[TreeItem]


class PublishedSlideItem(TypedDict):
    index: int
    heading: str
    content: str
    hard_ref: Dict


class KGExtractionResult(TypedDict):
    nodes: List[Dict]
    edges: List[Dict]
    clusters: List[Dict]


class TranscriptSegment(TypedDict):
    t_lo: float
    t_hi: float
    text: str


class TranscriptResult(TypedDict):
    duration: float
    segments: List[TranscriptSegment]


def new_report(course_name: str, run_id: str = "local",
               started_at: str = None) -> Dict:
    return {
        "course": course_name,
        "run_id": run_id,
        "status": REPORT_RUNNING,
        "started_at": started_at or datetime.now(timezone.utc).isoformat(),
        "textbooks": [],
        "slides": [],
        "videos": [],
        "anchors": 0,
        "errors": [],
        "stage_timings_ms": {},
    }
