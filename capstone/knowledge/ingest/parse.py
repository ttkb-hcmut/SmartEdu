from typing import Dict, List

from core.config import Ingest_param
from core.util.file_extractor import extract_pdf, extract_tree
from knowledge.ingest.fetch import temp_pdf


def parse_slide_pdf(file_bytes: bytes, cfg: Ingest_param) -> List[Dict]:
    with temp_pdf(file_bytes) as path:
        return extract_pdf(
            path, pages_per_batch=cfg.PAGE_PER_SLIDE,
            step=cfg.PAGE_PER_SLIDE - cfg.slide_overlap
        )


def parse_textbook_tree(file_bytes: bytes) -> Dict:
    with temp_pdf(file_bytes) as path:
        return extract_tree(path)


def parse_textbook_chunks(file_bytes: bytes, cfg: Ingest_param) -> List[Dict]:
    with temp_pdf(file_bytes) as path:
        return extract_pdf(path, pages_per_batch=cfg.PAGE_PER_TB)
