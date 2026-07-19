import re
from typing import Dict, List, Tuple

try:
    import fitz
except ImportError:
    fitz = None

from core.schema.graph.type import Ref
from core.repo.storage.minio_repo import make_topic_name


def clean_content(text: str) -> str:
    pattern = r'==== PAGE \d+ ====\s*\n?'
    return re.sub(pattern, '', text).strip()


def clean_slide_name(name: str) -> str:
    if not name: return ""

    name = re.sub(r'\.[a-z0-9]+$', '', name, flags=re.IGNORECASE)

    name = re.sub(r'^(chapter|chap|ch|slide|lecture|bài|chương)\s*\d*\s*[:\-\.]?\s*', '', name, flags=re.IGNORECASE)

    name = re.sub(r'[^\w\s]', ' ', name).lower().strip()
    name = re.sub(r'\s+', ' ', name)

    return name


def slice_pages_pdf(doc, start_page: int, end_page: int) -> bytes:
    sub = fitz.open()
    sub.insert_pdf(doc, from_page=start_page - 1, to_page=end_page - 1)
    data = sub.tobytes()
    sub.close()
    return data


def publish_slide_chunks(minio_repo, file_bytes: bytes, chunks: List[Dict],
                         course_name: str, file_name: str) -> Tuple[List, List, List]:
    ## per chunk: page-pdf slice + text -> minio topic folder, Ref built once here
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed — cannot split page pdfs.")
    big_doc = fitz.open(stream=file_bytes, filetype="pdf")

    texts, hard_refs, items = [], [], []
    for i, c in enumerate(chunks):
        content = c.get('content', None)
        if not content: continue
        content: str = clean_content(text=content)

        heading = c.get('heading') or file_name
        texts.append((heading, content))

        chunk_id = c.get("chunk_id")
        topic = make_topic_name(file_name=file_name, heading=c.get("heading"), chunk_id=chunk_id)

        start_p, end_p = c["page_num"]
        page_pdf = slice_pages_pdf(big_doc, start_p, end_p)
        minio_repo.upload_topic_pdf(topic=topic, course_name=course_name, file_data=page_pdf)

        uri = minio_repo.upload_chunk(
            chunk_id=chunk_id, content=c.get("content"), topic=topic, course_name=course_name
        )

        ref = Ref(
            db="minio",
            id=uri,
            name=clean_slide_name(heading),
            summary=f"{content[:100]} ......".replace("\n", " "),
            p_num=c["page_num"]
        )
        hard_refs.append(ref)
        items.append((i, heading, content, ref))

    big_doc.close()
    return texts, hard_refs, items
