# Util package
import hashlib
from time import time
from typing import List
import os
# Core Package
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HierarchicalChunker
# Project Modules
from core.schema.graph.type import Ref
import re
from core.config import PAGE_PER_PDF

def _convert_pdf(file_path: str):
    ## page image backlog, long-book RAM spike
    options = PdfPipelineOptions(ocr_batch_size=1, layout_batch_size=1,
                                 table_batch_size=1, queue_max_size=4)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    result = converter.convert(file_path)
    if result.status != ConversionStatus.SUCCESS or result.errors:
        raise RuntimeError(f"PDF conversion incomplete: {result.status}; {len(result.errors)} errors")
    return result.document

def extract_pdf(file_path: str, pages_per_batch: int = 15, step = None) -> List[dict]:
    start = time()
    doc = _convert_pdf(file_path)
    if step is None:
        step: int = int(pages_per_batch/3*2+1)
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    doc_name = os.path.splitext(os.path.basename(file_path))[0]

    raw_pages = {} 
    for item, _ in doc.iterate_items():
        if hasattr(item, 'text') and item.text:
            text = item.text.strip()
            if "GLYPH" in text or len(text) < 5: continue
            if re.search(r'Lecturer|Contact|PhD|University|Faculty', text, re.I): continue
            
            page_no = item.prov[0].page_no if item.prov else 1
            if page_no not in raw_pages:
                raw_pages[page_no] = []
            raw_pages[page_no].append(text)

    sorted_page_nos = sorted(raw_pages.keys())
    if not sorted_page_nos: return []
    
    max_page = max(sorted_page_nos)
    page_contents = {p: f"==== PAGE {p} ====\n" + "\n".join(raw_pages.get(p, [])) 
                     for p in range(1, max_page + 1) if raw_pages.get(p)}

    chunks = []
    for idx, start_page in enumerate(range(1, max_page + 1, step)):
        end_page: int = min(start_page + pages_per_batch - 1, max_page)
        
        window_text_list = []
        for p in range(start_page, end_page + 1):
            if page_contents.get(p):
                window_text_list.append(page_contents[p])

        combined_text = "\n".join(window_text_list)
        if not combined_text.strip(): continue

        chuk = {
            "doc_id": file_hash,
            "doc_name": doc_name,
            "chunk_id": f"{file_hash}_slide{idx}",
            "heading": f"{doc_name} - Part {idx+1}",
            "content": combined_text, 
            "page_num": (start_page, end_page)
        }
        chunks.append(chuk)
        #print(f"Heading: {chuk["heading"]} \n Content: {combined_text}")

        #print("==========================================================================================")


    print("==========================================================================================")
    print(f"Extraction completed in {time() - start:.2f}s. Created {len(chunks)} batch chunks.")
    return chunks

def create_refs(chunks: List[dict], storage_type: str = "minio") -> List[Ref]:
    return [
        Ref(
            db=storage_type,
            id=c["chunk_id"],
            name=c["doc_name"],
            summary=c["content"][:200],
            p_num=c["page_num"]  # already a (start, end) tuple
        )
        for c in chunks
    ]


def extract_tree(file_path: str) -> dict:
    ## docling authored hierarchy -> {sections tree, fine items} for :Section/:Passage build
    doc = _convert_pdf(file_path)

    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    doc_name = os.path.splitext(os.path.basename(file_path))[0]

    root_id = f"{file_hash}_sec_root"
    sections = {
        "": {"id": root_id, "title": doc_name, "level": 0,
             "parent_id": None, "p_num": (1, 1), "order": 0}
    }
    items: List[dict] = []

    for order, ch in enumerate(HierarchicalChunker().chunk(doc)):
        text = (ch.text or "").strip()
        if len(text) < 5:
            continue

        meta = ch.meta
        pages = [it.prov[0].page_no for it in getattr(meta, "doc_items", []) if it.prov]
        p_lo, p_hi = (min(pages), max(pages)) if pages else (1, 1)
        headings = [h for h in (getattr(meta, "headings", None) or []) if h and h.strip()]

        parent_id, path = root_id, []
        for lvl, h in enumerate(headings):
            path.append(h.strip())
            key = " > ".join(path)
            if key not in sections:
                sid = f"{file_hash}_sec_{hashlib.md5(key.encode()).hexdigest()[:8]}"
                sections[key] = {"id": sid, "title": h.strip(), "level": lvl + 1,
                                 "parent_id": parent_id, "p_num": (p_lo, p_hi), "order": len(sections)}
            else:  # widen ancestor span as children stream in
                lo, hi = sections[key]["p_num"]
                sections[key]["p_num"] = (min(lo, p_lo), max(hi, p_hi))
            parent_id = sections[key]["id"]

        items.append({"id": f"{file_hash}_item_{order}", "section_id": parent_id,
                      "text": text, "p_num": (p_lo, p_hi), "order": order})

    # widen root span to cover the whole doc
    if items:
        sections[""]["p_num"] = (min(i["p_num"][0] for i in items),
                                 max(i["p_num"][1] for i in items))

    return {"doc_id": file_hash, "doc_name": doc_name,
            "sections": list(sections.values()), "items": items}
