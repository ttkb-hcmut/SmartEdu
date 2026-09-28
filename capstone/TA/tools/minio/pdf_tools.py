import json
import io
from typing import List, Union, Type, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

try:
    import fitz # PyMuPDF
except ImportError:
    fitz = None

from core.repo.graph.graphdb import GraphDB
from core.repo.storage.minio_repo import MinioDB, resolve_pdf_document

class ConceptInput(BaseModel):
    concept: str = Field(description="Concept name to locate in the document.")

class GetConcept(BaseTool):
    name: str = "get_concept_page"
    description: str = "Find the page of a concept in the document fast via the Knowledge Graph."
    args_schema: Type[BaseModel] = ConceptInput
    engine: GraphDB

    def _run(self, concept: str) -> str:
        # concept -> anchored passage -> (minio pdf, page) for FEToPage
        res = self.engine.get_concept_page(concept)
        if not res:
            return json.dumps({"error": f"Concept '{concept}' not found in the document."})
        return json.dumps(
            {"concept_name": res.get("concept"), "destination": res["uri"], "page": res["page"]},
            ensure_ascii=False
        )

class PagesInput(BaseModel):
    pages: List[int] = Field(description="Pages to read content from (e.g. [5, 6]).")
    destination: str = Field(description="MinIO file path (from active_resource).")

class GetPages(BaseTool):
    name: str = "get_pdf_pages"
    description: str = "Extract text content from specific pages of the open PDF file."
    args_schema: Type[BaseModel] = PagesInput
    minio: MinioDB

    def _run(self, pages: List[int], destination: str) -> str:
        if fitz is None:
            return json.dumps({"error": "PyMuPDF (fitz) is not installed."})
        try:
            # Parse destination: courses/ML/slide_1/slide_1.pdf
            obj_name = destination.replace(f"minio://{self.minio.bucket_name}/", "")
            if obj_name.startswith("minio://"):
                obj_name = obj_name.split("/", 3)[-1]

            resolved, _ = resolve_pdf_document(obj_name)
            if resolved:
                obj_name = resolved
            
            response = self.minio.client.get_object(self.minio.bucket_name, obj_name)
            pdf_data = response.read()
            response.close()
            response.release_conn()
            
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            content = []
            for p in pages:
                if p > 0 and p <= len(doc):
                    page_obj = doc.load_page(p - 1)
                    content.append(f"--- PAGE {p} ---\n{page_obj.get_text()}")
                else:
                    content.append(f"--- PAGE {p} (DOES NOT EXIST) ---")
            
            return "\n\n".join(content)
        except Exception as e:
            return json.dumps({"error": f"Error reading PDF from MinIO: {str(e)}"})

class FEToolInput(BaseModel):
    page: int = Field(description="Page number for the frontend to flip to.")
    destination: str = Field(description="Target document (from active_resource).")

class FEToPage(BaseTool):
    name: str = "navigate_frontend_page"
    description: str = "Emit a control signal for the frontend to flip to a specific page of the document."
    args_schema: Type[BaseModel] = FEToolInput
    
    def _run(self, page: int, destination: str) -> str:
        resolved, _ = resolve_pdf_document(destination)
        if resolved and "/chunks/" in destination:
            prefix = destination.split("/", 3)[:3] if destination.startswith("minio://") else []
            destination = "/".join(prefix) + "/" + resolved if prefix else resolved
            page = 1

        res = {
            "action": "NAVIGATE_PDF",
            "destination": destination,
            "page": page
        }
        return json.dumps(res, ensure_ascii=False)
