from typing import List, Dict
from TA.tools.neo.retriever import EntityFinder, RhetoricalRetriever, EdgeExplorer
from TA.tools.retrieval import SemanticSearch, TextbookSearch
from TA.tools.neo.explore import RecommendNew, CourseBackbone, CourseRelevance, OptimalPath
from TA.tools.neo.course_tree import CourseTree
from TA.tools.minio.pdf_tools import GetConcept, GetPages, FEToPage
from TA.tools.student.context_tools import RecallToolResults, RecallThoughts, InspectChatHistory

from core.repo.graph.graphdb import GraphDB


class ToolFactory:
    def __init__(self, graph_db: GraphDB, milvus_db, embedder, minio, tracker=None):
        self.graph_db = graph_db
        self.milvus_db = milvus_db
        self.embedder = embedder
        self.minio = minio
        self.tracker = tracker

        self.agent_tools = {
            "RAG": self.get_retrieve_tools,
            "TA": self.get_teach_tools,
            "Generator": lambda: []
        }

    def get_retrieve_tools(self) -> List:
        return [
            EntityFinder(engine=self.graph_db, tracker=self.tracker),
            RhetoricalRetriever(engine=self.graph_db, tracker=self.tracker),
            EdgeExplorer(engine=self.graph_db, tracker=self.tracker),
            RecommendNew(engine=self.graph_db, tracker=self.tracker),
            CourseBackbone(engine=self.graph_db, tracker=self.tracker),
            CourseTree(engine=self.graph_db, tracker=self.tracker,
                       mongo=self.tracker.mongodb if self.tracker else None),
            CourseRelevance(engine=self.graph_db, tracker=self.tracker),
            OptimalPath(engine=self.graph_db, tracker=self.tracker),
            SemanticSearch(milvus_db=self.milvus_db, embedder=self.embedder),
            TextbookSearch(graph_db=self.graph_db, embedder=self.embedder),
        ]

    def get_teach_tools(self) -> List:
        """
        TA agent tools (always bound at startup, context injected via ContextVar per-request):
        - get_concept_page        : find which PDF page a concept is on
        - get_pdf_pages           : read actual page content (TA must call this before lecturing)
        - navigate_frontend_page  : signal FE to navigate to a page
        - recall_tool_results     : retrieve recent tool outputs from session context
        - recall_thoughts         : retrieve recent agent reasoning from session context
        """
        return [
            GetConcept(engine=self.graph_db),
            GetPages(minio=self.minio),
            FEToPage(),
            RecallToolResults(),
            RecallThoughts(),
            InspectChatHistory(),
        ]

    def get_teach_lookup_tools(self) -> Dict:
        """Raw tool instances for deterministic Teach_Lookup node (no LLM)."""
        return {
            "get_concept": GetConcept(engine=self.graph_db),
            "get_pages": GetPages(minio=self.minio),
        }

    def get_tools(self, agent_name: str):
        if agent_name not in self.agent_tools:
            print(f"Unknown agent: {agent_name}")
            return []
        return self.agent_tools[agent_name]()
