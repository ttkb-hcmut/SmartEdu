from sympy import false

from TA.helper.schema import RAGCore, RAGDeep, RoadmapExplore, TeachLectureOutput
from core.schema.wf_state import TAOutput

AGENT_SPECS = {
    "RAG": {
        "tools": lambda tf: tf.get_tools(agent_name="RAG"),
        "schema": RAGCore | RAGDeep | RoadmapExplore,
        "debug": False
    },
    "TA": {
        "tools": lambda tf: tf.get_teach_tools(),
        "schema": TAOutput | TeachLectureOutput,
        "debug": False
    }
}
