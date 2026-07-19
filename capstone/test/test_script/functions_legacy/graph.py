from time import time
from core.repo.graph.test.doc import DOC
from core.repo.graph.graphdb import GraphDB
from core.llm.llm_engine import CoreLLMEngine
from knowledge.engine.extract import GraphExtractionService

def test_extract():
    texts = ["""
            The primary Machine Learning Objective is to build a good model. 
             A Machine Learning Algorithm includes Decision Tree and Linear Regression. 
             Linear Regression is a estimate output via linear equation, while Logistic Regression use various statistic method
             A fast method is needed for real-time inference because good performance is essential. 
             However, the main limitation of RNN is vanishing gradient. 
             To clarify, Deep Learning definition involves layers of neural networks, whereas applications of Computer Vision include face recognition. 
             Finally, the Xyz123 Algorithm is a hypothetical concept used for testing.
    """]

    llm_engine = CoreLLMEngine()
    
    extractor = GraphExtractionService(llm_engine=llm_engine)

    kg = extractor.extract_and_build(texts=DOC, save="kg_v1.json")
    """   
    start = time()
    graphDB = GraphDB(uri="bolt://localhost:7687", auth=("neo4j", "graph123"))
    print("================================ Start Inserting ================================ ")
    graphDB.reset("test")
    graphDB.import_graph("test", kg)

    print("insertion takes: ", time() - start)
    """
if __name__ == "__main__":
    test_extract()