from langchain.globals import set_verbose
from loguru import logger

from application.rag.retriever import ContextRetriever
from infrastructure.opik_utils import configure_opik

if __name__ == "__main__":
    configure_opik()
    set_verbose(True)

    query = """
        My name is Aquib Ali Khan
        
        Could you draft a LinkedIn post discussing RAG systems?
        I'm particularly interested in:
            - how RAG works
            - how it is integrated with vector DBs and large language models (LLMs).
        """

    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=9)

    logger.info("Retrieved documents:")
    for rank, document in enumerate(documents):
        logger.info(f"{rank + 1}: {document}")
