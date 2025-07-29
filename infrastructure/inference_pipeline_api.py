import opik
from fastapi import FastAPI, HTTPException
from opik import opik_context
from pydantic import BaseModel

from application.rag.retriever import ContextRetriever
from application.utils import misc
from domain.embedded_chunks import EmbeddedChunk
from infrastructure.opik_utils import configure_opik
from model.inference import InferenceExecutor, LLMInferenceSagemakerEndpoint

from dotenv import load_dotenv
import os

# Load from .env file in current or parent directory
from dotenv import load_dotenv
load_dotenv()

configure_opik()

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=os.getenv("SAGEMAKER_ENDPOINT_INFERENCE"), inference_component_name=None
    )
    answer = InferenceExecutor(llm, query, context).execute()

    return answer


@opik.track
def rag(query: str) -> str:
    retriever = ContextRetriever(mock=False)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)

    answer = call_llm_service(query, context)

    opik_context.update_current_trace(
        tags=["rag"],
        metadata={
            "model_id": os.getenv("HF_MODEL_ID"),
            "embedding_model_id": os.getenv("TEXT_EMBEDDING_MODEL_ID"),
            "temperature": float(os.getenv("TEMPERATURE_INFERENCE")),
            "query_tokens": misc.compute_num_tokens(query),
            "context_tokens": misc.compute_num_tokens(context),
            "answer_tokens": misc.compute_num_tokens(answer),
        },
    )

    return answer


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
