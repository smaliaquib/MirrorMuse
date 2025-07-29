from __future__ import annotations

from domain.inference import Inference
from dotenv import load_dotenv
import os

# Load from .env file in current or parent directory
from dotenv import load_dotenv
load_dotenv()


class InferenceExecutor:
    def __init__(
        self,
        llm: Inference,
        query: str,
        context: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.query = query
        self.context = context if context else ""

        if prompt is None:
            self.prompt = """
            You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
                User query: {query}
                Context: {context}
            """
        else:
            self.prompt = prompt

    def execute(self) -> str:
        self.llm.set_payload(
            inputs=self.prompt.format(query=self.query, context=self.context),
            parameters={
                "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS_INFERENCE")),
                "repetition_penalty": 1.1,
                "temperature": float(os.getenv("TEMPERATURE_INFERENCE")),
            },
        )
        answer = self.llm.inference()[0]["generated_text"]

        return answer
