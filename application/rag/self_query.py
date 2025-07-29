import opik
from langchain_openai import ChatOpenAI
from loguru import logger

from application import utils
from domain.documents import UserDocument
from domain.queries import Query
from dotenv import load_dotenv
import os

# Load from .env file in current or parent directory
from dotenv import load_dotenv
load_dotenv()

from .base import RAGStep
from .prompt_templates import SelfQueryTemplate


class SelfQuery(RAGStep):
    # @opik.track(name="SelfQuery.generate")
    def generate(self, query: Query) -> Query:
        if self._mock:
            return query

        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(model=os.getenv("OPENAI_MODEL_ID"), api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

        chain = prompt | model

        response = chain.invoke({"question": query})
        user_full_name = response.content.strip("\n ")

        if user_full_name == "none":
            return query

        first_name, last_name = utils.split_user_full_name(user_full_name)
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

        query.author_id = user.id
        query.author_full_name = user.full_name

        return query


if __name__ == "__main__":
    query = Query.from_str("I am Aquib Ali Khan. Write an article about the best types of advanced RAG methods.")
    self_query = SelfQuery()
    query = self_query.generate(query)
    logger.info(f"Extracted author_id: {query.author_id}")
    logger.info(f"Extracted author_full_name: {query.author_full_name}")
