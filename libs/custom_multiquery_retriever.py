from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.chains.llm import LLMChain
import logging

logger = logging.getLogger(__name__)


class CustomMultiQueryRetriever(MultiQueryRetriever):
    def _get_relevant_documents(
        self,
        query: str,
        history: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get relevant documents given a user query.

        Args:
            question: user query

        Returns:
            Unique union of relevant documents from all generated queries
        """
        queries = self.generate_queries(query, history, run_manager)
        if self.include_original:
            queries.append(query)
        documents = self.retrieve_documents(queries, run_manager)
        return self.unique_union(documents)

    def generate_queries(
        self, question: str, history: str, run_manager: CallbackManagerForRetrieverRun
    ) -> List[str]:
        """Generate queries based upon user input.

        Args:
            question: user query

        Returns:
            List of LLM generated queries that are similar to the user input
        """
        response = self.llm_chain.invoke(
            {"question": question, "history": history},
            config={"callbacks": run_manager.get_child()},
        )
        if isinstance(self.llm_chain, LLMChain):
            lines = response["text"]
        else:
            lines = response
        if self.verbose:
            logger.info(f"Generated queries: {lines}")
        return lines
