import time
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
from .model import RerankerModel


class RerankerRetriever(BaseRetriever):
    _vector_retriever = PrivateAttr()
    _reranker = PrivateAttr()
    _top_k_vector = PrivateAttr(default=20)
    _top_k_final = PrivateAttr(default=5)

    def __init__(self, vector_retriever, reranker: Optional[RerankerModel] = None, top_k_vector: int = 20, top_k_final: int = 5):
        super().__init__()
        self._vector_retriever = vector_retriever
        self._reranker = reranker
        self._top_k_vector = top_k_vector
        self._top_k_final = top_k_final

    def _get_relevant_documents(self, query: str) -> List[Document]:
        start_time = time.time()
        vector_docs = self._vector_retriever.get_relevant_documents(query)
        _ = time.time() - start_time
        if self._reranker is None:
            return vector_docs[:self._top_k_final]
        try:
            return self._reranker.rerank(query, vector_docs, self._top_k_final)
        except Exception:
            return vector_docs[:self._top_k_final]


