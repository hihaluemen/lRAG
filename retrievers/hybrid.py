from typing import List, Dict, Any
from core.retriever import BaseRetriever
from core.document import Document


class HybridRetriever(BaseRetriever):
    """融合检索器"""

    def __init__(
            self,
            retrievers: List[BaseRetriever],
            top_k: int = 5,
            reranker=None
    ):
        self.retrievers = retrievers
        self.top_k = top_k
        self.reranker = reranker

    def add_documents(self, documents: List[Document]) -> None:
        """向所有检索器添加文档"""
        for retriever in self.retrievers:
            retriever.add_documents(documents)

    def retrieve(self, query: str) -> List[Document]:
        """执行融合检索"""
        # 获取所有检索器的结果
        all_docs = []
        seen_ids = set()  # 用于去重的id集合

        # 从每个检索器获取结果并去重
        for retriever in self.retrievers:
            docs = retriever.retrieve(query)
            for doc in docs:
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    all_docs.append(doc)

        # 如果有重排序器，进行重排序
        if self.reranker is not None:
            all_docs = self.reranker.rerank(query, all_docs)

        # 返回top_k个文档
        return all_docs[:self.top_k]

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return {
            "type": self.__class__.__name__,
            "retrievers": [
                retriever.__class__.__name__
                for retriever in self.retrievers
            ],
            "top_k": self.top_k
        }