from typing import List, Dict, Any, Optional
from core.retriever import BaseRetriever
from core.document import Document
import os
import pickle


class HybridRetriever(BaseRetriever):
    """融合检索器"""

    def __init__(
            self,
            retrievers: List[BaseRetriever],
            top_k: int = 5,
            reranker = None,
            retriever_weights: Optional[List[float]] = None,  # 检索器权重
            pre_rerank_top_k: Optional[int] = None  # 重排序前保留的文档数量
    ):
        """
        初始化融合检索器
        
        Args:
            retrievers: 检索器列表
            top_k: 最终返回的文档数量
            reranker: 重排序器
            retriever_weights: 各检索器的权重，默认为None(均等权重)
            pre_rerank_top_k: 重排序前保留的文档数量，默认为None(全部保留)
        """
        self.retrievers = retrievers
        self.top_k = top_k
        self.reranker = reranker
        
        # 设置检索器权重
        if retriever_weights is None:
            self.retriever_weights = [1.0] * len(retrievers)
        else:
            if len(retriever_weights) != len(retrievers):
                raise ValueError("检索器权重数量必须与检索器数量相同")
            self.retriever_weights = retriever_weights
            
        self.pre_rerank_top_k = pre_rerank_top_k or top_k * 3  # 默认为top_k的3倍

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
        for retriever, weight in zip(self.retrievers, self.retriever_weights):
            if weight <= 0:  # 跳过权重为0的检索器
                continue
                
            docs = retriever.retrieve(query)
            for doc in docs:
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    all_docs.append(doc)
                    
            # 控制重排序前的文档数量
            if len(all_docs) >= self.pre_rerank_top_k:
                all_docs = all_docs[:self.pre_rerank_top_k]
                break

        # 如果有重排序器，进行重排序
        if self.reranker is not None and all_docs:
            all_docs = self.reranker.rerank(query, all_docs)

        # 返回top_k个文档
        return all_docs[:self.top_k]

    def save(self, path: str) -> None:
        """
        保存检索器状态
        
        Args:
            path: 保存目录路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存各个检索器
        for i, retriever in enumerate(self.retrievers):
            retriever_path = os.path.join(path, f"retriever_{i}")
            os.makedirs(retriever_path, exist_ok=True)
            retriever.save(retriever_path)
            
        # 保存配置
        config = {
            "top_k": self.top_k,
            "retriever_weights": self.retriever_weights,
            "pre_rerank_top_k": self.pre_rerank_top_k
        }
        config_path = os.path.join(path, "config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(config, f)

    def load(self, path: str) -> None:
        """
        加载检索器状态
        
        Args:
            path: 保存目录路径
        """
        # 加载各个检索器
        for i, retriever in enumerate(self.retrievers):
            retriever_path = os.path.join(path, f"retriever_{i}")
            retriever.load(retriever_path)
            
        # 加载配置
        config_path = os.path.join(path, "config.pkl")
        with open(config_path, "rb") as f:
            config = pickle.load(f)
            
        self.top_k = config["top_k"]
        self.retriever_weights = config["retriever_weights"]
        self.pre_rerank_top_k = config["pre_rerank_top_k"]

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        retriever_stats = []
        for retriever, weight in zip(self.retrievers, self.retriever_weights):
            stats = retriever.get_stats()
            stats["weight"] = weight
            retriever_stats.append(stats)
            
        return {
            "type": self.__class__.__name__,
            "retrievers": retriever_stats,
            "top_k": self.top_k,
            "pre_rerank_top_k": self.pre_rerank_top_k,
            "has_reranker": self.reranker is not None
        }