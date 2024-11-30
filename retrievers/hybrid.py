from typing import List, Dict, Any, Optional
from core.retriever import BaseRetriever
from core.document import Document
import os
import pickle
import numpy as np


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

    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """执行融合检索并返回分数

        Args:
            query: 查询文本

        Returns:
            List[tuple[Document, float]]: 文档和融合后分数的元组列表
        """
        # 获取所有检索器的结果和分数
        retriever_results = []
        for retriever in self.retrievers:
            results = retriever.retrieve_with_scores(query)
            if results:  # 如果有结果，进行归一化
                scores = np.array([score for _, score in results])
                min_score = np.min(scores)
                max_score = np.max(scores)
                if max_score > min_score:  # 避免除以零
                    normalized_scores = (scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores = np.ones_like(scores)
                retriever_results.append([(doc, float(score)) for (doc, _), score 
                                        in zip(results, normalized_scores)])
            else:
                retriever_results.append([])

        # 融合结果
        doc_scores = {}  # 用于存储每个文档的加权分数
        seen_ids = set()  # 用于去重的id集合
        all_results = []

        # 从每个检索器获取结果并计算加权分数
        for results, weight in zip(retriever_results, self.retriever_weights):
            if weight <= 0:  # 跳过权重为0的检索器
                continue
                
            for doc, score in results:
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    if doc.id not in doc_scores:
                        doc_scores[doc.id] = 0.0
                    doc_scores[doc.id] += score * weight
                    all_results.append((doc, doc_scores[doc.id]))

        # 如果有重排序器，进行重排序
        if self.reranker is not None and all_results:
            docs = [doc for doc, _ in all_results]
            reranked_docs = self.reranker.rerank(query, docs)
            # 使用重排序后的顺序重新计算分数
            all_results = [(doc, 1.0 - i/len(reranked_docs)) for i, doc in enumerate(reranked_docs)]

        # 按分数排序并返回top_k个结果
        all_results = sorted(all_results, key=lambda x: x[1], reverse=True)
        return all_results[:self.top_k]

    def retrieve(self, query: str) -> List[Document]:
        """检索相似文档（仅返回文档，不返回分数）"""
        return [doc for doc, _ in self.retrieve_with_scores(query)]

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