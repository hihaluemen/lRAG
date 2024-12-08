from typing import List, Optional
import numpy as np
import faiss
import pickle
import os
from core.retriever import BaseRetriever
from core.document import Document
from core.embedding import BaseEmbedding


class VectorRetriever(BaseRetriever):
    """基于FAISS的向量检索器实现"""

    def __init__(
            self,
            embedding_model: BaseEmbedding,
            dimension: Optional[int] = None,  # 向量维度
            top_k: int = 5,
            score_threshold: Optional[float] = None,
            index_type: str = "L2"  # L2 或 IP (内积)
    ):
        """
        初始化向量检索器

        Args:
            embedding_model: 向量化模型
            dimension: 向量维度，如果为None则在首次添加文档时自动设置
            top_k: 返回的最相似文档数量
            score_threshold: 相似度阈值
            index_type: 索引类型，支持"L2"(欧氏距离)或"IP"(内积)
        """
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.index_type = index_type
        
        self.documents: List[Document] = []
        self.index = None
        
    def _init_index(self, dimension: int):
        """初始化FAISS索引"""
        if self.index_type == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        else:  # IP
            self.index = faiss.IndexFlatIP(dimension)
        self.dimension = dimension
            
    def add_documents(self, documents: List[Document]):
        """添加文档到检索器"""
        if not documents:
            return
            
        # 获取文档向量
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        # 归一化向量
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # 首次添加文档时初始化索引
        if self.index is None:
            self._init_index(embeddings.shape[1])
            
        # 添加向量到索引
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(documents)
        
    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """检索相似文档，并返回文档和相似度分数

        Args:
            query: 查询文本

        Returns:
            List[tuple[Document, float]]: 文档和原始相似度分数的元组列表
        """
        if not self.documents or self.index is None:
            return []
            
        # 计算查询向量并归一化
        query_embedding = self.embedding_model.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # 搜索最相似的向量
        scores, indices = self.index.search(
            query_embedding.astype(np.float32), 
            min(self.top_k, len(self.documents))
        )
        
        # 应用相似度阈值并组合文档和原始分数
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if self.score_threshold is None or score >= self.score_threshold:
                results.append((self.documents[idx], float(score)))
            
        return results
        
    def retrieve(self, query: str) -> List[Document]:
        """检索相似文档（仅返回文档，不返回分数）"""
        return [doc for doc, _ in self.retrieve_with_scores(query)]
        
    def save(self, path: str):
        """
        保存检索器状态
        
        Args:
            path: 保存目录路径
        """
        os.makedirs(path, exist_ok=True)
        
        # ��存FAISS索引
        index_path = os.path.join(path, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # 保存文档和配置
        state = {
            "documents": self.documents,
            "dimension": self.dimension,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "index_type": self.index_type
        }
        state_path = os.path.join(path, "state.pkl")
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
            
    def load(self, path: str):
        """
        加载检索器状态
        
        Args:
            path: 保存目录路径
        """
        # 加载FAISS索引
        index_path = os.path.join(path, "index.faiss")
        self.index = faiss.read_index(index_path)
        
        # 加载文档和配置
        state_path = os.path.join(path, "state.pkl")
        with open(state_path, "rb") as f:
            state = pickle.load(f)
            
        self.documents = state["documents"]
        self.dimension = state["dimension"]
        self.top_k = state["top_k"]
        self.score_threshold = state["score_threshold"]
        self.index_type = state["index_type"]
        
    def get_stats(self) -> dict:
        """获取检索器统计信息"""
        return {
            "document_count": len(self.documents),
            "index_type": self.index_type,
            "dimension": self.dimension,
            "type": self.__class__.__name__
        }
        
    def delete_documents(self, doc_ids: List[str]) -> List[str]:
        """删除指定ID的文档"""
        if not self.documents or self.index is None:
            return []
            
        # 找出要保留的文档索引
        keep_indices = []
        deleted_ids = []
        
        for i, doc in enumerate(self.documents):
            if doc.id in doc_ids:
                deleted_ids.append(doc.id)
            else:
                keep_indices.append(i)
                
        if not deleted_ids:
            return []
            
        # 重建索引
        keep_docs = [self.documents[i] for i in keep_indices]
        if keep_docs:
            # 获取保留文档的向量
            embeddings = self.embedding_model.encode([doc.content for doc in keep_docs])
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # 重新初始化索引
            self._init_index(self.dimension)
            self.index.add(embeddings.astype(np.float32))
        else:
            # 如果没有文档了，重置索引
            self.index = None
            
        self.documents = keep_docs
        return deleted_ids