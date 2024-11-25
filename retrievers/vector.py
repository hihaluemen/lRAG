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
        
        # 首次添加文档时初始化索引
        if self.index is None:
            self._init_index(embeddings.shape[1])
            
        # 添加向量到索引
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(documents)
        
    def retrieve(self, query: str) -> List[Document]:
        """检索相似文档"""
        if not self.documents or self.index is None:
            return []
            
        # 计算查询向量
        query_embedding = self.embedding_model.encode(query)
        
        # 搜索最相似的向量
        scores, indices = self.index.search(
            query_embedding.astype(np.float32), 
            min(self.top_k, len(self.documents))
        )
        
        # 应用相似度阈值
        if self.score_threshold is not None:
            mask = scores[0] >= self.score_threshold
            indices = indices[0][mask]
        else:
            indices = indices[0]
            
        # 返回相似文档
        return [self.documents[idx] for idx in indices]
        
    def save(self, path: str):
        """
        保存检索器状态
        
        Args:
            path: 保存目录路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存FAISS索引
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