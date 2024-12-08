from typing import List, Optional, Dict, Any
import numpy as np
from core.retriever import BaseRetriever
from core.document import Document
from rank_bm25 import BM25Okapi
from utils.tokenizer import ChineseSimpleTokenizer
import pickle
import os
import jieba


class BM25Retriever(BaseRetriever):
    """基于BM25算法的检索器"""

    def __init__(
            self,
            top_k: int = 5,
            score_threshold: Optional[float] = None,
            stopwords_files: Optional[List[str]] = None,
            user_dict_path: Optional[str] = None
    ):
        """
        初始化BM25检索器

        Args:
            top_k: 返回的最相似文档数量
            score_threshold: 相似度阈值
            stopwords_files: 停用词文件路径列表
            user_dict_path: 用户自定义词典路径
        """
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        # 初始化分词器
        self.tokenizer = ChineseSimpleTokenizer(stopwords_files)
        
        # 加载用户词典
        if user_dict_path:
            self._load_user_dict(user_dict_path)

        # 初始化存储
        self.documents: List[Document] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25 = None

    def _load_user_dict(self, dict_path: str):
        """加载用户自定义词典"""
        try:
            jieba.load_userdict(dict_path)
        except Exception as e:
            print(f"Warning: Failed to load user dictionary: {e}")

    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到检索器"""
        if not documents:
            return
            
        self.documents.extend(documents)
        
        # 对所有文档进行分词
        self.tokenized_docs = [
            self.tokenizer.tokenize(doc.content)
            for doc in self.documents
        ]
        
        # 创建BM25索引
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """检索相关文档并返回分数

        Args:
            query: 查询文本

        Returns:
            List[tuple[Document, float]]: 文档和原始BM25分数的元组列表
        """
        if not self.documents or not self.bm25:
            return []

        # 对查询进行分词
        tokenized_query = self.tokenizer.tokenize(query)
        if not tokenized_query:
            return []

        # 计算相似度得分
        scores = self.bm25.get_scores(tokenized_query)
        
        # 应用相似度阈值
        if self.score_threshold is not None:
            mask = scores >= self.score_threshold
            valid_indices = np.where(mask)[0]
        else:
            valid_indices = np.arange(len(scores))
            
        # 获取top-k文档的索引和分数
        top_indices = np.argsort(scores[valid_indices])[-self.top_k:][::-1]
        result_indices = valid_indices[top_indices]
        result_scores = scores[result_indices]
        
        # 返回文档和原始分数的元组列表
        return [(self.documents[idx], float(score)) 
                for idx, score in zip(result_indices, result_scores)]

    def retrieve(self, query: str) -> List[Document]:
        """检索相关文档（仅返回文档，不返回分数）"""
        return [doc for doc, _ in self.retrieve_with_scores(query)]

    def save(self, path: str) -> None:
        """
        保存检索器状态
        
        Args:
            path: 保存目录路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存文档、分词结果和配置
        state = {
            "documents": self.documents,
            "tokenized_docs": self.tokenized_docs,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold
        }
        state_path = os.path.join(path, "state.pkl")
        with open(state_path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """
        加载检索器状态
        
        Args:
            path: 保存目录路径
        """
        state_path = os.path.join(path, "state.pkl")
        with open(state_path, "rb") as f:
            state = pickle.load(f)
            
        self.documents = state["documents"]
        self.tokenized_docs = state["tokenized_docs"]
        self.top_k = state["top_k"]
        self.score_threshold = state["score_threshold"]
        
        # 重新创建BM25索引
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return {
            "document_count": len(self.documents),
            "type": self.__class__.__name__,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold
        }

    def delete_documents(self, doc_ids: List[str]) -> List[str]:
        """删除指定ID的文档"""
        if not self.documents:
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
            
        # 更新文档列表和分词结果
        self.documents = [self.documents[i] for i in keep_indices]
        self.tokenized_docs = [self.tokenized_docs[i] for i in keep_indices]
        
        # 重建BM25索引
        if self.documents:
            self.bm25 = BM25Okapi(self.tokenized_docs)
        else:
            self.bm25 = None
            
        return deleted_ids