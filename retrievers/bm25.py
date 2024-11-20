from typing import List, Optional, Union, Tuple, Set
import numpy as np
from core.document import Document
from rank_bm25 import BM25Okapi
from utils.tokenizer import ChineseSimpleTokenizer
import jieba


class BM25Retriever:
    """基于BM25算法的检索器"""

    def __init__(
        self,
        stopwords_files: List[str] = None,
        user_dict_path: Optional[str] = None
    ):
        """初始化检索器
        
        Args:
            stopwords_files: 额外的停用词文件路径列表
            user_dict_path: 用户自定义词典路径
        """
        # 初始化分词器
        self.tokenizer = ChineseSimpleTokenizer(stopwords_files)
        
        # 加载用户词典
        if user_dict_path:
            self._load_user_dict(user_dict_path)
            
        self.documents: List[Document] = []
        self.bm25 = None
        self.tokenized_docs = None

    def _load_user_dict(self, dict_path: str):
        """加载用户自定义词典"""
        try:
            jieba.load_userdict(dict_path)
        except Exception as e:
            print(f"Warning: Failed to load user dictionary: {e}")

    def preprocess(self, text: str) -> List[str]:
        """文本预处理"""
        return self.tokenizer.tokenize(text)

    def add_documents(self, documents: List[Document]):
        """添加文档到索引"""
        self.documents.extend(documents)
        # 对所有文档进行分词
        self.tokenized_docs = [
            self.preprocess(doc.content)
            for doc in self.documents
        ]
        # 创建BM25索引
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(
            self,
            query: str,
            top_k: int = 5,
            return_scores: bool = False
    ) -> List[Union[Document, Tuple[Document, float]]]:
        """检索相关文档

        Args:
            query: 查询文本
            top_k: 返回前k个相关文档
            return_scores: 是否返回分数

        Returns:
            如果return_scores为False，返回Document列表
            如果return_scores为True，返回(Document, score)元组列表
        """
        if not self.bm25:
            return []

        # 对查询进行分词
        tokenized_query = self.preprocess(query)
        if not tokenized_query:
            return []

        # 计算相似度得分
        scores = self.bm25.get_scores(tokenized_query)

        # 获取top-k文档的索引
        top_indices = np.argsort(scores)[-top_k:][::-1]

        if return_scores:
            return [(self.documents[i], float(scores[i])) for i in top_indices]

        return [self.documents[i] for i in top_indices]

    def get_stopwords(self) -> Set[str]:
        """获取当前使用的停用词集合"""
        return self.tokenizer.get_stopwords()