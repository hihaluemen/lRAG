from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from core.document import Document


class BaseRetriever(ABC):
    """检索器基类，定义检索器的标准接口"""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        添加文档到检索器

        Args:
            documents: 文档列表
        """
        pass

    @abstractmethod
    def retrieve(self, query: str) -> List[Document]:
        """
        根据查询检索相关文档

        Args:
            query: 查询文本

        Returns:
            List[Document]: 相关文档列表
        """
        pass

    def batch_retrieve(self, queries: List[str]) -> List[List[Document]]:
        """
        批量检索文档（可选实现）

        Args:
            queries: 查询文本列表

        Returns:
            List[List[Document]]: 每个查询对应的相关文档列表
        """
        return [self.retrieve(query) for query in queries]
    
    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """
        检索相关文档并返回分数
        """
        pass

    def save(self, path: str) -> None:
        """
        保存检索器状态（可选实现）

        Args:
            path: 保存路径
        """
        raise NotImplementedError("Save method not implemented")

    def load(self, path: str) -> None:
        """
        加载检索器状态（可选实现）

        Args:
            path: 加载路径
        """
        raise NotImplementedError("Load method not implemented")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取检索器统计信息（可选实现）

        Returns:
            Dict[str, Any]: 统计信息字典
        """
        return {
            "document_count": 0,
            "type": self.__class__.__name__
        }

    @abstractmethod
    def delete_documents(self, doc_ids: List[str]) -> List[str]:
        """
        从检索器中删除指定ID的文档

        Args:
            doc_ids: 要删除的文档ID列表

        Returns:
            List[str]: 成功删除的文档ID列表
        """
        pass