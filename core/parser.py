from abc import ABC, abstractmethod
from typing import List
from .document import Document


class BaseParser(ABC):
    """解析器基类"""
    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """解析文档
        Args:
            file_path: 文件路径
        Returns:
            Document列表
        """
        pass
    
    @abstractmethod
    def split(self, doc: Document, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """分割文档
        Args:
            doc: 输入文档
            chunk_size: 分块大小
            chunk_overlap: 重叠大小
        Returns:
            分割后的Document列表
        """
        pass