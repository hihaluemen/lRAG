from typing import List
from pathlib import Path
from core.parser import BaseParser
from core.document import Document
from llama_index.readers.file import PDFReader
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter


class LlamaPDFParser(BaseParser):
    """基于LlamaIndex的PDF解析器"""

    def __init__(
            self,
            chunk_size: int = 512,
            chunk_overlap: int = 50,
            include_metadata: bool = True,
            include_prev_next_rel: bool = True  # 是否包含前后文关系
    ):
        """
        初始化解析器

        Args:
            chunk_size: 分块大小，默认512字符
            chunk_overlap: 重叠大小，默认50字符
            include_metadata: 是否包含元数据
            include_prev_next_rel: 是否在元数据中包含前后文关系
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel

        # 初始化PDF阅读器和分句器
        self.pdf_reader = PDFReader()
        self.parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel
        )

    def parse(self, file_path: str) -> List[Document]:
        """
        解析PDF文件

        Args:
            file_path: 文件路径

        Returns:
            Document列表
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件未找到: {file_path}")

        # 使用PDFReader读取文件
        llama_docs = self.pdf_reader.load_data(
            file=str(file_path),
        )

        # 解析文档获取节点
        nodes = self.parser.get_nodes_from_documents(llama_docs)

        # 转换为我们的Document格式
        documents = []
        for i, node in enumerate(nodes):
            metadata = {
                'source': str(file_path),
                'type': 'pdf',
                'chunk_id': i,
                'file_name': file_path.name,
                'page_number': node.metadata.get('page_number', None)
            }

            # 添加LlamaIndex节点的元数据
            if node.metadata:
                metadata.update(node.metadata)

            # 添加前后文关系
            if self.include_prev_next_rel:
                if i > 0:
                    metadata['prev_chunk_id'] = i - 1
                if i < len(nodes) - 1:
                    metadata['next_chunk_id'] = i + 1

            documents.append(Document(
                content=node.text,
                metadata=metadata
            ))

        return documents

    def get_stats(self) -> dict:
        """获取解析器统计信息"""
        return {
            "type": self.__class__.__name__,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "include_metadata": self.include_metadata,
            "include_prev_next_rel": self.include_prev_next_rel
        } 