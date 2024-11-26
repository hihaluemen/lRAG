from typing import List, Optional
from core.parser import BaseParser
from core.document import Document
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from pathlib import Path


class LlamaMarkdownParser(BaseParser):
    """基于LlamaIndex的Markdown解析器"""

    def __init__(
            self,
            chunk_size: int = 512,
            chunk_overlap: int = 50,
            split_mode: str = "markdown",  # 'markdown' or 'sentence'
            include_metadata: bool = True,
            include_prev_next_rel: bool = True  # 是否包含前后文关系
    ):
        """
        初始化解析器

        Args:
            chunk_size: 分块大小，默认512字符
            chunk_overlap: 重叠大小，默认50字符
            split_mode: 分割模式，可选 'markdown' 或 'sentence'
            include_metadata: 是否包含元数据
            include_prev_next_rel: 是否在元数据中包含前后文关系
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_mode = split_mode
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel

        # 初始化解析器
        if split_mode == "markdown":
            self.parser = MarkdownNodeParser.from_defaults(
                include_metadata=include_metadata,
                include_prev_next_rel=include_prev_next_rel
            )
        else:  # sentence mode
            self.parser = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                include_metadata=include_metadata,
                include_prev_next_rel=include_prev_next_rel
            )

    def parse(self, file_path: str) -> List[Document]:
        """
        解析Markdown文件

        Args:
            file_path: 文件路径

        Returns:
            Document列表
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件未找到: {file_path}")

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 创建LlamaIndex文档
        llama_doc = LlamaDocument(
            text=content,
            metadata={
                'source': str(file_path),
                'type': 'markdown',
                'file_name': file_path.name
            }
        )

        # 解析文档获取节点
        nodes = self.parser.get_nodes_from_documents([llama_doc])

        # 转换为我们的Document格式
        documents = []
        for i, node in enumerate(nodes):
            metadata = {
                'source': str(file_path),
                'type': 'markdown',
                'chunk_id': i,
                'file_name': file_path.name
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
            "split_mode": self.split_mode,
            "include_metadata": self.include_metadata,
            "include_prev_next_rel": self.include_prev_next_rel
        }


# 使用示例
if __name__ == "__main__":
    # 测试不同的分割模式
    test_file = "../examples/test_docs/sample.md"

    # Markdown模式
    print("\n=== Markdown 模式 ===")
    md_parser = LlamaMarkdownParser(
        split_mode="markdown",
        chunk_size=512,
        chunk_overlap=50
    )
    md_docs = md_parser.parse(test_file)
    print(f"解析得到 {len(md_docs)} 个文档块")
    
    # 打印第一个文档的内容和元数据
    if md_docs:
        print("\n第一个文档块:")
        print(f"内容: {md_docs[0].content[:200]}...")
        print(f"元数据: {md_docs[0].metadata}")

    # 句子模式
    print("\n=== 句子模式 ===")
    sent_parser = LlamaMarkdownParser(
        split_mode="sentence",
        chunk_size=512,
        chunk_overlap=50
    )
    sent_docs = sent_parser.parse(test_file)
    print(f"解析得到 {len(sent_docs)} 个文档块")
    
    # 打印第一个文档的内容和元数据
    if sent_docs:
        print("\n第一个文档块:")
        print(f"内容: {sent_docs[0].content[:200]}...")
        print(f"元数据: {sent_docs[0].metadata}") 