import re
from pathlib import Path
from typing import List
from core.parser import BaseParser
from core.document import Document


class MarkdownParser(BaseParser):
    """Markdown文档解析器"""

    def __init__(self):
        # 用于提取markdown标题的正则表达式
        self.header_pattern = re.compile(r'^#+\s+(.+)$', re.MULTILINE)

    def parse(self, file_path: str) -> List[Document]:
        """解析Markdown文件

        策略:
        1. 读取文件内容
        2. 提取元数据(如标题等)
        3. 返回Document对象
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取第一个标题作为文档标题
        title_match = self.header_pattern.search(content)
        title = title_match.group(1) if title_match else file_path.stem

        # 创建元数据
        metadata = {
            'title': title,
            'source': str(file_path),
            'type': 'markdown'
        }

        # 创建完整文档
        full_doc = Document(content=content, metadata=metadata)

        # 分割文档
        return self.split(full_doc)

    def split(self, doc: Document, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """分割Markdown文档

        策略:
        1. 按段落分割
        2. 保持markdown格式
        3. 确保分割点在合适的位置
        """
        content = doc.content
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', content)

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # 如果当前段落加入后超过chunk_size,则启动新的chunk
            if current_size + para_size > chunk_size and current_chunk:
                # 创建新的Document
                chunk_content = '\n\n'.join(current_chunk)
                chunk_metadata = doc.metadata.copy()
                chunk_metadata['chunk_id'] = len(chunks)
                chunks.append(Document(content=chunk_content, metadata=chunk_metadata))

                # 处理重叠
                overlap_size = 0
                current_chunk = []
                for prev_para in reversed(current_chunk):
                    if overlap_size + len(prev_para) > chunk_overlap:
                        break
                    current_chunk.insert(0, prev_para)
                    overlap_size += len(prev_para)

                current_size = sum(len(p) for p in current_chunk)

            current_chunk.append(para)
            current_size += para_size

        # 处理最后一个chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunk_metadata = doc.metadata.copy()
            chunk_metadata['chunk_id'] = len(chunks)
            chunks.append(Document(content=chunk_content, metadata=chunk_metadata))

        return chunks