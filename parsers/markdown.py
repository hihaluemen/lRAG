import re
from pathlib import Path
from typing import List
from core.parser import BaseParser
from core.document import Document


class MarkdownParser(BaseParser):
    """Markdown文档解析器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """初始化解析器

        Args:
            chunk_size: 分块大小，默认500字符
            chunk_overlap: 重叠大小，默认100字符
        """
        # 用于提取markdown标题的正则表达式
        self.header_pattern = re.compile(r'^#+\s+(.+)$', re.MULTILINE)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, file_path: str) -> List[Document]:
        """解析Markdown文件"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取第一个标题作为文档标题
        title_match = self.header_pattern.search(content)
        title = title_match.group(1) if title_match else file_path.stem

        # 按章节分割
        sections = self._split_sections(content)

        documents = []
        for i, section in enumerate(sections):
            # 创建元数据
            metadata = {
                'title': title,
                'source': str(file_path),
                'type': 'markdown',
                'chunk_id': i
            }
            # 添加章节标题到元数据（如果有）
            section_title = self._extract_section_title(section)
            if section_title:
                metadata['section_title'] = section_title

            documents.append(Document(content=section, metadata=metadata))

        return documents

    def _split_sections(self, content: str) -> List[str]:
        """按章节分割内容"""
        # 首先按二级标题分割
        sections = re.split(r'\n(?=##\s+[^\n]+\n)', content)

        # 对每个部分进行大小检查和进一步分割
        result = []
        for section in sections:
            if not section.strip():
                continue

            if len(section) <= self.chunk_size:
                result.append(section)
            else:
                # 如果章节太大，按段落分割
                result.extend(self._split_by_paragraphs(section))

        return result

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割大段落"""
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = []
        current_size = 0

        # 保存当前章节的标题（如果有）
        section_title = self._extract_section_title(text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # 如果当前段落加入后会超过chunk_size，且当前chunk非空
            if current_size + para_size > self.chunk_size and current_chunk:
                # 添加已有内容为一个chunk
                chunk_text = '\n\n'.join(current_chunk)
                if section_title and not chunk_text.startswith('#'):
                    chunk_text = f"{section_title}\n\n{chunk_text}"
                chunks.append(chunk_text)

                # 从重叠部分开始新的chunk
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        # 处理最后一个chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if section_title and not chunk_text.startswith('#'):
                chunk_text = f"{section_title}\n\n{chunk_text}"
            chunks.append(chunk_text)

        return chunks

    def _extract_section_title(self, text: str) -> str:
        """提取段落的标题"""
        match = self.header_pattern.search(text)
        return match.group(0) if match else ""