from typing import List, Optional
from pathlib import Path
import pandas as pd
from core.parser import BaseParser
from core.document import Document


class QAExcelParser(BaseParser):
    """QA对Excel解析器"""
    
    def __init__(
        self,
        question_col: str = "question",      # 问题列名
        answer_col: str = "answer",          # 答案列名
        combine_qa: bool = True,             # 是否合并问答
        add_prefix: bool = True,             # 是否添加前缀
        extra_cols: Optional[List[str]] = None,  # 需要保留的额外列
        chunk_size: Optional[int] = None,    # 答案过长时的分块大小
        chunk_overlap: int = 50              # 分块重叠大小
    ):
        self.question_col = question_col
        self.answer_col = answer_col
        self.combine_qa = combine_qa
        self.add_prefix = add_prefix
        self.extra_cols = extra_cols or []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, file_path: str) -> List[Document]:
        """解析Excel文件
        
        Args:
            file_path: Excel文件路径
            
        Returns:
            List[Document]: 解析后的文档列表
        """
        # 检查文件是否存在
        if not Path(file_path).exists():
            raise FileNotFoundError(f"找不到文件: {file_path}")
            
        # 读取Excel文件
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            raise ValueError(f"Excel文件读取失败: {str(e)}")
            
        # 检查必需列是否存在
        required_cols = [self.question_col, self.answer_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Excel缺少必需列: {missing_cols}")
            
        documents = []
        
        # 处理每一行
        for idx, row in df.iterrows():
            question = str(row[self.question_col]).strip()
            answer = str(row[self.answer_col]).strip()
            
            # 跳过空值或nan值
            if (pd.isna(row[self.question_col]) or 
                pd.isna(row[self.answer_col]) or 
                not question or 
                not answer or 
                question.lower() == 'nan' or 
                answer.lower() == 'nan'):
                continue
            
            # 收集元数据
            metadata = {
                'qa_id': idx,
                'source': Path(file_path).name
            }
            for col in self.extra_cols:
                if col in df.columns:
                    metadata[col] = row[col]
            
            if self.combine_qa:
                # 合并问答为一个文档
                if self.add_prefix:
                    content = f"问题：{question}\n答案：{answer}"
                else:
                    content = f"{question}\n{answer}"
                    
                # 如果合并后的内容过长且设置了chunk_size，进行分块
                if self.chunk_size and len(content) > self.chunk_size:
                    chunks = self._split_text(content)
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            content=chunk,
                            metadata={
                                **metadata,
                                'type': 'qa_pair',
                                'chunk_id': i,
                                'total_chunks': len(chunks),
                                'original_question': question  # 保留原始问题
                            }
                        )
                        documents.append(doc)
                else:
                    doc = Document(
                        content=content,
                        metadata={
                            **metadata,
                            'type': 'qa_pair'
                        }
                    )
                    documents.append(doc)
                    
            else:
                # 分别创建问题和答案文档
                q_doc = Document(
                    content="问题：" + question if self.add_prefix else question,
                    metadata={
                        **metadata,
                        'type': 'question'
                    }
                )
                documents.append(q_doc)
                
                # 如果答案过长且设置了chunk_size，进行分块
                if self.chunk_size and len(answer) > self.chunk_size:
                    chunks = self._split_text(answer)
                    for i, chunk in enumerate(chunks):
                        a_doc = Document(
                            content="答案：" + chunk if self.add_prefix else chunk,
                            metadata={
                                **metadata,
                                'type': 'answer',
                                'chunk_id': i,
                                'total_chunks': len(chunks)
                            }
                        )
                        documents.append(a_doc)
                else:
                    a_doc = Document(
                        content="答案：" + answer if self.add_prefix else answer,
                        metadata={
                            **metadata,
                            'type': 'answer'
                        }
                    )
                    documents.append(a_doc)
                    
        return documents
    
    def _split_text(self, text: str) -> List[str]:
        """将长文本分块，保持问题完整"""
        if not self.chunk_size:
            return [text]
            
        # 首先分离问题和答案
        parts = text.split('\n', 1)
        question = parts[0]
        answer = parts[1] if len(parts) > 1 else ""
        
        chunks = []
        # 如果只有问题部分也超过了chunk_size，则单独作为一个块
        if len(question) > self.chunk_size:
            chunks.append(question)
            current_text = answer
        else:
            current_text = text
        
        start = 0
        text_len = len(current_text)
        
        while start < text_len:
            # 确定当前块的结束位置
            end = start + self.chunk_size
            
            # 如果不是最后一块，尝试在标点或空格处断开
            if end < text_len:
                # 在chunk_size范围内找最后的标点或空格
                for i in range(min(end + self.chunk_overlap, text_len) - 1, start, -1):
                    if current_text[i] in '。.!?！？\n':
                        end = i + 1
                        break
            else:
                end = text_len
                
            # 如果是第一块且包含完整问题，直接添加文本片段
            if start == 0 and len(question) <= self.chunk_size:
                chunks.append(current_text[start:end])
            else:
                # 否则，对于答案部分的块，确保包含问题信息
                if self.add_prefix:
                    chunk = f"{question}\n{current_text[start:end]}"
                else:
                    chunk = f"{question}\n{current_text[start:end]}"
                chunks.append(chunk)
            
            # 更新起始位置，考虑重叠
            start = max(end - self.chunk_overlap, start + 1)
            
        return chunks