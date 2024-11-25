from typing import List, Set, Optional
import jieba
import os


class ChineseSimpleTokenizer:
    """简单中文分词器"""

    def __init__(self, stopwords_files: Optional[List[str]] = None):
        """
        初始化分词器

        Args:
            stopwords_files: 停用词文件路径列表
        """
        self.stopwords = set()

        # 加载停用词
        if stopwords_files:
            for file_path in stopwords_files:
                self._load_stopwords(file_path)

    def _load_stopwords(self, file_path: str):
        """加载停用词文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.stopwords.add(word)
        except Exception as e:
            print(f"Warning: Failed to load stopwords file {file_path}: {e}")

    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词

        Args:
            text: 输入文本

        Returns:
            分词结果列表
        """
        # 使用jieba进行分词
        tokens = jieba.lcut(text)

        # 过滤停用词和空字符
        tokens = [
            token for token in tokens
            if token and token not in self.stopwords
        ]

        return tokens

    def get_stopwords(self) -> Set[str]:
        """获取停用词集合"""
        return self.stopwords.copy()