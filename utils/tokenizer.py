import jieba
from typing import List, Set
from pathlib import Path
import os
import platform


class ChineseSimpleTokenizer:
    """中文分词器"""

    def __init__(self, stopwords_files: List[str] = None):
        """初始化分词器

        Args:
            stopwords_files: 停用词文件路径列表，如果为None则只加载基础停用词表
        """
        # 加载默认停用词表
        self.stopwords = set()
        self._load_base_stopwords()

        # 加载额外的停用词表
        if stopwords_files:
            for file_path in stopwords_files:
                self._load_stopwords_file(file_path)

        # 仅在POSIX系统（Linux/Unix/MacOS）启用并行分词
        self._enable_parallel_if_possible()

    def _enable_parallel_if_possible(self):
        """如果可能的话启用并行模式"""
        try:
            if platform.system() in ('Linux', 'Darwin'):  # Linux 或 MacOS
                jieba.enable_parallel()
        except Exception as e:
            print(f"Warning: Failed to enable parallel mode: {e}")

    def _get_base_stopwords_path(self) -> str:
        """获取基础停用词表路径"""
        current_dir_ = Path(__file__).parent.parent
        return os.path.join(current_dir_, "resources", "stopwords", "base_stopwords.txt")

    def _load_base_stopwords(self) -> Set[str]:
        """加载基础停用词表"""
        base_stopwords_path = self._get_base_stopwords_path()
        try:
            return self._load_stopwords_file(base_stopwords_path)
        except Exception as e:
            print(f"Warning: Failed to load base stopwords file: {e}")
            return set()

    def _load_stopwords_file(self, file_path: str) -> Set[str]:
        """从文件加载停用词

        Args:
            file_path: 停用词文件路径

        Returns:
            停用词集合

        Raises:
            FileNotFoundError: 文件不存在
            UnicodeDecodeError: 文件编码错误
        """
        stopwords = set()
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Stopwords file not found: {file_path}")

        try:
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        words = {line.strip() for line in f if line.strip()}
                        stopwords.update(words)
                    break
                except UnicodeDecodeError:
                    continue
            if not stopwords:
                raise UnicodeDecodeError(
                    f"Failed to decode {file_path} with encodings: {encodings}"
                )

            self.stopwords.update(stopwords)
            return stopwords

        except Exception as e:
            raise Exception(f"Error loading stopwords file {file_path}: {e}")

    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """分词处理

        Args:
            text: 输入文本
            remove_stopwords: 是否去除停用词

        Returns:
            分词结果列表
        """
        # 使用jieba分词
        tokens = jieba.cut(text)

        # 去除空白符和停用词
        tokens = [token.strip() for token in tokens if token.strip()]
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]

        return tokens

    def get_stopwords(self) -> Set[str]:
        """获取当前的停用词集合"""
        return self.stopwords.copy()


if __name__ == '__main__':
    chinese_tokenizer = ChineseSimpleTokenizer()
    print(chinese_tokenizer.get_stopwords())
    # 测试分词
    text = "这是一个人工智能相关的项目"
    tokens = chinese_tokenizer.tokenize(text)
    print(tokens)