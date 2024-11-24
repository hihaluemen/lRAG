from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np


class BaseEmbedding(ABC):
    """向量化基类，定义向量化模型的标准接口"""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        将文本转换为向量表示

        Args:
            texts: 单条文本或文本列表
            batch_size: 批处理大小

        Returns:
            numpy.ndarray: 文本的向量表示
        """
        pass

    def __call__(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        return self.encode(texts, batch_size)