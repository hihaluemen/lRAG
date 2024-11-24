import torch
from typing import List, Union
import numpy as np
from transformers import AutoTokenizer, AutoModel
from core.embedding import BaseEmbedding


class TransformersEmbedding(BaseEmbedding):
    """基于 Transformers 的向量化实现"""

    def __init__(
            self,
            model_name: str = "BAAI/bge-small-zh-v1.5",
            device: str = None,
            max_length: int = 512
    ):
        """
        初始化向量化模型

        Args:
            model_name: 模型名称或路径
            device: 运行设备，默认自动选择
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.max_length = max_length

        # 设置设备
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """将文本转换为向量表示"""
        # 确保输入是列表格式
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # 批处理编码
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # 编码和截断
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # 移动到指定设备
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # 获取向量表示
            with torch.no_grad():
                outputs = self.model(**encoded)
                # 使用 [CLS] token 的输出作为句子表示
                embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                all_embeddings.append(embeddings)

        # 合并所有批次的结果
        return np.vstack(all_embeddings)