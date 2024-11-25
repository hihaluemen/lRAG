from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from core.document import Document
from FlagEmbedding import LayerWiseFlagLLMReranker


class BaseReranker(ABC):
    """重排序器基类"""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """重新排序文档列表"""
        pass


class BGELayerwiseReranker(BaseReranker):
    """基于BGE分层的重排序器"""
    
    def __init__(
            self,
            model_name: str = "BAAI/bge-reranker-v2-minicpm-layerwise",
            use_fp16: bool = False,
            use_bf16: bool = False,
            batch_size: int = 32,
            cutoff_layers: List[int] = [28]  # 默认使用第28层
    ):
        """
        初始化BGE分层重排序器
        
        Args:
            model_name: 模型名称或路径
            use_fp16: 是否使用FP16
            use_bf16: 是否使用BF16
            batch_size: 批处理大小
            cutoff_layers: 使用哪些层的输出进行重排序
        """
        self.batch_size = batch_size
        self.cutoff_layers = cutoff_layers
        
        # 初始化重排序模型
        self.reranker = LayerWiseFlagLLMReranker(
            model_name,
            use_fp16=use_fp16,
            use_bf16=use_bf16
        )

    def _batch_score(self, query: str, texts: List[str]) -> List[float]:
        """批量计算文档得分"""
        all_scores = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # 准备输入对
            pairs = [[query, text] for text in batch_texts]
            
            # 计算得分
            scores = self.reranker.compute_score(
                pairs,
                cutoff_layers=self.cutoff_layers
            )
            all_scores.extend(scores)
                
        return all_scores

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """重新排序文档列表"""
        if not documents:
            return documents
            
        # 获取文档文本
        texts = [doc.content for doc in documents]
        
        # 计算重排序得分
        scores = self._batch_score(query, texts)
        
        # 将文档和得分打包并排序
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的文档
        return [doc for doc, _ in doc_scores]


if __name__ == '__main__':
    # 测试代码
    reranker = BGELayerwiseReranker(
        model_name="../models/bge-reranker-v2-minicpm-layerwise",
        use_fp16=True,
        batch_size=32
    )
    
    # 测试文档
    query = "什么是熊猫？"
    documents = [
        Document(content="熊猫是一种生活在中国的濒危动物。"),
        Document(content="大熊猫(Ailuropoda melanoleuca)是中国特有的珍稀动物。"),
        Document(content="这是一个无关的文档。")
    ]
    
    # 测试重排序
    results = reranker.rerank(query, documents)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.content}")