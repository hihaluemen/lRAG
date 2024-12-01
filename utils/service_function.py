import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from embeddings.mytransformers import TransformersEmbedding
from retrievers.vector import VectorRetriever
from retrievers.bm25 import BM25Retriever
from retrievers.hybrid import HybridRetriever
from parsers.excel import QAExcelParser
from parsers.pdf import LlamaPDFParser
from parsers.llama_markdown import LlamaMarkdownParser
from rerankers.base import BGELayerwiseReranker
from core.document import Document
from core.retriever import BaseRetriever


class RAGService:
    """RAG服务类，提供文档管理和检索服务"""
    
    def __init__(
        self,
        data_root: str = "./data/retriever",
        embedding_model: str = "./models/bge-small-zh-v1.5",
        reranker_model: str = "./models/bge-reranker-v2-minicpm-layerwise",
        use_reranker: bool = True
    ):
        """初始化RAG服务
        
        Args:
            data_root: 数据根目录
            embedding_model: Embedding模型路径
            reranker_model: 重排序模型路径
            use_reranker: 是否使用重排序
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # 初始化embedding模型
        self.embedding_model = TransformersEmbedding(embedding_model)
        
        # 初始化重排序器
        self.reranker = None
        if use_reranker:
            self.reranker = BGELayerwiseReranker(
                model_name=reranker_model,
                use_fp16=True,
                batch_size=32,
                cutoff_layers=[28]
            )
    
    def create_knowledge_base(
        self,
        name: str,
        retriever_type: str = "hybrid",
        vector_top_k: int = 5,
        bm25_top_k: int = 5,
        final_top_k: int = 3,
        pre_rerank_top_k: int = 6,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> str:
        """创建知识库
        
        Args:
            name: 知识库名称
            retriever_type: 检索器类型 ("vector", "bm25", "hybrid")
            vector_top_k: 向量检索返回数量
            bm25_top_k: BM25检索返回数量
            final_top_k: 最终返回数量
            pre_rerank_top_k: 重排序前数量
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
            
        Returns:
            知识库路径
        """
        kb_path = self.data_root / name
        
        if retriever_type == "vector":
            retriever = VectorRetriever(
                embedding_model=self.embedding_model,
                top_k=final_top_k,
                index_type="IP"
            )
        elif retriever_type == "bm25":
            retriever = BM25Retriever(
                top_k=final_top_k,
                score_threshold=0.1
            )
        elif retriever_type == "hybrid":
            vector_retriever = VectorRetriever(
                embedding_model=self.embedding_model,
                top_k=vector_top_k,
                index_type="IP"
            )
            bm25_retriever = BM25Retriever(
                top_k=bm25_top_k,
                score_threshold=0.1
            )
            retriever = HybridRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                retriever_weights=[vector_weight, bm25_weight],
                top_k=final_top_k,
                pre_rerank_top_k=pre_rerank_top_k,
                reranker=self.reranker
            )
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")
            
        # 保存检索器
        os.makedirs(kb_path, exist_ok=True)
        retriever.save(str(kb_path))
        
        return str(kb_path)
    
    def add_documents(
        self,
        kb_name: str,
        file_path: str,
        file_type: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **parser_kwargs
    ) -> bool:
        """向知识库添加文档
        
        Args:
            kb_name: 知识库名称
            file_path: 文件路径
            file_type: 文件类型 ("excel", "pdf", "markdown")
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            **parser_kwargs: 解析器额外参数
            
        Returns:
            是否添加成功
        """
        kb_path = self.data_root / kb_name
        if not kb_path.exists():
            raise ValueError(f"知识库不存在: {kb_name}")
            
        # 初始化解析器
        if file_type == "excel":
            parser = QAExcelParser(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **parser_kwargs
            )
        elif file_type == "pdf":
            parser = LlamaPDFParser(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                include_metadata=True,
                include_prev_next_rel=True,
                **parser_kwargs
            )
        elif file_type == "markdown":
            parser = LlamaMarkdownParser(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                split_mode="markdown",
                include_metadata=True,
                include_prev_next_rel=True,
                **parser_kwargs
            )
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
            
        try:
            # 解析文档
            documents = parser.parse(file_path)
            
            # 加载检索器
            retriever = self._load_retriever(str(kb_path))
            
            # 添加文档
            retriever.add_documents(documents)
            
            # 保存检索器
            retriever.save(str(kb_path))
            
            return True
            
        except Exception as e:
            print(f"添加文档失败: {str(e)}")
            return False
    
    def search(
        self,
        kb_name: str,
        query: str,
        return_scores: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """搜索知识库
        
        Args:
            kb_name: 知识库名称
            query: 查询文本
            return_scores: 是否返回相关度分数
            **kwargs: 检索器额外参数
            
        Returns:
            检索结果列表
        """
        kb_path = self.data_root / kb_name
        if not kb_path.exists():
            raise ValueError(f"知识库不存在: {kb_name}")
            
        # 加载检索器
        retriever = self._load_retriever(str(kb_path))
        
        # 执行检索
        if return_scores:
            results = retriever.retrieve_with_scores(query, **kwargs)
            return [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
            ]
        else:
            results = retriever.retrieve(query, **kwargs)
            return [
                {
                    "content": doc.content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
    
    def list_knowledge_bases(self) -> List[str]:
        """列出所有知识库"""
        return [kb.name for kb in self.data_root.glob("*")]
    
    def get_kb_info(self, kb_name: str) -> Dict[str, Any]:
        """获取知识库信息"""
        kb_path = self.data_root / kb_name
        if not kb_path.exists():
            raise ValueError(f"知识库不存在: {kb_name}")
            
        retriever = self._load_retriever(str(kb_path))
        return retriever.get_stats()
    
    def _load_retriever(self, kb_path: str) -> BaseRetriever:
        """加载检索器
        
        Args:
            kb_path: 知识库路径
            
        Returns:
            检索器实例
        """
        kb_path = Path(kb_path)
        
        # 检查是否是混合检索器（查找retriever_0, retriever_1等子目录）
        if list(kb_path.glob("retriever_*")):
            # Hybrid检索器
            vector_retriever = VectorRetriever(
                embedding_model=self.embedding_model
            )
            bm25_retriever = BM25Retriever()
            retriever = HybridRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                reranker=self.reranker
            )
        # 检查是否是向量检索器
        elif (kb_path / "index.faiss").exists():
            retriever = VectorRetriever(
                embedding_model=self.embedding_model
            )
        # 否则是BM25检索器
        else:
            retriever = BM25Retriever()
            
        retriever.load(str(kb_path))
        return retriever


# 使用示例
if __name__ == "__main__":
    # 初始化服务
    service = RAGService(
        data_root="./data/retriever",
        embedding_model="./models/bge-small-zh-v1.5",
        reranker_model="./models/bge-reranker-v2-minicpm-layerwise"
    )
    
    # 创建知识库
    kb_name = "test_kb"
    service.create_knowledge_base(
        name=kb_name,
        retriever_type="hybrid"
    )
    
    # 添加文档
    service.add_documents(
        kb_name=kb_name,
        file_path="../examples/test_docs/qa1.xlsx",
        file_type="excel",
        question_col="question",
        answer_col="answer",
        combine_qa=True
    )
    
    # 搜索
    results = service.search(
        kb_name=kb_name,
        query="什么是机器学习？"
    )
    
    # 打印结果
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 相关度: {result['score']:.4f}")
        print(f"内容: {result['content']}")
        print("-" * 30) 