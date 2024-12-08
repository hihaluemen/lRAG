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
import numpy as np
from rank_bm25 import BM25Okapi
import shutil


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
        
        # 统一处理检索结果
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
    
    def get_kb_documents(
        self,
        kb_name: str,
        page: int = 1,
        page_size: int = 10
    ) -> Dict[str, Any]:
        """获取知识库中的所有文档（分页）"""
        kb_path = self.data_root / kb_name
        if not kb_path.exists():
            raise ValueError(f"知识库不存在: {kb_name}")
        
        # 加载检索器
        retriever = self._load_retriever(str(kb_path))
        
        # 获取所有文档
        if isinstance(retriever, HybridRetriever):
            # 对于混合检索器，使用第一个检索器（通常是向量检索器）的文档
            all_docs = retriever.retrievers[0].documents
        else:
            all_docs = retriever.documents
        
        total_docs = len(all_docs)
        
        # 计算总页数
        total_pages = (total_docs + page_size - 1) // page_size
        
        # 验证页码
        if page < 1 or (total_docs > 0 and page > total_pages):
            raise ValueError(f"无效的页码: {page}, 总页数: {total_pages}")
        
        # 计算当前页的文档
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_docs)
        current_docs = list(all_docs)[start_idx:end_idx]
        
        # 构建返回数据
        return {
            "total": total_docs,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "documents": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata
                }
                for doc in current_docs
            ]
        }
    
    def update_document(
        self,
        kb_name: str,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """更新知识库中的指定文档"""
        kb_path = self.data_root / kb_name
        if not kb_path.exists():
            raise ValueError(f"知识库不存在: {kb_name}")
        
        # 加载检索器
        retriever = self._load_retriever(str(kb_path))
        
        # 获取文档集合
        if isinstance(retriever, HybridRetriever):
            retrievers = retriever.retrievers
        else:
            retrievers = [retriever]
        
        success = False
        for r in retrievers:
            # 查找文档
            doc_index = None
            for i, doc in enumerate(r.documents):
                if doc.id == doc_id:
                    doc_index = i
                    break
                
            if doc_index is not None:
                # 创建新文档
                old_doc = r.documents[doc_index]
                new_content = content if content is not None else old_doc.content
                new_metadata = {**old_doc.metadata, **(metadata or {})}
                
                # 创建新文档对象（保持原ID）
                new_doc = Document(
                    content=new_content,
                    metadata=new_metadata,
                    id=doc_id
                )
                
                # 更新文档
                if isinstance(r, VectorRetriever):
                    # 对于向量检索器，重建整个索引
                    r.documents[doc_index] = new_doc
                    # 重新计算所有文档的向量并创建新索引
                    embeddings = r.embedding_model.encode([doc.content for doc in r.documents])
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    
                    # 重新初始化索引
                    r._init_index(r.dimension)
                    # 添加所有向量
                    r.index.add(embeddings.astype(np.float32))
                    
                elif isinstance(r, BM25Retriever):
                    # 对于BM25检索器，需要重新分词
                    r.documents[doc_index] = new_doc
                    r.tokenized_docs[doc_index] = r.tokenizer.tokenize(new_content)
                    r.bm25 = BM25Okapi(r.tokenized_docs)
                
                success = True
        
        if success:
            # 保存更新后的检索器
            retriever.save(str(kb_path))
            return True
        else:
            raise ValueError(f"未找到ID为 {doc_id} 的文档")
    
    def delete_documents(self, kb_name: str, doc_ids: List[str]) -> List[str]:
        """从知识库中删除指定文档
        
        Args:
            kb_name: 知识库名称
            doc_ids: 要删除的文档ID列表
            
        Returns:
            List[str]: 成功删除的文档ID列表
        """
        kb_path = self.data_root / kb_name
        if not kb_path.exists():
            raise ValueError(f"知识库不存在: {kb_name}")
            
        # 加载检索器
        retriever = self._load_retriever(str(kb_path))

        print(doc_ids)
        
        # 删除文档
        deleted_ids = retriever.delete_documents(doc_ids)
        
        # 保存更新后的检索器
        if deleted_ids:
            retriever.save(str(kb_path))
            
        return deleted_ids
    
    def delete_knowledge_base(self, kb_name: str) -> bool:
        """删除整个知识库
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            bool: 是否删除成功
        """
        kb_path = self.data_root / kb_name
        if not kb_path.exists():
            raise ValueError(f"知识库不存在: {kb_name}")
        
        try:
            # 递归删除知识库目录
            shutil.rmtree(kb_path)
            return True
        except Exception as e:
            print(f"删除知识库失败: {str(e)}")
            return False


# 使用示例
if __name__ == "__main__":
    # 初始化服务
    service = RAGService(
        data_root="./data/retriever",
        embedding_model="../models/bge-small-zh-v1.5",
        reranker_model="../models/bge-reranker-v2-minicpm-layerwise"
    )
    
    # 创建知识库
    kb_name = "test_kb_2"
    service.create_knowledge_base(
        name=kb_name,
        retriever_type="bm25"
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