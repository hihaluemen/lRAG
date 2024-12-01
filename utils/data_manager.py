import os
import sys
from typing import List, Optional
from pathlib import Path

# 将项目根目录添加到系统路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from embeddings.mytransformers import TransformersEmbedding
from retrievers.vector import VectorRetriever
from retrievers.bm25 import BM25Retriever
from retrievers.hybrid import HybridRetriever
from parsers.excel import QAExcelParser
from parsers.pdf import LlamaPDFParser
from parsers.llama_markdown import LlamaMarkdownParser
from core.retriever import BaseRetriever


def list_retriever_databases(root_path: str = "./data/retriever") -> List[Path]:
    """列出所有可用的检索器数据库
    
    Args:
        root_path: 数据根目录
        
    Returns:
        数据库路径列表
    """
    root = Path(root_path)
    if not root.exists():
        return []
    return list(root.glob("*"))


def show_retriever_data(
    retriever_path: str, 
    retriever_type: str = "vector",
    embedding_model_path: str = "../models/bge-small-zh-v1.5"
) -> Optional[BaseRetriever]:
    """显示指定检索器的数据，并返回加载好的检索器实例
    
    Args:
        retriever_path: 检索器路径
        retriever_type: 检索器类型 ("vector", "bm25", "hybrid")
        embedding_model_path: embedding模型路径
        
    Returns:
        加载好的检索器实例，如果加载失败则返回None
    """
    print(f"\n=== 检查{retriever_type}检索器数据 ===")
    print(f"路径: {retriever_path}")
    
    try:
        # 初始化检索器
        if retriever_type == "vector":
            embedding_model = TransformersEmbedding(embedding_model_path)
            retriever = VectorRetriever(
                embedding_model=embedding_model,
                top_k=3,
                index_type="IP"
            )
        elif retriever_type == "bm25":
            retriever = BM25Retriever(
                top_k=3,
                score_threshold=0.1
            )
        elif retriever_type == "hybrid":
            vector_retriever = VectorRetriever(
                embedding_model=TransformersEmbedding(embedding_model_path),
                top_k=5,
                index_type="IP"
            )
            bm25_retriever = BM25Retriever(
                top_k=5,
                score_threshold=0.1
            )
            retriever = HybridRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                retriever_weights=[0.7, 0.3],
                top_k=3
            )
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")
            
        # 加载检索器
        print("\n加载检索器...")
        retriever.load(retriever_path)
        
        # 打印检索器状态
        print("\n检索器状态:")
        stats = retriever.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
        # 打印所有文档
        print(f"\n文档列表 (共{len(retriever.documents)}个):")
        for i, doc in enumerate(retriever.documents, 1):
            print(f"\n文档 {i}:")
            print(f"ID: {doc.id}")
            print(f"元数据: {doc.metadata}")
            print(f"内容预览: {doc.content[:200]}...")
            print("-" * 50)
            
        return retriever
            
    except Exception as e:
        print(f"错误: {str(e)}")
        return None


def add_documents_to_retriever(
    file_path: str,
    file_type: str,
    retriever_path: str,
    retriever_type: str = "vector",
    embedding_model_path: str = "../models/bge-small-zh-v1.5",
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> bool:
    """向指定检索器添加新文档
    
    Args:
        file_path: 要添加的文件路径
        file_type: 文件类型 ("excel", "pdf", "markdown")
        retriever_path: 检索器保存路径
        retriever_type: 检索器类型 ("vector", "bm25", "hybrid")
        embedding_model_path: embedding模型路径
        chunk_size: 分块大小
        chunk_overlap: 分块重叠大小
        
    Returns:
        是否添加成功
    """
    try:
        # 初始化解析器
        if file_type == "excel":
            parser = QAExcelParser(
                question_col="question",
                answer_col="answer",
                combine_qa=True,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif file_type == "pdf":
            parser = LlamaPDFParser(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                include_metadata=True,
                include_prev_next_rel=True
            )
        elif file_type == "markdown":
            parser = LlamaMarkdownParser(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                split_mode="markdown",
                include_metadata=True,
                include_prev_next_rel=True
            )
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
            
        # 解析文档
        print(f"\n解析{file_type}文件: {file_path}")
        documents = parser.parse(file_path)
        print(f"解析得到{len(documents)}个文档片段")
        
        # 加载现有检索器或创建新检索器
        retriever = None
        if Path(retriever_path).exists():
            print("\n加载现有检索器...")
            retriever = show_retriever_data(retriever_path, retriever_type, embedding_model_path)
            
        if retriever is None:
            print("\n创建新检索器...")
            if retriever_type == "vector":
                embedding_model = TransformersEmbedding(embedding_model_path)
                retriever = VectorRetriever(
                    embedding_model=embedding_model,
                    top_k=3,
                    index_type="IP"
                )
            elif retriever_type == "bm25":
                retriever = BM25Retriever(
                    top_k=3,
                    score_threshold=0.1
                )
            elif retriever_type == "hybrid":
                vector_retriever = VectorRetriever(
                    embedding_model=TransformersEmbedding(embedding_model_path),
                    top_k=5,
                    index_type="IP"
                )
                bm25_retriever = BM25Retriever(
                    top_k=5,
                    score_threshold=0.1
                )
                retriever = HybridRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    retriever_weights=[0.7, 0.3],
                    top_k=3
                )
                
        # 添加文档
        print("\n添加文档到检索器...")
        retriever.add_documents(documents)
        
        # 保存检索器
        print(f"\n保存检索器到: {retriever_path}")
        os.makedirs(os.path.dirname(retriever_path), exist_ok=True)
        retriever.save(retriever_path)
        
        return True
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="检索器数据管理工具")
    parser.add_argument("--root_path", type=str, default="../data/retriever",
                      help="数据根目录路径")
    parser.add_argument("--embedding_model", type=str, 
                      default="../models/bge-small-zh-v1.5",
                      help="Embedding模型路径")
    args = parser.parse_args()
    
    # 列出所有数据库
    print("=== 可用的检索器数据库 ===")
    databases = list_retriever_databases(root_path=args.root_path)
    for i, db in enumerate(databases, 1):
        print(f"{i}. {db.name}")
        
    if not databases:
        print("未找到任何检索器数据")
        exit(0)
        
    # 选择要查看的数据库
    choice = input("\n请选择要查看的数据库编号 (直接回车退出): ").strip()
    if not choice:
        exit(0)
        
    try:
        selected = databases[int(choice) - 1]
        
        # 根据文件夹名称判断检索器类型
        if "vector" in selected.name:
            retriever_type = "vector"
        elif "bm25" in selected.name:
            retriever_type = "bm25"
        elif "hybrid" in selected.name:
            retriever_type = "hybrid"
        else:
            retriever_type = input("请指定检索器类型 (vector/bm25/hybrid): ").strip()
            
        # 显示数据库内容
        show_retriever_data(str(selected), retriever_type, args.embedding_model)
        
        # 询问是否要添加新文档
        add_new = input("\n是否要添加新文档? (y/n): ").strip().lower()
        if add_new == 'y':
            file_path = input("请输入文件路径: ").strip()
            file_type = input("请输入文件类型 (excel/pdf/markdown): ").strip()
            chunk_size = int(input("请输入分块大小 (默认512): ").strip() or "512")
            chunk_overlap = int(input("请输入分块重叠大小 (默认50): ").strip() or "50")
            
            add_documents_to_retriever(
                file_path=file_path,
                file_type=file_type,
                retriever_path=str(selected),
                retriever_type=retriever_type,
                embedding_model_path=args.embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
    except (IndexError, ValueError) as e:
        print(f"错误: {str(e)}")