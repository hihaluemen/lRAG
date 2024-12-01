import os
from pathlib import Path
from embeddings.mytransformers import TransformersEmbedding
from retrievers.vector import VectorRetriever
from retrievers.bm25 import BM25Retriever
from retrievers.hybrid import HybridRetriever


def show_retriever_data(retriever_path: str, retriever_type: str = "vector"):
    """显示指定路径下的检索器数据
    
    Args:
        retriever_path: 检索器保存路径
        retriever_type: 检索器类型，可选 "vector", "bm25", "hybrid"
    """
    print(f"\n=== 检查{retriever_type}检索器数据 ===")
    print(f"路径: {retriever_path}")
    
    try:
        # 根据类型初始化检索器
        if retriever_type == "vector":
            embedding_model = TransformersEmbedding("../models/bge-small-zh-v1.5")
            retriever = VectorRetriever(embedding_model=embedding_model)
        elif retriever_type == "bm25":
            retriever = BM25Retriever()
        elif retriever_type == "hybrid":
            vector_retriever = VectorRetriever(
                embedding_model=TransformersEmbedding("../models/bge-small-zh-v1.5")
            )
            bm25_retriever = BM25Retriever()
            retriever = HybridRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                retriever_weights=[0.7, 0.3]
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
            
    except FileNotFoundError:
        print(f"错误: 找不到检索器数据，路径: {retriever_path}")
    except Exception as e:
        print(f"错误: {str(e)}")
        raise e


if __name__ == "__main__":
    # 数据根目录
    data_root = Path(__file__).parent.parent / "data" / "retriever"
    
    # 显示所有可用的检索器数据
    print("=== 可用的检索器数据 ===")
    if data_root.exists():
        retrievers = list(data_root.glob("*"))
        for i, path in enumerate(retrievers, 1):
            print(f"{i}. {path.name}")
    else:
        print("未找到任何检索器数据")
        exit(0)
        
    # 让用户选择要查看的检索器
    choice = input("\n请选择要查看的检索器编号 (默认1): ").strip() or "1"
    try:
        selected = retrievers[int(choice) - 1]
        
        # 根据文件夹名称判断检索器类型
        if "vector" in selected.name:
            retriever_type = "vector"
        elif "bm25" in selected.name:
            retriever_type = "bm25"
        elif "hybrid" in selected.name:
            retriever_type = "hybrid"
        else:
            retriever_type = input("请指定检索器类型 (vector/bm25/hybrid): ").strip()
            
        show_retriever_data(str(selected), retriever_type)
        
    except (IndexError, ValueError):
        print("无效的选择")