from embeddings.mytransformers import TransformersEmbedding
from retrievers.vector import VectorRetriever
from retrievers.bm25 import BM25Retriever
from retrievers.hybrid import HybridRetriever
from rerankers.base import BGELayerwiseReranker
from core.document import Document
from pathlib import Path


def test_search():
    print("初始化模型...")
    # 初始化向量模型
    embedding_model = TransformersEmbedding(
        model_name="../models/bge-small-zh-v1.5"
    )
    
    # 初始化检索器
    vector_retriever = VectorRetriever(
        embedding_model=embedding_model,
        top_k=5,
        index_type="IP"
    )
    
    bm25_retriever = BM25Retriever(
        top_k=5,
        score_threshold=0.1
    )
    
    # 初始化重排序器
    reranker = BGELayerwiseReranker(
        model_name="../models/bge-reranker-v2-minicpm-layerwise",
        use_fp16=True,
        batch_size=32,
        cutoff_layers=[28]
    )
    
    # 初始化融合检索器
    hybrid_retriever = HybridRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        retriever_weights=[0.7, 0.3],  # 设置权重
        top_k=5,
        pre_rerank_top_k=10,  # 重排序前保留10个文档
        reranker=reranker
    )

    # 准备测试文档
    documents = [
        Document(content="Python是一种简单易学的编程语言，广泛应用于数据分析和人工智能领域"),
        Document(content="机器学习是人工智能的一个重要子领域，它能够从数据中学习规律"),
        Document(content="深度学习是机器学习中的一种方法，通过神经网络进行模型训练"),
        Document(content="自然语言处理是人工智能的重要应用，用于理解和生成人类语言"),
        Document(content="计算机视觉主要处理图像和视频数据，是深度学习的主要应用场景之一"),
        Document(content="强化学习通过与环境交互来学习决策策略，在游戏和机器人控制中表现出色"),
        Document(content="迁移学习利用预训练模型来提升新任务的学习效果，减少训练数据需求"),
        Document(content="集成学习combines多个基础模型的预测结果，提高整体预测准确性")
    ]
    
    print(f"\n添加文档，共{len(documents)}篇...")
    hybrid_retriever.add_documents(documents)

    # 测试查询
    test_queries = [
        "什么是机器学习",
        "深度学习的应用",
        "Python可以用来做什么",
        "人工智能的主要技术",
    ]

    print("\n开始测试检索...")
    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 50)
        
        # 获取检索结果
        results = hybrid_retriever.retrieve(query)
        
        # 打印结果
        print("\n检索结果:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.content}")
        print("-" * 50)

    # 打印检索器状态
    print("\n检索器状态:")
    stats = hybrid_retriever.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_search()