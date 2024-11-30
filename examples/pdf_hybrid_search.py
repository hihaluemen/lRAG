import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from pathlib import Path
from parsers.pdf import LlamaPDFParser
from embeddings.mytransformers import TransformersEmbedding
from retrievers.vector import VectorRetriever
from retrievers.bm25 import BM25Retriever
from retrievers.hybrid import HybridRetriever
from rerankers.base import BGELayerwiseReranker


def test_pdf_hybrid_search():
    print("初始化模型...")
    
    # 初始化PDF解析器
    parser = LlamaPDFParser(
        chunk_size=512,
        chunk_overlap=50,
        include_metadata=True,
        include_prev_next_rel=True
    )
    
    # 初始化向量模型和检索器
    embedding_model = TransformersEmbedding(
        model_name="../models/bge-small-zh-v1.5"
    )
    
    vector_retriever = VectorRetriever(
        embedding_model=embedding_model,
        top_k=5,
        index_type="IP"  # 使用内积相似度
    )
    
    # 初始化BM25检索器
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
    
    # 初始化混合检索器
    hybrid_retriever = HybridRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        retriever_weights=[0.7, 0.3],  # 设置权重
        top_k=3,  # 最终返回的文档数量
        pre_rerank_top_k=6,  # 重排序前保留的文档数量
        reranker=reranker
    )

    # 解析PDF文件
    current_dir = Path(__file__).parent.parent
    pdf_file = current_dir / "examples" / "test_docs" / "yg.pdf"
    save_dir = current_dir / "data" / "retriever"
    
    # 确保保存目录存在
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 解析文档
        print("\n开始解析PDF文档...")
        documents = parser.parse(str(pdf_file))
        print(f"成功解析文档，共{len(documents)}个片段")
        
        # 添加到混合检索器
        print("\n添加文档到检索器...")
        hybrid_retriever.add_documents(documents)
        
        # 保存检索器状态
        save_path = save_dir / "pdf_hybrid_retriever"
        print(f"\n保存检索器状态到: {save_path}")
        hybrid_retriever.save(str(save_path))
        
        # 加载检索器（可选，用于验证保存是否成功）
        print("\n重新加载检索器...")
        new_hybrid_retriever = HybridRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            retriever_weights=[0.7, 0.3],
            top_k=3,
            pre_rerank_top_k=6,
            reranker=reranker
        )
        new_hybrid_retriever.load(str(save_path))

        # 测试查询
        test_queries = [
            "因果建模是指",
            "ATE是什么？",
            "因果建模适合的场景是？",
            "因果推断的主要方法有哪些？"
        ]

        print("\n开始测试检索...")
        for query in test_queries:
            print(f"\n查询: {query}")
            print("-" * 50)
            
            # 获取检索结果和分数
            results = new_hybrid_retriever.retrieve_with_scores(query)
            
            # 打印结果
            print("\n检索结果:")
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n{i}. 混合相关度得分: {score:.4f}")
                print(f"页码: {doc.metadata.get('page_number', '未知')}")
                print(f"块ID: {doc.metadata.get('chunk_id', '未知')}")
                
                # 打印前后文关系
                if 'prev_chunk_id' in doc.metadata:
                    print(f"前一块: {doc.metadata['prev_chunk_id']}")
                if 'next_chunk_id' in doc.metadata:
                    print(f"后一块: {doc.metadata['next_chunk_id']}")
                
                print(f"内容:\n{doc.content}")
                print("-" * 30)

        # 打印检索器状态
        print("\n检索器状态:")
        stats = new_hybrid_retriever.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

    except FileNotFoundError:
        print(f"找不到测试文件: {pdf_file}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise e


if __name__ == "__main__":
    print("=== 测试PDF混合检索 ===")
    test_pdf_hybrid_search() 