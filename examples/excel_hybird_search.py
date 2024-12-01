import os
from pathlib import Path
from parsers.excel import QAExcelParser
from embeddings.mytransformers import TransformersEmbedding
from retrievers.vector import VectorRetriever
from retrievers.bm25 import BM25Retriever
from retrievers.hybrid import HybridRetriever
from rerankers.base import BGELayerwiseReranker


def test_excel_qa_search():
    print("初始化模型...")
    
    # 初始化Excel解析器
    parser = QAExcelParser(
        question_col="question",
        answer_col="answer",
        combine_qa=True,
        chunk_size=512,
        chunk_overlap=50,
    )
    
    # 初始化向量检索器
    vector_retriever = VectorRetriever(
        embedding_model=TransformersEmbedding("../models/bge-small-zh-v1.5"),
        top_k=5,
        index_type="IP"
    )
    
    # 初始化BM25检索器
    bm25_retriever = BM25Retriever(
        top_k=5,
        score_threshold=0.1
    )
    
    # 初始化重排序器
    reranker = BGELayerwiseReranker("../models/bge-reranker-v2-minicpm-layerwise")
    
    # 创建混合检索器
    hybrid_retriever = HybridRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        retriever_weights=[0.7, 0.3],
        top_k=3,
        reranker=reranker
    )
    
    # 设置文件路径
    excel_file = "./test_docs/qa1.xlsx"  # 修改Excel文件路径
    save_path = "./data/retriever/excel_qa_retriever"
    
    try:
        print("\n开始解析Excel文档...")
        documents = parser.parse(excel_file)
        print(f"成功解析文档，共{len(documents)}个片段")
        
        # 添加数据预览
        print("\n数据预览:")
        for i, doc in enumerate(documents[:3], 1):
            print(f"\n文档 {i}:")
            print(f"类型: {doc.metadata.get('type')}")
            print(f"QA ID: {doc.metadata.get('qa_id')}")
            print(f"内容预览: {doc.content[:100]}...")
        
        print("\n添加文档到检索器...")
        hybrid_retriever.add_documents(documents)
        
        # 保存检索器状态
        print(f"\n保存检索器状态到: {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        hybrid_retriever.save(save_path)
        
        # 重新加载检索器
        print("\n重新加载检索器...")
        new_hybrid_retriever = HybridRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            retriever_weights=[0.7, 0.3],
            top_k=3,
            reranker=reranker
        )
        new_hybrid_retriever.load(save_path)
        
        print("\n开始测试检索...")
        test_queries = [
            "介绍一下人工智能吧？",
            "我想做蛋糕",
            "AI是什么？"
        ]
        
        for query in test_queries:
            print(f"\n查询: {query}")
            print("-" * 50)
            
            results = new_hybrid_retriever.retrieve_with_scores(query)
            
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n{i}. 混合相关度得分: {score:.4f}")
                if 'category' in doc.metadata:
                    print(f"类别: {doc.metadata['category']}")
                if 'chunk_id' in doc.metadata:
                    print(f"块ID: {doc.metadata['chunk_id']}")
                    print(f"总块数: {doc.metadata['total_chunks']}")
                print(f"内容:\n{doc.content}")
                print("-" * 30)
        
        # 打印检索器状态
        print("\n检索器状态:")
        stats = new_hybrid_retriever.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except FileNotFoundError:
        print(f"找不到测试文件: {excel_file}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise e


if __name__ == "__main__":
    print("=== 测试Excel问答对检索 ===")
    test_excel_qa_search()