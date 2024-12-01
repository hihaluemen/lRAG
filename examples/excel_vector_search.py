import os
from pathlib import Path
from parsers.excel import QAExcelParser
from embeddings.mytransformers import TransformersEmbedding
from retrievers.vector import VectorRetriever


def test_excel_vector_search():
    print("初始化模型...")
    
    # 初始化Excel解析器
    parser = QAExcelParser(
        question_col="question",
        answer_col="answer",
        combine_qa=True,
        chunk_size=512,
        chunk_overlap=50
    )
    
    # 初始化embedding模型和检索器
    embedding_model = TransformersEmbedding(
        model_name="../models/bge-small-zh-v1.5"
    )
    retriever = VectorRetriever(
        embedding_model=embedding_model,
        top_k=3,
        index_type="IP"  # 使用内积相似度
    )

    # 设置文件路径
    current_dir = Path(__file__).parent.parent
    excel_file = current_dir / "examples" / "test_docs" / "qa1.xlsx"
    save_dir = current_dir / "data" / "retriever"
    
    # 确保保存目录存在
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 解析文档
        print("\n开始解析Excel文档...")
        documents = parser.parse(str(excel_file))
        print(f"成功解析文档，共{len(documents)}个片段")
        
        # 添加到检索器
        print("\n添加文档到检索器...")
        retriever.add_documents(documents)
        
        # 保存检索器状态
        save_path = save_dir / "excel_vector_retriever"
        print(f"\n保存检索器状态到: {save_path}")
        retriever.save(str(save_path))
        
        # 加载检索器（可选，用于验证保存是否成功）
        print("\n重新加载检索器...")
        new_retriever = VectorRetriever(embedding_model=embedding_model)
        new_retriever.load(str(save_path))

        # 测试查询
        test_queries = [
            "介绍一下人工智能吧？",
            "我想做蛋糕",
            "AI是什么？"
        ]

        print("\n开始测试检索...")
        for query in test_queries:
            print(f"\n查询: {query}")
            print("-" * 50)
            
            # 获取检索结果和分数
            results = new_retriever.retrieve_with_scores(query)
            
            # 打印结果
            print("\n检索结果:")
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n{i}. 相关度得分: {score:.4f}")
                print(f"QA ID: {doc.metadata.get('qa_id', '未知')}")
                if 'chunk_id' in doc.metadata:
                    print(f"块ID: {doc.metadata['chunk_id']}")
                    print(f"总块数: {doc.metadata['total_chunks']}")
                print(f"内容:\n{doc.content}")
                print("-" * 30)

    except FileNotFoundError:
        print(f"找不到测试文件: {excel_file}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise e


if __name__ == "__main__":
    print("=== 测试Excel向量检索 ===")
    test_excel_vector_search() 