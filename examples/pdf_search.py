from pathlib import Path
from parsers.pdf import LlamaPDFParser
from retrievers.bm25 import BM25Retriever


def test_pdf_search():
    print("初始化模型...")
    # 初始化解析器和检索器
    parser = LlamaPDFParser(
        chunk_size=512,
        chunk_overlap=50,
        include_metadata=True,
        include_prev_next_rel=True
    )
    
    retriever = BM25Retriever(
        top_k=3,
        score_threshold=0.5
    )

    # 解析PDF文件
    current_dir = Path(__file__).parent.parent
    pdf_file = current_dir / "examples" / "test_docs" / "yg.pdf"

    try:
        # 解析文档
        documents = parser.parse(str(pdf_file))
        print(f"\n成功解析文档，共{len(documents)}个片段")
        
        # 添加到检索器
        print("\n添加文档到检索器...")
        retriever.add_documents(documents)
        
        # 打印检索器状态
        stats = retriever.get_stats()
        print("\n检索器状态:")
        for key, value in stats.items():
            print(f"{key}: {value}")

        # 测试查询
        test_queries = [
            "因果建模是指",
            "ATE是什么？",
            "因果建模适合的场景是？"
        ]

        print("\n开始测试检索...")
        for query in test_queries:
            print(f"\n查询: {query}")
            print("-" * 50)
            
            # 获取检索结果
            results = retriever.retrieve(query)
            
            # 打印结果
            print("\n检索结果:")
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. 页码: {doc.metadata.get('page_number', '未知')}")
                print(f"块ID: {doc.metadata.get('chunk_id', '未知')}")
                if 'prev_chunk_id' in doc.metadata:
                    print(f"前一块: {doc.metadata['prev_chunk_id']}")
                if 'next_chunk_id' in doc.metadata:
                    print(f"后一块: {doc.metadata['next_chunk_id']}")
                print(f"内容:\n{doc.content}")
                print("-" * 30)

    except FileNotFoundError:
        print(f"找不到测试文件: {pdf_file}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise e


if __name__ == "__main__":
    test_pdf_search() 