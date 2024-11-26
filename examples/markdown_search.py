from pathlib import Path
from core.document import Document
from parsers.markdown import MarkdownParser
from parsers.llama_markdown import LlamaMarkdownParser
from retrievers.bm25 import BM25Retriever


def test_markdown_search():
    print("初始化模型...")
    # 初始化解析器和检索器
    parser = MarkdownParser()
    retriever = BM25Retriever(
        top_k=3,
        score_threshold=0.5  # 可选的相似度阈值
    )

    # 解析markdown文件
    current_dir = Path(__file__).parent.parent
    md_file = current_dir / "examples" / "test_docs" / "sample.md"

    try:
        # 解析文档
        documents = parser.parse(str(md_file))
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
            "RAG系统的主要组成是什么？",
            "RAG技术有什么优势？",
            "RAG可以应用在哪些场景？",
            "检索系统支持哪些功能？"
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
                print(f"\n{i}. 标题: {doc.metadata.get('title', '无标题')}")
                print(f"内容:\n{doc.content}")
                print("-" * 30)

    except FileNotFoundError:
        print(f"找不到测试文件: {md_file}")
    except Exception as e:
        print(f"发生错误: {str(e)}")


def test_llama_markdown_search():
    print("\n=== 测试 LlamaIndex Markdown 解析器 ===")
    print("初始化模型...")
    
    # 初始化两种解析器
    llama_parser_md = LlamaMarkdownParser(
        chunk_size=512,
        chunk_overlap=50,
        split_mode="markdown"
    )
    
    llama_parser_sent = LlamaMarkdownParser(
        chunk_size=512,
        chunk_overlap=50,
        split_mode="sentence"
    )
    
    # 初始化检索器
    retriever = BM25Retriever(
        top_k=3,
        score_threshold=0.5
    )

    # 解析markdown文件
    current_dir = Path(__file__).parent.parent
    md_file = current_dir / "examples" / "test_docs" / "sample.md"

    try:
        # 使用两种模式解析文档
        print("\n1. Markdown模式解析")
        md_documents = llama_parser_md.parse(str(md_file))
        print(f"成功解析文档，共{len(md_documents)}个片段")
        
        print("\n2. Sentence模式解析")
        sent_documents = llama_parser_sent.parse(str(md_file))
        print(f"成功解析文档，共{len(sent_documents)}个片段")
        
        # 分别测试两种模式
        for mode, documents in [("Markdown", md_documents), ("Sentence", sent_documents)]:
            print(f"\n\n=== 测试 {mode} 模式 ===")
            
            # 重置检索器
            retriever = BM25Retriever(
                top_k=3,
                score_threshold=0.5
            )
            
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
                "RAG系统的主要组成是什么？",
                "RAG技术有什么优势？",
                "RAG可以应用在哪些场景？",
                "检索系统支持哪些功能？"
            ]

            print(f"\n开始测试 {mode} 模式检索...")
            for query in test_queries:
                print(f"\n查询: {query}")
                print("-" * 50)
                
                # 获取检索结果
                results = retriever.retrieve(query)
                
                # 打印结果
                print("\n检索结果:")
                for i, doc in enumerate(results, 1):
                    print(f"\n{i}. 块ID: {doc.metadata.get('chunk_id', '未知')}")
                    if 'prev_chunk_id' in doc.metadata:
                        print(f"前一块: {doc.metadata['prev_chunk_id']}")
                    if 'next_chunk_id' in doc.metadata:
                        print(f"后一块: {doc.metadata['next_chunk_id']}")
                    print(f"内容:\n{doc.content}")
                    print("-" * 30)

    except FileNotFoundError:
        print(f"找不到测试文件: {md_file}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise e


if __name__ == "__main__":
    # 测试原始解析器
    print("\n=== 测试原始 Markdown 解析器 ===")
    # test_markdown_search()
    
    # 测试LlamaIndex解析器
    test_llama_markdown_search()
