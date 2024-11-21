from pathlib import Path
from core.document import Document
from parsers.markdown import MarkdownParser
from retrievers.bm25 import BM25Retriever


def test_markdown_search():
    # 初始化解析器和检索器
    parser = MarkdownParser()
    retriever = BM25Retriever()

    # 解析markdown文件
    current_dir = Path(__file__).parent.parent
    md_file = current_dir / "examples" / "test_docs" / "sample.md"

    try:
        # 解析文档
        documents = parser.parse(str(md_file))
        print(f"\n成功解析文档，共{len(documents)}个片段:")

        # 打印解析结果
        for i, doc in enumerate(documents):
            print(f"\n片段 {i + 1}:")
            print(f"Title: {doc.metadata.get('title')}")
            print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
            print(f"Content:\n{doc.content}\n")
            print("-" * 50)

        # 添加到检索器
        retriever.add_documents(documents)

        # 测试查询
        test_queries = [
            "RAG系统的主要组成是什么？",
            "RAG技术有什么优势？",
            "RAG可以应用在哪些场景？",
            "检索系统支持哪些功能？"
        ]

        for query in test_queries:
            print(f"\n查询: {query}")

            # 打印查询的分词结果
            tokenized_query = retriever.preprocess(query)
            print(f"查询分词: {tokenized_query}")

            # 获取检索结果
            results = retriever.retrieve(query, top_k=2, return_scores=True)

            print("\n检索结果:")
            for doc, score in results:
                print(f"Score: {score:.4f}")
                print(f"Title: {doc.metadata.get('title')}")
                print(f"Content:\n{doc.content}")
                print("-" * 50)

    except FileNotFoundError:
        print(f"找不到测试文件: {md_file}")
        print("请确保test_docs目录下存在sample.md文件")
    except Exception as e:
        print(f"发生错误: {str(e)}")


def main():
    # 初始化检索器(可选择性加载额外停用词表和用户词典)
    retriever = BM25Retriever()

    # 创建测试文档
    docs = [
        Document(
            content="今天天气真不错，阳光明媚。",
            metadata={"source": "doc1"}
        ),
        Document(
            content="北京今天下雨了，天气很凉爽。",
            metadata={"source": "doc2"}
        ),
        Document(
            content="昨天下雪了，很冷。",
            metadata={"source": "doc3"}
        )
    ]

    # 添加文档到检索器
    retriever.add_documents(docs)

    # 测试查询
    query = "今天天气怎么样"
    print(f"\n查询: {query}")

    # 获取带分数的结果
    results_with_scores = retriever.retrieve(query, top_k=2, return_scores=True)

    # 打印结果
    print("\n检索结果:")
    for doc, score in results_with_scores:
        print(f"Score: {score:.4f}")
        print(f"Content: {doc.content}")
        print(f"Source: {doc.metadata['source']}")
        print("-" * 50)


if __name__ == "__main__":
    # main()
    test_markdown_search()
