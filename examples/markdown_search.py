from pathlib import Path
from core.document import Document
from parsers.markdown import MarkdownParser
from retrievers.bm25 import BM25Retriever


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
    main()