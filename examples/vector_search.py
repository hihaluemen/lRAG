from embeddings.mytransformers import TransformersEmbedding
from retrievers.vector import VectorRetriever
from core.document import Document

# 初始化
embedding_model = TransformersEmbedding(model_name="BAAI/bge-small-zh-v1.5")
retriever = VectorRetriever(
    embedding_model=embedding_model,
    top_k=3,
    index_type="IP"  # 使用内积相似度
)

# 添加文档
documents = [
    Document(content="Python是一种简单易学的编程语言"),
    Document(content="机器学习是人工智能的一个子领域"),
    Document(content="深度学习是机器学习中的一种方法"),
]
retriever.add_documents(documents)

# 保存检索器
retriever.save("data/retriever/retriever_state_faiss")

# 加载检索器
new_retriever = VectorRetriever(embedding_model=embedding_model)
new_retriever.load("data/retriever/retriever_state_faiss")

# 使用加载的检索器进行检索
results = new_retriever.retrieve("想学习机器学习")
for doc in results:
    print(doc.content)