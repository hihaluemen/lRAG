# lRAG

一个轻量级的检索增强生成(RAG)框架，支持多种文档格式和检索方式。

## 主要特性

- 支持多种文档格式的解析
  - PDF文档
  - Markdown文档
  - Excel问答对
  - (可扩展支持更多格式)

- 多样化的检索方式
  - 向量检索 (基于语义相似度)
  - BM25检索 (基于关键词匹配)
  - 混合检索 (结合向量检索和BM25)

- 灵活的向量存储支持
  - FAISS
  - Milvus
  
- 可扩展的架构设计
  - 模块化的文档解析器
  - 可自定义的检索器
  - 支持自定义重排序模型

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 使用服务层API

最简单的使用方式是通过 `RAGService` 统一管理文档和检索：

```python
from service import RAGService

# 初始化服务
service = RAGService(
    data_root="./data/retriever",              # 数据存储目录
    embedding_model="./models/bge-small-zh-v1.5",  # Embedding模型路径
    reranker_model="./models/bge-reranker-v2-minicpm-layerwise",  # 重排序模型路径
    use_reranker=True  # 是否使用重排序
)

# 创建知识库
kb_name = "my_kb"
service.create_knowledge_base(
    name=kb_name,
    retriever_type="hybrid",  # "vector", "bm25", "hybrid"
    vector_top_k=5,          # 向量检索返回数量
    bm25_top_k=5,           # BM25检索返回数量
    final_top_k=3,          # 最终返回数量
    vector_weight=0.7,      # 向量检索权重
    bm25_weight=0.3         # BM25检索权重
)

# 添加文档
service.add_documents(
    kb_name=kb_name,
    file_path="./examples/test_docs/sample.pdf",
    file_type="pdf",        # "pdf", "markdown", "excel"
    chunk_size=512,         # 分块大小
    chunk_overlap=50        # 分块重叠大小
)

# 搜索
results = service.search(
    kb_name=kb_name,
    query="你的问题",
    return_scores=True  # 是否返回相关度分数
)

# 查看结果
for result in results:
    print(f"相关度: {result['score']}")
    print(f"内容: {result['content']}")
    print(f"元数据: {result['metadata']}")
```

### 使用底层API

也可以直接使用各个组件进行更灵活的定制：

#### Excel问答对检索

```python
from parsers.excel import QAExcelParser
from retrievers.vector import VectorRetriever

# 初始化解析器
parser = QAExcelParser(
    question_col="question",  # 问题列名
    answer_col="answer",      # 答案列名
    combine_qa=True,          # 是否合并问答对
    chunk_size=512,          # 分块大小
    chunk_overlap=50         # 块重叠大小
)

# 解析Excel文件
documents = parser.parse("path/to/qa.xlsx")

# 添加到检索器
retriever.add_documents(documents)
```

#### PDF文档检索

```python
from parsers.pdf import LlamaPDFParser
from retrievers.hybrid import HybridRetriever

# 初始化解析器
parser = LlamaPDFParser(
    chunk_size=512,
    chunk_overlap=50,
    include_metadata=True,
    include_prev_next_rel=True  # 保留前后文关系
)

# 解析PDF文件
documents = parser.parse("path/to/doc.pdf")

# 使用混合检索
retriever = HybridRetriever(...)
retriever.add_documents(documents)
```

### 数据管理工具

使用命令行工具管理检索数据：
```bash
# 查看和管理检索器数据
python utils/data_manager.py

# 指定模型路径和数据目录
python utils/data_manager.py --embedding_model /path/to/model --root_path /path/to/data
```

## 项目结构

- `core/`: 核心接口定义
- `parsers/`: 文档解析器实现
- `retrievers/`: 检索器实现
- `embeddings/`: 向量编码器实现
- `rerankers/`: 重排序模型
- `utils/`: 工具函数
- `examples/`: 示例代码
- `models/`: 预训练模型配置
- `service.py`: 统一服务层API

## 核心依赖

- PyTorch
- Transformers
- sentence-transformers
- FlagEmbedding
- llama-index
- FAISS
- Milvus (可选)
- pandas (Excel支持)

## License

MIT License
