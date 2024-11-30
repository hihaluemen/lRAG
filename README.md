# lRAG

一个轻量级的检索增强生成(RAG)框架，支持多种文档格式和检索方式。

## 主要特性

- 支持多种文档格式的解析
  - PDF文档
  - Markdown文档
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

查看 `examples` 目录下的示例代码：

- `vector_search.py`: 基础的向量检索示例
- `hybrid_search.py`: 混合检索示例
- `pdf_search.py`: PDF文档检索示例
- `markdown_search.py`: Markdown文档检索示例

## 核心依赖

- PyTorch
- Transformers
- sentence-transformers
- FlagEmbedding
- llama-index
- FAISS
- Milvus (可选)

## 项目结构

- `core/`: 核心接口定义
- `parsers/`: 文档解析器实现
- `retrievers/`: 检索器实现
- `embeddings/`: 向量编码器实现
- `rerankers/`: 重排序模型
- `utils/`: 工具函数
- `examples/`: 示例代码
- `models/`: 预训练模型配置

## License

MIT License
