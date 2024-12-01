from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import shutil
import os
from pathlib import Path
from utils.service_function import RAGService
import uvicorn
import json


# 定义请求和响应模型
class CreateKBRequest(BaseModel):
    name: str
    retriever_type: str = "hybrid"
    vector_top_k: int = 5
    bm25_top_k: int = 5
    final_top_k: int = 3
    pre_rerank_top_k: int = 6
    vector_weight: float = 0.7
    bm25_weight: float = 0.3


class SearchRequest(BaseModel):
    kb_name: str
    query: str
    return_scores: bool = True
    kwargs: Dict[str, Any] = {}


class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


# 添加新的响应模型
class DocumentInfo(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]

class DocumentsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    documents: List[DocumentInfo]


class UpdateDocumentRequest(BaseModel):
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# 创建FastAPI应用
app = FastAPI(
    title="RAG Service API",
    description="基于RAG的知识库检索服务",
    version="1.0.0"
)

# 初始化RAG服务
rag_service = RAGService(
    data_root="./data/retriever",
    embedding_model="./models/bge-small-zh-v1.5",
    reranker_model="./models/bge-reranker-v2-minicpm-layerwise"
)

# 创建临时文件目录
TEMP_DIR = Path("./data/tmp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/kb/create", response_model=Dict[str, str])
async def create_knowledge_base(request: CreateKBRequest):
    """创建知识库"""
    try:
        kb_path = rag_service.create_knowledge_base(
            name=request.name,
            retriever_type=request.retriever_type,
            vector_top_k=request.vector_top_k,
            bm25_top_k=request.bm25_top_k,
            final_top_k=request.final_top_k,
            pre_rerank_top_k=request.pre_rerank_top_k,
            vector_weight=request.vector_weight,
            bm25_weight=request.bm25_weight
        )
        return {"message": f"知识库创建成功: {kb_path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/kb/add_documents", response_model=Dict[str, bool])
async def add_documents(
    kb_name: str = Form(...),
    file: UploadFile = File(...),
    file_type: str = Form(...),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    parser_kwargs: str = Form("{}")  # JSON字符串
):
    """添加文档到知识库"""
    try:
        # 解析parser_kwargs
        parser_kwargs_dict = json.loads(parser_kwargs)
        
        # 生成临时文件路径
        file_extension = Path(file.filename).suffix
        temp_file_path = TEMP_DIR / f"{kb_name}_{file.filename}"
        
        try:
            # 保存上传的文件
            with temp_file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # 添加文档
            success = rag_service.add_documents(
                kb_name=kb_name,
                file_path=str(temp_file_path),
                file_type=file_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **parser_kwargs_dict
            )
            
            return {"success": success}
            
        finally:
            # 清理临时文件
            if temp_file_path.exists():
                temp_file_path.unlink()
                
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/kb/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """搜索知识库"""
    try:
        results = rag_service.search(
            kb_name=request.kb_name,
            query=request.query,
            return_scores=request.return_scores,
            **request.kwargs
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/kb/list", response_model=List[str])
async def list_knowledge_bases():
    """列出所有知识库"""
    try:
        return rag_service.list_knowledge_bases()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/kb/{kb_name}/info", response_model=Dict[str, Any])
async def get_kb_info(kb_name: str):
    """获取知识库信息"""
    try:
        return rag_service.get_kb_info(kb_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/kb/{kb_name}/documents", response_model=DocumentsResponse)
async def get_kb_documents(
    kb_name: str,
    page: int = 1,
    page_size: int = 10
):
    """获取知识库中的所有文档（分页）
    
    Args:
        kb_name: 知识库名称
        page: 页码（从1开始）
        page_size: 每页文档数量
    """
    try:
        return rag_service.get_kb_documents(
            kb_name=kb_name,
            page=page,
            page_size=page_size
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kb/{kb_name}/documents/{doc_id}", response_model=Dict[str, bool])
async def update_document(
    kb_name: str,
    doc_id: str,
    request: UpdateDocumentRequest
):
    """更新知识库中的指定文档
    
    Args:
        kb_name: 知识库名称
        doc_id: 文档ID
        request: 更新请求
    """
    try:
        success = rag_service.update_document(
            kb_name=kb_name,
            doc_id=doc_id,
            content=request.content,
            metadata=request.metadata
        )
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        "service:app",
        host="0.0.0.0",
        port=28900,
        reload=False
    ) 