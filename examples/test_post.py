import requests
import json
import time
from pathlib import Path


base_url = "http://localhost:28900"


def test_create_kb():
    """测试创建知识库"""
    url = f"{base_url}/kb/create"
    data = {
        "name": "test_kb_post_1",
        "retriever_type": "vector",
        "vector_top_k": 5,
        # "bm25_top_k": 5,
        # "final_top_k": 3,
        # "pre_rerank_top_k": 6,
        # "vector_weight": 0.7,
        # "bm25_weight": 0.3,
    }
    response = requests.post(url, json=data)
    print("创建知识库结果:", response.json())


def test_add_documents():
    """测试添加文档"""
    url = f"{base_url}/kb/add_documents"
    
    # 测试不同类型的文件
    test_files = [
        {
            "file_path": "examples/test_docs/qa1.xlsx",
            "file_type": "excel",
            "parser_kwargs": {
                "question_col": "question",
                "answer_col": "answer",
                "combine_qa": True
            }
        },
        # {
        #     "file_path": "examples/test_docs/sample.md",
        #     "file_type": "markdown",
        #     "parser_kwargs": {
        #         "split_mode": "markdown"
        #     }
        # },
        # {
        #     "file_path": "examples/test_docs/test.pdf",
        #     "file_type": "pdf",
        #     "parser_kwargs": {}
        # }
    ]
    
    for test_file in test_files:
        file_path = Path(test_file["file_path"])
        if not file_path.exists():
            print(f"文件不存在: {file_path}")
            continue
            
        print(f"\n测试添加文件: {file_path.name}")
        
        # 准备表单数据
        form_data = {
            "kb_name": "test_kb_post_1",
            "file_type": test_file["file_type"],
            "chunk_size": "512",
            "chunk_overlap": "50",
            "parser_kwargs": json.dumps(test_file["parser_kwargs"])
        }
        
        # 准备文件
        files = {
            "file": (file_path.name, open(file_path, "rb"), "application/octet-stream")
        }
        
        try:
            # 发送请求
            response = requests.post(url, data=form_data, files=files)
            print(f"添加文件结果: {response.json()}")
        except Exception as e:
            print(f"添加文件失败: {str(e)}")
        finally:
            # 关闭文件
            files["file"][1].close()


def test_search():
    """测试搜索"""
    url = f"{base_url}/kb/search"
    queries = [
        "什么是机器学习？",
        "深度学习的应用场景有哪些？",
        "RAG系统的主要组成部分有哪些？"
    ]
    
    for query in queries:
        data = {
            "kb_name": "test_kb_post_1",
            "query": query,
            "return_scores": True
        }
        print(f"\n测试查询: {query}")
        response = requests.post(url, json=data)
        results = response.json()
        
        print("检索结果:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 相关度: {result.get('score', '未找到相关结果')}")
            print(f"  内容: {result.get('content', '未找到内容')}")


def test_list_knowledge_bases():
    """测试列出知识库"""
    url = f"{base_url}/kb/list"
    response = requests.get(url)
    print("列出知识库结果:", response.json())


def test_kb_info():
    """测试获取知识库信息"""
    kb_name = "test_kb_post_1"
    url = f"{base_url}/kb/{kb_name}/info"
    response = requests.get(url)
    print("获取知识库信息结果:", response.json())

def test_kb_documents():
    """测试获取知识库文档"""
    kb_name = "test_kb_post_1"
    params = {
        "page": 2,
        "page_size": 5
    }
    url = f"{base_url}/kb/{kb_name}/documents"
    response = requests.get(url, params=params)
    print("获取知识库文档结果:", response.json())


def test_update_document():
    """测试更新文档"""
    kb_name = "test_kb_post_1"
    # doc_id = "04007de2-1cf8-4db0-9c90-a0403642c9cd"
    doc_id = "d25ab184-5868-49a4-b1af-fb3e692a60a6"
    params = {
        "content": "问题：人为什么要睡觉？\n答案：睡觉对人体非常重要，在睡眠过程中，身体可以进行自我修复和调整。比如，大脑会整理和巩固白天学习到的知识和记忆，身体各器官也能得到休息，恢复能量，调节新陈代谢等，以维持身体的正常运转和健康状态。",
        "metadata": {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "status": "updated"
        }
    }
    url = f"{base_url}/kb/{kb_name}/documents/{doc_id}"
    response = requests.post(url, json=params)
    print("更新文档结果:", response.json())


def test_delete_documents():
    """测试删除文档"""
    response = requests.post(
        f"{base_url}/kb/documents/delete",
        json={
            "kb_name": "tt",
            "doc_ids": ["af57d496-818e-4009-b5f7-77f5edde9db7", "2774cc31-ec2d-4971-a2db-4b05a9987743"]
        }
    )
    print(response.json())


def test_delete_kb():
    """测试删除知识库"""
    response = requests.post(
        f"{base_url}/kb/delete",
        json={"kb_name": "tt"}
    )
    print("删除知识库结果:", response.json())


if __name__ == "__main__":
    # test_create_kb()
    # test_add_documents()
    # test_search()
    # test_list_knowledge_bases()
    # test_kb_info()
    # test_update_document()
    # test_kb_documents()
    # test_delete_documents()
    test_delete_kb()

