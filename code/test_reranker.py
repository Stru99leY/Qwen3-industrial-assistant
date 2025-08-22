# test_reranker.py

import sys
import os
import torch
from reranker import RerankerModel

# 简单的Document类用于测试
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def test_reranker():
    """测试修改后的RerankerModel功能"""
    print("开始测试RerankerModel...")
    
    # 创建测试文档
    documents = [
        Document(page_content="北京是中国的首都。", metadata={"source": "test1"}),
        Document(page_content="上海是中国最大的城市。", metadata={"source": "test2"}),
        Document(page_content="广州是广东省的省会。", metadata={"source": "test3"}),
        Document(page_content="深圳是中国的经济特区。", metadata={"source": "test4"}),
    ]
    
    # 测试查询
    query = "中国的首都是哪里？"
    
    # 初始化RerankerModel
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        reranker = RerankerModel(
            model_name="Qwen/Qwen3-Reranker-0.6B",
            device=device,
            batch_size=2,
            max_length=8192
        )
        print("RerankerModel初始化成功")
        
        # 测试rerank方法
        print("\n测试rerank方法...")
        reranked_docs = reranker.rerank(query, documents)
        
        # 打印结果
        print("\n重排序结果:")
        for i, doc in enumerate(reranked_docs):
            score = doc.metadata.get('reranker_score', 'N/A')
            print(f"[{i+1}] 分数: {score:.4f} - 内容: {doc.page_content}")
        
        # 测试format_instruction方法
        print("\n测试format_instruction方法...")
        formatted = reranker.format_instruction(query, documents[0].page_content)
        print(formatted)
        
        # 测试process_inputs方法
        print("\n测试process_inputs方法...")
        pairs = [reranker.format_instruction(query, doc.page_content) for doc in documents[:2]]
        inputs = reranker.process_inputs(pairs)
        print(f"输入形状: {inputs['input_ids'].shape}")
        
        # 测试compute_logits方法
        print("\n测试compute_logits方法...")
        scores = reranker.compute_logits(inputs)
        print(f"计算的分数: {scores}")
        
        print("\n测试完成!")
        return True
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        return False

if __name__ == "__main__":
    test_reranker()