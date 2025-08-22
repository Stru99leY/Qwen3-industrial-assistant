# Qwen3-Reranker 使用说明

## 简介

本项目实现了基于Qwen3-Reranker-0.6B模型的文档重排序功能，用于提高检索系统的准确性。该实现基于官方推荐的CausalLM用法，通过计算"yes/no"的概率作为相关性分数。

## 模型加载

```python
from reranker import RerankerModel

# 初始化Reranker模型
reranker = RerankerModel(
    model_name="Qwen/Qwen3-Reranker-0.6B",  # 模型名称
    device="cuda",                        # 运行设备，如果CUDA不可用会自动切换到CPU
    batch_size=4,                        # 批处理大小
    max_length=8192                      # 最大序列长度
)
```

## 文档重排序

```python
# 假设documents是一个Document对象列表
query = "用户查询"
top_k = 5  # 返回的文档数量

# 对文档进行重排序
reranked_docs = reranker.rerank(query, documents, top_k)

# 查看重排序结果
for i, doc in enumerate(reranked_docs):
    score = doc.metadata.get('reranker_score', 'N/A')
    print(f"[{i+1}] 分数: {score:.4f} - 内容: {doc.page_content}")
```

## 实现细节

本实现基于官方推荐的CausalLM用法，主要包括以下步骤：

1. 使用`AutoModelForCausalLM`加载模型，而不是`SequenceClassification`头
2. 使用特定的提示模板格式化输入
3. 通过计算"yes"和"no"的概率作为相关性分数
4. 支持批处理以提高效率

## 注意事项

- 模型需要较大的显存，如果显存不足，可以减小`batch_size`或使用CPU模式
- 默认的`max_length`为8192，可以根据需要调整
- 如果处理非常长的文档，可能需要进行文档分块处理
