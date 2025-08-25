# Qwen3-Reranker 工业知识问答系统

## 项目简介

本项目实现了基于Qwen3-Reranker-0.6B模型的工业知识问答系统，采用模块化架构设计，专注于核心的问答功能。

## 系统架构

### 核心模块

1. **Reranker模块** (`code/reranker.py`)
   - `RerankerModel`: 基于Qwen3-Reranker的文档重排序模型
   - `RerankerRetriever`: 结合向量检索和重排序的检索器

2. **主应用** (`code/app.py`)
   - Streamlit Web界面
   - 集成Reranker模块
   - 支持连续对话和智能索引管理

## 主要特性

### 🎯 核心问答功能
- 基于PDF文档的知识库问答
- 支持连续对话，历史感知检索
- 智能文档检索和重排序

### 🚀 Reranker增强
- 基于Qwen3-Reranker-0.6B模型
- 两阶段检索：向量检索 + 重排序
- 提高文档相关性准确性

### 🛠️ 智能索引管理
- 自动检测索引维度匹配
- 一键重建损坏的索引
- 支持强制重建索引

### ⚙️ 灵活配置
- 支持启用/禁用Reranker模型
- 支持GPU/CPU切换
- 简洁的用户界面

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动应用

```bash
cd code
streamlit run app.py
```

### 3. 使用系统

1. 启用Reranker模型进行文档重排序
2. 在聊天界面输入问题
3. 系统自动检索相关文档并生成答案
4. 支持连续对话，系统会记住对话历史



### 基本用法

```python
from reranker import RerankerModel, RerankerRetriever

# 创建Reranker模型
reranker = RerankerModel("Qwen/Qwen3-Reranker-0.6B")

# 创建增强检索器
retriever = RerankerRetriever(
    vector_retriever=base_retriever,
    reranker=reranker,
    top_k_vector=20,
    top_k_final=5
)

## 性能优化

### 批处理配置
- 调整`batch_size`参数优化内存使用
- 根据GPU显存调整`max_length`参数
- 支持CPU回退，确保系统稳定性

### 检索优化
- 两阶段检索：向量检索 + 重排序
- 可配置检索数量：`top_k_vector` 和 `top_k_final`
- 智能索引管理，自动检测和重建

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小`batch_size`参数
   - 使用CPU模式运行

2. **Reranker模型未加载**
   - 检查模型是否正确下载
   - 确认设备配置（GPU/CPU）

3. **FAISS索引维度不匹配**
   - 使用"强制重建索引"选项
   - 运行`python emergency_fix.py`脚本

### 调试模式

启用详细日志输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 更新日志

### v3.0.0 (当前版本)
- ✨ 重构系统架构，专注于核心问答功能
- 🎯 集成Reranker增强检索
- 🔧 智能索引管理，自动检测和重建
- 🚀 支持连续对话，历史感知检索
- 💻 简洁的用户界面，易于使用

### v2.0.0
- 集成Qwen3-Reranker模型
- 支持文档重排序功能
- 基础评估指标实现

### v1.0.0
- 基础RAG系统实现
- 向量检索功能
- Streamlit Web界面
