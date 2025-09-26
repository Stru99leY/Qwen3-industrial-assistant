# Qwen3-base 工业知识问答系统(简易版) v3.0


## 🏗️ 架构特点

### 1. 模块化设计
- **reranker.py**: 专注于文档重排序功能
- **scoring_system.py**: 独立的评分系统，支持多种评分策略
- **scoring_display.py**: 完整的前端评分数据展示
- **app.py**: 主应用，集成所有模块

### 2. 评分系统优势(目前还有问题，需要解决)
- ✅ 评分功能完全独立，不依赖特定检索器
- ✅ 支持多种评分策略（Reranker、语义相似度等）
- ✅ 统一的评分接口和结果格式
- ✅ 完整的评估指标（Precision、Recall、F1、MRR、NDCG）

### 3. 前端展示增强
- 📊 评分仪表板：总览、统计、Ground Truth分析
- 📈 实时评分数据展示
- ⚡ 性能指标监控
- 🔍 评分历史记录和趋势分析

## 🎯 主要改进

1. **解耦评分逻辑**: 从reranker中移除评估代码
2. **统一评分接口**: 所有评分器实现相同接口
3. **完整数据展示**: 评分数据完整展示在前端
4. **性能监控**: 实时监控评分系统性能
5. **灵活配置**: 支持启用/禁用不同评分器

## 🚀 使用方法

1. 启动应用: `streamlit run app.py`
2. 在侧边栏导入评估数据
3. 启用Reranker模型
4. 点击"📊 评分仪表板"查看完整数据
5. 使用人工评估功能

## 📁 文件结构

```
code/
├── app.py              # 主应用
├── reranker.py         # 重排序模块（已重构）
├── scoring_system.py   # 独立评分系统
├── scoring_display.py  # 评分展示组件
├── main.py            # 文档处理
└── test_reranker.py   # 测试文件
```

## 🔧 技术特性

- 基于Qwen3-Reranker-0.6B模型
- 支持GPU/CPU切换
- 批处理优化
- 实时性能监控
- 完整的错误处理

## 🚨 常见问题解决

### FAISS索引维度不匹配错误

如果遇到 `assert d == self.d` 错误，这通常是由于：
1. 嵌入模型版本更新导致维度变化
2. 索引文件损坏或不完整
3. 不同环境间的模型差异

**紧急解决方案：**

1. **立即修复**：运行 `python emergency_fix.py` 删除损坏索引
2. **强制重建**：在侧边栏勾选"强制重建索引"并刷新页面
3. **手动重建**：在侧边栏点击"🔄 重建知识库索引"
4. **命令行工具**：运行 `python fix_index.py` 进行索引管理

**推荐解决步骤：**
```bash
# 1. 停止当前运行的Streamlit应用
# 2. 运行紧急修复脚本
cd code
python emergency_fix.py

# 3. 重新启动应用
streamlit run app.py
```

**索引管理工具功能：**
- 检查索引状态
- 备份现有索引
- 删除损坏索引
- 恢复备份索引

## 📁 完整文件结构

```
code/
├── app.py                    # 主应用
├── reranker.py              # 重排序模块
├── scoring_system.py        # 独立评分系统
├── scoring_display.py       # 评分展示组件
├── test_scoring_system.py   # 评分系统测试
├── fix_index.py             # 索引修复工具
├── emergency_fix.py         # 紧急修复脚本
├── main.py                  # 文档处理
└── test_reranker.py         # 原有测试文件
```
