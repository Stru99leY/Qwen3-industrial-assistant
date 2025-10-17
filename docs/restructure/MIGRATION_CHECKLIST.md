# 迁移完成检查清单（AIProject）

- 启动入口
  - [ ] 使用 `streamlit run src/app/app.py` 可正常启动
- 索引构建
  - [ ] 首次启动自动构建 `data/INDEX`
  - [ ] 侧边栏“强制重建索引”与“🔄 重建知识库索引”行为正常
- Reranker
  - [ ] 侧边栏启停 Reranker 不报错，回答可用
  - [ ] GPU 选项启用时可运行（无 GPU 则回退 CPU）
- 功能回归
  - [ ] 历史对话可驱动检索（history-aware）
  - [ ] 回答完全基于检索上下文
- 目录与文档
  - [ ] `README.md` 指引为新入口
  - [ ] `src/*` 与 `tests/*` 镜像结构齐全
  - [ ] 删除的旧文件有对应 `changes/` 记录
