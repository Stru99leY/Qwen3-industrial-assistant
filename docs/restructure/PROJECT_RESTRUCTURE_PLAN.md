# AIProject 重构方案与执行指引（v1）

本方案面向基于 Python + Streamlit + LangChain + FAISS + Ollama 的知识问答项目，目标：模块解耦、职责分离、测试规范化、变更记录自动化，不改动现有技术栈与运行路径。

## 一、技术栈与兼容性
- Python 3.10/3.11（与 torch 版本兼容）
- Streamlit、LangChain（core/community/huggingface）、FAISS
- Transformers、Torch、SentencePiece
- 本地 LLM：Ollama（qwen3:8b；示例还涉及 deepseek-r1:latest）

说明：仅调整代码组织结构与交互边界，不更换依赖。

## 二、重构后目录结构
```
AIProject/
  ├─ src/
  │   ├─ app/
  │   │   ├─ app.py
  │   │   └─ __init__.py
  │   ├─ ingestion/
  │   │   ├─ loader.py
  │   │   └─ __init__.py
  │   ├─ index/
  │   │   ├─ vector_store.py
  │   │   └─ __init__.py
  │   ├─ retrieval/
  │   │   ├─ rag_chain.py
  │   │   └─ __init__.py
  │   ├─ reranker/
  │   │   ├─ model.py
  │   │   ├─ retriever.py
  │   │   └─ __init__.py
  │   ├─ common/
  │   │   ├─ config.py
  │   │   ├─ logging.py
  │   │   ├─ paths.py
  │   │   ├─ types.py
  │   │   └─ __init__.py
  │   └─ __init__.py
  │
  ├─ tests/
  │   ├─ app/
  │   │   └─ test_app.py
  │   ├─ ingestion/
  │   │   └─ test_loader.py
  │   ├─ index/
  │   │   └─ test_vector_store.py
  │   ├─ retrieval/
  │   │   └─ test_rag_chain.py
  │   ├─ reranker/
  │   │   └─ test_reranker.py
  │   └─ resources/
  │       ├─ ingestion/
  │       ├─ index/
  │       ├─ retrieval/
  │       └─ reranker/
  │
  ├─ tools/
  │   └─ changelog/
  │       └─ create_change_doc.py
  │
  ├─ docs/
  │   ├─ restructure/
  │   │   └─ PROJECT_RESTRUCTURE_PLAN.md
  │   └─ templates/
  │       └─ change_doc_template.md
  │
  ├─ data/
  │   ├─ file/
  │   └─ INDEX/
  │
  ├─ requirements.txt
  └─ README.md
```

## 三、交互规则与依赖关系
- 边界：app 仅调用 ingestion/index/retrieval/reranker 暴露接口；各业务模块仅依赖 common；common 不反向依赖业务模块。
- 依赖（文本图）：
```
app ─┬─> ingestion
     ├─> index
     ├─> retrieval ──> index
     └─> reranker

ingestion ──┐
index     ───┼─> common
retrieval ───┘
reranker  ───┘
```

## 四、测试规范与迁移
- 镜像结构：`src/<module>/..` ↔ `tests/<module>/..`
- 将 `code/test_reranker.py` 迁移为 `tests/reranker/test_reranker.py` 并修正导入。
- 资源集中：`tests/resources/<module>/`。

## 五、变更记录自动化
- 各模块根下建立 `changes/`；命名 `YYYYMMDD_变更类型_文件名.md`（变更类型：新增/修改/删除）。
- 内容：变更时间、简述、原因、影响范围（是否需回归测试）。
- 使用：`python tools/changelog/create_change_doc.py --module src/reranker --type 修改 --file model.py --reason "优化批处理" --impact "需回归检索链"`

## 六、执行步骤摘录
- 从 `code/main.py` 拆分：`load_and_split_documents` → `src/ingestion/loader.py`；`create_vector_store` → `src/index/vector_store.py`
- 从 `code/app.py` 拆分：`get_project_root` → `src/common/paths.py`；`log` → `src/common/logging.py`；检索链 → `src/retrieval/rag_chain.py`；UI 保留在 `src/app/app.py`
- 从 `code/reranker.py` 拆分：`RerankerModel` → `src/reranker/model.py`；`RerankerRetriever` → `src/reranker/retriever.py`

## 七、示例
- 索引模块接口：`create_or_load_vector_store(docs, embeddings, index_path, batch_size)`（封装权限/维度/批处理/持久化）
- Reranker：`model.py` 负责打分，`retriever.py` 负责两阶段检索。
