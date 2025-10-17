import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_or_load_vector_store(
    documents,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    index_path: str = "data/INDEX",
    batch_size: int = 100,
    embeddings: HuggingFaceEmbeddings | None = None,
):
    """创建或加载本地 FAISS 向量数据库，兼容原 main.py 的行为。

    - 若 index_path 存在：直接加载（allow_dangerous_deserialization=True）。
    - 若不存在：分批写入并持久化。
    - 始终确保 index_path 上级目录可写。
    """
    current_dir = os.getcwd()
    absolute_index_path = os.path.abspath(index_path)
    print(f"当前工作目录: {current_dir}")
    print(f"向量库路径(绝对路径): {absolute_index_path}")

    parent_dir = os.path.dirname(absolute_index_path)
    if not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"无法写入到目录'{parent_dir}'")

    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print(f"创建新的嵌入模型实例，模型名称: {model_name}")
    else:
        print("使用外部提供的嵌入模型实例")

    if os.path.exists(index_path):
        print(f"正在从'{index_path}'加载向量库...")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("向量库加载成功。")
        return vector_store

    # 创建新索引
    if not documents:
        raise ValueError("文档列表不能为空")

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    print(f"已确保目录'{os.path.dirname(index_path)}'存在")

    total_docs = len(documents)
    print(f"总文档数量: {total_docs}")
    print(f"批次大小: {batch_size}")

    first_batch_size = min(batch_size, total_docs)
    first_batch = documents[:first_batch_size]
    print(f"初始化向量库，使用前 {first_batch_size} 个文档...")

    vector_store = FAISS.from_documents(first_batch, embeddings)
    print(f"向量库初始化完成，已包含 {vector_store.index.ntotal} 个向量")

    if total_docs > first_batch_size:
        for i in range(first_batch_size, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            batch_docs = documents[i:batch_end]
            print(f"处理批次 {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}，文档范围: {i+1}-{batch_end}/{total_docs}")
            vector_store.add_documents(batch_docs)
            print(f"已添加 {len(batch_docs)} 个文档到向量库。 当前总量: {vector_store.index.ntotal}")

    print("开始保存向量库...")
    vector_store.save_local(index_path)
    print(f"向量库已成功保存到'{index_path}'。")

    if os.path.exists(index_path):
        print(f"验证成功：向量库文件'{index_path}'已存在。")
        print(f"目录'{os.path.dirname(index_path)}'内容：")
        for file in os.listdir(os.path.dirname(index_path)):
            print(f"  - {file}")
    else:
        print(f"警告：向量库文件'{index_path}'不存在，保存可能失败。")
    return vector_store


