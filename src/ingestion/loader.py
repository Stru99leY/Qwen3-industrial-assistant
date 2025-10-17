import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split_documents(directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """加载指定目录下的 PDF 并切分为小块。

    兼容原 main.py 的实现与日志输出策略，并在异常时不中断整体流程。
    """
    print(f"正在从文件夹'{directory_path}'中加载所有pdf...")
    all_split_docs = []
    if not os.path.exists(directory_path):
        print(f"错误：目录不存在: {directory_path}")
        return all_split_docs

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                split_docs = text_splitter.split_documents(documents)
                all_split_docs.extend(split_docs)
                print(f"文档'{filename}'已切分为 {len(split_docs)} 块。")
            except Exception as e:
                print(f"处理文档'{filename}'时出错: {e}")
    print(f"所有文档已加载并切分，共 {len(all_split_docs)} 块。")
    return all_split_docs


