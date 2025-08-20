# main.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- 1. 加载和切分文档 ---
def load_and_split_documents(directory_path):
    """加载PDF并将其切分为小块"""
    print(f"正在从文件夹'{directory_path}'中加载所有pdf...")
    all_split_docs = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                split_docs = text_splitter.split_documents(documents)
                all_split_docs.extend(split_docs)
                print(f"文档'{filename}'已切分为 {len(split_docs)} 块。")
            except Exception as e:
                print(f"处理文档'{filename}'时出错: {e}")
    print(f"所有文档已加载并切分，共 {len(all_split_docs)} 块。")
    return all_split_docs

# --- 2. 文本嵌入和向量存储 ---
def create_vector_store(documents, model_name="moka-ai/m3e-base",index_path="faiss_index"):
    """创建或加载本地的FAISS向量数据库"""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    # 如果索引文件已经存在，则直接加载
    if os.path.exists(index_path):
        print(f"正在从'{index_path}'加载向量库...")
        vector_store = FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)
        print("向量库加载成功。")
    else:
        print("正在创建文本嵌入和向量库...")
        # 选用一个优秀的中英双语嵌入模型
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # 创建FAISS向量库
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(index_path)
        print(f"向量库已保存到'{index_path}'。")
    return vector_store

# --- 3. 创建RAG链 ---
def create_rag_chain(vector_store):
    """创建检索和问答链"""
    print("正在创建RAG链...")
    # 加载本地大模型
    llm = Ollama(model="deepseek-r1:latest")

    # 定义Prompt模板，指导模型如何回答问题
    prompt_template = ChatPromptTemplate.from_template(
        """
        你是一个专业的工业知识库专家。请根据下面提供的上下文信息来回答用户的问题。
        确保你的回答完全基于上下文，不要编造信息。如果上下文中没有相关信息，请明确告知用户。

        上下文:
        {context}

        问题:
        {input}
        """
    )
    
    # 创建一个用于将文档塞入prompt的链
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    
    # 创建检索器
    retriever = vector_store.as_retriever(search_kwargs={'k': 3}) # 返回最相关的3个文档块
    
    # 创建完整的检索链
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("RAG链创建成功。")
    return retrieval_chain


# --- 主流程 ---
if __name__ == '__main__':
    # 设置你的PDF文件路径
    directory_path = r"D:\HUT\WORK\AIProject\data\file" # <-- 把这里换成你的PDF文件名夹路径
    index_path = "data/index" # <-- 保存向量库的路径
    # 只有当索引不存在时，才需要去读取和处理原始PDF
    if not os.path.exists(index_path):
        docs = load_and_split_documents(directory_path)
        # 2. 创建向量库
        db = create_vector_store(docs,index_path=index_path)
    else:
        # 如果索引已存在，我们可以跳过文档加载，直接创建向量库（它会从本地加载）
        db = create_vector_store(None,index_path=index_path)

    # 3. 创建RAG链
    chain = create_rag_chain(db)
    # 4. 进行一个测试问答
    print("\n--- 开始测试问答 ---")
    test_question = "请告诉我EBOM到MBOM的转换步骤。" # <-- 换成你想问的问题
    response = chain.invoke({"input": test_question})
    
    print(f"问题: {test_question}")
    print(f"回答: {response['answer']}")