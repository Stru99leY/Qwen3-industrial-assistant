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
def create_vector_store(documents, model_name="Qwen/Qwen3-Embedding-0.6B", index_path="faiss_index", batch_size=100, embeddings=None):
    """创建或加载本地的FAISS向量数据库，支持批量处理文档"""
    # 打印当前工作目录和index_path的绝对路径
    current_dir = os.getcwd()
    absolute_index_path = os.path.abspath(index_path)
    print(f"当前工作目录: {current_dir}")
    print(f"向量库路径(绝对路径): {absolute_index_path}")
    
    # 检查是否有写入权限
    if not os.access(os.path.dirname(absolute_index_path), os.W_OK):
        print(f"错误：没有写入权限到目录'{os.path.dirname(absolute_index_path)}'")
        raise PermissionError(f"无法写入到目录'{os.path.dirname(absolute_index_path)}'")
    
    # 如果没有提供embeddings实例，则创建一个新的
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print(f"创建新的嵌入模型实例，模型名称: {model_name}")
    else:
        print(f"使用外部提供的嵌入模型实例")
    # 如果索引文件已经存在，则直接加载
    if os.path.exists(index_path):
        print(f"正在从'{index_path}'加载向量库...")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("向量库加载成功。")
    else:
        print("正在创建文本嵌入和向量库...")
        # 确保目标目录存在
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        print(f"已确保目录'{os.path.dirname(index_path)}'存在")
        
        # 如果没有提供embeddings实例，则创建一个新的
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            print(f"创建新的嵌入模型实例，模型名称: {model_name}")
        else:
            print(f"使用外部提供的嵌入模型实例")
        
        # 检查文档是否为空
        if not documents:
            print("错误：文档列表为空，无法创建向量库。")
            raise ValueError("文档列表不能为空")
        
        # 分批处理文档
        total_docs = len(documents)
        print(f"总文档数量: {total_docs}")
        print(f"批次大小: {batch_size}")
        
        # 初始化向量库（使用第一个批次）
        first_batch_size = min(batch_size, total_docs)
        first_batch = documents[:first_batch_size]
        print(f"初始化向量库，使用前 {first_batch_size} 个文档...")
        
        try:
            vector_store = FAISS.from_documents(first_batch, embeddings)
            print(f"向量库初始化完成，已包含 {vector_store.index.ntotal} 个向量")
        except Exception as e:
            print(f"初始化向量库时出错: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 处理剩余批次（如果有）
        if total_docs > first_batch_size:
            for i in range(first_batch_size, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                batch_docs = documents[i:batch_end]
                print(f"处理批次 {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}，文档范围: {i+1}-{batch_end}/{total_docs}")
                
                try:
                    # 向向量库中添加批次文档
                    vector_store.add_documents(batch_docs)
                    print(f"已添加 {len(batch_docs)} 个文档到向量库。")
                    print(f"当前向量库包含 {vector_store.index.ntotal} 个向量")
                except Exception as e:
                    print(f"添加批次文档时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
        
        print("开始保存向量库...")
        try:
            vector_store.save_local(index_path)
            print(f"向量库已成功保存到'{index_path}'。")
            # 验证文件是否存在
            if os.path.exists(index_path):
                print(f"验证成功：向量库文件'{index_path}'已存在。")
                # 打印目录内容
                print(f"目录'{os.path.dirname(index_path)}'内容：")
                for file in os.listdir(os.path.dirname(index_path)):
                    print(f"  - {file}")
            else:
                print(f"警告：向量库文件'{index_path}'不存在，保存可能失败。")
                # 打印目录内容
                print(f"目录'{os.path.dirname(index_path)}'内容：")
                for file in os.listdir(os.path.dirname(index_path)):
                    print(f"  - {file}")
        except Exception as e:
            print(f"保存向量库时出错: {e}")
            import traceback
            traceback.print_exc()
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
    index_path = "data/INDEX" # <-- 保存向量库的路径
    
    # 打印程序开始执行信息
    print("\n--- 程序开始执行 ---\n")
    print(f"PDF文件路径: {directory_path}")
    print(f"向量库保存路径(相对路径): {index_path}")
    
    # 获取绝对路径
    absolute_index_path = os.path.abspath(index_path)
    print(f"向量库保存路径(绝对路径): {absolute_index_path}")
    
    # 检查data目录下的内容
    data_dir = os.path.dirname(absolute_index_path)
    if os.path.exists(data_dir):
        print(f"目录'{data_dir}'下的内容:")
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            item_type = "目录" if os.path.isdir(item_path) else "文件"
            print(f"  - {item} ({item_type})")
    else:
        print(f"目录'{data_dir}'不存在")
    
    # 只有当索引不存在时，才需要去读取和处理原始PDF
    path_exists = os.path.exists(index_path)
    print(f"os.path.exists({index_path}) 返回: {path_exists}")
    
    if not path_exists:
        print(f"向量库路径'{index_path}'不存在，将创建新的向量库...")
        docs = load_and_split_documents(directory_path)
        print(f"所有文档已加载并切分，共 {len(docs)} 块。")
        # 2. 创建向量库（使用分批处理，批次大小为100）
        db = create_vector_store(docs, index_path=index_path, batch_size=100)
    else:
        print(f"向量库路径'{index_path}'已存在，将直接加载...")
        # 如果索引已存在，我们可以跳过文档加载，直接创建向量库（它会从本地加载）
        db = create_vector_store(None,index_path=index_path)

    print("\n向量库创建/加载完成，程序执行结束。\n")
    # 3. 创建RAG链
    # chain = create_rag_chain(db)
    # # 4. 进行一个测试问答
    # print("\n--- 开始测试问答 ---\n")
    # test_question = "请告诉我EBOM到MBOM的转换步骤。" # <-- 换成你想问的问题
    # response = chain.invoke({"input": test_question})
    # 
    # print(f"问题: {test_question}")
    # print(f"回答: {response['answer']}")