# app.py

import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from main import load_and_split_documents, create_vector_store, create_rag_chain

# --- 页面配置 ---
st.set_page_config(page_title="工业设备智能问答助手", page_icon="🤖", layout="wide")
st.title("🤖 工业设备智能问答助手")

# --- 全局变量和缓存 ---
PDF_FILE_PATH = "data/file" # <-- 同样，确保这是你的PDF文件名文件夹路径
INDEX_PATH = "data/index" # <-- 保存向量库的路径

# 使用Streamlit的缓存功能，避免每次都重新加载和创建向量库
@st.cache_resource
def get_rag_chain():
    """加载和创建RAG链"""
    # 1.设置PDF文件路径和向量库路径
    KNOWLEGE_BASE_PATH = "data/file"
    INDEX_PATH = "data/index"

    # 2.检查索引是否存在，如果不存在则创建
    if not os.path.exists(INDEX_PATH):
        print("首次运行或知识库更新：正在创建新的向量索引...")
        docs = load_and_split_documents(KNOWLEGE_BASE_PATH)
        db = create_vector_store(docs,index_path=INDEX_PATH)
    else:
        print("向量索引已存在，正在加载...")
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    # 创建RAG链
    chain = create_rag_chain(db)
    return chain

# --- 主程序 ---
try:
    # 获取RAG链
    chain = get_rag_chain()
    st.success(f"知识库 '{PDF_FILE_PATH}' 已成功加载！")

    # --- 用户交互 ---
    st.info("现在，您可以就这份文档的内容进行提问了。")
    user_question = st.text_input("请输入您的问题：")

    if user_question:
        with st.spinner("正在思考，请稍候..."):
            response = chain.invoke({"input": user_question})
            st.markdown("#### 回答:")
            st.write(response["answer"])

except FileNotFoundError:
    st.error(f"错误: 知识库文件 '{PDF_FILE_PATH}' 未找到。请检查文件路径是否正确。")
except Exception as e:
    st.error(f"发生了一个错误: {e}")