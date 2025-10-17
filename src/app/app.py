import time
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import get_project_root, get_data_paths
from common.logging import log
from ingestion.loader import load_and_split_documents
from index.vector_store import create_or_load_vector_store
from retrieval.rag_chain import build_history_aware_retriever, build_conversational_rag
from reranker.model import RerankerModel
from reranker.retriever import RerankerRetriever


st.set_page_config(page_title="工业知识问答助手 (重构入口)", page_icon="🤖", layout="wide")
st.title("🤖 工业知识问答助手（重构版入口）")


def init_vector_store():
    root = get_project_root()
    paths = get_data_paths(root)
    index_path = paths["index_dir"]
    model_name = "Qwen/Qwen3-Embedding-0.6B"

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    force_rebuild = st.session_state.get("force_rebuild_index", False)
    need_rebuild = force_rebuild or (not os.path.exists(index_path))

    if need_rebuild:
        docs = load_and_split_documents(paths["knowledge_dir"])
        db = create_or_load_vector_store(docs, model_name=model_name, index_path=index_path, embeddings=embeddings)
        if force_rebuild:
            st.session_state.force_rebuild_index = False
    else:
        db = create_or_load_vector_store(None, model_name=model_name, index_path=index_path, embeddings=embeddings)
    return db, embeddings


with st.sidebar:
    st.header("系统配置")
    if 'use_reranker' not in st.session_state:
        st.session_state.use_reranker = True
    if 'use_gpu' not in st.session_state:
        st.session_state.use_gpu = True

    use_reranker = st.checkbox("启用Reranker模型", value=st.session_state.use_reranker)
    if use_reranker != st.session_state.use_reranker:
        st.session_state.use_reranker = use_reranker
        st.rerun()

    use_gpu = st.checkbox("使用GPU加速", value=st.session_state.use_gpu)
    if use_gpu != st.session_state.use_gpu:
        st.session_state.use_gpu = use_gpu
        st.rerun()

    if st.checkbox("强制重建索引"):
        st.session_state.force_rebuild_index = True
        st.warning("已启用强制重建索引，请刷新页面")

    if st.button("🔄 重建知识库索引"):
        try:
            import shutil
            paths = get_data_paths(get_project_root())
            index_dir = paths["index_dir"]
            if os.path.exists(index_dir):
                shutil.rmtree(index_dir)
                st.success("索引已删除，请刷新页面重建")
                st.rerun()
            else:
                st.info("索引不存在，无需删除")
        except Exception as e:
            st.error(f"重建索引失败: {e}")


import os

try:
    log("应用启动...")
    db, embeddings = init_vector_store()
    base_retriever = db.as_retriever(search_kwargs={'k': 20})

    # 可选 Reranker 两阶段检索
    reranker = None
    if st.session_state.get('use_reranker', True):
        device = "cuda" if st.session_state.get('use_gpu', False) else "cpu"
        try:
            reranker = RerankerModel(device=device)
        except Exception as e:
            st.warning(f"Reranker 加载失败，将退回基础检索：{e}")
            reranker = None

    effective_retriever = base_retriever if reranker is None else RerankerRetriever(
        vector_retriever=base_retriever,
        reranker=reranker,
        top_k_vector=20,
        top_k_final=5,
    )

    llm = Ollama(model="qwen3:8b")
    retriever_chain = build_history_aware_retriever(llm, effective_retriever)
    conversation_rag_chain = build_conversational_rag(retriever_chain)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="你好！我是你的专属知识库助手，有什么可以帮到你？")]

    for message in st.session_state.chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
            st.markdown(message.content)

    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        with st.chat_message("Human"):
            st.markdown(prompt)
        with st.chat_message("AI"):
            with st.spinner("正在思考中..."):
                start_time = time.time()
                response = conversation_rag_chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": prompt
                })
                _ = time.time() - start_time
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.chat_history.append(AIMessage(content=answer))

except Exception as e:
    st.error(f"发生错误: {e}")


