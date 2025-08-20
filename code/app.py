from cmd import PROMPT
import streamlit as st
import os
import sys
import traceback
# 确保你的 main.py 中有以下这些函数，并且它们是可导入的
from main import load_and_split_documents, create_vector_store

# 设置日志级别，增加终端输出
DEBUG = True

def log(message):
    """打印日志到终端"""
    if DEBUG:
        print(f"[APP LOG] {message}", file=sys.stderr)
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# --- 页面配置 ---
st.set_page_config(page_title="工业知识问答助手 v2.0", page_icon="🤖", layout="wide")
st.title("🤖 工业知识问答助手 v2.0 (支持连续对话)")

# --- 核心功能函数 ---

# 使用Streamlit的缓存功能，避免每次都重新加载和创建向量库
@st.cache_resource
def get_vector_store():
    """创建并缓存向量数据库"""
    KNOWLEDGE_BASE_DIR = "data/file"
    INDEX_PATH = "data/INDEX"  # 统一使用大写，与main.py保持一致
    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # 显式指定嵌入模型，与main.py保持一致
    
    # 打印当前工作目录和INDEX_PATH的绝对路径
    current_dir = os.getcwd()
    absolute_index_path = os.path.abspath(INDEX_PATH)
    st.write(f"当前工作目录: {current_dir}")
    st.write(f"索引路径(绝对路径): {absolute_index_path}")
    st.write(f"索引路径是否存在: {os.path.exists(INDEX_PATH)}")
    st.write(f"使用的嵌入模型: {MODEL_NAME}")
    
    if not os.path.exists(INDEX_PATH):
        st.info("首次运行或知识库更新：正在创建新的向量索引，请稍候...")
        log("索引路径不存在，开始创建新索引...")
        try:
            docs = load_and_split_documents(KNOWLEDGE_BASE_DIR)
            log(f"成功加载并切分文档，共 {len(docs)} 个文档块")
            db = create_vector_store(docs, model_name=MODEL_NAME, index_path=INDEX_PATH)
            log("向量库创建成功")
            st.success("知识库已成功创建！")  # 明确是创建而不是加载
        except Exception as e:
            log(f"创建向量库时出错: {str(e)}")
            log(f"错误堆栈:\n{traceback.format_exc()}")
            st.error(f"创建知识库时出错: {e}")
            raise
    else:
        # 如果索引已存在，直接加载
        log("索引路径存在，开始加载索引...")
        try:
            db = create_vector_store(None, model_name=MODEL_NAME, index_path=INDEX_PATH)
            log("向量库加载成功")
            st.success("知识库已成功加载！")  # 明确是加载
        except Exception as e:
            log(f"加载向量库时出错: {str(e)}")
            log(f"错误堆栈:\n{traceback.format_exc()}")
            st.error(f"加载知识库时出错: {e}")
            raise
    return db

def get_context_retriever_chain(_vector_store):
    """创建能够感知历史的检索链"""
    log("创建历史感知检索链...")
    try:
        llm = Ollama(model="deepseek-r1:latest")
        log("成功加载Ollama模型")
        retriever = _vector_store.as_retriever()
        log("成功创建检索器")
    
        # 这个prompt用于将用户的新问题，根据历史记录，改写成一个独立的、更完整的查询
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "根据上面的对话历史，生成一个独立的、无需上下文就能理解的搜索查询。")
        ])
    
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        log("历史感知检索链创建成功")
        return retriever_chain
    except Exception as e:
        log(f"创建历史感知检索链时出错: {str(e)}")
        log(f"错误堆栈:\n{traceback.format_exc()}")
        raise

def get_conversational_rag_chain(_retriever_chain):
    """创建最终的对话式RAG链"""
    log("创建对话式RAG链...")
    try:
        llm = Ollama(model="deepseek-r1:latest")
        log("成功加载Ollama模型")
    
        # 这个prompt是最终用于生成答案的
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            **指令：** 你是一个AI助手，请根据用户的问题，从下面的文档中检索相关信息，并生成答案。
            **要求：**
            1. 只根据文档中的信息回答问题，不要编造信息。
            2. 如果文档中没有相关信息， politely 拒绝回答。
            3. 如果文档中有多个相关段落，请按照段落顺序依次回答。
            4. 请先输出思考过程（用```think```包裹），再输出最终答案。
            **文档：** {context}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
    
        stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
        log("文档处理链创建成功")
    
        rag_chain = create_retrieval_chain(_retriever_chain, stuff_documents_chain)
        log("对话式RAG链创建成功")
        return rag_chain
    except Exception as e:
        log(f"创建对话式RAG链时出错: {str(e)}")
        log(f"错误堆栈:\n{traceback.format_exc()}")
        raise

# --- 主程序 ---

try:
    log("应用启动...")
    # 1. 加载向量数据库
    log("步骤1: 加载向量数据库...")
    vector_store = get_vector_store()
    log("向量数据库加载/创建成功")

    # 2. 创建核心的RAG链
    log("步骤2: 创建核心的RAG链...")
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    log("RAG链创建成功")

    # 3. 初始化会话状态，用于存储聊天记录
    log("步骤3: 初始化会话状态...")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="你好！我是你的专属知识库助手，有什么可以帮到你？"),
        ]
        log("聊天历史初始化成功")

    # 4. 渲染聊天历史记录
    log("步骤4: 渲染聊天历史记录...")
    for message in st.session_state.chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
            st.markdown(message.content)
    log("聊天历史渲染完成")

    # 5. 获取用户输入
    log("步骤5: 等待用户输入...")
    if PROMPT := st.chat_input("请输入您的问题..."):
        log(f"收到用户输入: {PROMPT}")
        # 将用户输入添加到历史记录并显示
        st.session_state.chat_history.append(HumanMessage(content=PROMPT))
        with st.chat_message("Human"):
            st.markdown(PROMPT)
        # 调用RAG链获取AI回答
        with st.chat_message("AI"):
            with st.spinner("正在思考中..."):
                log("调用RAG链生成回答...")
                try:
                    # 这里是核心调用，传入了完整的历史记录
                    response = conversation_rag_chain.invoke({
                        "chat_history": st.session_state.chat_history,
                        "input": PROMPT
                    })
                    log("RAG链调用成功")
                    # 我们只显示最终的答案
                    # 提取思考过程和最终答案
                    answer = response["answer"]
                    think_start = answer.find('```think')
                    think_end = answer.find('```', think_start + 6)
                    
                    if think_start != -1 and think_end != -1:
                        thought_process = answer[think_start+6:think_end].strip()
                        final_answer = answer[think_end+3:].strip()  # 调整索引以跳过结束标记
                    else:
                        thought_process = "模型未提供思考过程"
                        final_answer = answer
                    
                    # 显示最终答案
                    st.markdown(final_answer)
                    
                    # 用expander隐藏思考过程
                    with st.expander("查看模型思考过程"):
                        st.markdown(thought_process)
                    # 将AI的回答也添加到历史记录
                    st.session_state.chat_history.append(AIMessage(content=response["answer"]))
                    log("回答已添加到聊天历史")
                except Exception as rag_e:
                    log(f"RAG链调用出错: {str(rag_e)}")
                    log(f"错误堆栈:\n{traceback.format_exc()}")
                    st.error(f"生成回答时出错: {rag_e}")


except Exception as e:
    log(f"应用运行出错: {str(e)}")
    log(f"错误堆栈:\n{traceback.format_exc()}")
    st.error(f"发生了一个错误: {e}\n\n详细错误信息已输出到终端，请查看。")