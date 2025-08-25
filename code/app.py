from cmd import PROMPT
import streamlit as st
import os
import sys
import time
import traceback
from langchain_huggingface import HuggingFaceEmbeddings
from main import load_and_split_documents, create_vector_store
from reranker import RerankerModel, RerankerRetriever

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
st.set_page_config(page_title="工业知识问答助手 v3.0", page_icon="🤖", layout="wide")
st.title("🤖 工业知识问答助手 v3.0 (支持连续对话 + Reranker增强)")

# --- 核心功能函数 ---

def get_project_root():
    """获取项目根目录"""
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == "code":
        # 如果在code目录中，需要回到上级目录
        return os.path.dirname(current_dir)
    else:
        return current_dir

# 不使用缓存，确保每次都重新检查索引状态
def get_vector_store():
    """创建并缓存向量数据库，返回向量库和嵌入模型实例"""
    # 获取项目根目录
    project_root = get_project_root()
    
    KNOWLEDGE_BASE_DIR = os.path.join(project_root, "data", "file")
    INDEX_PATH = os.path.join(project_root, "data", "INDEX")  # 统一使用大写，与main.py保持一致
    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # 显式指定嵌入模型，与main.py保持一致
    
    # 打印当前工作目录和INDEX_PATH的绝对路径
    current_dir = os.getcwd()
    absolute_index_path = os.path.abspath(INDEX_PATH)
    st.write(f"当前工作目录: {current_dir}")
    st.write(f"索引路径(绝对路径): {absolute_index_path}")
    st.write(f"索引路径是否存在: {os.path.exists(INDEX_PATH)}")
    st.write(f"使用的嵌入模型: {MODEL_NAME}")
    
    # 显式配置嵌入模型参数，确保与main.py一致
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # 强制重建索引的逻辑
    force_rebuild = st.session_state.get('force_rebuild_index', False)
    
    # 检查是否需要重建索引
    need_rebuild = False
    
    # 输出嵌入模型维度信息，用于调试
    try:
        # 使用一个简单的句子测试嵌入维度
        test_embedding = embeddings.embed_query("测试句子")
        embedding_dim = len(test_embedding)
        st.write(f"嵌入模型维度: {embedding_dim}")
        log(f"嵌入模型维度: {embedding_dim}")
        
        # 如果索引存在且不是强制重建，检查维度是否匹配
        if os.path.exists(INDEX_PATH) and not force_rebuild:
            try:
                # 尝试加载索引来检查维度
                temp_db = create_vector_store(None, model_name=MODEL_NAME, index_path=INDEX_PATH, embeddings=embeddings)
                log("索引维度检查通过")
            except Exception as dim_check_e:
                if "assert d == self.d" in str(dim_check_e):
                    log(f"索引维度不匹配，需要重建: {dim_check_e}")
                    need_rebuild = True
                else:
                    log(f"索引检查出错: {dim_check_e}")
                    need_rebuild = True
        
    except Exception as e:
        st.error(f"获取嵌入模型维度时出错: {e}")
        log(f"获取嵌入模型维度时出错: {e}")
        need_rebuild = True
    
    # 如果需要重建或索引不存在，则创建新索引
    if need_rebuild or not os.path.exists(INDEX_PATH) or force_rebuild:
        if need_rebuild or force_rebuild:
            st.warning("正在重建索引...")
            # 删除旧的索引文件
            try:
                import shutil
                if os.path.exists(INDEX_PATH):
                    shutil.rmtree(INDEX_PATH)
                    log("旧索引已删除")
                    st.success("旧索引已删除")
            except Exception as del_e:
                log(f"删除旧索引失败: {del_e}")
                st.error(f"删除旧索引失败: {del_e}")
        else:
            st.info("首次运行或知识库更新：正在创建新的向量索引，请稍候...")
        
        log("开始创建新索引...")
        try:
            docs = load_and_split_documents(KNOWLEDGE_BASE_DIR)
            log(f"成功加载并切分文档，共 {len(docs)} 个文档块")
            db = create_vector_store(docs, model_name=MODEL_NAME, index_path=INDEX_PATH, embeddings=embeddings)
            log("向量库创建成功")
            st.success("知识库已成功创建！")
            
            # 重置强制重建标志
            if force_rebuild:
                st.session_state.force_rebuild_index = False
                
        except Exception as e:
            log(f"创建向量库时出错: {str(e)}")
            log(f"错误堆栈:\n{traceback.format_exc()}")
            st.error(f"创建知识库时出错: {e}")
            raise
    else:
        # 索引存在且维度匹配，直接加载
        log("索引路径存在且维度匹配，开始加载索引...")
        try:
            db = create_vector_store(None, model_name=MODEL_NAME, index_path=INDEX_PATH, embeddings=embeddings)
            log("向量库加载成功")
            st.success("知识库已成功加载！")
        except Exception as e:
            log(f"加载向量库时出错: {str(e)}")
            log(f"错误堆栈:\n{traceback.format_exc()}")
            st.error(f"加载知识库时出错: {e}")
            raise
    
    return db, embeddings

# 不使用缓存，确保每次都根据当前状态决定是否加载模型
def get_reranker_model():
    """创建reranker模型
    
    返回:
        RerankerModel或None: 如果启用了Reranker则返回模型实例，否则返回None
    """
    try:
        # 检查是否启用reranker（从session_state中获取，默认为True）
        use_reranker = st.session_state.get('use_reranker', True)
        log(f"当前Reranker状态: {'启用' if use_reranker else '禁用'}")
        
        if not use_reranker:
            log("Reranker已禁用，跳过加载，返回None")
            # 当用户取消勾选"启用Reranker模型"时，返回None表示不使用reranker
            return None
            
        log("加载Reranker模型...")
        # 默认使用CPU，如果有CUDA则使用GPU
        device = "cuda" if st.session_state.get('use_gpu', False) else "cpu"
        model_name = "Qwen/Qwen3-Reranker-0.6B"  # 可以根据需要更换模型
        
        start_time = time.time()
        reranker = RerankerModel(model_name=model_name, device=device)
        load_time = time.time() - start_time
        log(f"Reranker模型加载完成，耗时: {load_time:.2f}秒")
        
        return reranker
    except Exception as e:
        log(f"加载Reranker模型时出错: {str(e)}")
        log(f"错误堆栈:\n{traceback.format_exc()}")
        st.warning(f"Reranker模型加载失败: {e}，将使用基础检索")
        return None

def get_context_retriever_chain(_vector_store, embeddings):
    """创建能够感知历史的检索链"""
    log("创建历史感知检索链...")
    try:
        llm = Ollama(model="qwen3:8b")
        log("成功加载Ollama模型")
        
        # 获取基础检索器
        base_retriever = _vector_store.as_retriever(search_kwargs={'k': 20})  # 增加检索数量以供reranker筛选
        
        # 检查是否使用reranker（当用户取消勾选"启用Reranker模型"时，get_reranker_model()返回None）
        reranker = get_reranker_model()
        log(f"获取到的reranker模型: {reranker}")
        
        if reranker is not None:
            log("使用Reranker增强检索")
            # 创建增强检索器 - 两阶段检索：先向量检索，再重排序
            try:
                retriever = RerankerRetriever(
                    vector_retriever=base_retriever,  # 基础向量检索器
                    reranker=reranker,               # 重排序模型
                    top_k_vector=20,                 # 向量检索返回的文档数量
                    top_k_final=5                    # 最终返回的文档数量
                )
                log("Reranker增强检索器创建成功")
            except Exception as e:
                log(f"创建RerankerRetriever时出错: {e}，将使用基础检索器")
                retriever = base_retriever
        else:
            # 当reranker为None时（用户取消勾选"启用Reranker模型"），直接使用基础向量检索器
            log("使用基础检索器（未启用Reranker）")
            retriever = base_retriever            # 直接使用向量检索结果
        log("成功创建检索器")
        # 验证嵌入维度
        try:
            test_query = "测试查询"
            test_embedding = embeddings.embed_query(test_query)
            embedding_dim = len(test_embedding)
            log(f"查询嵌入维度: {embedding_dim}")
        except Exception as e:
            log(f"验证嵌入维度时出错: {e}")
    
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
        llm = Ollama(model="qwen3:8b")
        log("成功加载Ollama模型")
    
        # 这个prompt是最终用于生成答案的
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            **指令：** 你是一个AI助手，请根据用户的问题，从下面的文档中检索相关信息，并生成答案。
            **要求：**
            1. 只根据文档中的信息回答问题，不要编造信息。
            2. 如果文档中没有相关信息， politely 拒绝回答。
            3. 如果文档中有多个相关段落，请按照段落顺序依次回答。
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

# --- 侧边栏配置 ---
with st.sidebar:
    st.header("系统配置")
    
    # Reranker配置 - 控制是否使用重排序模型来提高检索质量
    st.subheader("Reranker配置")
    # 初始化session_state中的use_reranker状态（默认为True，即启用Reranker）
    if 'use_reranker' not in st.session_state:
        st.session_state.use_reranker = True
    
    # 创建复选框UI元素，初始值为当前session_state中的值
    use_reranker = st.checkbox("启用Reranker模型", value=st.session_state.use_reranker, 
                            help="启用后将使用Qwen3-Reranker模型对检索结果进行重排序，提高相关性")
    
    # 当复选框状态发生变化时（用户点击了复选框）
    if use_reranker != st.session_state.use_reranker:
        # 更新session_state中的状态
        st.session_state.use_reranker = use_reranker
        # 重新运行应用以应用更改 - 这将触发重新加载模型和检索器
        # 当取消勾选时，get_reranker_model()将返回None，系统将使用基础检索器
        st.rerun()  # 重新运行应用以应用更改（注：st.experimental_rerun()已弃用，改用st.rerun()）
    
    # GPU配置
    st.subheader("硬件配置")
    if 'use_gpu' not in st.session_state:
        st.session_state.use_gpu = True
    
    use_gpu = st.checkbox("使用GPU加速", value=st.session_state.use_gpu)
    if use_gpu != st.session_state.use_gpu:
        st.session_state.use_gpu = use_gpu
        st.rerun()  # 重新运行应用以应用更改（注：st.experimental_rerun()已弃用，改用st.rerun()）
    

    
    # 索引管理
    st.subheader("🗄️ 索引管理")
    
    # 强制重建索引选项
    if st.checkbox("强制重建索引", help="勾选后将强制重建索引，解决维度不匹配问题"):
        st.session_state.force_rebuild_index = True
        st.warning("已启用强制重建索引，请刷新页面")
    
    if st.button("🔄 重建知识库索引"):
        try:
            import shutil
            INDEX_PATH = os.path.join(get_project_root(), "data", "INDEX")
            if os.path.exists(INDEX_PATH):
                shutil.rmtree(INDEX_PATH)
                st.success("索引已删除，请刷新页面重建")
                st.rerun()
            else:
                st.info("索引不存在，无需删除")
        except Exception as e:
            st.error(f"重建索引失败: {e}")

# --- 主程序 ---

try:
    log("应用启动...")
    # 1. 加载向量数据库和嵌入模型
    log("步骤1: 加载向量数据库和嵌入模型...")
    vector_store, embeddings = get_vector_store()
    log("向量数据库和嵌入模型加载/创建成功")

    # 2. 创建核心的RAG链
    log("步骤2: 创建核心的RAG链...")
    # 传递embeddings实例给检索链，确保查询时使用相同的嵌入模型
    retriever_chain = get_context_retriever_chain(vector_store, embeddings)
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
                    # 记录开始时间，用于性能评估
                    start_time = time.time()
                    
                    # 这里是核心调用，传入了完整的历史记录
                    response = conversation_rag_chain.invoke({
                        "chat_history": st.session_state.chat_history,
                        "input": PROMPT
                    })
                    
                    # 计算响应时间
                    response_time = time.time() - start_time
                    log(f"RAG链调用成功，总响应时间: {response_time:.2f}秒")
                    # 我们只显示最终的答案
                    # 提取思考过程和最终答案
                    answer = response["answer"]
                    
                    # 初始化变量
                    thought_process = "模型未提供思考过程"
                    final_answer = answer

                    # 使用if/elif/else结构确保只匹配一种格式
                    # 检测思考过程标记格式1: ```think...```
                    think_start = answer.find('```think')
                    if think_start != -1:
                        think_end = answer.find('```', think_start + 7)
                        if think_end != -1:
                            thought_process = answer[think_start+7:think_end].strip()
                            final_answer = answer[think_end+3:].strip()

                    # 检测思考过程标记格式2: <think>...</think>
                    elif answer.find('<think>') != -1:
                        think_start = answer.find('<think>')
                        think_end = answer.find('</think>', think_start + 7)
                        if think_end != -1:
                            thought_process = answer[think_start+7:think_end].strip()
                            final_answer = answer[think_end+8:].strip()

                    # 检测思考过程标记格式3: 思考过程:...答案:...
                    elif answer.find('思考过程:') != -1:
                        think_heading_start = answer.find('思考过程:')
                        answer_heading_start = answer.find('答案:', think_heading_start)
                        if answer_heading_start != -1:
                            thought_process = answer[think_heading_start+5:answer_heading_start].strip()
                            final_answer = answer[answer_heading_start+3:].strip()
                        else:
                            # 未找到"答案:"标记，将"思考过程:"后的所有内容作为思考过程
                            thought_process = answer[think_heading_start+5:].strip()
                            final_answer = answer[:think_heading_start].strip()
                    
                    # 确保最终答案不为空
                    if not final_answer:
                        final_answer = thought_process
                        thought_process = "模型未提供思考过程"
                    
                    # 显示最终答案
                    st.markdown(final_answer)
                    
                    # 用expander隐藏思考过程
                    with st.expander("查看模型思考过程"):
                        st.markdown(thought_process)
                    # 将AI的回答也添加到历史记录
                    st.session_state.chat_history.append(AIMessage(content=final_answer))
                    log("回答已添加到聊天历史")
                    

                except Exception as rag_e:
                    log(f"RAG链调用出错: {str(rag_e)}")
                    log(f"错误堆栈:\n{traceback.format_exc()}")
                    st.error(f"生成回答时出错: {rag_e}")


except Exception as e:
    log(f"应用运行出错: {str(e)}")
    log(f"错误堆栈:\n{traceback.format_exc()}")
    st.error(f"发生了一个错误: {e}\n\n详细错误信息已输出到终端，请查看。")