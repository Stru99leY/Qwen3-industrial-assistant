from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever


def build_history_aware_retriever(llm: Ollama, retriever):
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "根据上面的对话历史，生成一个独立的、无需上下文就能理解的搜索查询。"),
    ])
    return create_history_aware_retriever(llm, retriever, prompt)


def build_conversational_rag(retriever_chain):
    llm = Ollama(model="qwen3:8b")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            你是一个AI助手，请根据用户的问题，从下面的文档中检索相关信息，并生成答案。
            要求：
            1. 只根据文档中的信息回答问题，不要编造信息。
            2. 如果文档中没有相关信息，请明确说明。
            3. 如果文档中有多个相关段落，请按照段落顺序依次回答。
            文档：{context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


