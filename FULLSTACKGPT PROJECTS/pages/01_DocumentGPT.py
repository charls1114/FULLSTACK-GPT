import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

# 업로드한 파일 임베딩 후 retriever 리턴
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    st.write(file_path)
    with open(file_path,"wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    st.write(cache_dir.root_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=400,
        chunk_overlap=50,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriver = vectorstore.as_retriever()
    return retriver

# 채팅 내역 기록
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

# 채팅 내역 출력
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

# retriver 답변 형식화
def format_documents(docs):
    return "\n\n".join(document.page_content for document in docs)

# session_state(cache)에 메세지 저장
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

class ChatcallbackHandler(BaseCallbackHandler):

    message = ""
    def on_llm_start(self, *args, **kwargs):
        # 빈 위젯 생성
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        # 최종 답변 저장
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        # 스트리밍으로 ai 답변 출력
        self.message += token
        self.message_box.markdown(self.message)

# 모델 설정
llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatcallbackHandler()
    ],
)

# stuff 방식 prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    Answer the question using ONLY the following context. If you don't know the answer,
    just say you don't know. DON'T make anything up.

    context: {context}
    """),
    ("human","{question}"),

])

st.title("DocumentGPT")

st.markdown(
"""
    Welcome!

    Use this chatbot to ask question to an AI about your files!
"""
)

# 파일 업로드 부분
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type = ["pdf", "txt", "docx"]
        )

# 채팅 로직 구현 부분
if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save = False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_documents),
            "question": RunnablePassthrough()
        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []

