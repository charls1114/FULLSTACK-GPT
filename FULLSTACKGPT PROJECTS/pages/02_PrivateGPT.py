import streamlit as st
from langchain.chat_models import ChatOllama

# 대화 내역 저장
from langchain.memory import ConversationSummaryBufferMemory

# retriever 패키지들
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore

# 프롬프트 패키지들
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

# 스트리밍 표현 패키지
from langchain.callbacks.base import BaseCallbackHandler


# 업로드한 파일 임베딩 후 retriever 리턴
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path,"wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(model="mistral:latest")
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

st.set_page_config(
    page_title="Private GPT",
    page_icon="📃",
)

st.title("Private GPT")

st.markdown(
"""
    Welcome!

    Use this chatbot to ask question to an AI about your files in private!
"""
)

# 모델 설정
llm_model = ChatOllama(
    model= "mistral:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatcallbackHandler()
    ],
)

# chat_memory = ConversationSummaryBufferMemory(
#     llm = llm_model,
#     max_token_limit = 120,
#     memory_key = "chat_history",
#     return_messages = True,
# )

# stuff 방식 prompt
prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the following context and not your training data. If you don't know the answer just say you don't know. DON'T make anything up.
    
    Context: {context}
    Question:{question}
    """
)


# 파일 업로드 부분
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type = ["pdf", "txt", "docx"],
        accept_multiple_files=False,
    )

# 채팅 로직 구현 부분
if file:
    retriever = embed_file(file)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_documents),
            "question": RunnablePassthrough()
        } | prompt | llm_model
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    st.session_state["messages"] = []