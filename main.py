import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
from dotenv import load_dotenv
import os
import re

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] Multi File")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


# 전역 progress bar 선언
progress_bar = None
progress_text = None

st.title("한화생명 보험상품 QA💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 보험상품 파일 로드
    load_button = st.button("상품 정보 로드")

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 텍스트 정리 함수
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)  # 빈 줄 제거
    text = re.sub(r'<br>', '\n', text)  # 빈 줄 제거
    text = re.sub(r'\s+', ' ', text).strip()  # 여백 정리
    return text

@st.cache_resource(show_spinner="파일을 처리 중입니다...")
def load_files(base_dir="./.cache/files/hanwhalife"):
    vector_store_path = "./.cache/embeddings/hanwhalife_vectors"
    vector_store_file = f"{vector_store_path}/index.faiss"
    
    # 저장된 벡터스토어가 있다면 불러오기
    if os.path.exists(vector_store_file):
        print("저장된 벡터스토어를 불러옵니다.")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            vector_store_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    else:
        all_docs = []
        pdf_files = []

        # 단계 1: 파일 로드
        print(f"파일 로드 시작")
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        
        progress_text = "문서 로딩 중....."
        progress_bar = st.progress(0)

        for idx, file_path in enumerate(pdf_files):
            try:
                print(f"파일 로드 중: {file_path}")
                loader = PDFPlumberLoader(file_path)
                docs = loader.load()

                # 문서 전처리 & 각 문서에 메타데이터 추가
                file_name = os.path.basename(file_path)
                for doc in docs:
                    doc.page_content = clean_text(doc.page_content)
                    doc.metadata.update({
                        'source': file_path,
                        'file_name': file_name,
                        'page': doc.metadata.get('page', 0),
                        'total_pages': len(docs)
                    })
                
                all_docs.extend(docs)

                progress = (idx + 1) / len(pdf_files)
                progress_bar.progress(progress, text=f"{progress_text} ({idx + 1}/{len(pdf_files)})")

            except Exception as e:
                st.warning(f"파일 로드 실패: {file_path}\n에러: {str(e)}")
        
        if progress_bar is not None:
            progress_bar.empty()
            progress_bar = None

        # 단계 2: 문서 분할(Split Documents)
        print(f"문서 분할 중: {len(all_docs)}개")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(all_docs)

        # 단계 3: 임베딩(Embedding) 생성
        print(f"임베딩 생성 중")
        embeddings = OpenAIEmbeddings()

        # 단계 4: DB 생성(Create DB) 및 저장
        # 벡터스토어를 생성합니다.
        print(f"벡터스토어 생성 중: {len(split_documents)}개")
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

        vector_store_path = "./.cache/embeddings/hanwhalife_vectors"
        vectorstore.save_local(vector_store_path)

        # 단계 5: 검색기(Retriever) 생성
        # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
        print(f"검색기 생성 중")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        return retriever

# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    
    vector_store_path = "./.cache/embeddings/hanwhalife_vectors"
    vectorstore.save_local(vector_store_path)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


# 체인 생성
def create_chain(retriever, model_name="gpt-4o"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("prompts/pdf-rag-multi.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 파일이 업로드 되었을 때
#if uploaded_file:
if load_button:
    # 파일 로드 후 retriever 생성 (작업시간이 오래 걸릴 예정...)
    retriever = load_files()
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")
