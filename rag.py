# 질문을 할 때, 질문 답변에 참고할 자료를 미리 세팅 -> 자료를 찾아 같이 전달
# Retrieval(검색) Augmented(증강) Generation(생성)
# 벡터DB : 문장들을 임베딩해서 비슷한 문장을 분류해놓고 유사도 검색
# Augmented : 알고리즘을 이용해서 유사도 검색
import os
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

# 1. 폴더 안에 pdf를 읽어서 하나의 docs로 세팅
def load_pdfs(path):
    # 1-1.오류상황 1 -> 경로가 틀린 경우
    if not path:
        print(f'경로가 틀렸습니다.')
    # 1-2. 오류상황 2 -> 폴더에 pdf가 없는 경우
    pdf_lists = [os.path.join(path, x) for x in os.listdir(path) if 'pdf' in x]
    if len(pdf_lists) < 1:
        # raise 시스템 오류, 알림 실행
        raise FileNotFoundError(f'{path}에 pdf 파일이 존재하지 않습니다.')

    docs = []
    for pdf in pdf_lists:
        docs.extend(PyPDFLoader(pdf).load())

    print(f'{len(docs)} 개의 문서 취득')
    return docs

# 2. docs를 청크화
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_docs(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                              chunk_overlap = chunk_overlap)

    split_docs = splitter.split_documents(docs)
    return split_docs

# 3. 청크를 벡터화 -> 벡터DB
# https://docs.pinecone.io/
#
# Chroma에서 초기 데이터를 만드는 과정!
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
def vectorstore(dir, collection):
    return Chroma(
        collection_name=collection,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=dir,
    )

# 새로운 문서가 들어왔을 때, split한 후 기존의 vectordb에 추가
def add_to_vectorstore(vectordb, splits):
    vectordb.add_documents(splits)
    return vectordb

# 4. 리트리버
# K는? -> 유사도 검색 후, K개 만큼의 유사 문서를 return
def build_retrieval(vectordb, k):
    return vectordb.as_retriever(search_kwargs={'k':k})

# 5. 리트리버 얹은 chain 정의
def build_rag_chain(chat, retriever):
    
    prompt = ChatPromptTemplate.from_messages([
        ('human', ''' Answer the question using only the context below.
                    Q : {input}
                    C : {context}
                    ''')
    ])
    
    combine = create_stuff_documents_chain(chat, prompt, document_separator='\n\n')
    
    return create_retrieval_chain(retriever, combine)

if __name__ == '__main__':
    
    #내가 pdf를 특정 장소에 가지고 있는가?
    #내가 가진 pdf가 벡터 DB에 있는가?
    vectordb = vectorstore('./vectordb', 'pdf_docs')
    if len(vectordb.get(limit=1)['ids']) > 0:
    #있다 -> 추가 안해도 됨
        print(f'기존 파일 재사용')
    else:
    #없다 -> 추가해야함
        docs = load_pdfs('./pdf')
        split_doc = split_docs(docs)
        print(f'{len(docs)} -> {len(split_doc)} 개로 나누어짐')
        add_to_vectorstore(vectordb=vectordb, splits=split_doc)
        
    #chat, retriever
    chat = ChatOpenAI(temperature = 0, model = 'gpt-4o')
    retrieval = build_retrieval(vectordb=vectordb, k=3)
    rag_chain = build_rag_chain(chat, retriever=retrieval)

    question = input('경제 용어를 물어보세요 : \n')
    result = rag_chain.invoke({'input':question}, {'context':retrieval})
    print(result)