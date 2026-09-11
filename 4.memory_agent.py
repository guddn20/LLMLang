#   1. MemorySaver (checkpointer)
#       -> thread_id가 같으면 이전 대화를 기억한다.
#       -> thread_id가 다르면 새 대화로 시작한다
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# react-agent : React 패턴을 자동으로 구성
# memory-saver : 대화 이력을 RAM(내 컴퓨터 하드웨어)에 저장
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(
    top_k_results=1, doc_content_chars_max=500
))
# top_k_results = 검색 결과 중 k개만 가져와라.
# doc_content_chars_max = 검색 결과의 길이 제한
tools = [wiki]

# AI모델 셋업
chat = ChatOpenAI(model='gpt-4o', temperature=0)
# AI모델이 활용할 메모리 정의
memory = MemorySaver()
# 에이전트(react)
agent = create_react_agent(
    model=chat,
    tools=tools,
    checkpointer=memory
)

def chat_agent(query, id):
    config = {"configurable": {"thread_id":id}}
    response = agent.invoke({'messages':['human',query]},
                            config=config)
    print(response['messages'][-1].content)

if __name__ == '__main__':
    for i in range(5):
        if i%2 == 0:
            id = 'even'
        else:
            id = 'odd'
            
        query = input("무엇이 궁금하세요?")
        chat_agent(query, id)
