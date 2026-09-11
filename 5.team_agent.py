# 여러 에이전트(노드)가 협력해서 에세이를 작성하는 멀티 에이전트 시스템
# 그래프 흐름:
#   planner -> researcher_plan -> generate -|
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List

from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from templates import *

class AgentState(TypedDict):
    task : str
    plan : str
    draft : str
    critique : str
    content : List[str]
    revision_number : int
    max_revisions : int

# pydantic의 상속을 받음(BaseModel)
class Queries(BaseModel):
    queries : List[str]

model = ChatOpenAI(model='gpt-4o', temperature=0.7)

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1,
                                                         doc_content_chars_max=500))

def plan_node(state:AgentState):
    print(f'[plan node] .... 계획 수립 중 ....')
    response = model.invoke(
        [
            # 1. 시스템 프롬프트 정의
            SystemMessage(content=PLAN_PROMPT),
            # 2. 인간의 요청 넣기
            HumanMessage(content=state['task'])
        ])
    return {'plan':response.content}

# 주제에 관련된 내용을 문헌 검색
def research_node(state: AgentState):
    return _run_research(state, state['task'])


def generate_node(state: AgentState):
    pass


def reflection_node(state: AgentState):
    pass

# 검색 결과에 대해 평가
def critique_node(state: AgentState):
    return _run_research(state, state['critique'])

def _run_research(state:AgentState, user_content):
    #queries는 Queries라는 클래스의 output을 만드는 모델의 실행 결과
    queries_ = model.structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PROMPT),
        HumanMessage(content=user_content)
    ])
    
    content = list(state.get('content') or [])
    #결과물로 받은 List[str] 형태의 쿼리들을 for문 q로 하나씩 빼온다
    for q in queries_.queries:
        print(f'검색 중 ... : {q}')
        
        try :
            result = wiki.invoke({'query': q})
        except Exception as e:
            print(f'검색 실패...')
            continue
        
        content.append(result)
    
    return { content : content }

def should_continue(state: AgentState):
    pass


def build_graph():
    
    graph = StateGraph(AgentState)
    graph.add_node('planner', plan_node)
    graph.add_node('researcher', research_node)
    graph.add_node('generator', generate_node)
    graph.add_node('reflect', reflection_node)
    graph.add_node('critique', critique_node)
    
    #연결
    graph.set_entry_point('planner')
    graph.add_edge('planner', 'researcher')
    graph.add_edge('researcher', 'generator')
    graph.add_conditional_edges(
        'generator',
        should_continue,
        {END:END,
         'reflect':'reflect'}
    )
    graph.add_edge('reflect', 'critique')
    graph.add_edge('critique', 'generate')
    
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

if __name__ == '__main__':
    input('무엇이든 물어보세요 : \n')
    result = build_graph()
    print(result)
