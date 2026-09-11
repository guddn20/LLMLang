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

import wikipedia
wikipedia.set_user_agent('guddn20@gmail.com')
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2,
                                                         doc_content_chars_max=2000))



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
    print(f'[research_node] .... 리서치 중....')
    return _run_research(state, state['task'])

def generate_node(state: AgentState):
    print(f'[generate_node] .... 생성 중....')
    #생성 시도 제한
    rev = state.get('revision_number', 1)
    
    content_str = '\n'.join(state.get('content') or [])
    response = model.invoke([
        SystemMessage(content=WRITER_PROMPT.format(content=content_str)),
        HumanMessage(content=f"{state['content']} Here is my Plan : {state['plan']}")
    ])
    
    return {'draft' : response.content, 'revision_number': rev+1 }

# 비평가 내가 쓴 에세이
def reflection_node(state: AgentState):
    print(f'[reflection node] .... 비평 중 ....')
    response = model.invoke([
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ])
    return {'critique': response.content}

# 검색 결과에 대해 평가
def critique_node(state: AgentState):
    return _run_research(state, state['critique'])

def _run_research(state:AgentState, user_content):
    #queries는 Queries라는 클래스의 output을 만드는 모델의 실행 결과
    queries_ = model.with_structured_output(Queries).invoke([
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
    
    return { 'content' : content }

# 내가 state에 가지고 있는 revision_number가 max_revision을 넘으면 끝
# 그렇지 않으면 다시 reflect로 이동
def should_continue(state: AgentState):
    if state['revision_number'] > state['max_revisions']:
        return END
    else:
        return 'reflect'


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
    graph.add_edge('critique', 'generator')
    
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

if __name__ == '__main__':
    graph = build_graph()
    # print(graph.get_graph().print_ascii())

    task = input('어떤 주제에 대해 글을 쓸까요? : \n')
    thread_id = {'configurable' : {'thread_id' : 'essay-1'}}
    
    for s in graph.stream(
        {'task' : task,
        'max_revisions' : 5,
        'revision_number' : 1,
        'content' : []},
        thread_id):
        node_list = list(s.keys())[0]
        print(f'{node_list} 완료')
        
    final = graph.get_state(thread_id)
    print(final.values['draft'])