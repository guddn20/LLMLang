# Agent가 포함된 랭 그래프
from dotenv import load_dotenv

load_dotenv()

# Annotated[자료형, 덮어쓰기] -> '자료형' 데이터가 여기 들어올거야. 데이터가 새로 들어올 때 '덮어쓰기' 할거야.
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END, MessagesState

# 에이전트 구성을 위해 필요한 langchain 컴포넌드
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

# 툴을 사용해서 답변해줄 llm을 정의
from langchain_openai import ChatOpenAI

# 코랩용
import os
# from google.colab import userdta
# os.environ[변경불가한 약속어] = userdata.get(변경가능(내 코랩 이름))
# os.environ['OPEN_API_KEY'] = userdata.get('OPENAI_API_KEY')

# '툴'이라는 함수는 없었음
#'시스템 프롬프트' 안에 이런 함수들이 있다~ 알려주고,
# known_function = {'calculate':함수이름} < 이 함수 딕셔너리를 전달
#'툴'라이브러리를 차용
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from IPython.display import Image, display

@tool
def get_weather(location:str):
    '''Return the weather information'''
    if location in ['서울', '인천', '경기']:
        return '맑음'
    else:
        return '흐림'

@tool
def get_sum(a, b):
    '''Add some figures'''
    return a+b

tools = [get_weather, get_sum]

tool_node = ToolNode(tools)
# model은 tools를 인지하게 되고, 질문을 받았을 때 조회하면서 사용 가능
model_with_tools = ChatOpenAI(model='gpt-4o', temperature=0.5).bind_tools(tools)

def should_continue(state:MessagesState) -> Literal['tools', END]:
    message = state['messages']
    last_message = message[-1] #state가 가지고 있는 messages의 가장 마지막(최신)
    if last_message.tool_calls:
        return 'tools'
    return END

def call_openai(state:MessagesState):
    message = state['messages']
    response = model_with_tools.invoke(message)
    return {'messages':[response]}

OUT_DIR = './'
# compile()된 그래프는 get_graph()로 내부 구조(노드/엣지)를 꺼낼 수 있고
# 거기에 draw_mermaid()를 부르면 mermaid 다이어그램 문법이 문자열로 나옴
def save_graph(graph, name, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    code = graph.get_graph().draw_mermaid()

    # .md -> GitHub / VSCode 미리보기에서 바로 렌더링됨
    with open(os.path.join(out_dir, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(f"# {name}\n\n```mermaid\n{code}\n```\n")

    # .html -> 브라우저로 열면 CDN에서 mermaid를 받아 그려줌
    with open(os.path.join(out_dir, f"{name}.html"), "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>{name}</title>
<style>body{{font-family:system-ui,sans-serif;margin:40px;background:#fafafa}}
.box{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:24px}}</style>
</head><body>
<h1>{name}</h1>
<div class="box"><pre class="mermaid">{code}</pre></div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true}});</script>
</body></html>""")

    # .png -> mermaid.ink 외부 API를 쓰므로 인터넷 필요. 실패해도 죽지 않게 감쌈
    try:
        with open(os.path.join(out_dir, f"{name}.png"), "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        png = " + png"
    except Exception:
        png = " (png 실패 -> html 사용)"

    print(f"[save_graph] {name}.md + .html{png}")
    return code


# 툴이 포함된 그래프 만들기
workflow = StateGraph(MessagesState)
workflow.add_node('agent', call_openai)
workflow.add_node('tools', tool_node)

workflow.add_edge(START, 'agent')
workflow.add_conditional_edges('agent', should_continue)
workflow.add_edge('tools', 'agent')

graph = workflow.compile()

if __name__ == '__main__':
    # result = model_with_tools.invoke("서울 날씨는 어때?").tool_calls
    # result = model_with_tools.invoke("5+5는 뭐야?").tool_calls
    # result = model_with_tools.invoke("서울 날씨는 어때?").tool_calls
    # print(result)
    
    #그래프 저장
    # result = display(Image(graph.get_graph().draw_mermaid_png()))
    # save_graph(graph=graph,name='tool')
    
    query = input('무엇이든 물어보슈\n')
    
    response = graph.invoke({'messages' : [HumanMessage(content = query)]})
    print(response['messages'][-1].content)
    
    # for c in graph.stream({'messages' : ['human', '서울의 날씨는 어때?']},
    #                        stream_mode='values'):
    #     print(c['messages'][-1].pretty_print())
