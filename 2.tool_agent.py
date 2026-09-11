# Agent가 포함된 랭 그래프
from dotenv import load_dotenv
load_dotenv()

# Annotated[자료형, 덮어쓰기] -> '자료형' 데이터가 여기 들어올거야. 데이터가 새로 들어올 때 '덮어쓰기' 할거야.
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

# 에이전트 구성을 위해 필요한 langchain 컴포넌드
from langchain_core.messages import (AnyMessage, SystemMessage,
                                     HumanMessage, ToolMessage)
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_community.tools import YouTubeSearchTool

# 리스트를 합쳐줌 [] [] -> [] operator.add
import operator
import wikipedia

wikipedia.set_user_agent("CoredataLectureBot/1.0 (guddn20@gmail.com)")

from langchain_core.tools import tool

import requests
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# API가 주는 weather_code(WMO 국제 기상 코드) -> 사람이 읽는 말로 바꿔주는 표
WMO_CODES = {
    0: "맑음",
    1: "대체로 맑음",
    2: "구름 조금",
    3: "흐림",
    45: "안개",
    48: "짙은 안개",
    51: "약한 이슬비",
    53: "이슬비",
    55: "강한 이슬비",
    56: "약한 어는 이슬비",
    57: "강한 어는 이슬비",
    61: "약한 비",
    63: "비",
    65: "강한 비",
    66: "약한 어는 비",
    67: "강한 어는 비",
    71: "약한 눈",
    73: "눈",
    75: "강한 눈",
    77: "싸락눈",
    80: "약한 소나기",
    81: "소나기",
    82: "강한 소나기",
    85: "약한 눈 소나기",
    86: "강한 눈 소나기",
    95: "뇌우",
    96: "우박을 동반한 뇌우",
    99: "강한 우박 뇌우",
}


class AgentState(TypedDict):
    messages : Annotated[list[AnyMessage], operator.add]

# Agent의 탈을 쓴 그래프
class Agent:
    #이름만 Agent이고 langgraph로 구현함
    #langgraph를 구현
    def __init__(self, system, tools, model):
       self.system = system #시스템에 대한 전반적인 답변 매너를 정하는 프롬프트
       
       #그래프 정의
       graph = StateGraph(AgentState)
       graph.add_node('llm', self.call_openai) #1.채팅을 한다.
       graph.add_node('tool', self.take_action) #2.혹시 채팅중 도구 필요하면 쓴다.
       
       graph.add_conditional_edges('llm', self.exist_action,
                                   {True : 'tool',
                                    False : END})
       
       graph.add_edge('tool', 'llm')
       graph.set_entry_point('llm') #START 노트에서 시작하지 않았으므로 시작점이 llm 함수임을 알림
       #END -> LLM
       self.graph = graph.compile()
       
       
       self.tools = {t.name : t for t in tools}
       self.model = model.bind_tools(tools)
       
    #조건 판단 -> 툴이 있나?
    def exist_action(self, state):
        return len(state['messages'][-1].tool_calls) > 0
    
    #실행함(execute)
    def call_openai(self, state):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages':[message]}
    
    #도구 실행
    def take_action(self, state:AgentState):
        
        tool_calls = state['messages'][-1].tool_calls        
        results = []
        # t -> 1개의 개별 도구(함수, API)
        for t in tool_calls:
            print(f"도구 호출 : {t['name']} -> {t['args']}")
            
            if t['name'] not in self.tools:
                result = '존재하지 않는 도구입니다.'
            else:
                #self.tools -> 함수, api .invoke(도구 실행에 필요한 매개변수)
                result = self.tools[t['name']].invoke(t['args'])
            results.append(
                #tool의 id, name, content(툴을 쓴 결과)
                ToolMessage(tool_call_id=t['id'], name=t['name'],
                            content=str(result))
            )
            print(f'모델로 복귀\n')
            return {'messages':results}


def fetch_weather(city, country_code='KR'):
    try :
        # 인터넷 페이지에 요청하여 날씨 결과를 받아옴
        geo = requests.get(GEO_URL, params={'name':city,
                                            'count':1,
                                            'language':'ko',
                                            'country_code':country_code}).json()

        if not geo.get('result'):
            return f'{city}의 위도 경도를 찾지 못했습니다.'

        loc = geo['result'][0]
        weather = requests.get(WEATHER_URL, params={
            'latitude':loc['latitude'],
            'longitude':loc['longitude'],
            'timezone':'auto'
        }, timeout=10).json()

        if weather.get('error'):
            return f'날씨 API 오류 : {weather.get('reason')}'

        cur = weather['current']
    # 1. 인터넷 요청 실패, 2. 내용(요청) 잘못됨, 3.받아온 값이 이상함
    except (requests.RequestException, KeyError, ValueError) as e:
        return f'날씨 API 호출 실패 : 원인 {e}'

    sky = WMO_CODES.get(
        cur["weather_code"], f"알 수 없음(code {cur['weather_code']})"
    )
    return (
        f"{loc['name']} ({cur['time']} 기준) | {sky} | 기온 {cur['temperature_2m']}°C | "
        f"습도 {cur['relative_humidity_2m']}% | 강수량 {cur['precipitation']}mm | "
        f"풍속 {cur['wind_speed_10m']}km/h"
    )


# def 함수이름(매개변수:매개변수의 자료형 = 디폴트값)
@tool
def get_weather(city:str, country_code:str='KR') -> str:
    '''현재 city의 날씨 정보를 얻어옵니다.
    city의 이름은 영어로 되어야 합니다. 예시: seoul, busan, jeju
    country_code는 영어 2글자 표준 표기를 따릅니다. 예시 : KR, JP, US'''
    return fetch_weather(city, country_code)

# 진입점
if __name__ == '__main__':
    model = ChatOpenAI(temperature=0.7, model='gpt-4o')
    system = '''
        You are a smart research assistant.
        Using Wikipedia, find out what I ask for.
    '''
    wiki = WikipediaQueryRun(api_wrapper= WikipediaAPIWrapper(top_k_results=2,
                                                               doc_content_chars_max=1000))
    
    tools = [wiki, get_weather]
    #print(f'사용 도구 : {tools.name}')
    bot = Agent(system, tools, model)
    
    question = input('질문해주세요 : \n')
    message = [HumanMessage(content=question)]
    #bot.graph -> Agent.graph
    result = bot.graph.invoke({'messages':message})
    
    print(f'{result['messages'][-1].content}')
