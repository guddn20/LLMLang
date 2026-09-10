# build_seq_chain(라우터)은 LCEL만으로 분기했지만
# 여기서는 LangGraph의 조건부 엣지(add_conditional_edges)로 "그래프 자체"를 분기시킨다.
# 순서 : 리뷰 파싱 -> 상태(state) 정의 -> 노드 2개(감사/보완) -> 분기 함수 -> 그래프

import os
import pandas as pd

# 랭그래프 핵심요소 -> 상태, 노드, 엣지
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from templates import *
from typing import TypedDict

import main as m
from langchain_openai import ChatOpenAI

class State(TypedDict):
    comment : str
    label : str
    reply : str

def thanks_node(state, chat):
    chain = ChatPromptTemplate.from_template(THANKS_TEMPLATE) | chat | StrOutputParser()
    reply = chain.invoke({'comment':state['comment']}) #chain.invoke의 결과 -> gpt의 답변
    return {'reply':reply}

def sorry_node(state, chat):
    chain = ChatPromptTemplate.from_template(IMPROVE_TEMPLATE) | chat | StrOutputParser()
    reply = chain.invoke({'comment':state['comment']})
    return {'reply':reply}

def route_by_sentiment(state):
    if state['label'] == '1':
        return 'thanks'
    else:
        return 'sorry'

def build_graph(chat):
    graph = StateGraph(State)

    # thanks_node, sorry_node 모두 매개변수 필요
    # lambda : 함수를 축약하는 방법
    # def 함수이름():
    #   return x -> lambda x : x+2
    graph.add_node('thanks', lambda state : thanks_node(state, chat))
    graph.add_node("sorry", lambda state : sorry_node(state, chat))

    graph.add_conditional_edges(START, route_by_sentiment,
                                {'thanks':'thanks',
                                 'sorry':'sorry'})

    graph.add_edge('thanks', END)
    graph.add_edge('sorry', END)
    return graph.compile()


if __name__ == '__main__':
    #1. 리뷰 읽어오기
    df = m.load_reviews('./tarr_train.txt')
    chat = ChatOpenAI(temperature=1, model='gpt-4o')
    #print(df)
    graph = build_graph(chat)
    
    #2. 한 줄 한 줄 떼기
    for i in range(len(df)):
        comment = df.loc[i]['comment']
        label = df.loc[i]['label']
    #3. 그래프로 흘려보내기
    #result에는 그래프의 마지막 노드에서의 '상태'가 담겨있음
        result = graph.invoke({'label':str(label), 'comment':comment})    
        print(f'[{i}번째 댓글에 대한 답글] -> {result['label']} : {result['reply']}')
