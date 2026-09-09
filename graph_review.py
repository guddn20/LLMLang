#build_seq_chain(라우터)은 LCEL만으로 분기했지만
#여기서는 LangGraph의 조건부 엣지(add_conditional_edges)로 "그래프 자체"를 분기시킨다.
#순서 : 리뷰 파싱 -> 상태(state) 정의 -> 노드 2개(감사/보완) -> 분기 함수 -> 그래프

import os
import pandas as pd

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from templates import *
from typing import TypedDict

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

def build_graph():
    graph = StateGraph(State)
    
    graph.add_node('thanks', thanks_node)
    graph.add_node('sorry', sorry_node)
    
    graph.add_conditional_edges(START, route_by_sentiment,
                                {'thanks':'thanks',
                                 'sorry':'sorry'})
    
    graph.add_edge('thanks', END)
    graph.add_edge('sorry', END)