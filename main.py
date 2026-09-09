# 주어진 tarr_train.txt를 파싱하여
# main.py를 실행했을 때 각 리뷰에 대한 답글이 자동으로 생성되도록 하기
import pandas as pd
from templates import *
from dotenv import load_dotenv

# ChatPromptTemplate -> gpt나 LLm에 질문할 내용을 정리하는 것
from langchain_core.prompts import ChatPromptTemplate

# StrOutputParser -> output으로 나온 결과물을 정제
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI

#키를 가져오는 역할 -> .env
load_dotenv()

# 1. (pandas)라이브러리를 사용해서 txt 파일을 읽어오기
def load_reviews(path):
    df = pd.read_csv(path, sep='\t')
    #print(df.columns)
    #print(df['comment'])
    return df

# 2. 오늘 한 chain 함수를 이용해서 댓글 분류(선택) / 긍정-부정
def build_reply_chain(chat):

    # from_template, from_message
    prompt = ChatPromptTemplate.from_template(REPLY_TEMPLATE)

    # return 프롬프트 | 챗 | 파서 -> str, json, structured
    return prompt | chat | StrOutputParser()

# 리뷰를 한줄 한줄 읽어서 build_reply_chain에 넣어주는 함수
def generate_reply(reviews, chat):
    reply_chain = build_reply_chain(chat)
    
    for i in range(len(reviews)):
        #템플릿에서 뽑아올 2가지 요소
        #print(review.)
        #print(review[0])
        #print(reviews['label'][i])
        
        sentiment = '긍정' if reviews['label'][i] == '1' else "부정"
        comment = reviews['comment'][i]

        result = reply_chain.invoke({"sentiment":sentiment, "comment":comment})
        print(f'[답글 생성] 손님 댓글 : {comment} \n 사장님 댓글 : {result}')

# 3. 오늘 한 chain 함수를 이용해서 댓글에 대한 답글 생성(필수) -> '페르소나' 부여 가능

if __name__ == "__main__":
    # 1. (pandas)라이브러리를 사용해서 txt 파일을 읽어오기
    df = load_reviews('./tarr_train.txt')
    
    #temperature(0~1) : 창의성
    chat = ChatOpenAI(temperature = 0.7, model = 'gpt-4o')
    
    generate_reply(df, chat)
