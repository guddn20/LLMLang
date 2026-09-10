# https://github.com/langchain-ai/react-agent
# Re-Act 패턴의 에이전틱 AI를 구성해보자.
from dotenv import load_dotenv
load_dotenv() #OpenAI의 키가 담긴 .env 형식 가져오기
from templates import *

import re
from openai import OpenAI #openai에게 직접 통신 요청
client = OpenAI()

#'세션'
class Agent:
    def __init__(self, system):
        self.system = system
        self.messages = []
        if self.system:
            # role은 system(ai를 말함), user(사용자) -> content(내용)
            self.messages.append({'role' : 'system', 'content' : system})

    # 실제 대화를 주고받는 동작
    # __함수이름__ : 매직메서드, 던더 함수
    def __call__(self, message):
        self.messages.append({'role':'user', 'content':message})
        result = self.execute()
        self.messages.append({'role':'assistant', 'content':result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model = 'gpt-4o',
            temperature = 0.5,
            messages = self.messages
        )
        return completion.choices[0].message.content

# Tool, Memory
# 도구(에이전트가 활용 가능한 함수, API)
def calculate(what):
    #eval -> 문자를 수학식으로 해석해서 계산해주는 파이썬 내장함수
    #eval('4+5') => 4+5 계산
    return eval(what)


def average_dog_weight(name):
    if "Scottish Terrier" in name:
        return "Scottish Terriers average 20 lbs"
    elif "Border Collie" in name:
        return "a Border Collie's average weight is 37 lbs"
    elif "Toy Poodle" in name:
        return "a Toy Poodle's average weight is 7 lbs"
    else:
        return "An average dog weighs 50 lbs"

known_actions = {"calculate":calculate,
                "average_dog_weight" : average_dog_weight}


#실제 실행
#re(정규표현식) -> 문자열 패턴을 추출, 체크 re.match(문자열) -> '문자열'에서 내가 찾으려는 패턴 검사
#^Action :  -> Action: 으로 시작하는 문자열을 찾을게. ( ^ means startswith )
#(\w+)      -> \w (문자) + (1개 이상)
#(.*)       -> .(모든 문자) *(0개 이상 값)
action_re = re.compile(r'^Action: (\w+): (.*)$')

# if __name__ == '__main__':
#     line = "Action: calculate: 40*33"
#     match = action_re.match(line)
#     action, action_input = match.groups()
#     print(f'액션 : {action}')
#     print(f'인풋 : {action_input}')
# (Lang) C:\Users\user\Desktop\ChatBot>python ./1.react_agent.py
# 액션 : calculate
# 인풋 : 40*33

def query(question, max_turns=5):
    i = 0
    bot = Agent(SYSTEM_PROMPT)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(line) for line in result.split('\n') 
                   if action_re.match(line)]
        
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f'unknown action : {action} - {action_input}')
            print(f'running ... {action}( {action_input} )')
            observation = known_actions[action](action_input)
            
            print(f'observation ... {observation}')
            next_prompt = f'Observation: {observation}'
        else:
            return
        
if __name__ == '__main__':
    question = 'I have 2 dogs, a border collie and a scottish terrier. What is their combined weight?'
    query(question=question)