STYLE_TEMPLATE = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""


REVIEW_TEMPLATE = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

PROMPT_INFOS = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": "You are a very smart physics professor. You are great at answering "
        "questions about physics in a concise and easy to understand manner. "
        "When you don't know the answer to a question you admit that you don't know."
        "\n\nHere is a question:\n{input}",
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": "You are a very good mathematician. You are great at answering math "
        "questions. You break down hard problems into their component parts, "
        "answer the component parts, and then put them together."
        "\n\nHere is a question:\n{input}",
    },
    {
        "name": "History",
        "description": "Good for answering history questions",
        "prompt_template": "You are a very good historian. You have an excellent knowledge of "
        "people, events and contexts from a range of historical periods."
        "\n\nHere is a question:\n{input}",
    },
    {
        "name": "computer science",
        "description": "Good for answering computer science questions",
        "prompt_template": "You are a successful computer scientist. You are great at answering "
        "coding questions by describing the solution in imperative steps."
        "\n\nHere is a question:\n{input}",
    },
]

# {{{{ }}}} 4겹인 이유: .format() 한 번 + ChatPromptTemplate 한 번, 두 번 벗겨짐
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ "DEFAULT" or name of the prompt to use in {destinations}
    "next_inputs": string \\ a potentially modified version of the original input
}}}}
```

REMEMBER: The value of "destination" MUST match one of \
the candidate prompts listed below.\
If "destination" does not fit any of the specified prompts, set it to "DEFAULT."
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

REPLY_TEMPLATE = """\
다음은 가게에 대한 손님의 리뷰입니다. 이 리뷰는 {sentiment} 리뷰입니다.
사장님을 대신해서, 정중하고 친근한 어투로 리뷰에 대한 답글을 2~3문장으로 작성해주세요.
부정 리뷰라면 죄송한 마음을 담아 개선 의지를 보여주고,
긍정 리뷰라면 감사한 마음을 진심으로 표현해주세요.

리뷰: {comment}
"""

THANKS_TEMPLATE = """\
다음은 가게에 대한 손님의 긍정적인 리뷰입니다.
사장님을 대신해서, 진심을 담아 감사 인사를 2~3문장으로 작성해주세요.

리뷰: {comment}
"""

IMPROVE_TEMPLATE = """\
다음은 가게에 대한 손님의 부정적인 리뷰입니다.
사장님을 대신해서, 죄송한 마음과 함께 어떤 점을 개선하겠다는 의지를 담아
2~3문장으로 답글을 작성해주세요.

리뷰: {comment}
"""
