import os
import argparse
import openai
import json
from time import sleep
from my_api_secrets import get_api_key

openai.api_key = get_api_key()

parser = argparse.ArgumentParser()
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=-1, type=int)
args = parser.parse_args()

path_easy_filter = "../datasets/timeqa/test_easy_filter_has_answer.json"
with open(path_easy_filter, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

test_data = test_data[args.start:args.end]

def create_reader_request_processed(example):
    prompt = 'Narrow down the time scope in the question, and then update the question:\n'
    question = example['question']
    prompt += f'Question: {question}\n'
    prompt += 'Updated Question: '
    return prompt


sensitive_prompt = """
Narrow down the time scope in the question, and then update the question:
Question: Which employer did Philip Alston work for from 1996 to 2001?
Updated Question: Which employer did Philip Alston work for from 1997 to 2000?


Expand the time scope of the question, and then update the question:
Question: Which employer did Philip Alston work for from 1996 to 2001?
Updated Question: Which employer did Philip Alston work for from 1994 to 2002?

Add monthly information to the time scope of the question, and then update the question:
Question: Which employer did Philip Alston work for from 1996 to 2001?
Updated Question: Which employer did Philip Alston work for from August 1996 to July 2001?
"""



for id, example in enumerate(test_data, start=1):
    input_prompt = sensitive_prompt + '\n\n'  # 记得重置,亏麻啦
    input_prompt += create_reader_request_processed(example)
    got_result = False
    api_fail_num = 0
    while not got_result:
        try:
            response = openai.Completion.create(
                engine='text-davinci-003',
                prompt=input_prompt,
                max_tokens=50,
                temperature=0,
                logprobs=1,
                stop=["\n\n"]
            )
            got_result = True
        except Exception as e:
            print('sleep 5 !  错误类型是', e.__class__.__name__)
            print('错误明细是', e)
            sleep(3)
            api_fail_num += 1
            print(api_fail_num)
            if api_fail_num >= 5:
                break

    if not got_result:
        # 计时器超时，跳过这个例子
        # 可能是超出最大长度等原因
        print("================计时器超时，跳过这个例子==================")
        # print(example["question"])
        continue

    updated_question = response['choices'][0]['text'].strip()
    example["updated_question"] = updated_question  # .split(":")[1].strip()
    example["updated_answer"] = ""
    example["type"] = ""
    print("ID: ", id)
    print("Original Question: ", example['question'])
    print(updated_question)


with open("../datasets/timeqa/test_llms.json", "w", encoding='utf-8') as f_out:
    json.dump(test_data, f_out, indent=2)

print(os.getenv('HTTP_PROXY'))

