import os
import torch
import json
import argparse
import random
import openai
from time import sleep
from configs.model_config import *
from utils import compute_exact, compute_f1, create_logger
from datetime import datetime
from my_api_secrets import get_api_key

openai.api_key = get_api_key()

parser = argparse.ArgumentParser()
# data number
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=-1, type=int)
# data type
parser.add_argument("--update", default=1, type=int)  # 对照实验记得改这里！！！
parser.add_argument("--counterfactual", default=0, type=int)
parser.add_argument("--disorder", default=0, type=int)
parser.add_argument("--all_update", default=None, type=int)
# select base LLM
parser.add_argument("--llm_model", default="gpt-3.5-turbo", type=str)  # 选择 base LLM
args = parser.parse_args()

path_str = "Openai"
# vars() 函数返回对象object的属性和属性值的字典对象。
for arg in vars(args):
    if getattr(args, arg) == 1:
        path_str = path_str + "_" + arg
    print(arg, ':', getattr(args, arg))  # getattr() 函数是获取args中arg的属性值

if args.all_update is None:
    if args.update == 1 and args.counterfactual == 1:
        args.all_update = 1
    else:
        args.all_update = 0

def get_chatgpt_answer(input_prompt):
    got_result = False
    response = ""
    while not got_result:
        try:
            # ChatCompletion
            if args.llm_model == "gpt-3.5-turbo":
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,  # 0.0 to 2.0 (default 1.0)
                    top_p=1,  # 0.0 to 1.0 (default 1.0) (not used if temperature is set)
                    n=1,  # number (default 1) How many chat completion choices to generate for each input message.
                    stream=False,  # boolean (default False)
                    stop=["\n\n"],  # string or array (default None)
                    # 我们使用stop字段来控制生成的文本长度和格式。我们指定了两个停止标记，即换行符和"Here are some recommendations:"，
                    # 当模型生成文本中出现这些标记时，它将停止生成并返回生成的文本。这样，我们可以确保返回的文本不会太长，并按预期格式进行格式化。
                    max_tokens=25,  # inf (default 4096-prompt_token)
                    presence_penalty=0,  # -2.0 to 2.0 (default 0)
                    frequency_penalty=1,  # -2.0 to 2.0 (default 0)
                    messages=input_prompt
                    # messages=[
                    #     {"role": "system", "content": counterfactual_instruction},
                    #     {"role": "user", "content": input_sample},
                    # ]
                )
                response = completion.choices[0].message.content
                response = response.split("\n")[0]
            # Completion
            else:
                completion = openai.Completion.create(
                    engine=args.llm_model,
                    prompt=input_prompt,  # completion llms,
                    max_tokens=50,
                    temperature=0,
                    logprobs=1,
                    stop=["\n\n"]
                )
                response = completion['choices'][0]['text']
            # api访问失败，循环请求
            got_result = True
        except Exception as e:
            sleep(3)
            print('sleep 5 !  错误类型是', e.__class__.__name__)
            print('错误明细是', e)

    return response



instruction = "Instruction: Answer the question based on the context, the answer is the span in the context. If it's impossible to answer, output 'unanswerable':\n"
scope_instruction = "Instruction: get answers for the question based on the context. " \
        "If the time interval of when the event mentioned in the question occurred in the context, " \
        "the answer should be derived from substrings in the context. Else output 'unanswerable':\n"

scope_instruction = "Instruction: Answer the question based on the context.  " \
        "If the time interval of when the event mentioned in the question occurred in the context: " \
        "the answer is the span in the context. Else: output 'unanswerable':\n"

instruction = scope_instruction  # 覆盖
print(instruction)

# load
llm_model = args.llm_model

path_str = llm_model + "-" + path_str + ".log"
logger = create_logger("In Context Learning ", log_file=os.path.join("../docs/", path_str), silent=True)
run_start = datetime.now()  # 开始时间

print('loading datasets')
with open('../datasets/test_qq_1000_05.02final.json') as f:
    timeQA_test = json.load(f)
timeQA_test = timeQA_test[args.start:args.end]
print('number of examples: ', len(timeQA_test))


narrow_f1, narrow_em = 0, 0
expand_f1, expand_em = 0, 0
granularity_f1, granularity_em = 0, 0
# counterfactual_f1, counterfactual_em = 0, 0
# order_f1, order_em = 0, 0
answerable_f1, answerable_em = 0, 0
unanswerable_f1, unanswerable_em = 0, 0
sum_f1, sum_em = 0, 0
type_count = 0


for ids, example in enumerate(timeQA_test, start=1):
    sample_type = example["type"]
    if args.all_update:
        query = example["updated_question"]
        gold_answer = example["updated_answer"]
        # sample_type = example["type"]
    elif args.update:
        query = example["updated_question"] if example["type"] in ["narrow", "expand", "granularity"] else example["question"]
        gold_answer = example["updated_answer"] if example["type"] in ["narrow", "expand", "granularity"] else example["answer"]
    elif args.counterfactual:
        query = example["updated_question"] if example["type"] == "counterfactual" else example["question"]
        gold_answer = example["updated_answer"] if example["type"] == "counterfactual" else example["answer"]
    else:
        query = example["question"]
        gold_answer = example["answer"]

    # context
    if args.disorder:
        # context = []
        # for text in example["context"]:
        #     sentence_list = text["text"].split('.')
        #     random.shuffle(sentence_list)  # 是对原list做修改, 打乱context的text间的顺序
        #     context.append(".".join(sentence_list))
        context = [text["updated_text"] if text["updated_text"] != "" else text["text"] for text in example["context"]]
        random.shuffle(context)
    else:
        context = [text["text"] for text in example["context"]]

    # constructing input
    context = "\n".join(context)


    # if sample_type == "counterfactual":
    #     # counter setting
    #     main_query, counterfactuals = query.split(", if ")[0], query.split(", if ")[1]
    #     counterfactuals = "if " + counterfactuals
    #
    #     counterfactual_prompt = f"Bob read an article as follows:\n{context}. Bob imagine counterfactuals that \"{counterfactuals}\"\nQ:\"{main_query}\" in Bob's imagination."
    #     # messages = [{'role': 'system', 'content': "In Bob's imagination, provide an answer to the question, the answer must be a Noun phrase mentioned in the article, or 'unanswerable'."}]
    #     messages = [{'role': 'system', 'content': "In Bob's imagination, provide an answer to the question, the answer must be a Noun phrase mentioned in the article, or 'unanswerable'."}]
    #
    #     messages.append({'role': 'user', 'content': counterfactual_prompt})
    #     print(messages)
    #     response = get_chatgpt_answer(messages)
    #     print(response)
    # scope prompt
    input_sample = f'Context: {context}\nQuestion: {query}\nAnswer:'
    # feed to LLMs and  Generate
    # response = "Output_None"
    # prompt = instruction + input_sample
    message = [{'role': 'system', 'content': instruction},
               {"role": "user", "content": input_sample}]
    # print("输入样例: ", input_sample)
    response = get_chatgpt_answer(message)
    print('模型response答案:', response)


    # got_result = False
    # while not got_result:
    #     try:
    #         # ChatCompletion
    #         if args.llm_model == "gpt-3.5-turbo":
    #             completion = openai.ChatCompletion.create(
    #                 model="gpt-3.5-turbo",
    #                 temperature=0.0,  # 0.0 to 2.0 (default 1.0)
    #                 top_p=1,  # 0.0 to 1.0 (default 1.0) (not used if temperature is set)
    #                 n=1,  # number (default 1) How many chat completion choices to generate for each input message.
    #                 stream=False,  # boolean (default False)
    #                 stop=["\n\n"],  # string or array (default None)
    #                 # 我们使用stop字段来控制生成的文本长度和格式。我们指定了两个停止标记，即换行符和"Here are some recommendations:"，
    #                 # 当模型生成文本中出现这些标记时，它将停止生成并返回生成的文本。这样，我们可以确保返回的文本不会太长，并按预期格式进行格式化。
    #                 max_tokens=25,  # inf (default 4096-prompt_token)
    #                 presence_penalty=0,  # -2.0 to 2.0 (default 0)
    #                 frequency_penalty=1,  # -2.0 to 2.0 (default 0)
    #                 messages=[
    #                     {"role": "system", "content": instruction},
    #                     {"role": "user", "content": input_sample},
    #                 ]
    #             )
    #             response = completion.choices[0].message.content
    #         # Completion
    #         else:
    #             completion = openai.Completion.create(
    #                 engine=args.llm_model,
    #                 prompt=prompt,
    #                 max_tokens=50,
    #                 temperature=0,
    #                 logprobs=1,
    #                 stop=["\n\n"]
    #             )
    #             response = completion['choices'][0]['text']
    #         # api访问失败，循环请求
    #         got_result = True
    #
    #     except Exception as e:
    #         sleep(3)
    #         print('sleep 5 !  错误类型是', e.__class__.__name__)
    #         print('错误明细是', e)


    # evaluation
    pred_answer = response.split("\n")[0]
    f1 = compute_f1(pred_answer, gold_answer)
    em = compute_exact(pred_answer, gold_answer)

    # save result
    logger.info("第{}个问题: {}\r\n".format(ids, query))
    if args.all_update:
        logger.info("类型是: {}\r\n".format(example["type"]))
    logger.info("答案是: {}\r\n".format(gold_answer))
    logger.info("预测答案是: {}\r\n".format(pred_answer))
    logger.info("Sample F1: {} , EM: {} \r\n".format(f1, em))

    print("第{}个问题: {}".format(ids, query))
    print("答案是: {}".format(gold_answer))
    print("预测答案是: {}".format(pred_answer))
    print("ID: {} Sample F1: {} , EM: {} \n".format(ids, f1, em))
    # Compute different answer type performance
    if gold_answer == "unanswerable":
        unanswerable_f1 += f1
        unanswerable_em += em
        type_count += 1
    else:
        answerable_f1 += f1
        answerable_em += em

    if sample_type == "expand":
        expand_f1 += f1
        expand_em += em
    if sample_type == "narrow":
        narrow_f1 += f1
        narrow_em += em
    if sample_type == "granularity":
        granularity_f1 += f1
        granularity_em += em
    # Total performance
    sum_f1 += f1
    sum_em += em


logger.info("Scope F1=== expand_f1: {}, narrow_f1: {}, granularity_f1: {} ; "
            "Scope EM=== expand_em: {}, narrow_em:{}, granularity_em: {} \r\n"
            .format(100.0 * expand_f1/119, 100.0 * narrow_f1/161, 100.0 * granularity_f1/170,
                    100.0 * expand_em/119, 100.0 * narrow_em/161, 100.0 * granularity_em/170))
logger.info("answerable_f1: {} , answerable_em: {}\r\n".format(100.0 * answerable_f1 / (len(timeQA_test)-type_count), 100.0 * answerable_em / (len(timeQA_test)-type_count)))
print("answerable_f1: {} , answerable_em: {}\r\n".format(100.0 * answerable_f1 / (len(timeQA_test)-type_count), 100.0 * answerable_em / (len(timeQA_test)-type_count)))
if type_count:
    logger.info("unanswerable_f1 : {} , unanswerable_em : {}\r\n".format(100.0 * unanswerable_f1 / type_count, 100.0 * unanswerable_em / type_count))
    print("unanswerable_f1 : {} , unanswerable_em : {}\r\n".format(100.0 * unanswerable_f1 / type_count, 100.0 * unanswerable_em / type_count))
run_end = datetime.now()
print("运行时间====================: " + str((run_end - run_start).seconds / 60) + " minutes")
logger.info("Global F1: {} , Global EM: {} \r\n".format(100.0 * sum_f1 / len(timeQA_test), 100.0 * sum_em / len(timeQA_test)))
print("Global F1: {} , Global EM: {} \n".format(100.0 * sum_f1 / len(timeQA_test), 100.0 * sum_em / len(timeQA_test)))










