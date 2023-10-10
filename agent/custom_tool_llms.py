import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,9"
os.environ["OPENAI_API_KEY"] = "sk-d9oXCh"
os.environ['HTTP_PROXY'] = "http://"
os.environ['HTTPS_PROXY'] = "http://"
import re
import argparse
import json
import openai
import calendar
import random
from fan.utils import compute_exact, compute_f1, create_logger, moth_name_list, month_abbr_list
from datetime import datetime
from time import sleep
from langchain.agents.tools import Tool
from langchain.agents import AgentType, initialize_agent, load_tools
from models.chatglm_llm import ChatGLM
from models.llama_llm import Llama  # custom LLM
from configs.model_config import *  # 配置
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import infer_auto_device_map, init_empty_weights
from configs.model_config import *


parser = argparse.ArgumentParser()
# data number
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=-1, type=int)
# data type
parser.add_argument("--update", default=1, type=int)  # 对照实验记得改这里！！！
parser.add_argument("--counterfactual", default=1, type=int)
parser.add_argument("--disorder", default=1, type=int)
parser.add_argument("--all_update", default=None, type=int)
# select base LLM
parser.add_argument("--llm_model", default="llama-13b-hf", type=str)  # 选择 base LLM
args = parser.parse_args()

path_str = "tools_test"
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


# 搜索工具
class SearchTool(BaseTool):
    name = "Search"
    description = "如果我想知道天气，'鸡你太美'这两个问题时，请使用它"

    # 直接返回结果
    # return_direct = True

    def _run(self, query: str) -> str:
        print("\nSearchTool query: " + query)
        return "这个是一个通用的返回"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")


# 计算工具
class CalculatorTool(BaseTool):
    name = "Calculator"
    description = "如果是关于数学计算的问题，请使用它"

    def _run(self, query: str) -> str:
        print("\nCalculatorTool query: " + query)
        return "3"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂时不支持异步")


# time比较工具
class CompareTime(BaseTool):
    name = "Comparison Time Tool"
    # llama
    description = (
        "Useful for when you need to extract two time scopes from a question and its following context, "
        "Input should be a valid python command, the format such as:\n"
        "question_time=['2003-10', '2006']\n"
        "context_time=['2004-9', '2005']\n"
        # "and ['2003-10', '2006'] is the question time scope and ['2004-9', '2005'] is the context time scope"
        "representing the time interval in the question being asked and "
        "the time interval of when the event mentioned in the question occurred in the context, respectively."
    )
    # "Useful when you want to compare two sets of time scopes."
    # description = (
    #     "Useful for when you need to get time scopes. "  # llama
    #     # "Useful for when you need to answer a question about time. "
    #     # "The input of the tool should be two lists of time intervals, separated by a comma,"
    #     "The input of the tool should be two sets of time intervals, "
    #     "representing the time interval in the question and one time interval of when the event mentioned in the question occurred in the context, respectively."
    #     "The time interval may consist of a single time point, or it may consist of two time points representing the start and end of the interval."
    #     "format like: ['2004-09'], ['2003-10','2006']"
    # )

    # rather than ['September 2004'], ['October 2003', '2006']
    # print("打印tool description: ", description)

    return_direct = True  # 直接返回结果

    def _run(self, input: str) -> str:
        """extract time scopes from context and question, and compare two sets of time scopes"""
        flag = False
        print("Input type: ", type(input))
        # 处理query 一个和context 一个时间节点
        try:
            input = eval("[" + input + "]")  # [['June 1995', 'June 1997'], ['January 1995', 'June 1997']]
            # function:  month str ==> number str , such as: September ==> -09
            for j in range(2):  # 前两个 q和c
                for i in range(len(input[j])):
                    str_month_year = input[j][i].split(" ")
                    if str_month_year[0] in moth_name_list:
                        input[j][i] = str_month_year[1] + "-" + str(list(calendar.month_name).index(str_month_year[0]))
                    elif str_month_year[0] in month_abbr_list:
                        input[j][i] = str_month_year[1] + "-" + str(list(calendar.month_abbr).index(str_month_year[0]))
                    else:
                        continue

            query_time = input[0]
            context_time = input[1]
            query_time_len = len(query_time)
            context_time_len = len(context_time)
            print("query time: ", input[0])
            print("context time: ", input[1])
            # 正则话字符月份为数字月份 ↑↑↑↑↑↑↑↑↑↑
            if query_time_len == 2 and context_time_len == 2:
                if query_time[0] >= context_time[0] and query_time[1] <= context_time[1]:
                    flag = True
            # elif query_time_len == 1 and context_time_len == 2:
            #     if context_time[0] <= query_time[0] <= context_time[1]:
            #         flag = True
            # elif query_time_len == 2 and context_time_len == 1:  # 这个类型是数据本身的问题，不合理数据
            #     if not (query_time[0] <= context_time[0] <= query_time[1]):  # context在query时间区间内的否
            #         flag = True
            else:
                print("query和context时间区间不完整(<=1)，无法回答：")

            # print("成功执行Python time comparison tool !")

        except Exception as e:
            print('错误类型是', e.__class__.__name__)
            print('异常抛出错误明细是', e)
            flag = False

        if flag == True:
            # output = "The time scope mentioned in the question belongs to the time interval of real events in the context, " \
            #          "so it can be answered using information from the context."
            # output = "The time in the question is equivalent to this time: [" + " ,".join(input[1]) + "]"  # 防止context[1]越界
            output = input[1]
        else:
            output = "unanswerable"
            self.return_direct = True
        return output

    async def _arun(self, input: str) -> str:
        raise NotImplementedError("暂时不支持异步")


class PythonCompareTime(BaseTool):
    name = "Comparison Time Tool"
    # llama
    description = (
        "Useful for when you need to extract two time scopes from a question and its following context, "
        "Input should be a valid python command, the format such as:\n"
        "question_time = ['2003-10', '2006'] \n"
        "context_time = ['2004-9', '2005'] \n"
        # "and ['2003-10', '2006'] is the question time scope and ['2004-9', '2005'] is the context time scope"
        "Representing the time interval in the question being asked and "
        "the time interval of when the event mentioned in the question occurred in the context, respectively."
    )


    return_direct = True  # 直接返回结果

    def _run(self, input: str) -> str:
        """extract time scopes from context and question, and compare two sets of time scopes"""
        flag = False
        print("Input type: ", type(input), input)
        query_time = []
        context_time = []
        # 处理query 一个和context 一个时间节点
        try:
            # 在一个命名空间中执行代码
            namespace = {}
            exec(input, namespace)
            # 提取变量
            query_time = namespace['question_time']
            context_time = namespace['context_time']

            # function:  month str ==> number str , such as: September ==> -09
            for i in range(len(query_time)):
                str_month_year = query_time[i].split(" ")
                if str_month_year[0] in moth_name_list:
                    query_time[i] = str_month_year[1] + "-" + str(list(calendar.month_name).index(str_month_year[0]))
                elif str_month_year[0] in month_abbr_list:
                    query_time[i] = str_month_year[1] + "-" + str(list(calendar.month_abbr).index(str_month_year[0]))
                else:
                    continue

            for i in range(len(context_time)):
                str_month_year = context_time[i].split(" ")
                if str_month_year[0] in moth_name_list:
                    context_time[i] = str_month_year[1] + "-" + str(list(calendar.month_name).index(str_month_year[0]))
                elif str_month_year[0] in month_abbr_list:
                    context_time[i] = str_month_year[1] + "-" + str(list(calendar.month_abbr).index(str_month_year[0]))
                else:
                    continue

            # query_time = input[0]
            # context_time = input[1]
            query_time_len = len(query_time)
            context_time_len = len(context_time)
            print("query time: ", query_time)
            print("context time: ", context_time)
            # 正则话字符月份为数字月份 ↑↑↑↑↑↑↑↑↑↑
            if query_time_len == 2 and context_time_len == 2:
                if query_time[0] >= context_time[0] and query_time[1] <= context_time[1]:
                    flag = True
            # elif query_time_len == 1 and context_time_len == 2:
            #     if context_time[0] <= query_time[0] <= context_time[1]:
            #         flag = True
            # elif query_time_len == 2 and context_time_len == 1:  # 这个类型是数据本身的问题，不合理数据
            #     if not (query_time[0] <= context_time[0] <= query_time[1]):  # context在query时间区间内的否
            #         flag = True
            else:
                print("query和context时间区间不完整(<=1)，无法回答：")

            # print("成功执行Python time comparison tool !")

        except Exception as e:
            print('错误类型是', e.__class__.__name__)
            print('异常抛出错误明细是', e)
            flag = False

        if flag == True:
            # output = "The time scope mentioned in the question belongs to the time interval of real events in the context, " \
            #          "so it can be answered using information from the context."
            # output = "The time in the question is equivalent to this time: [" + " ,".join(input[1]) + "]"  # 防止context[1]越界
            output = context_time
        else:
            output = "unanswerable"
            self.return_direct = True
        return output

    async def _arun(self, input: str) -> str:
        raise NotImplementedError("暂时不支持异步")

# Equivalent transform 工具
class QueryTransform(BaseTool):
    name = "Query Transform Tool"
    description = (
        # "Useful for when you need to use an equivalent question to arrive at an answer, "
        "Useful for when you need to get an equivalent question,"
        "The input of the tool should be a original question and a time scope, "
        "Like: ['Dean Holdsworth played for which team from 1986-08 to 1989?', ['2003-10','2006']], "
        "Return an equivalent question."  # , and then continue to answer based on the equivalent question.
    )
    # print("打印tool description: ", description)

    return_direct = False  # 直接返回结果

    def _run(self, input: str) -> str:
        """Transform the original question into an equivalent new question, and answer based on the new question."""
        # print("\nMy Tool Input: ", input)
        # print("Input type: ", type(input))
        try:
            input = eval(input)
            question = input[0]
            time_list = input[1]
            print("query : ", input[0])
            print("time scope: ", input[1])

            time_match = re.findall(r"([0-9]{4})(-\d{2})?", question)
            print("time_match", time_match)
            query_time_list = ["".join(t) for t in time_match]
            print("query_time_list: ", query_time_list)
            query_time_list_len = len(query_time_list)
            time_list_len = len(time_list)
            if time_list_len == 2 and query_time_list_len == 2:
                # 将时间点按顺序用列表中的时间替换
                new_question = question.replace(query_time_list[0], time_list[0]).replace(query_time_list[1], time_list[1])
                print("成功执行Query Transform Tool !")
            else:
                # 能执行等价工具的话，说明以满足time comparison tool的时间区间要求
                # 需要正则处理
                print("query和context存在非区间时间，不做处理！输出原始问题")
                self.return_direct = True  # 执行等价替换就结束Agent
                return question
            self.return_direct = True  # 执行等价替换就结束Agent
            return new_question

        except Exception as e:
            print('错误类型是', e.__class__.__name__)
            print('异常抛出错误明细是', e)
            try:
                input = eval(input)
                question = input[0]
            except Exception as e1:
                print('内部子异常抛出错误明细是', e1)
                question = ""
            self.return_direct = True  # 执行等价替换就结束Agent
            return question  # 返回原始问题

    async def _arun(self, input: str) -> str:
        raise NotImplementedError("暂时不支持异步")


# Counterfactual hypothesis 工具
class Hypothesis_Imagine(BaseTool):
    name = "Hypothesis Tool"
    description = (
        "Useful for when you need to answer a question that involves a hypothetical condition. "
        "The input of the tool should be a question."
    )
    # print("打印tool description: ", description)

    return_direct = True  # 直接返回结果

    def _run(self, input: str) -> str:
        """Transform the original question into an equivalent new question, and answer based on the new question."""
        # print("\nMy Tool Input: ", input)
        # print("Input type: ", type(input))
        try:
            input = eval(input)
            question = input[0]
            time_list = input[1]
            print("query : ", input[0])
            print("time scope: ", input[1])

            time_match = re.findall(r"([0-9]{4})(-\d{2})?", question)
            print("time_match", time_match)
            query_time_list = ["".join(t) for t in time_match]
            print("query_time_list: ", query_time_list)
            query_time_list_len = len(query_time_list)
            time_list_len = len(time_list)
            if time_list_len == 2 and query_time_list_len == 2:
                # 将时间点按顺序用列表中的时间替换
                new_question = question.replace(query_time_list[0], time_list[0]).replace(query_time_list[1], time_list[1])
                print("成功执行Query Transform Tool !")
            else:
                # 能执行等价工具的话，说明以满足time comparison tool的时间区间要求
                # 需要正则处理
                print("query和context存在非区间时间，不做处理！输出原始问题")
                self.return_direct = True  # 执行等价替换就结束Agent
                return question
            self.return_direct = True  # 执行等价替换就结束Agent
            return new_question

        except Exception as e:
            print('错误类型是', e.__class__.__name__)
            print('异常抛出错误明细是', e)
            try:
                input = eval(input)
                question = input[0]
            except Exception as e1:
                print('内部子异常抛出错误明细是', e1)
                question = ""
            self.return_direct = True  # 执行等价替换就结束Agent
            return question  # 返回原始问题

    async def _arun(self, input: str) -> str:
        raise NotImplementedError("暂时不支持异步")


def get_chatgpt_answer(input_prompt):
    got_result = False
    response = ""
    while not got_result:
        try:
            # ChatCompletion
            if args.llm_model != "gpt-3.5-turbo":
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,  # 0.0 to 2.0 (default 1.0)
                    top_p=1,  # 0.0 to 1.0 (default 1.0) (not used if temperature is set)
                    n=1,  # number (default 1) How many chat completion choices to generate for each input message.
                    stream=False,  # boolean (default False)
                    stop=["\n\n"],  # string or array (default None)
                    # 我们使用stop字段来控制生成的文本长度和格式。我们指定了两个停止标记，即换行符和"Here are some recommendations:"，
                    # 当模型生成文本中出现这些标记时，它将停止生成并返回生成的文本。这样，我们可以确保返回的文本不会太长，并按预期格式进行格式化。
                    max_tokens=3096,  # inf (default 4096-prompt_token)
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


instruction = "Instruction: get answers for the question based on the context, where answers derived from substrings in the context or categorized as 'unanswerable':\n"



# load dataset
llm_model = args.llm_model
path_str = llm_model + "-" + path_str + ".log"
logger = create_logger("In Context Learning ", log_file=os.path.join("../docs/", path_str), silent=True)
run_start = datetime.now()  # 开始时间

print('loading datasets')
with open('../datasets/test_qq_1000_05.02.json') as f:
    timeQA_test = json.load(f)
timeQA_test = timeQA_test[args.start:args.end]
print('number of examples: ', len(timeQA_test))

narrow_f1, narrow_em = 0, 0
expand_f1, expand_em = 0, 0
granularity_f1, granularity_em = 0, 0

answerable_f1, answerable_em = 0, 0
unanswerable_f1, unanswerable_em = 0, 0
sum_f1, sum_em = 0, 0
type_count = 0
# load model
llm = Llama()
tokenizer, model = llm.load_model(model_name_or_path=llm_model_dict[llm_model], llm_device=LLM_DEVICE)

for ids, example in enumerate(timeQA_test, start=1):
    sample_type = "original"  # order and original
    if args.all_update:
        query = example["updated_question"]
        gold_answer = example["updated_answer"]
        sample_type = example["type"]
    elif args.update:
        query = example["updated_question"] if example["type"] in ["narrow", "expand", "granularity"] else example["question"]
        gold_answer = example["updated_answer"] if example["type"] in ["narrow", "expand", "granularity"] else example["answer"]
        sample_type = example["type"] if example["type"] in ["narrow", "expand", "granularity"] else "original"
    elif args.counterfactual:
        query = example["updated_question"] if example["type"] == "counterfactual" else example["question"]
        gold_answer = example["updated_answer"] if example["type"] == "counterfactual" else example["answer"]
        sample_type = example["type"] if example["type"] == "counterfactual" else "original"
    else:
        query = example["question"]
        gold_answer = example["answer"]

    # context
    if args.disorder:
        context = [text["updated_text"] if text["updated_text"] != "" else text["text"] for text in example["context"]]
        random.shuffle(context)
        context = "\n".join(context)
        # order prompt

    else:
        context = [text["text"] for text in example["context"]]
        context = "\n".join(context)

    # Agent 输入
    # constructing input
    response = ""
    # scope prompt,之后删掉 ，目前是不用tool
    if sample_type == "expand" or "granularity" or "narrow":
        scope_instruction = "Instruction: get answers for the question based on the context. " \
                            "If the time interval of when the event mentioned in the question occurred in the context, " \
                            "the answer should be derived from substrings in the context. Else output 'unanswerable':\n"

        generate_kwargs = {
            "max_new_tokens": 25,
            "min_new_tokens": 1,
            "temperature": 0.1,
            "do_sample": False,  # The three options below used together leads to contrastive search
            "top_k": 1,
            "penalty_alpha": 0.6,
            # "no_repeat_ngram_size": no_repeat_ngram_size,
            # **generation_config,
        }
        input_sample = f'Context: {context}\nQuestion: {query}\nAnswer:'
        # feed to LLMs
        prompt = scope_instruction + input_sample
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_len = input_ids.shape[1]
        input_ids = input_ids.to(0)
        # 将输入tensor移动到cuda:0设备上
        # inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
        generate_ids = model.generate(input_ids, **generate_kwargs)
        response = tokenizer.batch_decode(generate_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # evaluation
        response = response.split("\n")[0]
    elif sample_type == "expand":
        input_sample = f'Context: {context}\nQuestion: {query}'
        # print("输入: ", input_sample)
        tool_prompt = input_sample  # comparison_instruction +

        # turbo_llm = ChatOpenAI(
        #     temperature=0,
        #     model_name='gpt-3.5-turbo',
        # )
        # llm = turbo_llm
        tools = [PythonCompareTime(return_direct=True)]  # , QueryTransform(return_direct=True)
        agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        try:
            response = agent_chain.run(tool_prompt)
        except:
            response = "unanswerable"
        # print("original query: ", query)
        print("comparison time tool response: ", response)
        response = str(response)

    elif sample_type == "counterfactual":
        # counter setting
        main_query, counterfactuals = query.split(", if ")[0], query.split(", if ")[1]
        counterfactuals = "if " + counterfactuals

        counterfactual_instruction = "In Bob's imagination, provide an answer to the question, " \
                                     "the answer must be a Noun phrase mentioned in the article, or 'unanswerable'."
        counterfactual_prompt = f"Bob read an article as follows:\n{context}. Bob imagine counterfactuals that \"{counterfactuals}\"\nQuestion:\"{main_query}\" in Bob's imagination.\nAnswer:"
        input_prompt = counterfactual_instruction + counterfactual_prompt
        # print("input_prompt: ", input_prompt)
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
        input_len = input_ids.shape[1]
        input_ids = input_ids.to(0)

        generate_kwargs = {
            "max_new_tokens": 25,
            "min_new_tokens": 1,
            "temperature": 0.1,
            "do_sample": False,  # The three options below used together leads to contrastive search
            "top_k": 1,
            "penalty_alpha": 0.6,
            # "no_repeat_ngram_size": no_repeat_ngram_size,
            # **generation_config,
        }
        generate_ids = model.generate(input_ids, **generate_kwargs)
        response = tokenizer.batch_decode(generate_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # print("response: ", response)
        # evaluation

        response = response.split("\n")[0]
        response = response.strip('"')

    elif True:
        llama_instruction = "Instruction: get answers for the question based on the context, " \
                            "where answers derived from substrings in the context or categorized as 'unanswerable':\n"

        generate_kwargs = {
            "max_new_tokens": 25,
            "min_new_tokens": 1,
            "temperature": 0.1,
            "do_sample": False,  # The three options below used together leads to contrastive search
            "top_k": 1,
            "penalty_alpha": 0.6,
            # "no_repeat_ngram_size": no_repeat_ngram_size,
            # **generation_config,
        }
        input_sample = f'Context: {context}\nQuestion: {query}\nAnswer:'
        # feed to LLMs
        prompt = llama_instruction + input_sample
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_len = input_ids.shape[1]
        input_ids = input_ids.to(0)
        # 将输入tensor移动到cuda:0设备上
        # inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
        generate_ids = model.generate(input_ids, **generate_kwargs)
        response = tokenizer.batch_decode(generate_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # evaluation
        response = response.split("\n")[0]
    else:
        # other type
        input_sample = f'Context: {context}\nQuestion: {query}\nAnswer:'  # New time: {response}\n
        message = [{'role': 'system', 'content': instruction},
                   {"role": "user", "content": input_sample}]
        # print("输入样例: ", input_sample)
        response = get_chatgpt_answer(message)
        # print('模型答案:', response)


    # evaluation
    pred_answer = response
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
    print("类型是: {}".format(sample_type))
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
logger.info("answerable_f1: {} , answerable_em: {}\r\n".format(100.0 * answerable_f1 / (len(timeQA_test) - type_count),
                                                               100.0 * answerable_em / (len(timeQA_test) - type_count)))
print("answerable_f1: {} , answerable_em: {}\r\n".format(100.0 * answerable_f1 / (len(timeQA_test) - type_count),
                                                         100.0 * answerable_em / (len(timeQA_test) - type_count)))
if type_count:
    logger.info("unanswerable_f1 : {} , unanswerable_em : {}\r\n".format(100.0 * unanswerable_f1 / type_count,
                                                                         100.0 * unanswerable_em / type_count))
    print("unanswerable_f1 : {} , unanswerable_em : {}\r\n".format(100.0 * unanswerable_f1 / type_count,
                                                                   100.0 * unanswerable_em / type_count))
run_end = datetime.now()
print("运行时间====================: " + str((run_end - run_start).seconds / 60) + " minutes")
logger.info("Global F1: {} , Global EM: {} \r\n".format(100.0 * sum_f1 / len(timeQA_test), 100.0 * sum_em / len(timeQA_test)))
print("Global F1: {} , Global EM: {} \n".format(100.0 * sum_f1 / len(timeQA_test), 100.0 * sum_em / len(timeQA_test)))

