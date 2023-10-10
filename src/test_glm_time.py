import transformers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import sys
sys.path.append("..")
print(sys.path)
import torch
import json
import argparse
import random
import pandas as pd
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer, AutoModel, GenerationConfig
from configs.model_config import *
from utils import compute_exact, compute_f1, create_logger
from datetime import datetime



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
parser.add_argument("--llm_model", default="chatglm-6b", type=str)  # 选择 base LLM
args = parser.parse_args()

path_str = "result"
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


class ChatGLM:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load_model(self, model_name_or_path: str = "THUDM/chatglm-6b", llm_device=LLM_DEVICE):

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        # model
        if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
            self.model = (
                AutoModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True
                ).half().cuda()  #
            )
        else:
            self.model = (
                AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).float().to(llm_device)
            )

        self.model = self.model.eval()
        return self.tokenizer, self.model


# load
llm_model = args.llm_model
llm_device = LLM_DEVICE

path_str = llm_model + "-" + path_str + ".log"
logger = create_logger("In Context Learning ", log_file=os.path.join("../docs/", path_str), silent=True)
run_start = datetime.now()  # 开始时间

print("使用的device： ", llm_device)
device0 = torch.device("cuda:0")
model_name_or_path = llm_model_dict[llm_model]
chat_glm = ChatGLM()
print('loading model')
tokenizer, model = chat_glm.load_model(model_name_or_path=llm_model_dict[llm_model], llm_device=llm_device)



print('loading datasets')
with open('../datasets/test_qq_1000_05.02.json') as f:
    timeQA_test = json.load(f)
timeQA_test = timeQA_test[args.start:args.end]
print('number of examples: ', len(timeQA_test))

#  or categorized as 'unanswerable'
instruction = [("Instruction: ", "Answer the question based on the context, with answers derived from substrings in the context or categorized as 'unanswerable':\n")]
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
    # type
    sample_type = example["type"]
    # question
    if args.all_update:
        query = example["updated_question"]
        gold_answer = example["updated_answer"]
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
    input_sample = f'Context: {context}\nQuestion: {query}'
    # feed to LLMs
    response, history_ = model.chat(tokenizer, input_sample, history=instruction)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # evaluation

    pred_answer = response.split("\n")[0]
    f1 = compute_f1(pred_answer, gold_answer)
    em = compute_exact(pred_answer, gold_answer)

    print("第{}个问题: {}".format(ids, query))
    print("答案是: {}".format(gold_answer))
    print("预测答案是: {}".format(pred_answer))
    print("ID: {} Sample F1: {} , EM: {} \n".format(ids, f1, em))

    # save result
    logger.info("第{}个问题: {}\r\n".format(ids, query))
    logger.info("答案是: {}\r\n".format(gold_answer))
    logger.info("预测答案是: {}\r\n".format(pred_answer))
    logger.info("Sample F1: {} , EM: {} \r\n".format(f1, em))
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

    # print("ID: {} Sample F1: {} , EM: {} \n".format(ids, f1, em))
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








