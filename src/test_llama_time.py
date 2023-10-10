import transformers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,9"
import sys
sys.path.append("..")
# print(sys.path)
import torch
import json
import argparse
import random
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import infer_auto_device_map, init_empty_weights
from configs.model_config import *
from utils import compute_exact, compute_f1, create_logger
from datetime import datetime


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
parser.add_argument("--llm_model", default="llama-7b-hf", type=str)  # 选择 base LLM
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

# def get_device_map(model_name, device, do_int8):
#     if device == "a100-40g":
#         return "auto"
#
#     with init_empty_weights():
#         config = AutoConfig.from_pretrained(model_name)
#         model = AutoModelForCausalLM.from_config(config)
#
#     d = {0: "18GiB"}
#     for i in range(1, 6):
#         d[i] = "26GiB"
#     device_map = infer_auto_device_map(
#         model, max_memory=d, dtype=torch.int8 if do_int8 else torch.float16, no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer", "LlamaDecoderLayer"]
#     )
#     print(device_map)
#     del model
#     return device_map



class LlamaModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load_model(self, model_name_or_path: str = "decapoda-research/llama-13b-hf", llm_device=LLM_DEVICE):
        # print device map
        # config = AutoConfig.from_pretrained(model_name_or_path)
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_config(config)
        #
        # device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"])
        # print(device_map)

        # tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        # model
        if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
            hf_device_map = self.model.hf_device_map
            print(hf_device_map)
        else:
            self.model = (
                AutoModelForCausalLM.from_pretrained(model_name_or_path).float().to(llm_device)
            )
        # self.model = self.model.eval()
        return self.tokenizer, self.model


# load
llm_model = args.llm_model
llm_device = LLM_DEVICE
llm_model_dict = llama_model_dict  # custom llm dict
model_name_or_path = llm_model_dict[llm_model]

# device1 = torch.device("cuda:1")
max_memory_mapping = {0: "5GB", 1: "20GB", 2: "20GB", 3: "20GB"}  # 使用max_memory参数控制你想在每个GPU上分配的GPU RAM。
print("使用的device：{} 可用的GPU数量：{}".format(llm_device, torch.cuda.device_count()))

path_str = llm_model + "-" + path_str + ".log"
logger = create_logger("In Context Learning ", log_file=os.path.join("../docs/", path_str), silent=True)
run_start = datetime.now()  # 开始时间

print('loading model')
chat_glm = LlamaModel()
tokenizer, model = chat_glm.load_model(model_name_or_path=llm_model_dict[llm_model], llm_device=llm_device)


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

    instruction = "Instruction: get answers for the question based on the context, where answers derived from substrings in the context or categorized as 'unanswerable':\n"

    scope_instruction = "Instruction: get answers for the question based on the context. " \
        "If the time interval of when the event mentioned in the question occurred in the context, " \
        "the answer should be derived from substrings in the context. Else output 'unanswerable':\n"

    if sample_type == "counterfactual":
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
    else:
        input_sample = f'Context: {context}\nQuestion: {query}\nAnswer:'

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

        # feed to LLMs and  Generate
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


run_end = datetime.now()
print("运行时间: " + str((run_end - run_start).seconds / 60) + " minutes")
logger.info("Scope F1=== expand_f1: {}, narrow_f1: {}, granularity_f1: {} ; "
            "Scope EM=== expand_em: {}, narrow_em:{}, granularity_em: {} \r\n"
            .format(100.0 * expand_f1/119, 100.0 * narrow_f1/161, 100.0 * granularity_f1/170,
                    100.0 * expand_em/119, 100.0 * narrow_em/161, 100.0 * granularity_em/170))

print("Global F1: {} , Global EM: {} \n".format(100.0 * sum_f1 / len(timeQA_test), 100.0 * sum_em / len(timeQA_test)))
logger.info("Global F1: {} , Global EM: {} \r\n".format(100.0 * sum_f1 / len(timeQA_test), 100.0 * sum_em / len(timeQA_test)))
logger.info("answerable_f1: {} , answerable_em: {}\r\n".format(100.0 * answerable_f1 / (len(timeQA_test)-type_count), 100.0 * answerable_em / (len(timeQA_test)-type_count)))
if type_count:
    logger.info("unanswerable_f1 : {} , unanswerable_em : {}\r\n".format(100.0 * unanswerable_f1 / type_count, 100.0 * unanswerable_em / type_count))









