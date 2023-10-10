import torch
import json
import argparse
import threading

from accelerate import init_empty_weights, infer_auto_device_map
import transformers
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List, Union


def get_device_map(model_name, device, do_int8):
    if device == "a100-40g":
        return "auto"

    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    d = {0: "18GiB"}
    for i in range(1, 6):
        d[i] = "26GiB"
    device_map = infer_auto_device_map(
        model, max_memory=d, dtype=torch.int8 if do_int8 else torch.float16, no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer", "LlamaDecoderLayer"]
    )
    print(device_map)
    del model
    return device_map

# from configs.model_config import *
# class LlamaModel:
#     def __init__(self):
#         self.tokenizer = None
#         self.model = None
#
#     def load_model(self, model_name_or_path: str = "decapoda-research/llama-13b-hf", llm_device=LLM_DEVICE):
#         # print device map
#         # config = AutoConfig.from_pretrained(model_name_or_path)
#         # with init_empty_weights():
#         #     model = AutoModelForCausalLM.from_config(config)
#         #
#         # device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"])
#         # print(device_map)
#
#         # tokenizer
#         self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
#         # model
#         if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
#             self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
#             hf_device_map = self.model.hf_device_map
#         else:
#             self.model = (
#                 AutoModelForCausalLM.from_pretrained(model_name_or_path).float().to(llm_device)
#             )
#
#         # self.model = self.model.eval()
#         return self.tokenizer, self.model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../checkpoints/llama-7b-hf")
    parser.add_argument("--variant", type=str, default="65b", choices=["7b", "13b", "33b", "65b"])
    parser.add_argument(
        "--device", type=str, choices=["a100-40g", "v100-32g"], default="a100-40g"
    )
    parser.add_argument("--do_int8", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--port", type=int, default=12333)
    # my setting
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    # data type
    parser.add_argument("--update", default=1, type=int)  # 对照实验记得改这里！！！
    parser.add_argument("--counterfactual", default=1, type=int)
    parser.add_argument("--disorder", default=1, type=int)
    parser.add_argument("--all_update", default=None, type=int)
    args = parser.parse_args()

    # load llama_model:
    #
    # llama_model = LlamaModel()
    # tokenizer, model = llama_model.load_model(args.model_path, LLM_DEVICE)
    # prompt = "Puma is a "
    # inputs = tokenizer(prompt, return_tensors="pt").input_ids
    # inputs = inputs.to(0)
    # # 将输入tensor移动到cuda:0设备上
    # # inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
    # generate_ids = model.generate(inputs, max_length=512)
    # print(" ")
    # response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # # response, history_ = model.chat(tokenizer, input_sample, history=instruction)
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    #
    # with torch.no_grad():
    #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    #     assert len(input_ids) == 1, len(input_ids)
    #     if input_ids[0][-1] == 2:  # 2 is EOS, hack to remove. If the prompt is ending with EOS, often the generation will stop abruptly.
    #         input_ids = input_ids[:, :-1]
    #     input_ids = input_ids.to(0)
    #     # input_ids = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").input_ids.to(0)
    #     generated_ids = model.generate(
    #         input_ids,
    #         # stopping_criteria=stopping_criteria,
    #         # **generate_kwargs
    #     )
    #     result = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
    #     print(result)

    #
    model_id = f"{args.model_path}"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=get_device_map(model_id, args.device, args.do_int8),
        torch_dtype=torch.int8 if args.do_int8 else torch.float16,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        load_in_8bit=args.do_int8,
    )
    tokenizer = LlamaTokenizer.from_pretrained(f"{args.model_path}", use_fast="/opt" not in model_id)
    #tokenizer.pad_token_id = -1

    generate_kwargs = {
        "max_new_tokens": 50,
        "min_new_tokens": 1,
        "temperature": 0.1,
        "do_sample": False,  # The three options below used together leads to contrastive search
        "top_k": 1,
        "penalty_alpha": 0.6,
        #"no_repeat_ngram_size": no_repeat_ngram_size,
        #**generation_config,
    }
    with open('../datasets/test_qq_1000_51.json') as f:
        timeQA_test = json.load(f)
    timeQA_test = timeQA_test[args.start:args.end]
    print('number of examples: ', len(timeQA_test))

    for ids, example in enumerate(timeQA_test, start=1):
        # type
        # query_type = example["type"]
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
        else:
            context = [text["text"] for text in example["context"]]

        # constructing input
        instruction = "Instruction: Answer the question based on context, with answers derived from substrings in the context or categorized as 'unanswerable':\n"
        prompt = instruction + "Context:...\n" + "Question: XXXXXX ?\n" + "Answer: XXX\n\n"

        context = "\n".join(context)
        input_sample = f'{instruction}Context: {context}\nQuestion: {query}'

        with torch.no_grad():
            input_ids = tokenizer(prompt + input_sample, return_tensors="pt").input_ids
            input_len = input_ids.shape[1]
            assert len(input_ids) == 1, len(input_ids)
            if input_ids[0][-1] == 2:  # 2 is EOS, hack to remove. If the prompt is ending with EOS, often the generation will stop abruptly.
                input_ids = input_ids[:, :-1]
            input_ids = input_ids.to(0)
            # input_ids = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").input_ids.to(0)
            generated_ids = model.generate(
                input_ids,
                # stopping_criteria=stopping_criteria,
                **generate_kwargs
            )
            result = tokenizer.batch_decode(generated_ids[:, input_len:].cpu(), skip_special_tokens=True)
            print("ID: ", ids)
            print("Question: ", query)
            print("Answer: ",  result)



