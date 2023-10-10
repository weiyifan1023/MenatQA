from .chatglm_with_shared_memory_openai_llm import *
import os
os.environ['HTTP_PROXY'] = "http://cipzhao:cipzhao@210.75.240.136:10800"
os.environ['HTTPS_PROXY'] = "http://cipzhao:cipzhao@210.75.240.136:10800"
def get_api_key():
    return "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"