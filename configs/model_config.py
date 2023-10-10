import torch.cuda
import torch.backends


embedding_model_dict = {
    "ernie-tiny": "/home/weiyifan/LLMs/langchain_ChatGLM/ernie-3.0-nano-zh",
    "ernie-base": "/home/weiyifan/LLMs/langchain_ChatGLM/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# supported LLM models
llm_model_dict = {
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/chatglm-6b",
    "vicuna-13b-delta-v0": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/vicuna-13b-delta-v0",
    "bloom-7b1": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/bloom-7b1",
    "opt-6.7b": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/opt-6.7b",
    "opt-13b": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/opt-13b",
    "gpt-neox-20b": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/gpt-neox-20b",
    "gpt-j-6b": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/gpt-j-6b",
    "llama-7b-hf-int8": "decapoda-research/llama-7b-hf-int8",
    "llama-7b-hf": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/llama-7b-hf",
    "llama-13b-hf-int4": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/llama-13b-hf-int4",
    "llama-13b-hf": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/llama-13b-hf",

}

llama_model_dict = {
    "llama-7b-hf-int8": "decapoda-research/llama-7b-hf-int8",
    "llama-7b-hf": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/llama-7b-hf",
    "llama-13b-hf-int4": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/llama-13b-hf-int4",
    "llama-13b-hf": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/llama-13b-hf",
}

gpt_model_dict = {
    "gpt-neox-20b": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/gpt-neox-20b",
    "gpt-j-6b": "/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/gpt-j-6b",
}

# LLM model name
LLM_MODEL = "chatglm-6b"


# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# api key


DEVICE_MAP = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 0, 'model.layers.17': 0, 'model.layers.18': 0, 'model.layers.19': 0, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29.self_attn.q_proj': 1, 'model.layers.29.self_attn.k_proj': 1, 'model.layers.29.self_attn.v_proj': 1, 'model.layers.29.self_attn.o_proj': 1, 'model.layers.29.self_attn.rotary_emb': 1, 'model.layers.29.mlp': 1, 'model.layers.29.input_layernorm': 1, 'model.layers.29.post_attention_layernorm': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.norm': 0, 'lm_head': 0}