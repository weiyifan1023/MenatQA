import gzip
import re
from collections import Counter, OrderedDict
import re
import string
import warnings
import logging


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(examples: dict, references: dict):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    if len(examples) != len(references):
        warnings.warn('The length of the prediction and reference are not the same')
        assert len(examples) < len(references), 'prediction should be a subset'

    for idx, prediction in examples.items():
        reference = references[idx]
        assert isinstance(reference, list), reference

        exact_scores[idx] = max(compute_exact(a, prediction) for a in reference)
        f1_scores[idx] = max(compute_f1(a, prediction) for a in reference)

    return OrderedDict(
        [
            ("exact", 100.0 * sum(exact_scores.values()) / len(exact_scores)),
            ("f1", 100.0 * sum(f1_scores.values()) / len(f1_scores)),
            ("total", len(examples)),
        ]
    )


def readGZip(file_name):
    if file_name.endswith('gz'):
        with gzip.GzipFile(file_name, 'r') as fin:  # 4. gzip
            json_bytes = fin.read()  # 3. bytes (i.e. UTF-8)

        json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
        data = json.loads(json_str)  # 1. data
        return data
    else:
        with open(file_name, 'r') as fin:
            data = json.load(fin)
        return data


# 计算token个数
def count_input_token(input_text):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("../checkpoints/gpt-j-6B")  # load 本地
    # 初始化 GPT-2 tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 待计算的文本
    # input_text = "Hello, world! This is a sample text for computing the length and token count."
    token_count = len(tokenizer.tokenize(input_text))

    # 计算文本长度和 token 数量
    # text_length = len(input_text)
    # token_count = len(tokenizer.encode(text))

    # 打印结果
    # print("Text length:", text_length)
    print("Token count: ", token_count)
    return token_count


def create_logger(name, silent=False, to_disk=True, log_file=None):
    """Logger wrapper
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False

    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)  # 往屏幕上输出
        ch.setLevel(logging.INFO)  # 设置日志级别
        ch.setFormatter(formatter)  # 设置屏幕上显示的格式
        log.addHandler(ch)  # 把对象加到logger里
    if to_disk:
        log_file = log_file if log_file is not None else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log


month_abbr_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
moth_name_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

Month_dictionary = {
    "January": "01",
}
