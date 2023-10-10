# MenatQA
This is the repository for the EMNLP 2023 paper [MenatQA: A New Dataset for Testing the Temporal Comprehension and Reasoning Abilities of Large Language Models](https://arxiv.org/pdf/2310.05157.pdf)

## Introduction
![Image text](https://github.com/weiyifan1023/MenatQA/blob/main/time_example.png)

## Code Usage

First you should first set up your API key in the `src/my_api_secrets.py` file, 
and then make sure to pre-configure the checkpoints storage path for LLMs in the `config/model_config.py` file.

And then, please change the working directory to `src/`.

Next, please choose specific LLMs for execution based on corresponding file names. 
For example, You can evaluate "LLAMA" LLMs to run the `src/test_llama_time.py` script.

Prompt methods and tool comparison tools, as described in the paper, refer to the scripts of "LLAMA" and "ChatGPT" for reference.

## Framework
![Image text](https://github.com/weiyifan1023/MenatQA/blob/main/time%20tool.png)

## Citation
If you use this code for your research, please kindly cite our EMNLP 2023 paper:

```
@misc{wei2023menatqa,
      title={MenatQA: A New Dataset for Testing the Temporal Comprehension and Reasoning Abilities of Large Language Models}, 
      author={Yifan Wei and Yisong Su and Huanhuan Ma and Xiaoyan Yu and Fangyu Lei and Yuanzhe Zhang and Jun Zhao and Kang Liu},
      year={2023},
      eprint={2310.05157},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact

Yifan Wei: weiyifan2021@ia.ac.cn (Preferred)  &&  weiyifan21@mails.ucas.ac.cn 
