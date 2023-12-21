# MenatQA
This is the repository for the EMNLP 2023 paper [MenatQA: A New Dataset for Testing the Temporal Comprehension and Reasoning Abilities of Large Language Models](https://arxiv.org/pdf/2310.05157.pdf)

This paper presents a new dataset, MenatQA, for testing the time comprehension and reasoning abilities of large language models (LLMs).

![Image text](https://github.com/weiyifan1023/MenatQA/blob/main/time_example.png)

we consider three temporal factors, i.e., scope factor, order factor, and counterfactual factor. After testing multiple models on this dataset, we find that LLMs may fall behind smaller temporal reasoning models. Also, scope factor and counterfactual factor generally impact more on LLMs, and LLMs struggle more with reasoning compared with extractions. 

## Requirements

1. BigBird-Specific Requirements
      - Transformers 4.8.2
      - Pytorch 1.8.1+cu102
2. FiD-Specific Requirements
      - Transformers 3.0.2
      - Pytorch 1.6.0
3. LLMs Requirements
      - langchain ==0.0.166
      - transformers==4.28.1
      - Pytorch 2.0.0




## Code Usage
To compare with traditional models (Bigbird and FiD), I would recommend referring to the following GitHub repository: https://github.com/wenhuchen/Time-Sensitive-QA/tree/main. 
### BigBird
Extractive QA baseline model, first switch to the BigBird Conda environment:

#### Initialize from NQ checkpoint
Running Training (Hard)
```
    python -m BigBird.main model_id=nq dataset=hard cuda=[DEVICE] mode=train per_gpu_train_batch_size=8
```


#### Initialize from TriviaQA checkpoint
Running Training (Hard)
```
    python -m BigBird.main model_id=triviaqa dataset=hard cuda=[DEVICE] mode=train per_gpu_train_batch_size=2
```


### Fusion-in Decoder (FiD)
Generative QA baseline model, first switch to the FiD Conda environment and downaload the checkpoints from [Google Drive](https://drive.google.com/file/d/19DnItecTwqUqhw09zH3eR61iz_22dX-u/view?usp=sharing):
#### Initialize from NQ checkpoint
Running Training (Hard)
```
    python -m FiD.main mode=train dataset=hard model_path=/data2/wenhu/Time-Sensitive-QA/FiD/pretrained_models/nq_reader_base/
```


#### Initialize from TriviaQA checkpoint
Running Training (Hard)
```
    python -m FiD.main mode=train dataset=hard model_path=/data2/wenhu/Time-Sensitive-QA/FiD/pretrained_models/tqa_reader_base/
```


So far, you can evaluate Fine-tuning models on MenatQA.

### LLMs
First you should first set up your API key in the `src/my_api_secrets.py` file, 
and then make sure to pre-configure the checkpoints storage path for LLMs in the `config/model_config.py` file.

And then, please change the working directory to `src/`.

Next, please choose specific LLMs for execution based on corresponding file names. 
For example, You can evaluate "LLAMA" LLMs to run the `src/test_llama_time.py` script.
```
    python3 test_llama_time.py
```

Prompt methods and tool comparison tools, as described in the paper, refer to the scripts of "LLAMA" and "ChatGPT" for reference.
The time comparison tool is implemented based on Langchain. You can further optimize or customize your tool in the `agent/` file by following the instructions in the official documentation at https://python.langchain.com/docs/get_started/introduction.

## Framework
![Image text](https://github.com/weiyifan1023/MenatQA/blob/main/time%20tool.png)

## Citation
If you use this code for your research, please kindly cite our EMNLP 2023 paper:

```
@inproceedings{wei2023menatqa,
  title={MenatQA: A New Dataset for Testing the Temporal Comprehension and Reasoning Abilities of Large Language Models},
  author={Wei, Yifan and Su, Yisong and Ma, Huanhuan and Yu, Xiaoyan and Lei, Fangyu and Zhang, Yuanzhe and Zhao, Jun and Liu, Kang},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={1434--1447},
  year={2023}
}
```

## Contact

Yifan Wei: weiyifan2021@ia.ac.cn (Preferred)  &&  weiyifan21@mails.ucas.ac.cn 
