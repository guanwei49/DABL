import os, sys
import random
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import numpy as np
from tqdm import tqdm

from prompt import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model

llm_path = "./llama-2-13b-chat-hf"

# peft_path = None
peft_path = 'llama-13b-int4-dolly'

# dataset_size=10
dataset_size=None
max_length = 512
batch_size = 4

def generate_prompt(trace, label=None, cause=None, all=True, prompt_template=prompt_template):
    p1 = preface1[random.randint(0, len(preface1) - 1)]
    p2 = preface2[random.randint(0, len(preface2) - 1)]
    ask_c = ask_cause[random.randint(0, len(ask_cause) - 1)]

    if cause:
        if all:
            res = prompt_template["prompt_with_cause_all"].format(p1=p1, p2=p2, ask_c=ask_c,
                                                                  trace=trace, label=label, cause=cause)
        else:
            res = prompt_template["prompt_with_cause_q"].format(p1=p1, p2=p2, ask_c=ask_c,
                                                                trace=trace, label=label)
    else:
        if all:
            res = prompt_template["prompt_no_cause_all"].format(p1=p1, p2=p2,
                                                                trace=trace, label=label)
        else:
            res = prompt_template["prompt_no_cause_q"].format(p1=p1, p2=p2,
                                                              trace=trace)
    return res


def tokenize(tokenizer, prompt, max_length=max_length, add_eos_token=False):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None)

    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    # user prompt has no response
    user_prompt = generate_prompt(
        data_point["trace"],
        data_point["label"],
        data_point["cause"],
        all=False
    )
    return {'prompt': user_prompt}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--data_path', type=str, default='dataset/test_dataset_2.jsonl', help='Specify the path to the test dataset.')

    args = parser.parse_args()

    data_path = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(llm_path, padding_side="left")
    # tokenizer.pad_token = tokenizer.bos_token
    tokenizer.pad_token = tokenizer.unk_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # load the model into memory using 4-bit precision
        bnb_4bit_use_double_quant=False,  # use double quantition
        bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
        bnb_4bit_compute_dtype=torch.bfloat16  # use hf for computing when we need
    )

    # model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16, device_map='auto')

    # model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=bnb_config, use_cache=False,
    #                                              device_map='auto')

    model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=bnb_config, torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True, device_map='auto')

    if peft_path is not None:
        print(f'load {peft_path}')
        model = PeftModel.from_pretrained(
            model,
            peft_path,
            torch_dtype=torch.float16,
        )

    ########################################################
    print(f'dataset: {data_path}')
    dataset = datasets.load_dataset("json",
                                    data_files=data_path, cache_dir='cache_data')



    cols = ["trace"]
    if dataset_size is not None:
        dataset = dataset['train'].train_test_split(train_size=dataset_size, shuffle=False)
    dataset = dataset['train'].map(generate_and_tokenize_prompt, remove_columns=cols)

    model.eval()

    ################eval
    pre = 0
    all_ad_Pred = []
    all_ad_GT = []
    all_cause_Pred=[]
    all_cause_GT=[]

    with torch.no_grad():
        for bathc_i in tqdm(range(batch_size, len(dataset) + batch_size, batch_size)):
            if bathc_i <= len(dataset):
                right = bathc_i
            else:
                right = len(dataset)

            prompts = dataset['prompt'][pre:right]


            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')
            # print(inputs['input_ids'][0].__len__())

            generate_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                # max_length = 500,
                # min_length = 300,
                num_beams=4,
                # num_beam_groups=2,
                top_k=5,  # 用于在生成下一个token时，限制模型只能考虑前k个概率最高的token，这个策略可以降低模型生成无意义或重复的输出的概率
                # temperature=0.1,  # 该参数用于控制生成文本的随机性和多样性，
                # repetition_penalty=1., #避免重复,1表示不进行惩罚
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id)

            output_all = tokenizer.batch_decode(generate_ids)
            # print(output_all)
            gen_ = generate_ids[:, inputs['input_ids'].shape[1]:]  # 只取出生成部分，不要问题部分
            output = tokenizer.batch_decode(gen_)

            if 'cause' not in data_path:
                ad_Pred = []
                for text in output:
                    matches = re.findall(r'The trace is (.*?)\.</s>', text)
                    if len(matches) > 0:
                        ad_Pred.append(matches[0])
                    else:
                        ad_Pred.append('')
                print(ad_Pred)
                ad_GT = dataset['label'][pre:right]
                print("gt:{}".format(ad_GT))
                all_ad_Pred +=ad_Pred
                all_ad_GT += ad_GT

            else:
                cause_Pred=[]
                for text in output:
                    matches = re.findall(r'\\n (.*?)</s>', text)
                    if len(matches) > 0:
                        cause_Pred.append(matches[0])
                    else:
                        cause_Pred.append('')
                print(cause_Pred)
                cause_GT = dataset['cause'][pre:right]
                print("Cause gt:{}".format(cause_GT))
                all_cause_Pred += cause_Pred
                all_cause_GT += cause_GT

            pre = bathc_i


    if 'cause' not in data_path:
        all_ad_GT = np.array(all_ad_GT)
        all_ad_Pred = np.array(all_ad_Pred)

        print(all_ad_Pred)
        print(all_ad_GT)

        precision = precision_score(all_ad_GT, all_ad_Pred, average="binary", pos_label='anomalous')
        recall = recall_score(all_ad_GT, all_ad_Pred, average="binary", pos_label='anomalous')
        f = f1_score(all_ad_GT, all_ad_Pred, average="binary", pos_label='anomalous')

        acc = accuracy_score(all_ad_GT, all_ad_Pred)

        num_anomalous = (all_ad_GT == 'anomalous').sum()
        num_normal = (all_ad_GT == 'normal').sum()

        print(f'Number of anomalous traces: {num_anomalous}; number of normal traces: {num_normal}')

        det_num_anomalous = (all_ad_Pred == 'anomalous').sum()
        det_num_normal = (all_ad_Pred == 'normal').sum()

        print(f'Number of detected anomalous traces: {det_num_anomalous}; number of detected normal traces: {det_num_normal}')

        print(f'precision: {precision}, recall: {recall}, f1: {f}, accuracy: {acc}')

    else:
        from rouge import Rouge

        rouge = Rouge()

        scores = rouge.get_scores(all_cause_Pred, all_cause_GT, avg=True)
        rouge2_socres= scores['rouge-2']

        rouge2_r = rouge2_socres['r']
        rouge2_p = rouge2_socres['p']
        rouge2_f = rouge2_socres['f']

        print(f'rouge2_r: {rouge2_r}; rouge2_p: {rouge2_p}; rouge2_f: {rouge2_f}')

        rougel_socres= scores['rouge-l']

        rougel_r = rougel_socres['r']
        rougel_p = rougel_socres['p']
        rougel_f = rougel_socres['f']

        print(f'rougel_r: {rougel_r}; rougel_p: {rougel_p}; rougel_f: {rougel_f}')

