import json
import os, sys
import random
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import numpy as np
from tqdm import tqdm


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
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

checkpoints_save_path = 'checkpoints'
# peft_path = None
peft_path = 'llama-13b-int4-dolly'

dataset_size=None
max_length = 512

def write_file(data_list, path):
    # 写入文件
    with open(path, 'a') as file:
        for dictionary in data_list:
            file.write(json.dumps(dictionary))
            file.write("\n")  # 在字典之间插入空行

def pre_dataset():
    from pm4py.objects.log.importer.xes import importer as xes_importer
    log = xes_importer.apply(data_path)  # 引号中的为文件地址
    # variants = pm4py.get_variants(log)
    traces = set()
    data_list = []
    print(f"number of traces:{len(log)}")
    for case in log:
        trace = []
        for event in case:
            # if event['lifecycle:transition'] == 'COMPLETE':
            #     trace.append(event['concept:name'])
            trace.append(event['concept:name'])
        traces.add(tuple(trace))
    for trace in traces:
        data_list.append('[' + ','.join(trace) + ']')
    print(f"number of variants:{len(data_list)}")
    return data_list
    # print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--data_path', type=str, default='dataset/BPIC20_PermitLog.xes', help='Specify the path to the test dataset.')

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
    data_list = pre_dataset()

    file_name = os.path.basename(data_path)
    # 分离文件前缀和扩展名
    file_prefix, _ = os.path.splitext(file_name)
    res_file_path = file_prefix+'.txt'
    with open(res_file_path, 'w') as file:
        file.write('')


    with torch.no_grad():
        for i in tqdm(range(len(data_list))):
            trace = data_list[i]

            prompts = [f'In the following business process trace, each executed activity is separated by a comma: {trace}. \\n Is this trace normal or anomalous?']

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

            output = tokenizer.batch_decode(generate_ids)[0]
            # print(output_all)
            # gen_ = generate_ids[:, inputs['input_ids'].shape[1]:]  # 只取出生成部分，不要问题部分
            # output = tokenizer.batch_decode(gen_)[0]

            # print(output)
            matches = re.findall(r'The trace is (.*?)\.</s>', output)
            if len(matches)>0 and 'anomalous' == matches[0]:
                prompts = [
                    f'In the following business process trace, each executed activity is separated by a comma: {trace}. Is this trace normal or anomalous? \\n The trace is anomalous. \\n What makes this trace anomalous?']

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

                output = tokenizer.batch_decode(generate_ids)[0]
            print(output)

            with open(res_file_path, 'a') as file:
                file.write(output + '\n')