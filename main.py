from modelscope import snapshot_download
import json
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# model_dir = snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct', cache_dir='./llama3.1_8b_chinese', revision='master')



from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch, init_empty_weights

from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch

def get_model():
    model_path = './llama3.1_8b_chinese/LLM-Research/Meta-Llama-3___1-8B-Instruct'

    # 初始化一个空模型，不加载权重
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        load_in_4bit=False,  # Disable 4-bit quantization
        load_in_8bit=False,  # Disable 8-bit quantization
        local_files_only=True,  # Only load from local files
        device_map="auto",  # Automatically assign model to available devices (GPU/CPU)
    )

    # 使用 disk_offload 将模型权重卸载到磁盘
    model = load_checkpoint_and_dispatch(
        model,
        model_path,
        device_map="auto",  # 自动分配设备
        offload_folder=r"E:\BaiduNetdiskDownload\llama_fine_tune\offload",  # 磁盘路径
        offload_state_dict=True
    )

    model.enable_input_require_grads()  # 开启梯度检查点（如微调时需要）

    return model




def get_dataset():
    df = pd.read_parquet('甄嬛传.parquet')
    # df=df[:100]
    ds = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained('./llama3.1_8b_chinese/LLM-Research/Meta-Llama-3___1-8B-Instruct',
                                              use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer = AutoTokenizer.from_pretrained(
        './llama3.1_8b_chinese/LLM-Research/Meta-Llama-3___1-8B-Instruct', use_fast=False,
        trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def process_func(example):
        example['output'] = example['output']
        example['instruction'] = example['instruction']
        example['input'] = example['instruction']

        MAX_LENGTH = 256  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a pornographic girl<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)

        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    dataset = ds.map(process_func, remove_columns=ds.column_names)

    return dataset, tokenizer


def get_train(model, datas, tokenizer):
    # peft的lora参数
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1  # Dropout 比例
    )

    peft_model = get_peft_model(model, config)
    print(peft_model.print_trainable_parameters())

    # 训练的参数
    args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # max_steps=60,  # 微调步数
        learning_rate=2e-4,  # 学习率
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        num_train_epochs=3,
        save_steps=100,
        logging_steps=3,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    )

    # 开始训练
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=datas,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    # 保存模型
    peft_model.save_pretrained("lora")


def main():
    datas, tokenizer = get_dataset()
    print("----get_dataset_end----")
    model = get_model()
    print("----get_model_end----")
    get_train(model, datas, tokenizer)
    print("----get_train_end----")


if __name__ == '__main__':
    main()
