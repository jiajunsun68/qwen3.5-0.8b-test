"""
这里是训练tastypear的sft代码片段，数据集来自datasets/lmsys-chat-lewd-filter.csv
数据集内容：第一列是propmt，第二列是chosen
"""

import os
import pandas as pd
from datasets import Dataset
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainerCallback
import transformers.utils.hub as transformers_hub
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

if not hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
    from huggingface_hub.constants import HF_HUB_CACHE

    transformers_hub.TRANSFORMERS_CACHE = HF_HUB_CACHE

from trl import SFTTrainer, DPOTrainer, SFTConfig, DPOConfig

# --- 1. 基础配置 ---
MODEL_PATH = "qwen"                         # 你的模型路径
CSV_PATH = "datasets/lmsys-chat-lewd-filter.csv"
MAX_SEQ_LENGTH = 2048                       # 最大序列长度
MAX_PROMPT_LENGTH = 512                     # DPO中prompt的最大长度
LORA_RANK = 16                              # LoRA的秩，控制可训练参数量
LORA_ALPHA = 16                             # LoRA的缩放参数，通常设为与rank相同
OUTPUT_DIR_SFT = "./qwen_output_sft"             # SFT阶段输出目录
OUTPUT_DIR_DPO = "./qwen_output_dpo"             # DPO阶段输出目录

# 根据硬件调整训练批次大小
BATCH_SIZE_SFT = 4
BATCH_SIZE_DPO = 2


def format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TqdmTrainCallback(TrainerCallback):
    def __init__(self, desc: str) -> None:
        self.desc = desc
        self.progress_bar = None
        self.last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = int(state.max_steps) if state.max_steps else 0
        self.progress_bar = tqdm(
            total=total_steps,
            desc=self.desc,
            dynamic_ncols=True,
            leave=True,
        )
        self.last_step = 0
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.progress_bar is None:
            return control
        step_delta = int(state.global_step) - self.last_step
        if step_delta > 0:
            self.progress_bar.update(step_delta)
            self.last_step = int(state.global_step)
        remaining_steps = max((self.progress_bar.total or 0) - self.progress_bar.n, 0)
        rate = self.progress_bar.format_dict.get("rate")
        eta_seconds = (remaining_steps / rate) if rate else None
        self.progress_bar.set_postfix_str(f"ETA {format_seconds(eta_seconds)}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
        return control


def format_sft_instruction(example):
    """为SFT阶段格式化数据：将prompt和chosen拼接成可直接用于训练的文本"""
    # 这里以一个简单的格式为例，实际可替换为模型的chat_template
    return {"text": f"用户: {example['prompt']}\n助手: {example['chosen']}"}

def prepare_dpo_pairs(example):
    """为DPO阶段格式化数据：使用prompt作为prompt，chosen作为chosen，并构造一个rejected样本"""
    # 这是一个关键步骤，你需要根据你的数据创建被拒绝的响应
    # 例如：使用一个简单的回显作为rejected，或其他模型的生成结果
    # 这里用原始回答稍加修改示意，实际需替换为真实对比数据
    rejected_text = f"对不起，我无法回答这个问题。"
    return {
        "prompt": example['prompt'],
        "chosen": example['chosen'],
        "rejected": rejected_text
    }

def load_dataset(csv_path=CSV_PATH):
    """加载CSV数据集，并转换为HuggingFace Dataset格式"""
    df = pd.read_csv(csv_path, header=None, names=["prompt", "chosen"])
    return Dataset.from_pandas(df)

def load_model_and_tokenizer(model_path=MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA):
    """加载基础模型，并从SFT输出目录载入已训练好的LoRA适配器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    if os.path.isdir(OUTPUT_DIR_SFT):
        model = PeftModel.from_pretrained(model, OUTPUT_DIR_SFT, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
    model.to(device)
    model.config.use_cache = False
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.print_trainable_parameters()
    return model, tokenizer

def get_lora_adapter(model, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA):
    """为模型创建LoRA适配器"""
    return model

def sfttrainer(model, tokenizer, sft_dataset):
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_dataset,
        args=SFTConfig(
            output_dir=OUTPUT_DIR_SFT,
            dataset_text_field="text",
            per_device_train_batch_size=BATCH_SIZE_SFT,
            gradient_accumulation_steps=2,    # 梯度累积步数，等效于增大batch size
            warmup_steps=5,                   # 预热步数
            num_train_epochs=1,               # 训练轮数，可根据实际情况调整
            learning_rate=2e-4,               # 学习率，SFT常用2e-4
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,                 # 每10步输出一次日志
            optim="adamw_torch",             # 使用标准AdamW优化器，兼容普通trl/transformers训练
            seed=3407,
            report_to="none",                 # 禁用外部日志记录
        )
    )
    trainer.add_callback(TqdmTrainCallback("SFT Training"))
    return trainer

def dpotrainer(model, tokenizer, dpo_dataset):
    trainer = DPOTrainer(
        model=model,
        ref_model=None,                     # DPO需要参考模型，为None时DPOTrainer会自动从model复制一份[reference:10]
        args=DPOConfig(
            output_dir=OUTPUT_DIR_DPO,
            per_device_train_batch_size=BATCH_SIZE_DPO,
            gradient_accumulation_steps=4,
            warmup_steps=10,                # 使用固定步数预热，兼容当前trl版本
            num_train_epochs=1,
            learning_rate=5e-5,             # DPO阶段学习率通常比SFT稍低
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_torch",
            seed=3407,
            report_to="none",
            max_length=MAX_SEQ_LENGTH,      # 序列总长度上限[reference:12]
            max_prompt_length=MAX_PROMPT_LENGTH, # Prompt的最大长度[reference:13]
            beta=0.1,                       # DPO的温度参数，控制模型对偏好的敏感度
        ),
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )
    trainer.add_callback(TqdmTrainCallback("DPO Training"))
    return trainer

if __name__ == "__main__":
    # 设置gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # --- 2. 加载并处理数据集 ---
    print("加载数据集...")
    dataset = load_dataset()
    print(f"数据集大小: {len(dataset)} 条样本")

    # --- 3. 加载模型与分词器 ---
    print("加载SFT完成的模型...")
    model, tokenizer = load_model_and_tokenizer()
    print(f"模型加载完成，LoRA适配器已添加。可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- 5. 第二阶段：DPO训练 (加载已完成的SFT模型) ---
    print("开始DPO训练...")
    # 准备DPO数据集
    dpo_dataset = dataset.map(prepare_dpo_pairs, remove_columns=dataset.column_names)
    print(f"DPO数据集准备完成，包含 {len(dpo_dataset)} 条样本。")

    # 因为模型是在SFT之后，所以不再需要单独的reference model
    dpo_trainer = dpotrainer(model, tokenizer, dpo_dataset)

    # 开始DPO训练
    dpo_trainer.train()
    dpo_trainer.save_model(OUTPUT_DIR_DPO)
    print("DPO训练完成。模型已保存。")

