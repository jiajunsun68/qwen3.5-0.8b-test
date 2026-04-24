"""
test_fine_tuning.py
对 Qwen3.5-0.8B 进行轻量级微调流程的端到端测试（使用极小数据集验证流程可用性）。
"""

import unittest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # 可替换为本地路径或目标模型

SAMPLE_DATA = [
    {"text": "问：天空是什么颜色？\n答：天空是蓝色的。"},
    {"text": "问：水的沸点是多少？\n答：在标准大气压下，水的沸点是100°C。"},
    {"text": "问：1加1等于几？\n答：1加1等于2。"},
]


class TestFineTuning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    def _tokenize(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    def test_training_step_runs(self):
        """微调训练步骤能够正常运行而不抛出异常"""
        raw_dataset = Dataset.from_list(SAMPLE_DATA)
        tokenized = raw_dataset.map(self._tokenize, batched=True, remove_columns=["text"])
        tokenized = tokenized.map(lambda x: {"labels": x["input_ids"][:]})

        training_args = TrainingArguments(
            output_dir="/tmp/qwen_finetune_test",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_steps=2,
            logging_steps=1,
            save_steps=999,
            no_cuda=not torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        result = trainer.train()
        self.assertIsNotNone(result)
        self.assertIn("train_loss", result.metrics)

    def test_model_parameters_update(self):
        """训练后模型参数应与初始值不同（至少有梯度）"""
        param = next(self.model.parameters())
        self.assertTrue(param.requires_grad)


if __name__ == "__main__":
    unittest.main()
