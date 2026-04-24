"""
test_inference.py
测试 Qwen3.5-0.8B 模型的文本生成（推理）功能。
"""

import unittest
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # 可替换为本地路径或目标模型


class TestInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        cls.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        cls.model.eval()

    def _generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def test_basic_generation(self):
        """模型能够生成非空文本"""
        result = self._generate("你好，请介绍一下自己。")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)

    def test_max_new_tokens_respected(self):
        """生成长度不超过 max_new_tokens"""
        prompt = "请写一首关于春天的诗："
        result = self._generate(prompt, max_new_tokens=20)
        token_count = len(self.tokenizer(result)["input_ids"])
        self.assertLessEqual(token_count, 20)

    def test_deterministic_output(self):
        """greedy decode 结果应保持一致"""
        prompt = "1 + 1 = "
        result1 = self._generate(prompt)
        result2 = self._generate(prompt)
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
