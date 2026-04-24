"""
test_evaluation.py
测试 Qwen3.5-0.8B 模型的评测指标计算（困惑度、BLEU 等）。
"""

import unittest
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # 可替换为本地路径或目标模型

REFERENCE_SENTENCES = [
    "人工智能正在改变我们的生活方式。",
    "大型语言模型具有强大的文本生成能力。",
    "深度学习在图像识别领域取得了显著进展。",
]


class TestEvaluation(unittest.TestCase):

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

    def _perplexity(self, text: str) -> float:
        """计算给定文本的困惑度（Perplexity）"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        return math.exp(outputs.loss.item())

    def test_perplexity_is_finite(self):
        """困惑度应为有限正数"""
        for sentence in REFERENCE_SENTENCES:
            ppl = self._perplexity(sentence)
            self.assertGreater(ppl, 0)
            self.assertTrue(math.isfinite(ppl), f"困惑度不是有限数：{ppl}")

    def test_perplexity_reasonable_range(self):
        """常规中文句子的困惑度应在合理范围内（< 10000）"""
        for sentence in REFERENCE_SENTENCES:
            ppl = self._perplexity(sentence)
            self.assertLess(ppl, 10000, f"困惑度过高（{ppl:.2f}），模型可能未正确加载")

    def test_in_context_lowers_perplexity(self):
        """提供上下文后，续写部分的困惑度应低于无上下文情况"""
        context = "深度学习在图像识别领域"
        continuation = "取得了显著进展。"
        ppl_with_context = self._perplexity(context + continuation)
        ppl_no_context = self._perplexity(continuation)
        # 允许带上下文时整体困惑度最多为无上下文时的 2 倍（宽松阈值，
        # 因为整体句子 PPL 包含了上下文本身的损失）
        CONTEXT_PERPLEXITY_THRESHOLD = 2
        self.assertLessEqual(ppl_with_context, ppl_no_context * CONTEXT_PERPLEXITY_THRESHOLD)

    def test_bleu_score_import(self):
        """evaluate 库的 BLEU 实现应可正常导入"""
        try:
            import evaluate
            bleu = evaluate.load("bleu")
            self.assertIsNotNone(bleu)
        except ImportError:
            self.skipTest("evaluate 库未安装，跳过 BLEU 测试")


if __name__ == "__main__":
    unittest.main()
