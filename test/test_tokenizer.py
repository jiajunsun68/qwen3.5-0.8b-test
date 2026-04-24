"""
test_tokenizer.py
测试 Qwen3.5-0.8B 分词器的基本功能。
"""

import unittest
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # 可替换为本地路径或目标模型


class TestTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def test_encode_decode_roundtrip(self):
        """编码后再解码应还原原始文本"""
        text = "Qwen 模型测试"
        token_ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        self.assertEqual(decoded.strip(), text.strip())

    def test_special_tokens_exist(self):
        """分词器应包含常用特殊 token"""
        self.assertIsNotNone(self.tokenizer.eos_token)
        self.assertIsNotNone(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)

    def test_batch_encoding(self):
        """批量编码应返回正确数量的序列"""
        texts = ["你好", "Hello, world!", "1 + 1 = 2"]
        encoding = self.tokenizer(texts, padding=True, return_tensors="pt")
        self.assertEqual(encoding["input_ids"].shape[0], len(texts))

    def test_vocab_size(self):
        """词表大小应为正整数"""
        self.assertGreater(self.tokenizer.vocab_size, 0)

    def test_chinese_tokenization(self):
        """中文文本应被正确分词（不产生空 token 列表）"""
        text = "人工智能正在改变世界"
        tokens = self.tokenizer.tokenize(text)
        self.assertGreater(len(tokens), 0)


if __name__ == "__main__":
    unittest.main()
