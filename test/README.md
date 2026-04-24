# Test 目录说明

本目录包含对 Qwen 小模型（默认使用 `Qwen/Qwen2.5-0.5B`，可替换为目标模型路径）的各类测试，每个测试模块独立存放：

| 文件 | 说明 |
|------|------|
| `test_inference.py` | 模型推理/文本生成测试 |
| `test_tokenizer.py` | 分词器功能测试 |
| `test_fine_tuning.py` | 模型微调流程测试 |
| `test_evaluation.py` | 模型评测指标测试 |

## 环境依赖

```bash
pip install torch transformers datasets evaluate
```

## 快速运行

```bash
# 运行所有测试
python -m pytest test/

# 单独运行某个测试文件
python test/test_inference.py
```
