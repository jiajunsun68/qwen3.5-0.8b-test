# 验证kv cache

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 加载模型和 tokenizer（本地目录）
model_path = "././qwen"          # 与 py 文件同目录下的 qwen 文件夹
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,   # 可改为 float32，如果内存足够
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 2. 构造两个输入：基础序列和加一个 token 后的序列
base_text = "Hello world how are"          # 4 个 token（具体 token 数取决于分词）
extended_text = base_text + " you"         # 预期多一个 token

# 为了方便控制 token 长度，也可以手动指定 token ids，但用文本更直观
inputs_base = tokenizer(base_text, return_tensors="pt").to(model.device)
inputs_ext = tokenizer(extended_text, return_tensors="pt").to(model.device)

print(f"base 序列 token 数: {inputs_base['input_ids'].shape[1]}")
print(f"extended 序列 token 数: {inputs_ext['input_ids'].shape[1]}")

# 3. 前向传播，要求返回所有层的 hidden states
with torch.no_grad():
    outputs_base = model(**inputs_base, output_hidden_states=True)
    outputs_ext = model(**inputs_ext, output_hidden_states=True)

# 4. 提取 hidden_states
# hidden_states 是一个 tuple，长度 = 层数 + 1（第 0 项是 embedding 输出）
# 第一层的输出（即第二层的输入）索引为 1
hidden_base_layer1 = outputs_base.hidden_states[1]   # shape: [1, seq_len_base, hidden_size]
hidden_ext_layer1 = outputs_ext.hidden_states[1]     # shape: [1, seq_len_ext, hidden_size]

print(f"base 第一层输出形状: {hidden_base_layer1.shape}")
print(f"extended 第一层输出形状: {hidden_ext_layer1.shape}")

# 5. 比较前 base_len 个 token 的 hidden state
base_len = hidden_base_layer1.size(1)
hidden_ext_prefix = hidden_ext_layer1[:, :base_len, :]   # 取前 base_len 个 token

# 计算绝对差异
diff = (hidden_base_layer1 - hidden_ext_prefix).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()

print(f"\n最大绝对差异: {max_diff:.6e}")
print(f"平均绝对差异: {mean_diff:.6e}")

# 6. 判断是否相同（考虑浮点误差）
if max_diff < 1e-5:
    print("✅ 验证通过：增加一个 token 后，前几个 token 在第一层的输出（即第二层的输入）数值完全相同。")
else:
    print("❌ 验证失败：存在明显差异，可能原因包括位置编码或层归一化依赖全局统计。")
    # 为了调试，打印第一个 token 的前几个维度
    print("\n示例：第一个 token 前 5 个维度")
    print("base:     ", hidden_base_layer1[0, 0, :5])
    print("extended: ", hidden_ext_prefix[0, 0, :5])