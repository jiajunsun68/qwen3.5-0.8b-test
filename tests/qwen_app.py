"""这只是个简单的qwen大模型对话机器人实现"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


MODEL_DIR = Path(__file__).resolve().parents[1] / "qwen"
SYSTEM_PROMPT = "你是一个智能助手，名叫傻妞。我叫小智。请用友好、热情的语气回答我的问题。"


def load_model() -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dtype = torch.float16 if device.type == "cuda" else torch.float32

	tokenizer = AutoTokenizer.from_pretrained(
		str(MODEL_DIR),
		trust_remote_code=True,
		use_fast=True,
		local_files_only=True,
	)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token
	model = AutoModelForCausalLM.from_pretrained(
		str(MODEL_DIR),
		dtype=dtype,
		trust_remote_code=True,
		local_files_only=True,
	)
	if hasattr(model.config, "use_mamba_kernels"):
		model.config.use_mamba_kernels = False
	model.to(device)
	model.eval()
	return tokenizer, model, device


def stream_reply(
	tokenizer: AutoTokenizer,
	model: AutoModelForCausalLM,
	device: torch.device,
	messages: list[dict[str, str]],
	max_new_tokens: int = 256,
) -> str:
	input_text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
	)
	inputs = tokenizer(
		input_text,
		return_tensors="pt",
		add_special_tokens=False,
	).to(device)
	streamer = TextIteratorStreamer(
		tokenizer,
		skip_special_tokens=True,
		clean_up_tokenization_spaces=False,
	)

	generation_kwargs = {
		**inputs,
		"max_new_tokens": max_new_tokens,
		"do_sample": False,
		"use_cache": True,
		"pad_token_id": tokenizer.pad_token_id,
		"streamer": streamer,
	}
	thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
	thread.start()

	assistant_chunks: list[str] = []
	for chunk in streamer:
		assistant_chunks.append(chunk)
		sys.stdout.write(chunk)
		sys.stdout.flush()

	thread.join()
	print()
	return "".join(assistant_chunks).strip()


def main() -> None:
	tokenizer, model, device = load_model()

	messages = [
		{"role": "system", "content": SYSTEM_PROMPT},
	]

	print("Local Qwen chat started. Type 'q' or 'quit' to stop.\n")
	while True:
		try:
			user_text = input("You: ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\nBye.")
			break

		if not user_text:
			continue
		if user_text.lower() in {"q", "quit"}:
			print("Bye.")
			break

		messages.append({"role": "user", "content": user_text})
		sys.stdout.write("Qwen: ")
		sys.stdout.flush()
		assistant_text = stream_reply(tokenizer, model, device, messages)
		messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
	sys.exit(main())
