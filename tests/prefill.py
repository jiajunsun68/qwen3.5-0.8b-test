from __future__ import annotations

import gc
import html
import sys
from pathlib import Path

import torch
from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
	QApplication,
	QFrame,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QMainWindow,
	QMessageBox,
	QPushButton,
	QPlainTextEdit,
	QSizePolicy,
	QTextEdit,
	QVBoxLayout,
	QWidget,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parents[1] / "qwen"
SYSTEM_PROMPT = "你是一个智能助手，名叫傻妞。我叫小智。请用友好、热情的语气回答我的问题。"


class GenerationWorker(QObject):
	tokenReady = pyqtSignal(int, str, object, object)
	finished = pyqtSignal(str)
	error = pyqtSignal(str)

	def __init__(
		self,
		model: object,
		tokenizer: object,
		device: torch.device,
		past_key_values: object,
		logits: torch.Tensor,
		max_new_tokens: int = 1024,
	) -> None:
		super().__init__()
		self.model = model
		self.tokenizer = tokenizer
		self.device = device
		self.past_key_values = past_key_values
		self.logits = logits
		self.max_new_tokens = max_new_tokens
		self._stopped = False

	def stop(self) -> None:
		self._stopped = True

	def run(self) -> None:
		generated_text_parts: list[str] = []
		try:
			for _ in range(self.max_new_tokens):
				if self._stopped:
					break
				next_token_id = int(torch.argmax(self.logits, dim=-1).item())
				eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
				pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
				if next_token_id in {eos_token_id, pad_token_id}:
					break

				token_text = self.tokenizer.decode(
					[next_token_id],
					skip_special_tokens=False,
					clean_up_tokenization_spaces=False,
				)
				generated_text_parts.append(token_text)
				input_ids = torch.tensor([[next_token_id]], device=self.device)
				with torch.no_grad():
					outputs = self.model(
						input_ids=input_ids,
						past_key_values=self.past_key_values,
						use_cache=True,
					)
				self.past_key_values = outputs.past_key_values
				self.logits = outputs.logits[:, -1, :].detach()
				self.tokenReady.emit(next_token_id, token_text, self.past_key_values, self.logits)
				if self._stopped:
					break
			self.finished.emit("".join(generated_text_parts))
		except Exception as exc:  # pragma: no cover - worker error path
			self.error.emit(str(exc))


class TokenWorkbench(QWidget):
	def __init__(self) -> None:
		super().__init__()
		self._updating_input = False
		self._generation_active = False
		self._generation_thread: QThread | None = None
		self._generation_worker: GenerationWorker | None = None
		self._chat_events: list[str] = []
		self._conversation_messages: list[dict[str, str]] = []
		self._draft_token_ids: list[int] = []
		self._committed_token_ids: list[int] = []
		self._draft_user_text: str = ""
		self._state_stack: list[tuple[object, torch.Tensor]] = []
		self._load_error: str | None = None

		self._build_ui()
		self._load_model()

	# UI
	def _build_ui(self) -> None:
		self.setWindowTitle("Qwen token prefill workbench")
		self.resize(1280, 960)
		self.setStyleSheet(
			"""
			QWidget {
				background: #0f1218;
				color: #e8ecf1;
				font-family: "Microsoft YaHei UI";
				font-size: 13px;
			}
			QLabel#SectionTitle {
				color: #8fd3ff;
				font-size: 14px;
				font-weight: 600;
				letter-spacing: 0.4px;
			}
			QGroupBox {
				border: 1px solid #273040;
				border-radius: 14px;
				margin-top: 12px;
				padding: 10px;
				background: #121722;
			}
			QGroupBox::title {
				subcontrol-origin: margin;
				left: 12px;
				padding: 0 6px;
				color: #8fd3ff;
				font-weight: 600;
			}
			QTextEdit, QPlainTextEdit, QLineEdit {
				background: #151b26;
				border: 1px solid #2a3446;
				border-radius: 10px;
				padding: 8px;
				selection-background-color: #4c82ff;
			}
			QTextEdit:focus, QPlainTextEdit:focus, QLineEdit:focus {
				border: 1px solid #57a8ff;
			}
			QPushButton {
				background: #222b3b;
				border: 1px solid #334058;
				border-radius: 10px;
				padding: 8px 12px;
				color: #e8ecf1;
			}
			QPushButton:hover {
				background: #2c3951;
				border-color: #4c82ff;
			}
			QPushButton:pressed {
				background: #1d2534;
			}
			"""
		)

		root_layout = QVBoxLayout(self)
		root_layout.setSpacing(12)
		root_layout.setContentsMargins(14, 14, 14, 14)

		chat_title = QLabel("Chat Area")
		chat_title.setObjectName("SectionTitle")
		root_layout.addWidget(chat_title)

		self.chat_view = QTextEdit()
		self.chat_view.setReadOnly(True)
		self.chat_view.setMinimumHeight(300)
		self.chat_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		root_layout.addWidget(self.chat_view, stretch=1)

		state_box = QGroupBox("Model State")
		state_layout = QVBoxLayout(state_box)
		self.model_state_view = QPlainTextEdit()
		self.model_state_view.setReadOnly(True)
		self.model_state_view.setMinimumHeight(140)
		state_layout.addWidget(self.model_state_view)
		root_layout.addWidget(state_box)

		input_frame = QFrame()
		input_frame.setFrameShape(QFrame.StyledPanel)
		input_frame.setStyleSheet(
			"QFrame { background: #121722; border: 1px solid #273040; border-radius: 14px; }"
		)
		input_layout = QVBoxLayout(input_frame)
		input_layout.setContentsMargins(12, 12, 12, 12)
		input_layout.setSpacing(10)

		model_title = QLabel("Model Input Area")
		model_title.setObjectName("SectionTitle")
		input_layout.addWidget(model_title)

		logits_box = QGroupBox("Top 7 Logits")
		logits_layout = QVBoxLayout(logits_box)
		self.logits_view = QPlainTextEdit()
		self.logits_view.setReadOnly(True)
		self.logits_view.setMinimumHeight(160)
		logits_layout.addWidget(self.logits_view)
		input_layout.addWidget(logits_box)

		token_row = QHBoxLayout()
		self.minus_button = QPushButton("-")
		self.minus_button.setFixedWidth(56)
		self.minus_button.clicked.connect(self._rollback_last_token)
		token_row.addWidget(self.minus_button)

		self.input_edit = QLineEdit()
		self.input_edit.setPlaceholderText("输入内容，完整 token 会自动进入模型")
		self.input_edit.textChanged.connect(self._on_input_changed)
		token_row.addWidget(self.input_edit, stretch=1)

		self.send_button = QPushButton("发送")
		self.send_button.setFixedWidth(72)
		self.send_button.clicked.connect(self._send_user_message)
		token_row.addWidget(self.send_button)
		input_layout.addLayout(token_row)

		root_layout.addWidget(input_frame)

	# Model lifecycle
	def _load_model(self) -> None:
		try:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
			self.tokenizer = AutoTokenizer.from_pretrained(
				str(MODEL_DIR),
				trust_remote_code=True,
				use_fast=True,
				local_files_only=True,
			)
			self.model = AutoModelForCausalLM.from_pretrained(
				str(MODEL_DIR),
				torch_dtype=self.dtype,
				trust_remote_code=True,
				local_files_only=True,
			)
			self.model.to(self.device)
			self.model.eval()
			self._conversation_messages = [
				{"role": "system", "content": SYSTEM_PROMPT},
			]
			self._draft_token_ids = []
			self._draft_user_text = ""
			self._rebuild_draft_state()
			self._append_event("系统模型已加载")
			self._refresh_views()
		except Exception as exc:  # pragma: no cover - UI error path
			self._load_error = str(exc)
			self.input_edit.setEnabled(False)
			self.minus_button.setEnabled(False)
			self.send_button.setEnabled(False)
			self._append_event(f"模型加载失败: {html.escape(self._load_error)}")
			self._refresh_views()
			QMessageBox.critical(self, "加载失败", f"无法加载模型：\n{exc}")

	def _prompt_history_messages(self) -> list[dict[str, str]]:
		messages = list(self._conversation_messages)
		if messages and messages[-1]["role"] == "assistant" and messages[-1]["content"] == "生成中...":
			messages = messages[:-1]
		return messages

	def _build_open_user_prompt_text(self) -> str:
		messages = self._prompt_history_messages()
		if messages and messages[0]["role"] == "system":
			system_content = messages[0]["content"]
			base_prompt = f"<|im_start|>system\n{system_content}<|im_end|>\n"
		else:
			base_prompt = ""
		return base_prompt + "<|im_start|>user\n" + self._draft_user_text

	def _build_generation_prompt_text(self) -> str:
		messages = self._prompt_history_messages()
		return self.tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
		)

	def _rebuild_draft_state(self) -> None:
		if self._load_error is not None:
			return
		prompt_text = self._build_open_user_prompt_text()
		encoded = self.tokenizer(
			prompt_text,
			add_special_tokens=False,
			return_tensors="pt",
		)
		input_ids = encoded["input_ids"].to(self.device)
		with torch.no_grad():
			outputs = self.model(input_ids=input_ids, use_cache=True)
		self._state_stack = [(outputs.past_key_values, outputs.logits[:, -1, :].detach())]
		self._committed_token_ids = list(encoded["input_ids"][0].tolist())

	def _rebuild_generation_state(self) -> None:
		prompt_text = self._build_generation_prompt_text()
		encoded = self.tokenizer(
			prompt_text,
			add_special_tokens=False,
			return_tensors="pt",
		)
		input_ids = encoded["input_ids"].to(self.device)
		with torch.no_grad():
			outputs = self.model(input_ids=input_ids, use_cache=True)
		self._state_stack = [(outputs.past_key_values, outputs.logits[:, -1, :].detach())]
		self._committed_token_ids = list(encoded["input_ids"][0].tolist())

	def _release_kv_cache(self) -> None:
		self._state_stack.clear()
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def closeEvent(self, event) -> None:
		if self._generation_worker is not None:
			self._generation_worker.stop()
		if self._generation_thread is not None:
			self._generation_thread.quit()
			self._generation_thread.wait(3000)
		try:
			if hasattr(self, "model") and self.model is not None:
				self.model.to("cpu")
		except Exception:
			pass
		self._state_stack.clear()
		self._draft_token_ids.clear()
		self._committed_token_ids.clear()
		self._conversation_messages.clear()
		self._chat_events.clear()
		del self._state_stack
		del self._draft_token_ids
		del self._committed_token_ids
		del self._conversation_messages
		del self._chat_events
		del self.model
		del self.tokenizer
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		super().closeEvent(event)

	# Conversation / generation
	def _append_event(self, message: str) -> None:
		self._chat_events.append(message)
		if len(self._chat_events) > 200:
			self._chat_events = self._chat_events[-200:]

	def _add_message(self, role: str, content: str) -> None:
		self._conversation_messages.append({"role": role, "content": content})
		if len(self._conversation_messages) > 200:
			self._conversation_messages = self._conversation_messages[-200:]

	def _current_prefill_text(self) -> str:
		return self.tokenizer.decode(
			self._committed_token_ids,
			skip_special_tokens=False,
			clean_up_tokenization_spaces=False,
		)

	def _decode_token_for_display(self, token_id: int) -> str:
		token_text = self.tokenizer.decode(
			[token_id],
			skip_special_tokens=False,
			clean_up_tokenization_spaces=False,
		)
		return token_text.replace("\n", "\\n").replace("\r", "\\r")

	def _tokenize_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
		try:
			encoding = self.tokenizer(
				text,
				add_special_tokens=False,
				return_offsets_mapping=True,
				return_tensors=None,
			)
			return list(encoding["input_ids"]), list(encoding["offset_mapping"])
		except Exception:
			token_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
			offsets: list[tuple[int, int]] = []
			cursor = 0
			for token_id in token_ids:
				token_piece = self._decode_token_for_display(token_id)
				start = cursor
				cursor += len(token_piece)
				offsets.append((start, cursor))
			return token_ids, offsets

	def _commit_token(self, token_id: int, token_text: str) -> None:
		self._draft_token_ids.append(token_id)
		self._draft_user_text += token_text
		self._append_event(f"提交 token: {html.escape(token_text)}")

	def _process_pending_text(self, current_text: str | None = None) -> None:
		if self._load_error is not None:
			return
		remaining = self.input_edit.text() if current_text is None else current_text
		if not remaining:
			self._rebuild_draft_state()
			self._refresh_views()
			return
		updated = False
		while remaining:
			token_ids, offsets = self._tokenize_with_offsets(remaining)
			if not token_ids or not offsets:
				break
			start, end = offsets[0]
			if end <= start:
				break
			token_id = token_ids[0]
			token_text = remaining[start:end]
			self._commit_token(token_id, token_text)
			remaining = remaining[end:]
			updated = True
		if updated and remaining != self.input_edit.text():
			self._set_input_text(remaining, process=False)
		if updated:
			self._rebuild_draft_state()
		self._refresh_views()

	def _rollback_last_token(self) -> None:
		if not self._draft_token_ids:
			return
		removed_token_id = self._draft_token_ids.pop()
		removed_text = self._decode_token_for_display(removed_token_id)
		if self._draft_user_text.endswith(removed_text):
			self._draft_user_text = self._draft_user_text[: -len(removed_text)]
		self._set_input_text(removed_text + self.input_edit.text(), process=False)
		self._append_event(f"回退 token: {html.escape(removed_text)}")
		self._release_kv_cache()
		self._rebuild_draft_state()
		self._refresh_views()

	def _set_input_text(self, text: str, process: bool = True) -> None:
		self._updating_input = True
		self.input_edit.setText(text)
		self._updating_input = False
		if process:
			self._process_pending_text(self.input_edit.text())

	def _on_input_changed(self, text: str) -> None:
		if self._updating_input:
			return
		self._process_pending_text(text)

	def _send_user_message(self) -> None:
		if self._generation_active:
			return
		pending_text = self.input_edit.text()
		message = self._draft_user_text + pending_text
		if not message.strip() or self._load_error is not None:
			return
		self._add_message("user", message)
		self._append_event(f"发送 user: {html.escape(message)}")
		self._set_input_text("", process=False)
		self._draft_token_ids.clear()
		self._draft_user_text = ""
		self._refresh_views()
		self._add_message("assistant", "生成中...")
		self._refresh_views()
		QTimer.singleShot(0, self._start_generation)

	def _set_input_controls_enabled(self, enabled: bool) -> None:
		self.input_edit.setEnabled(enabled)
		self.minus_button.setEnabled(enabled)
		self.send_button.setEnabled(enabled)

	def _start_generation(self) -> None:
		if self._generation_active or self._load_error is not None or not self._state_stack:
			return
		self._generation_active = True
		self._set_input_controls_enabled(False)
		self._rebuild_generation_state()
		prompt_cache, prompt_logits = self._state_stack[-1]
		self._generation_thread = QThread(self)
		self._generation_worker = GenerationWorker(
			self.model,
			self.tokenizer,
			self.device,
			prompt_cache,
			prompt_logits,
		)
		self._generation_worker.moveToThread(self._generation_thread)
		self._generation_thread.started.connect(self._generation_worker.run)
		self._generation_worker.tokenReady.connect(self._on_generation_token)
		self._generation_worker.finished.connect(self._on_generation_finished)
		self._generation_worker.error.connect(self._on_generation_error)
		self._generation_worker.finished.connect(self._generation_thread.quit)
		self._generation_worker.error.connect(self._generation_thread.quit)
		self._generation_thread.finished.connect(self._cleanup_generation_worker)
		self._generation_thread.start()

	def _assistant_message_index(self) -> int | None:
		for index in range(len(self._conversation_messages) - 1, -1, -1):
			if self._conversation_messages[index]["role"] == "assistant":
				return index
		return None

	def _append_to_assistant_message(self, token_text: str) -> None:
		assistant_index = self._assistant_message_index()
		if assistant_index is None:
			return
		current_content = self._conversation_messages[assistant_index]["content"]
		if current_content == "生成中...":
			self._conversation_messages[assistant_index]["content"] = token_text
		else:
			self._conversation_messages[assistant_index]["content"] = current_content + token_text

	def _on_generation_token(self, token_id: int, token_text: str, cache: object, logits: object) -> None:
		self._committed_token_ids.append(token_id)
		self._state_stack.append((cache, logits))
		self._append_to_assistant_message(token_text)
		self._refresh_views()

	def _on_generation_finished(self, generated_text: str) -> None:
		assistant_index = self._assistant_message_index()
		if assistant_index is not None:
			current_content = self._conversation_messages[assistant_index]["content"]
			if not generated_text and current_content == "生成中...":
				self._conversation_messages[assistant_index]["content"] = ""
		self._append_event("assistant 生成完成")
		self._refresh_views()
		self._generation_active = False
		self._set_input_controls_enabled(True)

	def _on_generation_error(self, message: str) -> None:
		self._append_event(f"生成失败: {html.escape(message)}")
		assistant_index = self._assistant_message_index()
		if assistant_index is not None and not self._conversation_messages[assistant_index]["content"]:
			self._conversation_messages[assistant_index]["content"] = "生成失败"
		self._refresh_views()
		self._generation_active = False
		self._set_input_controls_enabled(True)

	def _cleanup_generation_worker(self) -> None:
		if self._generation_worker is not None:
			self._generation_worker.deleteLater()
			self._generation_worker = None
		if self._generation_thread is not None:
			self._generation_thread.deleteLater()
			self._generation_thread = None

	def _commit_text_to_state(self, text: str) -> None:
		for token_id in self.tokenizer.encode(text, add_special_tokens=False):
			self._commit_token(token_id, self._decode_token_for_display(token_id))
		self._rebuild_draft_state()

	# Rendering
	def _render_top7_logits(self) -> str:
		if not self._state_stack:
			return ""
		logits = self._state_stack[-1][1][0]
		top_k = min(7, logits.shape[-1])
		top_values, top_indices = torch.topk(logits, top_k)
		probs = torch.softmax(logits.float(), dim=-1)
		lines = []
		for rank, (token_id, value) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
			token_text = self._decode_token_for_display(token_id)
			probability = probs[token_id].item()
			lines.append(
				f"{rank:>2}. {token_text:<24} p={probability:.6f}  logit={value:.6f}  id={token_id}"
			)
		return "\n".join(lines)

	def _render_message_block(self, role: str, content: str) -> str:
		avatar = "SYS" if role == "system" else ("YOU" if role == "user" else "AI")
		content_html = html.escape(content).replace("\n", "<br>")
		if role == "user":
			return f"""
			<table width="100%" cellspacing="0" cellpadding="0" style="margin: 8px 0 14px 0;">
				<tr>
					<td style="max-width: 78%;">
						<table cellspacing="0" cellpadding="0" align="right">
							<tr>
								<td style="background: #1b2331; border: 1px solid #334058; border-radius: 14px; padding: 10px 12px; color: #e8ecf1; white-space: pre-wrap; text-align: left;">{content_html}</td>
								<td style="width: 10px;"></td>
								<td style="width: 44px; height: 44px; border-radius: 22px; background: #4c82ff; color: white; text-align: center; vertical-align: middle; font-weight: 700;">{avatar}</td>
							</tr>
						</table>
					</td>
					<td style="width: 100%;"></td>
				</tr>
			</table>
			"""
		if role == "assistant":
			return f"""
			<table width="100%" cellspacing="0" cellpadding="0" style="margin: 8px 0 14px 0;">
				<tr>
					<td style="max-width: 78%;">
						<table cellspacing="0" cellpadding="0">
							<tr>
								<td style="width: 44px; height: 44px; border-radius: 22px; background: #89e7a6; color: #07111a; text-align: center; vertical-align: middle; font-weight: 700;">{avatar}</td>
								<td style="width: 10px;"></td>
								<td style="background: #151b26; border: 1px solid #2a3446; border-radius: 14px; padding: 10px 12px; color: #e8ecf1; white-space: pre-wrap; text-align: left;">{content_html}</td>
							</tr>
						</table>
					</td>
					<td style="width: 100%;"></td>
				</tr>
			</table>
			"""
		return f"""
		<div style="text-align: center; margin: 10px 0 14px 0; color: #8fd3ff; font-weight: 700;">system prompt：{content_html}</div>
		"""

	def _refresh_views(self) -> None:
		self.logits_view.setPlainText(self._render_top7_logits())
		self.model_state_view.setPlainText(self._render_model_state())

		chat_html_parts = [
			self._render_message_block(message["role"], message["content"])
			for message in self._conversation_messages
		]
		html_text = f"""
		<div style="font-family: 'Microsoft YaHei UI'; line-height: 1.5;">
			{''.join(chat_html_parts)}
		</div>
		"""
		self.chat_view.setHtml(html_text)
		scrollbar = self.chat_view.verticalScrollBar()
		scrollbar.setValue(scrollbar.maximum())

	def _render_model_state(self) -> str:
		if not self._state_stack:
			return ""
		past_key_values = self._state_stack[-1][0]
		layer_count = len(past_key_values)
		seq_len = int(past_key_values.get_seq_length()) if hasattr(past_key_values, "get_seq_length") else 0
		layer_names = []
		if hasattr(past_key_values, "layers"):
			layer_names = [layer.__class__.__name__ for layer in past_key_values.layers]
		lines = [
			f"layers: {layer_count}",
			f"cache seq len: {seq_len}",
			f"cache layer types: {layer_names}",
			f"token ids: {self._committed_token_ids}",
			"decoded tokens:",
		]
		for index, token_id in enumerate(self._committed_token_ids, start=1):
			visible = self._decode_token_for_display(token_id).replace("\n", "\\n").replace("\r", "\\r")
			lines.append(f"{index:>4}. id={token_id} token={visible}")
		return "\n".join(lines)


def main() -> int:
	app = QApplication(sys.argv)
	window = QMainWindow()
	workbench = TokenWorkbench()
	window.setCentralWidget(workbench)
	window.resize(1280, 960)
	window.show()
	return app.exec_()


if __name__ == "__main__":
	raise SystemExit(main())
