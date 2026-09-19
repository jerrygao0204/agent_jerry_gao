# ⚠️ [已废弃 / DEPRECATED] 本模块是切换到 vllm+litellm 网关统一架构之前的
# 直连 vLLM（MODEL_BACKEND=local/vllm/multi_vllm）实现，现已被 factory/model_factory.py
# 完全取代。生产代码中没有任何一处引用本模块，目前仅 tests/test_model_factory_backends_smoke.py
# 还在 import 它（该测试本身也已过时，将在测试清理阶段一并处理）。
# 待测试清理完成后，本文件应整体删除，请勿在新代码中引入依赖。
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import requests
import torch
import yaml


logger = logging.getLogger(__name__)


class BaseChatHandle(ABC):
    @abstractmethod
    def chat(
        self,
        messages,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        stream: bool = False,
        **extra_kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class VLLMEndpointRouter:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ vLLM 端点配置文件不存在: {config_path}")
        with open(config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError("❌ vLLM 端点配置格式错误，需为 YAML 对象")
        return data

    def resolve(self, backend: str, role: str = "llm") -> Tuple[str, str]:
        profiles = self._config.get("profiles", {})
        if backend not in profiles:
            raise KeyError(
                f"❌ vLLM 配置缺失 backend='{backend}'，可用: {list(profiles.keys())}"
            )

        backend_profile = profiles.get(backend, {}) or {}
        role_cfg = backend_profile.get(role)
        if role_cfg is None and role != "llm":
            role_cfg = backend_profile.get("llm")
        if role_cfg is None:
            raise KeyError(
                f"❌ vLLM 配置缺失 role='{role}'，backend='{backend}'"
            )

        base_url = (role_cfg.get("base_url") or "").rstrip("/")
        model = role_cfg.get("model")

        if not base_url or not model:
            raise ValueError(
                f"❌ vLLM role 配置不完整，backend='{backend}', role='{role}' 需包含 base_url 与 model"
            )
        return base_url, model


def _normalize_messages(messages, role_hint: str = "llm") -> List[Dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    if not isinstance(messages, list):
        raise TypeError("messages 必须为字符串或消息列表")

    normalized = []
    for index, item in enumerate(messages):
        if not isinstance(item, dict):
            raise TypeError(f"messages[{index}] 必须是字典")

        role = item.get("role") or "user"
        content = item.get("content", "")

        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    raise TypeError("messages.content block 必须是字典")
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type in ("image", "file"):
                    raise NotImplementedError(
                        f"角色 '{role_hint}' 暂未接入多模态 block: {block_type}"
                    )
                else:
                    raise ValueError(f"未知 content block 类型: {block_type}")
            normalized.append({"role": role, "content": "".join(text_parts)})
            continue

        raise TypeError("messages.content 必须是字符串或 block 列表")

    return normalized


class LocalChatHandle(BaseChatHandle):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def chat(
        self,
        messages,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        stream: bool = False,
        **extra_kwargs,
    ):
        if stream:
            raise NotImplementedError("local handle.chat 暂不直接支持 stream=True，请走旧 generate+streamer 流程")

        normalized = _normalize_messages(messages, role_hint="llm")
        prompt = self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
        }
        generation_kwargs.update(extra_kwargs)

        outputs = self.model.generate(**inputs, **generation_kwargs)
        completion = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return completion

    def close(self):
        return None


class RemoteVLLMHandle(BaseChatHandle):
    _UNSUPPORTED_PARAMS = {
        "no_repeat_ngram_size",
    }

    def __init__(self, base_url: str, model: str, role: str = "llm", timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.role = role
        self.timeout = timeout
        self.session = requests.Session()
        self._health_check()

    def _health_check(self):
        health_url = f"{self.base_url}/health"
        models_url = f"{self.base_url}/v1/models"

        health_error = None
        try:
            response = self.session.get(health_url, timeout=self.timeout)
            if response.status_code == 200:
                logger.info("✅ vLLM 健康检查通过: %s", health_url)
                return
            health_error = f"{health_url} 返回 {response.status_code}"
        except Exception as exc:
            health_error = str(exc)

        try:
            response = self.session.get(models_url, timeout=self.timeout)
            if response.status_code == 200:
                logger.info("✅ vLLM 模型探测通过: %s", models_url)
                return
            fallback_error = f"{models_url} 返回 {response.status_code}"
        except Exception as exc:
            fallback_error = str(exc)

        raise ConnectionError(
            "❌ RemoteVLLMHandle 初始化失败：健康检查未通过。"
            f" health 错误: {health_error}; models 错误: {fallback_error}"
        )

    def _extract_and_warn_unsupported(self, extra_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        clean_kwargs = dict(extra_kwargs)
        for key in list(clean_kwargs.keys()):
            if key in self._UNSUPPORTED_PARAMS:
                logger.warning("⚠️ vLLM 后端不支持参数 '%s'，已显式丢弃。", key)
                clean_kwargs.pop(key, None)
        return clean_kwargs

    def _chat_stream(
        self,
        payload: Dict[str, Any],
    ) -> Generator[str, None, None]:
        url = f"{self.base_url}/v1/chat/completions"
        with self.session.post(url, json=payload, timeout=300, stream=True) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if not raw_line.startswith("data:"):
                    continue
                line = raw_line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                chunk = delta.get("content")
                if chunk:
                    yield chunk

    def chat(
        self,
        messages,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        stream: bool = False,
        **extra_kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        normalized = _normalize_messages(messages, role_hint=self.role)
        clean_kwargs = self._extract_and_warn_unsupported(extra_kwargs)

        payload = {
            "model": self.model,
            "messages": normalized,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        payload["top_p"] = 1.0 if not do_sample else clean_kwargs.pop("top_p", 0.95)

        if clean_kwargs:
            payload["extra_body"] = clean_kwargs

        if stream:
            return self._chat_stream(payload)

        url = f"{self.base_url}/v1/chat/completions"
        response = self.session.post(url, json=payload, timeout=300)
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"]

    def close(self):
        if self.session is not None:
            self.session.close()


class _SimpleBatchEncoding(dict):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, _device):
        return self


class LegacyTokenizerAdapter:
    def __init__(self, handle: RemoteVLLMHandle):
        self.handle = handle
        self.eos_token_id = 0
        self._last_template_text = None
        self._last_template_messages = None

    @staticmethod
    def _encode_text(text: str) -> List[int]:
        data = text.encode("utf-8")
        return [int(b) + 1 for b in data]

    @staticmethod
    def _decode_ids(ids: List[int]) -> str:
        byte_values = []
        for token_id in ids:
            if token_id <= 0:
                continue
            value = token_id - 1
            if 0 <= value <= 255:
                byte_values.append(value)
        return bytes(byte_values).decode("utf-8", errors="ignore")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        normalized = _normalize_messages(messages, role_hint=self.handle.role)
        lines = []
        for item in normalized:
            lines.append(f"<{item['role']}>: {item['content']}")
        if add_generation_prompt:
            lines.append("<assistant>:")
        rendered = "\n".join(lines)
        self._last_template_text = rendered
        self._last_template_messages = normalized
        if tokenize:
            return self._encode_text(rendered)
        return rendered

    def __call__(self, text, return_tensors="pt", **_kwargs):
        if isinstance(text, list):
            text = text[0]
        encoded = self._encode_text(text)
        input_ids = torch.tensor([encoded], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return _SimpleBatchEncoding(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self._decode_ids(token_ids)

    def recover_messages_from_prompt(self, prompt_text: str):
        if self._last_template_text == prompt_text and self._last_template_messages is not None:
            return self._last_template_messages
        return [{"role": "user", "content": prompt_text}]


class LegacyModelAdapter:
    def __init__(self, handle: RemoteVLLMHandle, tokenizer: LegacyTokenizerAdapter):
        self.handle = handle
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        streamer=None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ):
        prompt_text = ""
        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                ids = input_ids[0].tolist()
            else:
                ids = input_ids[0]
            prompt_text = self.tokenizer.decode(ids, skip_special_tokens=True)

        messages = self.tokenizer.recover_messages_from_prompt(prompt_text)

        if streamer is not None:
            stream_iter = self.handle.chat(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                stream=True,
                **kwargs,
            )
            for chunk in stream_iter:
                streamer.on_finalized_text(chunk, stream_end=False)
            streamer.on_finalized_text("", stream_end=True)
            return None

        content = self.handle.chat(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            stream=False,
            **kwargs,
        )

        prompt_ids = []
        if input_ids is not None:
            prompt_ids = input_ids[0].tolist() if isinstance(input_ids, torch.Tensor) else input_ids[0]

        output_ids = self.tokenizer._encode_text(content)
        full_ids = prompt_ids + output_ids
        return torch.tensor([full_ids], dtype=torch.long)


class LegacyTupleAdapter:
    def __init__(self, handle: RemoteVLLMHandle):
        self.tokenizer = LegacyTokenizerAdapter(handle)
        self.model = LegacyModelAdapter(handle, self.tokenizer)
