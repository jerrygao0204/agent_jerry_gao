import pytest
import torch

from factory.model_factory import ModelFactory
from factory.llm_backends import RemoteVLLMHandle


class _FakeLocalModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def generate(self, input_ids=None, **_kwargs):
        if input_ids is None:
            return torch.tensor([[1, 2, 3]], dtype=torch.long)
        tail = torch.tensor([[11, 12, 13]], dtype=torch.long)
        return torch.cat([input_ids, tail], dim=1)


class _FakeLocalBatch(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, _device):
        return self


class _FakeLocalTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "\n".join([f"{m['role']}:{m['content']}" for m in messages])
        if add_generation_prompt:
            text += "\nassistant:"
        return [1, 2, 3] if tokenize else text

    def __call__(self, text, return_tensors="pt", **_kwargs):
        if isinstance(text, list):
            text = text[0]
        ids = torch.tensor([[ord(ch) % 255 + 1 for ch in text]], dtype=torch.long)
        return _FakeLocalBatch(ids)

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        chars = [chr(max(0, tid - 1)) for tid in token_ids if tid > 0]
        return "".join(chars)


@pytest.fixture(autouse=True)
def _cleanup_factory_state():
    ModelFactory.destroy_llm_model()
    yield
    ModelFactory.destroy_llm_model()


def test_smoke_local_backend(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "local")

    fake_model = _FakeLocalModel()
    fake_tokenizer = _FakeLocalTokenizer()

    def _fake_build_local(self, llm_short_name):
        ModelFactory._LLM_MODEL = fake_model
        ModelFactory._LLM_TOKENIZER = fake_tokenizer
        return fake_model, fake_tokenizer

    monkeypatch.setattr(ModelFactory, "_build_local_llm_pair", _fake_build_local)

    factory = ModelFactory(prompt_hub_path="prompt_hub.yaml")
    model, tokenizer = factory.get_llm_model("Qwen/Qwen3-32B")

    assert model is fake_model
    assert tokenizer is fake_tokenizer


def test_smoke_vllm_backend(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "vllm")
    monkeypatch.setenv(
        "VLLM_ENDPOINTS_CONFIG",
        "/workspace/hf-conda/RAG/问答机器人/config/vllm_endpoints.yaml",
    )

    monkeypatch.setattr(RemoteVLLMHandle, "_health_check", lambda self: None)
    monkeypatch.setattr(
        RemoteVLLMHandle,
        "chat",
        lambda self, messages, max_new_tokens=1024, temperature=0.7, do_sample=True, stream=False, **kwargs: (
            "remote-ok" if not stream else iter(["remote", "-", "ok"])
        ),
    )

    factory = ModelFactory(prompt_hub_path="prompt_hub.yaml")
    model, tokenizer = factory.get_llm_model("Qwen/Qwen3-32B")

    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=16)
    text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    assert text == "remote-ok"


def test_smoke_multi_vllm_backend(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "multi_vllm")
    monkeypatch.setenv(
        "VLLM_ENDPOINTS_CONFIG",
        "/workspace/hf-conda/RAG/问答机器人/config/vllm_endpoints.yaml",
    )

    monkeypatch.setattr(RemoteVLLMHandle, "_health_check", lambda self: None)
    monkeypatch.setattr(
        RemoteVLLMHandle,
        "chat",
        lambda self, messages, max_new_tokens=1024, temperature=0.7, do_sample=True, stream=False, **kwargs: (
            f"multi-{self.model}" if not stream else iter(["multi"])
        ),
    )

    factory = ModelFactory(prompt_hub_path="prompt_hub.yaml")
    model, tokenizer = factory.get_llm_model("Qwen/Qwen3-32B")

    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    assert "multi-Qwen/Qwen3-32B" == text


def test_smoke_remote_failure_fallback_to_local(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "vllm")
    monkeypatch.setenv("MODEL_BACKEND_SAFE_FALLBACK", "1")
    monkeypatch.setenv(
        "VLLM_ENDPOINTS_CONFIG",
        "/workspace/hf-conda/RAG/问答机器人/config/vllm_endpoints.yaml",
    )

    fake_model = _FakeLocalModel()
    fake_tokenizer = _FakeLocalTokenizer()

    def _fake_build_local(self, llm_short_name):
        ModelFactory._LLM_MODEL = fake_model
        ModelFactory._LLM_TOKENIZER = fake_tokenizer
        return fake_model, fake_tokenizer

    monkeypatch.setattr(ModelFactory, "_build_local_llm_pair", _fake_build_local)
    monkeypatch.setattr(
        RemoteVLLMHandle,
        "_health_check",
        lambda self: (_ for _ in ()).throw(ConnectionError("remote down")),
    )

    factory = ModelFactory(prompt_hub_path="prompt_hub.yaml")
    model, tokenizer = factory.get_llm_model("Qwen/Qwen3-4B")

    assert model is fake_model
    assert tokenizer is fake_tokenizer
    assert ModelFactory._LLM_LOCAL_FALLBACK_ACTIVE is True
