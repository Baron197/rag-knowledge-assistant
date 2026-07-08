"""LLM (answer generation) providers behind one interface, with token usage
surfaced for cost accounting.

- `OpenAILLM` calls the real chat-completions API (paid, needs a key).
- `HuggingFaceLLM` runs a Hugging Face instruct model **locally** via the
  `transformers` library -- free, no API key, downloads the model once.
- `FakeLLM` produces a deterministic, grounded-looking answer built from the
  retrieved context (and correctly refuses when there is no context) so the full
  request path works with no API key and no model download.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import Settings, resolve_hf_device


@dataclass
class LLMResult:
    """A generation result: the answer `text`, token counts, and the `model` name
    (everything downstream needs to compute cost; local models cost $0)."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


class LLM(Protocol):
    """Structural interface every LLM provider must satisfy."""

    def complete(self, system: str, user: str) -> LLMResult:
        """Generate an answer given a system prompt and a user prompt."""
        ...


class FakeLLM:
    """Deterministic, keyless stand-in LLM.

    Not "smart" -- it echoes the first retrieved passage as a grounded-looking
    answer, and returns the exact refusal sentence when no context was provided.
    Enough to exercise the entire pipeline and tests offline.
    """

    model = "fake-llm"

    def complete(self, system: str, user: str) -> LLMResult:
        """Return a grounded stand-in answer, or the refusal sentence if no context."""
        # Match the full "no context" block emitted by build_user_prompt, not a
        # bare substring — otherwise a question or a retrieved passage that happens
        # to contain this phrase would spuriously trigger the refusal path.
        has_context = "Context passages:\n\n(no relevant context found)\n\n" not in user
        if not has_context:
            text = "I don't have enough information in the documentation to answer that."
        else:
            # Echo the first context passage ([1]) as a stand-in grounded answer.
            snippet = ""
            if "[1]" in user:
                after = user.split("[1]", 1)[1]
                body = after.split("\n", 1)[1] if "\n" in after else after
                snippet = body.split("\n\n", 1)[0].strip()[:280]
            text = f"{snippet} [1]" if snippet else "Based on the documentation. [1]"
        # Rough token accounting (word counts) so the fake path still reports usage.
        pt = len((system + user).split())
        ct = len(text.split())
        return LLMResult(text=text, prompt_tokens=pt, completion_tokens=ct, model=self.model)


class HuggingFaceLLM:
    """Local, free text generation via Hugging Face `transformers` -- no API key.

    Loads an instruct model once (downloaded on first use, then cached), formats
    the messages with the model's chat template, and generates greedily
    (deterministic). Runs on CPU by default; set HF_DEVICE to a GPU index to use a
    GPU. Cost is always $0 because it runs on your machine.
    """

    def __init__(self, settings: Settings) -> None:
        # Imported lazily so the keyless path never needs torch/transformers.
        from transformers import pipeline

        self.model = settings.hf_llm_model
        self.max_new_tokens = settings.hf_max_new_tokens
        # Same device string for both providers ("cpu", "cuda:0", ...).
        self._pipe = pipeline(
            "text-generation", model=self.model, torch_dtype="auto",
            device=resolve_hf_device(settings.hf_device),
        )
        self._tok = self._pipe.tokenizer

    def complete(self, system: str, user: str) -> LLMResult:
        """Build a chat prompt, generate greedily, and return text + token counts."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Most instruct models ship a chat template; fall back to a plain prompt.
        if getattr(self._tok, "chat_template", None):
            prompt = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{system}\n\n{user}\n\nAnswer:"

        out = self._pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,            # greedy -> reproducible for evaluation
            return_full_text=False,     # return only the newly generated text
        )
        text = (out[0]["generated_text"] or "").strip()
        prompt_tokens = len(self._tok.encode(prompt))
        completion_tokens = len(self._tok.encode(text))
        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
        )


class OpenAILLM:
    """Real answer generation via the OpenAI chat-completions API (paid)."""

    def __init__(self, settings: Settings) -> None:
        # Lazy import so the keyless path never needs the openai package/key.
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for llm_provider=openai. "
                "Set it in .env, or use LLM_PROVIDER=fake (offline) or "
                "LLM_PROVIDER=hf (free local Hugging Face) instead."
            )
        # Timeout + retries so a transient API failure or rate limit doesn't
        # fail the query outright (the SDK retries with exponential backoff).
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_s,
            max_retries=settings.openai_max_retries,
        )
        self.model = settings.openai_llm_model

    def complete(self, system: str, user: str) -> LLMResult:
        """Call the chat API at temperature 0 and return text + token usage."""
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,  # deterministic -> stable evaluation
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        return LLMResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            model=self.model,
        )


def get_llm(settings: Settings) -> LLM:
    """Factory: pick the LLM provider named in settings."""
    if settings.llm_provider == "openai":
        return OpenAILLM(settings)
    if settings.llm_provider == "hf":
        return HuggingFaceLLM(settings)
    return FakeLLM()
