"""
Stage2/llm.py
-------------
LLM 调用封装 + token 用量日志。
"""

from __future__ import annotations
import os
import time
from typing import Optional
from openai import OpenAI
from config.config import OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE, OPENAI_MODEL, TOKEN_LOG_ENABLED, TOKEN_LOG_PATH2
import tiktoken
import json as _json


def token_count(
            messages,
            model="gpt-4o"
        ):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    tokens = len(encoding.encode(messages))
    return tokens

def evaluator(prompt: str, model: str = OPENAI_MODEL, token_log_path: str = TOKEN_LOG_PATH2) -> str:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    completion = client.chat.completions.create(
        model=model,
        stream=False,
        messages=[
            {"role": "system", "content": "You are a patent text parsing assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE
    )
    try:
        usage = {}
        if hasattr(completion, 'usage') and completion.usage is not None:
            # some SDKs provide usage as dict-like
            usage = {k: int(v) for k, v in completion.usage.items()}
        else:
            # fallback: try dict access
            usage = dict(getattr(completion, 'usage', {}) or {})
    except Exception:
        usage = {}

    # Append token log as JSONL
    try:
        if token_log_path:
            os.makedirs(os.path.dirname(token_log_path), exist_ok=True)
            record = {
                'ts': int(time.time()),
                'model': model,
                'input_tokens': token_count(prompt),
                'output_tokens': token_count(completion.choices[0].message.content),
                'usage': usage,
                'prompt_snippet': prompt[:200],
                
            }
            with open(token_log_path, 'a', encoding='utf-8') as fh:
                fh.write(_json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        pass

    return completion.choices[0].message.content

def generator(prompt: str, model: str = OPENAI_MODEL, token_log_path: str = TOKEN_LOG_PATH2) -> str:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    completion = client.chat.completions.create(
        model=model,
        stream=False,
        messages=[
            {"role": "system", "content": "You are a patent text parsing assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE
    )
    try:
        usage = {}
        if hasattr(completion, 'usage') and completion.usage is not None:
            # some SDKs provide usage as dict-like
            usage = {k: int(v) for k, v in completion.usage.items()}
        else:
            # fallback: try dict access
            usage = dict(getattr(completion, 'usage', {}) or {})
    except Exception:
        usage = {}

    try:
        if TOKEN_LOG_ENABLED:
            os.makedirs(os.path.dirname(token_log_path), exist_ok=True)
            record = {
                'ts': int(time.time()),
                'model': model,
                'input_tokens': token_count(prompt),
                'output_tokens': token_count(completion.choices[0].message.content),
                'usage': usage,
                'prompt_snippet': prompt[:200],
                
            }
            with open(token_log_path, 'a', encoding='utf-8') as fh:
                fh.write(_json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        pass

    return completion.choices[0].message.content

class LLMInterface:
    def __init__(self, model: str = OPENAI_MODEL) -> None:
        self.model = model

    def call(self, prompt: str, model: Optional[str] = None) -> str:
        try:
            return generator(prompt) or ""
        except Exception as exc:
            print(f"[LLMInterface] failed: {exc}")
            return ""
