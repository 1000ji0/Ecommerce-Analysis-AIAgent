"""
llm_factory.py
LLM 인스턴스 팩토리

.env 설정으로 LLM 전환:
  USE_CUSTOM_LLM=false (기본) → Gemini
  USE_CUSTOM_LLM=true         → 커스텀 LLM (Ollama 등)

커스텀 LLM 설정:
  CUSTOM_LLM_BASE_URL=http://localhost:11434/v1
  CUSTOM_LLM_MODEL=mju-ecommerce-sft
  CUSTOM_LLM_API_KEY=ollama
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
import sys

from pydantic import SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GEMINI_MODEL, GOOGLE_API_KEY


@lru_cache(maxsize=1)
def get_llm():
    use_custom = os.getenv("USE_CUSTOM_LLM", "false").lower() == "true"
    if use_custom:
        return _build_custom_llm()
    return _build_gemini()


def _build_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def _build_custom_llm():
    from langchain_openai import ChatOpenAI

    base_url = os.getenv("CUSTOM_LLM_BASE_URL", "http://localhost:11434/v1")
    model    = os.getenv("CUSTOM_LLM_MODEL",    "mju-ecommerce-sft")
    api_key  = SecretStr(os.getenv("CUSTOM_LLM_API_KEY", "ollama"))

    print(f"[LLM] 커스텀 LLM 사용: {model} @ {base_url}")

    try:
        return ChatOpenAI(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_completion_tokens=4096,
        )
    except TypeError:
        return ChatOpenAI(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=0.1,
        )


def get_llm_info() -> dict:
    use_custom = os.getenv("USE_CUSTOM_LLM", "false").lower() == "true"
    if use_custom:
        return {
            "provider": "custom",
            "model":    os.getenv("CUSTOM_LLM_MODEL", "mju-ecommerce-sft"),
            "base_url": os.getenv("CUSTOM_LLM_BASE_URL", "http://localhost:11434/v1"),
        }
    return {
        "provider": "google",
        "model":    GEMINI_MODEL,
    }