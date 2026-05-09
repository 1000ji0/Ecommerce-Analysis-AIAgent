"""Simple HTTP client for local SLLM/RAG service

Sends POST requests to the SLLM endpoint and returns parsed JSON.
"""
from __future__ import annotations

import typing
import requests
import uuid

SLLM_URL = "http://localhost:8000/chat"
DEFAULT_TIMEOUT = 10.0


def query_sllm(query: str, session_id: typing.Optional[str] = None, timeout: float = DEFAULT_TIMEOUT) -> dict:
    if session_id is None:
        session_id = str(uuid.uuid4())
    payload = {"query": query, "session_id": session_id}
    resp = requests.post(SLLM_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}
