"""Optional OpenAI answer refinement over verified intelligence context."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from urllib import error, request

from sales_forecasting.data.intelligence import IntelligenceAnswer


DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"


def refine_answer(question: str, answer: IntelligenceAnswer) -> IntelligenceAnswer:
    """Use OpenAI only when configured; otherwise preserve the grounded answer."""
    api_key = os.getenv("SALES_LLM_API_KEY")
    if not api_key:
        return answer

    payload = {
        "model": os.getenv("SALES_LLM_MODEL", DEFAULT_MODEL),
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a sales intelligence assistant. Answer only from the supplied verified "
                    "context. Do not invent metrics, opportunities, policies, or sources. If the context "
                    "does not support an answer, say so clearly. Keep the answer concise and actionable."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "verified_context": answer.as_dict()},
                    ensure_ascii=True,
                ),
            },
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        os.getenv("SALES_LLM_BASE_URL", DEFAULT_BASE_URL),
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"].strip()
    except (error.HTTPError, error.URLError, TimeoutError, KeyError, IndexError, ValueError):
        return answer
    if not content:
        return answer
    return replace(
        answer,
        answer=content,
        sources=[*answer.sources, {"type": "llm", "source": payload["model"]}],
    )