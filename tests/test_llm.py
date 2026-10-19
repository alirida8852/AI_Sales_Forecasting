import json

from sales_forecasting.data.intelligence import IntelligenceAnswer
from sales_forecasting.data.llm import refine_answer


def test_refine_answer_keeps_grounded_response_without_key(monkeypatch) -> None:
    monkeypatch.delenv("SALES_LLM_API_KEY", raising=False)
    answer = IntelligenceAnswer("analytics", "answered", "There are 3 deals.", {"count": 3}, [])

    assert refine_answer("How many deals?", answer) == answer


def test_refine_answer_uses_mocked_openai_response(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "There are 3 verified deals."}}]}
            ).encode("utf-8")

    monkeypatch.setenv("SALES_LLM_API_KEY", "test-key")
    monkeypatch.setattr("sales_forecasting.data.llm.request.urlopen", lambda *args, **kwargs: FakeResponse())
    answer = IntelligenceAnswer("analytics", "answered", "There are 3 deals.", {"count": 3}, [])

    refined = refine_answer("How many deals?", answer)

    assert refined.answer == "There are 3 verified deals."
    assert refined.sources == [{"type": "llm", "source": "gpt-4o-mini"}]