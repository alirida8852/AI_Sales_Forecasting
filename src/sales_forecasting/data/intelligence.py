"""Deterministic routing between sales analytics and company knowledge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from sales_forecasting.data.database import DEFAULT_DATABASE_PATH
from sales_forecasting.data.knowledge import search_documents
from sales_forecasting.data.repository import SQLiteSalesRepository


ANALYTICS_TERMS = {
    "pipeline",
    "opportunities",
    "opportunity",
    "revenue",
    "deals",
    "deal",
    "win rate",
    "stage",
    "owner",
    "agent",
    "account",
}
KNOWLEDGE_TERMS = {
    "playbook",
    "policy",
    "recommend",
    "recommendation",
    "procedure",
    "product",
    "discount",
    "objection",
}
STOP_WORDS = {
    "a",
    "about",
    "and",
    "are",
    "can",
    "does",
    "after",
    "for",
    "happen",
    "how",
    "i",
    "is",
    "our",
    "should",
    "the",
    "what",
    "with",
}
KNOWLEDGE_QUERY_WORDS = STOP_WORDS | {
    "document",
    "playbook",
    "policy",
    "procedure",
    "recommend",
    "recommendation",
    "sales",
    "say",
}


@dataclass(frozen=True)
class IntelligenceAnswer:
    """A grounded answer that can later be rendered by an API or LLM."""

    intent: str
    status: str
    answer: str
    data: dict[str, object] | None
    sources: list[dict[str, object]]
    next_action: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def route_question(
    question: str,
    database_path: Path = DEFAULT_DATABASE_PATH,
    access_scope: str = "internal",
) -> IntelligenceAnswer:
    """Route a question to verified analytics or document retrieval."""
    normalized = question.strip().lower()
    if not normalized:
        return IntelligenceAnswer("unsupported", "needs_question", "Please ask a sales question.", None, [])

    if any(term in normalized for term in ANALYTICS_TERMS):
        return _answer_analytics(normalized, database_path)
    if any(term in normalized for term in KNOWLEDGE_TERMS):
        return _answer_knowledge(question, database_path, access_scope)
    knowledge_answer = _answer_knowledge(question, database_path, access_scope)
    if knowledge_answer.status == "retrieved":
        return knowledge_answer
    return IntelligenceAnswer(
        "unsupported",
        "unsupported_question",
        "I can answer pipeline questions or search approved company documents.",
        None,
        [],
    )


def _answer_analytics(question: str, database_path: Path) -> IntelligenceAnswer:
    repository = SQLiteSalesRepository(database_path)
    summary = repository.pipeline_summary()
    if "stage" in question:
        stages = repository.stage_counts()
        return IntelligenceAnswer(
            "analytics",
            "answered",
            "Here is the current opportunity count by stage.",
            {"stages": stages},
            [{"type": "database", "source": "opportunities"}],
            "Prioritize open stages before reviewing closed outcomes.",
        )
    if any(term in question for term in ("show", "find", "search", "which")) and any(
        term in question for term in ("deal", "opportunity", "opportunities", "account", "owner", "agent")
    ):
        search = _lookup_term(question)
        results = repository.search_opportunities(search=search or None, limit=5)
        if not results:
            return IntelligenceAnswer(
                "analytics",
                "no_results",
                "I could not find matching opportunities in the sales database.",
                {"results": []},
                [{"type": "database", "source": "opportunities"}],
                "Try an opportunity ID, account, product, or owner name.",
            )
        return IntelligenceAnswer(
            "analytics",
            "answered",
            f"I found {len(results)} matching opportunities.",
            {"results": results},
            [{"type": "database", "source": "opportunities"}],
            "Open the Pipeline view to review the matching opportunities.",
        )
    if "revenue" in question:
        answer = f"Closed revenue is ${summary['closed_revenue']:,.2f}."
        data = {"closed_revenue": summary["closed_revenue"]}
        next_action = "Review the Forecast view for the current outlook."
    elif "open" in question:
        answer = f"There are {summary['open_opportunities']:,} open opportunities."
        data = {"open_opportunities": summary["open_opportunities"]}
        next_action = "Ask for matching deals or open the Pipeline view to prioritize them."
    elif "win rate" in question:
        closed = summary["won_opportunities"] + summary["lost_opportunities"]
        rate = summary["won_opportunities"] / closed if closed else 0
        answer = f"The closed-opportunity win rate is {rate:.1%}."
        data = {"win_rate": rate}
        next_action = "Compare the win rate with team or stage breakdowns before acting."
    else:
        answer = f"The pipeline contains {summary['total_opportunities']:,} opportunities."
        data = {"total_opportunities": summary["total_opportunities"]}
        next_action = "Ask about a stage, account, owner, or specific opportunity for detail."
    return IntelligenceAnswer(
        "analytics",
        "answered",
        answer,
        data,
        [{"type": "database", "source": "opportunities"}],
        next_action,
    )


def _answer_knowledge(question: str, database_path: Path, access_scope: str) -> IntelligenceAnswer:
    words = re.sub(r"[^\w\s]", "", question.lower()).split()
    terms = " ".join(
        word for word in words if word not in KNOWLEDGE_QUERY_WORDS
    )
    results = search_documents(terms or question, database_path=database_path, access_scope=access_scope)
    if not results:
        return IntelligenceAnswer(
            "knowledge",
            "no_results",
            "I could not find supporting information in the approved company documents.",
            None,
            [],
            "Upload an approved playbook, policy, or product document before asking knowledge questions.",
        )
    sources = [
        {
            "type": "document",
            "document_id": result["document_id"],
            "source": result["source"],
            "title": result["title"],
            "excerpt": result["content"],
        }
        for result in results
    ]
    return IntelligenceAnswer(
        "knowledge",
        "retrieved",
        "I found supporting information in the approved company documents.",
        None,
        sources,
        "Use the cited document section as the basis for the next sales action.",
    )


def _lookup_term(question: str) -> str:
    """Extract a useful lookup phrase without treating question words as data."""
    words = re.sub(r"[^\w\s]", "", question.lower()).split()
    ignored = STOP_WORDS | {
        "show",
        "find",
        "search",
        "which",
        "deals",
        "opportunities",
        "opportunity",
        "account",
        "owner",
        "agent",
        "pipeline",
    }
    return " ".join(word for word in words if word not in ignored)