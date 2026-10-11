"""
TokenWise - end-to-end efficiency report (goals 5 and 6).

Pure function: given the token counts the pipeline already produces, it returns
what would have been sent without TokenWise vs. what is actually sent, plus an
optional dollar estimate. The price is always supplied by the caller.
"""

from typing import Any, Dict, Optional


def _pct(saved: int, base: int) -> float:
    return round(saved / base * 100, 1) if base > 0 else 0.0


def pipeline_report(
    original_doc_tokens: int,
    context_tokens: int,
    question_tokens_before: int,
    question_tokens_after: int,
    price_per_million: Optional[float] = None,
) -> Dict[str, Any]:
    """
    baseline = full raw document + the question as typed
    sent     = retrieved/optimized context + the optimized question
    """
    doc_before = max(0, int(original_doc_tokens))
    doc_after = max(0, int(context_tokens))
    q_before = max(0, int(question_tokens_before))
    q_after = max(0, int(question_tokens_after))

    baseline = doc_before + q_before
    sent = doc_after + q_after
    saved = max(0, baseline - sent)

    report: Dict[str, Any] = {
        "baseline_tokens": baseline,
        "sent_tokens": sent,
        "tokens_saved": saved,
        "reduction_percent": _pct(saved, baseline),
        "stages": {
            "document": {
                "before": doc_before,
                "after": doc_after,
                "saved": max(0, doc_before - doc_after),
                "reduction_percent": _pct(max(0, doc_before - doc_after), doc_before),
            },
            "question": {
                "before": q_before,
                "after": q_after,
                "saved": max(0, q_before - q_after),
                "reduction_percent": _pct(max(0, q_before - q_after), q_before),
            },
        },
        "cost": None,
    }

    if price_per_million is not None and price_per_million >= 0:
        base_usd = baseline / 1_000_000 * price_per_million
        sent_usd = sent / 1_000_000 * price_per_million
        saved_usd = max(0.0, base_usd - sent_usd)
        report["cost"] = {
            "price_per_million_input_tokens": price_per_million,
            "baseline_usd": round(base_usd, 6),
            "optimized_usd": round(sent_usd, 6),
            "saved_usd": round(saved_usd, 6),
            "saved_usd_per_1000_queries": round(saved_usd * 1000, 2),
        }
    return report
