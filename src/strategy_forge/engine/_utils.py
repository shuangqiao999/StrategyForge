"""Shared utilities for the deduction pipeline."""


def extract_text(response) -> str:
    """Extract text content from various LLM response formats."""
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        c = response.content
        if isinstance(c, list):
            from strategy_forge.core.llm_client import TextBlock
            return "".join(b.text for b in c if isinstance(b, TextBlock))
        return str(c)
    if isinstance(response, dict):
        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        return str(response)
    return str(response)
