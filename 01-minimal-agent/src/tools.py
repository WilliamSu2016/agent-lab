from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current, factual information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused web search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def web_search(query: str) -> dict[str, Any]:
    """Return up to five results from DuckDuckGo's public instant-answer endpoint."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("web_search requires a non-empty query")
    if len(query) > 300:
        raise ValueError("web_search query must be 300 characters or fewer")

    url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1"
    request = Request(url, headers={"User-Agent": "minimal-research-agent/0.1"})
    with urlopen(request, timeout=10) as response:
        payload = json.load(response)

    results: list[dict[str, str]] = []
    if payload.get("AbstractURL"):
        results.append(
            {
                "title": payload.get("Heading") or "DuckDuckGo instant answer",
                "url": payload["AbstractURL"],
                "snippet": payload.get("AbstractText", ""),
            }
        )

    for topic in payload.get("RelatedTopics", []):
        if len(results) == 5:
            break
        if "Topics" in topic:
            topic_items = topic["Topics"]
        else:
            topic_items = [topic]
        for item in topic_items:
            if len(results) == 5:
                break
            if item.get("FirstURL") and item.get("Text"):
                results.append(
                    {
                        "title": item["Text"].split(" - ", 1)[0],
                        "url": item["FirstURL"],
                        "snippet": item["Text"],
                    }
                )

    return {"query": query, "results": results}

