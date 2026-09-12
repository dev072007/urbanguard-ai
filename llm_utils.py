"""
Shared LLM provider helper for the UrbanGuard AI prototype.

Adapted from `llm_utils.py` in jvmkumar81/agenticAI (Agentic AI Workshop —
Hands-On Labs, https://github.com/jvmkumar81/agenticAI), which supports
Anthropic, OpenAI or a local Ollama model so nobody is blocked by which API
key they happen to have.

One change from the original: when no provider is configured at all, this
version falls back to a small deterministic DemoLLM instead of raising an
error. That means a recruiter or mentor can clone this project and run
`python urbanguard_agent.py` immediately, with no API key and no network
call, and still see the full Supervisor -> specialist -> writer graph
execute end to end. Set LLM_PROVIDER (+ the matching API key) in a .env
file to swap in a real model with no code changes.

Usage:
    from llm_utils import get_llm
    llm = get_llm()
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from langchain_core.messages import AIMessage, SystemMessage

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class DemoLLM:
    """Deterministic offline stand-in used only when no real provider is
    configured. Keeps the graph fully runnable for grading/demo purposes
    without any API key. NOT used when LLM_PROVIDER / an API key is set.
    """

    def __init__(self):
        self._tools = []

    def bind_tools(self, tools):
        self._tools = tools
        return self

    def invoke(self, messages):
        # Full concatenated text (used to detect *which node* is calling —
        # supervisor vs. writer) plus the citizen's own message in isolation
        # (used for the actual routing decision, so words in the system
        # prompt itself never leak into the classification).
        full_text = " ".join(
            getattr(m, "content", "") if not isinstance(m, tuple) else m[1]
            for m in messages
        ).lower()
        citizen_text = ""
        for m in messages:
            content = getattr(m, "content", "") if not isinstance(m, tuple) else m[1]
            if not isinstance(m, SystemMessage):
                citizen_text = content.lower()
                break

        if "route the citizen" in full_text or "exactly one word" in full_text:
            is_question = "?" in citizen_text or any(
                k in citizen_text for k in ["risk of", "will it", "chance of", "at risk"]
            )
            is_report = any(
                k in citizen_text for k in ["waterlog", "since this morning", "entering", "complain", "report"]
            )
            if is_report and not is_question:
                return AIMessage(content="grievance_handler")
            if is_question:
                return AIMessage(content="risk_advisor")
            return AIMessage(content="grievance_handler" if "flood" in citizen_text or "drain" in citizen_text else "risk_advisor")

        if "final citizen-facing reply" in full_text or "write a clear" in full_text:
            return AIMessage(content=self._compose_reply(messages))

        return AIMessage(content="Acknowledged.")

    @staticmethod
    def _compose_reply(messages) -> str:
        """Pull the structured tool JSON out of the conversation so the demo
        reply is grounded in real values instead of a single canned line."""
        import json as _json
        import re as _re

        full_text = " ".join(getattr(m, "content", "") for m in messages)

        ticket_match = _re.search(r'"ticket_id":\s*"([^"]+)"', full_text)
        severity_match = _re.search(r'"severity":\s*"([^"]+)"', full_text)
        risk_match = _re.search(r'"risk_level":\s*"([^"]+)"', full_text)
        ward_match = _re.search(r'"ward":\s*"([^"]+)"', full_text)
        action_match = _re.search(r'"recommended_action":\s*"([^"]+)"', full_text)
        dept_match = _re.search(r'"department":\s*"([^"]+)"', full_text)

        if ticket_match:
            parts = [f"Logged for {ward_match.group(1)}" if ward_match else "Logged"]
            if severity_match:
                parts.append(f"severity {severity_match.group(1)}")
            parts_str = ", ".join(parts)
            dept = f" Routed to {dept_match.group(1)}." if dept_match else ""
            return f"{parts_str}. Ticket #{ticket_match.group(1)} created.{dept} A ward officer reviews high-severity tickets before crew dispatch."

        if risk_match:
            ward = ward_match.group(1) if ward_match else "your area"
            action = action_match.group(1) if action_match else "Please stay alert for updates."
            return f"Flood risk for {ward} is currently {risk_match.group(1)}. {action}"

        return (
            "Thanks for the update — this has been logged and routed per the "
            "matching SOP. See the ticket details above for status and next steps."
        )


def get_llm(temperature: float = 0):
    """Return a LangChain chat model driven by .env / environment variables,
    falling back to DemoLLM when nothing is configured."""
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        model = os.environ.get("LLM_MODEL", DEFAULT_OLLAMA_MODEL)
        base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)

    use_anthropic = provider == "anthropic" or (provider == "" and os.environ.get("ANTHROPIC_API_KEY"))
    if use_anthropic and os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        model = os.environ.get("LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)
        return ChatAnthropic(model=model, temperature=temperature)

    use_openai = provider == "openai" or (provider == "" and os.environ.get("OPENAI_API_KEY"))
    if use_openai and os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.environ.get("LLM_MODEL", DEFAULT_OPENAI_MODEL)
        return ChatOpenAI(model=model, temperature=temperature)

    print(
        "[llm_utils] No LLM_PROVIDER / API key configured — running with the "
        "bundled offline DemoLLM so the graph still executes end to end.\n"
        "            Set LLM_PROVIDER + an API key in .env to use a real model."
    )
    return DemoLLM()
