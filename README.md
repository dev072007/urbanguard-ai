# UrbanGuard AI — Prototype Code

Runnable prototype for the **UrbanGuard AI** capstone project
(1M1B AI for Sustainability Virtual Internship, IBM SkillsBuild & AICTE).

**Author:** Dev Kumar — KCC Institute of Technology and Management

## What this is

A LangGraph implementation of the Supervisor / Hub-Spoke multi-agent
pattern for UrbanGuard AI's two modules:

- **`risk_advisor`** — predicts ward-level flood risk from rainfall,
  drainage capacity and elevation profile (`predict_flood_risk` tool).
- **`grievance_handler`** — extracts the ward and issue from a citizen
  message, cross-checks today's flood-risk output, retrieves the matching
  municipal SOP with a Retrieval-Augmented Generation (RAG) style search
  (`search_municipal_sop`), classifies severity, and creates a ticket
  (`create_grievance_ticket`), always flagging High-severity tickets for
  human ward-officer review before dispatch.

A `supervisor` node reads the citizen's message and routes it to whichever
specialist applies; a `writer` node turns the specialist's structured
findings into a plain-language reply.

## Credit / adapted from

The graph structure, tool-binding style, and Supervisor/Hub-Spoke routing
pattern are adapted from the **Agentic AI Workshop — Hands-On Labs**
repository:

> https://github.com/jvmkumar81/agenticAI

Specifically:
- `lab2_tool_agent.py` → the ReAct tool-calling pattern and the
  keyword-overlap RAG-tool approach used here in `search_municipal_sop`.
- `lab3_multi_agent.py` → the Supervisor/Hub-Spoke routing pattern,
  adapted to route between `risk_advisor` and `grievance_handler` instead
  of the original workshop's `researcher` / `writer`.
- `llm_utils.py` → the multi-provider LLM selection helper. This project's
  version adds a bundled offline `DemoLLM` fallback so the graph can be
  run and graded with **zero API keys and no network access**.

## Run it

```bash
pip install -r requirements.txt
python urbanguard_agent.py
```

No `.env` file is required — with no `LLM_PROVIDER` / API key configured,
it automatically uses the bundled offline `DemoLLM` and still executes the
full `supervisor → specialist → writer` graph end to end, printing a
sample citizen-report and a sample risk-query conversation.

To use a real model instead, create a `.env` file:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
# or LLM_PROVIDER=openai / ollama — see llm_utils.py
```

## Files

| File | Purpose |
|---|---|
| `urbanguard_agent.py` | Main graph: tools, supervisor, specialists, writer |
| `llm_utils.py` | Multi-provider LLM helper + offline DemoLLM fallback |
| `requirements.txt` | Python dependencies |

## Responsible AI note

Per the project's Responsible AI section, this prototype never finalizes
crew dispatch on its own — `create_grievance_ticket` always sets
`human_review_required: true` for High-severity tickets, matching SOP DR-05
in the mock municipal knowledge base.
