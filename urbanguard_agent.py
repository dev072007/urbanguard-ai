"""
UrbanGuard AI — Smart Flood Risk Prediction & Civic Grievance Assistant
========================================================================
1M1B AI for Sustainability Virtual Internship (IBM SkillsBuild & AICTE)
Dev Kumar — KCC Institute of Technology and Management

WHAT THIS FILE IS
------------------
A runnable prototype of the two-module UrbanGuard AI workflow described in
the project submission:
  1. a Flood Risk Prediction tool (ward-level rainfall/drainage/terrain ->
     risk score), and
  2. a Civic Grievance Assistant (citizen message -> entity extraction ->
     risk cross-check -> RAG lookup over municipal SOPs -> ticket routing).

A Supervisor node reads the citizen's message and routes it to whichever
specialist applies, then a Writer node turns the specialist's structured
findings into a plain-language reply — the Supervisor / Hub-Spoke
multi-agent pattern.

CREDIT / ADAPTED FROM
----------------------
The graph structure, tool-binding style and Supervisor/Hub-Spoke pattern
are adapted from the "Agentic AI Workshop — Hands-On Labs" repository:
    https://github.com/jvmkumar81/agenticAI
        - lab2_tool_agent.py  -> the ReAct tool-calling pattern and the
          keyword-overlap RAG tool used here for `search_municipal_sop`
        - lab3_multi_agent.py -> the Supervisor/Hub-Spoke routing pattern,
          adapted here to route between `risk_advisor` and
          `grievance_handler` instead of `researcher` / `writer`
        - llm_utils.py        -> the multi-provider LLM selection helper
          (this project's `llm_utils.py` adds an offline DemoLLM fallback
          so the graph runs with zero API keys for grading/demo purposes)

Run:
    python urbanguard_agent.py
"""

import json
import re
import hashlib
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm_utils import get_llm

# ---------------------------------------------------------------------------
# Reference data: wards and a small municipal-SOP knowledge base.
# In production these would be a live drainage-sensor feed and a real
# vector store; this keeps the prototype dependency-free while showing the
# exact shape of the retrieval and prediction steps.
# ---------------------------------------------------------------------------

WARDS = {
    "ward 14": {"name": "Ward 14 – Lakeview Rd", "drainage_capacity": 2, "elevation_profile": "low-lying"},
    "ward 7": {"name": "Ward 7 – Hill Road", "drainage_capacity": 4, "elevation_profile": "elevated"},
    "ward 22": {"name": "Ward 22 – Old Town Market", "drainage_capacity": 1, "elevation_profile": "low-lying"},
}

SOP_DOCS = [
    ("DR-07", "market area waterlogging",
     "Waterlogging in a market or commercial area classified High severity requires a pump crew dispatch "
     "within 2 hours and a temporary barricade if standing water exceeds 15cm."),
    ("DR-12", "school area waterlogging",
     "Any waterlogging report near a school is auto-escalated to High severity regardless of rainfall data, "
     "with a mandatory site visit before the next school day."),
    ("DR-03", "residential drainage complaint",
     "Residential waterlogging complaints are classified by depth: under 10cm is Low (logged, next scheduled "
     "cleaning), 10 to 25cm is Medium (crew within 24 hours), over 25cm is High (crew within 4 hours)."),
    ("DR-19", "road closure flooding",
     "Roads with flooding deep enough to submerge a two-wheeler's wheels must be flagged for barricading and "
     "reported to the traffic police control room in addition to the drainage department."),
    ("DR-05", "general escalation policy",
     "Any High-severity ticket must be reviewed by a human ward officer before crew dispatch is confirmed; "
     "the assistant may draft the recommended action but may not finalize dispatch on its own."),
]


def _keyword_score(query: str, text: str) -> int:
    """Simple keyword-overlap scorer — same lightweight approach used for
    `search_knowledge_base` in lab2_tool_agent.py, kept dependency-free."""
    q_words = set(re.findall(r"[a-z]+", query.lower()))
    t_words = set(re.findall(r"[a-z]+", text.lower()))
    return len(q_words & t_words)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def predict_flood_risk(ward_key: str, rainfall_mm: float) -> str:
    """Predict today's flood-risk level for a ward. `ward_key` must be one
    of: 'ward 14', 'ward 7', 'ward 22'. `rainfall_mm` is today's forecast
    rainfall in millimetres. Returns a JSON string with risk_level,
    confidence and recommended_action. Do NOT use this for wards outside
    the given list — it will not have drainage/elevation data for them.
    """
    ward = WARDS.get(ward_key.lower())
    if ward is None:
        return json.dumps({"error": f"No data for '{ward_key}'. Known wards: {list(WARDS)}"})

    low_lying = ward["elevation_profile"] == "low-lying"
    poor_drainage = ward["drainage_capacity"] <= 2

    if rainfall_mm > 50 and poor_drainage and low_lying:
        risk_level, confidence = "High", 0.82
    elif rainfall_mm > 25 and (poor_drainage or low_lying):
        risk_level, confidence = "Medium", 0.65
    else:
        risk_level, confidence = "Low", 0.7

    action = {
        "High": f"Alert {ward['name']} residents; pre-position pump crew nearby.",
        "Medium": f"Monitor {ward['name']}; notify ward officer, no alert push yet.",
        "Low": f"No action needed for {ward['name']} today.",
    }[risk_level]

    return json.dumps({
        "ward": ward["name"],
        "risk_level": risk_level,
        "confidence": confidence,
        "recommended_action": action,
    })


@tool
def search_municipal_sop(query: str) -> str:
    """Search the municipal drainage SOP knowledge base for the passage most
    relevant to a citizen complaint (e.g. market flooding, school-area
    waterlogging, residential drainage, road closures). Returns the SOP code
    and its text. This is a Retrieval-Augmented Generation (RAG) style tool
    — it grounds the assistant's severity call in an actual written policy
    rather than a guess.
    """
    ranked = sorted(SOP_DOCS, key=lambda d: _keyword_score(query, d[1] + " " + d[2]), reverse=True)
    code, topic, text = ranked[0]
    return json.dumps({"sop_code": code, "topic": topic, "text": text})


@tool
def create_grievance_ticket(ward_key: str, issue_type: str, severity: str) -> str:
    """Create a grievance ticket once severity has been classified. Returns
    a ticket ID and the department it was routed to. `severity` must be one
    of: Low, Medium, High. High-severity tickets are always flagged for
    human ward-officer review before dispatch, per SOP DR-05 — this tool
    never finalizes dispatch on its own.
    """
    ward = WARDS.get(ward_key.lower(), {"name": ward_key})
    digest = hashlib.sha256(f"{ward_key}{issue_type}{severity}".encode()).hexdigest()[:4].upper()
    ticket_id = f"UG-{digest}"
    department = "Drainage Dept" if "drain" in issue_type.lower() or "water" in issue_type.lower() else "Public Works"
    human_review = severity == "High"
    return json.dumps({
        "ticket_id": ticket_id,
        "ward": ward["name"],
        "department": department,
        "severity": severity,
        "human_review_required": human_review,
    })


TOOLS = [predict_flood_risk, search_municipal_sop, create_grievance_ticket]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

llm = get_llm()
llm_with_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Graph state and nodes — Supervisor / Hub-Spoke pattern (adapted from
# lab3_multi_agent.py), with two specialists instead of researcher/writer.
# ---------------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


SUPERVISOR_PROMPT = (
    "You are a supervisor for a city flood-risk and civic-grievance assistant. "
    "Route the citizen's message to exactly one specialist: 'risk_advisor' "
    "(the citizen is asking about upcoming flood risk / forecast for an area, "
    "no complaint) or 'grievance_handler' (the citizen is reporting an actual "
    "problem — waterlogging, drainage, flooding — that needs a ticket). "
    "Reply with exactly one word: risk_advisor or grievance_handler."
)


def _extract_ward(text: str) -> str:
    for key in WARDS:
        if key in text.lower():
            return key
    return "ward 14"  # sensible default for this demo


def supervisor_node(state: State) -> State:
    decision = llm_with_tools.invoke([SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"])
    choice = decision.content.strip().lower()
    return {"next": "grievance_handler" if "grievance" in choice else "risk_advisor"}


def risk_advisor_node(state: State) -> State:
    user_text = state["messages"][0].content
    ward_key = _extract_ward(user_text)
    rainfall_match = re.search(r"(\d+)\s*mm", user_text.lower())
    rainfall_mm = float(rainfall_match.group(1)) if rainfall_match else 60.0

    result = predict_flood_risk.invoke({"ward_key": ward_key, "rainfall_mm": rainfall_mm})
    note = HumanMessage(content=f"Flood risk prediction to use in your reply: {result}")
    return {"messages": [note], "next": "writer"}


def grievance_handler_node(state: State) -> State:
    user_text = state["messages"][0].content
    ward_key = _extract_ward(user_text)

    risk_result = json.loads(predict_flood_risk.invoke({"ward_key": ward_key, "rainfall_mm": 60.0}))
    sop_result = json.loads(search_municipal_sop.invoke({"query": user_text}))

    # Severity rule: market/school areas + an already-High-risk ward => High;
    # SOP-implied depth language otherwise decides Medium vs Low.
    text_l = user_text.lower()
    if risk_result.get("risk_level") == "High" and any(k in text_l for k in ["market", "school", "shop"]):
        severity = "High"
    elif any(k in text_l for k in ["market", "school"]):
        severity = "Medium"
    else:
        severity = "Low"

    ticket = json.loads(create_grievance_ticket.invoke(
        {"ward_key": ward_key, "issue_type": "waterlogging", "severity": severity}
    ))

    note = HumanMessage(content=(
        "Grievance processing result to use in your reply — "
        f"risk_context: {json.dumps(risk_result)}; "
        f"matched_sop: {json.dumps(sop_result)}; "
        f"ticket: {json.dumps(ticket)}"
    ))
    return {"messages": [note], "next": "writer"}


def writer_node(state: State) -> State:
    response = llm.invoke(
        [SystemMessage(
            content="Write a clear, 2-3 sentence final citizen-facing reply for UrbanGuard AI, "
                    "using any risk prediction, SOP match, or ticket details already present in "
                    "the conversation above. Mention the ticket ID if one was created."
        )] + state["messages"]
    )
    return {"messages": [response], "next": END}


def route(state: State) -> str:
    return state["next"]


graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("risk_advisor", risk_advisor_node)
graph_builder.add_node("grievance_handler", grievance_handler_node)
graph_builder.add_node("writer", writer_node)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor", route, {"risk_advisor": "risk_advisor", "grievance_handler": "grievance_handler"}
)
graph_builder.add_edge("risk_advisor", "writer")
graph_builder.add_edge("grievance_handler", "writer")
graph_builder.add_edge("writer", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    demo_messages = [
        "Heavy waterlogging near Lakeview Rd market since this morning, water entering two shops. Ward 14.",
        "Is Ward 7 at risk of flooding if we get 40mm of rain tomorrow?",
    ]
    for msg in demo_messages:
        print(f"Citizen: {msg}")
        result = graph.invoke({"messages": [HumanMessage(content=msg)], "next": ""})
        print(f"UrbanGuard AI: {result['messages'][-1].content}\n")

    print("Reference implementation adapted from jvmkumar81/agenticAI "
          "(Agentic AI Workshop — Hands-On Labs), Labs 2 & 3.")
