# =============================================================================
# Support Ticket Triage Agent -- A LangGraph Learning Project
# =============================================================================
#
# This project teaches you how LangGraph works by building a support ticket
# triage assistant.
#
# WHAT THIS DOES:
# A user submits a support issue (e.g. "After logging in, I can't access my dashboard"). The system
# runs 3 specialist nodes in PARALLEL (intent classification, sentiment
# analysis, technical impact assessment), then a decision node picks the
# support route and routes to either STANDARD support or ESCALATED handling
# based on severity, urgency, and impact.
#
# LANGGRAPH CONCEPTS COVERED:
# 1. State Management (Pydantic) -- the ticket flows through the graph
# 2. Nodes -- each function does one job (classify intent, sentiment, etc.)
# 3. Parallel Execution -- 3 specialist nodes run at the same time
# 4. Fan-in -- waiting for all 3 specialist outputs before deciding the route
# 5. Conditional Edges -- routing to standard vs escalated based on the decision
# 6. Graph Compilation -- turning the graph definition into a runnable app
#
# GRAPH STRUCTURE:
#
#                          START
#                            |
#                      receive_ticket
#                            |
#      +---> classify_ticket_intent -------+
#      |                                   |
#      +---> analyze_customer_sentiment ---+---> choose_support_route
#      |                                   |            |
#      +---> estimate_technical_impact ----+       (conditional)
#                                                  /            \
#                                            standard?      escalated?
#                                                |                |
#                                  standard_support_response  escalated_support_response
#                                                |                |
#                                               END              END
#
# HOW TO RUN:
# python support_ticket_graph.py
#
# DEPENDENCIES (same as requirements.txt):
# langgraph, langchain-openai, python-dotenv, pydantic
#
# =============================================================================

import sys
import operator
import json
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


class SupportTicketState(BaseModel):
    ticket_text: str = ""
    intent_analysis: str = ""
    sentiment_analysis: str = ""
    technical_impact: str = ""
    needs_escalation: bool = False
    route_reason: str = ""
    final_response: str = ""
    messages: Annotated[list, operator.add] = []


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)


def receive_ticket(state: SupportTicketState) -> dict:
    response = llm.invoke(
        f"You are an experienced customer-care lead. "
        f"A customer submitted this support ticket: '{state.ticket_text}'. "
        f"Acknowledge the ticket has been received in 1-2 sentences, sounding human and never blaming the customer."
    )
    return {
        "messages": [f"[receive_ticket] {response.content}"]
    }


def classify_ticket_intent(state: SupportTicketState) -> dict:
    response = llm.invoke(
        f"You are a ticket classification specialist. "
        f"The customer's ticket is: '{state.ticket_text}'. "
        f"Determine the product area affected and the type of request "
        f"(e.g. bug report, how-to question, billing issue, access issue). "
        f"Keep it under 5 sentences."
    )
    return {
        "intent_analysis": response.content,
        "messages": [f"[classify_ticket_intent] Done"]
    }


def analyze_customer_sentiment(state: SupportTicketState) -> dict:
    response = llm.invoke(
        f"You are a customer sentiment analyst. "
        f"The customer's ticket is: '{state.ticket_text}'. "
        f"Detect the level of frustration, urgency, and any escalation signals "
        f"(e.g. threats to cancel, mentions of lost business, repeated contact). "
        f"Keep it under 5 sentences."
    )
    return {
        "sentiment_analysis": response.content,
        "messages": [f"[analyze_customer_sentiment] Done"]
    }


def estimate_technical_impact(state: SupportTicketState) -> dict:
    response = llm.invoke(
        f"You are a technical impact assessor. "
        f"The customer's ticket is: '{state.ticket_text}'. "
        f"Assess the likely number of affected users, which functions are blocked, "
        f"and the possible severity (LOW, MEDIUM, HIGH, CRITICAL). "
        f"Keep it under 5 sentences."
    )
    return {
        "technical_impact": response.content,
        "messages": [f"[estimate_technical_impact] Done"]
    }


def choose_support_route(state: SupportTicketState) -> dict:
    response = llm.invoke(
        f"You are a support routing decision system. The customer's ticket is: '{state.ticket_text}'.\n\n"
        f"Here are three specialist assessments:\n\n"
        f"INTENT:\n{state.intent_analysis}\n\n"
        f"SENTIMENT:\n{state.sentiment_analysis}\n\n"
        f"TECHNICAL IMPACT:\n{state.technical_impact}\n\n"
        f"Decide: should this ticket follow STANDARD support (routine requests, low/medium "
        f"impact, manageable sentiment) or ESCALATED handling (high frustration, high urgency, "
        f"high/critical technical impact, or clear escalation signals)?\n\n"
        f"Reply STRICTLY in this JSON format (no other text):\n"
        f'{{"needs_escalation": true/false, "reason": "one sentence explanation"}}'
    )

    try:
        result = json.loads(response.content)
        needs_escalation = result["needs_escalation"]
        reason = result["reason"]
    except (json.JSONDecodeError, KeyError):
        needs_escalation = False
        reason = "Could not parse decision, defaulting to standard support."

    return {
        "needs_escalation": needs_escalation,
        "route_reason": reason,
        "messages": [f"[choose_support_route] escalation={needs_escalation}"]
    }


def standard_support_response(state: SupportTicketState) -> dict:
    response = llm.invoke(
        f"You are an experienced customer-care lead. The customer's ticket is: '{state.ticket_text}'.\n\n"
        f"Based on these specialist assessments, write a standard support response:\n\n"
        f"INTENT: {state.intent_analysis}\n"
        f"SENTIMENT: {state.sentiment_analysis}\n"
        f"TECHNICAL IMPACT: {state.technical_impact}\n\n"
        f"Acknowledge the issue, explain the next steps and a realistic timeline, "
        f"sound human, never blame the customer, and avoid unsupported promises. "
        f"Include a short subject line, greeting, body, and sign-off."
    )
    return {
        "final_response": f"STANDARD SUPPORT RESPONSE\n{'='*45}\n{response.content}",
        "messages": [f"[standard_support_response] Generated standard response"]
    }


def escalated_support_response(state: SupportTicketState) -> dict:
    response = llm.invoke(
        f"You are an experienced customer-care lead handling an escalated case. "
        f"The customer's ticket is: '{state.ticket_text}'.\n\n"
        f"Based on these specialist assessments, write an escalated support response:\n\n"
        f"INTENT: {state.intent_analysis}\n"
        f"SENTIMENT: {state.sentiment_analysis}\n"
        f"TECHNICAL IMPACT: {state.technical_impact}\n\n"
        f"Acknowledge the severity and urgency, confirm this has been flagged to a senior "
        f"team or specialist, give a clear and realistic next step and timeline, sound human, "
        f"never blame the customer, and avoid unsupported promises. "
        f"Include a short subject line, greeting, body, and sign-off."
    )
    return {
        "final_response": f"ESCALATED SUPPORT RESPONSE\n{'='*45}\n{response.content}",
        "messages": [f"[escalated_support_response] Generated escalated response"]
    }


def support_route(state: SupportTicketState) -> str:
    if state.needs_escalation:
        return "escalated"
    else:
        return "standard"


graph = StateGraph(SupportTicketState)

graph.add_node("receive_ticket", receive_ticket)
graph.add_node("classify_ticket_intent", classify_ticket_intent)
graph.add_node("analyze_customer_sentiment", analyze_customer_sentiment)
graph.add_node("estimate_technical_impact", estimate_technical_impact)
graph.add_node("choose_support_route", choose_support_route)
graph.add_node("standard_support_response", standard_support_response)
graph.add_node("escalated_support_response", escalated_support_response)

graph.add_edge(START, "receive_ticket")
graph.add_edge("receive_ticket", "classify_ticket_intent")
graph.add_edge("receive_ticket", "analyze_customer_sentiment")
graph.add_edge("receive_ticket", "estimate_technical_impact")

graph.add_edge("classify_ticket_intent", "choose_support_route")
graph.add_edge("analyze_customer_sentiment", "choose_support_route")
graph.add_edge("estimate_technical_impact", "choose_support_route")

graph.add_conditional_edges(
    "choose_support_route",
    support_route,
    {
        "standard": "standard_support_response",
        "escalated": "escalated_support_response",
    }
)

graph.add_edge("standard_support_response", END)
graph.add_edge("escalated_support_response", END)

app = graph.compile()


def run_support_triage(ticket_text: str):
    print("=" * 55)
    print(" SUPPORT TICKET TRIAGE AGENT")
    print(f" Ticket: \"{ticket_text}\"")
    print("=" * 55)

    result = app.invoke({
        "ticket_text": ticket_text,
        "messages": [],
    })

    print("\n" + "=" * 55)
    print(" FINAL SUPPORT RESPONSE")
    print("=" * 55)
    print(f"\n{result['final_response']}")

    print("\n" + "-" * 55)
    print(" MESSAGE LOG")
    print("-" * 55)
    for msg in result["messages"]:
        print(f"  {msg}")

    return result


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print(" SUPPORT TICKET TRIAGE AGENT")
    print("=" * 55)
    print("\n Describe the support issue and I'll analyze it and")
    print(" route it to the right handling path.")
    print(" Type 'quit' to exit.\n")

    while True:
        ticket_text = input(" Describe the issue > ").strip()
        if ticket_text.lower() in ("quit", "exit", "q"):
            print("\n Goodbye!\n")
            break
        if not ticket_text:
            continue

        run_support_triage(ticket_text)
        print("\n")
