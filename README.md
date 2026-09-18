```markdown
# Support Ticket Triage Agent - Learn LangGraph Step by Step

A beginner-friendly LangGraph project that evaluates a customer support ticket
and routes it to the right handling path.

The project demonstrates a clear LangGraph pattern:

```
[Support Ticket]
|

receive_ticket
      |
      
      +--> classify_ticket_intent -----+
      
      +--> analyze_customer_sentiment -+--> choose_support_route
      
      +--> estimate_technical_impact --+          |
                                             
                                             conditional
                                          /              \
                              
                              standard_support_response  escalated_support_response
                                          |                          |
                                         END                        END
```

---

## What This Project Does

A user submits a support ticket such as:

- `The export feature keeps crashing and I've missed two client deadlines`
- `Can't find the option to change my billing plan`
- `Our entire team is locked out and can't process orders`

The graph then:

1. Receives the raw support ticket.
2. Runs three specialist nodes in parallel:
   - `classify_ticket_intent` — determines product area and request type
   - `analyze_customer_sentiment` — detects frustration, urgency, and escalation signals
   - `estimate_technical_impact` — assesses affected users, blocked functions, and severity
3. Uses a decision node, `choose_support_route`, to decide whether the ticket needs:
   - standard support handling, or
   - escalated handling
4. Routes to the correct final node via the conditional `support_route`.
5. Prints the final response and the message log.

---

## LangGraph Concepts Covered

| Concept              | Where It Appears                                                                                                                                        |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| State                | `SupportTicketState` Pydantic model                                                                                                                     |
| Nodes                | `receive_ticket`, `classify_ticket_intent`, `analyze_customer_sentiment`, `estimate_technical_impact`, `choose_support_route`, `standard_support_response`, `escalated_support_response` |
| Parallel execution   | Three specialist nodes run after `receive_ticket`                                                                                                       |
| Fan-in               | All three specialist outputs flow into `choose_support_route`                                                                                            |
| Conditional edges    | `support_route` sends the graph to standard or escalated handling                                                                                        |
| Final output         | `standard_support_response` or `escalated_support_response`                                                                                              |
| Message accumulation | `messages: Annotated[list, operator.add]`                                                                                                                |

---

## Project Files

```
support_ticket_graph.py    Main LangGraph project
architecture.md             Architecture explanation
architecture.drawio          Diagram source file
requirements.txt            Python dependencies
.env.example                Example environment file
.gitignore                  Ignored local files
```

---

## Setup

### 1. Create and activate a virtual environment

```
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:

```
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Configure your OpenAI API key

```
copy .env.example .env
```

Edit `.env` and add your API key:

```
OPENAI_API_KEY=sk-...
```

Never commit your real `.env` file.

### 4. Run the project

```
python support_ticket_graph.py
```

---

## Expected Flow

Example input:

```
Our entire team has been locked out of the platform since this morning and
we can't process any customer orders. This is extremely urgent.
```

The graph will:

1. Receive the ticket.
2. Classify the product area and request type.
3. Analyze sentiment, urgency, and escalation signals.
4. Estimate affected users, blocked functions, and severity.
5. Decide whether the ticket needs standard or escalated handling.
6. Print the final support response.
7. Print the message log showing which nodes executed.

---

## Code Walkthrough

| Step | What Happens                        | File                     |
| ---- | ------------------------------------ | -------------------------- |
| 1    | Define `SupportTicketState`          | `support_ticket_graph.py` |
| 2    | Initialize `ChatOpenAI`              | `support_ticket_graph.py` |
| 3    | Define graph node functions          | `support_ticket_graph.py` |
| 4    | Define `support_route`               | `support_ticket_graph.py` |
| 5    | Add nodes and edges to `StateGraph`  | `support_ticket_graph.py` |
| 6    | Compile graph as `app`               | `support_ticket_graph.py` |
| 7    | Run with `run_support_triage()`      | `support_ticket_graph.py` |

---

## Important Note

This is a learning project, not a production incident-management system. The
routing decision is a starting point for triage, not a substitute for human
judgment on genuinely critical or safety-related issues.

---

## Key Takeaways

1. State holds the data that travels through the graph.
2. Nodes are normal Python functions that read state and return updates.
3. Parallel execution happens when one node connects to multiple next nodes.
4. Fan-in happens when multiple nodes connect into one later node.
5. Conditional edges let the graph choose the next path at runtime.
