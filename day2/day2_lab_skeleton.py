# ============================================================
# DAY 2 LAB — SKELETON: Build a Multi-Agent Research Team
# ============================================================
# Fill in every TODO. Don't open the solution (day2_lab_solution.py)
# until you pass the self-check at the bottom.
#
# WHAT CHANGES FROM DAY 1 — read this table twice:
#
#   Day 1 (single agent)              Day 2 (multi-agent)
#   ─────────────────────             ─────────────────────────────
#   nodes = Python functions          nodes = LLM agents w/ personas
#   routing = your if/else            routing = supervisor LLM decides
#   one prompt for everything         one system prompt PER agent
#   tools available everywhere        tools SCOPED (only researcher
#                                       can search the web)
#   loop = quality-score retry        loop = critic sends draft back
#                                       to writer for revision
#
# What does NOT change: State + Nodes + Edges. A multi-agent system
# is STILL just a StateGraph. If you can build Day 1, you can build
# this — the new ideas are personas, the supervisor, and guardrails.
#
# The system you're building (the SUPERVISOR pattern):
#
#              ┌──────────── supervisor ─────────────┐
#              │       (LLM decides who's next)      │
#     ┌────────┼───────────┬───────────┬─────────────┤
#     ↓        ↓           ↓           ↓             ↓
#  researcher  analyst    writer     critic       FINISH
#     │        │           │           │             ↓
#     └────────┴───────────┴───────────┘            END
#          (every worker reports back to the supervisor)
#
# Recommended reading BEFORE you start (~25 min):
#   1. Multi-agent concepts (architectures, supervisor pattern):
#      https://docs.langchain.com/oss/python/langgraph/multi-agent
#   2. Refresh: conditional branching + loops (you need both again):
#      https://docs.langchain.com/oss/python/langgraph/use-graph-api#conditional-branching
#   3. Structured output (the supervisor's decision is structured!):
#      https://docs.langchain.com/oss/python/langchain/structured-output
#
# Setup: same as Day 1 — `uv sync`, keys in .env, or USE_FAKE=1.
# ============================================================

import os
import operator
from datetime import datetime
from typing import Annotated, List, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# TODO STEP 0 — same imports as Day 1:
# StateGraph, START, END from langgraph.graph
# InMemorySaver from langgraph.checkpoint.memory 




MAX_REVISIONS = 2    # cap on writer↔critic loops
MAX_TURNS = 12         # cap on total supervisor decisions


# ============================================================
# STEP 1 — SHARED STATE: the team's "blackboard"
# ============================================================
# Day 1's state was a data PIPELINE (each field filled once, in
# order). Day 2's state is a BLACKBOARD: every agent reads all of
# it and writes only its own section; the supervisor reads it to
# decide who goes next.
#
# Define a TypedDict with:
#   task (str)
#   research_notes  <- List[str], APPEND-ONLY (which reducer? Day 1!)
#   analysis (str), draft (str), critique (str)
#   revision_count (int), turn_count (int)
#   next_agent (str)   <- the supervisor writes its decision HERE
#   execution_logs     <- append-only, same as Day 1
#
# ASK YOURSELF: why must research_notes append but draft overwrite?
# What would happen to the revision loop if draft used operator.add?

class TeamState(TypedDict):
    task: str
    research_notes: Annotated[List[str], operator.add]
    analysis: str
    draft: str
    critique: str
    revision_count: int
    turn_count: int
    next_agent: str
    execution_logs: Annotated[List[str], operator.add]


# ============================================================
# STEP 2 — STRUCTURED ROUTING DECISION
# ============================================================
# Day 1: structured output produced a quality SCORE.
# Day 2: structured output produces a ROUTING DECISION — this is
# the trick that turns an LLM into a supervisor. Literal[...] means
# the model CANNOT invent an agent that doesn't exist.
#
# WHERE TO LOOK: structured-output docs (same page as Day 1).

class RouterDecision(BaseModel):
    """The supervisor's choice of who acts next."""
    next_agent: Literal["researcher", "analyst", "writer", "critic", "FINISH"]
    reason: str = Field(description="One sentence explaining the choice")


# ============================================================
# STEP 3 — ONE LLM, FOUR PERSONAS (+ tools scoped per agent)
# ============================================================
# A multi-agent "team" doesn't need four models — it needs four
# SYSTEM PROMPTS. (In production you might also vary the model per
# agent: cheap model for the critic, big one for the writer.)
#
# TODO:
# 1. Write a PERSONAS dict: role -> system prompt, for
#    "researcher", "analyst", "writer", "critic".
#    Each persona must say what the agent DOES and what it MUST NOT
#    do (e.g. the researcher never analyzes). Boundaries between
#    agents live in the prompts — write them sharp.
# 2. Create llm (ChatOpenAI + OpenRouter, exactly like Day 1) and
#    search_tool (TavilySearch(max_results=4)).
# 3. supervisor_llm = llm.with_structured_output(RouterDecision)
# 4. Helper: run_persona(role, user_content) → invoke llm with
#    [SystemMessage(PERSONAS[role]), HumanMessage(user_content)]
#    and return response.content.
#
# TOOL SCOPING: only the researcher node may call search_tool.
# That's a deliberate design decision, not a limitation — ask
# yourself what could go wrong if the critic could search.

PERSONAS = {
    "researcher": """
You are the Researcher.
Your job is to gather relevant and reliable information for the given task.
Focus only on collecting and summarizing factual information from the provided search results.
Do NOT analyze the information, write the final answer, or critique other agents' work.
""",

    "analyst": """
You are the Analyst.
Your job is to analyze the research notes provided by the researcher.
Identify key findings, patterns, causes, implications, and useful insights.
Do NOT perform web searches, write the final draft, or critique the writer's work.
""",

    "writer": """
You are the Writer.
Your job is to create a clear, accurate, and well-structured draft using the research and analysis provided.
If a previous draft and critique are provided, revise the draft according to the critic's feedback.
Do NOT perform web searches or make unsupported claims.
""",

    "critic": """
You are the Critic.
Your job is to evaluate the current draft for accuracy, clarity, completeness, and alignment with the task.
If the draft needs improvement, begin your response with REVISE and explain what should be fixed.
If the draft is satisfactory, begin your response with APPROVED.
Do NOT rewrite the draft yourself and do NOT perform web searches.
"""
}

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

search_tool = TavilySearch(max_results=4)

supervisor_llm = llm.with_structured_output(RouterDecision)


def run_persona(role: str, user_content: str) -> str:
    response = llm.invoke([
        SystemMessage(content=PERSONAS[role]),
        HumanMessage(content=user_content),
    ])
    return response.content

# TODO: llm, search_tool, supervisor_llm, run_persona


# ============================================================
# STEP 4 — THE SUPERVISOR NODE (the piece Day 1 didn't have)
# ============================================================
# The supervisor node must:
# 1. Increment turn_count.
# 2. Build a STATUS SUMMARY of the blackboard (which sections are
#    filled? what does the critique say? how many revisions?).
#    Don't dump the full text of everything — the supervisor needs
#    STATUS, not content. (Why? Think tokens and attention.)
# 3. Ask supervisor_llm for a RouterDecision.
# 4. GUARDRAILS — never trust an LLM to terminate a loop:
#      a) if turn_count > MAX_TURNS → force FINISH
#      b) if the LLM picks writer/critic but revision_count >=
#         MAX_REVISIONS and a draft exists → force FINISH
#    This is Day 1's iteration cap wearing a new hat. Same lesson:
#    the LLM proposes, YOUR CODE disposes.
# 5. Return {"next_agent": ..., "turn_count": ..., "execution_logs": [...]}
#
# WHERE TO LOOK: multi-agent docs → "Supervisor" section.

def supervisor_node(state: TeamState):
    """Decide which worker should act next."""

    turn_count = state["turn_count"] + 1

    status = f"""
Task: {state["task"]}

Blackboard status:
- Research notes available: {bool(state["research_notes"])}
- Analysis available: {bool(state["analysis"])}
- Draft available: {bool(state["draft"])}
- Critique: {state["critique"] or "None"}
- Revision count: {state["revision_count"]}
- Turn count: {turn_count}
"""

    decision = supervisor_llm.invoke(
        [
            SystemMessage(
                content="""
You are the supervisor of a multi-agent research team.

Your job is to decide which agent should work next.

Workflow:
1. researcher gathers information.
2. analyst analyzes the research.
3. writer creates or revises the draft.
4. critic reviews the draft.
5. If the draft is approved, finish.

Choose only one of:
researcher, analyst, writer, critic, FINISH.

Use the current blackboard status to make the decision.
"""
            ),
            HumanMessage(content=status),
        ]
    )

    next_agent = decision.next_agent
    reason = decision.reason

    # Guardrail (a): prevent an infinite loop.
    if turn_count > MAX_TURNS:
        next_agent = "FINISH"
        reason = f"Maximum turn limit ({MAX_TURNS}) reached."

    # Guardrail (b): stop unnecessary writer/critic loops.
    elif (
        state["draft"]
        and state["revision_count"] >= MAX_REVISIONS
        and next_agent in {"writer", "critic"}
    ):
        next_agent = "FINISH"
        reason = f"Maximum revisions ({MAX_REVISIONS}) reached."

    return {
        "next_agent": next_agent,
        "turn_count": turn_count,
        "execution_logs": [
            f"Supervisor → {next_agent}: {reason}"
        ],
    }


# ============================================================
# STEP 5 — WORKER AGENT NODES
# ============================================================
# Each worker: read the blackboard → act in persona → return a
# PARTIAL update with ONLY its own section (Day 1 rule, unchanged).

def researcher_node(state: TeamState):
    """Search the web (ONLY this agent may), condense to notes."""

    results = search_tool.invoke(
        {"query": state["task"]}
    )["results"]

    raw = "\n\n".join(
        f"Title: {item.get('title', '')}\n"
        f"Content: {item.get('content', '')}\n"
        f"URL: {item.get('url', '')}"
        for item in results
    )

    notes = run_persona(
        "researcher",
        f"""Task: {state["task"]}

Search results:

{raw}

Summarize the most relevant factual information for the team.
"""
    )

    return {
        "research_notes": [notes],
        "execution_logs": [
            "Researcher completed web research and added research notes."
        ],
    }
    """Search the web (ONLY this agent may), condense to notes."""
    # TODO:
    # 1. results = search_tool.invoke({"query": state["task"]})["results"]
    # 2. Format results into a raw text block (title, content, url)
    # 3. notes = run_persona("researcher", f"Task ...\n\nSearch results:\n{raw}")
    # 4. return {"research_notes": [notes], "execution_logs": [...]}
    #    ^ note the LIST — research_notes is append-only!
    pass


def analyst_node(state: TeamState):
    """Turn raw notes into analysis."""

    research = "\n\n".join(state["research_notes"])

    analysis = run_persona(
        "analyst",
        f"""Task: {state["task"]}

Research notes:

{research}

Analyze the research notes.
Identify the key findings, patterns, causes, implications, and useful insights.
Do not perform web searches.
"""
    )

    return {
        "analysis": analysis,
        "execution_logs": [
            "Analyst completed analysis of the research notes."
        ],
    }
    """Turn raw notes into analysis."""
    # TODO: run_persona("analyst", ...) → {"analysis": ..., "execution_logs": [...]}
    pass


def writer_node(state: TeamState):
    """Write the draft — or REVISE it if a critique is present."""

    revising = (
        bool(state["critique"])
        and state["critique"].strip().upper().startswith("REVISE")
    )

    research = "\n\n".join(state["research_notes"])

    if revising:
        prompt = f"""Task: {state["task"]}

Research notes:
{research}

Analysis:
{state["analysis"]}

Previous draft:
{state["draft"]}

Critic feedback:
{state["critique"]}

Revise the previous draft based on the critic's feedback.
Keep useful content, fix the identified problems, and produce a better final draft.
"""
    else:
        prompt = f"""Task: {state["task"]}

Research notes:
{research}

Analysis:
{state["analysis"]}

Create a clear, accurate, and well-structured draft that answers the task.
"""

    draft = run_persona("writer", prompt)

    revision_count = (
        state["revision_count"] + 1
        if revising
        else state["revision_count"]
    )

    return {
        "draft": draft,
        "critique": "",
        "revision_count": revision_count,
        "execution_logs": [
            "Writer created a draft."
            if not revising
            else "Writer revised the draft based on critic feedback."
        ],
    }
    """Write the draft — or REVISE it if a critique is present."""
    # TODO:
    # 1. revising = critique exists and starts with "REVISE"
    # 2. Build the prompt; when revising, include the previous draft
    #    AND the critique so the writer knows what to fix.
    # 3. return {"draft": ...,
    #            "critique": "",   <- WHY reset this? (see self-check)
    #            "revision_count": +1 only when revising,
    #            "execution_logs": [...]}
    pass


def critic_node(state: TeamState):
    """Review the draft against the research notes."""

    research = "\n\n".join(state["research_notes"])

    critique = run_persona(
        "critic",
        f"""Task: {state["task"]}

Research notes:
{research}

Analysis:
{state["analysis"]}

Current draft:
{state["draft"]}

Review the draft for:
- accuracy
- clarity
- completeness
- alignment with the task
- unsupported claims

If the draft is good enough, respond with:
APPROVED

If it needs changes, respond with:
REVISE: followed by specific improvements.
"""
    )

    return {
        "critique": critique,
        "execution_logs": [
            f"Critic reviewed the draft: {critique}"
        ],
    }
    """Review the draft against the research notes."""
    # TODO: run_persona("critic", ...) → the persona replies either
    # "APPROVED" or "REVISE: <fixes>". Store it in critique.
    pass


# ============================================================
# STEP 6 — ROUTING FUNCTION + WIRE THE GRAPH
# ============================================================
# The conditional-edge function is now TRIVIAL — it just reads the
# supervisor's decision:
#
#     def route_from_supervisor(state) -> str:
#         return state["next_agent"]
#
# Compare with Day 1, where all decision logic lived inside
# quality_router. The intelligence MOVED from the edge into a node.
#
# Wiring checklist:
# 1. add all five nodes
# 2. START → supervisor
# 3. add_conditional_edges("supervisor", route_from_supervisor,
#        {"researcher": "researcher", "analyst": "analyst",
#         "writer": "writer", "critic": "critic", "FINISH": END})
# 4. EVERY worker gets an edge BACK to supervisor — the
#    hub-and-spoke shape that defines the supervisor pattern.
#    (A for-loop over the four worker names is idiomatic.)

# TODO: route_from_supervisor + graph wiring
def route_from_supervisor(state: TeamState) -> str:
    """Route to the agent selected by the supervisor."""
    return state["next_agent"]


builder = StateGraph(TeamState)

# Add all nodes
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("analyst", analyst_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)

# Start with the supervisor
builder.add_edge(START, "supervisor")

# Supervisor decides which worker runs next
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "critic": "critic",
        "FINISH": END,
    },
)

# Every worker reports back to the supervisor
for worker in ["researcher", "analyst", "writer", "critic"]:
    builder.add_edge(worker, "supervisor")


# ============================================================
# STEP 7 — COMPILE, VISUALIZE, RUN
# ============================================================
# Same as Day 1: compile with InMemorySaver, print the Mermaid
# diagram (it should look like a STAR, not Day 1's chain), stream
# with stream_mode="values" and a thread_id, print the final draft.
#
# EXPERIMENT 1: set MAX_REVISIONS = 0. What happens to quality?
# EXPERIMENT 2: delete guardrail (a) and make the critic always
#   say REVISE. Watch the turn cap save you — then delete guardrail
#   (b) too and meet your old friend GraphRecursionError.
# EXPERIMENT 3: swap the analyst's persona for a terrible one
#   ("you are vague and generic"). How far does the damage spread
#   through the team? This is why persona boundaries matter.


if __name__ == "__main__":
    initial_state = {
        "task": "Should our company adopt multi-agent AI systems in 2026?",
        "research_notes": [],
        "analysis": "",
        "draft": "",
        "critique": "",
        "revision_count": 0,
        "turn_count": 0,
        "next_agent": "",
        "execution_logs": [],
    }
    # TODO: compile, visualize, stream, print final draft + stats

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

print("\n=== MERMAID GRAPH ===")
print(graph.get_graph().draw_mermaid())

config = {
    "configurable": {
        "thread_id": "day2-demo"
    }
}

final_state = None

for state in graph.stream(
    initial_state,
    config=config,
    stream_mode="values",
):
    final_state = state

print("\n=== FINAL DRAFT ===")
print(final_state["draft"])

print("\n=== STATS ===")
print("Turns:", final_state["turn_count"])
print("Revisions:", final_state["revision_count"])

print("\n=== EXECUTION LOG ===")
for log in final_state["execution_logs"]:
    print("-", log)

# ============================================================
# SELF-CHECK before you look at the solution
# ============================================================
# [ ] I can explain the supervisor pattern in one sentence
# [ ] My routing function reads state — the DECISION was made in a node
# [ ] research_notes appends; draft overwrites; I know why each
# [ ] The writer RESETS critique — I can explain what breaks if not
#     (hint: what does the supervisor see on the turn after a revision?)
# [ ] Only researcher_node touches search_tool
# [ ] My supervisor has BOTH guardrails, and I triggered EXPERIMENT 2
# [ ] My Mermaid diagram is a star: supervisor in the middle
# [ ] I can name one task where Day 1's single agent is the BETTER
#     design (multi-agent is not free: more calls, more latency,
#     more places to break — coordination must earn its cost)
#
# Stuck? Debugging order that works:
#   1. stream_mode="updates" — watch each supervisor decision + reason
#   2. print the status summary your supervisor_node builds — is the
#      LLM seeing an accurate picture of the blackboard?
#   3. check your conditional-edge dict covers ALL five decisions
#   4. only THEN open day2_lab_solution.py
# ============================================================
