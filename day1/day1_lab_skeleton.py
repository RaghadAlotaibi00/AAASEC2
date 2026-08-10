# ============================================================
# DAY 1 LAB — SKELETON: Build the Research Agent Yourself
# ============================================================
# Fill in every TODO. Each step tells you exactly WHERE in the
# LangGraph docs to look. Don't copy from the solution file
# (day1_lab_solution.py) until you've tried each step —
# the point of Day 1 is learning to THINK in state graphs.
#
# The system you're building:
#
#   START → collect → store_memory → analyze → evaluate
#              ↑                                  │
#              └── quality < 7 (max 3 tries) ─────┤
#                                                 └ quality >= 7
#                                                       ↓
#                                          report → audit → END
#
# Recommended reading order BEFORE you start (30 min total):
#   1. "Thinking in LangGraph" (the mental model):
#      https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph
#   2. Graph API concepts (State, Nodes, Edges):
#      https://docs.langchain.com/oss/python/langgraph/graph-api
#   3. Using the Graph API (code patterns you'll copy):
#      https://docs.langchain.com/oss/python/langgraph/use-graph-api
#
# API reference (exact signatures when docs aren't enough):
#   https://reference.langchain.com/python/langgraph/
#
# Setup: `uv sync`, then create .env (or set USE_FAKE=1 — see README.md).
# ============================================================

import os
import operator
import typing
from datetime import datetime
from types import SimpleNamespace
from typing import Annotated, List, Dict

try:
    from typing_extensions import TypedDict  # type: ignore
except ImportError:
    TypedDict = typing.TypedDict  # type: ignore

from dotenv import load_dotenv
import dotenv # type: ignore

import pydantic # type: ignore
from pydantic import BaseModel, Field  # type: ignore

class QualityScore(BaseModel):
    """Evaluation of research quality."""
    score: int = Field(ge=1, le=10)
    reasoning: str = Field(description="One-sentence justification")




try:
    from langchain_core.messages import HumanMessage  # type: ignore
except ImportError:  # pragma: no cover
    from langchain.schema import HumanMessage  # type: ignore

     

# TODO STEP 0 — import the graph building blocks from langgraph.
# You need: StateGraph, START, END from langgraph.graph
#           InMemorySaver from langgraph.checkpoint.memory
# WHERE TO LOOK: "Graph API" docs, first code example on the page.
# from langgraph.graph import ...
# from langgraph.checkpoint.memory import ...

from langgraph.graph import StateGraph, START, END  # type: ignore
from langgraph.checkpoint.memory import InMemorySaver  # type: ignore
from langchain_core.vectorstores import InMemoryVectorStore  # type: ignore

class DeterministicFakeEmbedding:
    def __init__(self, size: int = 3) -> None:
        self.size = size

    def _vectorize(self, text: str):
        values = [((ord(char) * 13 + idx) % 100) / 100.0 for idx, char in enumerate(text[: self.size])]
        return values + [0.0] * max(0, self.size - len(values))

    def embed_documents(self, texts):
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text):
        return self._vectorize(text)

class FakeInMemoryVectorStore:
    def __init__(self, embedding):
        self.embedding = embedding
        self.docs = []

    def add_texts(self, texts, **kwargs):
        docs = []
        vectors = self.embedding.embed_documents(texts)
        for text, vector in zip(texts, vectors):
            doc = SimpleNamespace(page_content=text, metadata={}, vector=vector)
            docs.append(doc)
        self.docs.extend(docs)
        return docs

    def similarity_search(self, query, k=4, **kwargs):
        query_vector = self.embedding.embed_query(query)
        scored = []
        for doc in self.docs:
            score = self._cosine_similarity(query_vector, doc.vector)
            scored.append((doc, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored[:k]]

    def _cosine_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

load_dotenv()

USE_FAKE = os.getenv("USE_FAKE", "0") == "1"
if not USE_FAKE:
    missing_openai = not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY") or os.getenv("OPENAI_WORKLOAD_IDENTITY"))
    missing_tavily = not os.getenv("TAVILY_API_KEY")
    if missing_openai or missing_tavily:
        print("WARNING: Missing OpenAI/Tavily credentials; switching to offline fake mode.")
        USE_FAKE = True


# ============================================================
# STEP 1 — THE STATE  (the "digital clipboard" from the slides)
# ============================================================
# Define a TypedDict with everything the workflow needs to remember:
#   topic (str), search_query (str), collected_data (List[Dict]),
#   analyzed_data (List[Dict]), quality_score (int),
#   iteration_count (int), final_report (str), execution_logs
#
# KEY IDEA: execution_logs should use a REDUCER so every node can
# APPEND log lines instead of overwriting the list:
#     execution_logs: Annotated[List[str], operator.add]
#
# WHERE TO LOOK: Graph API docs → "State" section → "Reducers".
#   https://docs.langchain.com/oss/python/langgraph/graph-api
# ASK YOURSELF: what happens to a plain (non-reducer) key when two
# nodes write it? What happens with operator.add?

from typing import List, Dict, Annotated
import operator

class ResearchState(typing.TypedDict):
    topic: str
    search_query: str
    collected_data: typing.List[typing.Dict]
    analyzed_data: typing.List[typing.Dict]
    quality_score: int
    iteration_count: int
    final_report: str
    execution_logs: typing.Annotated[typing.List[str], operator.add]




class AgentState(typing.TypedDict):
    topic: str
    search_query: str
    collected_data: typing.List[typing.Dict]
    analyzed_data: typing.List[typing.Dict]
    quality_score: int
    iteration_count: int
    final_report: str
    execution_logs: typing.Annotated[typing.List[str], operator.add]


# ============================================================
# STEP 2 — MODEL, SEARCH TOOL, EMBEDDINGS
# ============================================================
# Create:
#   llm          = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#   search_tool  = TavilySearch(max_results=5)   # langchain_tavily!
#   vector_store = a Chroma or InMemoryVectorStore with embeddings
#
# ------------------------------------------------------------
# USING OPENROUTER (free models — recommended for this course)
# ------------------------------------------------------------
# OpenRouter is OpenAI-compatible, so ChatOpenAI works as-is —
# you only change the key, the base_url, and the model name.
#
# 1. Get a key at https://openrouter.ai/keys  (starts with sk-or-)
# 2. Put in your .env:
#        OPENAI_API_KEY=sk-or-...
# 3. Create the model like this:
#
#    llm = ChatOpenAI(
#        model="nvidia/nemotron-3-super-120b-a12b:free",
#        temperature=0,
#        base_url="https://openrouter.ai/api/v1",
#    )
#
# Free NVIDIA Nemotron models (the ":free" suffix is REQUIRED —
# without it you'll be billed):
#   nvidia/nemotron-3-super-120b-a12b:free   <- use this one
#   nvidia/nemotron-3-nano-30b-a3b:free      <- fallback if rate-limited
#   nvidia/nemotron-3-ultra-550b-a55b:free   <- biggest, often congested
#   deepseek/deepseek-v4-flash-0731:free     <- try it, could work
# Full list: https://openrouter.ai/collections/free-models
#
# KNOW THE LIMITS: free models are rate-limited (~20 req/min and a
# small daily cap). This lab makes ~5-10 LLM calls per run, so you
# have plenty — but don't run it in a tight loop, and if you get
# HTTP 429, wait a minute or switch to the nano model.
#
# CAVEAT for Step 3: with_structured_output() needs tool/function
# calling. Nemotron supports it, but if a free model ever returns
# an error there, either (a) try another :free model, or (b) pass
# method="json_schema" to with_structured_output.
#
# NOTE: OpenRouter has NO embeddings endpoint. For the vector store
# use InMemoryVectorStore + local HuggingFaceEmbeddings
# (uv sync --group embeddings), or DeterministicFakeEmbedding —
# embeddings only power the memory-retrieval bonus, not the core graph.
# ------------------------------------------------------------
#
# GOTCHA: the old imports you'll find in 2023-24 tutorials
# (langchain.vectorstores, langchain_community.tools.tavily_search)
# are DEAD. Current homes:
#   - TavilySearch:      https://docs.langchain.com/oss/python/integrations/providers/tavily
#   - Chat models:       https://docs.langchain.com/oss/python/langchain/models
#   - InMemoryVectorStore: langchain_core.vectorstores
#
# NOTE: TavilySearch.invoke({"query": q}) returns a DICT — the
# actual sources are under the "results" key. print() it once to see.

# TODO: your code here 



if USE_FAKE:
    # ---------- deterministic fakes: run the graph offline ----------

    class FakeLLM:
        """Just enough of the ChatModel surface for this lab."""

        def invoke(self, messages):
            class _Resp:
                content = (
                    "Key findings: multi-agent orchestration, state-graph "
                    "workflows, and guardrails dominate enterprise agentic "
                    "AI adoption in 2026."
                )
            return _Resp()

    class FakeEvaluator:
        """Scores low on the first pass so the retry loop fires."""

        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return QualityScore(score=5, reasoning="Only one shallow pass over the sources.")
            return QualityScore(score=8, reasoning="Second pass added breadth and depth.")

    class FakeSearch:
        def invoke(self, payload):
            q = payload["query"]
            return {
                "results": [
                    {
                        "title": f"Fake source A for: {q}",
                        "url": "https://example.com/a",
                        "content": f"Deterministic content about {q} — trends, tooling, adoption.",
                    },
                    {
                        "title": f"Fake source B for: {q}",
                        "url": "https://example.com/b",
                        "content": f"Deterministic content about {q} — risks, governance, ROI.",
                    },
                ]
            }

    llm = FakeLLM()
    evaluator = FakeEvaluator()
    search_tool = FakeSearch()
    embeddings = DeterministicFakeEmbedding(size=256)
    vector_store = FakeInMemoryVectorStore(embeddings)

else:
    # ---------- real providers ----------
    import importlib

    ChatOpenAI = None
    for module_name in ["langchain.chat_models", "langchain_openai"]:
        try:
            module = importlib.import_module(module_name)
            ChatOpenAI = getattr(module, "ChatOpenAI")
            break
        except Exception:
            continue
    if ChatOpenAI is None:
        raise ImportError(
            "Could not import ChatOpenAI from langchain.chat_models or langchain_openai"
        )

    try:
        from langchain_tavily import TavilySearch  # type: ignore
    except ImportError:
        TavilySearch = None  # type: ignore

    # OpenRouter is OpenAI-compatible: same ChatOpenAI class, different
    # base_url + model name. OPENAI_API_KEY in .env must be your
    # sk-or-... key. The ":free" suffix is REQUIRED to avoid billing.
    llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    )

    # STEP 3 — the structured evaluator. Returns a QualityScore OBJECT,
    # not a string: result.score is already a validated int in [1, 10].
    evaluator = llm.with_structured_output(QualityScore)

    if TavilySearch is None:
        raise ImportError(
            "langchain_tavily is not installed; install it or set USE_FAKE=1 to run offline."
        )
    search_tool = TavilySearch(max_results=5)  # needs TAVILY_API_KEY

    # OpenRouter has no embeddings endpoint → local HF embeddings if
    # installed (uv sync --group embeddings), else deterministic fakes.
    # Embeddings only power the RAG bonus, not the core graph.
    try:
        from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        embeddings = DeterministicFakeEmbedding(size=256)

    vector_store = InMemoryVectorStore(embeddings)




# ============================================================
# STEP 4 — NODES

# ============================================================
# STEP 4 — NODES
# ============================================================
# A node is just a function: takes state, returns a PARTIAL update
# (a dict with ONLY the keys it changed). LangGraph merges it in.
# Do NOT mutate state in place; do NOT return the whole state.
#
# WHERE TO LOOK: Use Graph API docs → "Define and update state".
#   https://docs.langchain.com/oss/python/langgraph/use-graph-api




#                                  STEP 4 — NODES

def collect_node(state: AgentState):
    """Search the web. On retries, CHANGE the query — a loop that
    repeats the identical action can never produce a different result."""
    iteration = state["iteration_count"] + 1

    # A different angle per iteration (rule (a) of loop termination):
    angles = {
        1: f"{state['topic']} overview 2026",
        2: f"{state['topic']} case studies implementation challenges",
        3: f"{state['topic']} ROI metrics production deployments",
    }
    query = angles.get(iteration, f"{state['topic']} latest developments")

    results = search_tool.invoke({"query": query})["results"]

    sources = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in results
    ]

    return {
        "search_query": query,
        "collected_data": sources,
        "iteration_count": iteration,
        "execution_logs": [
            f"[{datetime.now():%H:%M:%S}] collect (iter {iteration}): "
            f"'{query}' → {len(sources)} sources"
        ],
    }


def store_memory_node(state: AgentState):
    """Save source contents into the vector store (long-term memory)."""
    texts = [s["content"] for s in state["collected_data"] if s["content"]]
    if texts:
        vector_store.add_texts(texts)
    return {
        "execution_logs": [
            f"[{datetime.now():%H:%M:%S}] store_memory: {len(texts)} chunks embedded"
        ]
    }


def analyze_node(state: AgentState):
    """LLM-analyze each source, enriched with related past research
    retrieved from the vector store — that retrieval step is the RAG."""
    analyzed = []
    for source in state["collected_data"]:
        related = vector_store.similarity_search(source["content"], k=2)
        related_context = "\n".join(d.page_content[:200] for d in related)

        prompt = (
            f"Topic: {state['topic']}\n\n"
            f"Source: {source['title']}\n{source['content']}\n\n"
            f"Related prior research:\n{related_context}\n\n"
            "Extract the 2-3 most important insights as concise bullet points."
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        analyzed.append(
            {
                "title": source["title"],
                "url": source["url"],
                "insights": response.content,
            }
        )

    return {
        "analyzed_data": analyzed,
        "execution_logs": [
            f"[{datetime.now():%H:%M:%S}] analyze: {len(analyzed)} sources analyzed"
        ],
    }


def evaluate_node(state: AgentState):
    """Score the research with the STRUCTURED evaluator. result is a
    QualityScore object — no fragile int() parsing of free text."""
    summary = "\n".join(a["insights"] for a in state["analyzed_data"])
    result = evaluator.invoke(
        [
            HumanMessage(
                content=(
                    f"Rate this research on '{state['topic']}' from 1-10 for "
                    f"depth, breadth, and usefulness to an enterprise reader.\n\n"
                    f"{summary}"
                )
            )
        ]
    )
    return {
        "quality_score": result.score,
        "execution_logs": [
            f"[{datetime.now():%H:%M:%S}] evaluate: score={result.score} "
            f"({result.reasoning})"
        ],
    }


def report_node(state: AgentState):
    """Generate the enterprise report from analyzed_data."""
    insights = "\n\n".join(
        f"### {a['title']}\nSource: {a['url']}\n{a['insights']}"
        for a in state["analyzed_data"]
    )
    response = llm.invoke(
        [
            HumanMessage(
                content=(
                    f"Write a concise enterprise research report on "
                    f"'{state['topic']}' with an executive summary, key "
                    f"findings, and recommendations, based on:\n\n{insights}"
                )
            )
        ]
    )
    return {
        "final_report": response.content,
        "execution_logs": [f"[{datetime.now():%H:%M:%S}] report: generated"],
    }


def audit_node(state: AgentState):
    """Log completion stats — the compliance trail."""
    return {
        "execution_logs": [
            f"[{datetime.now():%H:%M:%S}] audit: done | "
            f"iterations={state['iteration_count']} | "
            f"final_score={state['quality_score']} | "
            f"sources={len(state['collected_data'])}"
        ]
    }




# ============================================================
# STEP 5 — THE CONDITIONAL EDGE (the heart of this lab)
# ============================================================
# Write a router function: takes state, RETURNS THE NAME of the
# next node as a string.
#
# CRITICAL — loops must terminate. Two rules:
#   a) every retry must change something (your query, Step 4.2),
#   b) hard-cap the retries with iteration_count.
# Without both, same search → same score → infinite loop → LangGraph
# kills the run at recursion limit 25 with GraphRecursionError.
#
# WHERE TO LOOK (read BOTH):
#   - "Conditional branching":
#     https://docs.langchain.com/oss/python/langgraph/use-graph-api#conditional-branching
#   - "Create and control loops":
#     https://docs.langchain.com/oss/python/langgraph/use-graph-api#create-and-control-loops
#
# EXPERIMENT: comment out the iteration cap, force low scores, run,
# and read the GraphRecursionError message. Now you understand why
# the docs insist on termination conditions.



def quality_router(state: AgentState) -> str:
    if state["quality_score"] >= 7:
        return "report"
    if state["iteration_count"] >= 3:
        return "report"  # give up gracefully, ship what we have
    return "collect"


# ============================================================
# STEP 6 — WIRE THE GRAPH
# ============================================================
# 1. workflow = StateGraph(AgentState)
# 2. add_node(...) for all six nodes
# 3. add_edge(START, "collect")        <- START, not set_entry_point
# 4. linear edges: collect → store_memory → analyze → evaluate
# 5. add_conditional_edges("evaluate", quality_router,
#        {"collect": "collect", "report": "report"})
#    (the dict maps router RETURN VALUES to NODE NAMES)
# 6. report → audit → END
#
# WHERE TO LOOK: Graph API docs → "Edges".

# TODO: your code here



workflow = StateGraph(AgentState)

workflow.add_node("collect", collect_node)
workflow.add_node("store_memory", store_memory_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("report", report_node)
workflow.add_node("audit", audit_node)

workflow.add_edge(START, "collect")
workflow.add_edge("collect", "store_memory")
workflow.add_edge("store_memory", "analyze")
workflow.add_edge("analyze", "evaluate")

# The dict maps router RETURN VALUES to NODE NAMES.
workflow.add_conditional_edges(
    "evaluate",
    quality_router,
    {"collect": "collect", "report": "report"},
)

workflow.add_edge("report", "audit")
workflow.add_edge("audit", END)



# ============================================================
# STEP 7 — COMPILE with a checkpointer, VISUALIZE, RUN
# ============================================================
# 1. app = workflow.compile(checkpointer=InMemorySaver())
#    A checkpointer saves state after every node → enables resume,
#    time-travel debugging, and human-in-the-loop.
#    WHERE TO LOOK: https://docs.langchain.com/oss/python/langgraph/persistence
#
# 2. Visualize what you built:
#       print(app.get_graph().draw_mermaid())
#    → paste the output into https://mermaid.live
#    Does the picture match the diagram at the top of this file?
#
# 3. Run with STREAMING so you watch state evolve node by node:
#       config = {"configurable": {"thread_id": "run-1"}}  # required
#       for chunk in app.stream(initial_state, config,
#                               stream_mode="values"):
#           ...
#    WHERE TO LOOK: https://docs.langchain.com/oss/python/langgraph/streaming
#
# 4. BONUS — human-in-the-loop: compile with
#       interrupt_before=["report"]
#    then inspect state and resume. WHERE TO LOOK:
#       https://docs.langchain.com/oss/python/langgraph/interrupts

 
# The runtime entrypoint is the __main__ block below.


if __name__ == "__main__":
    app = workflow.compile(checkpointer=InMemorySaver())

    print("=" * 60)
    print("GRAPH (paste into https://mermaid.live):")
    print("=" * 60)
    print(app.get_graph().draw_mermaid())

    initial_state = {
        "topic": "Enterprise Agentic AI Systems",
        "search_query": "",
        "collected_data": [],
        "analyzed_data": [],
        "quality_score": 0,
        "iteration_count": 0,
        "final_report": "",
        "execution_logs": [],
    }

    config = {"configurable": {"thread_id": "run-1"}}  # required by checkpointer

    print("\n" + "=" * 60)
    print(f"RUN (USE_FAKE={USE_FAKE})")
    print("=" * 60)

    final_state = None
    for chunk in app.stream(initial_state, config, stream_mode="values"):
        final_state = chunk
        if chunk["execution_logs"]:
            print(chunk["execution_logs"][-1])

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(final_state["final_report"])

    print("\n" + "=" * 60)
    print("FULL EXECUTION LOG")
    print("=" * 60)
    for line in final_state["execution_logs"]:
        print(line)



# ============================================================
# SELF-CHECK before you look at the solution
# ============================================================
# [ ] My nodes return partial dicts, never the whole mutated state
# [ ] execution_logs uses a reducer, and I can explain why
# [ ] My router has BOTH a quality exit AND an iteration cap
# [ ] Retried searches use a different query than the first attempt
# [ ] I saw the Mermaid diagram and it matches the intended flow
# [ ] I know what GraphRecursionError is and how to trigger it
# [ ] The quality score comes from with_structured_output, not int()
#
# Stuck? Debugging order that works:
#   1. print() the raw return of search_tool.invoke — check its shape
#   2. run app.stream(..., stream_mode="updates") — shows exactly
#      which node produced which state update
#   3. compare your edge wiring against the diagram at the top
#   4. only THEN open day1_lab_solution.py
# ============================================================
