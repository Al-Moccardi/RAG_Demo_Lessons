"""
advanced/react_agent.py — ReAct (Reasoning + Acting) Agent
============================================================
THE IDEA  (Yao et al., 2023 — "ReAct: Synergizing Reasoning and Acting")
-------------------------------------------------------------------------
A plain RAG pipeline does ONE retrieval and ONE generation. That's
fine for simple questions but breaks on multi-step ones like:

    "Compare the chunk size and embedding dimension I'm using right now."

To answer that the system has to:
  1. Look up "chunk size" → find the value
  2. Look up "embedding dimension" → find the value
  3. Compute a comparison

ReAct lets the LLM PLAN, EXECUTE, and OBSERVE in a loop:

    Thought 1: I need to find the chunk size first.
    Action 1: search("chunk size")
    Observation 1: <retrieved chunks>

    Thought 2: Now I need the embedding dimension.
    Action 2: search("embedding dimension")
    Observation 2: <retrieved chunks>

    Thought 3: I have both values. Compute the comparison.
    Action 3: finish("The chunk size is 500 chars; the embedding is 768-dim...")

THE TOOLS WE EXPOSE
-------------------
  • search(query)    — does a FAISS lookup, returns top-3 chunks
  • calculator(expr) — evaluates an arithmetic expression (safely)
  • finish(answer)   — terminate the loop with the final answer

EXTENSIBILITY
-------------
Real ReAct agents (LangChain, AutoGen) expose dozens of tools:
calendar, web search, SQL, code interpreter, etc. We keep two tools
so students can SEE the loop without getting lost in plumbing.

A WARNING ON SMALL LLMS
-----------------------
ReAct depends on the LLM's ability to follow a strict format:

    Thought: ...
    Action: tool_name(arg)
    Observation: ...

Models below ~3B parameters often break the format mid-loop. Our
parser is defensive — if a step doesn't look right, we stop the loop
and synthesize the best answer we can.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from langchain_core.documents import Document


# ────────────────────────────────────────────────────────────
# Trace data structures (for the playground UI)
# ────────────────────────────────────────────────────────────

@dataclass
class ReActStep:
    """A single Thought → Action → Observation triplet."""
    thought: str = ""
    action_name: str = ""
    action_input: str = ""
    observation: str = ""


@dataclass
class ReActTrace:
    """Full trace of a ReAct run, for visualization."""
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    elapsed: float = 0.0
    n_iterations: int = 0
    stopped_reason: str = ""  # "finish", "max_iterations", "parse_error"


# ────────────────────────────────────────────────────────────
# Tools (safe, sandboxed)
# ────────────────────────────────────────────────────────────

def _tool_search(query: str, vectorstore, k: int = 3) -> str:
    """FAISS retrieval tool — returns a compact text observation."""
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return "No matching chunks found."

    # Format compactly so the LLM can fit it in its context.
    parts = []
    for i, (doc, score) in enumerate(results, 1):
        snippet = doc.page_content[:250].replace("\n", " ")
        parts.append(f"[{i}] (score={score:.2f}) {snippet}")
    return "\n".join(parts)


def _tool_calculator(expression: str) -> str:
    """
    Tiny calculator. Uses eval() but only on a heavily restricted set of
    characters — no names, no function calls. For a teaching project
    this is acceptable; for production you'd use a real AST walker.
    """
    if not re.fullmatch(r"[\d\s\+\-\*\/\.\(\)\%]+", expression):
        return "Calculator error: only numbers and + - * / ( ) % are allowed."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{result}"
    except Exception as e:
        return f"Calculator error: {e}"


# ────────────────────────────────────────────────────────────
# Prompt template
# ────────────────────────────────────────────────────────────

_REACT_PROMPT = """You are a ReAct agent. Answer the user's question by reasoning step-by-step and using tools.

You have access to these tools:
  - search(query)         : look up information in a knowledge base
  - calculator(expression): evaluate a basic arithmetic expression
  - finish(answer)        : output the final answer and STOP

Use this exact format for each step:

Thought: <what you're thinking and what you need to do next>
Action: <tool_name>(<argument>)
Observation: <the tool's response will appear here>

Repeat the Thought/Action/Observation block as many times as needed.
When you have enough information, use Action: finish(<your answer>).

Question: {question}

{scratchpad}"""


# Regex to parse a Thought + Action line out of the LLM output.
_ACTION_RE = re.compile(
    r"Thought:\s*(?P<thought>.+?)\n+Action:\s*(?P<name>search|calculator|finish)\s*\(\s*(?P<arg>.*?)\s*\)",
    re.DOTALL | re.IGNORECASE,
)


# ────────────────────────────────────────────────────────────
# Main ReAct loop
# ────────────────────────────────────────────────────────────

def run_react(
    question: str,
    llm,
    vectorstore,
    max_iterations: int = 4,
) -> ReActTrace:
    """
    Run the ReAct loop until the model emits `finish(...)` or we
    hit `max_iterations`.

    Args:
        question       : the user's question
        llm            : a HuggingFacePipeline
        vectorstore    : a FAISS instance (for the search tool)
        max_iterations : hard cap on loop iterations (safety)

    Returns:
        A ReActTrace with every Thought/Action/Observation triplet,
        the final answer, and timing info.
    """
    t0 = time.time()
    trace = ReActTrace()

    # The "scratchpad" is the running conversation log we feed back
    # to the LLM at each iteration so it remembers what it has done.
    scratchpad = ""

    hf_pipe = llm.pipeline

    for iteration in range(max_iterations):
        prompt = _REACT_PROMPT.format(question=question, scratchpad=scratchpad)

        # Generate ONE step. Lower temp than for normal RAG — we want
        # crisp formatting compliance, not creativity.
        try:
            outputs = hf_pipe(
                prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                return_full_text=False,
                pad_token_id=hf_pipe.tokenizer.pad_token_id
                              or hf_pipe.tokenizer.eos_token_id,
            )
            raw = outputs[0]["generated_text"].strip() if outputs else ""
        except Exception as e:
            trace.stopped_reason = f"llm_error: {e}"
            break

        # Parse the next Thought + Action.
        m = _ACTION_RE.search(raw)
        if not m:
            # The LLM broke the format — bail out gracefully.
            trace.stopped_reason = "parse_error"
            # Salvage whatever it generated as the final answer.
            trace.final_answer = (
                raw.split("Observation:")[0]
                   .replace("Thought:", "")
                   .replace("Action:", "")
                   .strip()
            )
            break

        thought      = m.group("thought").strip()
        action_name  = m.group("name").lower().strip()
        action_input = m.group("arg").strip().strip('"\'')

        step = ReActStep(
            thought=thought,
            action_name=action_name,
            action_input=action_input,
        )

        # Dispatch to the requested tool.
        if action_name == "finish":
            step.observation = "(loop terminated)"
            trace.final_answer = action_input
            trace.steps.append(step)
            trace.stopped_reason = "finish"
            break
        elif action_name == "search":
            step.observation = _tool_search(action_input, vectorstore)
        elif action_name == "calculator":
            step.observation = _tool_calculator(action_input)
        else:
            step.observation = f"Unknown tool: {action_name}"

        trace.steps.append(step)

        # Append this step to the scratchpad so the next iteration sees it.
        scratchpad += (
            f"Thought: {thought}\n"
            f"Action: {action_name}({action_input})\n"
            f"Observation: {step.observation}\n\n"
        )

    else:
        # Loop exhausted without finish().
        trace.stopped_reason = "max_iterations"
        # Take the best guess: use the last observation as the answer.
        if trace.steps:
            trace.final_answer = (
                "I ran out of iterations. Best partial answer:\n\n"
                f"{trace.steps[-1].observation}"
            )

    trace.n_iterations = len(trace.steps)
    trace.elapsed = time.time() - t0
    return trace
