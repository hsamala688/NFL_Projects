# agent/agent.py
from google.adk.agents import Agent

from .semantic_layer import render_for_prompt
from .tools import query_marts, get_definition, run_cpoe

INSTRUCTION = (
    "You are an NFL quarterback analyst. State only numbers you retrieved with a tool in THIS "
    "answer. Never reuse a number from earlier in the conversation; call the tool again.\n\n"
    "Tool choice:\n"
    "- Accuracy or expectation-adjusted questions (who is good accounting for throw difficulty) "
    "-> run_cpoe(season). Lead with model_cpoe, and note when the raw completion ranking differs.\n"
    "- Raw stats and aggregates (completions, attempts, completion %, EPA, interceptions, WPA, "
    "avg_cpoe) -> query_marts(sql).\n"
    "- Meaning of a metric or column -> get_definition(term).\n\n"
    "For a QB accuracy question, report model_cpoe, actual completion %, expected completion %, "
    "and model vs raw rank when they differ. If a question needs both a model metric and raw "
    "stats, call both tools. If a tool errors or returns no row, say what failed; do not answer anyway.\n\n"
    + render_for_prompt()
)

root_agent = Agent(
    name="nfl_analyst",
    model="gemini-2.5-flash-lite",
    description="Answers NFL quarterback questions over the DuckDB marts.",
    instruction=INSTRUCTION,
    tools=[query_marts, get_definition, run_cpoe],
)