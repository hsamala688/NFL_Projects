"""
Orchestration layer: run the ADK agent and assemble a structured AnalystResponse
from the real event stream. Evidence and tool_calls are captured from execution;
the model produces only answer, confidence, and caveats.
"""
import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel
import google.genai.types as genai_types
from google.adk.runners import InMemoryRunner

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class EvidenceRow(BaseModel):
    tool: str
    query: str
    rows: list[dict[str, Any]]


class ToolCallRecord(BaseModel):
    tool: str
    inputs: dict[str, Any]
    play_count: int | None = None


class AnalystResponse(BaseModel):
    answer: str
    evidence: list[EvidenceRow]
    tool_calls: list[ToolCallRecord]
    confidence: str
    caveats: str


async def ask_async(question: str) -> AnalystResponse:
    from agent.agent import root_agent

    runner = InMemoryRunner(agent=root_agent, app_name="nfl_analyst")
    session = await runner.session_service.create_session(
        app_name="nfl_analyst", user_id="cli"
    )

    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=question)],
    )

    pending_calls: dict[str, tuple[str, dict]] = {}
    tool_calls: list[ToolCallRecord] = []
    evidence: list[EvidenceRow] = []
    raw_answer = ""

    async for event in runner.run_async(
        user_id="cli",
        session_id=session.id,
        new_message=message,
    ):
        for fc in event.get_function_calls():
            args = fc.args or {}
            call_id = fc.id or f"{fc.name}_{len(pending_calls)}"
            pending_calls[call_id] = (fc.name, args)
            tool_calls.append(ToolCallRecord(tool=fc.name, inputs=args))

        for fr in event.get_function_responses():
            result = fr.response or {}
            call_id = fr.id
            if call_id and call_id in pending_calls:
                tool_name, args = pending_calls.pop(call_id)
            else:
                tool_name = fr.name
                args = {}

            if tool_name == "query_marts" and result.get("status") == "success":
                evidence.append(EvidenceRow(
                    tool="query_marts",
                    query=args.get("sql", ""),
                    rows=result.get("rows", []),
                ))

            elif tool_name == "run_cpoe" and result.get("status") == "success":
                season = args.get("season", "")
                play_count = result.get("play_count")
                evidence.append(EvidenceRow(
                    tool="run_cpoe",
                    query=f"run_cpoe(season={season})",
                    rows=result.get("rows", []),
                ))
                for tc in reversed(tool_calls):
                    if tc.tool == "run_cpoe" and tc.inputs == args:
                        tc.play_count = play_count
                        break

        if event.is_final_response() and event.content:
            for part in event.content.parts or []:
                if part.text:
                    raw_answer += part.text

    answer, confidence, caveats = _parse_response(raw_answer.strip())

    return AnalystResponse(
        answer=answer,
        evidence=evidence,
        tool_calls=tool_calls,
        confidence=confidence,
        caveats=caveats,
    )


def _parse_response(text: str) -> tuple[str, str, str]:
    confidence = ""
    caveats = ""
    body_lines = []
    for line in text.splitlines():
        if line.startswith("CONFIDENCE:"):
            confidence = line[len("CONFIDENCE:"):].strip()
        elif line.startswith("CAVEATS:"):
            caveats = line[len("CAVEATS:"):].strip()
        else:
            body_lines.append(line)
    answer = "\n".join(body_lines).strip()
    return answer, confidence, caveats


def ask(question: str) -> AnalystResponse:
    return asyncio.run(ask_async(question))
