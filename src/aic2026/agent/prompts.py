from __future__ import annotations

from importlib.resources import files
from aic2026.models import Query

COMMON_GUARDRAILS = files("aic2026.agent").joinpath("templates/system_agent_en.md").read_text(encoding="utf-8").strip()

PLANNER_SYSTEM = COMMON_GUARDRAILS + """

Your task is retrieval planning, not final-answer generation.
Produce 1-6 short visual query variants that preserve stated objects, actions, attributes, and context.
For Vietnamese queries, an English variant is allowed. Do not add facts absent from the user query.
For TRAKE, split the sequence into atomic events in chronological order from video start to end.
"""

JUDGE_SYSTEM = COMMON_GUARDRAILS + """

Your task is evidence selection after retrieval.
`selected_vector_ids` may contain only IDs present in the evidence list.
KIS: prioritize the frame that visually matches the described event; do not generate prose answers.
Q&A: select relevant frames only; your textual answer is ignored because a local VLM is the image-answer source.
TRAKE: choose evidence from videos with a plausible event sequence; deterministic DP selects one frame per event.
"""


def task_instruction(query: Query) -> str:
    if query.type == "kis":
        return "Every candidate output must contain exactly: video_id, frame_id."
    if query.type == "qa":
        return "Every candidate output must contain exactly: video_id, frame_id, answer from local VLM."
    if query.type == "trake":
        return f"Every candidate output must contain video_id and {len(query.events)} frame IDs in event order."
    return "An extension query type must register a task handler and output adapter before execution."
