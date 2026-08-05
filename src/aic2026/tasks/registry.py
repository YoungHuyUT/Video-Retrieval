from __future__ import annotations

from collections.abc import Callable

from aic2026.models import Candidate, Query

TaskHandler = Callable[[Query, list[Candidate]], list[Candidate]]


class TaskRegistry:
    """Đăng ký query type mới mà không thay đổi orchestration/retrieval lõi."""
    def __init__(self) -> None:
        self._handlers: dict[str, TaskHandler] = {}

    def register(self, query_type: str, handler: TaskHandler) -> None:
        if not query_type.strip():
            raise ValueError("query_type must not be empty")
        self._handlers[query_type] = handler

    def handler_for(self, query_type: str) -> TaskHandler:
        try:
            return self._handlers[query_type]
        except KeyError as exc:
            known = ", ".join(sorted(self._handlers))
            raise ValueError(f"Unsupported query type '{query_type}'. Registered: {known}") from exc


def _identity(query: Query, candidates: list[Candidate]) -> list[Candidate]:
    return candidates


default_registry = TaskRegistry()
default_registry.register("kis", _identity)
default_registry.register("qa", _identity)
default_registry.register("trake", _identity)
