"""Estabilizacion temporal de mensajes y confirmacion de captura."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from src.longitudinal_quality_rules import DECISION_MESSAGES


MESSAGES = DECISION_MESSAGES


@dataclass(frozen=True)
class TemporalResult:
    raw_decision: str
    stable_decision: str
    stable_message: str
    capture_confirmed: int
    capture_streak: int
    window_size_used: int


class TemporalGuidance:
    """Suaviza decisiones y exige una racha antes de autorizar captura."""

    def __init__(
        self,
        window_size: int = 5,
        required_capture_streak: int = 3,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size debe ser >= 1.")
        if required_capture_streak < 1:
            raise ValueError("required_capture_streak debe ser >= 1.")
        self.decisions = deque(maxlen=window_size)
        self.required_capture_streak = required_capture_streak
        self.capture_streak = 0
        self.last_non_capture = "liver_without_reference"

    def _majority_recent_tiebreak(self) -> str:
        counts = Counter(self.decisions)
        maximum = max(counts.values())
        tied = {name for name, count in counts.items() if count == maximum}
        for decision in reversed(self.decisions):
            if decision in tied:
                return decision
        return self.decisions[-1]

    def update(self, raw_decision: str) -> TemporalResult:
        if raw_decision not in MESSAGES:
            raise ValueError(f"Decision no soportada: {raw_decision}")
        self.decisions.append(raw_decision)
        if raw_decision == "capture":
            self.capture_streak += 1
        else:
            self.capture_streak = 0
            self.last_non_capture = raw_decision

        if raw_decision == "no_structure":
            stable = "no_structure"
        else:
            stable = self._majority_recent_tiebreak()
            if stable == "capture" and self.capture_streak < self.required_capture_streak:
                stable = self.last_non_capture

        confirmed = int(
            stable == "capture"
            and self.capture_streak >= self.required_capture_streak
        )
        return TemporalResult(
            raw_decision=raw_decision,
            stable_decision=stable,
            stable_message=MESSAGES[stable],
            capture_confirmed=confirmed,
            capture_streak=self.capture_streak,
            window_size_used=len(self.decisions),
        )
