"""Prueba funcional de estabilizacion temporal sin cargar modelos."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.temporal_guidance import TemporalGuidance


if __name__ == "__main__":
    guidance = TemporalGuidance(window_size=5, required_capture_streak=3)
    sequence = [
        "liver_without_reference",
        "capture",
        "capture",
        "capture",
    ]
    results = [guidance.update(decision) for decision in sequence]
    assert results[-1].capture_confirmed == 1
    assert results[-1].stable_decision == "capture"

    emergency = guidance.update("no_structure")
    assert emergency.stable_decision == "no_structure"
    assert emergency.capture_streak == 0

    print("TemporalGuidance OK")
    for result in results:
        print(result)
