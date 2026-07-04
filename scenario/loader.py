"""Load scenario timelines from maca/scenario/scenarios/*.json."""
import json
from pathlib import Path
from typing import List

from .schema import Scenario, scenario_from_dict

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"


def available_scenarios() -> List[str]:
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.json"))


def load_scenario(name: str) -> Scenario:
    path = SCENARIOS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Scenario '{name}' not found. Available: {', '.join(available_scenarios())}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return scenario_from_dict(json.load(f))
