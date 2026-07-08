"""场景加载器：从 maca/scenario/scenarios/*.json 读取场景时间线。"""
import json
from pathlib import Path
from typing import List

from .schema import Scenario, scenario_from_dict

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"


def available_scenarios() -> List[str]:
    """列出全部可用场景名（即 JSON 文件名，不含扩展名）。"""
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.json"))


def load_scenario(name: str) -> Scenario:
    """按名称加载场景；不存在时报错并列出可选项。"""
    path = SCENARIOS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Scenario '{name}' not found. Available: {', '.join(available_scenarios())}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return scenario_from_dict(json.load(f))
