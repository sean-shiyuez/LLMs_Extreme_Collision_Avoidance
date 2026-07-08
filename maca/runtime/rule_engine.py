"""规则引擎 —— 在线平面执行的"编译后规律"。

一条规则 = 特征语言上的一组守卫（GUARDS）+ 应急策略树载荷（payload）。
所有守卫都是可选项，这正是 schema 可无限扩展的原因：工厂将来启用任何
新特征，无需改动引擎即可被匹配。

守卫两种形式：
  离散守卫：  "approach_sector": ["front_left", "left"]   —— 成员匹配
  数值守卫：  "ttc_s": {"max": 1.3} / {"min": 4.0} / {"min": a, "max": b}

匹配代价最坏 O(规则数 × 守卫数)，任何现实规模的库都是微秒级。
"""
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .. import config
from .features import FeatureVector


@dataclass
class Rule:
    """单条规则。status 生命周期：candidate（候选）-> active（现役）->
    deprecated（废弃，复验失败或被证据修订后退役）。"""
    rule_id: str
    guards: Dict[str, object]         # 守卫条件（特征名 -> 离散列表/数值区间）
    policy: dict                      # 载荷：应急策略树
    rationale: str = ""               # 规则依据（人可读）
    status: str = "candidate"
    priority: int = 0                 # 同特异度平局时的优先级（种子规则较高）
    version: int = 1
    provenance: dict = field(default_factory=dict)   # 来源（场景族可复实例化）
    stats: dict = field(default_factory=dict)        # 电池验证统计

    @property
    def specificity(self) -> int:
        """特异度 = 守卫条数；更具体的规则在匹配排序中优先。"""
        return len(self.guards)

    def matches(self, fv: FeatureVector) -> bool:
        """守卫求值：全部满足才算命中；特征缺失视为不命中。"""
        for key, guard in self.guards.items():
            value = fv.get(key)
            if value is None:
                return False
            if isinstance(guard, list):          # 离散：成员匹配
                if value not in guard:
                    return False
            elif isinstance(guard, dict):        # 数值：闭区间（None 端开放）
                if guard.get("min") is not None and value < guard["min"]:
                    return False
                if guard.get("max") is not None and value > guard["max"]:
                    return False
            else:                                # 标量：相等
                if value != guard:
                    return False
        return True

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "version": self.version,
                "status": self.status, "priority": self.priority,
                "guards": self.guards, "policy": self.policy,
                "rationale": self.rationale, "provenance": self.provenance,
                "stats": self.stats}

    @staticmethod
    def from_dict(d: dict) -> "Rule":
        return Rule(rule_id=d["rule_id"], guards=d["guards"], policy=d["policy"],
                    rationale=d.get("rationale", ""), status=d.get("status", "candidate"),
                    priority=int(d.get("priority", 0)), version=int(d.get("version", 1)),
                    provenance=d.get("provenance", {}), stats=d.get("stats", {}))


class RuleLibrary:
    """规则库：从 rules.jsonl 加载，提供 active 过滤与排序匹配。"""

    def __init__(self, rules: List[Rule]):
        self.rules = rules

    @staticmethod
    def load(path=None) -> "RuleLibrary":
        path = path or config.RULES_PATH
        rules = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rules.append(Rule.from_dict(json.loads(line)))
        return RuleLibrary(rules)

    def active(self) -> List[Rule]:
        """仅现役规则参与在线匹配。"""
        return [r for r in self.rules if r.status == "active"]

    def match(self, fv: FeatureVector) -> Tuple[Optional[Rule], List[Rule], float]:
        """返回 (最优规则, 全部命中规则, 耗时 ms)。

        排序：特异度降序 > 优先级降序 > 电池通过率降序 —— 更具体的规则
        （守卫更多）优先于更宽泛的规则。
        """
        start = time.perf_counter()
        matches = [r for r in self.active() if r.matches(fv)]
        matches.sort(key=lambda r: (-r.specificity, -r.priority,
                                    -(r.stats.get("pass_rate") or 0.0)))
        elapsed = (time.perf_counter() - start) * 1000
        return (matches[0] if matches else None), matches, elapsed
