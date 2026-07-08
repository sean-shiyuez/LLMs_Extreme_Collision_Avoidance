"""案例 k-NN 索引 —— 基于"本地数值特征向量"的近邻检索。

查询路径上没有任何 API embedding 调用：向量由 features.py 从场景物理量
本地构造，检索是纯 numpy 余弦相似度 —— 10^5 案例暴力检索约 1 ms 量级，
案例库规模不威胁实时性。

案例覆盖"扫描发现无明显规律"的区域（有害的严重度接近平局、涉及弱势
道路使用者的权衡），由离线委员会标注产生。
"""
import json
import time
from typing import List, Optional, Tuple

import numpy as np

from .. import config
from .features import FeatureVector


class CaseLibrary:
    """案例库：加载 cases.jsonl（跳过已退役案例）并建立归一化向量矩阵。"""

    def __init__(self, cases: List[dict]):
        self.cases = cases
        self._matrix = None
        if cases:
            m = np.array([c["features"]["vector"] for c in cases], dtype=np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = m / norms   # 行归一化：点积即余弦相似度

    @staticmethod
    def load(path=None) -> "CaseLibrary":
        path = path or config.CASES_PATH
        cases = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        case = json.loads(line)
                        # 被离线冲突裁决否决的案例已退役，不进在线索引
                        if not case.get("retired"):
                            cases.append(case)
        return CaseLibrary(cases)

    def knn(self, fv: FeatureVector, k: int = config.CASE_KNN_K
            ) -> Tuple[Optional[dict], List[Tuple[dict, float]], float]:
        """返回 (首个过阈值的案例, top-k [(案例, 相似度)], 耗时 ms)。"""
        start = time.perf_counter()
        if self._matrix is None:
            return None, [], (time.perf_counter() - start) * 1000
        q = np.asarray(fv.vector, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            return None, [], (time.perf_counter() - start) * 1000
        sims = self._matrix @ (q / qn)          # 一次矩阵-向量乘 = 全库相似度
        order = np.argsort(-sims)[:k]
        top = [(self.cases[i], float(sims[i])) for i in order]
        best = next((c for c, s in top if s >= config.CASE_SIM_THRESHOLD), None)
        return best, top, (time.perf_counter() - start) * 1000
