"""
Flow scoring utilities: Q, P_flow update, UCB/PUCT score calculation.
All core formulae centralized here for easier inspection.
"""

import math
import numpy as np
from typing import Dict, Any
from config.config import CPUCT, ALPHA_FLOW, GAMMA_DECAY

EPS = 1e-12


def q_value(edge_R_sum: Dict[Any, float], edge_N: Dict[Any, int], child) -> float:
    n = edge_N.get(child, 0)
    if n <= 0:
        return 0.0
    return edge_R_sum.get(child, 0.0) / (n + EPS)


def pflow_from_edges(parent_F: float, edge_F: Dict[Any, float], child) -> float:
    # fallback to uniform if parent_F==0
    if parent_F <= 0:
        return 1.0
    return edge_F.get(child, 0.0) / (parent_F + EPS)


def pflow_update(old_p: float, r_tilde: float, alpha: float = ALPHA_FLOW) -> float:
    return (1 - alpha) * old_p + alpha * r_tilde


def normalize_pflows(pdict: Dict[Any, float]) -> Dict[Any, float]:
    total = sum(pdict.values())
    if total <= 0:
        # fallback to uniform
        n = len(pdict) if pdict else 1
        return {k: 1.0 / n for k in pdict}
    return {k: v / total for k, v in pdict.items()}


def ucb_puct_score(parent, child, c_puct: float = CPUCT) -> float:
    """
    Compute UCB/PUCT score per formula:
    U = Q + c_puct * P_flow * sqrt(N(parent)) / (1 + N(parent->child))
    - parent: node object with attributes edge_R_sum, edge_N, P_flow (dict), N, F
    - child: child node object
    """
    # Q
    q = q_value(parent.edge_R_sum, parent.edge_N, child)

    # P_flow prior
    p = 0.0
    if hasattr(parent, 'P_flow') and child in parent.P_flow:
        p = parent.P_flow.get(child, 0.0)
    else:
        # fallback to edge ratio
        p = pflow_from_edges(parent.F, parent.edge_F, child)

    # visits
    parent_N = max(1, getattr(parent, 'N', 0))
    edge_n = parent.edge_N.get(child, 0)

    # if child never visited, give strong exploration boost
    if edge_n == 0:
        exploration = float('inf')
    else:
        exploration = c_puct * p * math.sqrt(parent_N) / (1 + edge_n)

    return q + exploration


def decayed_reward(raw_reward: float, t: int, T: int, gamma: float = GAMMA_DECAY) -> float:
    # R_tilde = R * gamma^(T - t)
    return raw_reward * (gamma ** (T - t))
