"""
ml/agents/digital_twin.py
=========================
Digital Twin — Treatment Outcome Simulation (12-month PFS prediction).

Lightweight ODE-based simulation using TCGA-inspired parameters per tissue type.
Outputs a Kaplan-Meier-style progression-free survival (PFS) curve for 12 months.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class PfsPrediction:
    pfs_12mo: float
    curve_points: list[dict]
    model: str
    assumptions: list[str]


_TCGA_PARAMS = {
    "lung_adenocarcinoma": {
        "base_hazard": 0.08,
        "growth_rate": 0.12,
        "treatment_effect": 0.18,
    },
    "lung_squamous_cell_carcinoma": {
        "base_hazard": 0.09,
        "growth_rate": 0.14,
        "treatment_effect": 0.15,
    },
    "colon_adenocarcinoma": {
        "base_hazard": 0.07,
        "growth_rate": 0.10,
        "treatment_effect": 0.14,
    },
}


def simulate_pfs(tissue_type: str, horizon_months: int = 12) -> PfsPrediction:
    """
    Simulate a 12-month progression-free survival curve.

    Uses a simple tumor burden ODE:
      dB/dt = (growth_rate - treatment_effect) * B

    Hazard is scaled by tumor burden to approximate PFS dynamics:
      hazard = base_hazard * (1 + 0.6 * (B - 1))

    Args:
        tissue_type: Tissue type string.
        horizon_months: Months to simulate (default 12).

    Returns:
        PfsPrediction with curve points and summary.
    """
    params = _TCGA_PARAMS.get(tissue_type, {"base_hazard": 0.08, "growth_rate": 0.11, "treatment_effect": 0.14})
    base_hazard = params["base_hazard"]
    growth_rate = params["growth_rate"]
    treatment_effect = params["treatment_effect"]

    dt = 0.1
    total_steps = int(horizon_months / dt)

    burden = 1.0
    survival = 1.0
    points: List[dict] = []

    for step in range(total_steps + 1):
        t = step * dt
        # Diminishing treatment effect after 6 months
        eff = treatment_effect * (1.0 if t <= 6.0 else 0.6)
        burden = max(0.2, burden + (growth_rate - eff) * burden * dt)

        hazard = base_hazard * (1.0 + 0.6 * max(0.0, burden - 1.0))
        survival *= math.exp(-hazard * dt)

        if abs(t - round(t)) < 1e-6:
            points.append({"month": int(round(t)), "survival": round(survival, 4)})

    assumptions = [
        "TCGA-inspired hazard and growth parameters",
        "Single-regimen effect; no resistance modeling",
        "Population-level approximation; not patient-specific",
    ]

    return PfsPrediction(
        pfs_12mo=round(points[-1]["survival"], 4) if points else 0.0,
        curve_points=points,
        model="ODE tumor burden + hazard scaling",
        assumptions=assumptions,
    )
