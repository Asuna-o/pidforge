"""Closed-loop performance metrics.

Computes the standard time-domain criteria used to compare controllers:

* transient: rise time, settling time, overshoot, steady-state error;
* integral criteria: IAE, ISE, ITAE, ITSE;
* actuator activity: total variation of the control signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulate import SimulationResult


@dataclass
class Metrics:
    """Time-domain performance metrics of a closed-loop simulation."""

    rise_time: float
    settling_time: float
    overshoot_pct: float
    steady_state_error: float
    iae: float
    ise: float
    itae: float
    itse: float
    total_variation: float

    def as_dict(self) -> dict:
        """Plain float dictionary, convenient for tables and CSV export."""
        return {
            "rise_time": self.rise_time,
            "settling_time": self.settling_time,
            "overshoot_pct": self.overshoot_pct,
            "steady_state_error": self.steady_state_error,
            "iae": self.iae,
            "ise": self.ise,
            "itae": self.itae,
            "itse": self.itse,
            "total_variation": self.total_variation,
        }


def compute_metrics(
    result: SimulationResult,
    settle_tolerance: float = 0.02,
    rise_band: tuple = (0.10, 0.90),
) -> Metrics:
    """Compute metrics for a step-oriented simulation result.

    ``settle_tolerance`` is the fraction of the step change allowed in the
    settling band (default 2%). ``rise_band`` is the rise-time band.
    """
    t = result.time
    y = result.output
    u = result.control
    sp = result.setpoint

    target = float(sp[-1])
    y0 = float(y[0])
    step = target - y0
    dt = float(t[1] - t[0]) if t.size > 1 else 1.0

    # Rise time: first crossing of 10% -> 90% band.
    rise_time = float("nan")
    if abs(step) > 1e-12:
        lo, hi = (y0 + rise_band[0] * step, y0 + rise_band[1] * step)
        cross_lo = np.flatnonzero((y - lo) * step >= 0)
        cross_hi = np.flatnonzero((y - hi) * step >= 0)
        if cross_lo.size and cross_hi.size:
            t_lo, t_hi = t[cross_lo[0]], t[cross_hi[0]]
            if t_hi >= t_lo:
                rise_time = t_hi - t_lo

    # Settling time: the last time the output leaves the tolerance band.
    band = settle_tolerance * abs(step)
    outside = np.abs(y - target) > band
    if outside.any():
        settling_time = float(t[int(np.flatnonzero(outside)[-1])])
    else:
        settling_time = 0.0

    # Overshoot relative to the step change.
    if abs(step) > 1e-12:
        overshoot = max(0.0, (np.max(y - target) * step) / (step * step))
        overshoot_pct = overshoot * 100.0
    else:
        overshoot_pct = 0.0

    # Steady-state error.
    steady_state_error = abs(target - float(y[-1]))

    # Integral criteria (approximated with the rectangle rule).
    e = sp - y
    iae = float(np.sum(np.abs(e)) * dt)
    ise = float(np.sum(e * e) * dt)
    itae = float(np.sum(t * np.abs(e)) * dt)
    itse = float(np.sum(t * e * e) * dt)

    # Actuator activity.
    total_variation = float(np.sum(np.abs(np.diff(u))))

    return Metrics(
        rise_time=rise_time,
        settling_time=settling_time,
        overshoot_pct=overshoot_pct,
        steady_state_error=steady_state_error,
        iae=iae,
        ise=ise,
        itae=itae,
        itse=itse,
        total_variation=total_variation,
    )


def compare(results: dict) -> dict:
    """Compute metrics for a mapping ``name -> SimulationResult``."""
    return {name: compute_metrics(res) for name, res in results.items()}


__all__: list[str] = ["Metrics", "compute_metrics", "compare"]
