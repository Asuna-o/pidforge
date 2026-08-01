"""Optional plotting helpers (requires ``pip install pidforge[plot]``).

All functions import matplotlib lazily so that the core library works
without it.
"""

from __future__ import annotations

from .metrics import Metrics, compute_metrics
from .simulate import SimulationResult


def _require_plt():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotting requires matplotlib; install with `pip install pidforge[plot]`"
        ) from exc


def plot_response(
    result: SimulationResult,
    metrics: Metrics | None = None,
    title: str | None = None,
):
    """Plot setpoint, process output and control signal for one simulation.

    Returns the matplotlib ``Figure``.
    """
    plt = _require_plt()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_y, ax_u = axes
    ax_y.plot(result.time, result.setpoint, "k--", lw=1.2, label="setpoint")
    ax_y.plot(result.time, result.output, color="#1f77b4", lw=1.8, label="process output")
    ax_y.set_ylabel("Process variable")
    ax_y.legend(loc="best")
    ax_y.grid(True, alpha=0.3)
    if metrics is not None:
        ax_y.set_title(
            f"{title or 'Closed-loop response'} — "
            f"overshoot {metrics.overshoot_pct:.1f}%, "
            f"settling {metrics.settling_time:.2f} s"
        )

    ax_u.plot(result.time, result.control, color="#d62728", lw=1.4, label="control")
    ax_u.set_xlabel("Time (s)")
    ax_u.set_ylabel("Control output")
    ax_u.legend(loc="best")
    ax_u.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_comparison(
    results: dict[str, SimulationResult],
    metrics: dict[str, Metrics] | None = None,
):
    """Overlay several closed-loop responses (one curve per result)."""
    plt = _require_plt()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax_y, ax_u = axes
    colors = plt.cm.tab10.colors  # type: ignore

    if metrics is None:
        metrics = {name: compute_metrics(res) for name, res in results.items()}

    for idx, (name, res) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        m = metrics[name]
        ax_y.plot(
            res.time, res.output, color=color, lw=1.6,
            label=f"{name} (OS {m.overshoot_pct:.1f}%, ts {m.settling_time:.2f}s)",
        )
        ax_u.plot(res.time, res.control, color=color, lw=1.2, alpha=0.85, label=name)
        ax_y.plot(res.time, res.setpoint, "k--", lw=1.0, alpha=0.6)

    ax_y.set_ylabel("Process variable")
    ax_y.legend(loc="best", fontsize=8)
    ax_y.grid(True, alpha=0.3)
    ax_u.set_xlabel("Time (s)")
    ax_u.set_ylabel("Control output")
    ax_u.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


__all__: list[str] = ["plot_response", "plot_comparison"]
