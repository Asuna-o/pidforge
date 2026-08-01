"""Compare tuning rules head-to-head on the same FOPDT plant.

Shows why tuning choice matters: Ziegler-Nichols is fast but oscillatory,
SIMC balances speed and robustness, IMC with a slow lambda is gentle.
"""

from pidforge import FOPDT, PIDController, compute_metrics, simulate
from pidforge.tuning import (
    cohen_coon,
    imc_pi,
    simc_pi,
    ziegler_nichols_open_loop,
)

PLANT = FOPDT(gain=2.0, tau=5.0, theta=1.0)

RULES = {
    "Ziegler-Nichols": ziegler_nichols_open_loop(PLANT, "PID"),
    "Cohen-Coon": cohen_coon(PLANT, "PID"),
    "IMC (tau_c=3)": imc_pi(PLANT, tau_c=3.0),
    "SIMC": simc_pi(PLANT),
}

results = {}
for name, params in RULES.items():
    controller = PIDController(kp=params.kp, ki=params.ki, kd=params.kd)
    res = simulate(PLANT, controller, setpoint=1.0, dt=0.01, t_end=80.0)
    results[name] = res

print(f"{'Rule':<18} {'OS %':>6} {'ts (s)':>8} {'IAE':>8} {'TV':>8}")
print("-" * 52)
for name, res in results.items():
    m = compute_metrics(res)
    print(
        f"{name:<18} {m.overshoot_pct:6.1f} {m.settling_time:8.2f} "
        f"{m.iae:8.3f} {m.total_variation:8.1f}"
    )

try:
    from pidforge.plotting import plot_comparison

    fig = plot_comparison(results)
    fig.savefig("tuning_comparison.png", dpi=150)
    print("\nPlot saved to tuning_comparison.png")
except ImportError:
    print("\n(matplotlib not installed; skipping plot)")
