"""Quickstart: tune, simulate and evaluate a temperature-like FOPDT loop."""

from pidforge import FOPDT, PIDController, compute_metrics, simulate
from pidforge.tuning import simc_pi

# A heated-tank temperature loop: K = 2 C/% valve, tau = 5 s, dead time 1 s.
plant = FOPDT(gain=2.0, tau=5.0, theta=1.0)

# SIMC tuning (PI).
params = simc_pi(plant)
print("Tuning:", params.summary())

controller = PIDController(
    kp=params.kp, ki=params.ki, output_limits=(0.0, 100.0)
)

# Setpoint step from 0 to 50 C.
result = simulate(
    plant,
    controller,
    setpoint=50.0,
    dt=0.01,
    t_end=60.0,
)

metrics = compute_metrics(result)
print("\nPerformance (setpoint step):")
for key, value in metrics.as_dict().items():
    print(f"  {key:<20} {value:.4g}")

try:
    from pidforge.plotting import plot_response

    fig = plot_response(result, metrics=metrics, title="pidforge quickstart")
    fig.savefig("quickstart_response.png", dpi=150)
    print("\nPlot saved to quickstart_response.png")
except ImportError:
    print("\n(matplotlib not installed; skipping plot)")
