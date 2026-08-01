"""The full industrial workflow in one script:

1. perform a step test on the real (unknown) plant,
2. fit a FOPDT model from the recorded data,
3. tune a PI controller from the fitted model,
4. verify the controller on the *real* plant.

This is exactly what a control engineer does before commissioning a loop.
"""

import numpy as np
from pidforge import PIDController, compute_metrics, open_loop_step, simulate
from pidforge.identification import fit_fopdt
from pidforge.models import FOPDT
from pidforge.tuning import simc_pi

# ---------------------------------------------------------------------------
# 1. Step test on the "real" plant (with measurement noise)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
real_plant = FOPDT(gain=2.0, tau=5.0, theta=1.0)

step_test = open_loop_step(real_plant, u=1.0, dt=0.02, t_end=60.0)
noisy = step_test.output + rng.normal(0.0, 0.01, step_test.output.shape)

# ---------------------------------------------------------------------------
# 2. Identify a FOPDT model
# ---------------------------------------------------------------------------
fit = fit_fopdt(step_test.time, noisy, input_step=1.0, method="area")
print("Identified:", fit.summary())
print("True plant:", real_plant)

# ---------------------------------------------------------------------------
# 3. Tune the controller from the fitted model
# ---------------------------------------------------------------------------
params = simc_pi(fit.model)
print("Tuning:", params.summary())

# ---------------------------------------------------------------------------
# 4. Verify on the real plant (model-plant mismatch is realistic)
# ---------------------------------------------------------------------------
controller = PIDController(kp=params.kp, ki=params.ki)
res = simulate(
    real_plant,
    controller,
    setpoint=1.0,
    dt=0.02,
    t_end=80.0,
    disturbance_time=40.0,
    disturbance_magnitude=0.5,
)
m = compute_metrics(res)
print("\nClosed-loop performance on the real plant:")
print(f"  overshoot     {m.overshoot_pct:.2f} %")
print(f"  settling time {m.settling_time:.2f} s")
print(f"  IAE           {m.iae:.4f}")

try:
    from pidforge.plotting import plot_response

    fig = plot_response(res, metrics=m, title="PID tuned from step-test data")
    fig.savefig("identify_workflow.png", dpi=150)
    print("\nPlot saved to identify_workflow.png")
except ImportError:
    print("\n(matplotlib not installed; skipping plot)")
