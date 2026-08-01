import numpy as np

from pidforge import FOPDT, PIDController, compute_metrics, simulate
from pidforge.tuning import simc_pi


def _well_tuned_result(t_end=80.0):
    plant = FOPDT(gain=2.0, tau=5.0, theta=1.0)
    params = simc_pi(plant)
    ctrl = PIDController(kp=params.kp, ki=params.ki)
    return simulate(plant, ctrl, setpoint=1.0, dt=0.01, t_end=t_end)


def test_metrics_values_are_finite_and_sane():
    res = _well_tuned_result()
    m = compute_metrics(res)
    assert np.isfinite(m.iae) and m.iae > 0
    assert m.ise > 0 and m.itae > 0 and m.itse > 0
    assert m.total_variation > 0
    assert m.steady_state_error < 5e-2
    assert 0 <= m.overshoot_pct < 30
    assert m.settling_time < 80.0
    assert m.rise_time > 0


def test_p_only_has_no_overshoot():
    plant = FOPDT(gain=1.0, tau=5.0, theta=0.0)
    ctrl = PIDController(kp=2.0)
    res = simulate(plant, ctrl, setpoint=1.0, dt=0.01, t_end=60.0)
    m = compute_metrics(res)
    assert m.overshoot_pct < 1e-6


def test_aggressive_pid_overshoots():
    from pidforge.tuning import ziegler_nichols_open_loop

    plant = FOPDT(gain=2.0, tau=5.0, theta=1.0)
    params = ziegler_nichols_open_loop(plant, "PID")
    ctrl = PIDController(kp=params.kp, ki=params.ki, kd=params.kd)
    res = simulate(plant, ctrl, setpoint=1.0, dt=0.01, t_end=40.0)
    m = compute_metrics(res)
    assert m.overshoot_pct > 5.0


def test_settling_time_definition():
    """Everything after the settling time must stay inside the band."""
    res = _well_tuned_result()
    m = compute_metrics(res, settle_tolerance=0.05)
    assert m.settling_time < 50.0
    tail = res.output[res.time > m.settling_time]  # strictly after the last crossing
    assert np.all(np.abs(tail - res.setpoint[-1]) <= 0.05)
