import numpy as np
import pytest

from pidforge import (
    FOPDT,
    IntegratorDelay,
    PIDController,
    simulate,
    step_setpoint,
)


def test_dead_time_is_exact():
    """Output must stay at zero until the dead time has elapsed."""
    plant = FOPDT(gain=1.0, tau=5.0, theta=2.0)
    ctrl = PIDController(kp=0.5)
    res = simulate(plant, ctrl, setpoint=1.0, dt=0.01, t_end=10.0)
    before = res.output[res.time < 1.9]
    assert np.all(np.abs(before) < 1e-6)
    # After the delay the loop begins responding.
    after = res.output[res.time > 2.4]
    assert np.any(after > 1e-3)


def test_setpoint_trajectory_and_step_helper():
    plant = FOPDT(gain=1.0, tau=1.0, theta=0.0)
    ctrl = PIDController(kp=1.0, ki=0.5)
    sp_func = step_setpoint([0.0, 10.0], [0.0, 1.0])
    assert sp_func(5.0) == 0.0
    assert sp_func(10.0) == 1.0

    res = simulate(plant, ctrl, setpoint=sp_func, dt=0.01, t_end=40.0)
    assert res.setpoint[0] == 0.0
    assert res.setpoint[-1] == 1.0
    assert abs(res.output[-1] - 1.0) < 1e-2


def test_per_sample_setpoint_array():
    plant = FOPDT(gain=1.0, tau=0.5, theta=0.0)
    ctrl = PIDController(kp=2.0)
    n = 2001
    sp = np.zeros(n)
    sp[1000:] = 1.0
    res = simulate(plant, ctrl, setpoint=sp, dt=0.01, t_end=20.0)
    assert res.n == n
    assert res.setpoint[500] == 0.0
    assert res.setpoint[-1] == 1.0


def test_disturbance_step():
    plant = FOPDT(gain=1.0, tau=1.0, theta=0.0)
    ctrl = PIDController(kp=1.0, ki=0.5)
    res = simulate(
        plant,
        ctrl,
        setpoint=1.0,
        dt=0.01,
        t_end=40.0,
        disturbance_time=20.0,
        disturbance_magnitude=0.5,
    )
    # The disturbance causes a visible bump around t=20.
    mid = res.output[(res.time > 20.0) & (res.time < 22.0)]
    assert mid.max() > 1.05
    # ... which is rejected back to setpoint.
    assert abs(res.output[-1] - 1.0) < 1e-2


def test_euler_vs_rk4_consistency():
    plant = FOPDT(gain=1.0, tau=2.0, theta=0.5)
    ctrl_e = PIDController(kp=1.0, ki=0.3)
    ctrl_r = PIDController(kp=1.0, ki=0.3)
    re = simulate(plant, ctrl_e, dt=0.001, t_end=20.0, method="euler")
    rr = simulate(plant, ctrl_r, dt=0.001, t_end=20.0, method="rk4")
    assert np.allclose(re.output, rr.output, atol=1e-3)


def test_integrating_process_high_p_gain_oscillates():
    """An integrating process with dead time becomes oscillatory for a P gain
    above the critical gain ``pi / (2 * K * theta)``."""
    plant = IntegratorDelay(gain=1.0, theta=1.0)
    ctrl = PIDController(kp=4.0)  # critical gain ~= 1.57
    res = simulate(plant, ctrl, setpoint=1.0, dt=0.01, t_end=60.0)
    # Growing oscillation: swings well below zero and overshoots heavily.
    assert res.output.min() < -0.5
    assert res.output.max() > 2.0

    # A PI controller stabilises the same plant (SIMC integrator tuning).
    from pidforge.tuning import simc_integrator

    params = simc_integrator(plant)
    ctrl2 = PIDController(kp=params.kp, ki=params.ki)
    res2 = simulate(plant, ctrl2, setpoint=1.0, dt=0.01, t_end=120.0)
    assert abs(res2.output[-1] - 1.0) < 1e-2
    assert res2.output.max() < 1.5


def test_invalid_args():
    plant = FOPDT(gain=1.0, tau=1.0, theta=0.0)
    ctrl = PIDController(kp=1.0)
    with pytest.raises(ValueError):
        simulate(plant, ctrl, dt=0.0)
    with pytest.raises(ValueError):
        simulate(plant, ctrl, t_end=-1.0)
    with pytest.raises(ValueError):
        simulate(plant, ctrl, method="gear")
