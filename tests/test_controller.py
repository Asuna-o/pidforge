import numpy as np

from pidforge import FOPDT, PIDController, simulate


def test_p_control_reaches_predicted_steady_state():
    plant = FOPDT(gain=1.0, tau=5.0, theta=0.0)
    kp = 2.0
    ctrl = PIDController(kp=kp)
    res = simulate(plant, ctrl, setpoint=1.0, dt=0.02, t_end=100.0)
    expected = kp / (1.0 + kp)  # P-only offset
    assert abs(res.output[-1] - expected) < 1e-3


def test_pi_removes_offset():
    plant = FOPDT(gain=1.0, tau=1.0, theta=0.0)
    ctrl = PIDController(kp=1.0, ki=0.5)
    res = simulate(plant, ctrl, setpoint=1.0, dt=0.01, t_end=50.0)
    assert abs(res.output[-1] - 1.0) < 1e-3


def test_output_limits_clamp_control():
    plant = FOPDT(gain=1.0, tau=1.0, theta=0.0)
    ctrl = PIDController(kp=5.0, ki=1.0, output_limits=(0.0, 1.0))
    res = simulate(plant, ctrl, setpoint=5.0, dt=0.01, t_end=10.0)
    assert res.control.max() <= 1.0 + 1e-9
    assert res.control.min() >= -1e-9
    # Once saturated, the loop must still recover (no integrator windup).
    assert res.output[-1] > 0.9


def test_direct_acting_controller():
    FOPDT(gain=1.0, tau=1.0, theta=0.0)
    ctrl = PIDController(kp=1.0, direction="direct")
    # PV above SP => positive output for a direct-acting loop (cooling case).
    u = ctrl.update(2.0, dt=0.1, setpoint=1.0)
    assert u > 0.0
    ctrl2 = PIDController(kp=1.0, direction="reverse")
    assert ctrl2.update(2.0, dt=0.1, setpoint=1.0) < 0.0


def test_derivative_filter_no_nan_on_step():
    plant = FOPDT(gain=1.0, tau=2.0, theta=0.0)
    ctrl = PIDController(kp=1.0, ki=0.2, kd=1.0, derivative_filter=10.0)
    res = simulate(plant, ctrl, setpoint=1.0, dt=0.01, t_end=20.0)
    assert np.isfinite(res.control).all()
    assert np.isfinite(res.output).all()


def test_reset():
    ctrl = PIDController(kp=1.0, ki=0.5)
    ctrl.update(0.0, dt=0.1, setpoint=1.0)
    ctrl.update(0.5, dt=0.1, setpoint=1.0)
    ctrl.reset()
    u = ctrl.update(0.0, dt=0.1, setpoint=1.0)
    # After reset the controller behaves like a fresh controller: the first
    # step contains the proportional term plus one integral increment.
    assert abs(u - (1.0 + 0.5 * 0.1)) < 1e-9
