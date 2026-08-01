import numpy as np
import pytest

from pidforge import FOPDT, fit_fopdt, open_loop_step, tune_from_step_data
from pidforge.identification import fit_area, fit_two_point


def _open_loop_step_response(plant, dt=0.01, t_end=60.0, u_step=1.0):
    """Simulate an open-loop step test with a constant input step."""
    res = open_loop_step(plant, u=u_step, dt=dt, t_end=t_end)
    return res.time, res.output


TRUE = FOPDT(gain=2.0, tau=5.0, theta=1.0)


def test_two_point_recovers_fopdt():
    time, out = _open_loop_step_response(TRUE)
    fit = fit_two_point(time, out)
    assert fit.gain == pytest.approx(2.0, rel=0.01)
    assert fit.tau == pytest.approx(5.0, rel=0.05)
    assert fit.theta == pytest.approx(1.0, abs=0.1)


def test_area_recovers_fopdt():
    time, out = _open_loop_step_response(TRUE)
    fit = fit_area(time, out)
    assert fit.gain == pytest.approx(2.0, rel=0.01)
    assert fit.tau == pytest.approx(5.0, rel=0.03)
    assert fit.theta == pytest.approx(1.0, abs=0.05)


def test_fit_fopdt_with_input_step_scaling():
    time, out = _open_loop_step_response(TRUE, u_step=0.5)
    fit = fit_fopdt(time, out, input_step=0.5, method="area")
    assert fit.gain == pytest.approx(2.0, rel=0.01)
    assert fit.model.gain == pytest.approx(2.0, rel=0.01)
    assert isinstance(fit.model, FOPDT)


def test_tune_from_step_data_pipeline():
    time, out = _open_loop_step_response(TRUE)
    model, params = tune_from_step_data(
        time, out, input_step=1.0, fit_method="area", tuning="simc"
    )
    assert isinstance(model, FOPDT)
    # SIMC with default tau_c == theta: kp = tau / (K * 2 * theta)
    expected = model.tau / (model.gain * 2.0 * model.theta)
    assert params.kp == pytest.approx(expected, rel=1e-6)


def test_no_change_raises():
    time = np.linspace(0, 10, 100)
    out = np.zeros_like(time)
    with pytest.raises(ValueError):
        fit_fopdt(time, out)


def test_bad_method_raises():
    time, out = _open_loop_step_response(TRUE)
    with pytest.raises(ValueError):
        fit_fopdt(time, out, method="magic")


def test_noisy_data_area_still_reasonable():
    rng = np.random.default_rng(7)
    time, out = _open_loop_step_response(TRUE)
    noise = rng.normal(0.0, 0.01, out.shape)
    fit = fit_area(time, out + noise)
    assert fit.gain == pytest.approx(2.0, rel=0.05)
    assert fit.tau == pytest.approx(5.0, rel=0.2)
    assert fit.theta == pytest.approx(1.0, abs=0.3)
