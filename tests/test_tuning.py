import pytest

from pidforge import FOPDT, IntegratorDelay, PIDController, simulate
from pidforge.tuning import (
    PIDParams,
    cohen_coon,
    imc_pi,
    imc_pid,
    simc_integrator,
    simc_pi,
    tyreus_luyben,
    ziegler_nichols_closed_loop,
    ziegler_nichols_open_loop,
)

PLANT = FOPDT(gain=2.0, tau=5.0, theta=1.0)


def test_zn_open_loop():
    r = ziegler_nichols_open_loop(PLANT, "PID")
    assert r.kp == pytest.approx(3.0)
    assert r.ki == pytest.approx(1.5)
    assert r.kd == pytest.approx(1.5)
    assert r.ti == pytest.approx(2.0)
    assert r.td == pytest.approx(0.5)

    r_pi = ziegler_nichols_open_loop(PLANT, "PI")
    assert r_pi.kp == pytest.approx(0.9 * 5.0 / 2.0)
    assert r_pi.ti == pytest.approx(3.33)

    r_p = ziegler_nichols_open_loop(PLANT, "P")
    assert r_p.kp == pytest.approx(5.0 / 2.0)


def test_zn_closed_loop():
    r = ziegler_nichols_closed_loop(ku=4.0, tu=6.0, controller="PID")
    assert r.kp == pytest.approx(2.4)
    assert r.ti == pytest.approx(3.0)
    assert r.td == pytest.approx(0.75)

    r_pi = ziegler_nichols_closed_loop(ku=4.0, tu=6.0, controller="PI")
    assert r_pi.kp == pytest.approx(1.8)
    assert r_pi.ti == pytest.approx(5.0)


def test_cohen_coon():
    r = cohen_coon(PLANT, "PID")
    assert r.kp == pytest.approx(2.5 * (4.0 / 3.0 + 0.05))
    assert r.ti == pytest.approx(33.2 / 14.6)
    assert r.td == pytest.approx(4.0 / 11.4)


def test_imc():
    r = imc_pi(PLANT)  # tau_c defaults to theta
    assert r.kp == pytest.approx(5.0 / (2.0 * 2.0))
    assert r.ti == pytest.approx(5.0)

    r_pid = imc_pid(PLANT, tau_c=1.0)
    assert r_pid.kp == pytest.approx((5.0 + 0.5) / (2.0 * 2.0))
    assert r_pid.ti == pytest.approx(5.5)


def test_simc():
    r = simc_pi(PLANT)  # tau_c defaults to theta = 1
    assert r.kp == pytest.approx(5.0 / (2.0 * 2.0))
    assert r.ti == pytest.approx(min(5.0, 8.0))

    r2 = simc_pi(PLANT, tau_c=2.0)
    assert r2.kp == pytest.approx(5.0 / (2.0 * 3.0))
    assert r2.ti == pytest.approx(min(5.0, 12.0))


def test_simc_integrator():
    plant = IntegratorDelay(gain=1.0, theta=1.0)
    r = simc_integrator(plant)
    assert r.kp == pytest.approx(0.5)
    assert r.ti == pytest.approx(8.0)


def test_tyreus_luyben():
    r = tyreus_luyben(ku=4.0, tu=6.0, controller="PI")
    assert r.kp == pytest.approx(1.25)
    assert r.ti == pytest.approx(13.2)


def test_pid_params_properties():
    p = PIDParams(kp=2.0, ki=1.0, kd=0.5, controller="PID")
    assert p.ti == 2.0
    assert p.td == 0.25
    assert p.with_controller("PI").controller == "PI"


def test_auto_tune_dispatch():
    from pidforge.tuning import auto_tune

    assert auto_tune(PLANT, method="simc").method == "SIMC"
    assert auto_tune(PLANT, method="imc").method == "IMC (Lambda)"
    assert auto_tune(PLANT, method="zn-open", controller="PI").controller == "PI"
    with pytest.raises(ValueError):
        auto_tune(PLANT, method="zn-closed")
    with pytest.raises(ValueError):
        auto_tune(PLANT, method="nope")


def test_tuning_produces_stable_loop():
    """A SIMC-tuned PI loop must settle close to the setpoint."""
    from pidforge.tuning import simc_pi

    params = simc_pi(PLANT)
    ctrl = PIDController(kp=params.kp, ki=params.ki)
    res = simulate(PLANT, ctrl, setpoint=1.0, dt=0.01, t_end=80.0)
    assert abs(res.output[-1] - 1.0) < 5e-2
    assert res.output.max() < 1.2
