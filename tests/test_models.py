import numpy as np
import pytest

from pidforge.models import (
    FOPDT,
    SOPDT,
    IntegratorDelay,
    Oscillator,
    parse_model,
)


def test_fopdt_steady_state():
    plant = FOPDT(gain=2.0, tau=5.0, theta=1.0)
    assert plant.n_states == 1
    assert plant.dead_time == 1.0
    np.testing.assert_allclose(plant.steady_state(3.0), [6.0])
    # derivative at rest: dx/dt = (K*u - x)/tau
    np.testing.assert_allclose(plant.derivative(np.array([0.0]), 1.0), [2.0 / 5.0])


def test_fopdt_validation():
    with pytest.raises(ValueError):
        FOPDT(tau=0.0)
    with pytest.raises(ValueError):
        FOPDT(theta=-1.0)


def test_sopdt():
    plant = SOPDT(gain=2.0, tau1=3.0, tau2=4.0, theta=0.5)
    assert plant.n_states == 2
    np.testing.assert_allclose(plant.steady_state(1.0), [2.0, 2.0])
    # At steady state the derivative is zero.
    np.testing.assert_allclose(
        plant.derivative(plant.steady_state(1.0), 1.0), [0.0, 0.0]
    )


def test_oscillator():
    plant = Oscillator(gain=1.5, wn=2.0, zeta=0.3)
    np.testing.assert_allclose(plant.steady_state(1.0), [1.5, 0.0])
    np.testing.assert_allclose(
        plant.derivative(plant.steady_state(1.0), 1.0), [0.0, 0.0]
    )


def test_integrator_delay():
    plant = IntegratorDelay(gain=2.0, theta=0.5)
    np.testing.assert_allclose(plant.derivative(np.array([1.0]), 3.0), [6.0])
    with pytest.raises(NotImplementedError):
        plant.step_gain()


def test_parse_model():
    plant = parse_model("fopdt:2,5,1")
    assert isinstance(plant, FOPDT)
    assert plant.gain == 2.0 and plant.tau == 5.0 and plant.theta == 1.0

    assert isinstance(parse_model("sopdt:1,2,3,0.1"), SOPDT)
    assert isinstance(parse_model("oscillator:2,1,0.3"), Oscillator)
    assert isinstance(parse_model("integrator:1,0.5"), IntegratorDelay)
    assert isinstance(parse_model("fopdt"), FOPDT)

    with pytest.raises(ValueError):
        parse_model("bogus")
    with pytest.raises(ValueError):
        parse_model("fopdt:1,2,x")
