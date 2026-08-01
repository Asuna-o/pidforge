import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import matplotlib.pyplot as plt  # noqa: E402

from pidforge import FOPDT, PIDController, compute_metrics, simulate  # noqa: E402
from pidforge.plotting import plot_comparison, plot_response  # noqa: E402
from pidforge.tuning import simc_pi, ziegler_nichols_open_loop  # noqa: E402


def _result():
    plant = FOPDT(gain=2.0, tau=5.0, theta=1.0)
    params = simc_pi(plant)
    ctrl = PIDController(kp=params.kp, ki=params.ki)
    return simulate(plant, ctrl, setpoint=1.0, dt=0.02, t_end=60.0)


def test_plot_response_returns_figure():
    fig = plot_response(_result(), metrics=compute_metrics(_result()))
    assert fig is not None
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_response_without_metrics():
    fig = plot_response(_result())
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_comparison():
    plant = FOPDT(gain=2.0, tau=5.0, theta=1.0)
    results = {}
    for name, params in (
        ("SIMC", simc_pi(plant)),
        ("ZN", ziegler_nichols_open_loop(plant, "PID")),
    ):
        ctrl = PIDController(kp=params.kp, ki=params.ki, kd=params.kd)
        results[name] = simulate(plant, ctrl, setpoint=1.0, dt=0.02, t_end=60.0)
    fig = plot_comparison(results)
    assert len(fig.axes) == 2
    plt.close(fig)
