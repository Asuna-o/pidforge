"""Closed-loop simulation engine.

Integrates a :class:`~pidforge.models.Process` model (continuous-time
state-space, optionally with dead time) driven by a
:class:`~pidforge.controller.PIDController`. Dead time is modelled exactly
with an input buffer; the process ODE is integrated with a fixed-step RK4
solver by default.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from .controller import PIDController
from .models import Process

SetpointLike = Union[float, Callable[[float], float], np.ndarray, tuple]


@dataclass
class SimulationResult:
    """The outcome of a closed-loop simulation."""

    time: np.ndarray
    output: np.ndarray
    control: np.ndarray
    setpoint: np.ndarray

    @property
    def n(self) -> int:
        """Number of samples."""
        return int(self.time.size)

    @property
    def final_value(self) -> float:
        """Measured process output at the end of the horizon."""
        return float(self.output[-1])

    def as_dict(self) -> dict:
        """Plain dictionary, convenient for CSV export."""
        return {
            "time": self.time,
            "setpoint": self.setpoint,
            "output": self.output,
            "control": self.control,
        }


def _resolve_setpoint(sp: SetpointLike, t: float, i: int, n: int) -> float:
    """Evaluate the setpoint at sample ``i``/time ``t``.

    Accepts a constant, a callable ``t -> value``, a per-sample array of
    length ``n``, or a ``(times, values)`` pair describing a piecewise
    step trajectory.
    """
    if callable(sp):
        return float(sp(t))
    if isinstance(sp, (int, float, np.floating, np.integer)):
        return float(sp)
    arr = np.asarray(sp, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        if arr.shape[0] == n:
            return float(arr[i])
        if arr.shape[0] == 2:
            # Interpreted as (times, values) with an implied constant value.
            return float(arr[1])
        raise ValueError(
            f"setpoint array of length {arr.shape[0]} does not match "
            f"simulation length {n}"
        )
    if arr.ndim == 2 and arr.shape[0] == 2:
        times, values = arr[0], arr[1]
        idx = int(np.searchsorted(times, t, side="right")) - 1
        return float(values[max(idx, 0)])
    raise ValueError("unsupported setpoint specification")


def step_setpoint(times: list[float], values: list[float]) -> Callable[[float], float]:
    """Build a piecewise-constant (step) setpoint function.

    ``times`` and ``values`` must have the same length; the setpoint takes
    ``values[i]`` for ``t >= times[i]`` (and ``values[0]`` before that).
    """
    if len(times) != len(values):
        raise ValueError("times and values must have the same length")
    t_arr = np.asarray(times, dtype=float)
    v_arr = np.asarray(values, dtype=float)

    def _sp(t: float) -> float:
        idx = int(np.searchsorted(t_arr, t, side="right")) - 1
        return float(v_arr[max(idx, 0)])

    return _sp


def _rk4_step(
    process: Process, state: np.ndarray, u: float, t: float, dt: float
) -> np.ndarray:
    k1 = process.derivative(state, u, t)
    k2 = process.derivative(state + 0.5 * dt * k1, u, t + 0.5 * dt)
    k3 = process.derivative(state + 0.5 * dt * k2, u, t + 0.5 * dt)
    k4 = process.derivative(state + dt * k3, u, t + dt)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _euler_step(
    process: Process, state: np.ndarray, u: float, t: float, dt: float
) -> np.ndarray:
    return state + dt * process.derivative(state, u, t)


def simulate(
    process: Process,
    controller: PIDController,
    setpoint: SetpointLike = 1.0,
    dt: float = 0.01,
    t_end: float = 100.0,
    *,
    method: str = "rk4",
    disturbance: float | Callable[[float], float] | None = None,
    disturbance_time: float | None = None,
    disturbance_magnitude: float = 1.0,
    initial_control: float = 0.0,
) -> SimulationResult:
    """Simulate the closed loop over ``[0, t_end]`` with step ``dt``.

    Parameters
    ----------
    process:
        Process model to control.
    controller:
        PID controller (its ``update`` method is called every ``dt``).
    setpoint:
        Constant, callable ``t -> value``, per-sample array or
        ``(times, values)`` step trajectory.
    dt:
        Fixed sample/integration time step (also the controller period).
    t_end:
        Simulation horizon in seconds.
    method:
        ``"rk4"`` (default, accurate) or ``"euler"``.
    disturbance:
        Constant load disturbance or callable ``t -> value`` added to the
        measured output. If ``disturbance_time`` is given instead, a step of
        magnitude ``disturbance_magnitude`` is applied at that time.
    initial_control:
        Controller output held before the first sample (used as the process
        input during the dead-time period).

    Returns
    -------
    SimulationResult
        Time, setpoint, measured output and control arrays.
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if method not in ("rk4", "euler"):
        raise ValueError(f"unknown integrator {method!r}; use 'rk4' or 'euler'")
    if t_end <= 0:
        raise ValueError(f"t_end must be > 0, got {t_end}")

    n = int(round(t_end / dt)) + 1
    time = np.linspace(0.0, t_end, n)
    state = process.initial_state(u0=initial_control)

    # Exact dead time: the ODE input is the control value from theta seconds ago.
    delay_steps = max(1, int(round(process.dead_time / dt)))
    past_control: deque = deque([float(initial_control)] * delay_steps)

    step = _rk4_step if method == "rk4" else _euler_step

    out = np.empty(n)
    ctrl = np.empty(n)
    sp = np.empty(n)

    for i in range(n):
        t = time[i]
        u_applied = past_control[0]

        # Advance the process with the *delayed* input.
        state = step(process, state, u_applied, t, dt)
        y = float(process.output(state))

        # Load disturbance (additive on the measurement).
        if disturbance_time is not None and t >= disturbance_time:
            y += disturbance_magnitude
        elif callable(disturbance):
            y += float(disturbance(t))
        elif disturbance is not None:
            y += float(disturbance)

        sp_val = _resolve_setpoint(setpoint, t, i, n)
        u_ctrl = float(controller.update(y, dt, sp_val))

        past_control.append(u_ctrl)
        past_control.popleft()

        out[i] = y
        ctrl[i] = u_ctrl
        sp[i] = sp_val

    return SimulationResult(time=time, output=out, control=ctrl, setpoint=sp)


def open_loop_step(
    process: Process,
    u: float = 1.0,
    dt: float = 0.01,
    t_end: float = 60.0,
    *,
    method: str = "rk4",
) -> SimulationResult:
    """Simulate an *open-loop step test*: the input is held constant at ``u``.

    This is the standard plant experiment used for identification: bump the
    manipulated variable and record the response. ``control`` is constant
    ``u`` and ``setpoint`` mirrors ``u`` for convenience.
    """
    if dt <= 0 or t_end <= 0:
        raise ValueError("dt and t_end must be positive")
    n = int(round(t_end / dt)) + 1
    time = np.linspace(0.0, t_end, n)
    state = process.steady_state(0.0)
    delay_steps = max(1, int(round(process.dead_time / dt)))
    # The input is held at 0 before the step, so the step only reaches the
    # process after the dead time has elapsed.
    past_control: deque = deque([0.0] * delay_steps)
    step = _rk4_step if method == "rk4" else _euler_step

    out = np.empty(n)
    for i in range(n):
        u_applied = past_control[0]
        state = step(process, state, u_applied, time[i], dt)
        past_control.append(float(u))
        past_control.popleft()
        out[i] = float(process.output(state))

    return SimulationResult(
        time=time, output=out, control=np.full(n, float(u)), setpoint=np.full(n, float(u))
    )


def tuned_simulation(
    process: Process,
    tuning,
    *,
    dt: float = 0.01,
    t_end: float = 100.0,
    setpoint: SetpointLike = 1.0,
    output_limits=None,
    **kwargs,
) -> SimulationResult:
    """Build a controller from a tuning result and simulate immediately.

    ``tuning`` may be a :class:`~pidforge.tuning.TuningResult` (or any
    object with ``kp``/``ki``/``kd`` attributes) or a method name accepted
    by :func:`pidforge.auto_tune`.
    """
    from .tuning import auto_tune  # local import avoids a cycle

    if isinstance(tuning, str):
        params = auto_tune(process, method=tuning)
    else:
        params = tuning
    controller = PIDController(
        kp=params.kp, ki=params.ki, kd=params.kd, output_limits=output_limits
    )
    return simulate(process, controller, setpoint=setpoint, dt=dt, t_end=t_end, **kwargs)


__all__: list[str] = [
    "SimulationResult",
    "simulate",
    "open_loop_step",
    "step_setpoint",
    "tuned_simulation",
]
