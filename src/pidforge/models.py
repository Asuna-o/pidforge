"""Continuous-time process models for control-loop simulation.

All models are linear, single-input/single-output (SISO) transfer functions
expressed in continuous-time state-space form, optionally with pure dead
time (transport delay). The simulator in :mod:`pidforge.simulate` integrates
these models with a fixed time step and models the dead time with an input
buffer, so the delay is exact regardless of the integration step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Process(ABC):
    """Base class for a continuous-time SISO process model."""

    name: str = "generic"
    dead_time: float = 0.0

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of continuous states of the model."""

    @abstractmethod
    def derivative(self, state: np.ndarray, u: float, t: float = 0.0) -> np.ndarray:
        """Return ``dx/dt`` for the given state, input and time."""

    @abstractmethod
    def output(self, state: np.ndarray) -> float:
        """Return the measured output ``y`` for the given state."""

    @abstractmethod
    def steady_state(self, u: float) -> np.ndarray:
        """Return the state vector at steady state for constant input ``u``."""

    def initial_state(self, u0: float = 0.0) -> np.ndarray:
        """Return a suitable initial state (steady state at ``u0``)."""
        return self.steady_state(u0)

    def step_gain(self) -> float:
        """Steady-state gain of the model (delta-y per delta-u)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dead_time={self.dead_time:g})"


@dataclass
class FOPDT(Process):
    """First-order plus dead-time (FOPDT) model.

    .. math::

        G(s) = \\frac{K}{\\tau s + 1} \\, e^{-\\theta s}

    The most widely used model in process automation for tuning PID
    controllers; parameters are commonly obtained from a step test.
    """

    gain: float = 1.0
    tau: float = 1.0
    theta: float = 0.0
    name: str = "FOPDT"

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.theta < 0:
            raise ValueError(f"theta must be >= 0, got {self.theta}")
        self.dead_time = float(self.theta)

    @property
    def n_states(self) -> int:
        return 1

    def derivative(self, state: np.ndarray, u: float, t: float = 0.0) -> np.ndarray:
        x = float(state[0])
        return np.array([(self.gain * u - x) / self.tau])

    def output(self, state: np.ndarray) -> float:
        return float(state[0])

    def steady_state(self, u: float) -> np.ndarray:
        return np.array([self.gain * u])

    def step_gain(self) -> float:
        return float(self.gain)

    def __repr__(self) -> str:
        return (
            f"FOPDT(gain={self.gain:g}, tau={self.tau:g}, theta={self.theta:g})"
        )


@dataclass
class SOPDT(Process):
    """Second-order plus dead-time (SOPDT) model with two lags in series.

    .. math::

        G(s) = \\frac{K}{(\\tau_1 s + 1)(\\tau_2 s + 1)} \\, e^{-\\theta s}

    Represents processes with two dominant time constants, e.g. cascaded
    tanks or heat exchangers with both wall and fluid dynamics.
    """

    gain: float = 1.0
    tau1: float = 1.0
    tau2: float = 1.0
    theta: float = 0.0
    name: str = "SOPDT"

    def __post_init__(self) -> None:
        if self.tau1 <= 0 or self.tau2 <= 0:
            raise ValueError("tau1 and tau2 must be > 0")
        if self.theta < 0:
            raise ValueError(f"theta must be >= 0, got {self.theta}")
        self.dead_time = float(self.theta)

    @property
    def n_states(self) -> int:
        return 2

    def derivative(self, state: np.ndarray, u: float, t: float = 0.0) -> np.ndarray:
        x1, x2 = float(state[0]), float(state[1])
        dx1 = (self.gain * u - x1) / self.tau1
        dx2 = (x1 - x2) / self.tau2
        return np.array([dx1, dx2])

    def output(self, state: np.ndarray) -> float:
        return float(state[1])

    def steady_state(self, u: float) -> np.ndarray:
        return np.array([self.gain * u, self.gain * u])

    def step_gain(self) -> float:
        return float(self.gain)

    def __repr__(self) -> str:
        return (
            f"SOPDT(gain={self.gain:g}, tau1={self.tau1:g}, "
            f"tau2={self.tau2:g}, theta={self.theta:g})"
        )


@dataclass
class Oscillator(Process):
    """Underdamped second-order oscillator plus dead time.

    .. math::

        G(s) = \\frac{K \\omega_n^2}{s^2 + 2 \\zeta \\omega_n s + \\omega_n^2}
        \\, e^{-\\theta s}

    Useful for servo systems, gimbal/turntable dynamics and other lightly
    damped mechanical plants that appear in automation and robotics.
    """

    gain: float = 1.0
    wn: float = 1.0
    zeta: float = 0.3
    theta: float = 0.0
    name: str = "Oscillator"

    def __post_init__(self) -> None:
        if self.wn <= 0:
            raise ValueError(f"wn must be > 0, got {self.wn}")
        if not 0.0 < self.zeta < 1.0:
            raise ValueError("zeta must be in (0, 1) for an underdamped model")
        if self.theta < 0:
            raise ValueError(f"theta must be >= 0, got {self.theta}")
        self.dead_time = float(self.theta)

    @property
    def n_states(self) -> int:
        return 2

    def derivative(self, state: np.ndarray, u: float, t: float = 0.0) -> np.ndarray:
        x1, x2 = float(state[0]), float(state[1])
        dx1 = x2
        dx2 = self.wn**2 * (self.gain * u - x1) - 2.0 * self.zeta * self.wn * x2
        return np.array([dx1, dx2])

    def output(self, state: np.ndarray) -> float:
        return float(state[0])

    def steady_state(self, u: float) -> np.ndarray:
        return np.array([self.gain * u, 0.0])

    def step_gain(self) -> float:
        return float(self.gain)

    def __repr__(self) -> str:
        return (
            f"Oscillator(gain={self.gain:g}, wn={self.wn:g}, "
            f"zeta={self.zeta:g}, theta={self.theta:g})"
        )


@dataclass
class IntegratorDelay(Process):
    """Integrating process with dead time (level/position processes).

    .. math::

        G(s) = \\frac{K}{s} \\, e^{-\\theta s}

    Typical of tank level loops and motor/position drives: there is no
    self-regulation, so a controller with integral action (or the correct
    P-only gain) is mandatory.
    """

    gain: float = 1.0
    theta: float = 0.0
    name: str = "IntegratorDelay"

    def __post_init__(self) -> None:
        if self.theta < 0:
            raise ValueError(f"theta must be >= 0, got {self.theta}")
        self.dead_time = float(self.theta)

    @property
    def n_states(self) -> int:
        return 1

    def derivative(self, state: np.ndarray, u: float, t: float = 0.0) -> np.ndarray:
        return np.array([self.gain * u])

    def output(self, state: np.ndarray) -> float:
        return float(state[0])

    def steady_state(self, u: float) -> np.ndarray:
        return np.zeros(1)

    def step_gain(self) -> float:
        raise NotImplementedError(
            "An integrator has no finite steady-state gain; it ramps "
            "unboundedly for constant input."
        )

    def __repr__(self) -> str:
        return f"IntegratorDelay(gain={self.gain:g}, theta={self.theta:g})"


_PROCESS_ALIASES: dict[str, type] = {
    "fopdt": FOPDT,
    "firstorder": FOPDT,
    "sopdt": SOPDT,
    "secondorder": SOPDT,
    "oscillator": Oscillator,
    "osc": Oscillator,
    "integrator": IntegratorDelay,
    "intdelay": IntegratorDelay,
}


def parse_model(spec: str) -> Process:
    """Build a process model from a compact string specification.

    Supported forms: ``fopdt``, ``fopdt:2,5,1``, ``sopdt:1,4,2,0.5``,
    ``oscillator:2,1,0.3``, ``integrator:1,1``.

    Parameter order follows the constructor of the corresponding model
    class (omitted trailing parameters use the class defaults).
    """
    if not spec:
        raise ValueError("empty model specification")
    name, _, params = spec.partition(":")
    key = name.strip().lower()
    if key not in _PROCESS_ALIASES:
        raise ValueError(
            f"unknown model '{name}'; available: "
            f"{', '.join(sorted(_PROCESS_ALIASES))}"
        )
    cls = _PROCESS_ALIASES[key]
    args: list[float] = []
    if params.strip():
        for token in params.split(","):
            token = token.strip()
            if not token:
                raise ValueError(f"invalid parameter in '{spec}'")
            args.append(float(token))
    return cls(*args)


def fopdt_from_params(gain: float, tau: float, theta: float) -> FOPDT:
    """Convenience factory for a :class:`FOPDT` model."""
    return FOPDT(gain=gain, tau=tau, theta=theta)


__all__: list[str] = [
    "Process",
    "FOPDT",
    "SOPDT",
    "Oscillator",
    "IntegratorDelay",
    "parse_model",
    "fopdt_from_params",
]
