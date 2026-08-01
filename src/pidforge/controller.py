"""Discrete-time PID controller with industrial-grade features.

Implements the *position form* (absolute output) PID commonly used in DCS
and PLC implementations:

.. math::

    u(t) = K_p \\left[ b\\, r(t) - y(t) \\right]
         + K_i \\int_0^t e(\\tau)\\,d\\tau
         + K_d \\frac{d}{dt} \\left[ c\\, r(t) - y(t) \\right]

Features:

* derivative action on a filtered (low-pass) derivative of the measurement,
  which removes the "setpoint kick" and rejects high-frequency noise;
* setpoint weighting (``b`` for the proportional term, ``c`` for the
  derivative term);
* anti-windup via *conditional integration* (the integrator is recomputed so
  that the raw output never exceeds the actuator limits);
* direct/reverse acting control (for fail-safe actuators).
"""

from __future__ import annotations

DEFAULT_DERIVATIVE_FILTER = 10.0
"""Filter coefficient N in ``D_filter = Kd * N / (s + N)`` (higher = less filtered)."""


class PIDController:
    """Position-form PID controller with anti-windup and derivative filtering.

    Parameters
    ----------
    kp, ki, kd:
        Proportional, integral and derivative gains.
    setpoint:
        Initial setpoint (can be overridden per call to :meth:`update`).
    output_limits:
        ``(min, max)`` actuator limits; ``None`` means unbounded.
    derivative_filter:
        Filter coefficient ``N``; set to ``0`` or ``None`` to disable the
        derivative term entirely.
    setpoint_weight:
        Tuple ``(b, c)`` weighting the setpoint in the proportional and
        derivative terms.
    direction:
        ``"reverse"`` (default, direct-acting process: error = SP - PV) or
        ``"direct"`` (error = PV - SP) for fail-open actuators.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        *,
        setpoint: float = 0.0,
        output_limits: tuple[float | None, float | None] | None = None,
        derivative_filter: float | None = DEFAULT_DERIVATIVE_FILTER,
        setpoint_weight: tuple[float, float] = (1.0, 0.0),
        direction: str = "reverse",
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = float(setpoint)
        self.output_limits = (
            (None, None) if output_limits is None else tuple(output_limits)  # type: ignore[assignment]
        )
        self.derivative_filter = (
            None if derivative_filter is None else float(derivative_filter)
        )
        self.setpoint_weight = (float(setpoint_weight[0]), float(setpoint_weight[1]))
        if direction not in ("direct", "reverse"):
            raise ValueError("direction must be 'direct' or 'reverse'")
        self.direction = direction

        self._integral = 0.0
        self._filtered_derivative = 0.0
        self._last_derivative_source: float | None = None
        self._last_output = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        measurement: float,
        dt: float | None = None,
        setpoint: float | None = None,
    ) -> float:
        """Advance the controller by one sample and return the control output.

        ``dt`` may be omitted on the very first call; the controller then
        treats it as an infinitesimal step (proportional term only).
        """
        if setpoint is not None:
            self.setpoint = float(setpoint)

        y = float(measurement)
        sign = 1.0 if self.direction == "reverse" else -1.0

        if dt is None or dt <= 0.0:
            # First sample: only the proportional term is defined.
            p = self.kp * sign * (self.setpoint_weight[0] * self.setpoint - y)
            u = self._clamp(p)
            self._last_derivative_source = (
                self.setpoint_weight[1] * self.setpoint - y
            )
            return u

        e = sign * (self.setpoint - y)

        # Proportional term with setpoint weighting.
        p = self.kp * sign * (self.setpoint_weight[0] * self.setpoint - y)

        # Integral term with conditional-integration anti-windup.
        self._integral += self.ki * e * dt
        u_unclamped = p + self._integral + self.kd * self._filtered_derivative
        u = self._clamp(u_unclamped)
        if u != u_unclamped:
            # Keep the integrator consistent with the saturated output so the
            # loop recovers quickly when the actuator un-saturates.
            self._integral = u - p - self.kd * self._filtered_derivative

        # Filtered derivative on (c * setpoint - measurement).
        source = self.setpoint_weight[1] * self.setpoint - y
        if self.derivative_filter:
            if self._last_derivative_source is not None:
                raw = (source - self._last_derivative_source) / dt
                n = self.derivative_filter
                self._filtered_derivative = (n * raw + self._filtered_derivative / dt) / (
                    n + 1.0 / dt
                )
            self._last_derivative_source = source

        self._last_output = float(u)
        return float(u)

    def reset(self) -> None:
        """Reset all internal states (integral, derivative filter)."""
        self._integral = 0.0
        self._filtered_derivative = 0.0
        self._last_derivative_source = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _clamp(self, u: float) -> float:
        lo, hi = self.output_limits
        if lo is not None and u < lo:
            return float(lo)
        if hi is not None and u > hi:
            return float(hi)
        return u

    @property
    def output(self) -> float:
        """The most recently computed control output (0.0 before first call)."""
        return self._last_output

    def __repr__(self) -> str:
        return (
            f"PIDController(kp={self.kp:g}, ki={self.ki:g}, kd={self.kd:g}, "
            f"setpoint={self.setpoint:g}, limits={self.output_limits})"
        )


def pid_from_gains(
    kp: float,
    ki: float = 0.0,
    kd: float = 0.0,
    **kwargs: object,
) -> PIDController:
    """Create a :class:`PIDController` from gain values."""
    return PIDController(kp=kp, ki=ki, kd=kd, **kwargs)


__all__: list[str] = ["PIDController", "pid_from_gains", "DEFAULT_DERIVATIVE_FILTER"]
