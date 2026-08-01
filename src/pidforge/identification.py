"""Process identification: fit a FOPDT model from step-response data.

In industrial practice the first step of any tuning exercise is a *step
test*: bump the manipulated variable and record the process response. This
module converts that data into a :class:`~pidforge.models.FOPDT` model
using textbook reaction-curve methods:

* **two-point** (28.3 % / 63.2 % rule);
* **area (moments)** method, which uses the first two moments of the
  normalised response and is more robust to noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FOPDT
from .tuning import TuningResult


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration (version-independent numpy wrapper)."""
    if x.size < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


@dataclass
class FOPDTFit:
    """A fitted FOPDT model plus the identification metadata."""

    gain: float
    tau: float
    theta: float
    method: str

    @property
    def model(self) -> FOPDT:
        """The fitted model, ready for tuning."""
        return FOPDT(gain=self.gain, tau=self.tau, theta=self.theta)

    def summary(self) -> str:
        return (
            f"FOPDT ({self.method}): K={self.gain:.4g}, "
            f"tau={self.tau:.4g}, theta={self.theta:.4g}"
        )


def _normalise(
    time: np.ndarray, output: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Return the normalised response and the baseline/final values.

    The baseline and final value are estimated from the first and last 2 % of
    samples (not single samples) so that the fit is robust to measurement
    noise.
    """
    n = int(output.size)
    k = max(1, int(round(0.02 * n)))
    y0 = float(np.mean(output[:k]))
    y_end = float(np.mean(output[-k:]))
    span = y_end - y0
    if abs(span) < 1e-12:
        raise ValueError(
            "step response shows no change; cannot identify a process model"
        )
    return (output - y0) / span, y0, span


def _crossing_time(time: np.ndarray, y_norm: np.ndarray, level: float) -> float:
    """First time the normalised response reaches ``level``."""
    idx = int(np.flatnonzero(y_norm >= level)[0])
    if idx == 0:
        return float(time[0])
    # Linear interpolation for sub-sample accuracy.
    y0, y1 = y_norm[idx - 1], y_norm[idx]
    t0, t1 = time[idx - 1], time[idx]
    if y1 == y0:
        return float(t1)
    frac = (level - y0) / (y1 - y0)
    return float(t0 + frac * (t1 - t0))


def fit_two_point(time: np.ndarray, output: np.ndarray) -> FOPDTFit:
    """Fit a FOPDT model with the 28.3 % / 63.2 % reaction-curve rule."""
    y_norm, y0, span = _normalise(time, output)
    t28 = _crossing_time(time, y_norm, 0.283)
    t63 = _crossing_time(time, y_norm, 0.632)
    tau = 1.5 * (t63 - t28)
    theta = t63 - tau
    if tau <= 0:
        raise ValueError("identification failed: non-positive time constant")
    return FOPDTFit(gain=span, tau=tau, theta=max(theta, 0.0), method="two-point")


def fit_area(time: np.ndarray, output: np.ndarray) -> FOPDTFit:
    """Fit a FOPDT model with the area (moments) method.

    Uses ``A1 = int(1 - y_norm)dt = theta + tau`` and
    ``A2 = int(t(1 - y_norm))dt = theta^2/2 + theta*tau + tau^2``.
    """
    y_norm, y0, span = _normalise(time, output)
    a1 = _trapz(1.0 - y_norm, time)
    a2 = _trapz(time * (1.0 - y_norm), time)
    disc = 2.0 * a2 - a1 * a1
    if disc < 0.0:
        # With noisy data the moment estimates can drift slightly negative;
        # clamp to the degenerate case (theta -> 0) rather than failing.
        disc = 0.0
    theta = a1 - np.sqrt(disc)
    tau = a1 - theta
    if tau <= 0:
        raise ValueError("identification failed: non-positive time constant")
    return FOPDTFit(gain=span, tau=float(tau), theta=float(max(theta, 0.0)), method="area")


def fit_fopdt(
    time,
    output,
    input_step: float | np.ndarray | None = None,
    method: str = "two_point",
) -> FOPDTFit:
    """Fit a FOPDT model from a step-response experiment.

    Parameters
    ----------
    time:
        Sample times (seconds).
    output:
        Measured process output.
    input_step:
        The magnitude of the input step (a scalar or an array of the
        manipulated variable). When omitted it is assumed to be ``1.0`` and
        the reported gain is the normalised gain.
    method:
        ``"two_point"`` or ``"area"``.
    """
    time_arr = np.asarray(time, dtype=float)
    out_arr = np.asarray(output, dtype=float)
    if time_arr.ndim != 1 or out_arr.ndim != 1:
        raise ValueError("time and output must be 1-D")
    if time_arr.size != out_arr.size or time_arr.size < 3:
        raise ValueError("time and output must have the same length (>= 3)")
    if np.any(np.diff(time_arr) <= 0):
        raise ValueError("time must be strictly increasing")

    if method not in ("two_point", "area"):
        raise ValueError("method must be 'two_point' or 'area'")

    fit = fit_two_point(time_arr, out_arr) if method == "two_point" else fit_area(time_arr, out_arr)

    if input_step is not None:
        if isinstance(input_step, (int, float, np.floating, np.integer)):
            step = float(input_step)
        else:
            step_arr = np.asarray(input_step, dtype=float)
            step = float(step_arr[-1] - step_arr[0]) if step_arr.size > 1 else float(step_arr[0])
        if abs(step) < 1e-12:
            raise ValueError("input step must be non-zero")
        fit = FOPDTFit(
            gain=fit.gain / step, tau=fit.tau, theta=fit.theta, method=fit.method
        )
    return fit


def tune_from_step_data(
    time,
    output,
    input_step: float | np.ndarray | None = None,
    fit_method: str = "two_point",
    tuning: str = "simc",
    tau_c: float | None = None,
) -> tuple[FOPDT, TuningResult]:
    """End-to-end workflow: step data -> FOPDT model -> PID tuning.

    This mirrors the real engineering workflow: run a step test, identify
    the model, and obtain controller settings in one call.
    """
    from .tuning import auto_tune

    fit = fit_fopdt(time, output, input_step=input_step, method=fit_method)
    params = auto_tune(fit.model, method=tuning, tau_c=tau_c)
    return fit.model, params


__all__: list[str] = [
    "FOPDTFit",
    "fit_fopdt",
    "fit_two_point",
    "fit_area",
    "tune_from_step_data",
]
