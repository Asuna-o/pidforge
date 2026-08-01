"""Controller tuning rules (PID parameter synthesis).

Implements the classic, textbook tuning rules used in industrial process
automation:

* **Ziegler–Nichols** open-loop (step response) and closed-loop (ultimate
  gain / ultimate period) rules;
* **Cohen–Coon** reaction-curve rules;
* **IMC / Lambda** tuning (Rivera–Morari–Skogestad);
* **SIMC** (Skogestad's simple internal model control);
* **Tyreus–Luyben** conservative closed-loop rules.

Every rule returns a :class:`PIDParams` object holding ``kp``, ``ki``,
``kd`` plus the equivalent integral/reset time ``ti`` and derivative rate
``td`` (``ti = kp / ki``, ``td = kd / kp``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import FOPDT, IntegratorDelay, Process


@dataclass(frozen=True)
class PIDParams:
    """PID gains in *gain form* (kp, ki, kd) plus the controller structure."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    controller: str = "PID"  # one of "P", "PI", "PID"

    def __post_init__(self) -> None:
        if self.controller not in ("P", "PI", "PID"):
            raise ValueError(f"unknown controller type {self.controller!r}")

    @property
    def ti(self) -> float:
        """Integral (reset) time; ``inf`` when no integral action."""
        return float("inf") if self.ki == 0.0 else self.kp / self.ki

    @property
    def td(self) -> float:
        """Derivative (rate) time; 0 when no derivative action."""
        return 0.0 if self.kp == 0.0 else self.kd / self.kp

    def with_controller(self, controller: str) -> PIDParams:
        """Return a copy with the given controller structure (P/PI/PID)."""
        return PIDParams(self.kp, self.ki, self.kd, controller)


@dataclass(frozen=True)
class TuningResult(PIDParams):
    """A :class:`PIDParams` together with provenance metadata."""

    method: str = "custom"
    description: str = ""

    def summary(self) -> str:
        """One-line human-readable description."""
        td = f", td={self.td:g}" if self.kd else ""
        return (
            f"{self.controller} ({self.method}): kp={self.kp:g}, "
            f"ti={self.ti:g}{td}"
        )


# ---------------------------------------------------------------------------
# Ziegler-Nichols
# ---------------------------------------------------------------------------
def ziegler_nichols_open_loop(
    fopdt: FOPDT, controller: str = "PID"
) -> TuningResult:
    """Ziegler–Nichols open-loop (step-response) tuning for a FOPDT model.

    Uses the classical formulas with the process gain ``K``, time constant
    ``tau`` and dead time ``theta``.
    """
    K, tau, theta = fopdt.gain, fopdt.tau, fopdt.theta
    if theta <= 0:
        raise ValueError("Ziegler-Nichols requires a model with dead time > 0")
    if controller == "P":
        return TuningResult(kp=tau / (K * theta), controller="P", method="Z-N open loop")
    if controller == "PI":
        return TuningResult(
            kp=0.9 * tau / (K * theta),
            ki=(0.9 * tau / (K * theta)) / (3.33 * theta),
            controller="PI",
            method="Z-N open loop",
        )
    return TuningResult(
        kp=1.2 * tau / (K * theta),
        ki=(1.2 * tau / (K * theta)) / (2.0 * theta),
        kd=(1.2 * tau / (K * theta)) * (0.5 * theta),
        controller="PID",
        method="Z-N open loop",
    )


def ziegler_nichols_closed_loop(
    ku: float, tu: float, controller: str = "PID"
) -> TuningResult:
    """Ziegler–Nichols closed-loop (ultimate gain/period) tuning.

    ``ku`` is the ultimate gain (gain at sustained oscillation) and ``tu``
    the ultimate period.
    """
    if ku <= 0 or tu <= 0:
        raise ValueError("ku and tu must be positive")
    if controller == "P":
        return TuningResult(kp=0.5 * ku, controller="P", method="Z-N closed loop")
    if controller == "PI":
        return TuningResult(
            kp=0.45 * ku, ki=(0.45 * ku) / (tu / 1.2), controller="PI",
            method="Z-N closed loop",
        )
    return TuningResult(
        kp=0.6 * ku,
        ki=(0.6 * ku) / (tu / 2.0),
        kd=(0.6 * ku) * (tu / 8.0),
        controller="PID",
        method="Z-N closed loop",
    )


# ---------------------------------------------------------------------------
# Cohen-Coon
# ---------------------------------------------------------------------------
def cohen_coon(fopdt: FOPDT, controller: str = "PID") -> TuningResult:
    """Cohen–Coon reaction-curve tuning for a FOPDT model.

    Aims for quarter-amplitude damping and generally gives a faster (but
    more aggressive) response than Ziegler–Nichols.
    """
    K, tau, theta = fopdt.gain, fopdt.tau, fopdt.theta
    if theta <= 0:
        raise ValueError("Cohen-Coon requires dead time > 0")
    r = theta / tau
    if controller == "P":
        kp = (1.0 / K) * (tau / theta) * (1.0 + r / 3.0)
        return TuningResult(kp=kp, controller="P", method="Cohen-Coon")
    if controller == "PI":
        kp = (1.0 / K) * (tau / theta) * (0.9 + r / 12.0)
        ti = theta * (30.0 + 3.0 * r) / (9.0 + 20.0 * r)
        return TuningResult(
            kp=kp, ki=kp / ti, controller="PI", method="Cohen-Coon"
        )
    kp = (1.0 / K) * (tau / theta) * (4.0 / 3.0 + r / 4.0)
    ti = theta * (32.0 + 6.0 * r) / (13.0 + 8.0 * r)
    td = theta * 4.0 / (11.0 + 2.0 * r)
    return TuningResult(
        kp=kp, ki=kp / ti, kd=kp * td, controller="PID", method="Cohen-Coon"
    )


# ---------------------------------------------------------------------------
# IMC / Lambda
# ---------------------------------------------------------------------------
def imc_pi(fopdt: FOPDT, tau_c: float | None = None) -> TuningResult:
    """IMC (Lambda) PI tuning for a FOPDT model.

    ``tau_c`` is the desired closed-loop time constant; when omitted it
    defaults to the dead time ``theta`` (aggressive) — increase it for a
    more sluggish, robust loop.
    """
    K, tau, theta = fopdt.gain, fopdt.tau, fopdt.theta
    tau_c = theta if tau_c is None else float(tau_c)
    if tau_c + theta <= 0:
        raise ValueError("tau_c + theta must be > 0")
    kp = tau / (K * (tau_c + theta))
    return TuningResult(kp=kp, ki=kp / tau, controller="PI", method="IMC (Lambda)")


def imc_pid(fopdt: FOPDT, tau_c: float | None = None) -> TuningResult:
    """IMC (Lambda) PID tuning for a FOPDT model (dominant-dead-time case)."""
    K, tau, theta = fopdt.gain, fopdt.tau, fopdt.theta
    tau_c = theta if tau_c is None else float(tau_c)
    denom = K * (tau_c + theta)
    kp = (tau + theta / 2.0) / denom
    ti = tau + theta / 2.0
    td = tau * theta / (2.0 * tau + theta)
    return TuningResult(kp=kp, ki=kp / ti, kd=kp * td, controller="PID", method="IMC (Lambda)")


# ---------------------------------------------------------------------------
# SIMC (Skogestad)
# ---------------------------------------------------------------------------
def simc_pi(
    fopdt: FOPDT, tau_c: float | None = None
) -> TuningResult:
    """Skogestad's SIMC PI tuning for a FOPDT model.

    ``tau_c`` defaults to ``theta``, giving near-optimal disturbance
    rejection with robust margins.
    """
    K, tau, theta = fopdt.gain, fopdt.tau, fopdt.theta
    tau_c = theta if tau_c is None else float(tau_c)
    kp = tau / (K * (tau_c + theta))
    ti = min(tau, 4.0 * (tau_c + theta))
    return TuningResult(kp=kp, ki=kp / ti, controller="PI", method="SIMC")


def simc_integrator(
    plant: IntegratorDelay, tau_c: float | None = None
) -> TuningResult:
    """SIMC PI tuning for an integrating process ``K/s * exp(-theta*s)``.

    Integrators (level, position) need a much smaller proportional gain to
    keep the loop stable.
    """
    K, theta = plant.gain, plant.theta
    tau_c = theta if tau_c is None else float(tau_c)
    if tau_c + theta <= 0:
        raise ValueError("tau_c + theta must be > 0")
    kp = 1.0 / (K * (tau_c + theta))
    ti = 4.0 * (tau_c + theta)
    return TuningResult(kp=kp, ki=kp / ti, controller="PI", method="SIMC (integrator)")


# ---------------------------------------------------------------------------
# Tyreus-Luyben
# ---------------------------------------------------------------------------
def tyreus_luyben(ku: float, tu: float, controller: str = "PI") -> TuningResult:
    """Tyreus–Luyben closed-loop tuning (more conservative than Z–N)."""
    if ku <= 0 or tu <= 0:
        raise ValueError("ku and tu must be positive")
    if controller == "PI":
        return TuningResult(
            kp=ku / 3.2, ki=(ku / 3.2) / (2.2 * tu), controller="PI",
            method="Tyreus-Luyben",
        )
    return TuningResult(
        kp=ku / 2.2,
        ki=(ku / 2.2) / (2.2 * tu),
        kd=(ku / 2.2) * (tu / 6.3),
        controller="PID",
        method="Tyreus-Luyben",
    )


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------
def auto_tune(
    process: Process,
    method: str = "simc",
    controller: str = "PID",
    tau_c: float | None = None,
) -> TuningResult:
    """Dispatch to a tuning rule by name.

    ``method`` may be one of: ``"zn-open"``, ``"zn-closed"`` (needs
    ``ku``/``tu`` — see :func:`ziegler_nichols_closed_loop`), ``"cohen-coon"``,
    ``"imc"``, ``"simc"``, ``"tyreus-luyben"``.
    """
    name = method.lower().replace("_", "-")
    if name in ("zn", "zn-open", "ziegler-nichols"):
        if not isinstance(process, FOPDT):
            raise TypeError("Z-N open loop requires a FOPDT model")
        return ziegler_nichols_open_loop(process, controller)
    if name == "zn-closed":
        raise ValueError("use ziegler_nichols_closed_loop(ku, tu) directly")
    if name in ("cohen-coon", "cc"):
        if not isinstance(process, FOPDT):
            raise TypeError("Cohen-Coon requires a FOPDT model")
        return cohen_coon(process, controller)
    if name in ("imc", "lambda"):
        if not isinstance(process, FOPDT):
            raise TypeError("IMC tuning requires a FOPDT model")
        return imc_pid(process, tau_c) if controller == "PID" else imc_pi(process, tau_c)
    if name == "simc":
        if isinstance(process, IntegratorDelay):
            return simc_integrator(process, tau_c)
        if isinstance(process, FOPDT):
            return simc_pi(process, tau_c)
        raise TypeError("SIMC requires a FOPDT or IntegratorDelay model")
    if name in ("tyreus-luyben", "tl"):
        raise ValueError("use tyreus_luyben(ku, tu) directly")
    raise ValueError(f"unknown tuning method {method!r}")


def available_methods() -> list[str]:
    """Names of all tuning rules that can be used with :func:`auto_tune`."""
    return [
        "zn-open",
        "zn-closed",
        "cohen-coon",
        "imc",
        "simc",
        "tyreus-luyben",
    ]


__all__: list[str] = [
    "PIDParams",
    "TuningResult",
    "ziegler_nichols_open_loop",
    "ziegler_nichols_closed_loop",
    "cohen_coon",
    "imc_pi",
    "imc_pid",
    "simc_pi",
    "simc_integrator",
    "tyreus_luyben",
    "auto_tune",
    "available_methods",
]
