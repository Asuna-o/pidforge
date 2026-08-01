"""pidforge - control-loop design and PID tuning toolkit for automation engineers."""

from .controller import DEFAULT_DERIVATIVE_FILTER, PIDController, pid_from_gains
from .identification import FOPDTFit, fit_fopdt, tune_from_step_data
from .metrics import Metrics, compute_metrics
from .models import (
    FOPDT,
    SOPDT,
    IntegratorDelay,
    Oscillator,
    Process,
    fopdt_from_params,
    parse_model,
)
from .simulate import (
    SimulationResult,
    open_loop_step,
    simulate,
    step_setpoint,
    tuned_simulation,
)
from .tuning import (
    PIDParams,
    TuningResult,
    auto_tune,
    cohen_coon,
    imc_pi,
    imc_pid,
    simc_integrator,
    simc_pi,
    tyreus_luyben,
    ziegler_nichols_closed_loop,
    ziegler_nichols_open_loop,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Process",
    "FOPDT",
    "SOPDT",
    "Oscillator",
    "IntegratorDelay",
    "parse_model",
    "fopdt_from_params",
    "PIDController",
    "pid_from_gains",
    "DEFAULT_DERIVATIVE_FILTER",
    "PIDParams",
    "TuningResult",
    "auto_tune",
    "ziegler_nichols_open_loop",
    "ziegler_nichols_closed_loop",
    "cohen_coon",
    "imc_pi",
    "imc_pid",
    "simc_pi",
    "simc_integrator",
    "tyreus_luyben",
    "simulate",
    "open_loop_step",
    "step_setpoint",
    "tuned_simulation",
    "SimulationResult",
    "compute_metrics",
    "Metrics",
    "fit_fopdt",
    "FOPDTFit",
    "tune_from_step_data",
]
