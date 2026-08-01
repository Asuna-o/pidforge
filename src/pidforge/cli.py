"""Command-line interface: tune, simulate and identify from the terminal."""

from __future__ import annotations

import argparse
import csv
import sys

from .controller import PIDController
from .identification import fit_fopdt
from .metrics import compute_metrics
from .models import _PROCESS_ALIASES, parse_model
from .simulate import simulate
from .tuning import auto_tune, available_methods


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    widths = {
        h: max(len(h), *(len(f"{r[h]:.4g}") if isinstance(r[h], float) else len(str(r[h])) for r in rows))
        for h in headers
    }
    def fmt(h: str, v: object) -> str:
        if isinstance(v, float):
            return f"{v:.4g}".rjust(widths[h])
        return str(v).ljust(widths[h])

    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(fmt(h, row[h]) for h in headers))


def _cmd_models(args: argparse.Namespace) -> int:
    print("Available process models (use with --plant):")
    for name in sorted(_PROCESS_ALIASES):
        cls = _PROCESS_ALIASES[name]
        print(f"  {name:<14} {cls.__doc__.splitlines()[0] if cls.__doc__ else ''}")
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    process = parse_model(args.plant)
    if args.all:
        rows = []
        for method in available_methods():
            if method in ("zn-closed", "tyreus-luyben"):
                continue  # require ku/tu
            try:
                result = auto_tune(process, method=method, controller=args.controller)
                rows.append(
                    {
                        "method": method,
                        "controller": result.controller,
                        "kp": result.kp,
                        "ti": result.ti,
                        "td": result.td,
                    }
                )
            except (TypeError, ValueError) as exc:
                print(f"  {method}: skipped ({exc})", file=sys.stderr)
        _print_table(rows)
        return 0

    result = auto_tune(process, method=args.method, controller=args.controller, tau_c=args.tau_c)
    print(f"Plant: {process!r}")
    print(result.summary())
    print(f"  kp = {result.kp:.4g}")
    print(f"  ki = {result.ki:.4g}")
    if result.kd:
        print(f"  kd = {result.kd:.4g}")
    print(f"  ti = {result.ti:.4g}")
    if result.kd:
        print(f"  td = {result.td:.4g}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    process = parse_model(args.plant)
    if ":" in args.tuning:
        parts = [float(p) for p in args.tuning.split(":")]
        if len(parts) not in (1, 2, 3):
            raise SystemExit("--tuning kp:ki:kd expects 1 to 3 values")
        kp, ki, kd = (parts + [0.0, 0.0])[:3]
        controller = PIDController(kp=kp, ki=ki, kd=kd)
    else:
        params = auto_tune(process, method=args.tuning, controller=args.controller, tau_c=args.tau_c)
        controller = PIDController(kp=params.kp, ki=params.ki, kd=params.kd)

    disturbance = None
    disturbance_time = None
    disturbance_magnitude = 1.0
    if args.disturbance:
        mag, _, t_str = args.disturbance.partition(":")
        disturbance_magnitude = float(mag)
        disturbance_time = float(t_str) if t_str else None
        if disturbance_time is None:
            disturbance = disturbance_magnitude
            disturbance_magnitude = 1.0

    result = simulate(
        process,
        controller,
        setpoint=args.setpoint,
        dt=args.dt,
        t_end=args.horizon,
        disturbance=disturbance,
        disturbance_time=disturbance_time,
        disturbance_magnitude=disturbance_magnitude,
    )

    metrics = compute_metrics(result)
    print(f"Plant: {process!r}")
    print(f"Controller: {controller!r}")
    print("Metrics:")
    for key, value in metrics.as_dict().items():
        print(f"  {key:<20} {value:.4g}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time", "setpoint", "output", "control"])
            for i in range(result.n):
                writer.writerow(
                    [result.time[i], result.setpoint[i], result.output[i], result.control[i]]
                )
        print(f"CSV written to {args.csv}")

    if args.plot:
        try:
            from .plotting import plot_response
        except ImportError as exc:
            print(f"cannot plot: {exc}", file=sys.stderr)
            return 1
        fig = plot_response(result, metrics=metrics, title=f"pidforge simulate — {args.plant}")
        fig.savefig(args.plot, dpi=150)
        print(f"Plot saved to {args.plot}")
    return 0


def _cmd_identify(args: argparse.Namespace) -> int:
    import numpy as np

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    if not rows:
        raise SystemExit("empty CSV file")
    time = np.array([float(r["t"]) for r in rows])
    output = np.array([float(r["y"]) for r in rows])
    u = np.array([float(r["u"]) for r in rows]) if "u" in rows[0] else None

    fit = fit_fopdt(time, output, input_step=u, method=args.method)
    print(fit.summary())
    if args.tune:
        from .identification import tune_from_step_data

        model, params = tune_from_step_data(
            time, output, input_step=u, fit_method=args.method, tuning=args.tune
        )
        print(f"Tuned with '{args.tune}':")
        print(f"  kp = {params.kp:.4g}, ki = {params.ki:.4g}, kd = {params.kd:.4g}")
        print(f"  ti = {params.ti:.4g}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pidforge",
        description="Control-loop design & PID tuning toolkit for automation engineers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser("models", help="list supported process models")
    p_models.set_defaults(func=_cmd_models)

    p_tune = sub.add_parser("tune", help="compute PID gains with a tuning rule")
    p_tune.add_argument("--plant", default="fopdt:1,5,1", help="model spec, e.g. fopdt:2,5,1")
    p_tune.add_argument("--method", default="simc", help=f"tuning rule: {', '.join(available_methods())}")
    p_tune.add_argument("--controller", default="PID", choices=["P", "PI", "PID"])
    p_tune.add_argument("--tau-c", type=float, default=None, help="desired closed-loop time constant")
    p_tune.add_argument("--all", action="store_true", help="show every applicable rule")
    p_tune.set_defaults(func=_cmd_tune)

    p_sim = sub.add_parser("simulate", help="run a closed-loop simulation")
    p_sim.add_argument("--plant", default="fopdt:1,5,1")
    p_sim.add_argument("--tuning", default="simc", help="rule name or explicit kp:ki:kd")
    p_sim.add_argument("--controller", default="PID", choices=["P", "PI", "PID"])
    p_sim.add_argument("--tau-c", type=float, default=None)
    p_sim.add_argument("--setpoint", type=float, default=1.0)
    p_sim.add_argument("--dt", type=float, default=0.02)
    p_sim.add_argument("--horizon", type=float, default=60.0)
    p_sim.add_argument("--disturbance", default=None, help="magnitude[:time] load disturbance, e.g. 0.3:20")
    p_sim.add_argument("--csv", default=None, help="write samples to CSV")
    p_sim.add_argument("--plot", default=None, help="save response plot to PNG")
    p_sim.set_defaults(func=_cmd_simulate)

    p_id = sub.add_parser("identify", help="fit a FOPDT model from step-test CSV")
    p_id.add_argument("csv", help="CSV with columns t,y[,u]")
    p_id.add_argument("--method", default="two_point", choices=["two_point", "area"])
    p_id.add_argument("--tune", default=None, help="also tune, e.g. simc")
    p_id.set_defaults(func=_cmd_identify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, TypeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
