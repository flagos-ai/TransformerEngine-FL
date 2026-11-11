#!/usr/bin/env python3
"""Compute regression against a baseline FlagCICD report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    """Load a FlagCICD report or a raw ``coverage json`` report."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "flagcicd.report/v1" and "totals" not in payload:
        raise ValueError(f"unsupported schema_version: {payload.get('schema_version')}")
    return payload


def coverage_percent(report: dict[str, Any]) -> float | None:
    """Extract coverage percentage from a report."""
    if "totals" in report:
        return report["totals"].get("percent_covered")
    coverage = report.get("metrics", {}).get("coverage", {})
    if not coverage.get("available"):
        return None
    return coverage.get("percent", 0.0)


def regression_verdict(
    current: float | None, baseline: float | None, threshold: float
) -> dict[str, Any]:
    """Determine if current coverage regresses against baseline."""
    if current is None:
        return {
            "status": "no_coverage",
            "message": "Current report has no coverage data",
        }
    if baseline is None:
        return {
            "status": "no_baseline",
            "message": "Baseline report has no coverage data; cannot compare",
            "current_percent": current,
        }
    delta = current - baseline
    if delta < -threshold:
        return {
            "status": "regression",
            "message": f"Coverage dropped {abs(delta):.2f}% (threshold: {threshold:.2f}%)",
            "current_percent": current,
            "baseline_percent": baseline,
            "delta": delta,
        }
    return {
        "status": "acceptable",
        "message": f"Coverage change: {delta:+.2f}%",
        "current_percent": current,
        "baseline_percent": baseline,
        "delta": delta,
    }


def statistical_regression(
    current: float | None,
    historical: list[float],
    sigma_multiplier: float = 3.0,
) -> dict[str, Any]:
    """Check if current is within mean ± sigma_multiplier * std of historical."""
    if current is None:
        return {
            "status": "no_coverage",
            "message": "Current report has no coverage data",
        }
    if not historical:
        return {
            "status": "no_history",
            "message": "No historical data; cannot compute statistical baseline",
            "current_percent": current,
        }
    mean = sum(historical) / len(historical)
    if len(historical) == 1:
        return {
            "status": "insufficient_history",
            "message": "Only one historical point; assuming acceptable",
            "current_percent": current,
            "baseline_mean": mean,
        }
    variance = sum((value - mean) ** 2 for value in historical) / (len(historical) - 1)
    std = math.sqrt(variance)
    # Zero variance collapses both bounds onto the mean: any drop is an outlier.
    lower_bound = mean - sigma_multiplier * std
    upper_bound = mean + sigma_multiplier * std
    if current < lower_bound:
        return {
            "status": "outlier_below",
            "message": (
                f"Coverage {current:.2f}% is below {sigma_multiplier}σ bound ({lower_bound:.2f}%)"
                + (" (history has zero variance)" if std == 0 else "")
            ),
            "current_percent": current,
            "baseline_mean": mean,
            "baseline_std": std,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
    if current > upper_bound:
        return {
            "status": "outlier_above",
            "message": (
                f"Coverage {current:.2f}% is above {sigma_multiplier}σ bound ({upper_bound:.2f}%)"
            ),
            "current_percent": current,
            "baseline_mean": mean,
            "baseline_std": std,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
    return {
        "status": "within_bounds",
        "message": (
            f"Coverage {current:.2f}% is within {sigma_multiplier}σ bounds [{lower_bound:.2f}%,"
            f" {upper_bound:.2f}%]"
        ),
        "current_percent": current,
        "baseline_mean": mean,
        "baseline_std": std,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True, help="Current FlagCICD report")
    parser.add_argument("--baseline", type=Path, help="Single baseline FlagCICD report")
    parser.add_argument(
        "--historical", type=Path, help="JSON array of historical coverage percentages"
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0, help="Absolute threshold in percent"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Sigma multiplier for statistical bounds",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output regression verdict JSON")
    args = parser.parse_args()

    current_report = load_report(args.current)
    current_percent = coverage_percent(current_report)
    verdict: dict[str, Any] = {
        "schema_version": "flagcicd.regression/v1",
        "current_report": str(args.current),
        "method": "none",
    }

    if args.baseline:
        baseline_report = load_report(args.baseline)
        baseline_percent = coverage_percent(baseline_report)
        verdict["method"] = "absolute_threshold"
        verdict["threshold"] = args.threshold
        verdict["baseline_report"] = str(args.baseline)
        verdict["result"] = regression_verdict(current_percent, baseline_percent, args.threshold)
    elif args.historical:
        historical_data = json.loads(args.historical.read_text(encoding="utf-8"))
        if not isinstance(historical_data, list) or not all(
            isinstance(value, (int, float)) for value in historical_data
        ):
            raise ValueError("--historical must be a JSON array of numbers")
        verdict["method"] = "statistical"
        verdict["sigma"] = args.sigma
        verdict["historical_count"] = len(historical_data)
        verdict["result"] = statistical_regression(current_percent, historical_data, args.sigma)
    else:
        verdict["result"] = {
            "status": "no_comparison",
            "message": "No baseline or historical data provided",
            "current_percent": current_percent,
        }

    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Regression verdict: {verdict['result']['status']}")
    print(f"  {verdict['result']['message']}")
    return 1 if verdict["result"]["status"] in {"regression", "outlier_below"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
