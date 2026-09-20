"""Mass-axis calibration from reference calibration points.

Each calibration point pairs an *observed* m/z with a neutral formula,
adduct and charge; the theoretical monoisotopic m/z is computed with the
existing formula parsing, adduct application and electron-mass rules shared
with ``/api/v1/ion-envelope``. The observed mass axis is then modelled as a
linear function of the theoretical m/z::

    observed = slope * theoretical + intercept

The model is built with a median-based (Theil-Sen style) initial estimate
followed by one least-squares refinement on the inliers:

1. Every pair of valid points with distinct theoretical m/z contributes a
   slope ``(y_j - y_i) / (x_j - x_i)``; the initial slope is the median of
   those pairwise slopes. Each valid point then contributes an intercept
   ``y - slope0 * x`` and the initial intercept is their median.
2. A point is an inlier when its residual against the initial model,
   ``y - (slope0 * x + intercept0)``, is within ``tolerance_ppm`` in
   relative (ppm) terms.
3. The inliers are refit once with an ordinary least-squares line.

The fitted model maps *theoretical -> observed*; an observed m/z is
calibrated back onto the theoretical axis with its inverse::

    theoretical = (observed - intercept) / slope
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .adduct import AdductError, apply_adduct, parse_adduct
from .envelope import ELECTRON_MASS_DA, monoisotopic_mass
from .parser import FormulaError, ParsedFormula, parse_formula

# Accepted calibration-point count bounds.
MIN_POINTS = 3
MAX_POINTS = 200
# A refinement needs at least this many inliers (two points never determine
# a straight line uniquely through the median/OLS machinery).
MIN_INLIERS = 3


class CalibrationError(ValueError):
    """Request-level calibration failure carrying a machine-readable code."""

    def __init__(self, error: str, message: str) -> None:
        super().__init__(message)
        self.error = error
        self.message = message


@dataclass
class CalibrationPoint:
    """One calibration point plus its validation/fit result.

    Invalid points keep ``theoretical_mz``/``residual_ppm`` as ``None`` and
    carry the same machine-readable error metadata as the formula/adduct
    endpoints.
    """

    index: int
    observed_mz: float
    formula: str
    adduct: str
    charge: int
    valid: bool
    theoretical_mz: float | None = None
    residual_ppm: float | None = None
    inlier: bool = False
    error: str | None = None
    message: str | None = None
    position: int | None = None


@dataclass
class MassCalibrationResult:
    slope: float
    intercept: float
    inlier_count: int
    points: list[CalibrationPoint]


def theoretical_monomz(
    formula: str, adduct: str, charge: int
) -> tuple[float, ParsedFormula]:
    """Theoretical monoisotopic m/z of an adducted ion.

    Reuses :func:`parse_formula`, :func:`parse_adduct`/:func:`apply_adduct`
    and the ``(mass - charge * m_e) / |charge|`` electron-mass rule of the
    ion-envelope endpoints. Raises :class:`FormulaError` or
    :class:`AdductError` exactly as those endpoints do.
    """
    parsed = parse_formula(formula)
    pieces = parse_adduct(adduct)
    adjusted = apply_adduct(parsed, pieces)
    mono_mz = (
        monoisotopic_mass(adjusted) - charge * ELECTRON_MASS_DA
    ) / abs(charge)
    return mono_mz, adjusted


def _residual_ppm(x: float, y: float, slope: float, intercept: float) -> float:
    """Residual ``y - predicted`` in ppm relative to the predicted m/z."""
    predicted = slope * x + intercept
    return (y - predicted) / predicted * 1e6


def mass_calibration(
    points: list[CalibrationPoint], tolerance_ppm: float
) -> MassCalibrationResult:
    """Fit the observed -> theoretical mass-axis linear calibration.

    Invalid points (bad formula/adduct) are skipped but returned in place.
    Raises :class:`CalibrationError` when no usable point pair exists, fewer
    than :data:`MIN_INLIERS` points fall within ``tolerance_ppm`` of the
    initial model, or the resulting slope is not positive.
    """
    valid = [(p, p.theoretical_mz, p.observed_mz) for p in points if p.valid]

    pairwise_slopes: list[float] = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            dx = valid[j][1] - valid[i][1]
            if dx == 0.0:
                continue
            pairwise_slopes.append((valid[j][2] - valid[i][2]) / dx)

    if not pairwise_slopes:
        raise CalibrationError(
            "no_valid_pairs",
            "No pair of valid calibration points with distinct theoretical "
            "m/z; at least two valid points at different m/z are required",
        )

    slope0 = _median(pairwise_slopes)
    if slope0 <= 0.0:
        raise CalibrationError(
            "non_positive_slope",
            f"Initial median slope {slope0} is not positive; the observed "
            "m/z must increase with the theoretical m/z",
        )
    intercept0 = _median([y - slope0 * x for _, x, y in valid])

    inlier_points = [
        item
        for item in valid
        if abs(_residual_ppm(item[1], item[2], slope0, intercept0))
        <= tolerance_ppm
    ]
    for point, _, _ in inlier_points:
        point.inlier = True

    if len(inlier_points) < MIN_INLIERS:
        raise CalibrationError(
            "insufficient_inliers",
            f"Only {len(inlier_points)} of {len(valid)} valid calibration "
            f"points lie within {tolerance_ppm} ppm of the initial median "
            f"model; at least {MIN_INLIERS} inliers are required",
        )

    # One ordinary least-squares refinement on the inliers:
    # slope = Sxy / Sxx, intercept = mean(y) - slope * mean(x).
    n = len(inlier_points)
    xs = [item[1] for item in inlier_points]
    ys = [item[2] for item in inlier_points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        raise CalibrationError(
            "non_positive_slope",
            "The inlier points all share one theoretical m/z, so a "
            "least-squares slope cannot be determined",
        )
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if slope <= 0.0:
        raise CalibrationError(
            "non_positive_slope",
            f"Least-squares slope {slope} is not positive; the observed m/z "
            "must increase with the theoretical m/z",
        )

    for point, x, y in valid:
        point.residual_ppm = _residual_ppm(x, y, slope, intercept)

    return MassCalibrationResult(
        slope=slope,
        intercept=intercept,
        inlier_count=len(inlier_points),
        points=points,
    )


def _median(values: list[float]) -> float:
    """Median of an even-length list: mean of the two central values."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def calibrate_mz(observed_mz: float, slope: float, intercept: float) -> float:
    """Invert the fitted line to map an observed m/z onto the true axis."""
    return (observed_mz - intercept) / slope


__all__ = [
    "MIN_POINTS",
    "MAX_POINTS",
    "MIN_INLIERS",
    "CalibrationError",
    "CalibrationPoint",
    "MassCalibrationResult",
    "theoretical_monomz",
    "mass_calibration",
    "calibrate_mz",
]
