"""Tests for the median/least-squares mass-axis calibration module."""

import pytest

from app.mass_calibration import (
    CalibrationError,
    CalibrationPoint,
    MIN_INLIERS,
    calibrate_mz,
    mass_calibration,
    theoretical_monomz,
)

REFERENCES = [
    ("H2O", "+H", 1),
    ("CH4O", "+H", 1),
    ("C2H6O", "+H", 1),
    ("C3H8O", "+H", 1),
    ("C6H12O6", "+Na", 1),
]


def _point(index, formula, adduct, charge, observed_mz, valid=True, mono_mz=None):
    if valid and mono_mz is None:
        mono_mz, _ = theoretical_monomz(formula, adduct, charge)
    return CalibrationPoint(
        index=index,
        observed_mz=observed_mz,
        formula=formula,
        adduct=adduct,
        charge=charge,
        valid=valid,
        theoretical_mz=mono_mz,
    )


def _linear_points(slope, intercept, outlier_index=None, outlier_factor=1.01):
    points = []
    for i, (formula, adduct, charge) in enumerate(REFERENCES):
        x, _ = theoretical_monomz(formula, adduct, charge)
        y = slope * x + intercept
        if i == outlier_index:
            y *= outlier_factor
        points.append(_point(i, formula, adduct, charge, y, mono_mz=x))
    return points


def test_theoretical_monomz_reuses_electron_mass_rule():
    mz, _ = theoretical_monomz("C2H6O", "+H", 1)
    # neutral 46.041865 + proton fragment 1.007825, minus one electron, / 1
    assert mz == pytest.approx(47.049141, abs=1e-6)
    neg, _ = theoretical_monomz("C2H6O", "-H", -1)
    assert neg == pytest.approx(45.034588, abs=1e-6)
    doubly, _ = theoretical_monomz("C6H12O6", "+2H", 2)
    assert doubly == pytest.approx(91.038971, abs=1e-6)


def test_perfect_identity_line():
    points = _linear_points(1.0, 0.0)
    result = mass_calibration(points, 5.0)
    assert result.slope == pytest.approx(1.0, abs=1e-12)
    assert result.intercept == pytest.approx(0.0, abs=1e-10)
    assert result.inlier_count == len(REFERENCES)
    assert all(p.inlier for p in result.points)
    assert all(abs(p.residual_ppm) < 1e-6 for p in result.points)


def test_recovers_known_line_and_rejects_outlier():
    points = _linear_points(1.0002, -0.05, outlier_index=2, outlier_factor=1.01)
    result = mass_calibration(points, 10.0)
    assert result.slope == pytest.approx(1.0002, abs=1e-12)
    assert result.intercept == pytest.approx(-0.05, abs=1e-10)
    assert result.inlier_count == len(REFERENCES) - 1
    assert [p.inlier for p in result.points] == [True, True, False, True, True]
    # The outlier's residual is reported against the *final* refined line.
    assert abs(result.points[2].residual_ppm) == pytest.approx(10000.0, rel=1e-3)
    for i in (0, 1, 3, 4):
        assert abs(result.points[i].residual_ppm) < 1e-6


def test_inverse_transform_recovers_theoretical_mz():
    slope, intercept = 0.9997, 0.12
    points = _linear_points(slope, intercept)
    result = mass_calibration(points, 5.0)
    observed = slope * 123.456 + intercept
    assert calibrate_mz(observed, result.slope, result.intercept) == pytest.approx(
        123.456, abs=1e-9
    )


def test_invalid_points_are_skipped_but_kept_in_place():
    points = _linear_points(1.0001, 0.02)
    points[1] = CalibrationPoint(
        index=1,
        observed_mz=33.0,
        formula="C6Xx2",
        adduct="+H",
        charge=1,
        valid=False,
        error="invalid_formula",
        message="Unknown element symbol 'Xx'",
        position=2,
    )
    points[3] = CalibrationPoint(
        index=3,
        observed_mz=61.0,
        formula="C3H8O",
        adduct="-99H",
        charge=1,
        valid=False,
        error="invalid_adduct",
        message="negative count",
        position=0,
    )
    result = mass_calibration(points, 10.0)
    assert [p.index for p in result.points] == [0, 1, 2, 3, 4]
    assert points[1].inlier is False and points[1].residual_ppm is None
    assert points[3].inlier is False and points[3].theoretical_mz is None
    assert result.inlier_count == 3
    assert result.slope == pytest.approx(1.0001, abs=1e-10)
    assert result.intercept == pytest.approx(0.02, abs=1e-8)


def test_all_invalid_raises_no_valid_pairs():
    points = [
        CalibrationPoint(
            index=i, observed_mz=100.0 + i, formula="Xx", adduct="+H",
            charge=1, valid=False, error="invalid_formula",
        )
        for i in range(3)
    ]
    with pytest.raises(CalibrationError) as exc:
        mass_calibration(points, 5.0)
    assert exc.value.error == "no_valid_pairs"


def test_distinct_formulas_same_theoretical_mz_raises_no_valid_pairs():
    x, _ = theoretical_monomz("H2O", "+H", 1)
    points = [
        _point(0, "H2O", "+H", 1, x + 0.001, mono_mz=x),
        _point(1, "H2O", "+H", 1, x - 0.001, mono_mz=x),
        _point(2, "H2O", "+H", 1, x + 0.002, mono_mz=x),
    ]
    with pytest.raises(CalibrationError) as exc:
        mass_calibration(points, 1000.0)
    assert exc.value.error == "no_valid_pairs"


def test_fewer_than_three_inliers_raises():
    # Only 3 points: two on the line, one far away -> 2 inliers < 3.
    x0, _ = theoretical_monomz("H2O", "+H", 1)
    x1, _ = theoretical_monomz("C2H6O", "+H", 1)
    x2, _ = theoretical_monomz("C6H12O6", "+Na", 1)
    points = [
        _point(0, "H2O", "+H", 1, x0, mono_mz=x0),
        _point(1, "C2H6O", "+H", 1, x1, mono_mz=x1),
        _point(2, "C6H12O6", "+Na", 1, x2 * 1.05, mono_mz=x2),
    ]
    with pytest.raises(CalibrationError) as exc:
        mass_calibration(points, 10.0)
    assert exc.value.error == "insufficient_inliers"


def test_negative_median_slope_raises():
    # Observed m/z decreases while theoretical increases: every pairwise
    # slope is negative, so the median initial slope is non-positive.
    xs_mz = [theoretical_monomz(*ref)[0] for ref in REFERENCES]
    ordered = sorted(xs_mz)
    points = []
    for i, ((formula, adduct, charge), x) in enumerate(zip(REFERENCES, xs_mz)):
        rank = ordered.index(x)
        points.append(
            _point(
                i, formula, adduct, charge,
                300.0 - 5.0 * rank, mono_mz=x,
            )
        )
    with pytest.raises(CalibrationError) as exc:
        mass_calibration(points, 1e6)
    assert exc.value.error == "non_positive_slope"


def test_min_inliers_constant_is_three():
    assert MIN_INLIERS == 3
