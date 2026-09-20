"""API tests for POST /api/v1/mass-calibrate."""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.mass_calibration import theoretical_monomz

REFERENCES = [
    ("H2O", "+H", 1),
    ("CH4O", "+H", 1),
    ("C2H6O", "+H", 1),
    ("C3H8O", "+H", 1),
    ("C6H12O6", "+Na", 1),
]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def _payload(slope=1.0002, intercept=-0.05, tolerance=10.0, mz=None, bad=None):
    points = []
    for i, (formula, adduct, charge) in enumerate(REFERENCES):
        x, _ = theoretical_monomz(formula, adduct, charge)
        observed = slope * x + intercept
        if bad == "outlier" and i == 2:
            observed *= 1.01
        points.append(
            {
                "observed_mz": round(observed, 6),
                "formula": formula,
                "adduct": adduct,
                "charge": charge,
            }
        )
    return {
        "calibration_points": points,
        "tolerance_ppm": tolerance,
        "mz": [100.0, 203.052609] if mz is None else mz,
    }


def test_calibrate_recovers_line_and_keeps_input_order(client):
    resp = client.post("/api/v1/mass-calibrate", json=_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["slope"] == pytest.approx(1.0002, abs=1e-7)
    assert body["intercept"] == pytest.approx(-0.05, abs=1e-6)
    assert body["valid_point_count"] == 5
    assert body["inlier_count"] == 5
    assert [p["index"] for p in body["calibration_points"]] == [0, 1, 2, 3, 4]
    for point in body["calibration_points"]:
        assert point["valid"] is True
        assert point["inlier"] is True
        assert point["error"] is None
        assert abs(point["residual_ppm"]) < 1.0
    # Inverse transform: calibrated = (observed - intercept) / slope.
    expected = [(100.0 - body["intercept"]) / body["slope"]]
    assert body["calibrated_mz"][0] == pytest.approx(expected[0], abs=1e-5)
    assert body["mz"] == [100.0, 203.052609]
    assert len(body["calibrated_mz"]) == 2


def test_calibrate_marks_outlier_and_refits_inliers(client):
    resp = client.post(
        "/api/v1/mass-calibrate", json=_payload(bad="outlier", tolerance=10.0)
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["inlier_count"] == 4
    flags = [p["inlier"] for p in body["calibration_points"]]
    assert flags == [True, True, False, True, True]
    assert body["slope"] == pytest.approx(1.0002, abs=1e-7)
    outlier = body["calibration_points"][2]
    assert abs(outlier["residual_ppm"]) == pytest.approx(10000.0, rel=2e-3)
    # Inliers still sit on the refined line.
    for i in (0, 1, 3, 4):
        assert abs(body["calibration_points"][i]["residual_ppm"]) < 1.0


def test_invalid_points_reported_in_place_not_422(client):
    payload = _payload()
    payload["calibration_points"][1]["formula"] = "C6Xx2"
    payload["calibration_points"][3]["adduct"] = "-99H"
    resp = client.post("/api/v1/mass-calibrate", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid_point_count"] == 3
    assert body["inlier_count"] == 3
    points = body["calibration_points"]
    assert points[1]["valid"] is False
    assert points[1]["error"] == "invalid_formula"
    assert points[1]["position"] == 2
    assert points[1]["theoretical_mz"] is None
    assert points[1]["residual_ppm"] is None
    assert points[1]["inlier"] is False
    assert points[3]["valid"] is False
    assert points[3]["error"] == "invalid_adduct"
    assert points[3]["position"] is not None
    # Echoed fields retain input order.
    assert points[1]["formula"] == "C6Xx2"
    assert points[3]["adduct"] == "-99H"


def test_all_points_invalid_returns_422_no_valid_pairs(client):
    payload = _payload()
    for point in payload["calibration_points"]:
        point["formula"] = "Xx"
    resp = client.post("/api/v1/mass-calibrate", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "no_valid_pairs"


def test_fewer_than_three_inliers_returns_422(client):
    resp = client.post(
        "/api/v1/mass-calibrate", json=_payload(bad="outlier", tolerance=0.001)
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "insufficient_inliers"


def test_negative_slope_returns_422(client):
    # Descending observed m/z against ascending references.
    payload = _payload()
    descending = [500.0, 400.0, 300.0, 200.0, 100.0]
    for point, observed in zip(payload["calibration_points"], descending):
        point["observed_mz"] = observed
    resp = client.post("/api/v1/mass-calibrate", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "non_positive_slope"


def test_point_count_bounds(client):
    payload = _payload()
    payload["calibration_points"] = payload["calibration_points"][:2]
    assert client.post("/api/v1/mass-calibrate", json=payload).status_code == 422

    payload = _payload()
    base = payload["calibration_points"][0]
    payload["calibration_points"] = [dict(base) for _ in range(201)]
    assert client.post("/api/v1/mass-calibrate", json=payload).status_code == 422


def test_scalar_and_field_validation(client):
    # tolerance_ppm must be positive and finite.
    assert client.post(
        "/api/v1/mass-calibrate", json=_payload(tolerance=0.0)
    ).status_code == 422
    assert client.post(
        "/api/v1/mass-calibrate", json=_payload(tolerance=-1.0)
    ).status_code == 422

    # observed_mz must be positive.
    payload = _payload()
    payload["calibration_points"][0]["observed_mz"] = 0.0
    assert client.post("/api/v1/mass-calibrate", json=payload).status_code == 422

    # charge zero is rejected by the point model.
    payload = _payload()
    payload["calibration_points"][0]["charge"] = 0
    assert client.post("/api/v1/mass-calibrate", json=payload).status_code == 422

    # mz list must be non-empty and positive.
    assert client.post(
        "/api/v1/mass-calibrate", json=_payload(mz=[])
    ).status_code == 422
    assert client.post(
        "/api/v1/mass-calibrate", json=_payload(mz=[-5.0])
    ).status_code == 422
    # A literal NaN token is accepted by Python's JSON parser but rejected
    # by the field validator; send it as raw content because httpx refuses
    # to encode float('nan') itself.
    raw = client.post(
        "/api/v1/mass-calibrate",
        content='{"calibration_points": '
        + json.dumps(_payload()["calibration_points"])
        + ', "tolerance_ppm": 10.0, "mz": [NaN]}',
        headers={"Content-Type": "application/json"},
    )
    assert raw.status_code == 422


def test_result_mz_order_follows_input(client):
    payload = _payload(mz=[203.052609, 19.017841, 47.049141])
    body = client.post("/api/v1/mass-calibrate", json=payload).json()
    assert body["mz"] == [203.052609, 19.017841, 47.049141]
    slope, intercept = body["slope"], body["intercept"]
    assert body["calibrated_mz"] == pytest.approx(
        [(m - intercept) / slope for m in body["mz"]], abs=1e-5
    )


def test_theoretical_mz_uses_adduct_and_electron_mass(client):
    payload = _payload()
    body = client.post("/api/v1/mass-calibrate", json=payload).json()
    by_formula = {p["formula"]: p for p in body["calibration_points"]}
    assert by_formula["H2O"]["theoretical_mz"] == pytest.approx(19.017841, abs=1e-6)
    assert by_formula["C2H6O"]["theoretical_mz"] == pytest.approx(47.049141, abs=1e-6)
    assert by_formula["C6H12O6"]["theoretical_mz"] == pytest.approx(
        203.052609, abs=1e-6
    )
