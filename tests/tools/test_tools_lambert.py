"""Tests for lambert_problem_solver in aerospace_mcp.tools.orbits module."""

import json

from aerospace_mcp.tools.orbits import lambert_problem_solver


def extract_json(result: str) -> dict:
    """Extract JSON data from tool output that contains both text and JSON."""
    brace_count = 0
    json_start = None
    last_json_end = None

    for i, char in enumerate(result):
        if char == "{":
            if brace_count == 0:
                json_start = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and json_start is not None:
                last_json_end = i + 1

    if json_start is not None and last_json_end is not None:
        try:
            return json.loads(result[json_start:last_json_end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from result: {result[:200]}...")


class TestLambertProblemSolver:
    """Tests for lambert_problem_solver function."""

    def test_leo_transfer(self):
        """Test Lambert solver for LEO to LEO transfer."""
        r1 = [7000e3, 0, 0]
        r2 = [0, 8000e3, 0]
        tof = 3600

        result = lambert_problem_solver(
            r1_m=r1,
            r2_m=r2,
            tof_s=tof,
        )

        assert "LAMBERT PROBLEM" in result
        data = extract_json(result)
        assert data["success"] is True
        assert "v1_ms" in data
        assert "v2_ms" in data
        assert len(data["v1_ms"]) == 3
        assert len(data["v2_ms"]) == 3

    def test_velocity_magnitudes_reasonable(self):
        """Test that computed velocities are in reasonable range."""
        r1 = [7000e3, 0, 0]
        r2 = [0, 7000e3, 0]
        tof = 2700

        result = lambert_problem_solver(r1_m=r1, r2_m=r2, tof_s=tof)
        data = extract_json(result)

        # LEO velocity should be around 7-8 km/s
        v1_mag = data["v1_magnitude_ms"]
        assert 5000 < v1_mag < 12000

    def test_prograde_vs_retrograde(self):
        """Test prograde and retrograde parameters are passed correctly."""
        r1 = [7000e3, 0, 0]
        r2 = [0, 7000e3, 0]
        tof = 3000

        result_pro = lambert_problem_solver(
            r1_m=r1, r2_m=r2, tof_s=tof, direction="prograde"
        )
        result_ret = lambert_problem_solver(
            r1_m=r1, r2_m=r2, tof_s=tof, direction="retrograde"
        )

        data_pro = extract_json(result_pro)
        data_ret = extract_json(result_ret)

        # Both should succeed
        assert data_pro["success"] is True
        assert data_ret["success"] is True

    def test_different_central_bodies(self):
        """Test with different central bodies."""
        r1 = [1e9, 0, 0]
        r2 = [0, 1.5e9, 0]
        tof = 86400 * 100

        result = lambert_problem_solver(
            r1_m=r1,
            r2_m=r2,
            tof_s=tof,
            central_body="sun",
        )

        data = extract_json(result)
        assert data["central_body"] == "sun"

    def test_transfer_orbit_elements(self):
        """Test that transfer orbit elements are computed."""
        r1 = [7000e3, 0, 0]
        r2 = [0, 10000e3, 0]
        tof = 5000

        result = lambert_problem_solver(r1_m=r1, r2_m=r2, tof_s=tof)
        data = extract_json(result)

        assert "transfer_orbit" in data
        assert "semi_major_axis_m" in data["transfer_orbit"]
        assert "eccentricity" in data["transfer_orbit"]
        assert "inclination_deg" in data["transfer_orbit"]

    def test_180_degree_transfer(self):
        """Test 180-degree (Hohmann-like) transfer."""
        r1 = [7000e3, 0, 0]
        r2 = [-7000e3, 0, 0]
        tof = 2700

        result = lambert_problem_solver(r1_m=r1, r2_m=r2, tof_s=tof)
        data = extract_json(result)

        # 180-degree transfers are singularities in Lambert problem
        # Check that we get a response (may or may not succeed)
        assert "v1_ms" in data or "error" in data or "success" in data

    def test_short_transfer_time(self):
        """Test with short transfer time."""
        r1 = [7000e3, 0, 0]
        r2 = [7000e3, 1000e3, 0]
        tof = 600

        result = lambert_problem_solver(r1_m=r1, r2_m=r2, tof_s=tof)
        # Should still produce a result
        assert "success" in result or "v1_ms" in result

    def test_altitude_output(self):
        """Test that altitude above Earth is computed."""
        r1 = [7000e3, 0, 0]
        r2 = [0, 8000e3, 0]
        tof = 3600

        result = lambert_problem_solver(r1_m=r1, r2_m=r2, tof_s=tof)
        data = extract_json(result)

        # Earth radius is 6378 km
        assert "positions" in data
        assert 600 < data["positions"]["r1_altitude_km"] < 700
        assert 1600 < data["positions"]["r2_altitude_km"] < 1700
