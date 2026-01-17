"""Tests for aerospace_mcp.tools.performance module."""

import json

from aerospace_mcp.tools.performance import (
    density_altitude_calculator,
    fuel_reserve_calculator,
    landing_performance,
    stall_speed_calculator,
    takeoff_performance,
    true_airspeed_converter,
    weight_and_balance,
)


def extract_json(result: str) -> dict:
    """Extract JSON data from tool output that contains both text and JSON."""
    # Find the last complete JSON object in the string
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


class TestDensityAltitudeCalculator:
    """Tests for density_altitude_calculator function."""

    def test_sea_level_isa(self):
        """Test sea level ISA conditions."""
        result = density_altitude_calculator(
            pressure_altitude_ft=0,
            temperature_c=15,  # ISA sea level temp
        )
        assert "DENSITY ALTITUDE" in result
        data = extract_json(result)
        # Should be close to 0 at ISA
        assert -500 < data["density_altitude_ft"] < 500

    def test_hot_day(self):
        """Test hot day increases density altitude."""
        result_isa = density_altitude_calculator(
            pressure_altitude_ft=5000, temperature_c=5
        )
        result_hot = density_altitude_calculator(
            pressure_altitude_ft=5000, temperature_c=35
        )

        isa_data = extract_json(result_isa)
        hot_data = extract_json(result_hot)

        # Hot day should have higher density altitude
        assert hot_data["density_altitude_ft"] > isa_data["density_altitude_ft"]

    def test_high_altitude(self):
        """Test high altitude calculation."""
        result = density_altitude_calculator(
            pressure_altitude_ft=10000,
            temperature_c=-5,
        )
        assert "density_altitude_ft" in result


class TestTrueAirspeedConverter:
    """Tests for true_airspeed_converter function."""

    def test_sea_level_isa(self):
        """Test conversion at sea level ISA."""
        result = true_airspeed_converter(
            speed_value=250,
            speed_type="CAS",
            altitude_ft=0,
            temperature_c=15,
        )
        assert "AIRSPEED CONVERSION" in result
        data = extract_json(result)
        # At sea level ISA, TAS ≈ CAS
        assert 240 < data["TAS_kts"] < 260

    def test_high_altitude(self):
        """Test TAS conversion at altitude."""
        result = true_airspeed_converter(
            speed_value=250,
            speed_type="TAS",
            altitude_ft=35000,
            temperature_c=-50,
        )
        data = extract_json(result)
        # TAS at altitude should be higher than IAS
        # At high altitude, density is lower, so IAS < TAS
        assert data["TAS_kts"] >= data["IAS_kts"]

    def test_mach_input(self):
        """Test Mach number input."""
        result = true_airspeed_converter(
            speed_value=0.8,
            speed_type="MACH",
            altitude_ft=35000,
        )
        data = extract_json(result)
        # Mach should be preserved or close to input
        # Check that we get valid speeds
        assert data["TAS_kts"] > 0
        assert data["MACH"] > 0


class TestStallSpeedCalculator:
    """Tests for stall_speed_calculator function."""

    def test_basic_calculation(self):
        """Test basic stall speed calculation."""
        result = stall_speed_calculator(
            weight_kg=75000,
            wing_area_m2=122,
            cl_max_clean=1.5,
            altitude_ft=0,
        )
        assert "STALL SPEED" in result
        data = extract_json(result)
        # Sanity check: stall speed should be reasonable
        assert 50 < data["stall_speeds"]["VS1_clean_kts"] < 200

    def test_landing_config_lower(self):
        """Test that landing config has lower stall speed."""
        result = stall_speed_calculator(
            weight_kg=75000,
            wing_area_m2=122,
            cl_max_clean=1.5,
            cl_max_landing=2.5,
            altitude_ft=0,
        )
        data = extract_json(result)
        assert (
            data["stall_speeds"]["VS0_landing_kts"]
            < data["stall_speeds"]["VS1_clean_kts"]
        )

    def test_load_factor_effect(self):
        """Test that load factor increases stall speed."""
        result_1g = stall_speed_calculator(
            weight_kg=5000, wing_area_m2=20, cl_max_clean=1.5, load_factor=1.0
        )
        result_2g = stall_speed_calculator(
            weight_kg=5000, wing_area_m2=20, cl_max_clean=1.5, load_factor=2.0
        )

        data_1g = extract_json(result_1g)
        data_2g = extract_json(result_2g)

        # At 2g, stall speed should be ~41% higher (sqrt(2))
        ratio = (
            data_2g["stall_speeds"]["VS1_clean_kts"]
            / data_1g["stall_speeds"]["VS1_clean_kts"]
        )
        assert 1.35 < ratio < 1.45


class TestWeightAndBalance:
    """Tests for weight_and_balance function."""

    def test_basic_calculation(self):
        """Test basic W&B calculation."""
        result = weight_and_balance(
            basic_empty_weight_kg=1000,
            basic_empty_arm_m=2.5,
            fuel_kg=200,
            fuel_arm_m=2.8,
            payload_items=[
                {"weight_kg": 80, "arm_m": 2.0, "name": "Pilot"},
                {"weight_kg": 20, "arm_m": 3.5, "name": "Baggage"},
            ],
        )
        assert "WEIGHT AND BALANCE" in result
        data = extract_json(result)
        assert data["totals"]["total_weight_kg"] == 1300

    def test_cg_within_limits(self):
        """Test CG limit checking."""
        result = weight_and_balance(
            basic_empty_weight_kg=1000,
            basic_empty_arm_m=2.5,
            fuel_kg=200,
            fuel_arm_m=2.8,
            payload_items=[],
            forward_cg_limit_m=2.0,
            aft_cg_limit_m=3.0,
        )
        data = extract_json(result)
        assert data["status"]["cg_within_limits"] is True

    def test_overweight_warning(self):
        """Test overweight warning."""
        result = weight_and_balance(
            basic_empty_weight_kg=1000,
            basic_empty_arm_m=2.5,
            fuel_kg=500,
            fuel_arm_m=2.8,
            payload_items=[{"weight_kg": 200, "arm_m": 2.5, "name": "Cargo"}],
            max_takeoff_weight_kg=1500,
        )
        data = extract_json(result)
        assert data["status"]["weight_within_limits"] is False
        assert "OVERWEIGHT" in str(data["status"]["warnings"])


class TestTakeoffPerformance:
    """Tests for takeoff_performance function."""

    def test_basic_calculation(self):
        """Test basic takeoff calculation."""
        result = takeoff_performance(
            weight_kg=75000,
            pressure_altitude_ft=0,
            temperature_c=15,
            wind_kts=0,
            runway_slope_pct=0,
            runway_condition="dry",
        )
        assert "TAKEOFF PERFORMANCE" in result
        data = extract_json(result)
        assert "v_speeds_kts" in data
        assert "distances_m" in data

    def test_headwind_reduces_distance(self):
        """Test that headwind reduces takeoff distance."""
        result_calm = takeoff_performance(
            weight_kg=50000, pressure_altitude_ft=0, temperature_c=15, wind_kts=0
        )
        result_hw = takeoff_performance(
            weight_kg=50000, pressure_altitude_ft=0, temperature_c=15, wind_kts=20
        )

        data_calm = extract_json(result_calm)
        data_hw = extract_json(result_hw)

        assert (
            data_hw["distances_m"]["total_distance"]
            < data_calm["distances_m"]["total_distance"]
        )

    def test_wet_runway_increases_distance(self):
        """Test that wet runway increases factored distance."""
        result_dry = takeoff_performance(
            weight_kg=50000,
            pressure_altitude_ft=0,
            temperature_c=15,
            runway_condition="dry",
        )
        result_wet = takeoff_performance(
            weight_kg=50000,
            pressure_altitude_ft=0,
            temperature_c=15,
            runway_condition="wet",
        )

        data_dry = extract_json(result_dry)
        data_wet = extract_json(result_wet)

        assert (
            data_wet["distances_m"]["factored_distance"]
            > data_dry["distances_m"]["factored_distance"]
        )


class TestLandingPerformance:
    """Tests for landing_performance function."""

    def test_basic_calculation(self):
        """Test basic landing calculation."""
        result = landing_performance(
            weight_kg=65000,
            pressure_altitude_ft=0,
            temperature_c=15,
            wind_kts=0,
            runway_slope_pct=0,
            runway_condition="dry",
        )
        assert "LANDING PERFORMANCE" in result
        data = extract_json(result)
        assert "v_speeds_kts" in data
        assert "VS0" in data["v_speeds_kts"]

    def test_wet_runway_factor(self):
        """Test wet runway factor increases distance."""
        result_dry = landing_performance(
            weight_kg=50000,
            pressure_altitude_ft=0,
            temperature_c=15,
            runway_condition="dry",
        )
        result_wet = landing_performance(
            weight_kg=50000,
            pressure_altitude_ft=0,
            temperature_c=15,
            runway_condition="wet",
        )

        data_dry = extract_json(result_dry)
        data_wet = extract_json(result_wet)

        # Wet runway should have longer factored distance
        assert (
            data_wet["distances_m"]["factored_distance"]
            > data_dry["distances_m"]["factored_distance"]
        )


class TestFuelReserveCalculator:
    """Tests for fuel_reserve_calculator function."""

    def test_far91_reserves(self):
        """Test FAR 91 reserve calculation."""
        result = fuel_reserve_calculator(
            regulation="FAR_91",
            trip_fuel_kg=1000,
            cruise_fuel_flow_kg_hr=200,
            flight_time_min=120,
            alternate_fuel_kg=100,
        )
        assert "FUEL RESERVE" in result
        data = extract_json(result)
        # FAR 91 requires 45 min reserve
        assert data["reserves"]["final_reserve_kg"] > 0
        assert data["total_required_fuel_kg"] > data["trip_fuel_kg"]

    def test_far121_includes_contingency(self):
        """Test FAR 121 includes 10% contingency."""
        result = fuel_reserve_calculator(
            regulation="FAR_121",
            trip_fuel_kg=10000,
            cruise_fuel_flow_kg_hr=5000,
            flight_time_min=180,
        )
        data = extract_json(result)
        # FAR 121 requires 10% contingency
        assert data["reserves"]["contingency_kg"] == 1000  # 10% of 10000

    def test_icao_reserves(self):
        """Test ICAO reserve calculation."""
        result = fuel_reserve_calculator(
            regulation="ICAO",
            trip_fuel_kg=5000,
            cruise_fuel_flow_kg_hr=2000,
            flight_time_min=150,
            alternate_fuel_kg=500,
        )
        data = extract_json(result)
        # ICAO requires 5% contingency and 30 min final reserve
        assert data["reserves"]["contingency_kg"] >= 250  # At least 5% of trip
        assert data["reserves"]["final_reserve_kg"] > 0
