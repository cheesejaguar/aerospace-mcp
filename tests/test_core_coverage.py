"""Additional tests for 100% core.py coverage."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aerospace_mcp.core import (
    FlightPlanError,
    OpenAPError,
    PlanRequest,
    PlanResponse,
    _find_city_airports,
    airports_by_city,
    create_flight_plan,
    estimates_openap,
    get_health_status,
    health,
    plan_flight,
)


class TestHealthFunction:
    """Tests for the core health() function."""

    @pytest.mark.unit
    def test_health_returns_dict(self):
        """Test that health() returns proper structure."""
        result = health()
        assert isinstance(result, dict)
        assert "status" in result
        assert "openap" in result
        assert "airports_count" in result
        assert result["status"] == "ok"

    @pytest.mark.unit
    def test_health_with_openap_available(self):
        """Test health() when OpenAP is available."""
        with patch("aerospace_mcp.core.OPENAP_AVAILABLE", True):
            result = health()
            assert result["openap"] is True

    @pytest.mark.unit
    def test_health_without_openap(self):
        """Test health() when OpenAP is not available."""
        with patch("aerospace_mcp.core.OPENAP_AVAILABLE", False):
            result = health()
            assert result["openap"] is False


class TestAirportsByCityFunction:
    """Tests for the airports_by_city() function."""

    @pytest.mark.unit
    def test_airports_by_city_basic(self, mock_airports_iata):
        """Test airports_by_city with valid input."""
        results = airports_by_city("San Jose", "US")
        assert len(results) == 1
        assert results[0].iata == "SJC"

    @pytest.mark.unit
    def test_airports_by_city_no_country(self, mock_airports_iata):
        """Test airports_by_city without country."""
        results = airports_by_city("San Jose")
        assert len(results) >= 1

    @pytest.mark.unit
    def test_airports_by_city_empty(self, mock_airports_iata):
        """Test airports_by_city with non-existent city."""
        results = airports_by_city("NonexistentCity123")
        assert results == []


class TestGetHealthStatus:
    """Tests for get_health_status() function."""

    @pytest.mark.unit
    def test_get_health_status_basic(self):
        """Test get_health_status returns complete info."""
        with patch(
            "aerospace_mcp.integrations.get_domain_status",
            return_value={"atmosphere": True, "orbits": False},
        ):
            result = get_health_status()

            assert "status" in result
            assert "openap" in result
            assert "airports_count" in result
            assert "version" in result
            assert "domains" in result
            assert result["status"] == "ok"
            assert result["version"] == "0.1.0"
            assert result["domains"]["atmosphere"] is True
            assert result["domains"]["orbits"] is False


class TestPlanFlightFunction:
    """Tests for the plan_flight() function."""

    @pytest.fixture
    def mock_openap(self, mock_openap_flight_generator, mock_openap_fuel_flow):
        """Setup OpenAP mocks."""
        with patch("aerospace_mcp.core.OPENAP_AVAILABLE", True):
            with patch(
                "aerospace_mcp.core.FlightGenerator",
                return_value=mock_openap_flight_generator,
            ):
                with patch(
                    "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
                ):
                    with patch(
                        "aerospace_mcp.core.prop.aircraft",
                        return_value={"limits": {"MTOW": 85000.0}},
                    ):
                        yield

    @pytest.mark.unit
    def test_plan_flight_success(self, mock_airports_iata, mock_openap):
        """Test successful flight planning."""
        payload = {
            "depart_city": "San Jose",
            "arrive_city": "Tokyo",
            "depart_country": "US",
            "arrive_country": "JP",
            "ac_type": "A359",
        }

        result = plan_flight(payload)

        assert "engine" in result
        assert "depart" in result
        assert "arrive" in result
        assert "distance_km" in result
        assert "distance_nm" in result
        assert "polyline" in result
        assert "estimates" in result
        assert result["depart"]["iata"] == "SJC"
        assert result["arrive"]["iata"] == "NRT"

    @pytest.mark.unit
    def test_plan_flight_same_city_error(self, mock_airports_iata):
        """Test error when departure and arrival are the same city."""
        payload = {
            "depart_city": "San Jose",
            "arrive_city": "San Jose",
            "ac_type": "A320",
        }

        with pytest.raises(ValueError, match="identical"):
            plan_flight(payload)

    @pytest.mark.unit
    def test_plan_flight_same_city_with_different_iata(
        self, mock_airports_iata, mock_openap
    ):
        """Test same city works if IATA codes are specified."""
        payload = {
            "depart_city": "San Jose",
            "arrive_city": "San Jose",
            "prefer_depart_iata": "SJC",
            "prefer_arrive_iata": "NRT",
            "ac_type": "A320",
        }

        result = plan_flight(payload)
        assert result["depart"]["iata"] == "SJC"
        assert result["arrive"]["iata"] == "NRT"

    @pytest.mark.unit
    def test_plan_flight_invalid_airport(self, mock_airports_iata):
        """Test error with invalid airport."""
        payload = {
            "depart_city": "NonexistentCity",
            "arrive_city": "Tokyo",
            "ac_type": "A320",
        }

        with pytest.raises(ValueError, match="no airport"):
            plan_flight(payload)

    @pytest.mark.unit
    def test_plan_flight_invalid_iata(self, mock_airports_iata):
        """Test error with invalid IATA."""
        payload = {
            "depart_city": "San Jose",
            "arrive_city": "Tokyo",
            "prefer_depart_iata": "INVALID",
            "ac_type": "A320",
        }

        with pytest.raises(ValueError, match="not found"):
            plan_flight(payload)

    @pytest.mark.unit
    def test_plan_flight_openap_unavailable(self, mock_airports_iata):
        """Test error when OpenAP is unavailable."""
        with patch("aerospace_mcp.core.OPENAP_AVAILABLE", False):
            payload = {
                "depart_city": "San Jose",
                "arrive_city": "Tokyo",
                "ac_type": "A320",
            }

            with pytest.raises(RuntimeError, match="unavailable"):
                plan_flight(payload)

    @pytest.mark.unit
    def test_plan_flight_unknown_backend(self, mock_airports_iata, mock_openap):
        """Test error with unknown backend (validation)."""
        # This should raise a validation error since backend is a Literal["openap"]
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PlanRequest(
                depart_city="San Jose",
                arrive_city="Tokyo",
                ac_type="A320",
                backend="unknown_backend",
            )


class TestCreateFlightPlan:
    """Tests for the create_flight_plan() function."""

    @pytest.fixture
    def mock_openap(self, mock_openap_flight_generator, mock_openap_fuel_flow):
        """Setup OpenAP mocks."""
        with patch("aerospace_mcp.core.OPENAP_AVAILABLE", True):
            with patch(
                "aerospace_mcp.core.FlightGenerator",
                return_value=mock_openap_flight_generator,
            ):
                with patch(
                    "aerospace_mcp.core.FuelFlow", return_value=mock_openap_fuel_flow
                ):
                    with patch(
                        "aerospace_mcp.core.prop.aircraft",
                        return_value={"limits": {"MTOW": 85000.0}},
                    ):
                        yield

    @pytest.mark.unit
    def test_create_flight_plan_success(self, mock_airports_iata, mock_openap):
        """Test successful flight plan creation."""
        req = PlanRequest(
            depart_city="San Jose",
            arrive_city="Tokyo",
            depart_country="US",
            arrive_country="JP",
            ac_type="A359",
        )

        result = create_flight_plan(req)

        assert isinstance(result, PlanResponse)
        assert result.depart.iata == "SJC"
        assert result.arrive.iata == "NRT"
        assert result.distance_km > 0
        assert len(result.polyline) > 0

    @pytest.mark.unit
    def test_create_flight_plan_same_city_error(self, mock_airports_iata):
        """Test error when departure and arrival are the same city."""
        req = PlanRequest(
            depart_city="San Jose",
            arrive_city="San Jose",
            ac_type="A320",
        )

        with pytest.raises(FlightPlanError, match="identical"):
            create_flight_plan(req)

    @pytest.mark.unit
    def test_create_flight_plan_airport_resolution_error(self, mock_airports_iata):
        """Test airport resolution error handling."""
        req = PlanRequest(
            depart_city="NonexistentCity",
            arrive_city="Tokyo",
            ac_type="A320",
        )

        with pytest.raises(FlightPlanError, match="no airport"):
            create_flight_plan(req)

    @pytest.mark.unit
    def test_create_flight_plan_openap_error(self, mock_airports_iata):
        """Test OpenAP error handling."""
        with patch("aerospace_mcp.core.OPENAP_AVAILABLE", False):
            req = PlanRequest(
                depart_city="San Jose",
                arrive_city="Tokyo",
                ac_type="A320",
            )

            with pytest.raises(FlightPlanError, match="unavailable"):
                create_flight_plan(req)


class TestAirportDataEdgeCases:
    """Tests for edge cases in airport data handling."""

    @pytest.mark.unit
    def test_airport_without_iata_key(self):
        """Test that airports without iata key are skipped."""
        test_data = {
            "SJC": {
                "iata": "SJC",
                "icao": "KSJC",
                "name": "San Jose International Airport",
                "city": "San Jose",
                "country": "US",
                "lat": 37.3626,
                "lon": -121.929,
            },
            "XXX": {
                # Missing iata key
                "icao": "XXXX",
                "name": "No IATA Airport",
                "city": "San Jose",
                "country": "US",
                "lat": 0.0,
                "lon": 0.0,
            },
            "YYY": {
                "iata": "",  # Empty iata
                "icao": "YYYY",
                "name": "Empty IATA Airport",
                "city": "San Jose",
                "country": "US",
                "lat": 0.0,
                "lon": 0.0,
            },
        }

        with patch("aerospace_mcp.core._AIRPORTS_IATA", test_data):
            results = _find_city_airports("San Jose", "US")
            # Should only find SJC, not the ones without valid IATA
            iatas = [a.iata for a in results]
            assert "SJC" in iatas
            # XXX and YYY should be filtered out
            assert "" not in iatas

    @pytest.mark.unit
    def test_airport_with_empty_iata_entry(self):
        """Test handling of empty IATA entry key."""
        test_data = {
            "": {  # Empty key
                "iata": "",
                "icao": "XXXX",
                "name": "Empty Key Airport",
                "city": "San Jose",
                "country": "US",
                "lat": 0.0,
                "lon": 0.0,
            },
            "SJC": {
                "iata": "SJC",
                "icao": "KSJC",
                "name": "San Jose International Airport",
                "city": "San Jose",
                "country": "US",
                "lat": 37.3626,
                "lon": -121.929,
            },
        }

        with patch("aerospace_mcp.core._AIRPORTS_IATA", test_data):
            results = _find_city_airports("San Jose", "US")
            # Should only find SJC
            assert len(results) == 1
            assert results[0].iata == "SJC"


class TestEstimatesOpenAPEdgeCases:
    """Additional tests for edge cases in estimates_openap."""

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_estimates_mass_fallback_no_mtow_in_limits(self):
        """Test mass fallback when aircraft has no MTOW in limits."""
        mock_gen = MagicMock()
        climb_data = pd.DataFrame(
            {
                "t": [30],
                "s": [10000],
                "altitude": [25000],
                "groundspeed": [300],
                "vertical_rate": [1500],
            }
        )
        cruise_data = pd.DataFrame(
            {
                "t": [30],
                "s": [15000],
                "altitude": [35000],
                "groundspeed": [450],
                "vertical_rate": [0],
            }
        )
        descent_data = pd.DataFrame(
            {
                "t": [30],
                "s": [12000],
                "altitude": [15000],
                "groundspeed": [350],
                "vertical_rate": [-1200],
            }
        )
        mock_gen.climb.return_value = climb_data
        mock_gen.cruise.return_value = cruise_data
        mock_gen.descent.return_value = descent_data

        mock_ff = MagicMock()
        mock_ff.enroute.return_value = 0.5

        # Aircraft with no MTOW in limits but has mtow field
        mock_props_no_limits = {"mtow": 75000.0}

        with patch("aerospace_mcp.core.FlightGenerator", return_value=mock_gen):
            with patch("aerospace_mcp.core.FuelFlow", return_value=mock_ff):
                with patch(
                    "aerospace_mcp.core.prop.aircraft",
                    return_value=mock_props_no_limits,
                ):
                    estimates, _ = estimates_openap("A320", 35000, None, 5000.0)
                    # Should use 85% of mtow = 63750
                    assert estimates["assumptions"]["mass_kg"] == 75000.0 * 0.85

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_estimates_mass_fallback_no_mtow_at_all(self):
        """Test mass fallback when aircraft has no MTOW anywhere."""
        mock_gen = MagicMock()
        climb_data = pd.DataFrame(
            {
                "t": [30],
                "s": [10000],
                "altitude": [25000],
                "groundspeed": [300],
                "vertical_rate": [1500],
            }
        )
        cruise_data = pd.DataFrame(
            {
                "t": [30],
                "s": [15000],
                "altitude": [35000],
                "groundspeed": [450],
                "vertical_rate": [0],
            }
        )
        descent_data = pd.DataFrame(
            {
                "t": [30],
                "s": [12000],
                "altitude": [15000],
                "groundspeed": [350],
                "vertical_rate": [-1200],
            }
        )
        mock_gen.climb.return_value = climb_data
        mock_gen.cruise.return_value = cruise_data
        mock_gen.descent.return_value = descent_data

        mock_ff = MagicMock()
        mock_ff.enroute.return_value = 0.5

        # Aircraft with no MTOW anywhere
        mock_props_no_mtow = {"limits": {}}

        with patch("aerospace_mcp.core.FlightGenerator", return_value=mock_gen):
            with patch("aerospace_mcp.core.FuelFlow", return_value=mock_ff):
                with patch(
                    "aerospace_mcp.core.prop.aircraft", return_value=mock_props_no_mtow
                ):
                    estimates, _ = estimates_openap("A320", 35000, None, 5000.0)
                    # Should use default 60000.0
                    assert estimates["assumptions"]["mass_kg"] == 60000.0

    @pytest.mark.unit
    @patch("aerospace_mcp.core.OPENAP_AVAILABLE", True)
    def test_estimates_fuel_flow_exception_handling(self):
        """Test fuel flow calculation exception handling."""
        mock_gen = MagicMock()
        climb_data = pd.DataFrame(
            {
                "t": [30],
                "s": [10000],
                "altitude": [25000],
                "groundspeed": [300],
                "vertical_rate": [1500],
            }
        )
        cruise_data = pd.DataFrame(
            {
                "t": [30],
                "s": [15000],
                "altitude": [35000],
                "groundspeed": [450],
                "vertical_rate": [0],
            }
        )
        descent_data = pd.DataFrame(
            {
                "t": [30],
                "s": [12000],
                "altitude": [15000],
                "groundspeed": [350],
                "vertical_rate": [-1200],
            }
        )
        mock_gen.climb.return_value = climb_data
        mock_gen.cruise.return_value = cruise_data
        mock_gen.descent.return_value = descent_data

        mock_ff = MagicMock()
        # Fuel flow raises exception
        mock_ff.enroute.side_effect = Exception("Fuel flow error")

        with patch("aerospace_mcp.core.FlightGenerator", return_value=mock_gen):
            with patch("aerospace_mcp.core.FuelFlow", return_value=mock_ff):
                with patch(
                    "aerospace_mcp.core.prop.aircraft",
                    return_value={"limits": {"MTOW": 85000.0}},
                ):
                    estimates, _ = estimates_openap("A320", 35000, None, 5000.0)
                    # Fuel should be 0 when exception occurs
                    assert estimates["climb"]["fuel_kg"] == 0.0
                    assert estimates["cruise"]["fuel_kg"] == 0.0
                    assert estimates["descent"]["fuel_kg"] == 0.0


class TestFlightPlanErrorException:
    """Tests for FlightPlanError exception class."""

    @pytest.mark.unit
    def test_flight_plan_error_creation(self):
        """Test FlightPlanError exception can be created."""
        error = FlightPlanError("Test error message")
        assert str(error) == "Test error message"

    @pytest.mark.unit
    def test_flight_plan_error_raised(self):
        """Test FlightPlanError can be raised and caught."""
        with pytest.raises(FlightPlanError) as exc_info:
            raise FlightPlanError("Test error")
        assert "Test error" in str(exc_info.value)


class TestOpenAPImportFailure:
    """Tests for OpenAP import failure handling."""

    @pytest.mark.unit
    def test_openap_unavailable_constant(self):
        """Test that OPENAP_AVAILABLE can be False."""
        with patch("aerospace_mcp.core.OPENAP_AVAILABLE", False):
            from aerospace_mcp import core

            with patch.object(core, "OPENAP_AVAILABLE", False):
                with pytest.raises(OpenAPError, match="unavailable"):
                    estimates_openap("A320", 35000, None, 1000.0)


class TestSearchAirportsByCity:
    """Tests for search_airports_by_city function."""

    @pytest.mark.unit
    def test_search_airports_by_city(self, mock_airports_iata):
        """Test search_airports_by_city function."""
        from aerospace_mcp.core import search_airports_by_city

        results = search_airports_by_city("San Jose", "US")
        assert len(results) == 1
        assert results[0].iata == "SJC"

    @pytest.mark.unit
    def test_search_airports_by_city_no_country(self, mock_airports_iata):
        """Test search_airports_by_city without country."""
        from aerospace_mcp.core import search_airports_by_city

        results = search_airports_by_city("Tokyo")
        assert len(results) >= 1


class TestPlanFlightGenericException:
    """Tests for generic exception handling in plan_flight."""

    @pytest.mark.unit
    def test_plan_flight_generic_exception(self, mock_airports_iata):
        """Test plan_flight handles generic exceptions."""
        payload = {
            "depart_city": "San Jose",
            "arrive_city": "Tokyo",
            "ac_type": "A320",
        }

        # Mock to raise a generic exception during flight planning
        with patch("aerospace_mcp.core.OPENAP_AVAILABLE", True):
            with patch(
                "aerospace_mcp.core.FlightGenerator",
                side_effect=Exception("Generic error"),
            ):
                with pytest.raises(RuntimeError, match="Flight planning failed"):
                    plan_flight(payload)


class TestCreateFlightPlanUnknownBackend:
    """Tests for unknown backend handling in create_flight_plan."""

    @pytest.mark.unit
    def test_create_flight_plan_backend_handling(self, mock_airports_iata):
        """Test create_flight_plan with backend handling."""
        from aerospace_mcp.core import FlightPlanError, PlanRequest, create_flight_plan

        # Create request with valid backend
        req = PlanRequest(
            depart_city="San Jose",
            arrive_city="Tokyo",
            ac_type="A320",
        )

        # Mock OpenAP to be unavailable to test backend error
        with patch("aerospace_mcp.core.OPENAP_AVAILABLE", False):
            with pytest.raises(FlightPlanError, match="unavailable"):
                create_flight_plan(req)
