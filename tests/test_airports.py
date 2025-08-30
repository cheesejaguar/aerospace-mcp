"""Tests for airport resolution functionality."""

from unittest.mock import patch

import pytest

from aerospace_mcp.core import (
    AirportResolutionError,
    _airport_from_iata,
    _find_city_airports,
    _resolve_endpoint,
)


class TestAirportFromIata:
    """Tests for _airport_from_iata function."""

    @pytest.mark.unit
    def test_valid_iata_code(self, mock_airports_iata, sjc_airport):
        """Test resolving a valid IATA code."""
        result = _airport_from_iata("SJC")
        assert result is not None
        assert result.iata == "SJC"
        assert result.icao == "KSJC"
        assert result.name == "San Jose International Airport"
        assert result.city == "San Jose"
        assert result.country == "US"
        assert result.lat == 37.3626
        assert result.lon == -121.929
        assert result.tz == "America/Los_Angeles"

    @pytest.mark.unit
    def test_case_insensitive_iata(self, mock_airports_iata):
        """Test that IATA codes are case insensitive."""
        result_lower = _airport_from_iata("sjc")
        result_upper = _airport_from_iata("SJC")
        result_mixed = _airport_from_iata("sJc")

        assert result_lower is not None
        assert result_upper is not None
        assert result_mixed is not None
        assert result_lower.iata == result_upper.iata == result_mixed.iata == "SJC"

    @pytest.mark.unit
    def test_invalid_iata_code(self, mock_airports_iata):
        """Test resolving an invalid IATA code."""
        result = _airport_from_iata("INVALID")
        assert result is None

    @pytest.mark.unit
    def test_empty_iata_code(self, mock_airports_iata):
        """Test resolving an empty IATA code."""
        result = _airport_from_iata("")
        assert result is None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "iata,expected_iata,should_exist",
        [
            ("SJC", "SJC", True),
            ("NRT", "NRT", True),
            ("INVALID", None, False),
            ("", None, False),
            ("XYZ", None, False),
        ],
    )
    def test_airport_from_iata_parametrized(
        self, mock_airports_iata, iata, expected_iata, should_exist
    ):
        """Parametrized test for various IATA codes."""
        result = _airport_from_iata(iata)
        if should_exist:
            assert result is not None
            assert result.iata == expected_iata
        else:
            assert result is None


class TestFindCityAirports:
    """Tests for _find_city_airports function."""

    @pytest.mark.unit
    def test_san_jose_us(self, mock_airports_iata):
        """Test finding airports for San Jose, US."""
        results = _find_city_airports("San Jose", "US")
        assert len(results) == 1
        assert results[0].iata == "SJC"
        assert results[0].city == "San Jose"
        assert results[0].country == "US"

    @pytest.mark.unit
    def test_case_insensitive_city_search(self, mock_airports_iata):
        """Test that city search is case insensitive."""
        results_lower = _find_city_airports("san jose", "US")
        results_upper = _find_city_airports("SAN JOSE", "US")
        results_mixed = _find_city_airports("San Jose", "US")

        assert len(results_lower) == len(results_upper) == len(results_mixed) == 1
        assert results_lower[0].iata == results_upper[0].iata == results_mixed[0].iata

    @pytest.mark.unit
    def test_city_without_country(self, mock_airports_iata):
        """Test finding airports for a city without specifying country."""
        results = _find_city_airports("San Jose")
        assert len(results) >= 1
        assert any(airport.iata == "SJC" for airport in results)

    @pytest.mark.unit
    def test_international_airport_preference(self, mock_airports_iata):
        """Test that international airports are preferred in sorting."""
        # Add another airport for the same city to test sorting
        with patch(
            "aerospace_mcp.core._AIRPORTS_IATA",
            {
                **mock_airports_iata,
                "SJC": {
                    "iata": "SJC",
                    "icao": "KSJC",
                    "name": "San Jose International Airport",
                    "city": "San Jose",
                    "country": "US",
                    "lat": 37.3626,
                    "lon": -121.929,
                },
                "SJO": {
                    "iata": "SJO",
                    "icao": "MROC",
                    "name": "Juan Santamaría Airport",
                    "city": "San Jose",
                    "country": "CR",
                    "lat": 9.9936,
                    "lon": -84.2081,
                },
            },
        ):
            results = _find_city_airports("San Jose")
            # International airports should come first
            international_airports = [
                a for a in results if "international" in a.name.lower()
            ]
            if international_airports:
                assert results[0] in international_airports

    @pytest.mark.unit
    def test_nonexistent_city(self, mock_airports_iata):
        """Test searching for a non-existent city."""
        results = _find_city_airports("NonexistentCity", "US")
        assert len(results) == 0

    @pytest.mark.unit
    def test_empty_city_name(self, mock_airports_iata):
        """Test searching with empty city name."""
        results = _find_city_airports("", "US")
        assert len(results) == 0

    @pytest.mark.unit
    def test_country_filtering(self, mock_airports_iata):
        """Test that country filtering works correctly."""
        results_us = _find_city_airports("San Jose", "US")
        results_cr = _find_city_airports("San Jose", "CR")

        # Should only get US airports
        for airport in results_us:
            assert airport.country == "US"

        # Should only get CR airports (none in our test data)
        assert len(results_cr) == 0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "city,country,expected_iatas",
        [
            ("San Jose", "US", ["SJC"]),
            ("San Jose", None, ["SJC"]),
            ("Tokyo", "JP", ["NRT"]),
            ("NonexistentCity", "US", []),
            ("", "US", []),
        ],
    )
    def test_find_city_airports_parametrized(
        self, mock_airports_iata, city, country, expected_iatas
    ):
        """Parametrized test for city airport search."""
        results = _find_city_airports(city, country)
        result_iatas = [airport.iata for airport in results]
        assert result_iatas == expected_iatas


class TestResolveEndpoint:
    """Tests for _resolve_endpoint function."""

    @pytest.mark.unit
    def test_resolve_with_preferred_iata(self, mock_airports_iata):
        """Test resolving with a preferred IATA code."""
        result = _resolve_endpoint(
            city="Any City", country=None, prefer_iata="SJC", role="departure"
        )
        assert result.iata == "SJC"
        assert result.city == "San Jose"

    @pytest.mark.unit
    def test_resolve_with_invalid_preferred_iata(self, mock_airports_iata):
        """Test resolving with an invalid preferred IATA code."""
        with pytest.raises(
            AirportResolutionError, match="departure: IATA 'INVALID' not found"
        ):
            _resolve_endpoint(
                city="Any City", country=None, prefer_iata="INVALID", role="departure"
            )

    @pytest.mark.unit
    def test_resolve_by_city(self, mock_airports_iata):
        """Test resolving by city name."""
        result = _resolve_endpoint(
            city="San Jose", country="US", prefer_iata=None, role="departure"
        )
        assert result.iata == "SJC"
        assert result.city == "San Jose"

    @pytest.mark.unit
    def test_resolve_nonexistent_city(self, mock_airports_iata):
        """Test resolving a non-existent city."""
        with pytest.raises(
            AirportResolutionError,
            match="departure: no airport for city='NonexistentCity'",
        ):
            _resolve_endpoint(
                city="NonexistentCity", country="US", prefer_iata=None, role="departure"
            )

    @pytest.mark.unit
    def test_resolve_with_role_in_error_message(self, mock_airports_iata):
        """Test that the role is included in error messages."""
        with pytest.raises(
            AirportResolutionError, match="arrival: IATA 'INVALID' not found"
        ):
            _resolve_endpoint(
                city="Any City", country=None, prefer_iata="INVALID", role="arrival"
            )

    @pytest.mark.unit
    def test_resolve_prefers_iata_over_city(self, mock_airports_iata):
        """Test that preferred IATA takes precedence over city."""
        result = _resolve_endpoint(
            city="Tokyo",  # Would normally resolve to NRT
            country="JP",
            prefer_iata="SJC",  # But we prefer SJC
            role="departure",
        )
        assert result.iata == "SJC"
        assert result.city == "San Jose"  # Not Tokyo


class TestAirportResolutionIntegration:
    """Integration tests for airport resolution."""

    @pytest.mark.integration
    def test_real_airport_data_sjc(self):
        """Test with real airport data - SJC should exist."""
        result = _airport_from_iata("SJC")
        if result:  # Only test if real data is available
            assert result.iata == "SJC"
            assert result.country == "US"
            assert "San Jose" in result.name or "San Jose" in result.city

    @pytest.mark.integration
    def test_real_airport_data_city_search(self):
        """Test city search with real airport data."""
        results = _find_city_airports("San Jose", "US")
        if results:  # Only test if real data is available
            # Should find at least one airport
            assert len(results) > 0
            # At least one should be SJC
            iatas = [airport.iata for airport in results]
            assert "SJC" in iatas

    @pytest.mark.integration
    def test_error_handling_with_real_data(self):
        """Test error handling with real airport data."""
        # This should always fail regardless of real data
        result = _airport_from_iata("DEFINITELYNOTREAL123")
        assert result is None

        results = _find_city_airports("DefinitelyNotARealCity123", "XX")
        assert len(results) == 0
