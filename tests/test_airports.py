"""Tests for airport resolution functionality (real airportsdata database)."""

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
    def test_valid_iata_code(self):
        """Test resolving a valid IATA code."""
        result = _airport_from_iata("SJC")
        assert result is not None
        assert result.iata == "SJC"
        assert result.icao == "KSJC"
        assert result.name == "Norman Y Mineta San Jose International Airport"
        assert result.city == "San Jose"
        assert result.country == "US"
        assert result.lat == pytest.approx(37.363, abs=0.01)
        assert result.lon == pytest.approx(-121.929, abs=0.01)
        assert result.tz == "America/Los_Angeles"

    @pytest.mark.unit
    def test_case_insensitive_iata(self):
        """Test that IATA codes are case insensitive."""
        result_lower = _airport_from_iata("sjc")
        result_upper = _airport_from_iata("SJC")
        result_mixed = _airport_from_iata("sJc")

        assert result_lower == result_upper == result_mixed
        assert result_upper is not None and result_upper.iata == "SJC"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "iata,expected_icao",
        [
            ("SJC", "KSJC"),
            ("NRT", "RJAA"),
            ("LHR", "EGLL"),
        ],
    )
    def test_known_airports(self, iata, expected_icao):
        """Known IATA codes resolve to the matching ICAO code."""
        result = _airport_from_iata(iata)
        assert result is not None
        assert result.iata == iata
        assert result.icao == expected_icao

    @pytest.mark.unit
    @pytest.mark.parametrize("iata", ["INVALID", "", "XYZ", "DEFINITELYNOTREAL123"])
    def test_unknown_iata_returns_none(self, iata):
        """Unknown or malformed IATA codes return None."""
        assert _airport_from_iata(iata) is None


class TestFindCityAirports:
    """Tests for _find_city_airports function."""

    @pytest.mark.unit
    def test_san_jose_us(self):
        """San Jose, US: SJC first, then the general-aviation field RHV."""
        results = _find_city_airports("San Jose", "US")
        assert [a.iata for a in results] == ["SJC", "RHV"]
        assert all(a.country == "US" for a in results)
        assert results[0].city == "San Jose"

    @pytest.mark.unit
    def test_case_insensitive_city_search(self):
        """Test that city search is case insensitive."""
        results_lower = _find_city_airports("san jose", "US")
        results_upper = _find_city_airports("SAN JOSE", "US")
        results_mixed = _find_city_airports("San Jose", "US")

        assert results_lower == results_upper == results_mixed

    @pytest.mark.unit
    def test_city_without_country_ranks_sjc_first(self):
        """Without a country every San Jose matches; SJC ranks first."""
        results = _find_city_airports("San Jose")
        iatas = [a.iata for a in results]
        assert iatas[0] == "SJC"
        assert {"SJC", "RHV", "SJO", "SYQ"} <= set(iatas)
        assert {a.country for a in results} >= {"US", "CR"}

    @pytest.mark.unit
    def test_international_airport_preference(self):
        """International airports rank above other airports of the same city."""
        us = _find_city_airports("San Jose", "US")
        assert "international" in us[0].name.lower()
        assert "international" not in us[-1].name.lower()

    @pytest.mark.unit
    def test_country_filtering(self):
        """Test that country filtering works correctly."""
        results_us = _find_city_airports("San Jose", "US")
        results_cr = _find_city_airports("San Jose", "CR")

        assert all(a.country == "US" for a in results_us)
        assert [a.iata for a in results_cr] == ["SJO", "SYQ"]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "city,country,expected_iatas",
        [
            ("San Jose", "US", ["SJC", "RHV"]),
            ("Tokyo", "JP", ["HND", "NRT"]),
            ("NonexistentCity", "US", []),
            ("DefinitelyNotARealCity123", "XX", []),
            ("", "US", []),
            ("   ", None, []),
        ],
    )
    def test_find_city_airports_parametrized(self, city, country, expected_iatas):
        """Parametrized test for city airport search."""
        results = _find_city_airports(city, country)
        assert [airport.iata for airport in results] == expected_iatas


class TestResolveEndpoint:
    """Tests for _resolve_endpoint function."""

    @pytest.mark.unit
    def test_resolve_with_preferred_iata(self):
        """Test resolving with a preferred IATA code."""
        result = _resolve_endpoint(
            city="Any City", country=None, prefer_iata="SJC", role="departure"
        )
        assert result.iata == "SJC"
        assert result.city == "San Jose"

    @pytest.mark.unit
    def test_resolve_with_invalid_preferred_iata(self):
        """Test resolving with an invalid preferred IATA code."""
        with pytest.raises(
            AirportResolutionError, match="departure: IATA 'INVALID' not found"
        ):
            _resolve_endpoint(
                city="Any City", country=None, prefer_iata="INVALID", role="departure"
            )

    @pytest.mark.unit
    def test_resolve_by_city(self):
        """Test resolving by city name picks the top-ranked airport."""
        result = _resolve_endpoint(
            city="San Jose", country="US", prefer_iata=None, role="departure"
        )
        assert result.iata == "SJC"

        tokyo = _resolve_endpoint(
            city="Tokyo", country="JP", prefer_iata=None, role="arrival"
        )
        assert tokyo.iata == "HND"

    @pytest.mark.unit
    def test_resolve_nonexistent_city(self):
        """Test resolving a non-existent city."""
        with pytest.raises(
            AirportResolutionError,
            match="departure: no airport for city='NonexistentCity'",
        ):
            _resolve_endpoint(
                city="NonexistentCity", country="US", prefer_iata=None, role="departure"
            )

    @pytest.mark.unit
    def test_resolve_with_role_in_error_message(self):
        """Test that the role is included in error messages."""
        with pytest.raises(
            AirportResolutionError, match="arrival: IATA 'INVALID' not found"
        ):
            _resolve_endpoint(
                city="Any City", country=None, prefer_iata="INVALID", role="arrival"
            )

    @pytest.mark.unit
    def test_resolve_prefers_iata_over_city(self):
        """Test that preferred IATA takes precedence over city."""
        result = _resolve_endpoint(
            city="Tokyo",  # Would normally resolve to HND
            country="JP",
            prefer_iata="SJC",
            role="departure",
        )
        assert result.iata == "SJC"
        assert result.city == "San Jose"
