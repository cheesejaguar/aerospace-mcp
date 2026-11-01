"""Tests for FastAPI health endpoint."""

import math

import airportsdata
import pytest

import aerospace_mcp.core as core


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @pytest.mark.unit
    def test_health_endpoint_with_openap(self, client):
        """OpenAP is installed in the test environment; the airport count is
        the size of the airportsdata IATA table."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "openap": True,
            "airports_count": len(airportsdata.load("IATA")),
        }

    @pytest.mark.unit
    def test_health_endpoint_without_openap(self, client, monkeypatch):
        """Health reports openap=false when the OpenAP package is missing."""
        monkeypatch.setattr(core, "OPENAP_AVAILABLE", False)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["openap"] is False

    @pytest.mark.unit
    def test_health_endpoint_response_format(self, client):
        """Test that health endpoint returns correct JSON format."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        data = response.json()
        assert set(data) == {"status", "openap", "airports_count"}
        assert isinstance(data["openap"], bool)
        # The real airport database has thousands of IATA entries
        assert data["airports_count"] > 1000


class TestAirportsByCity:
    """Tests for the /airports/by_city endpoint."""

    @pytest.mark.unit
    def test_airports_by_city_san_jose(self, client):
        """Test finding airports for San Jose."""
        response = client.get("/airports/by_city?city=San Jose&country=US")

        assert response.status_code == 200
        data = response.json()

        assert [airport["iata"] for airport in data] == ["SJC", "RHV"]

        airport = data[0]
        assert airport["icao"] == "KSJC"
        assert airport["city"] == "San Jose"
        assert airport["country"] == "US"
        assert airport["name"] == "Norman Y Mineta San Jose International Airport"

    @pytest.mark.unit
    def test_airports_by_city_without_country(self, client):
        """Test finding airports without specifying country."""
        response = client.get("/airports/by_city?city=San Jose")

        assert response.status_code == 200
        data = response.json()

        iatas = [airport["iata"] for airport in data]
        assert iatas[0] == "SJC"
        # Costa Rica's San Jose airports are included without a country
        assert {"SJO", "SYQ"} <= set(iatas)

    @pytest.mark.unit
    def test_airports_by_city_nonexistent(self, client):
        """Test finding airports for non-existent city."""
        response = client.get("/airports/by_city?city=NonexistentCity&country=US")

        assert response.status_code == 200
        data = response.json()

        assert data == []

    @pytest.mark.unit
    def test_airports_by_city_missing_parameter(self, client):
        """Test airports endpoint with missing city parameter."""
        response = client.get("/airports/by_city")

        # Should return 422 for missing required parameter
        assert response.status_code == 422

    @pytest.mark.unit
    def test_airports_by_city_empty_city(self, client):
        """Regression: empty city must not return the entire airport database."""
        response = client.get("/airports/by_city?city=")

        # min_length=1 rejects the empty string at validation time
        assert response.status_code == 422

    @pytest.mark.unit
    def test_airports_by_city_whitespace_city(self, client):
        """A whitespace-only city matches nothing (guarded in core)."""
        response = client.get("/airports/by_city?city=%20%20")

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.unit
    def test_airports_by_city_case_insensitive(self, client):
        """Test that city search is case insensitive."""
        responses = [
            client.get("/airports/by_city?city=san jose&country=US"),
            client.get("/airports/by_city?city=SAN JOSE&country=US"),
            client.get("/airports/by_city?city=San Jose&country=US"),
        ]

        # All responses should be successful and identical
        for response in responses:
            assert response.status_code == 200

        data_sets = [response.json() for response in responses]

        # All should return the same results
        assert all(data == data_sets[0] for data in data_sets)
        assert [a["iata"] for a in data_sets[0]] == ["SJC", "RHV"]


class TestPlanEndpoint:
    """Tests for the /plan endpoint (real airport data and OpenAP)."""

    @pytest.mark.unit
    def test_plan_endpoint_success(self, client):
        """Test successful flight planning."""
        request_data = {
            "depart_city": "San Jose",
            "arrive_city": "Tokyo",
            "depart_country": "US",
            "arrive_country": "JP",
            "prefer_arrive_iata": "NRT",
            "ac_type": "A359",
            "cruise_alt_ft": 35000,
            "route_step_km": 50.0,
        }

        response = client.post("/plan", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["engine"] == "openap"
        assert data["depart"]["iata"] == "SJC"
        assert data["arrive"]["iata"] == "NRT"

        assert 8200 < data["distance_km"] < 8400
        assert data["distance_nm"] == pytest.approx(data["distance_km"] / 1.852)

        # Polyline runs SJC -> NRT at the requested 50 km spacing
        polyline = data["polyline"]
        assert len(polyline) == math.ceil(data["distance_km"] / 50.0) + 1
        assert polyline[0] == pytest.approx(
            [data["depart"]["lat"], data["depart"]["lon"]]
        )
        assert polyline[-1] == pytest.approx(
            [data["arrive"]["lat"], data["arrive"]["lon"]]
        )

        estimates = data["estimates"]
        seg_km = sum(
            estimates[k]["distance_km"] for k in ("climb", "cruise", "descent")
        )
        assert seg_km == pytest.approx(data["distance_km"])
        assert estimates["block"]["fuel_kg"] > 0

    @pytest.mark.unit
    def test_plan_endpoint_same_city_error(self, client):
        """Test error when departure and arrival are the same city."""
        request_data = {
            "depart_city": "San Jose",
            "arrive_city": "San Jose",
            "ac_type": "A320",
        }

        response = client.post("/plan", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "identical" in data["detail"].lower()

    @pytest.mark.unit
    def test_plan_endpoint_invalid_airport(self, client):
        """Test error with invalid airport."""
        request_data = {
            "depart_city": "NonexistentCity",
            "arrive_city": "Tokyo",
            "ac_type": "A320",
        }

        response = client.post("/plan", json=request_data)

        assert response.status_code == 404
        data = response.json()
        assert "no airport" in data["detail"].lower()

    @pytest.mark.unit
    def test_plan_endpoint_preferred_iata(self, client):
        """Preferred IATA codes override unresolvable city names."""
        request_data = {
            "depart_city": "Any City",  # This would normally fail
            "arrive_city": "Other City",
            "prefer_depart_iata": "SJC",
            "prefer_arrive_iata": "NRT",
            "ac_type": "A359",
        }

        response = client.post("/plan", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["depart"]["iata"] == "SJC"
        assert data["arrive"]["iata"] == "NRT"

    @pytest.mark.unit
    def test_plan_endpoint_invalid_preferred_iata(self, client):
        """Test error with invalid preferred IATA."""
        request_data = {
            "depart_city": "San Jose",
            "arrive_city": "Tokyo",
            "prefer_depart_iata": "INVALID",
            "ac_type": "A320",
        }

        response = client.post("/plan", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.unit
    def test_plan_endpoint_unsupported_aircraft(self, client):
        """An aircraft OpenAP cannot model is a 400, not a 500."""
        response = client.post(
            "/plan",
            json={"depart_city": "San Jose", "arrive_city": "Tokyo", "ac_type": "C172"},
        )

        assert response.status_code == 400
        assert "not supported" in response.json()["detail"]

    @pytest.mark.unit
    def test_plan_endpoint_openap_unavailable(self, client, monkeypatch):
        """501 when the OpenAP package is not installed."""
        monkeypatch.setattr(core, "OPENAP_AVAILABLE", False)
        request_data = {
            "depart_city": "San Jose",
            "arrive_city": "Tokyo",
            "ac_type": "A320",
        }

        response = client.post("/plan", json=request_data)

        assert response.status_code == 501
        data = response.json()
        assert "unavailable" in data["detail"].lower()

    @pytest.mark.unit
    def test_plan_endpoint_validation_errors(self, client):
        """Test various validation errors."""
        # Missing required field
        response = client.post(
            "/plan",
            json={
                "depart_city": "San Jose",
                # Missing arrive_city and ac_type
            },
        )
        assert response.status_code == 422

        # Invalid altitude
        response = client.post(
            "/plan",
            json={
                "depart_city": "San Jose",
                "arrive_city": "Tokyo",
                "ac_type": "A320",
                "cruise_alt_ft": 100000,  # Too high
            },
        )
        assert response.status_code == 422

        # Invalid route step
        response = client.post(
            "/plan",
            json={
                "depart_city": "San Jose",
                "arrive_city": "Tokyo",
                "ac_type": "A320",
                "route_step_km": -10.0,  # Negative
            },
        )
        assert response.status_code == 422

    @pytest.mark.unit
    def test_plan_endpoint_custom_parameters(self, client):
        """Custom altitude, mass and step are honoured."""
        request_data = {
            "depart_city": "San Jose",
            "arrive_city": "Tokyo",
            "ac_type": "A320",
            "cruise_alt_ft": 41000,
            "mass_kg": 70000.0,
            "route_step_km": 100.0,
        }

        response = client.post("/plan", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assumptions = data["estimates"]["assumptions"]
        assert assumptions["cruise_alt_ft"] == 41000
        assert assumptions["mass_kg"] == 70000.0
        assert assumptions["mass_source"] == "user"
        assert len(data["polyline"]) == math.ceil(data["distance_km"] / 100.0) + 1
