"""Tests for FastAPI health endpoint."""

from unittest.mock import patch

import pytest


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @pytest.mark.unit
    def test_health_endpoint_with_openap(self, client):
        """Test health endpoint when OpenAP is available."""
        with patch("main.OPENAP_AVAILABLE", True):
            with patch(
                "main._AIRPORTS_IATA", {"SJC": {"iata": "SJC"}, "NRT": {"iata": "NRT"}}
            ):
                response = client.get("/health")

                assert response.status_code == 200
                data = response.json()

                assert "status" in data
                assert "openap" in data
                assert "airports_count" in data

                assert data["status"] == "ok"
                assert data["openap"] is True
                assert data["airports_count"] == 2

    @pytest.mark.unit
    def test_health_endpoint_without_openap(self, client):
        """Test health endpoint when OpenAP is not available."""
        with patch("main.OPENAP_AVAILABLE", False):
            with patch("main._AIRPORTS_IATA", {"SJC": {"iata": "SJC"}}):
                response = client.get("/health")

                assert response.status_code == 200
                data = response.json()

                assert data["status"] == "ok"
                assert data["openap"] is False
                assert data["airports_count"] == 1

    @pytest.mark.unit
    def test_health_endpoint_empty_airports(self, client):
        """Test health endpoint with no airports loaded."""
        with patch("main.OPENAP_AVAILABLE", True):
            with patch("main._AIRPORTS_IATA", {}):
                response = client.get("/health")

                assert response.status_code == 200
                data = response.json()

                assert data["status"] == "ok"
                assert data["openap"] is True
                assert data["airports_count"] == 0

    @pytest.mark.unit
    def test_health_endpoint_response_format(self, client):
        """Test that health endpoint returns correct JSON format."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        data = response.json()

        # Check required fields exist
        required_fields = ["status", "openap", "airports_count"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Check field types
        assert isinstance(data["status"], str)
        assert isinstance(data["openap"], bool)
        assert isinstance(data["airports_count"], int)

        # Check status value
        assert data["status"] == "ok"

        # Check airports_count is non-negative
        assert data["airports_count"] >= 0

    @pytest.mark.integration
    def test_health_endpoint_real_data(self, client):
        """Test health endpoint with real airport data."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ok"
        # Real airport data should have many airports
        assert data["airports_count"] > 1000
        # OpenAP availability depends on installation
        assert isinstance(data["openap"], bool)


class TestAirportsByCity:
    """Tests for the /airports/by_city endpoint."""

    @pytest.mark.unit
    def test_airports_by_city_san_jose(self, client, mock_airports_iata):
        """Test finding airports for San Jose."""
        response = client.get("/airports/by_city?city=San Jose&country=US")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 1

        airport = data[0]
        assert airport["iata"] == "SJC"
        assert airport["city"] == "San Jose"
        assert airport["country"] == "US"
        assert airport["name"] == "San Jose International Airport"

    @pytest.mark.unit
    def test_airports_by_city_without_country(self, client, mock_airports_iata):
        """Test finding airports without specifying country."""
        response = client.get("/airports/by_city?city=San Jose")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) >= 1

        # Should find at least SJC
        iatas = [airport["iata"] for airport in data]
        assert "SJC" in iatas

    @pytest.mark.unit
    def test_airports_by_city_nonexistent(self, client, mock_airports_iata):
        """Test finding airports for non-existent city."""
        response = client.get("/airports/by_city?city=NonexistentCity&country=US")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 0

    @pytest.mark.unit
    def test_airports_by_city_missing_parameter(self, client):
        """Test airports endpoint with missing city parameter."""
        response = client.get("/airports/by_city")

        # Should return 422 for missing required parameter
        assert response.status_code == 422

    @pytest.mark.unit
    def test_airports_by_city_case_insensitive(self, client, mock_airports_iata):
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
        assert len(data_sets[0]) == 1
        assert data_sets[0][0]["iata"] == "SJC"


class TestPlanEndpoint:
    """Tests for the /plan endpoint."""

    @pytest.mark.unit
    def test_plan_endpoint_success(
        self,
        client,
        mock_airports_iata,
        mock_openap_flight_generator,
        mock_openap_fuel_flow,
        mock_openap_props,
    ):
        """Test successful flight planning."""
        with patch("main.OPENAP_AVAILABLE", True):
            with patch(
                "main.FlightGenerator", return_value=mock_openap_flight_generator
            ):
                with patch("main.FuelFlow", return_value=mock_openap_fuel_flow):
                    with patch("main.prop.aircraft", return_value=mock_openap_props):
                        request_data = {
                            "depart_city": "San Jose",
                            "arrive_city": "Tokyo",
                            "depart_country": "US",
                            "arrive_country": "JP",
                            "ac_type": "A359",
                            "cruise_alt_ft": 35000,
                            "route_step_km": 50.0,
                        }

                        response = client.post("/plan", json=request_data)

                        assert response.status_code == 200
                        data = response.json()

                        # Check response structure
                        required_fields = [
                            "engine",
                            "depart",
                            "arrive",
                            "distance_km",
                            "distance_nm",
                            "polyline",
                            "estimates",
                        ]
                        for field in required_fields:
                            assert field in data

                        # Check airport data
                        assert data["depart"]["iata"] == "SJC"
                        assert data["arrive"]["iata"] == "NRT"

                        # Check distance reasonableness
                        assert 8500 < data["distance_km"] < 9500
                        assert 4500 < data["distance_nm"] < 5200  # km * NM_PER_KM

                        # Check polyline
                        assert isinstance(data["polyline"], list)
                        assert len(data["polyline"]) > 100  # Should have many points

                        # Check estimates structure
                        estimates = data["estimates"]
                        assert "block" in estimates
                        assert "climb" in estimates
                        assert "cruise" in estimates
                        assert "descent" in estimates

    @pytest.mark.unit
    def test_plan_endpoint_same_city_error(self, client, mock_airports_iata):
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
    def test_plan_endpoint_invalid_airport(self, client, mock_airports_iata):
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
    def test_plan_endpoint_preferred_iata(
        self,
        client,
        mock_airports_iata,
        mock_openap_flight_generator,
        mock_openap_fuel_flow,
        mock_openap_props,
    ):
        """Test planning with preferred IATA codes."""
        with patch("main.OPENAP_AVAILABLE", True):
            with patch(
                "main.FlightGenerator", return_value=mock_openap_flight_generator
            ):
                with patch("main.FuelFlow", return_value=mock_openap_fuel_flow):
                    with patch("main.prop.aircraft", return_value=mock_openap_props):
                        request_data = {
                            "depart_city": "Any City",  # This would normally fail
                            "arrive_city": "Any City",
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
    def test_plan_endpoint_invalid_preferred_iata(self, client, mock_airports_iata):
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
    def test_plan_endpoint_openap_unavailable(self, client, mock_airports_iata):
        """Test error when OpenAP is unavailable."""
        with patch("main.OPENAP_AVAILABLE", False):
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
    def test_plan_endpoint_custom_parameters(
        self,
        client,
        mock_airports_iata,
        mock_openap_flight_generator,
        mock_openap_fuel_flow,
        mock_openap_props,
    ):
        """Test planning with custom parameters."""
        with patch("main.OPENAP_AVAILABLE", True):
            with patch(
                "main.FlightGenerator", return_value=mock_openap_flight_generator
            ):
                with patch("main.FuelFlow", return_value=mock_openap_fuel_flow):
                    with patch("main.prop.aircraft", return_value=mock_openap_props):
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

                        # Check that custom parameters were used
                        assert (
                            data["estimates"]["assumptions"]["cruise_alt_ft"] == 41000
                        )
                        assert data["estimates"]["assumptions"]["mass_kg"] == 70000.0

                        # Polyline should have fewer points due to larger step size
                        assert (
                            len(data["polyline"]) < 200
                        )  # Fewer than default 25km step
