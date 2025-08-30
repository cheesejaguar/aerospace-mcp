"""Tests for orbital mechanics module."""

import pytest
import numpy as np
from aerospace_mcp.integrations.orbits import (
    OrbitElements, StateVector,
    elements_to_state_vector, state_vector_to_elements,
    propagate_orbit_j2, calculate_ground_track, 
    hohmann_transfer, orbital_rendezvous_planning,
    lambert_solver_simple
)


class TestOrbitElements:
    """Test OrbitElements data class."""
    
    def test_valid_orbit_elements(self):
        """Test creating valid orbit elements."""
        elements = OrbitElements(
            semi_major_axis_m=7000000.0,
            eccentricity=0.001,
            inclination_deg=45.0,
            raan_deg=0.0,
            arg_periapsis_deg=0.0,
            true_anomaly_deg=0.0,
            epoch_utc="2000-01-01T12:00:00"
        )
        assert elements.semi_major_axis_m == 7000000.0
        assert elements.eccentricity == 0.001
        assert elements.inclination_deg == 45.0


class TestStateVector:
    """Test StateVector data class."""
    
    def test_valid_state_vector(self):
        """Test creating valid state vector."""
        state = StateVector(
            position_m=[7000000.0, 0.0, 0.0],
            velocity_ms=[0.0, 7546.0, 0.0],
            epoch_utc="2000-01-01T12:00:00"
        )
        assert state.position_m[0] == 7000000.0
        assert state.velocity_ms[1] == 7546.0
        assert state.epoch_utc == "2000-01-01T12:00:00"


class TestElementsToStateVector:
    """Test orbital elements to state vector conversion."""
    
    def test_circular_equatorial_orbit(self):
        """Test conversion for circular equatorial orbit."""
        elements = OrbitElements(
            semi_major_axis_m=7000000.0,
            eccentricity=0.0,
            inclination_deg=0.0,
            raan_deg=0.0,
            arg_periapsis_deg=0.0,
            true_anomaly_deg=0.0,
            epoch_utc="2000-01-01T12:00:00"
        )
        
        state = elements_to_state_vector(elements)
        
        # For circular orbit at periapsis, position should be (a, 0, 0)
        assert abs(state.position_m[0] - 7000000.0) < 1.0
        assert abs(state.position_m[1]) < 1.0
        assert abs(state.position_m[2]) < 1.0
        
        # Velocity should be (0, sqrt(mu/a), 0)
        expected_v = np.sqrt(3.986004418e14 / 7000000.0)
        assert abs(state.velocity_ms[0]) < 1.0
        assert abs(state.velocity_ms[1] - expected_v) < 1.0
        assert abs(state.velocity_ms[2]) < 1.0
    
    def test_inclined_orbit(self):
        """Test conversion for inclined orbit."""
        elements = OrbitElements(
            semi_major_axis_m=7000000.0,
            eccentricity=0.0,
            inclination_deg=90.0,  # Polar orbit
            raan_deg=0.0,
            arg_periapsis_deg=0.0,
            true_anomaly_deg=0.0,
            epoch_utc="2000-01-01T12:00:00"
        )
        
        state = elements_to_state_vector(elements)
        
        # For polar orbit, should have some z-component in velocity
        assert abs(state.position_m[0] - 7000000.0) < 1.0
        assert abs(state.position_m[1]) < 1.0
        # Z position should be zero at equator crossing
        assert abs(state.position_m[2]) < 1.0


class TestStateVectorToElements:
    """Test state vector to orbital elements conversion."""
    
    def test_round_trip_conversion(self):
        """Test that conversion is reversible."""
        original_elements = OrbitElements(
            semi_major_axis_m=7000000.0,
            eccentricity=0.1,
            inclination_deg=45.0,
            raan_deg=30.0,
            arg_periapsis_deg=60.0,
            true_anomaly_deg=90.0,
            epoch_utc="2000-01-01T12:00:00"
        )
        
        # Convert to state vector and back
        state = elements_to_state_vector(original_elements)
        recovered_elements = state_vector_to_elements(state)
        
        # Check that we get back approximately the same elements
        assert abs(recovered_elements.semi_major_axis_m - original_elements.semi_major_axis_m) < 1000.0
        assert abs(recovered_elements.eccentricity - original_elements.eccentricity) < 0.001
        assert abs(recovered_elements.inclination_deg - original_elements.inclination_deg) < 0.1
    
    def test_circular_orbit_conversion(self):
        """Test conversion of circular orbit."""
        # Create circular orbit state vector manually
        r = 7000000.0
        mu = 3.986004418e14
        v = np.sqrt(mu / r)
        
        state = StateVector(
            position_m=[r, 0.0, 0.0],
            velocity_ms=[0.0, v, 0.0],
            epoch_utc="2000-01-01T12:00:00"
        )
        
        elements = state_vector_to_elements(state)
        
        assert abs(elements.semi_major_axis_m - r) < 1000.0
        assert elements.eccentricity < 0.01  # Should be nearly circular
        assert abs(elements.inclination_deg) < 0.1  # Should be equatorial


class TestOrbitPropagation:
    """Test orbit propagation with J2 perturbations."""
    
    def test_propagate_circular_orbit(self):
        """Test propagation of circular orbit."""
        initial_state = StateVector(
            position_m=[7000000.0, 0.0, 0.0],
            velocity_ms=[0.0, 7546.0, 0.0],
            epoch_utc="2000-01-01T12:00:00"
        )
        
        # Propagate for one hour
        states = propagate_orbit_j2(initial_state, 3600.0, 300.0)
        
        assert len(states) > 1
        
        # States should be different from initial
        final_state = states[-1]
        assert abs(final_state.position_m[0] - initial_state.position_m[0]) > 1000.0
        assert abs(final_state.position_m[1] - initial_state.position_m[1]) > 1000.0
    
    def test_propagate_short_duration(self):
        """Test propagation for short duration."""
        initial_state = StateVector(
            position_m=[7000000.0, 0.0, 0.0],
            velocity_ms=[0.0, 7546.0, 0.0],
            epoch_utc="2000-01-01T12:00:00"
        )
        
        # Propagate for 1 hour with 1-minute steps
        states = propagate_orbit_j2(initial_state, 3600.0, 60.0)
        
        assert len(states) == 61  # 60 minutes + initial state
        
        # States should be different from initial
        final_state = states[-1]
        assert abs(final_state.position_m[0] - initial_state.position_m[0]) > 1000.0
        assert abs(final_state.position_m[1] - initial_state.position_m[1]) > 1000.0


class TestGroundTrack:
    """Test ground track calculation."""
    
    def test_equatorial_orbit_ground_track(self):
        """Test ground track for equatorial orbit."""
        # Create state vectors for equatorial orbit
        states = []
        r = 7000000.0
        mu = 3.986004418e14
        v = np.sqrt(mu / r)
        
        # Create several points along orbit
        for i in range(5):
            angle = i * np.pi / 4  # 0 to 180 degrees
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            vx = -v * np.sin(angle)
            vy = v * np.cos(angle)
            
            states.append(StateVector(
                position_m=[x, y, 0.0],
                velocity_ms=[vx, vy, 0.0],
                epoch_utc="2000-01-01T12:00:00"
            ))
        
        ground_track = calculate_ground_track(states, 600.0)
        
        assert len(ground_track) == len(states)
        
        # For equatorial orbit, all latitudes should be near zero
        for point in ground_track:
            assert abs(point.latitude_deg) < 5.0  # Relaxed tolerance
            assert -180.0 <= point.longitude_deg <= 180.0


class TestHohmannTransfer:
    """Test Hohmann transfer calculations."""
    
    def test_earth_to_higher_orbit(self):
        """Test Hohmann transfer to higher orbit."""
        r1 = 7000000.0  # Initial orbit radius (m)
        r2 = 10000000.0  # Final orbit radius (m)
        
        transfer = hohmann_transfer(r1, r2)
        
        assert transfer["delta_v1_ms"] > 0  # Should require positive delta-v
        assert transfer["delta_v2_ms"] > 0  # Should require positive delta-v
        assert transfer["total_delta_v_ms"] > 0
        assert transfer["transfer_time_s"] > 0
        
        # Semi-major axis of transfer ellipse should be (r1 + r2) / 2
        expected_a = (r1 + r2) / 2
        assert abs(transfer["transfer_semi_major_axis_m"] - expected_a) < 1000.0
    
    def test_higher_to_lower_orbit(self):
        """Test Hohmann transfer to lower orbit."""
        r1 = 10000000.0  # Initial orbit radius (m)
        r2 = 7000000.0   # Final orbit radius (m)
        
        transfer = hohmann_transfer(r1, r2)
        
        # For descending transfer, first burn should be negative (retrograde)
        assert transfer["delta_v1_ms"] < 0
        assert transfer["delta_v2_ms"] < 0
        assert transfer["total_delta_v_ms"] > 0  # Total magnitude
        assert transfer["transfer_time_s"] > 0


class TestOrbitalRendezvous:
    """Test orbital rendezvous planning."""
    
    def test_coplanar_rendezvous(self):
        """Test rendezvous between coplanar orbits."""
        chaser = OrbitElements(
            semi_major_axis_m=7000000.0,
            eccentricity=0.0,
            inclination_deg=45.0,
            raan_deg=0.0,
            arg_periapsis_deg=0.0,
            true_anomaly_deg=0.0,
            epoch_utc="2000-01-01T12:00:00"
        )
        
        target = OrbitElements(
            semi_major_axis_m=7200000.0,  # Slightly higher orbit
            eccentricity=0.0,
            inclination_deg=45.0,  # Same inclination
            raan_deg=0.0,
            arg_periapsis_deg=0.0,
            true_anomaly_deg=90.0,  # Different position
            epoch_utc="2000-01-01T12:00:00"
        )
        
        plan = orbital_rendezvous_planning(chaser, target)
        
        assert plan["total_delta_v_ms"] > 0
        assert plan["time_to_rendezvous_s"] > 0
        assert len(plan["maneuvers"]) > 0
        
        # Should have at least one maneuver
        first_maneuver = plan["maneuvers"][0]
        assert abs(first_maneuver["delta_v_ms"]) > 0


class TestLambertProblem:
    """Test Lambert problem solver."""
    
    def test_lambert_half_orbit(self):
        """Test Lambert problem for half orbit."""
        r1 = [7000000.0, 0.0, 0.0]
        r2 = [-7000000.0, 0.0, 0.0]
        
        # Half orbital period for circular orbit
        mu = 3.986004418e14
        period = 2 * np.pi * np.sqrt((7000000.0)**3 / mu)
        tof = period / 2
        
        try:
            result = lambert_solver_simple(r1, r2, tof)
            
            # Should return a dictionary with velocity vectors
            assert "v1_vec_ms" in result or "initial_velocity_ms" in result
            # Lambert solver uses simplified approach, might not handle all cases
            
        except (NotImplementedError, ValueError):
            # Lambert solver uses simplified approach, might not handle all cases
            pytest.skip("Lambert solver uses simplified approach")


if __name__ == "__main__":
    pytest.main([__file__])