"""
Tests for atmosphere integration module.
"""

import pytest
import math
from aerospace_mcp.integrations.atmosphere import (
    get_atmosphere_profile,
    wind_model_simple,
    get_atmospheric_properties,
    _isa_manual,
    AMBIANCE_AVAILABLE
)


class TestAtmosphereProfile:
    """Test atmospheric profile calculations."""
    
    def test_sea_level_conditions(self):
        """Test standard sea level conditions."""
        profile = get_atmosphere_profile([0.0])
        point = profile[0]
        
        # ISA standard sea level conditions
        assert abs(point.pressure_pa - 101325.0) < 1.0
        assert abs(point.temperature_k - 288.15) < 0.1
        assert abs(point.density_kg_m3 - 1.225) < 0.001
        assert abs(point.speed_of_sound_mps - 340.3) < 0.5
    
    def test_multiple_altitudes(self):
        """Test profile at multiple altitudes."""
        altitudes = [0, 5000, 11000, 20000]
        profile = get_atmosphere_profile(altitudes)
        
        assert len(profile) == len(altitudes)
        
        # Check that pressure decreases with altitude
        for i in range(1, len(profile)):
            assert profile[i].pressure_pa < profile[i-1].pressure_pa
            assert profile[i].density_kg_m3 < profile[i-1].density_kg_m3
    
    def test_manual_isa_calculation(self):
        """Test manual ISA calculation against known values."""
        # Test troposphere (linear temperature gradient)
        pressure, temperature, density = _isa_manual(5000.0)
        
        # Expected values at 5000m
        expected_temp = 288.15 - 0.0065 * 5000  # ~255.65 K
        assert abs(temperature - expected_temp) < 0.1
        assert pressure < 101325.0  # Should be lower than sea level
        assert density < 1.225  # Should be lower than sea level
    
    def test_tropopause(self):
        """Test conditions at tropopause (11km)."""
        profile = get_atmosphere_profile([11000.0])
        point = profile[0]
        
        # At tropopause, temperature should be ~216.65K
        assert abs(point.temperature_k - 216.65) < 1.0
        assert point.pressure_pa < 30000.0  # Much lower pressure
    
    def test_altitude_limits(self):
        """Test altitude limits and validation."""
        # Test maximum altitude
        profile = get_atmosphere_profile([86000.0])
        assert len(profile) == 1
        
        # Test out of range
        with pytest.raises(ValueError, match="out of ISA range"):
            get_atmosphere_profile([100000.0])
        
        with pytest.raises(ValueError, match="out of ISA range"):
            get_atmosphere_profile([-1000.0])
    
    def test_invalid_model_type(self):
        """Test invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            get_atmosphere_profile([0.0], "INVALID")
    
    def test_atmospheric_properties_convenience(self):
        """Test convenience function for single altitude."""
        props = get_atmospheric_properties(5000.0)
        
        assert "altitude_m" in props
        assert "pressure_pa" in props 
        assert "temperature_k" in props
        assert "temperature_c" in props
        assert "density_kg_m3" in props
        assert "speed_of_sound_mps" in props
        
        # Check temperature conversion
        assert abs(props["temperature_c"] - (props["temperature_k"] - 273.15)) < 0.001


class TestWindModel:
    """Test wind profile models."""
    
    def test_logarithmic_wind_profile(self):
        """Test logarithmic wind model."""
        altitudes = [0, 10, 50, 100, 500]
        surface_wind = 10.0  # m/s
        
        wind_profile = wind_model_simple(
            altitudes, surface_wind, model="logarithmic"
        )
        
        assert len(wind_profile) == len(altitudes)
        
        # Wind should increase with height in logarithmic profile
        for i in range(1, len(wind_profile)):
            assert wind_profile[i].wind_speed_mps >= wind_profile[i-1].wind_speed_mps
        
        # At reference height (10m), should be close to surface wind
        ref_point = next(p for p in wind_profile if p.altitude_m == 10)
        assert abs(ref_point.wind_speed_mps - surface_wind) < 0.1
    
    def test_power_law_wind_profile(self):
        """Test power law wind model."""
        altitudes = [0, 10, 50, 100]
        surface_wind = 8.0
        
        wind_profile = wind_model_simple(
            altitudes, surface_wind, model="power"
        )
        
        # Wind should increase with height
        for i in range(1, len(wind_profile)):
            assert wind_profile[i].wind_speed_mps >= wind_profile[i-1].wind_speed_mps
    
    def test_below_ground(self):
        """Test wind calculation below surface."""
        wind_profile = wind_model_simple(
            [-10, 0, 10], 10.0, surface_altitude_m=0.0
        )
        
        # Below ground should be zero wind
        assert wind_profile[0].wind_speed_mps == 0.0
        # At surface should be interpolated
        assert wind_profile[1].wind_speed_mps >= 0.0
    
    def test_roughness_length_validation(self):
        """Test roughness length validation."""
        with pytest.raises(ValueError, match="Roughness length must be positive"):
            wind_model_simple([10, 50], 10.0, roughness_length_m=-0.1)
    
    def test_invalid_wind_model(self):
        """Test invalid wind model type."""
        with pytest.raises(ValueError, match="Unknown wind model"):
            wind_model_simple([10], 10.0, model="invalid")
    
    def test_surface_altitude_offset(self):
        """Test wind calculation with surface altitude offset."""
        altitudes = [100, 110, 150]  # Above 100m surface
        surface_alt = 100.0
        
        wind_profile = wind_model_simple(
            altitudes, 12.0, surface_altitude_m=surface_alt
        )
        
        # At surface altitude, should have minimal wind
        surface_point = wind_profile[0]
        assert surface_point.wind_speed_mps >= 0.0
        
        # Higher altitudes should have more wind
        assert wind_profile[-1].wind_speed_mps > wind_profile[0].wind_speed_mps
    
    def test_wind_profile_non_negative(self):
        """Test that wind speeds are always non-negative."""
        # Test with various parameters
        altitudes = [0, 5, 10, 20, 50, 100, 200]
        
        for surface_wind in [0.1, 5.0, 15.0, 30.0]:
            wind_profile = wind_model_simple(altitudes, surface_wind)
            
            for point in wind_profile:
                assert point.wind_speed_mps >= 0.0


class TestIntegration:
    """Integration tests combining atmosphere and wind models."""
    
    def test_atmosphere_wind_consistency(self):
        """Test that atmosphere and wind models work together."""
        altitudes = [0, 1000, 5000, 10000]
        
        # Get atmospheric profile
        atm_profile = get_atmosphere_profile(altitudes)
        
        # Get wind profile  
        wind_profile = wind_model_simple(altitudes, 10.0)
        
        assert len(atm_profile) == len(wind_profile)
        
        # Check altitude consistency
        for atm_point, wind_point in zip(atm_profile, wind_profile):
            assert atm_point.altitude_m == wind_point.altitude_m
    
    @pytest.mark.skipif(not AMBIANCE_AVAILABLE, reason="ambiance library not available")
    def test_ambiance_library_integration(self):
        """Test integration with ambiance library if available."""
        profile = get_atmosphere_profile([0, 5000, 11000])
        
        # Should have viscosity data if ambiance is available
        for point in profile:
            if point.viscosity_pa_s is not None:
                assert point.viscosity_pa_s > 0
    
    def test_consistency_across_models(self):
        """Test consistency of calculations across different conditions."""
        base_altitudes = [0, 2000, 5000, 8000, 11000]
        
        # Test multiple wind speeds
        for wind_speed in [5.0, 10.0, 20.0]:
            wind_profile = wind_model_simple(base_altitudes, wind_speed)
            
            # Verify monotonic increase (generally)
            for i in range(1, len(wind_profile)):
                # Allow for small variations due to numerical precision
                assert (wind_profile[i].wind_speed_mps >= 
                       wind_profile[i-1].wind_speed_mps - 0.001)


if __name__ == "__main__":
    pytest.main([__file__])