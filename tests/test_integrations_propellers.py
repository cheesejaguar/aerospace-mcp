"""
Tests for propeller and UAV energy analysis integration module.
"""

import pytest
import math
from aerospace_mcp.integrations.propellers import (
    propeller_bemt_analysis,
    uav_energy_estimate,
    motor_propeller_matching,
    get_propeller_database,
    get_battery_database,
    PropellerGeometry,
    UAVConfiguration,
    BatteryConfiguration,
    PROPELLER_DATABASE,
    BATTERY_DATABASE,
    AEROSANDBOX_AVAILABLE
)


class TestPropellerGeometry:
    """Test propeller geometry model."""
    
    def test_propeller_geometry_creation(self):
        """Test creating propeller geometry objects."""
        geometry = PropellerGeometry(
            diameter_m=0.254,  # 10 inches
            pitch_m=0.178,     # 7 inches
            num_blades=2,
            activity_factor=100,
            cl_design=0.5,
            cd_design=0.02
        )
        
        assert geometry.diameter_m == 0.254
        assert geometry.pitch_m == 0.178
        assert geometry.num_blades == 2
        assert geometry.activity_factor == 100
        assert geometry.cl_design == 0.5
        assert geometry.cd_design == 0.02
    
    def test_propeller_geometry_validation(self):
        """Test propeller geometry validation."""
        # Test minimum constraints
        with pytest.raises(ValueError):
            PropellerGeometry(
                diameter_m=0.0,  # Too small
                pitch_m=0.1,
                num_blades=2
            )
        
        with pytest.raises(ValueError):
            PropellerGeometry(
                diameter_m=0.2,
                pitch_m=0.1,
                num_blades=1  # Too few blades
            )


class TestPropellerAnalysis:
    """Test propeller performance analysis."""
    
    def test_basic_propeller_analysis(self):
        """Test basic propeller analysis functionality."""
        geometry = PropellerGeometry(
            diameter_m=0.254,  # 10 inch prop
            pitch_m=0.178,     # 7 inch pitch
            num_blades=2
        )
        
        rpm_list = [1000, 2000, 3000, 4000, 5000]
        
        # Static thrust analysis
        results = propeller_bemt_analysis(geometry, rpm_list, velocity_ms=0.0)
        
        assert len(results) == len(rpm_list)
        
        # Check basic propeller physics
        for i, result in enumerate(results):
            assert result.rpm == rpm_list[i]
            assert result.thrust_n >= 0      # Thrust should be non-negative
            assert result.power_w >= 0       # Power should be non-negative
            assert result.torque_nm >= 0     # Torque should be non-negative
            assert 0 <= result.efficiency <= 1.0  # Efficiency should be between 0 and 1
        
        # Higher RPM should generally produce more thrust (for static case)
        for i in range(len(results) - 1):
            assert results[i+1].thrust_n >= results[i].thrust_n
            assert results[i+1].power_w >= results[i].power_w
    
    def test_forward_flight_analysis(self):
        """Test propeller analysis in forward flight."""
        geometry = PropellerGeometry(
            diameter_m=0.3048,  # 12 inch prop
            pitch_m=0.2032,     # 8 inch pitch
            num_blades=2
        )
        
        rpm_list = [2000, 3000, 4000]
        velocity_ms = 20.0  # 20 m/s forward speed
        
        results = propeller_bemt_analysis(geometry, rpm_list, velocity_ms)
        
        assert len(results) == len(rpm_list)
        
        # Check advance ratios are calculated
        for result in results:
            expected_J = velocity_ms / (result.rpm/60.0 * geometry.diameter_m)
            assert abs(result.advance_ratio - expected_J) < 0.01
        
        # In forward flight, efficiency should be higher than static
        static_results = propeller_bemt_analysis(geometry, rpm_list, 0.0)
        
        for forward, static in zip(results, static_results):
            if forward.rpm == static.rpm and forward.power_w > 10:  # Avoid division by small numbers
                assert forward.efficiency >= static.efficiency - 0.1  # Allow some tolerance
    
    def test_altitude_effects(self):
        """Test altitude effects on propeller performance."""
        geometry = PropellerGeometry(
            diameter_m=0.254,
            pitch_m=0.152,  # 6 inch pitch
            num_blades=3
        )
        
        rpm_list = [3000, 5000]
        
        # Compare sea level vs altitude performance
        sea_level_results = propeller_bemt_analysis(geometry, rpm_list, 0.0, altitude_m=0)
        altitude_results = propeller_bemt_analysis(geometry, rpm_list, 0.0, altitude_m=3000)
        
        # At altitude, air density is lower, so thrust should decrease
        for sl, alt in zip(sea_level_results, altitude_results):
            if sl.rpm == alt.rpm:
                assert alt.thrust_n < sl.thrust_n  # Less thrust at altitude
                assert alt.power_w < sl.power_w    # Less power at altitude (due to density)
    
    def test_propeller_coefficients(self):
        """Test propeller thrust and power coefficients."""
        geometry = PropellerGeometry(
            diameter_m=0.2032,  # 8 inch
            pitch_m=0.1016,     # 4 inch pitch
            num_blades=2
        )
        
        results = propeller_bemt_analysis(geometry, [4000], 0.0)
        result = results[0]
        
        # Check coefficient calculations
        rho = 1.225  # Sea level density
        n = result.rpm / 60.0
        D = geometry.diameter_m
        
        expected_CT = result.thrust_n / (rho * n**2 * D**4)
        expected_CP = result.power_w / (rho * n**3 * D**5)
        
        assert abs(result.thrust_coefficient - expected_CT) < 0.001
        assert abs(result.power_coefficient - expected_CP) < 0.001
        
        # Coefficients should be in reasonable ranges
        assert 0.01 < result.thrust_coefficient < 0.3
        assert 0.01 < result.power_coefficient < 0.3
    
    def test_different_propeller_sizes(self):
        """Test analysis with different propeller sizes."""
        # Small prop
        small_prop = PropellerGeometry(
            diameter_m=0.152,  # 6 inch
            pitch_m=0.102,     # 4 inch pitch
            num_blades=2
        )
        
        # Large prop
        large_prop = PropellerGeometry(
            diameter_m=0.381,  # 15 inch
            pitch_m=0.254,     # 10 inch pitch  
            num_blades=2
        )
        
        rpm = 3000
        small_results = propeller_bemt_analysis(small_prop, [rpm], 0.0)
        large_results = propeller_bemt_analysis(large_prop, [rpm], 0.0)
        
        small_result = small_results[0]
        large_result = large_results[0]
        
        # Large prop should produce more thrust at same RPM
        assert large_result.thrust_n > small_result.thrust_n
        # Large prop should require more power
        assert large_result.power_w > small_result.power_w
        # Large prop may have better efficiency
        assert large_result.efficiency >= small_result.efficiency - 0.1


class TestUAVConfiguration:
    """Test UAV configuration models."""
    
    def test_uav_configuration_creation(self):
        """Test creating UAV configuration objects."""
        config = UAVConfiguration(
            mass_kg=2.5,
            wing_area_m2=0.8,
            cd0=0.025,
            cl_cruise=0.8,
            num_motors=1,
            motor_efficiency=0.88,
            esc_efficiency=0.96
        )
        
        assert config.mass_kg == 2.5
        assert config.wing_area_m2 == 0.8
        assert config.cd0 == 0.025
        assert config.cl_cruise == 0.8
        assert config.num_motors == 1
        assert config.motor_efficiency == 0.88
        assert config.esc_efficiency == 0.96
    
    def test_battery_configuration_creation(self):
        """Test creating battery configuration objects."""
        battery = BatteryConfiguration(
            capacity_ah=5.0,
            voltage_nominal_v=14.8,
            mass_kg=0.8,
            energy_density_wh_kg=180,
            discharge_efficiency=0.95
        )
        
        assert battery.capacity_ah == 5.0
        assert battery.voltage_nominal_v == 14.8
        assert battery.mass_kg == 0.8
        assert battery.energy_density_wh_kg == 180
        assert battery.discharge_efficiency == 0.95


class TestUAVEnergyAnalysis:
    """Test UAV energy analysis."""
    
    def test_fixed_wing_energy_analysis(self):
        """Test energy analysis for fixed-wing UAV."""
        uav_config = UAVConfiguration(
            mass_kg=2.0,
            wing_area_m2=0.6,
            cd0=0.03,
            cl_cruise=0.7,
            num_motors=1,
            motor_efficiency=0.85,
            esc_efficiency=0.95
        )
        
        battery_config = BatteryConfiguration(
            capacity_ah=4.0,
            voltage_nominal_v=14.8,
            mass_kg=0.6
        )
        
        mission_profile = {
            "velocity_ms": 15.0,
            "altitude_m": 100.0
        }
        
        result = uav_energy_estimate(uav_config, battery_config, mission_profile)
        
        # Check basic results
        assert result.flight_time_min > 0
        assert result.range_km > 0  # Should have range for fixed-wing
        assert result.hover_time_min is None  # No hover time for fixed-wing
        assert result.power_required_w > 0
        assert result.energy_consumed_wh > 0
        assert result.battery_energy_wh > 0
        assert 0 < result.efficiency_overall <= 1.0
        
        # Check energy balance
        assert result.energy_consumed_wh <= result.battery_energy_wh
        
        # Flight time should be reasonable (10 minutes to 3 hours)
        assert 10 < result.flight_time_min < 180
        
        # Range should be reasonable for small UAV
        assert 5 < result.range_km < 100
    
    def test_multirotor_energy_analysis(self):
        """Test energy analysis for multirotor UAV."""
        uav_config = UAVConfiguration(
            mass_kg=1.5,
            disk_area_m2=0.2,  # Total rotor disk area
            cd0=0.05,  # Higher drag for multirotor
            num_motors=4,
            motor_efficiency=0.85,
            esc_efficiency=0.95
        )
        
        battery_config = BatteryConfiguration(
            capacity_ah=5.0,
            voltage_nominal_v=22.2,  # 6S battery
            mass_kg=0.7
        )
        
        mission_profile = {
            "velocity_ms": 8.0,  # Slower than fixed-wing
            "altitude_m": 50.0
        }
        
        result = uav_energy_estimate(uav_config, battery_config, mission_profile)
        
        # Check basic results
        assert result.flight_time_min > 0
        assert result.range_km is None  # No range calculation for multirotor
        assert result.hover_time_min > 0  # Should have hover time
        assert result.hover_time_min == result.flight_time_min  # Should be equal for multirotor
        
        # Multirotor typically has shorter flight times than fixed-wing
        assert 5 < result.flight_time_min < 60  # Typical multirotor endurance
    
    def test_energy_scaling_effects(self):
        """Test how energy consumption scales with aircraft parameters."""
        base_uav = UAVConfiguration(
            mass_kg=2.0,
            wing_area_m2=0.5,
            cd0=0.03,
            cl_cruise=0.8
        )
        
        base_battery = BatteryConfiguration(
            capacity_ah=3.0,
            voltage_nominal_v=11.1,
            mass_kg=0.4
        )
        
        mission = {"velocity_ms": 12.0}
        
        # Test mass scaling
        heavy_uav = UAVConfiguration(
            mass_kg=4.0,  # Double the mass
            wing_area_m2=0.5,
            cd0=0.03,
            cl_cruise=0.8
        )
        
        base_result = uav_energy_estimate(base_uav, base_battery, mission)
        heavy_result = uav_energy_estimate(heavy_uav, base_battery, mission)
        
        # Heavier aircraft should require more power
        assert heavy_result.power_required_w > base_result.power_required_w
        # And have shorter flight time with same battery
        assert heavy_result.flight_time_min < base_result.flight_time_min
    
    def test_battery_scaling_effects(self):
        """Test how battery capacity affects flight time."""
        uav_config = UAVConfiguration(
            mass_kg=1.8,
            wing_area_m2=0.45,
            cd0=0.028,
            cl_cruise=0.75
        )
        
        small_battery = BatteryConfiguration(
            capacity_ah=2.0,
            voltage_nominal_v=11.1,
            mass_kg=0.3
        )
        
        large_battery = BatteryConfiguration(
            capacity_ah=6.0,  # Triple capacity
            voltage_nominal_v=11.1,
            mass_kg=0.9
        )
        
        mission = {"velocity_ms": 14.0}
        
        small_result = uav_energy_estimate(uav_config, small_battery, mission)
        large_result = uav_energy_estimate(uav_config, large_battery, mission)
        
        # Larger battery should provide longer flight time
        assert large_result.flight_time_min > small_result.flight_time_min
        # Should be roughly proportional to capacity (accounting for increased mass)
        capacity_ratio = large_battery.capacity_ah / small_battery.capacity_ah
        time_ratio = large_result.flight_time_min / small_result.flight_time_min
        
        # Time ratio should be less than capacity ratio due to increased battery mass
        assert 1.5 < time_ratio < capacity_ratio
    
    def test_velocity_effects(self):
        """Test velocity effects on energy consumption."""
        uav_config = UAVConfiguration(
            mass_kg=2.2,
            wing_area_m2=0.55,
            cd0=0.032,
            cl_cruise=0.85
        )
        
        battery_config = BatteryConfiguration(
            capacity_ah=4.5,
            voltage_nominal_v=14.8,
            mass_kg=0.65
        )
        
        # Test different velocities
        slow_result = uav_energy_estimate(uav_config, battery_config, {"velocity_ms": 10.0})
        fast_result = uav_energy_estimate(uav_config, battery_config, {"velocity_ms": 20.0})
        
        # There should be an optimal speed; very slow and very fast both inefficient
        # At minimum, power should increase with velocity^3 due to drag
        assert fast_result.power_required_w > slow_result.power_required_w
    
    def test_altitude_effects_energy(self):
        """Test altitude effects on energy consumption."""
        uav_config = UAVConfiguration(
            mass_kg=1.9,
            wing_area_m2=0.48,
            cd0=0.03,
            cl_cruise=0.8
        )
        
        battery_config = BatteryConfiguration(
            capacity_ah=3.5,
            voltage_nominal_v=11.1,
            mass_kg=0.45
        )
        
        sea_level_result = uav_energy_estimate(uav_config, battery_config, 
                                              {"velocity_ms": 13.0, "altitude_m": 0})
        altitude_result = uav_energy_estimate(uav_config, battery_config,
                                            {"velocity_ms": 13.0, "altitude_m": 2000})
        
        # At altitude, air density is lower, which affects both lift and drag
        # Results should be different
        assert altitude_result.power_required_w != sea_level_result.power_required_w


class TestDatabases:
    """Test propeller and battery databases."""
    
    def test_propeller_database_access(self):
        """Test propeller database functionality.""" 
        database = get_propeller_database()
        
        assert isinstance(database, dict)
        assert len(database) > 0
        
        # Check that standard propellers are present
        assert any("APC" in name for name in database.keys())
        
        # Check data structure
        for prop_name, data in database.items():
            assert "diameter_m" in data
            assert "pitch_m" in data
            assert "num_blades" in data
            assert "efficiency_max" in data
            
            # Check reasonable values
            assert 0.05 < data["diameter_m"] < 1.0    # 2-40 inch range
            assert 0.02 < data["pitch_m"] < 0.5       # Reasonable pitch range
            assert 2 <= data["num_blades"] <= 6       # Typical blade counts
            assert 0.5 < data["efficiency_max"] < 0.9  # Realistic efficiency range
    
    def test_battery_database_access(self):
        """Test battery database functionality."""
        database = get_battery_database()
        
        assert isinstance(database, dict)
        assert len(database) > 0
        
        # Check that standard battery types are present
        assert any("LiPo" in name for name in database.keys())
        
        # Check data structure
        for battery_name, data in database.items():
            assert "nominal_voltage_v" in data
            assert "energy_density_wh_kg" in data
            assert "discharge_efficiency" in data
            
            # Check reasonable values
            assert 3.0 < data["nominal_voltage_v"] < 30.0    # Typical voltages
            assert 50 < data["energy_density_wh_kg"] < 300   # Realistic energy densities
            assert 0.8 < data["discharge_efficiency"] < 1.0  # Reasonable efficiency


class TestMotorPropellerMatching:
    """Test motor-propeller matching analysis."""
    
    def test_motor_propeller_matching_analysis(self):
        """Test motor-propeller matching functionality."""
        motor_kv = 1000  # 1000 RPM/V
        battery_voltage = 11.1  # 3S battery
        propeller_options = ["APC_10x7", "APC_12x8"]
        thrust_required = 15.0  # 15N thrust requirement
        
        results = motor_propeller_matching(
            motor_kv, battery_voltage, propeller_options, thrust_required
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check results for each propeller
        for prop_name, analysis in results.items():
            assert prop_name in propeller_options
            assert "geometry" in analysis
            assert "operating_point" in analysis
            assert "thrust_error_n" in analysis
            assert "efficiency" in analysis
            assert "power_required_w" in analysis
            assert "rpm_recommended" in analysis
            assert "suitable" in analysis
            
            # Check operating point data
            op = analysis["operating_point"]
            assert op["thrust_n"] >= 0
            assert op["power_w"] >= 0
            assert 0 <= op["efficiency"] <= 1.0
            
            # RPM should be reasonable for the motor/battery combination
            max_rpm = motor_kv * battery_voltage * 0.85
            assert op["rpm"] <= max_rpm
    
    def test_thrust_requirement_matching(self):
        """Test matching to specific thrust requirements."""
        motor_kv = 850
        battery_voltage = 14.8  # 4S
        props = ["APC_10x7", "APC_15x10"] 
        
        # Test with different thrust requirements
        low_thrust_results = motor_propeller_matching(motor_kv, battery_voltage, props, 5.0)
        high_thrust_results = motor_propeller_matching(motor_kv, battery_voltage, props, 25.0)
        
        # Should get recommendations for both cases
        assert len(low_thrust_results) > 0
        assert len(high_thrust_results) > 0
        
        # Larger prop (APC_15x10) should be more suitable for high thrust
        if "APC_15x10" in high_thrust_results:
            large_prop_analysis = high_thrust_results["APC_15x10"]
            assert large_prop_analysis["thrust_error_n"] >= 0  # Should be able to meet requirement


class TestIntegration:
    """Integration tests for propellers module."""
    
    def test_complete_uav_design_workflow(self):
        """Test complete UAV design and analysis workflow."""
        # Define UAV requirements
        target_endurance_min = 45  # Target 45 minute flight
        target_speed_ms = 12.0
        payload_mass_kg = 0.5
        
        # Design UAV configuration
        uav_config = UAVConfiguration(
            mass_kg=2.5,  # Including payload
            wing_area_m2=0.6,
            cd0=0.028,
            cl_cruise=0.8,
            num_motors=1,
            motor_efficiency=0.86,
            esc_efficiency=0.95
        )
        
        # Select battery
        battery_config = BatteryConfiguration(
            capacity_ah=5.2,
            voltage_nominal_v=14.8,
            mass_kg=0.8
        )
        
        # Analyze energy requirements
        mission = {"velocity_ms": target_speed_ms, "altitude_m": 100.0}
        energy_result = uav_energy_estimate(uav_config, battery_config, mission)
        
        # Should meet endurance requirement
        assert energy_result.flight_time_min > 30  # At least reasonable endurance
        
        # Select propeller
        motor_kv = 800
        battery_voltage = 14.0  # Under load voltage
        propeller_options = ["APC_12x8", "APC_15x10"]
        
        # Estimate required thrust (approximately equal to weight for level flight)
        thrust_required = uav_config.mass_kg * 9.81 * 0.1  # 10% of weight for climb capability
        
        prop_analysis = motor_propeller_matching(
            motor_kv, battery_voltage, propeller_options, thrust_required
        )
        
        # Should get viable propeller options
        assert len(prop_analysis) > 0
        suitable_props = [name for name, data in prop_analysis.items() if data["suitable"]]
        assert len(suitable_props) > 0
        
        # Verify complete system integration
        assert energy_result.battery_energy_wh > energy_result.energy_consumed_wh
        assert energy_result.efficiency_overall > 0.7  # Should be reasonably efficient
    
    @pytest.mark.skipif(not AEROSANDBOX_AVAILABLE, reason="AeroSandbox not available")
    def test_aerosandbox_propeller_integration(self):
        """Test AeroSandbox propeller integration if available."""
        geometry = PropellerGeometry(
            diameter_m=0.254,
            pitch_m=0.178,
            num_blades=2
        )
        
        results = propeller_bemt_analysis(geometry, [3000, 4000], 10.0)
        
        # Should get more accurate results from AeroSandbox
        assert len(results) == 2
        for result in results:
            assert isinstance(result.efficiency, float)
            assert 0 <= result.efficiency <= 1.0
    
    def test_error_handling_propellers(self):
        """Test error handling for invalid inputs."""
        # Test invalid propeller geometry
        with pytest.raises(ValueError):
            PropellerGeometry(
                diameter_m=0.0,  # Invalid
                pitch_m=0.1,
                num_blades=2
            )
        
        # Test invalid UAV configuration
        with pytest.raises(ValueError):
            UAVConfiguration(mass_kg=0.0)  # Invalid mass
        
        # Test invalid battery configuration
        with pytest.raises(ValueError):
            BatteryConfiguration(
                capacity_ah=0.0,  # Invalid
                voltage_nominal_v=12.0,
                mass_kg=1.0
            )
    
    def test_performance_characteristics_propellers(self):
        """Test that propeller calculations complete in reasonable time."""
        import time
        
        geometry = PropellerGeometry(
            diameter_m=0.3,
            pitch_m=0.2,
            num_blades=3
        )
        
        start_time = time.time()
        
        # Run multiple analyses
        static_results = propeller_bemt_analysis(geometry, list(range(1000, 6001, 500)), 0.0)
        forward_results = propeller_bemt_analysis(geometry, list(range(1000, 6001, 500)), 15.0)
        
        # UAV analysis
        uav_config = UAVConfiguration(mass_kg=2.0, wing_area_m2=0.5, cd0=0.03, cl_cruise=0.8)
        battery_config = BatteryConfiguration(capacity_ah=4.0, voltage_nominal_v=14.8, mass_kg=0.6)
        energy_result = uav_energy_estimate(uav_config, battery_config, {"velocity_ms": 12.0})
        
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 3.0
        
        # Should get reasonable number of results
        assert len(static_results) == 11
        assert len(forward_results) == 11
        assert energy_result.flight_time_min > 0


if __name__ == "__main__":
    pytest.main([__file__])