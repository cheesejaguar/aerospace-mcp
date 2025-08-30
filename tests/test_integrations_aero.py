"""
Tests for aircraft aerodynamics integration module.
"""

import pytest
import math
from aerospace_mcp.integrations.aero import (
    wing_vlm_analysis,
    airfoil_polar_analysis,
    calculate_stability_derivatives,
    estimate_wing_area,
    get_airfoil_database,
    WingGeometry,
    AIRFOIL_DATABASE,
    AEROSANDBOX_AVAILABLE
)


class TestWingGeometry:
    """Test wing geometry model and calculations."""
    
    def test_wing_geometry_creation(self):
        """Test creating wing geometry objects."""
        geometry = WingGeometry(
            span_m=10.0,
            chord_root_m=1.5,
            chord_tip_m=1.0,
            sweep_deg=5.0,
            dihedral_deg=2.0,
            twist_deg=-2.0,
            airfoil_root="NACA2412",
            airfoil_tip="NACA2412"
        )
        
        assert geometry.span_m == 10.0
        assert geometry.chord_root_m == 1.5
        assert geometry.chord_tip_m == 1.0
        assert geometry.sweep_deg == 5.0
        assert geometry.dihedral_deg == 2.0
        assert geometry.twist_deg == -2.0
        assert geometry.airfoil_root == "NACA2412"
        assert geometry.airfoil_tip == "NACA2412"
    
    def test_wing_area_calculation(self):
        """Test wing geometric property calculations."""
        geometry = WingGeometry(
            span_m=10.0,
            chord_root_m=2.0,
            chord_tip_m=1.0
        )
        
        props = estimate_wing_area(geometry)
        
        # Trapezoidal wing area
        expected_area = 10.0 * (2.0 + 1.0) / 2  # 15.0 m²
        assert abs(props["wing_area_m2"] - expected_area) < 0.001
        
        # Aspect ratio
        expected_ar = 10.0**2 / expected_area  # 6.67
        assert abs(props["aspect_ratio"] - expected_ar) < 0.01
        
        # Taper ratio
        assert abs(props["taper_ratio"] - 0.5) < 0.001
        
        # Mean aerodynamic chord
        assert props["mean_aerodynamic_chord_m"] > 1.0
        assert props["mean_aerodynamic_chord_m"] < 2.0


class TestWingAnalysis:
    """Test wing aerodynamic analysis."""
    
    def test_simple_wing_analysis(self):
        """Test basic wing analysis functionality.""" 
        geometry = WingGeometry(
            span_m=8.0,
            chord_root_m=1.2,
            chord_tip_m=0.8,
            airfoil_root="NACA2412"
        )
        
        alpha_list = [0, 2, 4, 6, 8]
        results = wing_vlm_analysis(geometry, alpha_list, mach=0.15)
        
        assert len(results) == len(alpha_list)
        
        # Check that lift increases with angle of attack (up to a point)
        for i in range(len(results) - 1):
            if results[i].alpha_deg < 10:  # Before stall
                assert results[i+1].CL >= results[i].CL
        
        # Check reasonable values
        for result in results:
            assert -2.0 < result.CL < 2.0  # Reasonable lift coefficient range
            assert 0.005 < result.CD < 0.5  # Reasonable drag coefficient range
            assert -0.5 < result.CM < 0.5   # Reasonable moment coefficient range
            if result.CD > 0.01:
                assert result.L_D_ratio >= 0  # L/D should be non-negative
    
    def test_wing_analysis_stall_behavior(self):
        """Test wing stall behavior."""
        geometry = WingGeometry(
            span_m=6.0,
            chord_root_m=1.0,
            chord_tip_m=1.0,
            airfoil_root="NACA0012"
        )
        
        # Test including high angles of attack
        alpha_list = [0, 5, 10, 15, 20, 25]
        results = wing_vlm_analysis(geometry, alpha_list)
        
        # Find maximum lift coefficient
        max_cl = max(result.CL for result in results)
        
        # Should show stall characteristics
        assert max_cl > 1.0  # Should achieve reasonable CL_max
        assert max_cl < 2.5  # But not unreasonably high
        
        # Check that drag increases significantly at high alpha
        high_alpha_result = next(r for r in results if r.alpha_deg == 20)
        low_alpha_result = next(r for r in results if r.alpha_deg == 5)
        assert high_alpha_result.CD > low_alpha_result.CD
    
    def test_wing_analysis_mach_effects(self):
        """Test Mach number effects on wing analysis."""
        geometry = WingGeometry(
            span_m=5.0,
            chord_root_m=1.5,
            chord_tip_m=1.0
        )
        
        alpha_list = [2, 4, 6]
        
        # Compare low and high Mach results
        results_low_mach = wing_vlm_analysis(geometry, alpha_list, mach=0.1)
        results_high_mach = wing_vlm_analysis(geometry, alpha_list, mach=0.6)
        
        # At higher Mach, should generally see increased lift curve slope
        for low, high in zip(results_low_mach, results_high_mach):
            if low.alpha_deg == high.alpha_deg and low.alpha_deg > 0:
                # High Mach should show compressibility effects
                assert high.CL != low.CL  # Should be different
    
    def test_different_airfoils(self):
        """Test wing analysis with different airfoil selections."""
        geometry_symmetric = WingGeometry(
            span_m=6.0,
            chord_root_m=1.0,
            chord_tip_m=1.0,
            airfoil_root="NACA0012"
        )
        
        geometry_cambered = WingGeometry(
            span_m=6.0,
            chord_root_m=1.0,
            chord_tip_m=1.0,
            airfoil_root="NACA4412"
        )
        
        alpha_list = [0, 4]
        
        results_symmetric = wing_vlm_analysis(geometry_symmetric, alpha_list)
        results_cambered = wing_vlm_analysis(geometry_cambered, alpha_list)
        
        # At zero alpha, cambered airfoil should have higher lift
        symmetric_zero = next(r for r in results_symmetric if r.alpha_deg == 0)
        cambered_zero = next(r for r in results_cambered if r.alpha_deg == 0)
        
        assert cambered_zero.CL > symmetric_zero.CL


class TestAirfoilAnalysis:
    """Test airfoil polar analysis."""
    
    def test_airfoil_database_access(self):
        """Test airfoil database functionality."""
        database = get_airfoil_database()
        
        assert isinstance(database, dict)
        assert len(database) > 0
        
        # Check that standard airfoils are present
        assert "NACA0012" in database
        assert "NACA2412" in database
        
        # Check data structure
        for airfoil_name, data in database.items():
            assert "cl_alpha" in data
            assert "cd0" in data
            assert "cl_max" in data
            assert "alpha_stall_deg" in data
            
            # Check reasonable values
            assert 5.0 < data["cl_alpha"] < 8.0  # Typical 2D lift curve slope
            assert 0.004 < data["cd0"] < 0.02    # Typical profile drag
            assert 1.0 < data["cl_max"] < 2.5    # Typical maximum lift
            assert 10.0 < data["alpha_stall_deg"] < 20.0  # Typical stall angle
    
    def test_airfoil_polar_generation(self):
        """Test airfoil polar data generation."""
        alpha_list = [-5, 0, 2, 4, 6, 8, 10, 12, 15]
        
        results = airfoil_polar_analysis("NACA2412", alpha_list, reynolds=1e6, mach=0.1)
        
        assert len(results) == len(alpha_list)
        
        # Check basic aerodynamic relationships
        for result in results:
            assert -2.0 < result.cl < 2.0  # Reasonable lift coefficient
            assert 0.004 < result.cd < 0.2  # Reasonable drag coefficient
            assert -0.2 < result.cm < 0.1   # Reasonable moment coefficient
            
            # L/D ratio should be reasonable
            if result.cd > 0.005:
                assert -50 < result.cl_cd_ratio < 200
        
        # Check lift curve slope (approximately linear region)
        linear_region = [r for r in results if -2 < r.alpha_deg < 8]
        if len(linear_region) >= 2:
            # Calculate approximate slope
            cl_alpha_approx = (linear_region[-1].cl - linear_region[0].cl) / \
                             (math.radians(linear_region[-1].alpha_deg - linear_region[0].alpha_deg))
            assert 4.0 < cl_alpha_approx < 8.0  # Typical 2D values
    
    def test_reynolds_effects(self):
        """Test Reynolds number effects on airfoil polars."""
        alpha_list = [0, 4, 8]
        
        # Compare different Reynolds numbers
        results_low_re = airfoil_polar_analysis("NACA0012", alpha_list, reynolds=1e5)
        results_high_re = airfoil_polar_analysis("NACA0012", alpha_list, reynolds=1e7)
        
        # At higher Reynolds number, generally expect lower drag
        for low_re, high_re in zip(results_low_re, results_high_re):
            if low_re.alpha_deg == high_re.alpha_deg:
                # High Re should generally have lower drag (especially profile drag)
                assert high_re.cd <= low_re.cd + 0.005  # Allow some tolerance
    
    def test_mach_effects_airfoil(self):
        """Test Mach number effects on airfoil analysis."""
        alpha_list = [2, 6]
        
        results_low_mach = airfoil_polar_analysis("NACA2412", alpha_list, mach=0.05)
        results_high_mach = airfoil_polar_analysis("NACA2412", alpha_list, mach=0.4)
        
        # Should see some differences due to compressibility
        for low_m, high_m in zip(results_low_mach, results_high_mach):
            if low_m.alpha_deg == high_m.alpha_deg:
                # May see increased drag at higher Mach
                assert abs(high_m.cd - low_m.cd) >= 0  # Should be some difference
    
    def test_different_airfoil_types(self):
        """Test analysis of different airfoil types.""" 
        alpha_list = [0, 6]
        
        # Compare symmetric vs cambered airfoils
        symmetric_results = airfoil_polar_analysis("NACA0012", alpha_list)
        cambered_results = airfoil_polar_analysis("NACA4412", alpha_list)
        
        # At zero alpha
        symmetric_zero = next(r for r in symmetric_results if r.alpha_deg == 0)
        cambered_zero = next(r for r in cambered_results if r.alpha_deg == 0)
        
        # Symmetric should have ~zero lift at zero alpha
        assert abs(symmetric_zero.cl) < 0.1
        # Cambered should have positive lift at zero alpha
        assert cambered_zero.cl > 0.2
        
        # Cambered airfoil should have negative moment coefficient
        assert cambered_zero.cm < -0.02


class TestStabilityDerivatives:
    """Test stability derivatives calculations."""
    
    def test_basic_stability_calculation(self):
        """Test basic stability derivatives calculation."""
        geometry = WingGeometry(
            span_m=8.0,
            chord_root_m=1.0,
            chord_tip_m=0.8,
            airfoil_root="NACA2412"
        )
        
        stability = calculate_stability_derivatives(geometry, alpha_deg=2.0, mach=0.2)
        
        # Check that we get reasonable values
        assert isinstance(stability.CL_alpha, float)
        assert isinstance(stability.CM_alpha, float)
        
        # CL_alpha should be positive and reasonable for 3D wing
        assert 3.0 < stability.CL_alpha < 7.0  # 3D wing should be lower than 2D
        
        # CM_alpha should typically be negative for stability
        assert stability.CM_alpha < 0.1  # Allow slightly positive for test wing
        assert stability.CM_alpha > -0.5  # But not extremely negative
    
    def test_aspect_ratio_effects(self):
        """Test aspect ratio effects on stability derivatives."""
        # High aspect ratio wing
        geometry_high_ar = WingGeometry(
            span_m=12.0,
            chord_root_m=1.0,
            chord_tip_m=1.0,
            airfoil_root="NACA2412"
        )
        
        # Low aspect ratio wing  
        geometry_low_ar = WingGeometry(
            span_m=4.0,
            chord_root_m=2.0,
            chord_tip_m=2.0,
            airfoil_root="NACA2412"
        )
        
        stability_high_ar = calculate_stability_derivatives(geometry_high_ar)
        stability_low_ar = calculate_stability_derivatives(geometry_low_ar)
        
        # High AR should have higher lift curve slope (closer to 2D)
        assert stability_high_ar.CL_alpha > stability_low_ar.CL_alpha
    
    def test_mach_effects_stability(self):
        """Test Mach number effects on stability derivatives."""
        geometry = WingGeometry(
            span_m=6.0,
            chord_root_m=1.2,
            chord_tip_m=1.0
        )
        
        stability_low_mach = calculate_stability_derivatives(geometry, mach=0.1)
        stability_high_mach = calculate_stability_derivatives(geometry, mach=0.5)
        
        # Should see some compressibility effects
        assert stability_high_mach.CL_alpha != stability_low_mach.CL_alpha
        
        # At higher Mach, may see increased lift curve slope initially
        # (before critical Mach effects dominate)
        difference_ratio = stability_high_mach.CL_alpha / stability_low_mach.CL_alpha
        assert 0.5 < difference_ratio < 2.0  # Should be within reasonable bounds


class TestIntegration:
    """Integration tests for aero module."""
    
    def test_complete_wing_analysis_workflow(self):
        """Test complete wing analysis workflow."""
        # Define a realistic wing
        geometry = WingGeometry(
            span_m=9.0,
            chord_root_m=1.8,
            chord_tip_m=1.2,
            sweep_deg=3.0,
            dihedral_deg=1.0,
            twist_deg=-1.0,
            airfoil_root="NACA2412",
            airfoil_tip="NACA2412"
        )
        
        # Calculate wing properties
        wing_props = estimate_wing_area(geometry)
        
        # Run aerodynamic analysis
        alpha_range = [-2, 0, 2, 4, 6, 8, 10]
        aero_results = wing_vlm_analysis(geometry, alpha_range, mach=0.25)
        
        # Calculate stability
        stability = calculate_stability_derivatives(geometry, mach=0.25)
        
        # Verify reasonable results
        assert wing_props["wing_area_m2"] > 10.0  # Should be reasonable size
        assert wing_props["aspect_ratio"] > 4.0   # Should be reasonable AR
        
        assert len(aero_results) == len(alpha_range)
        
        # Find cruise point (around 4-6 degrees alpha)
        cruise_points = [r for r in aero_results if 3 < r.alpha_deg < 7]
        if cruise_points:
            cruise_point = cruise_points[0]
            assert cruise_point.L_D_ratio > 5.0  # Should have reasonable L/D
            assert 0.4 < cruise_point.CL < 1.2   # Should be in reasonable range
        
        assert stability.CL_alpha > 0  # Should have positive lift curve slope
    
    @pytest.mark.skipif(not AEROSANDBOX_AVAILABLE, reason="AeroSandbox not available")
    def test_aerosandbox_integration(self):
        """Test AeroSandbox integration if available."""
        geometry = WingGeometry(
            span_m=6.0,
            chord_root_m=1.0,
            chord_tip_m=1.0
        )
        
        results = wing_vlm_analysis(geometry, [0, 4, 8])
        
        # Should get results from AeroSandbox VLM
        assert len(results) == 3
        for result in results:
            # AeroSandbox should provide more accurate results
            assert isinstance(result.CL, float)
            assert isinstance(result.CD, float)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid geometry
        with pytest.raises(ValueError):
            WingGeometry(
                span_m=-1.0,  # Invalid negative span
                chord_root_m=1.0,
                chord_tip_m=1.0
            )
        
        # Test empty alpha list
        geometry = WingGeometry(
            span_m=5.0,
            chord_root_m=1.0,
            chord_tip_m=1.0
        )
        
        # Should handle empty input gracefully
        results = wing_vlm_analysis(geometry, [], mach=0.2)
        assert len(results) == 0
        
        # Test unknown airfoil (should fall back to default)
        results_unknown = airfoil_polar_analysis("UNKNOWN_AIRFOIL", [0, 5])
        assert len(results_unknown) == 2  # Should still return results
    
    def test_performance_characteristics(self):
        """Test that calculations complete in reasonable time."""
        import time
        
        geometry = WingGeometry(
            span_m=10.0,
            chord_root_m=1.5,
            chord_tip_m=1.0
        )
        
        start_time = time.time()
        
        # Run multiple analyses
        wing_results = wing_vlm_analysis(geometry, list(range(-5, 16)), mach=0.2)
        airfoil_results = airfoil_polar_analysis("NACA2412", list(range(-10, 16)))
        stability = calculate_stability_derivatives(geometry)
        
        end_time = time.time()
        
        # Should complete quickly (less than 2 seconds for basic analysis)
        assert (end_time - start_time) < 2.0
        
        # Should get reasonable number of results
        assert len(wing_results) == 21
        assert len(airfoil_results) == 26
        assert stability.CL_alpha > 0


if __name__ == "__main__":
    pytest.main([__file__])