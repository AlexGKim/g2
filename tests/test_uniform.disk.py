"""
Unit tests for the source module.

This test suite validates the functionality of all source classes including:
- AbstractSource base class methods
- ChaoticSource temporal coherence functions
- PointSource analytical solutions
- UniformDisk analytical vs FFT comparison
"""

import unittest
import numpy as np
from g2.models.base import AbstractSource, ChaoticSource
from g2.models.sources.simple import UniformDiskFixR
from g2.models.sources.grid_source import GridSource
from g2.core import Observation
from scipy.special import j1, jv

class TestUniformDisk(unittest.TestCase):
    """Test cases for UniformDisk class."""
    
    def setUp(self):
        """Set up test fixtures for source."""
        self.spectral_exitance = 1e-26  # W/m²/Hz
        self.radius_arcsec = 1e-5
        self.radius_rad = self.radius_arcsec / 3600 * np.pi /180
        self.radius_m = 1. # m
        self.distance = 1/ self.radius_rad # m.

        self.disk = UniformDiskFixR(self.spectral_exitance, self.radius_m, self.distance)
        # self.griddisk = GridSource.getUniformDisk()
        
        """Set up test fixtures for baseline"""
        self.nu_0 = 5e14  # 600 nm
        c = 2.99792458e8  # Speed of light in m/s
        self.lam = c / self.nu_0  # Wavelength in meters
        self.L_res = 1.22 * self.lam / (2 * self.radius_rad)

        # Calculate inverse noise for a baseline measurement half the resolution limit
        baseline = np.array([self.L_res, 0, 0.0])
        baseline2 = np.array([self.L_res*.5, -self.L_res*.3, 0.0])
        self.baselines = np.array([baseline, baseline*1.1, baseline*0.9, baseline2, baseline2*0.9 ])
    
        """Set up test fixtures for observation"""
        # Observational parameters
        telescope_area = 1.0  # m²
        integration_time = 3600  # 1 hour
        detector_jitter = 1e-11  # 10 ps
        throughput = 1.

        self.obs = Observation(
            integration_time=integration_time,
            telescope_area=telescope_area,
            throughput=throughput,
            detector_jitter=detector_jitter
        )
        
    def test_initialization(self):
        """Test UniformDisk initialization."""
        self.assertIsInstance(self.disk, UniformDiskFixR)
        self.assertIsInstance(self.disk, ChaoticSource)
        self.assertIsInstance(self.disk, AbstractSource)

        self.assertEqual(self.disk.spectral_exitance, self.spectral_exitance)
        self.assertEqual(self.disk.radius_m, self.radius_m)
        self.assertEqual(self.disk.distance, self.distance)
        
        # # Check surface brightness calculation
        # expected_brightness = self.flux_density / (np.pi * self.radius**2)
        # self.assertAlmostEqual(self.disk.surface_brightness, expected_brightness)
        
    def test_V(self):
        """Test total flux is conserved."""
        for baseline in self.baselines:
            u = baseline[0]
            v = baseline[1]
            rho = np.sqrt(u**2 + v**2)
            xi = np.pi * rho * (2*self.radius_rad) / self.lam
        
            if rho ==0:
                V=1.
            else:
                V=2 * j1(xi)/xi
            print(xi, V)
            self.assertAlmostEqual(
                self.disk.V(self.nu_0, baseline), V, places=6)
        
    # def test_intensity_inside_disk(self):
    #     """Test intensity inside the disk."""
    #     # Point at center
    #     n_hat_center = np.array([0.0, 0.0])
    #     intensity_center = self.disk.intensity(self.nu_0, n_hat_center)
    #     self.assertAlmostEqual(intensity_center, self.disk.surface_brightness)
        
    #     # Point at edge
    #     n_hat_edge = np.array([self.radius, 0.0])
    #     intensity_edge = self.disk.intensity(self.nu_0, n_hat_edge)
    #     self.assertAlmostEqual(intensity_edge, self.disk.surface_brightness)
        
    # def test_intensity_outside_disk(self):
    #     """Test intensity outside the disk is zero."""
    #     n_hat_outside = np.array([2 * self.radius, 0.0])
    #     intensity_outside = self.disk.intensity(self.nu_0, n_hat_outside)
    #     self.assertEqual(intensity_outside, 0.0)
        
    # def test_V_zero_baseline(self):
    #     """Test V at zero baseline equals 1."""
    #     baseline_zero = np.array([0.0, 0.0, 0.0])
    #     result = self.disk.V(self.nu_0, baseline_zero)
    #     self.assertAlmostEqual(abs(result), 1.0, places=10)
        
    # def test_V_airy_function(self):
    #     """Test V implements correct Airy function in zeta units."""
    #     from scipy.special import j1
        
    #     baseline = np.array([100.0, 0.0, 0.0])
    #     baseline_length = np.linalg.norm(baseline[:2])
        
    #     # Calculate zeta parameter: ζ = πρθ/λ where θ is angular diameter
    #     theta = 2 * self.radius  # Angular diameter
    #     zeta = np.pi * baseline_length * theta / self.wavelength
        
    #     # Calculate expected Airy function value: V = 2J₁(ζ)/ζ
    #     if zeta == 0:
    #         expected = 1.0
    #     else:
    #         expected = 2 * j1(zeta) / zeta
            
    #     result = self.disk.V(self.nu_0, baseline)
    #     self.assertAlmostEqual(abs(result), abs(expected), places=10,
    #                           msg=f"ζ={zeta:.3f}: Expected={expected:.6f}, Got={abs(result):.6f}")
        
    # def test_V_first_zero(self):
    #     """Test V first zero occurs at correct zeta value."""
    #     # First zero of Airy function: 2J₁(ζ)/ζ = 0 when ζ = 3.8317...
    #     # For uniform disk: ζ = πρθ/λ where θ is angular diameter
        
    #     first_zero_zeta = 3.8317  # First zero of 2J₁(ζ)/ζ
    #     theta = 2 * self.radius  # Angular diameter
        
    #     # Calculate baseline length for first zero: ρ = ζλ/(πθ)
    #     baseline_first_zero = first_zero_zeta * self.wavelength / (np.pi * theta)
        
    #     baseline = np.array([baseline_first_zero, 0.0, 0.0])
    #     result = self.disk.V(self.nu_0, baseline)
        
    #     # Verify zeta calculation
    #     calculated_zeta = np.pi * baseline_first_zero * theta / self.wavelength
        
    #     # Should be very close to zero
    #     self.assertLess(abs(result), 0.01,
    #                    msg=f"ζ={calculated_zeta:.3f}: |V|={abs(result):.6f} should be ≈0")
    #     self.assertAlmostEqual(calculated_zeta, first_zero_zeta, places=3,
    #                           msg=f"Calculated ζ={calculated_zeta:.3f} should equal target ζ={first_zero_zeta:.3f}")
        
    # def test_V_symmetry(self):
    #     """Test V is symmetric for different baseline orientations."""
    #     baseline_length = 50.0
        
    #     baselines = [
    #         np.array([baseline_length, 0.0, 0.0]),  # E-W
    #         np.array([0.0, baseline_length, 0.0]),  # N-S
    #         np.array([baseline_length/np.sqrt(2), baseline_length/np.sqrt(2), 0.0])  # Diagonal
    #     ]
        
    #     results = []
    #     for baseline in baselines:
    #         result = self.disk.V(self.nu_0, baseline)
    #         results.append(abs(result))
            
    #     # All should be equal (symmetric disk)
    #     for i in range(1, len(results)):
    #         self.assertAlmostEqual(results[i], results[0], places=8)
            
    # def test_V_zeta_parameter_scaling(self):
    #     """Test visibility function scaling with zeta parameter."""
    #     from scipy.special import j1
        
    #     # Test multiple zeta values by varying baseline length
    #     zeta_targets = [0.5, 1.0, 2.0, 3.0, 3.8317, 5.0]  # Include first zero
    #     theta = 2 * self.radius  # Angular diameter
        
    #     for zeta_target in zeta_targets:
    #         with self.subTest(zeta=zeta_target):
    #             # Calculate baseline length for target zeta: ρ = ζλ/(πθ)
    #             baseline_length = zeta_target * self.wavelength / (np.pi * theta)
    #             baseline = np.array([baseline_length, 0.0, 0.0])
                
    #             # Get visibility result
    #             result = self.disk.V(self.nu_0, baseline)
                
    #             # Calculate expected Airy function value: V = 2J₁(ζ)/ζ
    #             if zeta_target == 0:
    #                 expected = 1.0
    #             else:
    #                 expected = 2 * j1(zeta_target) / zeta_target
                
    #             # Verify zeta calculation
    #             calculated_zeta = np.pi * baseline_length * theta / self.wavelength
                
    #             self.assertAlmostEqual(calculated_zeta, zeta_target, places=6,
    #                                   msg=f"Calculated ζ={calculated_zeta:.6f} should equal target ζ={zeta_target:.6f}")
    #             self.assertAlmostEqual(abs(result), abs(expected), places=8,
    #                                   msg=f"ζ={zeta_target:.3f}: |V|={abs(result):.6f}, Expected={abs(expected):.6f}")
                
    #             # Special check for first zero
    #             if abs(zeta_target - 3.8317) < 0.001:
    #                 self.assertLess(abs(result), 0.01,
    #                                msg=f"First zero at ζ={zeta_target:.3f}: |V|={abs(result):.6f} should be ≈0")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)