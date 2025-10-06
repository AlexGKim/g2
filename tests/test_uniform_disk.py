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
from g2.core import Observation, inverse_noise
from scipy.special import j0, j1, jv

import matplotlib.pyplot as plt

class TestUniformDisk(unittest.TestCase):
    """Test cases for UniformDisk class."""
    
    def setUp(self):
        """Set up test fixtures for source."""
        self.spectral_exitance = 1e-10  # W/m²/Hz
        self.radius_arcsec = 1e-5
        self.radius_rad = self.radius_arcsec / 3600 * np.pi /180
        self.radius_m = 1. # m
        self.distance = 1/ self.radius_rad # m.

        self.disk = UniformDiskFixR(self.spectral_exitance, self.radius_m, self.distance)
        
        """Set up test fixtures for baseline"""
        self.nu_0 = 5e14  # 600 nm
        c = 2.99792458e8  # Speed of light in m/s
        self.lam = c / self.nu_0  # Wavelength in meters
        self.L_res = 1.22 * self.lam / (2 * self.radius_rad)

        ngrid = 2048*2
        wavelength_grid = np.array([self.lam*1e-10-0.5, self.lam*1e-10+0.5])
        flux_grid = np.zeros((2,ngrid,ngrid))
        pixel_scale_m = self.radius_m/128
        pixel_radius = (self.radius_m/pixel_scale_m)
        for i in range(ngrid//2 - np.ceil(pixel_radius).astype(int), ngrid//2 + np.ceil(pixel_radius).astype(int)):
            for j in range(ngrid//2 - np.ceil(pixel_radius).astype(int), ngrid//2 + np.ceil(pixel_radius).astype(int)):
                if (i-ngrid//2)**2 + (j-ngrid//2)**2 < pixel_radius**2:
                    flux_grid[:, i, j] = self.spectral_exitance * pixel_scale_m**2

        self.griddisk = GridSource(wavelength_grid, flux_grid, pixel_scale_m, self.distance)

        # Calculate inverse noise for a baseline measurement half the resolution limit
        baseline = np.array([self.L_res*.25, 0, 0.0])
        baseline2 = np.array([self.L_res*.4, -self.L_res*.3, 0.0])
        self.baselines = np.array([baseline, baseline*0.9, baseline2, baseline2*0.9 ])
    
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

        self.assertIsInstance(self.griddisk, GridSource)
        self.assertIsInstance(self.griddisk, ChaoticSource)
        self.assertIsInstance(self.griddisk, AbstractSource)

        # # Check surface brightness calculation
        # expected_brightness = self.flux_density / (np.pi * self.radius**2)
        # self.assertAlmostEqual(self.disk.surface_brightness, expected_brightness)
        
    def test_V(self):
        """Test V"""
        for baseline in self.baselines:
            u = baseline[0]
            v = baseline[1]
            rho = np.sqrt(u**2 + v**2)
            xi = np.pi * rho * (2*self.radius_rad) / self.lam
        
            if rho ==0:
                V=1.
            else:
                V=2 * j1(xi)/xi
            self.assertAlmostEqual(
                self.disk.V(self.nu_0, baseline), V, places=6)
            
            self.assertAlmostEqual(
                np.abs(self.griddisk.V(self.nu_0, baseline)), np.abs(V), delta=0.02)

        
    def test_SNR_s(self):
        """Test SNR_s"""
        for baseline in self.baselines:
            u = baseline[0]
            v = baseline[1]
            rho = np.sqrt(u**2 + v**2)
            xi = np.pi * rho * (2*self.radius_rad) / self.lam
        
            if rho ==0:
                SNR_s=0
            else:
                # J_0 - J_2 - 2J_1/r = -2 * J_2
                SNR_s= 2 * inverse_noise(self.disk, self.nu_0, self.obs) * 4 * np.abs(j1(xi)/xi * jv(2,xi))
            self.assertAlmostEqual(
                self.disk.SNR_s(self.nu_0, baseline, self.obs), SNR_s, places=6)       
               

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)