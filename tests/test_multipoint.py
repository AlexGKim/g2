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
from g2.models.sources.simple import MultiPoint
from g2.models.sources.grid_source import GridSource
from g2.core import Observation, inverse_noise
from scipy.special import j0, j1, jv

import matplotlib.pyplot as plt

class TestMultiPoint(unittest.TestCase):
    """Test cases for UniformDisk class."""
    
    def setUp(self):
        """Set up test fixtures for source."""
        self.spectral_exitance = 1e-10  # W/m²/Hz
        self.radius_arcsec = 1e-5
        self.radius_rad = self.radius_arcsec / 3600 * np.pi /180
        self.radius_m = 1. # m
        self.distance = 1/ self.radius_rad # m.

        self.spectral_exitances = np.array([self.spectral_exitance,self.spectral_exitance/3.45])
        self.positions = np.array([[0,0], [self.radius_rad,0]])
        self.multipoint = MultiPoint(self.spectral_exitances, self.positions,
                                     np.array([1.,1.]))
        
        """Set up test fixtures for baseline"""
        self.nu_0 = 5e14  # 600 nm
        c = 2.99792458e8  # Speed of light in m/s
        self.lam = c / self.nu_0  # Wavelength in meters

        bx_coords = np.linspace(-1.22 * self.lam/ self.radius_rad, -1.22 * self.lam/ self.radius_rad, 11)
        by_coords = np.linspace(-1.22 * self.lam/ self.radius_rad, -1.22 * self.lam/ self.radius_rad, 11)
        xx, yy = np.meshgrid(bx_coords, by_coords)
        self.baselines = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(len(bx_coords) * len(by_coords))])
    
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
        self.assertIsInstance(self.multipoint, MultiPoint)
        self.assertIsInstance(self.multipoint, ChaoticSource)
        self.assertIsInstance(self.multipoint, AbstractSource)


        
    # def test_V_squared_jacobian(self):
    #     """Test V_squared_jacobian"""


    #     for baseline in self.baselines:

    #         u = baseline[0]
    #         v = baseline[1]
    #         rho = np.sqrt(u**2 + v**2)
    #         xi = np.pi * rho * (2*self.radius_rad) / self.lam

    #         dVds = 2 * jv(2,xi)
    #         if rho ==0:
    #             V=1.
    #         else:
    #             V=2 * j1(xi)/xi

    #         V_squared_jacobian = 2 * V * dVds
    #         self.assertAlmostEqual(
    #             self.griddisk.V_squared_jacobian(self.nu_0, baseline, {'s': 1.})['s'] / V_squared_jacobian,
    #             1, delta=0.05)

    #         self.assertAlmostEqual(
    #             self.disk.V_squared_jacobian(self.nu_0, baseline)['s'],
    #             V_squared_jacobian, places=6)
            

    def test_V(self):
        """Test V"""
        dx, dy = self.positions[1]-self.positions[0]
        for baseline in self.baselines:
            u = baseline[0]/self.lam
            v = baseline[1]/self.lam
            V = self.spectral_exitances[0] + self.spectral_exitances[1] *np.exp(-2j * np.pi * (u * dx + v * dy))
            V = V / self.spectral_exitances.sum()

            V_multipoint = self.multipoint.V(self.nu_0, baseline)
            self.assertAlmostEqual(
                V_multipoint.real, V.real, places=6)
            self.assertAlmostEqual(
                V_multipoint.imag, V.imag, places=6)

        
    # def test_SNR_s(self):
    #     """Test SNR_s"""
    #     for baseline in self.baselines:
    #         u = baseline[0]
    #         v = baseline[1]
    #         rho = np.sqrt(u**2 + v**2)
    #         xi = np.pi * rho * (2*self.radius_rad) / self.lam
        
    #         if rho ==0:
    #             SNR_s=0
    #         else:
    #             # J_0 - J_2 - 2J_1/r = -2 * J_2
    #             SNR_s= 2 * inverse_noise(self.disk, self.nu_0, self.obs) * 4 * np.abs(j1(xi)/xi * jv(2,xi))
    #         self.assertAlmostEqual(
    #             self.disk.SNR_s(self.nu_0, baseline, self.obs), SNR_s, places=6)

    #         self.assertAlmostEqual(
    #             self.disk.SNR_s(self.nu_0, baseline, self.obs), SNR_s, places=6)
               

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)