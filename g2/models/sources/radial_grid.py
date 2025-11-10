"""
Radial Grid Source Model
========================

This module implements a radial coordinate-based source model using the polar DFT algorithms
from II.ipynb. The RadialGrid class uses intensity data as a function of wavelength and 
impact parameter (radial coordinate) to calculate visibility functions.

Key Features:
- Polar DFT algorithms for visibility calculations
- Radial coordinate system (impact parameter)
- Direct implementation of II.ipynb algorithms (dft_polar, dgamma2ds)
- Support for SN2011fe HDF5 data format
"""

import numpy as np
from typing import Union, Dict, Any
import os
from pathlib import Path
import pandas as pd

from ..base import source
import jax.numpy as jnp
import jax
from jax.numpy.fft import fftshift, fftfreq
from scipy.special import jv


class RadialGrid(source.ChaoticSource):
    """
    Radial coordinate-based source model using polar DFT algorithms.
    
    This class implements a source model using radial coordinates (impact parameter)
    and applies the polar DFT algorithms from II.ipynb for efficient visibility
    calculations.
    
    The intensity data is provided as I_nu_p(lambda, p) where:
    - lambda: wavelength grid
    - p: impact parameter (radial coordinate)
    """
    
    def __init__(self, lambdas: np.ndarray, I_nu_p: np.ndarray, p_rays: np.ndarray,
                 distance: float = 204379200000000.0, phi_B: float = 0.0):
        """
        Initialize RadialGrid with wavelength, intensity, and radial coordinate data.
        
        Parameters
        ----------
        lambdas : np.ndarray
            Wavelength grid in Angstrom, shape (n_wavelengths,)
        I_nu_p : np.ndarray
            Intensity data, shape (n_wavelengths, n_radial_points)
        p_rays : np.ndarray
            Impact parameter coordinates in meters, shape (n_radial_points,)
        distance : float
            Distance to source in meters
        phi_B : float
            Baseline orientation angle in radians
        """
        c = 2.99792458e8  # m/s
        
        # Store input data
        self.lambdas = np.array(lambdas)  # [Angstrom]
        self.I_nu_p = np.array(I_nu_p)    # [intensity units from data]
        self.p_rays = np.array(p_rays)    # In meters
        
        # Calculate frequency grid
        self.frequency_grid = c / (self.lambdas * 1e-10)  # [Hz]
        
        # Store parameters
        self.distance = distance
        self.phi_B = phi_B
        
        # Get dimensions
        self.n_wavelengths, self.n_radial_points = self.I_nu_p.shape
        
        # Validate input dimensions
        if len(self.lambdas) != self.n_wavelengths:
            raise ValueError(f"Wavelength grid length {len(self.lambdas)} doesn't match intensity wavelength dimension {self.n_wavelengths}")
        if len(self.p_rays) != self.n_radial_points:
            raise ValueError(f"Radial coordinate length {len(self.p_rays)} doesn't match intensity radial dimension {self.n_radial_points}")
        
        # Store frequency range for reference
        self.freq_min = np.min(self.frequency_grid)
        self.freq_max = np.max(self.frequency_grid)

    def dft_polar(self, y: np.ndarray, norder: int = None) -> np.ndarray:
        """
        Polar DFT algorithm from II.ipynb.
        
        Computes the polar discrete Fourier transform using Bessel functions.
        This is the core algorithm for calculating gamma (visibility function).
        
        Parameters
        ----------
        y : np.ndarray
            Input radial intensity profile
        norder : int, optional
            Number of output points. If None, uses len(y)
            
        Returns
        -------
        np.ndarray
            Polar DFT result (gamma values)
        """
        ny = len(y)
        
        if norder is None:
            norder = ny
            
        rhos = np.arange(norder) / ny
        ans = np.zeros(norder)
        theta = np.arange(ny)
        
        for i, rho in enumerate(rhos):
            integrand = y * jv(0, 2 * np.pi * rho * theta) * theta
            ans[i] = np.trapz(integrand)
            
        return 2 * np.pi * ans

    def dft_polar_der(self, y: np.ndarray, norder: int = None) -> np.ndarray:
        """
        Derivative of polar DFT from II.ipynb.
        
        Computes the derivative of the polar DFT for use in Jacobian calculations.
        
        Parameters
        ----------
        y : np.ndarray
            Input radial intensity profile
        norder : int, optional
            Number of output points. If None, uses len(y)
            
        Returns
        -------
        np.ndarray
            Derivative of polar DFT
        """
        ny = len(y)
        
        if norder is None:
            norder = ny
            
        rhos = np.arange(norder) / ny
        ans = np.zeros(norder)
        theta = np.arange(ny)
        
        for i, rho in enumerate(rhos):
            integrand = y * jv(1, 2 * np.pi * rho * theta) * theta**2
            ans[i] = np.trapz(integrand)
            
        return -(2 * np.pi)**2 * ans

    def dgamma2ds(self, y: np.ndarray, norder: int = None) -> np.ndarray:
        """
        Calculate derivative of |gamma|^2 with respect to size parameter s.
        
        This implements the dgamma2ds algorithm from II.ipynb for calculating
        the Jacobian of |V|^2 with respect to the size parameter.
        
        Parameters
        ----------
        y : np.ndarray
            Input radial intensity profile
        norder : int, optional
            Number of output points. If None, uses len(y)
            
        Returns
        -------
        np.ndarray
            Derivative of |gamma|^2 with respect to size parameter
        """
        ny = len(y)
        
        if norder is None:
            norder = ny
            
        rhos = np.arange(norder) / ny
        
        gamma = self.dft_polar(y, norder=norder)
        dgamma_drho = self.dft_polar_der(y, norder=norder)
        
        return -2 * gamma * rhos * dgamma_drho

    def intensity(self, nu: Union[float, np.ndarray], n_hat: np.ndarray, params=None) -> Union[float, np.ndarray]:
        """
        Calculate specific intensity I_nu(nu, n_hat) using radial interpolation.
        
        Parameters
        ----------
        nu : float or array_like
            Frequency in Hz
        n_hat : array_like, shape (2,)
            Direction vector on sky in radians
        params : dict, optional
            Source parameters
            
        Returns
        -------
        intensity : float or array_like
            Specific intensity in appropriate units
        """
        if params is None:
            params = self.get_params()
        
        # Convert direction to radial distance
        r = np.sqrt(n_hat[0]**2 + n_hat[1]**2)
        
        # Convert radial angle to impact parameter
        # Apply scaling parameter if present
        impact_parameter = r * self.distance * params.get('s', 1.0)
        
        # Handle scalar vs array frequency input
        if np.isscalar(nu):
            # Find closest frequency
            freq_idx = np.argmin(np.abs(self.frequency_grid - nu))
            intensity_profile = self.I_nu_p[freq_idx, :]
            
            # Interpolate at the requested impact parameter
            return np.interp(impact_parameter, self.p_rays, intensity_profile)
        else:
            # Array of frequencies
            nu_array = np.asarray(nu)
            results = np.zeros_like(nu_array)
            for i, freq in enumerate(nu_array):
                results[i] = self.intensity(freq, n_hat, params=params)
            return results

    def specific_flux(self, nu: float) -> float:
        """
        Calculate total flux F_nu = ∫ I_nu d²n̂.
        
        Parameters
        ----------
        nu : float
            Frequency in Hz
            
        Returns
        -------
        flux : float
            Total flux density
        """
        # Find closest frequency
        freq_idx = np.argmin(np.abs(self.frequency_grid - nu))
        intensity_profile = self.I_nu_p[freq_idx, :]
        
        # Integrate over radial coordinates
        # Convert to solid angle integration: d²n̂ = 2π r dr for radial symmetry
        # where r is angular radius
        angular_radii = self.p_rays / self.distance
        integrand = intensity_profile * angular_radii * 2 * np.pi
        
        return np.trapz(integrand, angular_radii)

    def V(self, nu_0: float, baseline: np.ndarray, params: dict = None) -> complex:
        """
        Calculate the spatial visibility function V using polar DFT.
        
        This implements the core visibility calculation using the dft_polar
        algorithm from II.ipynb.
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz]
        params : dict, optional
            Source parameters
            
        Returns
        -------
        V : complex
            Complex fringe visibility
        """
        if params is None:
            params = self.get_params()
        
        # Find the closest frequency index
        freq_idx = np.argmin(np.abs(self.frequency_grid - nu_0))
        
        # Get intensity profile for this frequency
        intensity_profile = self.I_nu_p[freq_idx, :]
        
        # Normalize intensity profile
        intensity_norm = intensity_profile / np.sum(intensity_profile*self.p_rays)
        
        # Apply polar DFT to get gamma (which equals V)
        gamma = self.dft_polar(intensity_norm)
        gamma = fftshift(gamma)  # Shift to center zero frequency
        
        # Convert baseline to spatial frequency
        c = 2.99792458e8
        wavelength = c / nu_0
        baseline_perp = baseline[:2]
        baseline_length = np.linalg.norm(baseline_perp)
        
        # Calculate spatial frequency u = |B|/λ
        u = baseline_length / wavelength
        
        # Create frequency coordinates using fftfreq and fftshift (like GridSource)
        # The spacing in the polar DFT corresponds to the angular sampling
        angular_spacing = np.max(self.p_rays) / self.distance / len(intensity_norm)
        u_coords = fftshift(fftfreq(len(gamma), d=angular_spacing))
        
        # Find the closest frequency coordinate
        u_idx = np.argmin(np.abs(u_coords - u))
        
        return gamma[u_idx] + 0.0j

    def V_squared_jacobian(self, nu_0: float, baseline: np.ndarray, params: dict = None) -> Dict[str, float]:
        """
        Calculate the Jacobian of |V|² with respect to source parameters.
        
        This implements the dgamma2ds algorithm from II.ipynb.
        
        Parameters
        ----------
        nu_0 : float
            Central frequency in Hz
        baseline : array_like, shape (3,)
            Baseline vector in meters [Bx, By, Bz]
        params : dict, optional
            Source parameters
            
        Returns
        -------
        jacobian : dict
            Dictionary with parameter derivatives
        """
        if params is None:
            params = self.get_params()
        
        # Find the closest frequency index
        freq_idx = np.argmin(np.abs(self.frequency_grid - nu_0))
        
        # Get intensity profile for this frequency
        intensity_profile = self.I_nu_p[freq_idx, :]
        
        # Normalize intensity profile
        intensity_norm = intensity_profile / np.sum(intensity_profile * self.p_rays)
        
        # Calculate dgamma2ds
        dgamma2ds_result = self.dgamma2ds(intensity_norm)
        dgamma2ds_result = fftshift(dgamma2ds_result)  # Shift to center zero frequency
        
        # Convert baseline to appropriate index (same logic as V method)
        c = 2.99792458e8
        wavelength = c / nu_0
        baseline_perp = baseline[:2]
        baseline_length = np.linalg.norm(baseline_perp)
        
        # Calculate spatial frequency u = |B|/λ
        u = baseline_length / wavelength
        
        # Create frequency coordinates using fftfreq and fftshift (like GridSource)
        angular_spacing = np.max(self.p_rays) / self.distance / len(intensity_norm)
        u_coords = fftshift(fftfreq(len(dgamma2ds_result), d=angular_spacing))
        
        # Find the closest frequency coordinate
        u_idx = np.argmin(np.abs(u_coords - u))
        
        jacobian = {}
        
        # Derivative with respect to size parameter 's'
        if 's' in params:
            jacobian['s'] = dgamma2ds_result[u_idx]
        
        # Derivative with respect to phi_B (rotation parameter)
        # This would require additional implementation similar to dgamma2ds
        # For now, set to zero
        if 'phi_B' in params:
            jacobian['phi_B'] = 0.0
        
        return jacobian

    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters that define the source model.
        
        Returns
        -------
        dict
            Dictionary containing source parameters
        """
        return {
            's': 1.0,  # Size parameter
            'phi_B': self.phi_B  # Baseline orientation angle
        }

    @classmethod
    def from_hdf5(cls, hdf_file: str, distance: float = 204379200000000.0,
                  phi_B: float = 0.0, p_rays_units: str = 'm') -> "RadialGrid":
        """
        Create RadialGrid instance from HDF5 file.
        
        Parameters
        ----------
        hdf_file : str
            Path to HDF5 file containing intensity data
        distance : float
            Distance to source in meters
        phi_B : float
            Baseline orientation angle in radians
        p_rays_units : str
            Units of p_rays in the HDF5 file. Options: 'm' (meters), 'cm' (centimeters)
            Default is 'm' (meters)
            
        Returns
        -------
        RadialGrid
            Configured RadialGrid instance
        """
        # Read the HDF5 file
        intensity = pd.read_hdf(hdf_file, key='intensity')
        
        # Extract data arrays
        lambdas = intensity.index.values      # Wavelength grid [Angstrom]
        I_nu_p = intensity.values            # Intensity data [n_wavelengths, n_radial_points]
        p_rays = intensity.columns.values    # Impact parameter [original units]
        
        # Convert p_rays to meters if needed
        if p_rays_units == 'cm':
            p_rays = p_rays * 1e-2  # Convert cm to meters
        elif p_rays_units == 'm':
            pass  # Already in meters
        else:
            raise ValueError(f"Unsupported p_rays_units: {p_rays_units}. Use 'm' or 'cm'.")
        
        # Flip arrays to ensure proper ordering (as done in II.ipynb)
        lambdas = np.flip(lambdas)
        I_nu_p = np.flip(I_nu_p, axis=0)
        
        return cls(lambdas, I_nu_p, p_rays, distance=distance, phi_B=phi_B)

    def get_spectrum_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded spectrum.
        
        Returns
        -------
        dict
            Dictionary with spectrum information
        """
        return {
            'wavelength_range_angstrom': (np.min(self.lambdas), np.max(self.lambdas)),
            'frequency_range_hz': (self.freq_min, self.freq_max),
            'radial_range_m': (np.min(self.p_rays), np.max(self.p_rays)),
            'wavelength_points': self.n_wavelengths,
            'radial_points': self.n_radial_points
        }