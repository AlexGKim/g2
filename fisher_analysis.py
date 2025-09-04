"""
Fisher Matrix Analysis for Intensity Interferometry
Based on "Probing H0 and resolving AGN disks with ultrafast photon counters" (Dalal et al. 2024)

This module implements Fisher matrix calculations for parameter estimation
from intensity interferometry observations of AGN.
"""

import numpy as np
import scipy.optimize as optimize
from typing import List, Dict, Callable, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class FisherResults:
    """Results from Fisher matrix analysis"""
    fisher_matrix: np.ndarray
    covariance_matrix: np.ndarray
    parameter_names: List[str]
    parameter_errors: np.ndarray
    correlation_matrix: np.ndarray
    
    def get_error(self, param_name: str) -> float:
        """Get 1σ error for a parameter"""
        if param_name in self.parameter_names:
            idx = self.parameter_names.index(param_name)
            return self.parameter_errors[idx]
        else:
            raise ValueError(f"Parameter {param_name} not found")
    
    def get_correlation(self, param1: str, param2: str) -> float:
        """Get correlation coefficient between two parameters"""
        idx1 = self.parameter_names.index(param1)
        idx2 = self.parameter_names.index(param2)
        return self.correlation_matrix[idx1, idx2]


class FisherMatrixCalculator:
    """
    Fisher matrix calculator for intensity interferometry observations
    
    Implements Equation (17) from the paper:
    F_αβ = (1/σ²|V|²) * (∂|V|²/∂p_α) * (∂|V|²/∂p_β)
    """
    
    def __init__(self, visibility_function: Callable, parameter_names: List[str]):
        """
        Initialize Fisher matrix calculator
        
        Parameters:
        -----------
        visibility_function : callable
            Function that returns |V|² given parameters and baseline info
            Signature: visibility_function(params, baseline_length, baseline_angle, wavelength)
        parameter_names : list
            Names of parameters to analyze
        """
        self.visibility_function = visibility_function
        self.parameter_names = parameter_names
        self.n_params = len(parameter_names)
    
    def numerical_derivative(self, params: np.ndarray, param_idx: int,
                           baseline_length: float, baseline_angle: float = 0.0,
                           wavelength: float = 550e-9, delta: float = 1e-6) -> float:
        """
        Calculate numerical derivative ∂|V|²/∂p_α
        
        Parameters:
        -----------
        params : np.ndarray
            Parameter values
        param_idx : int
            Index of parameter to differentiate
        baseline_length : float
            Baseline length in meters
        baseline_angle : float
            Baseline angle in radians
        wavelength : float
            Wavelength in meters
        delta : float
            Step size for numerical differentiation
            
        Returns:
        --------
        derivative : float
            ∂|V|²/∂p_α
        """
        # Use relative step size for better numerical stability
        step = max(delta, abs(params[param_idx]) * delta)
        
        # Forward difference
        params_plus = params.copy()
        params_plus[param_idx] += step
        
        params_minus = params.copy()
        params_minus[param_idx] -= step
        
        try:
            vis_plus = self.visibility_function(params_plus, baseline_length, baseline_angle, wavelength)
            vis_minus = self.visibility_function(params_minus, baseline_length, baseline_angle, wavelength)
            
            # Calculate |V|²
            vis2_plus = abs(vis_plus)**2 if np.iscomplex(vis_plus) else vis_plus**2
            vis2_minus = abs(vis_minus)**2 if np.iscomplex(vis_minus) else vis_minus**2
            
            derivative = (vis2_plus - vis2_minus) / (2 * step)
            
        except Exception as e:
            print(f"Error calculating derivative for parameter {param_idx}: {e}")
            derivative = 0.0
        
        return derivative
    
    def calculate_fisher_matrix(self, fiducial_params: np.ndarray,
                              baselines: List[Tuple[float, float]],
                              wavelengths: List[float],
                              sigma_v2_inv: float) -> np.ndarray:
        """
        Calculate Fisher matrix for given observation setup
        
        Parameters:
        -----------
        fiducial_params : np.ndarray
            Fiducial parameter values
        baselines : list of tuples
            List of (baseline_length, baseline_angle) pairs in meters and radians
        wavelengths : list
            List of wavelengths in meters
        sigma_v2_inv : float
            Inverse noise variance σ⁻¹|V|² from Equation (14)
            
        Returns:
        --------
        fisher_matrix : np.ndarray
            Fisher information matrix
        """
        fisher = np.zeros((self.n_params, self.n_params))
        
        # Sum over all baselines and wavelengths
        for baseline_length, baseline_angle in baselines:
            for wavelength in wavelengths:
                
                # Calculate derivatives for all parameters
                derivatives = np.zeros(self.n_params)
                for i in range(self.n_params):
                    derivatives[i] = self.numerical_derivative(
                        fiducial_params, i, baseline_length, baseline_angle, wavelength
                    )
                
                # Add contribution to Fisher matrix
                for i in range(self.n_params):
                    for j in range(self.n_params):
                        fisher[i, j] += sigma_v2_inv**2 * derivatives[i] * derivatives[j]
        
        return fisher
    
    def analyze_parameters(self, fiducial_params: np.ndarray,
                         baselines: List[Tuple[float, float]],
                         wavelengths: List[float],
                         sigma_v2_inv: float) -> FisherResults:
        """
        Perform complete Fisher matrix analysis
        
        Returns parameter errors, correlations, and covariance matrix
        """
        # Calculate Fisher matrix
        fisher_matrix = self.calculate_fisher_matrix(
            fiducial_params, baselines, wavelengths, sigma_v2_inv
        )
        
        # Calculate covariance matrix (inverse of Fisher matrix)
        try:
            covariance_matrix = np.linalg.inv(fisher_matrix)
        except np.linalg.LinAlgError:
            print("Warning: Fisher matrix is singular, using pseudo-inverse")
            covariance_matrix = np.linalg.pinv(fisher_matrix)
        
        # Extract parameter errors (diagonal elements)
        parameter_errors = np.sqrt(np.diag(covariance_matrix))
        
        # Calculate correlation matrix
        correlation_matrix = np.zeros_like(covariance_matrix)
        for i in range(self.n_params):
            for j in range(self.n_params):
                if parameter_errors[i] > 0 and parameter_errors[j] > 0:
                    correlation_matrix[i, j] = (
                        covariance_matrix[i, j] / (parameter_errors[i] * parameter_errors[j])
                    )
        
        return FisherResults(
            fisher_matrix=fisher_matrix,
            covariance_matrix=covariance_matrix,
            parameter_names=self.parameter_names,
            parameter_errors=parameter_errors,
            correlation_matrix=correlation_matrix
        )
    
    def plot_error_ellipse(self, results: FisherResults, param1: str, param2: str,
                          fiducial_values: np.ndarray, confidence_level: float = 0.68,
                          figsize: Tuple[float, float] = (8, 6)):
        """
        Plot error ellipse for two parameters
        
        Parameters:
        -----------
        results : FisherResults
            Results from Fisher analysis
        param1, param2 : str
            Names of parameters to plot
        fiducial_values : np.ndarray
            Fiducial parameter values
        confidence_level : float
            Confidence level (0.68 for 1σ, 0.95 for 2σ)
        """
        idx1 = self.parameter_names.index(param1)
        idx2 = self.parameter_names.index(param2)
        
        # Extract 2x2 covariance submatrix
        cov_sub = results.covariance_matrix[np.ix_([idx1, idx2], [idx1, idx2])]
        
        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(cov_sub)
        
        # Calculate ellipse parameters
        from scipy.stats import chi2
        chi2_val = chi2.ppf(confidence_level, df=2)
        
        # Semi-axes lengths
        a = np.sqrt(chi2_val * eigenvals[1])  # Major axis
        b = np.sqrt(chi2_val * eigenvals[0])  # Minor axis
        
        # Rotation angle
        angle = np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1])
        
        # Create ellipse
        theta = np.linspace(0, 2*np.pi, 100)
        ellipse_x = a * np.cos(theta)
        ellipse_y = b * np.sin(theta)
        
        # Rotate ellipse
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rot = ellipse_x * cos_angle - ellipse_y * sin_angle
        y_rot = ellipse_x * sin_angle + ellipse_y * cos_angle
        
        # Translate to fiducial values
        x_final = x_rot + fiducial_values[idx1]
        y_final = y_rot + fiducial_values[idx2]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_final, y_final, 'b-', linewidth=2, 
                label=f'{confidence_level*100:.0f}% confidence')
        ax.plot(fiducial_values[idx1], fiducial_values[idx2], 'ro', 
                markersize=8, label='Fiducial value')
        
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_title(f'Error Ellipse: {param1} vs {param2}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add error information
        error1 = results.get_error(param1)
        error2 = results.get_error(param2)
        correlation = results.get_correlation(param1, param2)
        
        info_text = f'σ({param1}) = {error1:.3f}\nσ({param2}) = {error2:.3f}\nρ = {correlation:.3f}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig, ax
    
    def plot_correlation_matrix(self, results: FisherResults, 
                               figsize: Tuple[float, float] = (10, 8)):
        """Plot correlation matrix as heatmap"""
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(results.correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(self.n_params))
        ax.set_yticks(range(self.n_params))
        ax.set_xticklabels(self.parameter_names, rotation=45, ha='right')
        ax.set_yticklabels(self.parameter_names)
        
        # Add correlation values as text
        for i in range(self.n_params):
            for j in range(self.n_params):
                text = ax.text(j, i, f'{results.correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(results.correlation_matrix[i, j]) < 0.5 else "white")
        
        ax.set_title("Parameter Correlation Matrix")
        fig.colorbar(im, ax=ax, label='Correlation Coefficient')
        
        plt.tight_layout()
        return fig, ax


def calculate_snr_scaling(photon_rate: float, observing_time: float,
                         timing_jitter: float, n_channels: int = 1) -> float:
    """
    Calculate σ⁻¹|V|² scaling factor from Equation (14)
    
    Parameters:
    -----------
    photon_rate : float
        Photon detection rate dΓ/dν in Hz/Hz
    observing_time : float
        Total observing time in seconds
    timing_jitter : float
        RMS timing jitter in seconds
    n_channels : int
        Number of spectral channels
        
    Returns:
    --------
    sigma_inv : float
        σ⁻¹|V|² scaling factor
    """
    sigma_inv = (
        np.sqrt(photon_rate) * 
        np.sqrt(observing_time / timing_jitter) *
        (128 * np.pi)**(-0.25) *
        np.sqrt(n_channels)
    )
    return sigma_inv


if __name__ == "__main__":
    print("Fisher Matrix Analysis Module")
    print("=" * 40)
    
    # Example: Calculate SNR scaling for paper's fiducial case
    photon_rate = 1.4e-7  # From paper
    observing_time = 1e5  # 10^5 seconds
    timing_jitter = 13e-12  # 13 ps RMS
    n_channels = 5000
    
    sigma_inv = calculate_snr_scaling(photon_rate, observing_time, timing_jitter, n_channels)
    print(f"σ⁻¹|V|² scaling factor: {sigma_inv:.0f}")
    print(f"Expected SNR for |V|=1: {sigma_inv}")
    
    # This should give ~1800 as in the paper
    print(f"Paper prediction: 1800")
    print(f"Ratio: {sigma_inv/1800:.2f}")