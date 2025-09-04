"""
Telescope Array Configuration Tools for Intensity Interferometry
Based on "Probing H0 and resolving AGN disks with ultrafast photon counters" (Dalal et al. 2024)

This module provides tools for configuring telescope arrays for intensity interferometry
observations of AGN, including baseline calculations, timing requirements, and array
optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import astropy.units as u
import astropy.constants as const


@dataclass
class Telescope:
    """Individual telescope configuration"""
    x: float  # Position in meters
    y: float  # Position in meters
    z: float = 0.0  # Elevation in meters
    area: float = 88.0  # Collecting area in m²
    efficiency: float = 0.8  # Overall efficiency
    timing_jitter_fwhm: float = 30e-12  # Timing jitter FWHM in seconds
    dead_time: float = 5e-9  # Dead time in seconds (for SPADs)
    
    @property
    def timing_jitter_rms(self) -> float:
        """RMS timing jitter (σ_t) from FWHM"""
        return self.timing_jitter_fwhm / 2.35
    
    @property
    def max_count_rate(self) -> float:
        """Maximum count rate before saturation (Hz)"""
        return 1.0 / self.dead_time if self.dead_time > 0 else np.inf


@dataclass
class Baseline:
    """Baseline between two telescopes"""
    tel1_idx: int
    tel2_idx: int
    length: float  # Baseline length in meters
    angle: float   # Position angle in radians
    u: float       # u-coordinate (B_perp/λ)
    v: float       # v-coordinate (B_perp/λ)
    
    def __post_init__(self):
        """Calculate u,v coordinates from length and angle"""
        self.u = self.length * np.cos(self.angle)
        self.v = self.length * np.sin(self.angle)


class TelescopeArray:
    """
    Telescope array for intensity interferometry observations
    
    Based on the CTA South MST array configuration from the paper:
    - 14 telescopes with 88 m² collecting area each
    - Baselines ranging from ~100m to ~16km for AGN observations
    - Timing precision of ~30 ps FWHM
    """
    
    def __init__(self, telescopes: List[Telescope], name: str = "Custom Array"):
        self.telescopes = telescopes
        self.name = name
        self.baselines = self._calculate_baselines()
        
    @classmethod
    def cta_south_mst_like(cls, baseline_max: float = 16000.0) -> 'TelescopeArray':
        """
        Create a CTA South MST-like array configuration
        
        Parameters:
        -----------
        baseline_max : float
            Maximum baseline length in meters (default 16 km as in paper)
        """
        n_telescopes = 14
        telescopes = []
        
        # Create a roughly circular array with some randomization
        # to avoid regular patterns that could cause systematic effects
        np.random.seed(42)  # For reproducible configurations
        
        for i in range(n_telescopes):
            if i == 0:
                # Central telescope
                x, y = 0.0, 0.0
            else:
                # Distribute telescopes in rough rings
                if i <= 6:
                    # Inner ring
                    radius = baseline_max * 0.1 + np.random.normal(0, baseline_max * 0.02)
                    angle = 2 * np.pi * i / 6 + np.random.normal(0, 0.2)
                else:
                    # Outer ring
                    radius = baseline_max * 0.4 + np.random.normal(0, baseline_max * 0.1)
                    angle = 2 * np.pi * (i - 6) / 7 + np.random.normal(0, 0.3)
                
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
            
            telescope = Telescope(
                x=x, y=y, z=0.0,
                area=88.0,  # m² as specified in paper
                efficiency=0.8,
                timing_jitter_fwhm=30e-12,  # 30 ps FWHM as in paper
                dead_time=5e-9  # 5 ns for SPADs
            )
            telescopes.append(telescope)
        
        return cls(telescopes, "CTA South MST-like Array")
    
    @classmethod
    def linear_array(cls, n_telescopes: int, spacing: float, 
                     telescope_area: float = 88.0) -> 'TelescopeArray':
        """Create a linear array configuration"""
        telescopes = []
        for i in range(n_telescopes):
            x = i * spacing
            telescope = Telescope(
                x=x, y=0.0, z=0.0,
                area=telescope_area,
                efficiency=0.8,
                timing_jitter_fwhm=30e-12
            )
            telescopes.append(telescope)
        
        return cls(telescopes, f"Linear Array ({n_telescopes} telescopes)")
    
    def _calculate_baselines(self) -> List[Baseline]:
        """Calculate all baselines between telescope pairs"""
        baselines = []
        n_tel = len(self.telescopes)
        
        for i in range(n_tel):
            for j in range(i + 1, n_tel):
                tel1, tel2 = self.telescopes[i], self.telescopes[j]
                
                dx = tel2.x - tel1.x
                dy = tel2.y - tel1.y
                length = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)
                
                baseline = Baseline(
                    tel1_idx=i,
                    tel2_idx=j,
                    length=length,
                    angle=angle,
                    u=0.0,  # Will be set when wavelength is specified
                    v=0.0
                )
                baselines.append(baseline)
        
        return baselines
    
    def update_uv_coordinates(self, wavelength: float):
        """
        Update u,v coordinates for all baselines at given wavelength
        
        Parameters:
        -----------
        wavelength : float
            Observing wavelength in meters
        """
        for baseline in self.baselines:
            baseline.u = baseline.length * np.cos(baseline.angle) / wavelength
            baseline.v = baseline.length * np.sin(baseline.angle) / wavelength
    
    @property
    def n_telescopes(self) -> int:
        """Number of telescopes in array"""
        return len(self.telescopes)
    
    @property
    def n_baselines(self) -> int:
        """Number of baselines (telescope pairs)"""
        return len(self.baselines)
    
    @property
    def total_collecting_area(self) -> float:
        """Total collecting area of all telescopes in m²"""
        return sum(tel.area for tel in self.telescopes)
    
    @property
    def baseline_lengths(self) -> np.ndarray:
        """Array of all baseline lengths in meters"""
        return np.array([b.length for b in self.baselines])
    
    @property
    def min_baseline(self) -> float:
        """Minimum baseline length in meters"""
        return np.min(self.baseline_lengths)
    
    @property
    def max_baseline(self) -> float:
        """Maximum baseline length in meters"""
        return np.max(self.baseline_lengths)
    
    def get_baseline_coverage(self, wavelength: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get u,v coverage for given wavelength
        
        Parameters:
        -----------
        wavelength : float
            Observing wavelength in meters
            
        Returns:
        --------
        u, v : np.ndarray
            u and v coordinates in units of wavelength
        """
        self.update_uv_coordinates(wavelength)
        u = np.array([b.u for b in self.baselines])
        v = np.array([b.v for b in self.baselines])
        return u, v
    
    def angular_resolution(self, wavelength: float) -> float:
        """
        Angular resolution at given wavelength (radians)
        
        Parameters:
        -----------
        wavelength : float
            Observing wavelength in meters
            
        Returns:
        --------
        resolution : float
            Angular resolution in radians (λ/B_max)
        """
        return wavelength / self.max_baseline
    
    def angular_resolution_microarcsec(self, wavelength: float) -> float:
        """Angular resolution in microarcseconds"""
        res_rad = self.angular_resolution(wavelength)
        return res_rad * (180 / np.pi) * 3600 * 1e6
    
    def photon_rate_per_telescope(self, source_magnitude: float, 
                                  wavelength: float = 550e-9,
                                  bandwidth: float = 100e-9) -> float:
        """
        Calculate photon detection rate per telescope
        
        Parameters:
        -----------
        source_magnitude : float
            Apparent magnitude of source
        wavelength : float
            Central wavelength in meters (default 550 nm)
        bandwidth : float
            Observing bandwidth in meters (default 100 nm)
            
        Returns:
        --------
        rate : float
            Photon detection rate in Hz
        """
        # Convert magnitude to flux density (Jy)
        # Using g=0 corresponds to 3730 Jy (from paper footnote)
        flux_jy = 3730 * 10**(-source_magnitude / 2.5)  # Jy
        flux_si = flux_jy * 1e-26  # W m⁻² Hz⁻¹
        
        # Convert to photon flux
        frequency = const.c.value / wavelength
        photon_energy = const.h.value * frequency
        photon_flux = flux_si / photon_energy  # photons m⁻² s⁻¹ Hz⁻¹
        
        # Total photon rate per telescope
        freq_bandwidth = const.c.value * bandwidth / wavelength**2
        rate = photon_flux * freq_bandwidth * self.telescopes[0].area * self.telescopes[0].efficiency
        
        return rate
    
    def check_saturation(self, source_magnitude: float, wavelength: float = 550e-9) -> bool:
        """
        Check if detectors will saturate for given source brightness
        
        Parameters:
        -----------
        source_magnitude : float
            Apparent magnitude of source
        wavelength : float
            Observing wavelength in meters
            
        Returns:
        --------
        saturated : bool
            True if any detector will saturate
        """
        rate = self.photon_rate_per_telescope(source_magnitude, wavelength)
        max_rate = self.telescopes[0].max_count_rate
        return rate > max_rate
    
    def plot_array_layout(self, figsize: Tuple[float, float] = (10, 8)):
        """Plot the telescope array layout"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot telescopes
        x_coords = [tel.x for tel in self.telescopes]
        y_coords = [tel.y for tel in self.telescopes]
        
        ax.scatter(x_coords, y_coords, s=100, c='red', marker='o', 
                  label=f'Telescopes ({self.n_telescopes})')
        
        # Plot baselines
        for baseline in self.baselines:
            tel1 = self.telescopes[baseline.tel1_idx]
            tel2 = self.telescopes[baseline.tel2_idx]
            ax.plot([tel1.x, tel2.x], [tel1.y, tel2.y], 'b-', alpha=0.3, linewidth=0.5)
        
        # Add telescope numbers
        for i, tel in enumerate(self.telescopes):
            ax.annotate(str(i), (tel.x, tel.y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'{self.name} Layout')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
        
        # Add statistics
        stats_text = f"""Array Statistics:
Telescopes: {self.n_telescopes}
Baselines: {self.n_baselines}
Total Area: {self.total_collecting_area:.0f} m²
Min Baseline: {self.min_baseline:.0f} m
Max Baseline: {self.max_baseline:.0f} m"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig, ax
    
    def plot_uv_coverage(self, wavelength: float = 550e-9, figsize: Tuple[float, float] = (10, 8)):
        """Plot u,v coverage at given wavelength"""
        u, v = self.get_baseline_coverage(wavelength)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot u,v points and their conjugates
        ax.scatter(u, v, c='blue', s=20, alpha=0.7, label='Baselines')
        ax.scatter(-u, -v, c='red', s=20, alpha=0.7, label='Conjugate baselines')
        
        ax.set_xlabel('u (wavelengths)')
        ax.set_ylabel('v (wavelengths)')
        ax.set_title(f'u,v Coverage at λ = {wavelength*1e9:.0f} nm')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
        
        # Add resolution information
        resolution_uas = self.angular_resolution_microarcsec(wavelength)
        ax.text(0.02, 0.98, f'Angular Resolution: {resolution_uas:.1f} μas', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig, ax
    
    def plot_baseline_distribution(self, figsize: Tuple[float, float] = (10, 6)):
        """Plot distribution of baseline lengths"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram of baseline lengths
        ax1.hist(self.baseline_lengths / 1000, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Baseline Length (km)')
        ax1.set_ylabel('Number of Baselines')
        ax1.set_title('Baseline Length Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_baselines = np.sort(self.baseline_lengths / 1000)
        cumulative = np.arange(1, len(sorted_baselines) + 1) / len(sorted_baselines)
        ax2.plot(sorted_baselines, cumulative, 'b-', linewidth=2)
        ax2.set_xlabel('Baseline Length (km)')
        ax2.set_ylabel('Cumulative Fraction')
        ax2.set_title('Cumulative Baseline Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def summary(self) -> Dict:
        """Return summary statistics of the array"""
        return {
            'name': self.name,
            'n_telescopes': self.n_telescopes,
            'n_baselines': self.n_baselines,
            'total_area_m2': self.total_collecting_area,
            'min_baseline_m': self.min_baseline,
            'max_baseline_m': self.max_baseline,
            'mean_baseline_m': np.mean(self.baseline_lengths),
            'median_baseline_m': np.median(self.baseline_lengths),
            'timing_jitter_rms_ps': self.telescopes[0].timing_jitter_rms * 1e12,
            'telescope_area_m2': self.telescopes[0].area,
            'telescope_efficiency': self.telescopes[0].efficiency
        }


def compare_arrays(arrays: List[TelescopeArray], wavelength: float = 550e-9):
    """Compare multiple telescope arrays"""
    print("Array Comparison:")
    print("=" * 80)
    
    for array in arrays:
        summary = array.summary()
        resolution_uas = array.angular_resolution_microarcsec(wavelength)
        
        print(f"\n{summary['name']}:")
        print(f"  Telescopes: {summary['n_telescopes']}")
        print(f"  Baselines: {summary['n_baselines']}")
        print(f"  Total Area: {summary['total_area_m2']:.0f} m²")
        print(f"  Baseline Range: {summary['min_baseline_m']:.0f} - {summary['max_baseline_m']:.0f} m")
        print(f"  Angular Resolution: {resolution_uas:.1f} μas")
        print(f"  Timing Precision: {summary['timing_jitter_rms_ps']:.1f} ps RMS")


if __name__ == "__main__":
    # Example usage
    print("Creating CTA South MST-like array for AGN observations...")
    
    # Create the fiducial array from the paper
    array = TelescopeArray.cta_south_mst_like(baseline_max=16000)
    
    # Print summary
    summary = array.summary()
    print(f"\nArray: {summary['name']}")
    print(f"Telescopes: {summary['n_telescopes']}")
    print(f"Baselines: {summary['n_baselines']}")
    print(f"Total collecting area: {summary['total_area_m2']:.0f} m²")
    print(f"Baseline range: {summary['min_baseline_m']:.0f} - {summary['max_baseline_m']:.0f} m")
    
    # Check angular resolution for AGN observations
    wavelength = 550e-9  # 550 nm
    resolution_uas = array.angular_resolution_microarcsec(wavelength)
    print(f"Angular resolution at 550 nm: {resolution_uas:.1f} μas")
    
    # Check photon rates for typical AGN
    agn_magnitude = 12  # g = 12 mag as in paper
    photon_rate = array.photon_rate_per_telescope(agn_magnitude)
    print(f"Photon rate per telescope (g={agn_magnitude}): {photon_rate:.2e} Hz")
    
    # Check for saturation
    saturated = array.check_saturation(agn_magnitude)
    print(f"Detector saturation risk: {'Yes' if saturated else 'No'}")
    
    # Plot array layout
    fig, ax = array.plot_array_layout()
    plt.savefig('array_layout.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot u,v coverage
    fig, ax = array.plot_uv_coverage(wavelength)
    plt.savefig('uv_coverage.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot baseline distribution
    fig, axes = array.plot_baseline_distribution()
    plt.savefig('baseline_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()