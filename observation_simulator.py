"""
Supernova Intensity Interferometry Observation Simulator
Integrates telescope arrays, supernova models, and intensity interferometry calculations

This module provides a complete simulation framework for planning and analyzing
supernova intensity interferometry observations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM
import pandas as pd

# Import our modules
from telescope_arrays import TelescopeArray, TelescopeArray
from supernova_models import SupernovaEjecta, SupernovaParameters
from intensity_interferometry import IntensityInterferometer, ObservationParameters
from fisher_analysis import FisherMatrixCalculator, FisherResults


@dataclass
class ObservationPlan:
    """Complete observation plan for supernova intensity interferometry"""
    target_name: str
    supernova: SupernovaEjecta
    telescope_array: TelescopeArray
    observation_params: ObservationParameters
    observing_time: float  # seconds
    wavelength_range: Tuple[float, float]  # meters
    n_wavelengths: int = 5
    
    @property
    def wavelengths(self) -> np.ndarray:
        """Array of observing wavelengths"""
        return np.linspace(self.wavelength_range[0], self.wavelength_range[1], self.n_wavelengths)


@dataclass
class ObservationResults:
    """Results from supernova intensity interferometry observation"""
    target_name: str
    snr_total: float
    snr_per_baseline: Dict[float, float]
    visibility_measurements: Dict[float, Dict[float, float]]  # {baseline: {wavelength: visibility}}
    parameter_constraints: Optional[FisherResults] = None
    detection_significance: float = 0.0
    
    def is_detection(self, threshold: float = 5.0) -> bool:
        """Check if observation constitutes a detection"""
        return self.snr_total >= threshold


class SupernovaInterferometrySimulator:
    """
    Complete simulator for supernova intensity interferometry observations
    """
    
    def __init__(self, cosmology: Optional[FlatLambdaCDM] = None):
        if cosmology is None:
            self.cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        else:
            self.cosmology = cosmology
    
    def simulate_observation(self, plan: ObservationPlan) -> ObservationResults:
        """
        Simulate a complete supernova intensity interferometry observation
        
        Parameters:
        -----------
        plan : ObservationPlan
            Complete observation plan
            
        Returns:
        --------
        results : ObservationResults
            Simulation results
        """
        print(f"Simulating observation of {plan.target_name}")
        print(f"SN Type: {plan.supernova.params.sn_type}")
        print(f"Distance: {plan.supernova.params.distance:.1f} Mpc")
        print(f"Angular size: {plan.supernova.params.angular_radius_microarcsec:.1f} μas")
        print(f"Array: {plan.telescope_array.name}")
        print(f"Observing time: {plan.observing_time/3600:.1f} hours")
        
        # Calculate apparent magnitude
        distance_modulus = 5 * np.log10(plan.supernova.params.distance * 1e6 / 10)
        apparent_mag = plan.supernova.params.absolute_magnitude + distance_modulus
        print(f"Apparent magnitude: {apparent_mag:.1f}")
        
        # Check if source is too bright (detector saturation)
        if plan.telescope_array.check_saturation(apparent_mag):
            print("Warning: Source may saturate detectors!")
        
        # Initialize results storage
        snr_per_baseline = {}
        visibility_measurements = {}
        total_snr_squared = 0.0
        
        # Create intensity interferometer
        interferometer = IntensityInterferometer(plan.observation_params)
        
        # Loop over baselines
        for baseline in plan.telescope_array.baselines:
            baseline_length = baseline.length
            baseline_angle = baseline.angle
            
            visibility_measurements[baseline_length] = {}
            baseline_snr_squared = 0.0
            
            # Loop over wavelengths
            for wavelength in plan.wavelengths:
                # Calculate visibility amplitude
                visibility = plan.supernova.visibility_amplitude(
                    baseline_length, wavelength, include_polarization=True
                )
                visibility_measurements[baseline_length][wavelength] = visibility
                
                # Calculate photon rate
                photon_rate = plan.telescope_array.photon_rate_per_telescope(
                    apparent_mag, wavelength
                )
                dgamma_dnu = photon_rate / plan.observation_params.bandwidth
                
                # Calculate SNR for this baseline and wavelength
                snr = interferometer.photon_correlation_snr(dgamma_dnu, visibility)
                
                # Add to total (SNRs add in quadrature)
                baseline_snr_squared += snr**2
            
            # Store baseline SNR
            baseline_snr = np.sqrt(baseline_snr_squared)
            snr_per_baseline[baseline_length] = baseline_snr
            total_snr_squared += baseline_snr_squared
        
        # Calculate total SNR
        total_snr = np.sqrt(total_snr_squared)
        
        # Calculate detection significance
        detection_significance = total_snr
        
        print(f"Total SNR: {total_snr:.1f}")
        print(f"Detection significance: {detection_significance:.1f}σ")
        
        return ObservationResults(
            target_name=plan.target_name,
            snr_total=total_snr,
            snr_per_baseline=snr_per_baseline,
            visibility_measurements=visibility_measurements,
            detection_significance=detection_significance
        )
    
    def optimize_observation_time(self, plan: ObservationPlan, 
                                target_snr: float = 10.0) -> float:
        """
        Find optimal observing time to reach target SNR
        
        Parameters:
        -----------
        plan : ObservationPlan
            Observation plan (observing_time will be ignored)
        target_snr : float
            Target signal-to-noise ratio
            
        Returns:
        --------
        optimal_time : float
            Optimal observing time in seconds
        """
        # SNR scales as sqrt(observing_time)
        # So we can scale from a reference observation
        
        # Use 1 hour as reference
        reference_plan = ObservationPlan(
            target_name=plan.target_name,
            supernova=plan.supernova,
            telescope_array=plan.telescope_array,
            observation_params=plan.observation_params,
            observing_time=3600.0,  # 1 hour
            wavelength_range=plan.wavelength_range,
            n_wavelengths=plan.n_wavelengths
        )
        
        # Simulate reference observation
        reference_results = self.simulate_observation(reference_plan)
        reference_snr = reference_results.snr_total
        
        # Scale to target SNR
        time_scaling = (target_snr / reference_snr)**2
        optimal_time = 3600.0 * time_scaling
        
        print(f"To reach SNR = {target_snr:.1f}, need {optimal_time/3600:.1f} hours")
        
        return optimal_time
    
    def survey_simulation(self, supernova_sample: List[SupernovaEjecta],
                         telescope_array: TelescopeArray,
                         observing_time: float = 3600.0) -> pd.DataFrame:
        """
        Simulate observations of multiple supernovae
        
        Parameters:
        -----------
        supernova_sample : list
            List of SupernovaEjecta objects
        telescope_array : TelescopeArray
            Telescope array configuration
        observing_time : float
            Observing time per target in seconds
            
        Returns:
        --------
        survey_results : pd.DataFrame
            Results for all targets
        """
        results_list = []
        
        for i, supernova in enumerate(supernova_sample):
            # Create observation plan
            obs_params = ObservationParameters(
                central_frequency=const.c.value / 550e-9,
                bandwidth=const.c.value * 100e-9 / (550e-9)**2,
                observing_time=observing_time,
                timing_jitter_rms=telescope_array.telescopes[0].timing_jitter_rms,
                n_channels=1000
            )
            
            plan = ObservationPlan(
                target_name=f"SN{i+1:03d}_{supernova.params.sn_type}",
                supernova=supernova,
                telescope_array=telescope_array,
                observation_params=obs_params,
                observing_time=observing_time,
                wavelength_range=(450e-9, 650e-9),
                n_wavelengths=5
            )
            
            # Simulate observation
            results = self.simulate_observation(plan)
            
            # Store results
            distance_modulus = 5 * np.log10(supernova.params.distance * 1e6 / 10)
            apparent_mag = supernova.params.absolute_magnitude + distance_modulus
            
            result_dict = {
                'target_name': results.target_name,
                'sn_type': supernova.params.sn_type,
                'explosion_time': supernova.params.explosion_time,
                'distance_mpc': supernova.params.distance,
                'apparent_mag': apparent_mag,
                'angular_radius_uas': supernova.params.angular_radius_microarcsec,
                'snr_total': results.snr_total,
                'detection': results.is_detection(),
                'detection_significance': results.detection_significance
            }
            
            results_list.append(result_dict)
        
        return pd.DataFrame(results_list)
    
    def plot_survey_results(self, survey_df: pd.DataFrame, 
                          figsize: Tuple[float, float] = (15, 10)):
        """Plot survey simulation results"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # SNR vs distance
        axes[0, 0].scatter(survey_df['distance_mpc'], survey_df['snr_total'], 
                          c=survey_df['apparent_mag'], cmap='viridis', s=50)
        axes[0, 0].axhline(5, color='red', linestyle='--', label='5σ detection')
        axes[0, 0].set_xlabel('Distance (Mpc)')
        axes[0, 0].set_ylabel('Total SNR')
        axes[0, 0].set_title('SNR vs Distance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # SNR vs apparent magnitude
        axes[0, 1].scatter(survey_df['apparent_mag'], survey_df['snr_total'], 
                          c=survey_df['distance_mpc'], cmap='plasma', s=50)
        axes[0, 1].axhline(5, color='red', linestyle='--', label='5σ detection')
        axes[0, 1].set_xlabel('Apparent Magnitude')
        axes[0, 1].set_ylabel('Total SNR')
        axes[0, 1].set_title('SNR vs Apparent Magnitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # SNR vs angular size
        axes[0, 2].scatter(survey_df['angular_radius_uas'], survey_df['snr_total'], 
                          c=survey_df['explosion_time'], cmap='coolwarm', s=50)
        axes[0, 2].axhline(5, color='red', linestyle='--', label='5σ detection')
        axes[0, 2].set_xlabel('Angular Radius (μas)')
        axes[0, 2].set_ylabel('Total SNR')
        axes[0, 2].set_title('SNR vs Angular Size')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Detection rate by SN type
        detection_by_type = survey_df.groupby('sn_type')['detection'].agg(['sum', 'count'])
        detection_rate = detection_by_type['sum'] / detection_by_type['count']
        
        axes[1, 0].bar(detection_rate.index, detection_rate.values)
        axes[1, 0].set_xlabel('Supernova Type')
        axes[1, 0].set_ylabel('Detection Rate')
        axes[1, 0].set_title('Detection Rate by SN Type')
        axes[1, 0].grid(True, alpha=0.3)
        
        # SNR distribution
        axes[1, 1].hist(survey_df['snr_total'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(5, color='red', linestyle='--', label='5σ detection')
        axes[1, 1].set_xlabel('Total SNR')
        axes[1, 1].set_ylabel('Number of Targets')
        axes[1, 1].set_title('SNR Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Detection significance vs time
        axes[1, 2].scatter(survey_df['explosion_time'], survey_df['detection_significance'],
                          c=survey_df['sn_type'].astype('category').cat.codes, cmap='tab10', s=50)
        axes[1, 2].axhline(5, color='red', linestyle='--', label='5σ detection')
        axes[1, 2].set_xlabel('Time since explosion (days)')
        axes[1, 2].set_ylabel('Detection Significance (σ)')
        axes[1, 2].set_title('Detection vs Explosion Time')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def create_observation_summary(self, results: ObservationResults) -> str:
        """Create a summary report of observation results"""
        summary = f"""
Supernova Intensity Interferometry Observation Summary
====================================================

Target: {results.target_name}
Total SNR: {results.snr_total:.1f}
Detection Significance: {results.detection_significance:.1f}σ
Detection Status: {'DETECTED' if results.is_detection() else 'NOT DETECTED'}

Baseline Performance:
"""
        
        for baseline_length, snr in results.snr_per_baseline.items():
            summary += f"  {baseline_length/1000:.1f} km: SNR = {snr:.1f}\n"
        
        summary += f"""
Visibility Measurements:
  Number of baselines: {len(results.visibility_measurements)}
  Wavelength range: {min(list(results.visibility_measurements.values())[0].keys())*1e9:.0f}-{max(list(results.visibility_measurements.values())[0].keys())*1e9:.0f} nm
  
Recommendations:
"""
        
        if results.snr_total < 5:
            summary += "  - Increase observing time or use larger telescopes\n"
            summary += "  - Consider observing at earlier epochs when SN is brighter\n"
        elif results.snr_total < 10:
            summary += "  - Marginal detection - consider longer integration\n"
        else:
            summary += "  - Excellent detection - suitable for detailed analysis\n"
            summary += "  - Consider parameter estimation with Fisher matrix analysis\n"
        
        return summary


def create_example_observation_plan() -> ObservationPlan:
    """Create an example observation plan for demonstration"""
    # Create a Type Ia supernova
    supernova = SupernovaEjecta(SupernovaParameters(
        sn_type="Ia",
        explosion_time=10,  # 10 days post-explosion
        expansion_velocity=10000,  # km/s
        distance=20.0,  # Mpc
        absolute_magnitude=-19.46
    ))
    
    # Create telescope array
    telescope_array = TelescopeArray.cta_south_mst_like()
    
    # Create observation parameters
    obs_params = ObservationParameters(
        central_frequency=const.c.value / 550e-9,  # 550 nm
        bandwidth=const.c.value * 100e-9 / (550e-9)**2,  # 100 nm bandwidth
        observing_time=3600.0,  # 1 hour
        timing_jitter_rms=13e-12,  # 13 ps RMS
        n_channels=1000  # 1000 spectral channels
    )
    
    # Create observation plan
    plan = ObservationPlan(
        target_name="SN2024ex",
        supernova=supernova,
        telescope_array=telescope_array,
        observation_params=obs_params,
        observing_time=3600.0,  # 1 hour
        wavelength_range=(450e-9, 650e-9),  # 450-650 nm
        n_wavelengths=5
    )
    
    return plan


if __name__ == "__main__":
    # Example simulation
    print("Supernova Intensity Interferometry Observation Simulator")
    print("=" * 60)
    
    # Create simulator
    simulator = SupernovaInterferometrySimulator()
    
    # Create example observation plan
    plan = create_example_observation_plan()
    
    # Simulate observation
    results = simulator.simulate_observation(plan)
    
    # Print summary
    summary = simulator.create_observation_summary(results)
    print(summary)
    
    # Find optimal observing time
    optimal_time = simulator.optimize_observation_time(plan, target_snr=10.0)
    print(f"Optimal observing time for SNR=10: {optimal_time/3600:.1f} hours")
    
    # Create supernova sample for survey
    from supernova_models import create_supernova_sample
    sn_sample = create_supernova_sample()
    
    # Run survey simulation
    print(f"\nRunning survey simulation with {len(sn_sample)} targets...")
    survey_results = simulator.survey_simulation(sn_sample, plan.telescope_array)
    
    print(f"Survey Results:")
    print(f"Total targets: {len(survey_results)}")
    print(f"Detections (>5σ): {survey_results['detection'].sum()}")
    print(f"Detection rate: {survey_results['detection'].mean()*100:.1f}%")
    
    # Plot survey results
    fig, axes = simulator.plot_survey_results(survey_results)
    plt.savefig('survey_results.png', dpi=150, bbox_inches='tight')
    plt.show()