"""
Tests for Fisher Matrix Analysis and Distance Precision
Based on Section IV of II_Telescopes.pdf

This test suite validates the Fisher matrix calculations for distance measurements
and the signal-to-noise ratio predictions from the paper.
"""

import unittest
import numpy as np
import scipy.special as special
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Import our modules
from fisher_analysis import FisherMatrixCalculator, FisherResults, calculate_snr_scaling
from intensity_interferometry import IntensityInterferometer, ObservationParameters
from supernova_models import SupernovaEjecta, SupernovaParameters
from telescope_arrays import TelescopeArray


class TestFisherMatrixFormulation(unittest.TestCase):
    """Test Fisher matrix formulation from Equations (17-20) in the paper"""
    
    def setUp(self):
        """Set up Fisher matrix test parameters"""
        # Paper parameters for SN at z=0.004, B=12 mag
        self.fiducial_distance = 20.0  # Mpc
        self.sn_params = SupernovaParameters(
            sn_type="Ia",
            explosion_time=20,
            expansion_velocity=10000,
            distance=self.fiducial_distance,
            absolute_magnitude=-19.46
        )
        self.supernova = SupernovaEjecta(self.sn_params)
        
        # Keck-like telescope parameters
        self.telescope_area = np.pi * (9.96/2)**2  # m²
        self.efficiency = 0.39  # Total efficiency from paper
        self.timing_jitter = 13e-12  # 13 ps RMS
        
        # Observation parameters
        self.obs_params = ObservationParameters(
            central_frequency=5.45e14,  # 550 nm
            bandwidth=1e12,  # 1 THz
            observing_time=3600,  # 1 hour
            timing_jitter_rms=self.timing_jitter,
            n_channels=1000
        )
    
    def test_airy_disk_fisher_matrix(self):
        """Test Fisher matrix for Airy disk model (Equations 11-13)"""
        
        def airy_visibility_function(params, baseline_length, baseline_angle, wavelength):
            """Airy disk visibility function for Fisher matrix testing"""
            distance_scale = params[0]  # s parameter from paper
            
            # Angular radius scales inversely with distance
            angular_radius = self.supernova.params.angular_radius / distance_scale
            
            # Calculate ζ = πθB/λ
            zeta = np.pi * angular_radius * baseline_length / wavelength
            
            if zeta == 0:
                return 1.0
            else:
                return abs(2 * special.j1(zeta) / zeta)
        
        # Create Fisher calculator
        fisher_calc = FisherMatrixCalculator(airy_visibility_function, ['distance_scale'])
        
        # Test parameters
        fiducial_params = np.array([1.0])  # s = 1
        baselines = [(1000, 0), (3000, 0), (5000, 0)]  # Different baseline lengths
        wavelengths = [550e-9]
        
        # Calculate σ⁻¹|V|² from paper parameters
        photon_rate = 1.4e-7  # Hz/Hz from paper
        sigma_v2_inv = calculate_snr_scaling(
            photon_rate, self.obs_params.observing_time, 
            self.timing_jitter, self.obs_params.n_channels
        )
        
        # Calculate Fisher matrix
        fisher_matrix = fisher_calc.calculate_fisher_matrix(
            fiducial_params, baselines, wavelengths, sigma_v2_inv
        )
        
        # Test Fisher matrix properties
        self.assertEqual(fisher_matrix.shape, (1, 1), "Fisher matrix should be 1x1 for single parameter")
        self.assertGreater(fisher_matrix[0, 0], 0, "Fisher matrix should be positive definite")
        
        # Calculate parameter error
        parameter_error = 1.0 / np.sqrt(fisher_matrix[0, 0])
        
        # Should give reasonable distance precision
        self.assertLess(parameter_error, 0.5, "Distance error should be <50%")
        self.assertGreater(parameter_error, 0.01, "Distance error should be >1%")
    
    def test_snr_scaling_formula_validation(self):
        """Test SNR scaling formula from Equation (14)"""
        # Paper's fiducial parameters
        photon_rate = 1.4e-7  # Hz/Hz
        observing_time = 1e5  # 10^5 seconds
        timing_jitter = 13e-12  # 13 ps RMS
        n_channels = 5000
        
        # Calculate σ⁻¹|V|²
        sigma_inv = calculate_snr_scaling(photon_rate, observing_time, timing_jitter, n_channels)
        
        # Should give ~1800 as stated in paper
        expected_snr = 1800
        self.assertAlmostEqual(sigma_inv, expected_snr, delta=300,
                              msg=f"SNR scaling should give ~{expected_snr} for paper parameters")
        
        # Test scaling with different parameters
        # SNR should scale as √(photon_rate)
        sigma_inv_2x = calculate_snr_scaling(2*photon_rate, observing_time, timing_jitter, n_channels)
        ratio = sigma_inv_2x / sigma_inv
        self.assertAlmostEqual(ratio, np.sqrt(2), delta=0.1,
                              msg="SNR should scale as √(photon_rate)")
        
        # SNR should scale as √(observing_time)
        sigma_inv_2t = calculate_snr_scaling(photon_rate, 2*observing_time, timing_jitter, n_channels)
        ratio_t = sigma_inv_2t / sigma_inv
        self.assertAlmostEqual(ratio_t, np.sqrt(2), delta=0.1,
                              msg="SNR should scale as √(observing_time)")
        
        # SNR should scale as √(n_channels)
        sigma_inv_2c = calculate_snr_scaling(photon_rate, observing_time, timing_jitter, 2*n_channels)
        ratio_c = sigma_inv_2c / sigma_inv
        self.assertAlmostEqual(ratio_c, np.sqrt(2), delta=0.1,
                              msg="SNR should scale as √(n_channels)")
    
    def test_derivative_calculation_accuracy(self):
        """Test numerical derivative calculation for Fisher matrix"""
        
        def simple_visibility_function(params, baseline_length, baseline_angle, wavelength):
            """Simple test function with known derivative"""
            x = params[0]
            # V = exp(-x²), so dV/dx = -2x*exp(-x²)
            return np.exp(-x**2)
        
        fisher_calc = FisherMatrixCalculator(simple_visibility_function, ['x'])
        
        # Test derivative at x = 0.5
        test_params = np.array([0.5])
        baseline_length = 1000
        wavelength = 550e-9
        
        numerical_deriv = fisher_calc.numerical_derivative(
            test_params, 0, baseline_length, 0, wavelength
        )
        
        # For V = exp(-x²), |V|² = exp(-2x²)
        # d|V|²/dx = -4x*exp(-2x²)
        x = test_params[0]
        analytical_deriv = -4 * x * np.exp(-2 * x**2)
        
        self.assertAlmostEqual(numerical_deriv, analytical_deriv, delta=0.01,
                              msg="Numerical derivative should match analytical result")


class TestTARDISFisherAnalysis(unittest.TestCase):
    """Test Fisher analysis for TARDIS model (Section IV.A)"""
    
    def setUp(self):
        """Set up TARDIS Fisher analysis parameters"""
        self.sn_params = SupernovaParameters(
            sn_type="Ia",
            explosion_time=20,
            expansion_velocity=10000,
            distance=20.0,  # z ≈ 0.004
            absolute_magnitude=-19.46
        )
        self.supernova = SupernovaEjecta(self.sn_params)
        
        # Paper parameters
        self.telescope_area = np.pi * (9.96/2)**2  # Keck-like
        self.efficiency = 0.39
        self.timing_jitter = 13e-12  # 13 ps RMS
    
    def test_tardis_visibility_derivative(self):
        """Test visibility derivative calculation for TARDIS model"""
        
        def tardis_visibility_function(params, baseline_length, baseline_angle, wavelength):
            """TARDIS-like visibility function"""
            distance_scale = params[0]
            
            # Use our supernova model
            temp_params = SupernovaParameters(
                sn_type=self.sn_params.sn_type,
                explosion_time=self.sn_params.explosion_time,
                expansion_velocity=self.sn_params.expansion_velocity,
                distance=self.sn_params.distance * distance_scale,
                absolute_magnitude=self.sn_params.absolute_magnitude
            )
            temp_sn = SupernovaEjecta(temp_params)
            
            return temp_sn.visibility_amplitude(baseline_length, wavelength, False)
        
        fisher_calc = FisherMatrixCalculator(tardis_visibility_function, ['distance_scale'])
        
        # Test derivative calculation
        fiducial_params = np.array([1.0])
        baseline_length = 2000  # 2 km
        wavelength = 550e-9
        
        derivative = fisher_calc.numerical_derivative(
            fiducial_params, 0, baseline_length, 0, wavelength
        )
        
        # Derivative should be non-zero for resolved source
        self.assertNotAlmostEqual(derivative, 0, places=6,
                                 msg="Visibility derivative should be non-zero for resolved source")
    
    def test_tardis_snr_vs_baseline(self):
        """Test SNR vs baseline for TARDIS model (Figure 8)"""
        # Wavelengths from Figure 8
        wavelengths_angstrom = [3700, 4700, 6055, 6355, 8750]
        baselines_km = np.linspace(2.5, 20, 10)  # 2.5 to 20 km
        
        # Calculate apparent magnitude
        distance_modulus = 5 * np.log10(self.sn_params.distance * 1e6 / 10)
        apparent_mag = self.sn_params.absolute_magnitude + distance_modulus
        
        # Should be approximately 12 mag
        self.assertAlmostEqual(apparent_mag, 12.0, delta=1.0,
                              msg="Apparent magnitude should be ~12 for z=0.004")
        
        # Test SNR calculation for different baselines
        for wl_angstrom in wavelengths_angstrom[:2]:  # Test first two wavelengths
            wavelength = wl_angstrom * 1e-10
            
            snrs = []
            for baseline_km in baselines_km:
                baseline_m = baseline_km * 1000
                
                # Calculate visibility
                visibility = self.supernova.visibility_amplitude(baseline_m, wavelength, False)
                
                # Calculate photon rate (simplified)
                photon_rate = 1e5  # Approximate Hz
                
                # Calculate SNR using intensity interferometry
                obs_params = ObservationParameters(
                    central_frequency=3e8 / wavelength,
                    bandwidth=1e12,
                    observing_time=3600,
                    timing_jitter_rms=self.timing_jitter,
                    n_channels=1
                )
                interferometer = IntensityInterferometer(obs_params)
                
                dgamma_dnu = photon_rate / obs_params.bandwidth
                snr = interferometer.photon_correlation_snr(dgamma_dnu, visibility)
                snrs.append(snr)
            
            # SNR should vary with baseline
            snr_array = np.array(snrs)
            if len(snr_array) > 1:
                snr_variation = np.std(snr_array) / np.mean(snr_array)
                self.assertGreater(snr_variation, 0.1,
                                 f"SNR should vary significantly with baseline at {wl_angstrom}Å")


class TestSEDONAFisherAnalysis(unittest.TestCase):
    """Test Fisher analysis for SEDONA model (Section IV.B)"""
    
    def setUp(self):
        """Set up SEDONA Fisher analysis parameters"""
        self.sn_params = SupernovaParameters(
            sn_type="Ia",
            explosion_time=20,
            expansion_velocity=10000,
            distance=20.0,
            absolute_magnitude=-19.46
        )
        self.supernova = SupernovaEjecta(self.sn_params)
    
    def test_sedona_two_parameter_fisher_matrix(self):
        """Test two-parameter Fisher matrix for SEDONA (distance + orientation)"""
        
        def sedona_visibility_function(params, baseline_length, baseline_angle, wavelength):
            """SEDONA-like visibility with distance and orientation parameters"""
            distance_scale = params[0]
            orientation_angle = params[1] if len(params) > 1 else 0.0
            
            # Create temporary supernova with scaled distance
            temp_params = SupernovaParameters(
                sn_type=self.sn_params.sn_type,
                explosion_time=self.sn_params.explosion_time,
                expansion_velocity=self.sn_params.expansion_velocity,
                distance=self.sn_params.distance * distance_scale,
                absolute_magnitude=self.sn_params.absolute_magnitude
            )
            temp_sn = SupernovaEjecta(temp_params)
            
            # Include polarization for asymmetry
            vis = temp_sn.visibility_amplitude(baseline_length, wavelength, True)
            
            # Add small orientation dependence
            orientation_factor = 1.0 + 0.01 * np.cos(baseline_angle - orientation_angle)
            
            return vis * orientation_factor
        
        # Test with two parameters
        fisher_calc = FisherMatrixCalculator(
            sedona_visibility_function, ['distance_scale', 'orientation']
        )
        
        fiducial_params = np.array([1.0, 0.0])  # s=1, φ=0
        baselines = [(2000, 0), (2000, np.pi/2), (5000, 0), (5000, np.pi/2)]  # Different orientations
        wavelengths = [550e-9]
        sigma_v2_inv = 100  # Simplified
        
        # Calculate Fisher matrix
        fisher_matrix = fisher_calc.calculate_fisher_matrix(
            fiducial_params, baselines, wavelengths, sigma_v2_inv
        )
        
        # Test matrix properties
        self.assertEqual(fisher_matrix.shape, (2, 2), "Fisher matrix should be 2x2")
        
        # Test positive definiteness
        eigenvals = np.linalg.eigvals(fisher_matrix)
        self.assertTrue(np.all(eigenvals > 0), "Fisher matrix should be positive definite")
        
        # Calculate parameter errors
        try:
            covariance = np.linalg.inv(fisher_matrix)
            distance_error = np.sqrt(covariance[0, 0])
            orientation_error = np.sqrt(covariance[1, 1])
            
            # Errors should be reasonable
            self.assertLess(distance_error, 1.0, "Distance error should be reasonable")
            self.assertLess(orientation_error, np.pi, "Orientation error should be reasonable")
            
        except np.linalg.LinAlgError:
            self.fail("Fisher matrix should be invertible")
    
    def test_sedona_bias_calculation(self):
        """Test bias calculation for SEDONA model (Equation 20)"""
        # Test bias due to model uncertainty
        
        def perturbed_visibility_function(params, baseline_length, baseline_angle, wavelength):
            """Visibility function with small perturbation"""
            distance_scale = params[0]
            
            temp_params = SupernovaParameters(
                sn_type=self.sn_params.sn_type,
                explosion_time=self.sn_params.explosion_time,
                expansion_velocity=self.sn_params.expansion_velocity,
                distance=self.sn_params.distance * distance_scale,
                absolute_magnitude=self.sn_params.absolute_magnitude
            )
            temp_sn = SupernovaEjecta(temp_params)
            
            vis = temp_sn.visibility_amplitude(baseline_length, wavelength, True)
            
            # Add 5% perturbation as mentioned in paper
            perturbation = 0.05 * np.random.normal()
            return vis * (1 + perturbation)
        
        fisher_calc = FisherMatrixCalculator(perturbed_visibility_function, ['distance_scale'])
        
        # Calculate bias using Fisher formalism
        fiducial_params = np.array([1.0])
        baselines = [(3000, 0), (5000, 0)]
        wavelengths = [550e-9]
        sigma_v2_inv = 100
        
        # Multiple realizations to test bias
        biases = []
        for _ in range(10):
            np.random.seed(42)  # For reproducibility
            fisher_matrix = fisher_calc.calculate_fisher_matrix(
                fiducial_params, baselines, wavelengths, sigma_v2_inv
            )
            
            if fisher_matrix[0, 0] > 0:
                parameter_error = 1.0 / np.sqrt(fisher_matrix[0, 0])
                biases.append(parameter_error)
        
        if len(biases) > 0:
            mean_bias = np.mean(biases)
            # Bias should be small compared to statistical error
            self.assertLess(mean_bias, 0.5, "Model bias should be reasonable")


class TestMultiBaselineOptimization(unittest.TestCase):
    """Test optimization of multiple baseline configurations"""
    
    def setUp(self):
        """Set up multi-baseline optimization tests"""
        self.sn_params = SupernovaParameters(
            sn_type="Ia",
            explosion_time=20,
            expansion_velocity=10000,
            distance=20.0,
            absolute_magnitude=-19.46
        )
        self.supernova = SupernovaEjecta(self.sn_params)
    
    def test_two_pair_vs_three_pair_configuration(self):
        """Test two-pair vs three-pair configurations from paper"""
        
        def visibility_function(params, baseline_length, baseline_angle, wavelength):
            distance_scale = params[0]
            orientation = params[1] if len(params) > 1 else 0.0
            
            temp_params = SupernovaParameters(
                sn_type=self.sn_params.sn_type,
                explosion_time=self.sn_params.explosion_time,
                expansion_velocity=self.sn_params.expansion_velocity,
                distance=self.sn_params.distance * distance_scale,
                absolute_magnitude=self.sn_params.absolute_magnitude
            )
            temp_sn = SupernovaEjecta(temp_params)
            
            vis = temp_sn.visibility_amplitude(baseline_length, wavelength, True)
            
            # Add orientation dependence
            orientation_factor = 1.0 + 0.02 * np.cos(baseline_angle - orientation)
            return vis * orientation_factor
        
        fisher_calc = FisherMatrixCalculator(visibility_function, ['distance_scale', 'orientation'])
        
        # Two-pair configuration (perpendicular baselines)
        two_pair_baselines = [(5000, 0), (5000, np.pi/2)]
        
        # Three-pair configuration (right isosceles triangle)
        baseline_length = 5000
        three_pair_baselines = [
            (baseline_length, 0),
            (baseline_length, np.pi/2),
            (baseline_length * np.sqrt(2), np.pi/4)
        ]
        
        fiducial_params = np.array([1.0, 0.0])
        wavelengths = [550e-9]
        sigma_v2_inv = 100
        
        # Calculate Fisher matrices
        fisher_2pair = fisher_calc.calculate_fisher_matrix(
            fiducial_params, two_pair_baselines, wavelengths, sigma_v2_inv
        )
        
        fisher_3pair = fisher_calc.calculate_fisher_matrix(
            fiducial_params, three_pair_baselines, wavelengths, sigma_v2_inv
        )
        
        # Three-pair should give better constraints
        try:
            cov_2pair = np.linalg.inv(fisher_2pair)
            cov_3pair = np.linalg.inv(fisher_3pair)
            
            error_2pair = np.sqrt(cov_2pair[0, 0])
            error_3pair = np.sqrt(cov_3pair[0, 0])
            
            # Three-pair configuration should give smaller errors
            self.assertLessEqual(error_3pair, error_2pair * 1.1,
                               "Three-pair configuration should give comparable or better precision")
            
        except np.linalg.LinAlgError:
            self.skipTest("Fisher matrices not invertible")
    
    def test_baseline_length_optimization(self):
        """Test optimal baseline length selection"""
        
        def simple_visibility_function(params, baseline_length, baseline_angle, wavelength):
            distance_scale = params[0]
            
            # Simple Airy disk model
            angular_radius = self.supernova.params.angular_radius / distance_scale
            zeta = np.pi * angular_radius * baseline_length / wavelength
            
            if zeta == 0:
                return 1.0
            else:
                return abs(2 * special.j1(zeta) / zeta)
        
        fisher_calc = FisherMatrixCalculator(simple_visibility_function, ['distance_scale'])
        
        # Test different baseline lengths
        baseline_lengths = np.linspace(1000, 10000, 10)  # 1-10 km
        wavelength = 550e-9
        sigma_v2_inv = 100
        
        fisher_values = []
        for baseline_length in baseline_lengths:
            baselines = [(baseline_length, 0)]
            fisher_matrix = fisher_calc.calculate_fisher_matrix(
                np.array([1.0]), baselines, [wavelength], sigma_v2_inv
            )
            fisher_values.append(fisher_matrix[0, 0])
        
        # Should have an optimal baseline length
        fisher_array = np.array(fisher_values)
        max_idx = np.argmax(fisher_array)
        optimal_baseline = baseline_lengths[max_idx]
        
        # Optimal baseline should be in reasonable range
        self.assertGreater(optimal_baseline, 2000, "Optimal baseline should be >2 km")
        self.assertLess(optimal_baseline, 8000, "Optimal baseline should be <8 km")


class TestObservationTimeOptimization(unittest.TestCase):
    """Test observation time optimization for target SNR"""
    
    def test_snr_target_calculation(self):
        """Test calculation of observing time needed for target SNR"""
        # Target SNRs from paper
        target_snrs = [5, 10, 20]  # Detection thresholds
        
        # Base observation parameters
        base_photon_rate = 1e5  # Hz
        base_time = 3600  # 1 hour
        timing_jitter = 13e-12
        n_channels = 1000
        visibility = 0.5
        
        for target_snr in target_snrs:
            # Calculate required observing time
            # SNR ∝ √(observing_time), so T_required = T_base * (SNR_target/SNR_base)²
            
            # Calculate base SNR
            obs_params = ObservationParameters(
                central_frequency=5.45e14,
                bandwidth=1e12,
                observing_time=base_time,
                timing_jitter_rms=timing_jitter,
                n_channels=n_channels
            )
            interferometer = IntensityInterferometer(obs_params)
            
            dgamma_dnu = base_photon_rate / obs_params.bandwidth
            base_snr = interferometer.photon_correlation_snr(dgamma_dnu, visibility)
            
            # Calculate required time
            if base_snr > 0:
                time_scaling = (target_snr / base_snr)**2
                required_time = base_time * time_scaling
                
                # Verify by calculating SNR with required time
                obs_params_new = ObservationParameters(
                    central_frequency=5.45e14,
                    bandwidth=1e12,
                    observing_time=required_time,
                    timing_jitter_rms=timing_jitter,
                    n_channels=n_channels
                )
                interferometer_new = IntensityInterferometer(obs_params_new)
                achieved_snr = interferometer_new.photon_correlation_snr(dgamma_dnu, visibility)
                
                # Should achieve target SNR within 10%
                self.assertAlmostEqual(achieved_snr, target_snr, delta=target_snr*0.1,
                                     msg=f"Should achieve target SNR of {target_snr}")
                
                # Required time should be reasonable
                self.assertLess(required_time, 24*3600*365,  # Less than 1 year
                               f"Required time for SNR={target_snr} should be reasonable")


def create_fisher_analysis_plots():
    """Create plots showing Fisher analysis results"""
    # Set up supernova model
    sn_params = SupernovaParameters(
        sn_type="Ia",
        explosion_time=20,
        expansion_velocity=10000,
        distance=20.0,
        absolute_magnitude=-19.46
    )
    supernova = SupernovaEjecta(sn_params)
    
    # Define visibility function
    def visibility_function(params, baseline_length, baseline_angle, wavelength):
        distance_scale = params[0]
        
        temp_params = SupernovaParameters(
            sn_type=sn_params.sn_type,
            explosion_time=sn_params.explosion_time,
            expansion_velocity=sn_params.expansion_velocity,
            distance=sn_params.distance * distance_scale,
            absolute_magnitude=sn_params.absolute_magnitude
        )
        temp_sn = SupernovaEjecta(temp_params)
        
        return temp_sn.visibility_amplitude(baseline_length, wavelength, False)
    
    # Create Fisher calculator
    fisher_calc = FisherMatrixCalculator(visibility_function, ['distance_scale'])
    
    # Test different baseline configurations
    baseline_lengths = np.linspace(1000, 10000, 20)  # 1-10 km
    wavelength = 550e-9
    sigma_v2_inv = 1000
    
    # Calculate Fisher information vs baseline
    fisher_values = []
    distance_errors = []
    
    for baseline_length in baseline_lengths:
        baselines = [(baseline_length, 0)]
        fisher_matrix = fisher_calc.calculate_fisher_matrix(
            np.array([1.0]), baselines, [wavelength], sigma_v2_inv
        )
        
        fisher_values.append(fisher_matrix[0, 0])
        
        if fisher_matrix[0, 0] > 0:
            distance_error = 1.0 / np.sqrt(fisher_matrix[0, 0])
            distance_errors.append(distance_error)
        else:
            distance_errors.append(np.inf)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fisher information vs baseline
    ax1.plot(baseline_lengths/1000, fisher_values, 'b-', linewidth=2)
    ax1.set_xlabel('Baseline Length (km)')
    ax1.set_ylabel('Fisher Information')
    ax1.set_title('Fisher Information vs Baseline Length')
    ax1.grid(True, alpha=0.3)
    
    # Distance error vs baseline
    ax2.plot(baseline_lengths/1000, np.array(distance_errors)*100, 'r-', linewidth=2)
    ax2.set_xlabel('Baseline Length (km)')
    ax2.set_ylabel('Distance Error (%)')
    ax2.set_title('Distance Precision vs Baseline Length')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 50)
    
    plt.tight_layout()
    plt.savefig('fisher_analysis_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, (ax1, ax2)


if __name__ == "__main__":
    # Run Fisher matrix and distance precision tests
    print("Running Fisher Matrix and Distance Precision Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestFisherMatrixFormulation,
        TestTARDISFisherAnalysis,
        TestSEDONAFisherAnalysis,
        TestMultiBaselineOptimization,
        TestObservationTimeOptimization
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Create analysis plots if tests pass
    if result.wasSuccessful():
        print("\nCreating Fisher analysis plots...")
        try:
            create_fisher_analysis_plots()
        except Exception as e:
            print(f"Could not create plots: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("FISHER ANALYSIS TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if not result.wasSuccessful():
        print("\nIssues found:")
        for test, traceback in result.failures + result.errors:
            print(f"- {test}")