# Unit Tests for II_Telescopes.pdf Results

This directory contains comprehensive unit tests that validate the intensity interferometry implementation against the results presented in the paper "Measuring type Ia supernova angular-diameter distances with intensity interferometry" (II_Telescopes.pdf).

## Overview

The test suite validates all major components and results from the paper:

- **TARDIS model** emission profiles and visibility calculations (Section II.A)
- **SEDONA model** asymmetric profiles and polarization effects (Section II.B)
- **Normalized visibility** calculations and Fourier transforms (Section III)
- **Fisher matrix analysis** for distance precision (Section IV)
- **Signal-to-noise ratio** predictions and scaling laws
- **Telescope array** configurations and baseline requirements
- **Observation simulation** and detection thresholds

## Test Files

### 1. `test_ii_telescopes_results.py`
**Main test suite covering all paper results**

- `TestTARDISModel`: Validates TARDIS spectral synthesis results
- `TestSEDONAModel`: Tests SEDONA 3D radiative transfer results  
- `TestNormalizedVisibility`: Validates visibility function calculations
- `TestDistancePrecision`: Tests distance measurement precision
- `TestSignalToNoiseRatio`: Validates SNR predictions
- `TestTelescopeArrayConfiguration`: Tests array configurations
- `TestPolarizationEffects`: Validates polarization calculations
- `TestSupernova2011feData`: Tests against real SN2011fe data
- `TestObservationSimulator`: Tests complete observation pipeline
- `TestPaperFigureReproduction`: Reproduces specific paper figures

### 2. `test_tardis_sedona_models.py`
**Specialized tests for spectral synthesis models**

- `TestTARDISModelValidation`: Detailed TARDIS model validation
- `TestSEDONAModelValidation`: Detailed SEDONA model validation
- `TestTARDISvsSEDONAComparison`: Direct model comparisons
- `TestVisibilityCalculationMethods`: Tests different calculation methods
- `TestPolarizationEffectsDetailed`: Detailed polarization analysis

### 3. `test_fisher_distance_precision.py`
**Fisher matrix and distance precision tests**

- `TestFisherMatrixFormulation`: Tests Fisher matrix equations
- `TestTARDISFisherAnalysis`: Fisher analysis for TARDIS model
- `TestSEDONAFisherAnalysis`: Fisher analysis for SEDONA model
- `TestMultiBaselineOptimization`: Tests baseline configurations
- `TestObservationTimeOptimization`: Tests observing time optimization

### 4. `run_all_tests.py`
**Comprehensive test runner with detailed reporting**

## Key Paper Results Validated

### Section II: Supernova Emission Profiles

✅ **TARDIS Model (Figure 2)**
- Wavelength coverage: 3000-10000 Å
- Impact parameter range: 0-3×10¹⁰ km
- Specific wavelengths: 3700, 4700, 6055, 6355, 8750 Å
- Spherical symmetry assumption

✅ **SEDONA Model (Figure 4)**
- Asymmetric emission profiles
- Wavelengths: 3696.69, 4698.29, 6128.39, 6189.92, 8745.82 Å
- 3D structure effects
- Time-dependent evolution

### Section III: Normalized Visibility

✅ **Visibility Functions (Figures 5-7)**
- Airy disk visibility: |V| = |2J₁(ζ)/ζ|
- Wavelength-dependent deviations
- TARDIS vs SEDONA comparisons
- Asymmetry effects <0.05

### Section IV: Distance Precision

✅ **Fisher Matrix Analysis**
- SNR scaling: σ⁻¹|V|² = (dΓ/dν)^(1/2) × (T_obs/σ_t)^(1/2) × (128π)^(-1/4)
- Expected SNR ≈ 1800 for paper parameters
- Distance parameter derivatives
- Two-parameter analysis (distance + orientation)

✅ **Signal-to-Noise Predictions (Figure 8)**
- Keck-like telescopes: 9.96 m diameter, efficiency = 0.39
- Timing jitter: 30 ps FWHM (13 ps RMS)
- Baseline optimization: <10 km for z=0.004 supernovae

✅ **Polarization Effects (Figure 10)**
- Integrated polarization P < 0.5%
- Spatial polarization structure
- Visibility differences <5×10⁻⁴

### Section V: Physical Predictions

✅ **Supernova Rates (Figure 1)**
- Rate: 2.43×10⁻⁵ SNe yr⁻¹ Mpc⁻³ h₇₀³
- Detection limit: m < 12 mag → z ≈ 0.004
- Expected rate: ~1 SN Ia per year

✅ **Angular Sizes and Baselines**
- Angular diameter: ~few microarcseconds
- Required baselines: <10 km at λ = 4400 Å
- Resolution scaling: θ = 1.22λ/ρ

## Running the Tests

### Quick Start
```bash
# Run all tests with basic output
python run_all_tests.py

# Run with verbose output
python run_all_tests.py --verbose

# Run with summary plots
python run_all_tests.py --plots
```

### Individual Test Suites
```bash
# Run main test suite
python -m unittest test_ii_telescopes_results -v

# Run TARDIS/SEDONA model tests
python -m unittest test_tardis_sedona_models -v

# Run Fisher matrix tests
python -m unittest test_fisher_distance_precision -v
```

### Specific Test Classes
```bash
# Test only TARDIS model
python -m unittest test_ii_telescopes_results.TestTARDISModel -v

# Test only visibility calculations
python -m unittest test_ii_telescopes_results.TestNormalizedVisibility -v

# Test only Fisher matrix formulation
python -m unittest test_fisher_distance_precision.TestFisherMatrixFormulation -v
```

## Data Requirements

### Required Data Files
- `import/SN2011fe_MLE_intensity_maxlight.hdf`: TARDIS intensity data for SN2011fe
- `import/WaveGrid.npy`: Wavelength grid data
- `import/Phase0Flux.npy`: Flux data (optional)

### Optional Data Files
- `import/Phase0Q.npy`, `import/Phase0U.npy`: Polarization data
- `import/11feP027.fit`: Observational data for comparison

**Note**: Tests will run without data files but some validations will be skipped.

## Expected Test Results

### Success Criteria
- **>95% pass rate**: Implementation closely matches paper
- **85-95% pass rate**: Good agreement, minor issues to address
- **70-85% pass rate**: Moderate agreement, needs improvement
- **<70% pass rate**: Significant issues requiring debugging

### Key Validations
1. **TARDIS emission profiles** match Figure 2 characteristics
2. **SEDONA asymmetric profiles** show expected features
3. **Visibility calculations** reproduce Airy disk behavior
4. **SNR scaling** gives ~1800 for paper parameters
5. **Fisher matrix** calculations are numerically stable
6. **Polarization effects** are at <5×10⁻⁴ level
7. **Physical parameters** match paper predictions

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure all modules are in Python path
export PYTHONPATH=$PYTHONPATH:.
```

**Missing Data Files**
- Tests will skip data-dependent validations
- Download SN2011fe data from paper's data repository
- Check file paths in `import/` directory

**Numerical Precision Issues**
- Some tests may fail due to floating-point precision
- Adjust tolerance in test assertions if needed
- Check for platform-specific numerical differences

**Memory Issues**
- Large FFT calculations may require significant memory
- Reduce grid sizes in test parameters if needed
- Run tests individually if memory is limited

### Test Failures

**TARDIS Model Tests**
- Check wavelength coverage and impact parameter ranges
- Verify emission profile normalization
- Ensure spherical symmetry is maintained

**SEDONA Model Tests**
- Confirm asymmetric profiles are generated
- Check polarization implementation
- Verify 3D structure effects

**Fisher Matrix Tests**
- Ensure numerical derivatives are stable
- Check matrix positive definiteness
- Verify parameter scaling relationships

**SNR Tests**
- Confirm photon rate calculations
- Check timing jitter implementation
- Verify spectroscopic enhancement

## Extending the Tests

### Adding New Tests
1. Create test class inheriting from `unittest.TestCase`
2. Add to appropriate test file or create new file
3. Update `run_all_tests.py` to include new module
4. Document expected results and validation criteria

### Testing New Models
1. Implement model in appropriate module
2. Create visibility function for Fisher analysis
3. Add comparison tests against paper results
4. Validate against observational data if available

### Performance Testing
```python
import time
import cProfile

# Profile test execution
cProfile.run('unittest.main()', 'test_profile.stats')

# Time critical functions
start_time = time.time()
# ... test code ...
execution_time = time.time() - start_time
```

## References

1. **Primary Paper**: "Measuring type Ia supernova angular-diameter distances with intensity interferometry" (II_Telescopes.pdf)

2. **TARDIS Documentation**: https://tardis-sn.github.io/tardis/

3. **SEDONA Code**: Kasen et al. radiative transfer code

4. **Intensity Interferometry Theory**: Hanbury Brown & Twiss effect

5. **Fisher Matrix Analysis**: Parameter estimation theory

## Contact

For questions about the test suite or implementation:
- Review paper Section references for theoretical background
- Check module docstrings for implementation details
- Examine test assertions for expected behavior
- Compare with paper figures and results

---

**Test Suite Version**: 1.0  
**Paper Reference**: II_Telescopes.pdf  
**Last Updated**: 2024  
**Python Version**: 3.8+  
**Dependencies**: numpy, scipy, matplotlib, pandas, h5py, astropy, unittest