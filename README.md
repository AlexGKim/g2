# Supernova Intensity Interferometry Framework

A Python implementation for simulating supernova observations using intensity interferometry techniques, based on Dalal et al. 2024.

## Overview

This framework provides tools for:
- Modeling supernova ejecta and photospheres
- Calculating visibility functions for intensity interferometry
- Simulating telescope array observations
- Performing signal-to-noise ratio calculations
- Parameter estimation using Fisher matrix analysis

## Installation

```bash
pip install numpy scipy matplotlib astropy pandas h5py
pip install sncosmo  # Optional: for supernova spectral analysis
```

## Core Modules

- **`telescope_arrays.py`**: Telescope array configurations (CTA-like, linear, custom)
- **`intensity_interferometry.py`**: Core intensity interferometry calculations
- **`supernova_models.py`**: Supernova ejecta models with polarization effects
- **`fisher_analysis.py`**: Parameter estimation analysis
- **`observation_simulator.py`**: Complete observation simulation
- **`example_demonstration.py`**: Complete framework demonstration

## Quick Start

```python
from telescope_arrays import TelescopeArray
from supernova_models import SupernovaEjecta, SupernovaParameters
from observation_simulator import SupernovaInterferometrySimulator, ObservationPlan
from intensity_interferometry import ObservationParameters
import astropy.constants as const

# Create telescope array
array = TelescopeArray.cta_south_mst_like(baseline_max=16000)

# Create supernova model
supernova = SupernovaEjecta(SupernovaParameters(
    sn_type="Ia",
    explosion_time=10,  # days since explosion
    expansion_velocity=10000,  # km/s
    distance=20.0,  # Mpc
    absolute_magnitude=-19.46
))

# Set observation parameters
obs_params = ObservationParameters(
    central_frequency=const.c.value / 550e-9,  # 550 nm
    bandwidth=const.c.value * 100e-9 / (550e-9)**2,  # 100 nm bandwidth
    observing_time=3600.0,  # 1 hour
    timing_jitter_rms=13e-12,  # 13 ps RMS
    n_channels=1000
)

# Create and run observation
plan = ObservationPlan(
    target_name="SN2024example",
    supernova=supernova,
    telescope_array=array,
    observation_params=obs_params,
    observing_time=3600.0,
    wavelength_range=(450e-9, 650e-9),
    n_wavelengths=5
)

simulator = SupernovaInterferometrySimulator()
results = simulator.simulate_observation(plan)

print(f"Total SNR: {results.snr_total:.1f}")
print(f"Detection: {'YES' if results.is_detection() else 'NO'}")
```

## Run Complete Demo

```python
python example_demonstration.py
```

## Key Features

### Supernova Models
- Multiple SN types (Ia, II, Ib, Ic)
- Time evolution and expanding photosphere
- Polarization effects and limb darkening

### Telescope Arrays
- CTA-like configurations
- Custom array geometries
- Baseline analysis and u,v coverage
- Performance metrics

### Intensity Interferometry
- Visibility function calculations
- SNR based on photon correlation statistics
- Timing jitter effects
- Multi-channel spectroscopic enhancement

## Scientific Applications

- **Supernova Physics**: Probe ejecta structure and expansion dynamics
- **Distance Measurements**: Use Type Ia SNe for cosmology
- **Survey Science**: Optimize detection strategies

## Technical Details

### SNR Calculation
```
SNR = |V|² / σ|V|²
σ⁻¹|V|² = (dΓ/dν)^(1/2) * (T_obs/σ_t)^(1/2) * (128π)^(-1/4) * √n_c
```

### Visibility Functions
- Uniform disk: `|V| = |2J₁(πθB/λ) / (πθB/λ)|`
- Limb darkening with modified Bessel functions
- Polarization via 2D FFT of intensity distributions

### Array Performance
- 14 telescopes, 88 m² each (CTA-like)
- Baselines: ~100m to ~16 km
- Timing precision: 30 ps FWHM (13 ps RMS)
- Total collecting area: 1232 m²

## Example Results

For Type Ia SN at 20 Mpc:
- Angular size: ~50 μas at 10 days post-explosion
- SNR: ~20-50 for 1-hour observation
- Detection threshold: >5σ achievable
- Optimal baselines: 1-10 km

## Integration

Compatible with existing code in `import/` directory:
- Same SN rate and magnitude distributions
- TARDIS/SEDONA spectral model compatibility
- Extends polarization analysis
- Builds on angular size calculations

## References

1. Dalal, N., et al. (2024). "Probing H₀ and resolving AGN disks with ultrafast photon counters." arXiv:2403.15903
2. Hanbury Brown, R., & Twiss, R. Q. (1956). Nature, 177, 27-29.

## License

Provided for scientific research purposes. Please cite relevant papers when using this code.