from g2.models.sources.simple import  UniformDisk
from g2 import calculate_inverse_noise
import numpy as np

# Example: Create a uniform disk source with a flux density of 1e-26 W/m²/Hz and an angular radius of 0.5 milliarcseconds

# instance of a UniformDisk
source = UniformDisk(flux_density=1e-26, radius=0.5e-3)  # 0.5 milliarcseconds

c = 2.99792458e8  # Speed of light in m/s
# the observation bnadpass
nu_0 = 5e14  # 600 nm
lambda_0 = c / nu_0  # Wavelength in meters
# delta_nu = 1e12  # 1 THz bandwidth

# Baseline distance set to the resolution limit for a uniform disk source
D = (1.22 * lambda_0  / (2 * source.radius))

# Calculate inverse noise for a baseline measurement
baseline = np.array([D/2, 0.0, 0.0])

telescope_area = 1.0  # m²
integration_time = 3600  # 1 hour
detector_jitter = 1e-11  # 10 ps
throughput = 1.

# Calclate signal
print("V^2 for baseline", baseline)
print(source.V_squared(nu_0, baseline))

# Print source parameters
print("Source parameters:")
print(source.get_params())

# Print partial Jacobian of signal w.r.t. source parameters
print("Partial Jacobian of V^2 w.r.t. source parameters:")
print(source.V_squared_jacobian(nu_0, baseline))

# Calculate noise
print("Inverse noise (SNR) for 1 hour integration:")
print(calculate_inverse_noise(source, nu_0, baseline, integration_time, telescope_area=telescope_area,
                              throughput=throughput, detector_jitter=detector_jitter))

