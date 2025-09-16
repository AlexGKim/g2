import numpy as np
from g2 import calculate_inverse_noise

# summary-begin
def summary(source):
    c = 2.99792458e8  # Speed of light in m/s
    # the observation bnadpass
    nu_0 = 5e14  # 600 nm
    lambda_0 = c / nu_0  # Wavelength in meters

    # Baseline distance at the resolution limit for a uniform disk source
    D_res = (1.22 * lambda_0  / (2 * source.radius))

    # Calculate inverse noise for a baseline measurement half the resolution limit
    baseline = np.array([D_res/2, 0.0, 0.0])

    # Observational parameters
    telescope_area = 1.0  # m²
    integration_time = 3600  # 1 hour
    detector_jitter = 1e-11  # 10 ps
    throughput = 1.

    # Calclate signal
    print("|V|^2:", end=" ")
    print(source.V_squared(nu_0, baseline))

    # Print source parameters
    print("Source parameters:", end=" ")
    print(source.get_params())

    # Print partial Jacobian of signal w.r.t. source parameters
    print("Partial Jacobian of |V|^2 w.r.t. source parameters:", end=" ")
    print(source.V_squared_jacobian(nu_0, baseline))

    # Calculate noise
    print("Inverse noise (SNR) for 1 hour integration:", end=" ")
    print(calculate_inverse_noise(source, nu_0, baseline, integration_time, telescope_area=telescope_area,
                                throughput=throughput, detector_jitter=detector_jitter))
    
    return
# summary-end

# uniform_disk-begin
from g2.models.sources.simple import  UniformDisk

# instance of a UniformDisk
source = UniformDisk(flux_density=1e-26, radius=0.5e-3)  # 0.5 milliarcseconds
# uniform_disk-end

# summary-call-begin
summary(source)
# summary-call-end

# sn2011fe-begin
from g2.models.sources.grid_source import GridSource

# instance of a UniformDisk
source = GridSource.getSN2011feSource()  # 0.5 milliarcseconds
source.radius = source.pixel_scale *5  # Set radius to half the grid size
# sn2011fe-end

summary(source)