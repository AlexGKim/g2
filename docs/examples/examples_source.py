import numpy as np
from jax import numpy as jnp
import jax
from g2 import inverse_noise

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

    nus = jnp.array([nu_0, nu_0*1.1])
    baselines = jnp.array([baseline, baseline*1.1])
    print("")
    print("Source type:", type(source).__name__)  # Print source type
    print("Source parameters:", source.get_params())    # Print source parameters

    # Calclate signal
    print("|V|^2:")
    print(". nu scalar")
    print(source.V_squared(nu_0, baseline))
    print(". nu vectorized")
    print(jax.vmap(source.V_squared, in_axes=(0, None))(nus, baseline))
    print(". baseline vectorized")
    print(jax.vmap(source.V_squared, in_axes=(None, 0))(nu_0, baselines))
    print(". nu and baseline vectorized")
    print(jax.vmap(lambda nu: jax.vmap(lambda b: source.V_squared(nu, b))(baselines))(nus))

    print("")
    # Print partial Jacobian of signal w.r.t. source parameters
    
    print("Partial Jacobian of |V|^2 w.r.t. source parameters:", end=" ")
    print(". nu scalar")
    print(source.V_squared_jacobian(nu_0, baseline))
    print(". nu vectorized")
    print(jax.vmap(source.V_squared_jacobian, in_axes=(0, None))(nus, baseline))
    print(". baseline vectorized")
    print(jax.vmap(source.V_squared_jacobian, in_axes=(None, 0))(nu_0, baselines))
    print(". nu and baseline vectorized")
    print(jax.vmap(lambda nu: jax.vmap(lambda b: source.V_squared_jacobian(nu, b))(baselines))(nus))
 

    print("")
    # Calculate noise
    print("Inverse noise (SNR) for 1 hour integration:", end=" ")
    print(". nu scalar")
    print(inverse_noise(source, nu_0, baseline, integration_time, telescope_area=telescope_area,
                                throughput=throughput, detector_jitter=detector_jitter))
    print(". nu vectorized")
    print(jax.vmap(inverse_noise, in_axes=(None, 0, None, None, None, None, None))(source, nus, baseline, integration_time,
                                telescope_area,
                                throughput, detector_jitter))
    print(". baseline vectorized")
    print(jax.vmap(inverse_noise, in_axes=(None, None, 0, None, None, None, None))(source, nus, baseline, integration_time,
                                telescope_area,
                                throughput, detector_jitter))
    print(". nu and baseline vectorized")
    print(jax.vmap(lambda nu: jax.vmap(
        lambda b: inverse_noise(source, nu, b, integration_time, 
                               telescope_area, throughput, detector_jitter))(baseline))(nus))
    
    return
# summary-end

# multipoint-begin
from g2.models.sources.simple import  MultiPoint
# Create a simple binary system
flux_densities = [1e-26, 5e-27]  # W/m²/Hz - Primary and Secondary
positions = [[1e-8, 0], [-1e-8, 0]]  # radians - positions
source = MultiPoint(flux_densities, positions)
source.radius = 2e-8  # Set radius to 20 microarcseconds
#multipoint-end

summary(source)

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

# summary(source)