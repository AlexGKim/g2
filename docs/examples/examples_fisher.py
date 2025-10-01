import numpy as np
from jax import numpy as jnp
import jax
from g2 import inverse_noise, fisher_matrix, Observation

# summary-begin
def summary(source):
    c = 2.99792458e8  # Speed of light in m/s
    # the observation bandpass
    nu_0 = 5e14  # 600 nm
    lambda_0 = c / nu_0  # Wavelength in meters

    # Baseline distance at the resolution limit for a uniform disk source
    D_res = (1.22 * lambda_0  / (2 * source.radius))

    # Calculate inverse noise for a baseline measurement half the resolution limit
    baseline = np.array([D_res/2, -D_res/1.5, 0.0])
    baseline2 = np.array([D_res*1.1, -D_res*1.5, 0.0])

    # Observational parameters
    telescope_area = 1.0  # m²
    integration_time = 3600  # 1 hour
    detector_jitter = 1e-11  # 10 ps
    throughput = 1.

    nus = jnp.array([nu_0, nu_0*1.1])
    baselines = jnp.array([baseline, baseline*1.1, baseline*0.9, baseline2, baseline2*0.9 ])
    
    print("")
    print("Source type:", type(source).__name__)  # Print source type
    print("Source parameters:", source.get_params())    # Print source parameters

    # Calculate Fisher matrix
    print("Fisher matrix calculation:")
    
    # Create observation configuration (telescope/detector setup)
    obs = Observation(
        integration_time=integration_time,
        telescope_area=telescope_area,
        throughput=throughput,
        detector_jitter=detector_jitter
    )
    
    # Create lists of frequencies and baselines for Fisher matrix calculation
    nu_0_list = []
    baseline_list = []
    
    # Add measurements at different baselines
    for b in baselines:  # Use first two baselines
        for nu in [nu_0]:  # Use base frequency
            nu_0_list.append(float(nu))
            baseline_list.append(np.array(b))
    
    # Calculate Fisher matrix
    F = fisher_matrix(source, nu_0_list, baseline_list, obs)
    print(f"Fisher matrix shape: {F.shape}")
    print(f"Fisher matrix:\n{F}")
    
    # Calculate inverse noise for observations

    
    # Calculate parameter uncertainties if Fisher matrix is invertible
    try:
        cov_matrix = np.linalg.inv(F)
        uncertainties = np.sqrt(np.diag(cov_matrix))
        print("\nParameter uncertainties (Cramér-Rao bound):")
        param_names = list(source.get_params().keys())
        param_values = source.get_params()
        
        # Handle both scalar and array parameters
        uncertainty_idx = 0
        for param_name in param_names:
            param_value = param_values[param_name]
            if np.isscalar(param_value):
                # Scalar parameter
                uncertainty = uncertainties[uncertainty_idx]
                relative_uncertainty = uncertainty / abs(param_value) if param_value != 0 else float('inf')
                print(f"  {param_name}: {uncertainty:.2e} (relative: {relative_uncertainty:.2%})")
                uncertainty_idx += 1
            else:
                # Array parameter
                param_array = np.asarray(param_value)
                param_flat = param_array.flatten()
                for i, val in enumerate(param_flat):
                    uncertainty = uncertainties[uncertainty_idx]
                    relative_uncertainty = uncertainty / abs(val) if val != 0 else float('inf')
                    print(f"  {param_name}[{i}]: {uncertainty:.2e} (relative: {relative_uncertainty:.2%})")
                    uncertainty_idx += 1
                    
    except np.linalg.LinAlgError:
        print("Fisher matrix is singular - cannot compute uncertainties")
    
    return
# summary-end

# multipoint-begin
from g2.models.sources.simple import MultiPoint
# Create a simple binary system
flux_densities = [1e-26, 5e-27]  # W/m²/Hz - Primary and Secondary
positions = [[1e-8, 0], [-1e-8, 0.5e-8]]  # radians - positions
source = MultiPoint(flux_densities, positions)
source.radius = 2e-8  # Set radius to 20 microarcseconds
# multipoint-end

summary(source)

# uniform_disk-begin
from g2.models.sources.simple import UniformDisk

# instance of a UniformDisk
source = UniformDisk(flux_density=1e-26, radius=0.5e-3)  # 0.5 milliarcseconds
# uniform_disk-end

# summary-call-begin
summary(source)
# summary-call-end

# sn2011fe-begin
from g2.models.sources.grid_source import GridSource

# instance of a GridSource
source = GridSource.getSN2011feSource()
source.radius = source.pixel_scale_m/source.distance * 5  # Set radius to half the grid size
# sn2011fe-end

summary(source)

# shakura-sunyaev-begin
from g2.models.sources.agn import ShakuraSunyaevDisk
source = ShakuraSunyaevDisk(1e-15, 10, 6, 3, distance=1e25)
source.radius = 5e-13
# shakura-sunyaev-end
summary(source)