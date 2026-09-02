#!/home/viviana.caceres/.conda/envs/hierarchical-mergers/bin/python
import numpy as np
import detection
from tqdm import trange

# Specify the detector networks to analyze
# Given in the format "{detector}_{location}". 
# see: https://gitlab.com/sborhanian/gwbench/-/blob/master/example_scripts/README.md
network_specs = [
    ['aLIGO_H', 'aLIGO_L'],
    ['CE-40_CEA'],
    ['CE-40_CEA', 'ET_ET1'],
    ['CE-40_CEA', 'CE-20_CEB', 'ET_ET1']
]

# Create tags for the different detector networks
tag_map = {
    tuple(['aLIGO_H', 'aLIGO_L']): 'HL',
    tuple(['CE-40_CEA']): 'CE40',
    tuple(['CE-40_CEA', 'ET_ET1']): 'CE40ET',
    tuple(['CE-40_CEA', 'CE-20_CEB', 'ET_ET1']): 'CE40CE20ET'
}

labels = ['HL', 'CE40', 'CE40ET', 'CE40CE20ET']

# --- Convergence parameters ---
n_samples = 500

# Array to collect results
snr_values = {label: [] for label in labels}

for i in trange(n_samples):
    # Randomize extrinsic parameters
    ra = np.random.uniform(0, 2*np.pi)           # [0, 2pi)
    dec = np.arcsin(np.random.uniform(-1, 1))    # isotropic on the sphere
    psi = np.random.uniform(0, np.pi)            # [0, pi)
    # sample isotropic orientation: cos(theta) uniform in [-1, 1]
    cos_th = np.random.uniform(-1.0, 1.0)
    theta_jn = np.arccos(cos_th)              # inclination in [0, pi]
    phase = np.random.uniform(0, 2*np.pi)     # waveform phase

    # Build injection for this sky position
    gwbench_injection = detection.build_gwbench_injection_from_bilby_params(
        mass_1=50.0, mass_2=30.0, 
        a_1=0.01, a_2=0.5,
        tilt_1=0.0, tilt_2=0.0, 
        phi_12=0.0, phi_jl=0.0,
        theta_jn=theta_jn, luminosity_distance=1000.0, 
        ra=ra, dec=dec, psi=psi, 
        fref=-1, phase=phase
    )

    # Run Fisher analysis for this realization
    out_data = detection.run_fisher_analysis(
        approximant='IMRPhenomXPHM',
        network_specs=network_specs,
        injection_parameters=gwbench_injection,
        tag_map=tag_map,
    )

    # Store SNRs for each label
    for label in labels:
        snr_values[label].append(out_data[label]["snr"])

# Print them out
# or find a way to save them and load them into a different python file to make your plot or sumn
for label, values in snr_values.items():
    if values:  # skip empty lists
        avg = np.mean(values)
        print(f"{label}: average SNR = {avg:.3f}")
    else:
        print(f"{label}: no data")