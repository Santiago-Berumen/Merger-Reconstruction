# change this to your directory!
base_dir = "/Users/santiago/Documents/Genealogy/hierarchical-merger-efficiency"

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import sys
sys.path.append(f"{base_dir}/Packages")
import lal  # noqa: F401 -- imported for the swiglal stdio redirect suppressed above
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import bilby
import GWFish.modules as gw
import precession
import functions2
import ancestral

# output directories
fisher_matrix_dir = f"{base_dir}/gwYYMMDD_results"
genealogy_dir     = f"{base_dir}/tutorial"
os.makedirs(f"{base_dir}/data_csv/NGNG", exist_ok=True)
os.makedirs(f"{base_dir}/data_csv/posteriors", exist_ok=True)
os.makedirs(fisher_matrix_dir, exist_ok=True)

# number of cpus to use
NPOOL = 4

# function to create a random sky location
def sample_sky_location(rng):
    """
    Draw an isotropic sky position plus orientation/polarization/phase.
    """
    return {
        "ra":       rng.uniform(0, 2 * np.pi),
        "dec":      np.arcsin(rng.uniform(-1, 1)),
        "psi":      rng.uniform(0, np.pi),
        "theta_jn": np.arccos(rng.uniform(-1, 1)),
        "phase":    rng.uniform(0, 2 * np.pi),
    }


def load_merger_from_csv(merger_csv, index, sky, distance=4000):
    """
    Load merger parameters from the merger CSV and build a GWFish-ready DataFrame.

    merger_csv : path to NGNGmerger_branches CSV (has masses, spins, tilts)
    index      : which merger to use (0-indexed)
    sky        : dict from sample_sky_location() -- ra, dec, psi, theta_jn, phase
    distance   : luminosity distance in Mpc — adjust to get desired SNR
    """
    df_merger = pd.read_csv(merger_csv)

    # Get the merger row
    row = df_merger.iloc[index]

    # Print summary
    print(f"Loaded merger index {index}:")
    print(f"m1 = {row['mass_1']:.3f} Msun, a1 = {row['a_1']:.3f}, tilt1 = {row['tilt_1']:.3f}")
    print(f"m2 = {row['mass_2']:.3f} Msun, a2 = {row['a_2']:.3f}, tilt2 = {row['tilt_2']:.3f}")
    print(f"remnant_mass = {row['m_f']:.3f} Msun, spin = {row['a_f']:.3f}, vkick = {row['v_k']:.1f} km/s")

    # Build GWFish parameters DataFrame using the sampled sky/orientation values
    parameters = pd.DataFrame({
        "chirp_mass":          [bilby.gw.conversion.component_masses_to_chirp_mass(row['mass_1'], row['mass_2'])],
        "mass_ratio":          [(row['mass_2'] / row['mass_1'])],
        "luminosity_distance": [distance],
        "theta_jn":            [sky['theta_jn']],
        "ra":                  [sky['ra']],
        "dec":                 [sky['dec']],
        "psi":                 [sky['psi']],
        "phase":               [sky['phase']],
        "geocent_time":        [1.412725e+09],
        "a_1":                 [row['a_1']],
        "a_2":                 [row['a_2']],
        "tilt_1":              [row['tilt_1']],
        "tilt_2":              [row['tilt_2']],
        "phi_12":              [np.pi],
        "phi_jl":              [np.pi],
    })

    # True values for comparison after genealogy runs
    true_values = {
        "m1":    row['mass_1'],
        "m2":    row['mass_2'],
        "a1":    row['a_1'],
        "a2":    row['a_2'],
        "m_rem": row['m_f'],
        "a_rem": row['a_f'],
        "vkick": row['v_k'],
    }

    return parameters, true_values, df_merger

# Define some useful conversion functions
# Effective spin
def chi_eff(mass_1_source, mass_2_source, a_1, a_2, cos_tilt_1, cos_tilt_2, phi_12):
    """
    Calculate the effective spin parameter (chi_eff).

    Parameters:
        mass_1_source (float): Source mass of the primary black hole.
        mass_2_source (float): Source mass of the secondary black hole.
        a_1 (float): Dimensionless spin magnitude of the primary.
        a_2 (float): Dimensionless spin magnitude of the secondary.
        cos_tilt_1 (float): Cosine of the tilt angle for the primary spin.
        cos_tilt_2 (float): Cosine of the tilt angle for the secondary spin.
        phi_12 (float): Difference in azimuthal angles between the two spins.

    Returns:
        float: Effective spin parameter (chieff).
    """
    q = mass_2_source / mass_1_source

    # Effective spin parameter
    chieff = (a_1 * cos_tilt_1 + q * a_2 * cos_tilt_2) / (1. + q)

    return chieff

# Effective precession parameters
def chi_p(mass_1_source, mass_2_source, a_1, a_2, cos_tilt_1, cos_tilt_2, phi_12):
    """
    Calculate the effective precession parameter (chi_p).

    Parameters:
        mass_1_source (float): Source mass of the primary black hole.
        mass_2_source (float): Source mass of the secondary black hole.
        a_1 (float): Dimensionless spin magnitude of the primary.
        a_2 (float): Dimensionless spin magnitude of the secondary.
        cos_tilt_1 (float): Cosine of the tilt angle for the primary spin.
        cos_tilt_2 (float): Cosine of the tilt angle for the secondary spin.
        phi_12 (float): Difference in azimuthal angles between the two spins.

    Returns:
        float: Effective precession parameter (chi_p).
    """
    q = mass_2_source / mass_1_source

    # Calculate sin(tilt) using cos(tilt)
    sintilt1 = np.sqrt(1 - cos_tilt_1**2)
    sintilt2 = np.sqrt(1 - cos_tilt_2**2)

    # Effective precession parameter
    chip = np.maximum(a_1 * sintilt1,
                (q * (4. * q + 3.) / (4. + 3. * q)) * a_2 * sintilt2)

    return chip


def convert_m1_m2_to_q(parameters):
    """
    Function to convert between sampled parameters and constraint parameter.

    Parameters
    ----------
    parameters: dict
        Dictionary containing sampled parameter values, 'mass_1_source', 'mass_2_source'.

    Returns
    -------
    dict: Dictionary with constraint parameter 'mass_ratio' added.
    """
    converted_parameters = parameters.copy()
    converted_parameters['mass_ratio'] = parameters['mass_2_source']/parameters['mass_1_source']
    return converted_parameters

# Define the NR fits that will be used to obtain the final mass and spin.
# precession always returns shape-(1,) arrays. NumPy 2 removed the implicit
# float(size-1 array) conversion that ancestral.log_likelihood relies on, so squeeze
# to a 0-d array here -- otherwise every likelihood call raises
# "only 0-dimensional arrays can be converted to Python scalars".
def remnant_mass(mass_1_source, mass_2_source, a_1, a_2, cos_tilt_1, cos_tilt_2, phi_12):
    M = mass_1_source + mass_2_source
    q = mass_2_source / mass_1_source
    return np.squeeze(precession.remnantmass(
        np.arccos(cos_tilt_1),
        np.arccos(cos_tilt_2),
        q,
        a_1,
        a_2,
    )) * M

def remnant_spin(mass_1_source, mass_2_source, a_1, a_2, cos_tilt_1, cos_tilt_2, phi_12):
    q = mass_2_source / mass_1_source
    return np.squeeze(precession.remnantspin(
        np.arccos(cos_tilt_1),
        np.arccos(cos_tilt_2),
        phi_12,
        q,
        a_1,
        a_2,
    ))

def remnant_kick(mass_1_source, mass_2_source, a_1, a_2, cos_tilt_1, cos_tilt_2, phi_12):
    q = mass_2_source / mass_1_source
    return np.squeeze(precession.remnantkick(
        np.arccos(cos_tilt_1),
        np.arccos(cos_tilt_2),
        phi_12,
        q,
        a_1,
        a_2,
        kms = True,
    ))


# ---------------------------------------------------------------------------
# Everything below runs only in the parent process. The __main__ guard is
# required for npool > 1: without it, every spawned worker would re-execute the
# whole pipeline (merger tree, Fisher analysis, plots) on import.
# ---------------------------------------------------------------------------
def main():

    # ---------------------------
    # Priors
    # ---------------------------
    priors = bilby.core.prior.PriorDict()
    priors['mass_1'] = bilby.core.prior.analytical.PowerLaw(minimum=5, maximum=50, alpha=-2.3,
                                                            name='mass_1')
    priors['a_1'] = bilby.core.prior.Uniform(name='a_1', minimum=0, maximum=0.1)
    priors['tilt_1'] = bilby.core.prior.analytical.Sine(name='tilt_1', minimum=0, maximum=np.pi)
    priors['tilt_2'] = bilby.core.prior.analytical.Sine(name='tilt_2', minimum=0, maximum=np.pi)
    priors['phi_12'] = bilby.core.prior.Uniform(name='phi_12', minimum=0,
                                                maximum=2 * np.pi, boundary='periodic')
    priors['phi_jl'] = bilby.core.prior.Uniform(name='phi_jl', minimum=0,
                                                maximum=2 * np.pi, boundary='periodic')
    priors['phase'] = bilby.core.prior.Uniform(name='phase', minimum=0,
                                               maximum=2 * np.pi, boundary='periodic')
    priors['theta_jn'] = bilby.core.prior.analytical.Sine(name='theta_jn')


    seed = 2
    np.random.seed(seed)
    bilby.core.utils.random.seed(seed)

    n_initial = 300 # number of BHs in initial population
    n_branches = 1 # number of branches. I am only using 1 at the moment because the pipeline only looks at the first one anyways
    start_time = time.time()
    all_branches = []

    for i in tqdm(range(n_branches)):
        branch = functions2.create_ngng_chain_mergers(priors, n_initial, v_esc=500, beta1=1.6, beta2=0, verbose=False)
        branch_df = pd.DataFrame(branch)
    
        # Add remnant role column
        remnant_role = ['none']  # first row has no remnant progenitor
        for j in range(1, len(branch_df)):
            prev_mf = branch_df.iloc[j - 1]['m_f']
            curr_m1 = branch_df.iloc[j]['mass_1']
            curr_m2 = branch_df.iloc[j]['mass_2']
            if np.isclose(prev_mf, curr_m1, rtol=1e-10):
                remnant_role.append('m1')
            elif np.isclose(prev_mf, curr_m2, rtol=1e-10):
                remnant_role.append('m2')
            else:
                remnant_role.append('unknown')
        branch_df['remnant_role'] = remnant_role

        # Reorder: move m_f to column 3 and remnant_role to column 4
        cols = branch_df.columns.tolist()
        cols.remove('m_f')
        cols.remove('remnant_role')
        cols.insert(2, 'm_f')
        cols.insert(3, 'remnant_role')
        branch_df = branch_df[cols]

        branch_df["branch_id"] = i
        all_branches.append(branch_df)
    
        merger_csv_path = f"{base_dir}/data_csv/NGNG/NGNGmerger_branches_job{i+1}.csv"
        branch_df.to_csv(merger_csv_path, index=False)
        print(f"Saved merger branches to {merger_csv_path}")

    df_master = pd.concat(all_branches, ignore_index=True)
    master_csv_path = f"{base_dir}/data_csv/NGNG/merger_branches_master.csv"
    df_master.to_csv(master_csv_path, index=False)
    print(f"Saved combined merger branches to {master_csv_path}")

    end_time = time.time()
    print(f"Elapsed time in making merger branch: {end_time - start_time:.3f} seconds")

    # Define the waveform model to read
    waveform_model = 'IMRPhenomXPHM'



    merger_csv = f'{base_dir}/data_csv/NGNG/NGNGmerger_branches_job1.csv'
    df_merger = pd.read_csv(merger_csv)
    index = len(df_merger) - 1     # which merger row (0=first, 1=second, etc.). I have set it so that it grabs the last one of the CSV since they usually escape by the second or third
    distance   = 4000    # Mpc

    # The genealogy reconstructs the progenitors of merger `index`, which are the
    # components of merger `index - 1`. With a single merger in the branch there is
    # no progenitor merger, and `index - 1` would silently wrap around to the last row.
    if index < 1:
        raise ValueError(
            f"Branch has only {len(df_merger)} merger(s); the genealogy needs a merger whose "
            "progenitor is itself a merger in the branch. Re-run the branch generation."
        )

    # Sky position and orientation are prior draws, so sample them directly
    rng = np.random.default_rng(seed)
    sky = sample_sky_location(rng)

    parameters, true_values, df_merger = load_merger_from_csv(merger_csv, index, sky, distance=distance)
    print("\nParameters DataFrame:")
    print(parameters)
    print("\nTrue values:")
    print(true_values)

    # Set the reference frequency
    #f_ref = result[f"C00:{result_waveform_key}"]["meta_data"]["meta_data"]["f_ref"][0]
    f_ref = 5.0
    print("reference_phase: ", f_ref)

    # The networks are the combinations of detectors that will be used for the analysis
    # The detection_SNR is the minimum SNR for a detection:
    #   --> The first entry specifies the minimum SNR for a detection in a single detector
    #   --> The second entry specifies the minimum network SNR for a detection
    # Build the network once and reuse it -- compute_network_errors already returns the
    # network SNR, so there is no need for a separate get_snr call
    detectors = ['CE1', 'ET']
    population_name = 'BBH'
    snr_threshold = 8.
    network = gw.detection.Network(detector_ids = detectors, detection_SNR = (0., snr_threshold))

    # The fisher parameters are the parameters that will be used to calculate the Fisher matrix
    # and on which we will calculate the errors
    fisher_parameters = ['chirp_mass', 'mass_ratio', 'luminosity_distance','theta_jn', 'dec','ra',
                         'psi', 'phase', 'geocent_time', 'a_1', 'a_2', "tilt_1", "tilt_2", "phi_12", "phi_jl"]

    detected, network_snr, parameter_errors, sky_localization = gw.fishermatrix.compute_network_errors(
            network = network,
            parameter_values = parameters,
            fisher_parameters=fisher_parameters, 
            waveform_model = waveform_model,
            f_ref = f_ref,
            use_duty_cycle = True
            )   
            # save_matrices = False, # default is False anyway, put True if you want Fisher and covariance matrices in the output
            # save_matrices_path = None, # default is None anyway,
                                         # otherwise specify the folder
                                         # where to save the Fisher and
                                         # corresponding covariance matrices

    print('The event passed the detection threshold: ', detected)
    print('The network SNR of the event is ', network_snr)
    print('The sky localization of the event is ', sky_localization)

    # Choose percentile factor of sky localization and pass from rad2 to deg2
    percentile = 90.
    sky_localization_90cl = sky_localization * gw.fishermatrix.sky_localization_percentile_factor(percentile)
    print(f'The {percentile:.0f}% CL sky localization is ', sky_localization_90cl)

    # One can create a dictionary with the parameter errors, the order is the same as the one given in fisher_parameters
    parameter_errors_dict = {}
    for i, parameter in enumerate(fisher_parameters):
        parameter_errors_dict['err_' + parameter] = np.squeeze(parameter_errors)[i]

    print('The parameter errors of the event are ', parameter_errors_dict)


    data_folder = fisher_matrix_dir  # created at the top of the script
    gw.fishermatrix.analyze_and_save_to_txt(network = network,
                                            parameter_values  = parameters,
                                            fisher_parameters = fisher_parameters, 
                                            sub_network_ids_list = [list(range(len(detectors)))],  # all detectors
                                            population_name = population_name,
                                            waveform_model = waveform_model,
                                            f_ref = f_ref,
                                            save_path = data_folder,
                                            save_matrices = True)


    # Load the covariance matrix. Derive the filename from the network so it stays in
    # sync if `detectors` or `snr_threshold` change.
    net_tag = f"{'_'.join(detectors)}_{population_name}_SNR{int(snr_threshold)}"
    cov_matrix = np.load(f"{data_folder}/inv_fisher_matrices_{net_tag}.npy")[0, :, :]

    # Save the mean values to sample.
    # Order must match fisher_parameters, since that is the order of cov_matrix.
    mean_values = parameters[fisher_parameters].iloc[0] # mean values of the parameters are just the parameters we injected

    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-10

    rng = np.random.default_rng(42) # fix for reproducibility

    fisher_samples = pd.DataFrame(rng.multivariate_normal(mean_values, cov_matrix, int(1e6)), columns = fisher_parameters)

    param_lbs = {'chirp_mass': r'$\mathcal{M}_c$ $[M_{\odot}]$', 'mass_ratio': r'$q$', 'luminosity_distance': r'$d_L$ [Mpc]',
                    'dec': r'$\mathrm{DEC}$ [rad]', 'ra': r'$\mathrm{RA}$ [rad]', 'theta_jn': r'$\theta_{JN}$ [rad]', 'psi': r'$\Psi$ [rad]',
                    'phase': r'$\phi$ [rad]', 'geocent_time': r'$t_c$ [rad][s]', 'a_1': r'$a_1$', 'a_2': r'$a_2$',
                    'tilt_1': r'$\theta_1$ [rad]', 'tilt_2': r'$\theta_2$ [rad]', 'phi_12': r'$\phi_{12}$ [rad]',
                    'phi_jl': r'$\phi_{JL}$ [rad]'}

    # Plot the posterior distribution of the parameter of choice
    N = len(fisher_parameters)
    fig = plt.figure(figsize=(35, 35))

    for i in range(N):
        sel_param = fisher_parameters[i]
        plt.subplot(5, 3, i+1)  # 5 rows x 3 columns

        #No longer using truncated + priors, we are now only using the raw fisher analysis
        plt.hist(fisher_samples[fisher_parameters[i]], bins=50, density=True,
                    label='Fisher', alpha = .6, color = 'skyblue') 
    
        #plt.hist(samples_from_truncated_lkh[fisher_parameters[i]], bins=50, histtype='step', density=True,
                    #label='Fisher + Truncated', alpha = 1., color = 'blue')
    
        #plt.hist(samples_from_posterior[fisher_parameters[i]], bins=50, density=True,
                    #label='Fisher + Truncated + Priors', alpha = .7, color = 'skyblue')
        #plt.axvline(injections[fisher_parameters.index(fisher_parameters[i])], color='black', linestyle='--', label='True Value')
    
        plt.axvline(mean_values[fisher_parameters[i]], color='red', linestyle='--', label='True Value')
        plt.xlabel(param_lbs[sel_param])
        plt.ylabel('PDF')

    plt.savefig(f"{base_dir}/Fisher_Samples_Priors.pdf")
    plt.close(fig)

    fisher_posteriors_csv = f'{base_dir}/data_csv/posteriors/test_fisher_posteriors.csv'
    fisher_samples.to_csv(fisher_posteriors_csv, index=False)

    # Load fisher posteriors
    fisher_posteriors = pd.read_csv(fisher_posteriors_csv)
    fisher_posteriors = bilby.gw.conversion.generate_mass_parameters(fisher_posteriors)

    # Determine which mass to reconstruct based on remnant_role of current merger
    role = df_merger.iloc[index]['remnant_role']

    posteriors_candidate = pd.DataFrame()
    m1, m2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(
        fisher_posteriors["chirp_mass"], fisher_posteriors["mass_ratio"]
    )
    if role == 'm1':
        posteriors_candidate["mass_source"] = m1
        posteriors_candidate["spin"] = fisher_posteriors["a_1"]
    elif role == 'm2':
        posteriors_candidate["mass_source"] = m2
        posteriors_candidate["spin"] = fisher_posteriors["a_2"]
    else:  # 'none' or 'unknown' — first merger or unmatched
        print(f"Warning: remnant_role is '{role}' for index {index}. Defaulting to m2.")
        posteriors_candidate["mass_source"] = m2
        posteriors_candidate["spin"] = fisher_posteriors["a_2"]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    if role == 'm1':
        mass_true = true_values["m1"]
        spin_true = true_values["a1"]
        mass_label = r"$m_1$"
        spin_label = r"$a_1$"
        spin_col = fisher_posteriors["a_1"]
    else:
        mass_true = true_values["m2"]
        spin_true = true_values["a2"]
        mass_label = r"$m_2$"
        spin_label = r"$a_2$"
        spin_col = fisher_posteriors["a_2"]

    ax[0].hist(posteriors_candidate["mass_source"], histtype="step", bins=30, density=True)
    ax[0].axvline(mass_true, color='red', linestyle='--', label='True value')
    ax[0].set_xlabel(mass_label)
    ax[0].set_ylabel(rf"$p({mass_label[1:-1]})$")
    ax[0].legend()

    ax[1].hist(spin_col, histtype="step", bins=30, density=True)
    ax[1].axvline(spin_true, color='red', linestyle='--', label='True value')
    ax[1].set_xlabel(spin_label)
    ax[1].set_ylabel(rf"$p({spin_label[1:-1]})$")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f"{base_dir}/sample_mass_spin_posterior.png")
    plt.close(fig)

    # Define the parameters to sample for and that functions will take in
    parameter_names = ['mass_1_source', 'mass_2_source', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'phi_12']

    # Derive component masses
    fisher_posteriors["mass_1_source"] = m1
    fisher_posteriors["mass_2_source"] = m2

    # Derive cos_tilt
    fisher_posteriors["cos_tilt_1"] = np.cos(fisher_posteriors["tilt_1"])
    fisher_posteriors["cos_tilt_2"] = np.cos(fisher_posteriors["tilt_2"])

    #Priors on parent parameters
    #mass_1_source, mass_2_source, a_1, a_2, cos_tilt_1, cos_tilt_2, phi_12
    # Named separately from the population `priors` above so the two cannot clobber each other
    parent_priors = bilby.core.prior.PriorDict(conversion_function=convert_m1_m2_to_q)
    parent_priors["mass_1_source"] = bilby.core.prior.Uniform(minimum=5, maximum=100, latex_label=r'$m_{\mathrm{1, p}}$')
    parent_priors["mass_2_source"] = bilby.core.prior.Uniform(minimum=5, maximum=100, latex_label=r'$m_{\mathrm{2, p}}$')
    parent_priors["mass_ratio"] = bilby.core.prior.Constraint(minimum=1/6, maximum=.99)

    parent_priors["a_1"] = bilby.core.prior.Uniform(minimum=0.0, maximum=0.99, latex_label=r'$\chi_{\mathrm{1, p}}$')
    parent_priors["a_2"] = bilby.core.prior.Uniform(minimum=0.0, maximum=0.99, name="a_2", latex_label=r'$\chi_{\mathrm{2, p}}$')

    parent_priors["cos_tilt_1"] = bilby.core.prior.Uniform(minimum=-1, maximum=1, latex_label=r'$\mathrm{cos \theta_{\mathrm{1, p}}}$')
    parent_priors["cos_tilt_2"] = bilby.core.prior.Uniform(minimum=-1, maximum=1, latex_label=r'$\mathrm{cos \theta_{\mathrm{2, p}}}$')
    parent_priors["phi_12"] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, boundary='periodic', latex_label=r'$\phi_{\mathrm{12, p}}$')

    # Subsample to 5000 points
    posteriors_candidate_sub = posteriors_candidate.sample(n=5000, random_state=42).reset_index(drop=True)
    print("subsampled: ", len(posteriors_candidate_sub))
    print("full: ", len(posteriors_candidate))

    genealogy = ancestral.Genealogy_Reconstruction(
        posteriors_candidate=posteriors_candidate_sub,  # use subsampled version
        parameter_names=parameter_names,
        label='tutorial',
        outdir=genealogy_dir,
        Mfin_NRfit=remnant_mass,
        Chifin_NRfit=remnant_spin,
        Vkick_NRfit=remnant_kick,
        Chieff=chi_eff,
        chip=chi_p,
        priors=parent_priors,
        method='gmm',
        method_kwargs={
            'n_components': 5,
            'covariance_type': 'full',
            'max_iter': 1000,
            'tol': 0.001,
            'random_state': 42
        },
        interp_method='interp2d',
        # NOTE: passing sampler_kwargs REPLACES Genealogy_Reconstruction's default
        # dict wholesale, so anything omitted here (npool, dlogz, ...) is simply lost.
        # Without npool the run is single-core.
        sampler_kwargs={
            'nlive': 1000,
            'naccept': 20,
            'sample': 'acceptance-walk',
            'save': 'hdf5',
            'npool': NPOOL,
            'dlogz': 0.1,
        },
    )

    genealogy.run_analysis()

    # Reconstructed progenitor posteriors. The genealogy infers the parents of the
    # remnant involved in merger `index`, i.e. the components of merger `index - 1`.
    genealogy_posterior = genealogy.result.posterior

    fig, ax = plt.subplots(1, 4, figsize=(21, 4))

    current_row = df_merger.iloc[index - 1]
    true_vals = [current_row["mass_1"], current_row["a_1"], current_row["mass_2"], current_row["a_2"]]
    samples_list = [genealogy_posterior["mass_1_source"], genealogy_posterior["a_1"],
                    genealogy_posterior["mass_2_source"], genealogy_posterior["a_2"]]
    labels       = [r"$m_1$", r"$\chi_1$", r"$m_2$", r"$\chi_2$"]

    for i, (samples, label, true_val) in enumerate(zip(samples_list, labels, true_vals)):
        lo, hi = np.percentile(samples, [5, 95])
    
        # Compute histogram
        counts, bin_edges = np.histogram(samples, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        # Plot full histogram
        ax[i].hist(samples, bins=30, histtype="step", density=True, color="#2171b5")
    
        # Shade only the bins within the 90% CI
        mask = (bin_centers >= lo) & (bin_centers <= hi)
        ax[i].fill_between(bin_edges[:-1], 0, 
                            np.where(mask, counts, 0),
                            step='post', alpha=0.3, color="#2171b5", label='90% CI')
    
        ax[i].axvline(true_val, color='red', linestyle='--', label='True value')
        ax[i].set_xlabel(label)
        ax[i].tick_params(direction='in', which='both')
        ax[i].legend()

    ax[0].set_ylabel(r"$p$")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/genealogy_posteriors.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
