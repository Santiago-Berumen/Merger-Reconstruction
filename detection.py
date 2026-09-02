import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal
import gwbench
import numpy as np
import astropy.cosmology as apcosmo
import astropy.units as u
import utilities
import bilby

def build_gwbench_injection_from_bilby_params(mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, 
                                              theta_jn, luminosity_distance, ra, dec, psi, fref, phase):
    iota, chi1, chi2 = utilities.component_spins(
        theta_jn=theta_jn,
        phi_jl=phi_jl,  
        tilt_1=tilt_1,
        tilt_2=tilt_2,
        phi_12=phi_12, 
        a_1=a_2,
        a_2=a_1,
        mass_1=mass_1, 
        mass_2=mass_2,
        f_ref=fref,
        phase=phase,)
    
    gwbench_injection = {
        'Mc': bilby.gw.conversion.component_masses_to_chirp_mass(mass_1, mass_2),
        'eta': bilby.gw.conversion.component_masses_to_symmetric_mass_ratio(mass_1, mass_2),
        "chi1x" : chi1[0],
        "chi1y" : chi1[1],
        "chi1z" : chi1[2],
        "chi2x" : chi2[0],
        "chi2y" : chi2[1],
        "chi2z" : chi2[2],
        "DL" : luminosity_distance,
        "iota" : iota,
        "ra" : ra, 
        "dec" : dec, 
        "psi" : psi,
    }

    gwbench_injection['tc']   = 0.0
    gwbench_injection['phic'] = 0.0

    return gwbench_injection


def make_frequency_grid(injection_parameters, wf_model_name, wf_other_var_dic):
    # Innermost stable circular orbit (ISCO) frequency
    f_isco = gwbench.basic_relations.f_isco_Msolar(gwbench.basic_relations.M_of_Mc_eta(
      injection_parameters['Mc'],
      injection_parameters['eta']))
    
    # Set f_hi based on waveform validity
    if 'tf2' in wf_model_name:
      f_hi = f_isco
    elif 'lal_' in wf_model_name:
      if 'IMRPhenomD' in wf_other_var_dic['approximant']:
          f_hi = 4 * f_isco
      elif 'IMRPhenomPv2' in wf_other_var_dic['approximant']:
          f_hi = 4 * f_isco
      elif 'HM' in wf_other_var_dic['approximant']:
          f_hi = 8 * f_isco
    
    f_lo = 5.
    f_hi = np.minimum(2.**10,f_hi)
    df = 2.**-4
    f = np.arange(f_lo, f_hi + df, df)
    return f_lo, f_hi, df, f, f_isco


def run_fisher_analysis(approximant, network_specs, tag_map, injection_parameters,
                        wf_model_name='lal_bbh', deriv_symbols='Mc eta chi1z chi2z DL tc ra dec psi',
                        conv_cos=('dec','iota'), conv_log = ('Mc','DL'), derivs='num', step=1e-6, 
                        method='central', order=2, use_rot=True, only_net=True,):
    # Define the waveform
    wf_other_var_dic = {'approximant': approximant}

    # Frequency grid
    f_lo, f_hi, df, f, f_isco = make_frequency_grid(
        injection_parameters, wf_model_name, wf_other_var_dic
    )

    # Set up Fisher analysis
    net = gwbench.multi_network.MultiNetwork(network_specs,
                                             logger_level='ERROR',
                                             logger_level_network='ERROR')

    net.set_net_vars(
        wf_model_name = wf_model_name,
        wf_other_var_dic = wf_other_var_dic,
        deriv_symbs_string = deriv_symbols,
          conv_cos = conv_cos,
          conv_log = conv_log,
          use_rot = use_rot,
          f = f,
          inj_params = injection_parameters
    )

    net.setup_ant_pat_lpf_psds()

    net.calc_errors(
        only_net=only_net,
        derivs=derivs,
        step=step,
        method=method,
        order=order
    )

    out_data = {}
    for spec, sub in zip(network_specs, net.networks):
        tag = tag_map[tuple(spec)]
        out_data[tag] = {
          'snr':      sub.snr,
          'errs':     sub.errs,
          'cov':      sub.cov, # full covariance matrix
          'cond_num': sub.cond_num # condition number of Fisher matrix
        }

    return out_data