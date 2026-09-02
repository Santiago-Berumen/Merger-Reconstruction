"""
NOTATION:
mass_1: float (larger mass in solar masses)
mass_2: float (smaller mass in solar masses)
chi_1: array-like in the form [chi1x, chi1y, chi1z] (dimensionless spin)
chi_2: array-like in the form [chi2x, chi2y, chi2z] (dimensionless spin)
f_ref: float (reference frequency in Hz)
final_mass: float (final mass of the remnant in solar masses)
final_spin: array-like in the form [final_chi_x, final_chi_y, final_chi_z] (dimensionless spin of the remnant)
recoil_kick: array-like in the form [kick_x, kick_y, kick_z] (recoil kick velocity in km/s)
"""
from lalsimulation import nrfits
import lal, utilities
import numpy as np
from pycbc.conversions import get_final_from_initial


def final_parameters_surrogate(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2, phase, fref=-1):
    # Get the component spins
    iota, chi_1, chi_2 = utilities.component_spins(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
        mass_1, mass_2, f_ref=fref, phase=phase)
    
    # Get the NR fit
    remnant_properties = nrfits.eval_nrfit(mass_1 * lal.MSUN_SI, mass_2 * lal.MSUN_SI, 
                                            chi_1, chi_2, 
                                            'NRSur7dq4Remnant', 
                                            ['FinalMass', 'FinalSpin', 'RecoilKick'],
                                            f_ref=fref)

    # Get remnant properties
    final_mass = (remnant_properties["FinalMass"] / lal.MSUN_SI)[0] # in solar masses
    final_spin = remnant_properties["FinalSpin"]  # in dimensionless units
    recoil_kick = remnant_properties["RecoilKick"] * lal.C_SI / 1e3 # in km/s

    return final_mass, final_spin, recoil_kick


def kick_estimate(tilt_1, tilt_2, phi_12, q, a_1, a_2,
                maxphase=False, superkick=True, hangupkick=True, crosskick=True):
    """
    Estimate the kick of the merger remnant. We collect various numerical-relativity
    results, as described in Gerosa and Kesden 2016. Flags let you switch the
    various contributions on and off (all on by default): superkicks (Gonzalez et al. 2007a;
    Campanelli et al. 2007), hang-up kicks (Lousto & Zlochower 2011),
    cross-kicks (Lousto & Zlochower 2013). The orbital-plane kick components are
    implemented as described in Kesden et al. 2010a.  The final kick depends on
    the orbital phase at merger. By default, this is assumed to be uniformly
    distributed in [0,2pi]. The maximum kick is realized for Theta=0 and can be
    computed with the optional argument maxphase. This formula has to be applied *close to merger*, where
    numerical relativity simulations are available.
    """
    eta = q / (1 + q) ** 2 

    # Get unit vectors
    Lhat = np.array([0, 0, 1])  # L is aligned with z-axis
    S1hat = np.array([np.sin(tilt_1), 0, np.cos(tilt_1)])  # S1 in x-z plane
    S2hat = np.array([np.sin(tilt_2) * np.cos(phi_12),
                      np.sin(tilt_2) * np.sin(phi_12),
                      np.cos(phi_12)])  # S2 in x-y plane
    
    delta = -(q * a_1 * S2hat - a_1 * S1hat) / (1. + q)
    delta_parallel = np.dot(delta, Lhat)
    delta_perpendicular = np.linalg.norm(np.cross(delta, Lhat))

    chi_t = (q ** 2 * a_2 * S2hat + a_1 * S1hat) / ((1 + q) ** 2)
    chi_t_parallel = np.dot(chi_t, Lhat)
    chi_t_perpendicular = np.linalg.norm(np.cross(chi_t, Lhat))

    #Coefficients are quoted in km/s
    #vm and vperp from Kesden at 2010a. vpar from Lousto Zlochower 2013
    zeta=np.radians(145)
    A=1.2e4
    B=-0.93
    H=6.9e3

    # Multiply by 0/1 boolean flags to select terms to include
    V11 = 3677.76 * superkick
    VA = 2481.21 * hangupkick
    VB = 1792.45 * hangupkick
    VC = 1506.52 * hangupkick
    C2 = 1140 * crosskick
    C3 = 2481 * crosskick

    # max kick
    bigTheta = np.random.uniform(0., 2 * np.pi) * (not maxphase)

    vm = A * eta ** 2 * (1 + B * eta) * (1 - q) / (1 + q) 
    vperp = H * eta ** 2 * delta_parallel
    vpar = 16 * eta ** 2 * (delta_perpendicular * (V11 + 2 * VA * chi_t_parallel 
                                                   + 4 * VB * chi_t_parallel ** 2 
                                                   + 8 * VC * chi_t_parallel ** 3) 
                            + chi_t_perpendicular * delta_parallel * (2 * C2 + 4 * C3 * chi_t_parallel)) * np.cos(bigTheta)
    kick = np.array([vm + vperp * np.cos(zeta),vperp * np.sin(zeta), vpar])

    return kick

def params_in_surrogate_training_range(q, chi_1, chi_2, verbose):
    if (q > 4.010):
        if verbose:
            print(f"Mass ratio outside surrogate training range ({q:.3f} > 4.010).")
        return False
    elif (np.linalg.norm(chi_1) > 0.8100):
        if verbose:
            print(f"|chi_1| outside surrogate training range ({np.linalg.norm(chi_1):.4f} > 0.8100).")
        return False
    elif (np.linalg.norm(chi_2) > 0.8100):
        if verbose:
            print(f"|chi_2| outside surrogate training range ({np.linalg.norm(chi_2):.4f} > 0.8100).")
        return False
    else:
        return True

def final_parameters(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2, phase, fref=-1,
                         maxphase=False, superkick=True, hangupkick=True, crosskick=True, verbose=False):
    """
    Calculate the final kick magnitude using either the surrogate model or the Maggiore estimate.
    """
    q = mass_1 / mass_2

    # Check if the parameters are within the training range of the surrogate model
    in_training_range = params_in_surrogate_training_range(q, a_1, a_2, verbose)

    if in_training_range:
        mf, sf, rk = final_parameters_surrogate(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2, phase, fref)
        sf = np.linalg.norm(sf)  # Convert final spin to magnitude
    else:
        rk = kick_estimate(tilt_1, tilt_2, phi_12, q, a_1, a_2, maxphase, superkick, hangupkick, crosskick)
        iota, chi_1, chi_2 = utilities.component_spins(
            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
            mass_1, mass_2, f_ref=fref, phase=phase)
        mf, sf = get_final_from_initial(mass_1, mass_2, spin1x=chi_1[0], 
                                                        spin1y=chi_1[1],    
                                                        spin1z=chi_1[2],
                                                        spin2x=chi_2[0], 
                                                        spin2y=chi_2[1], 
                                                        spin2z=chi_2[2],
                                                        approximant='SEOBNRv4PHM')
    
    return mf, sf, rk
