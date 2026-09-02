from lalsimulation import (SimInspiralTransformPrecessingNewInitialConditions, 
                           SimInspiralTransformPrecessingWvf2PE)
import lal

def spin_angles(mass_1, mass_2, chi_1, chi_2, iota=0.0, fref=-1, phase=0.0):
    chi_1x = chi_1[0]
    chi_1y = chi_1[1]
    chi_1z = chi_1[2]
    chi_2x = chi_2[0]
    chi_2y = chi_2[1]
    chi_2z = chi_2[2]
    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2 = \
        SimInspiralTransformPrecessingWvf2PE(
            incl=iota, S1x=chi_1x, S1y=chi_1y, S1z=chi_1z, 
            S2x=chi_2x, S2y=chi_2y, S2z=chi_2z,
            m1=mass_1, m2=mass_2, fRef=float(fref), phiRef=float(phase))
    return theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2

def component_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1,
                    mass_2, f_ref, phase):
    iota, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z = \
            SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                a_1, a_2, mass_1 * lal.MSUN_SI, mass_2 * lal.MSUN_SI,
                float(f_ref), float(phase))
    return iota, [chi_1x, chi_1y, chi_1z], [chi_2x, chi_2y, chi_2z]
