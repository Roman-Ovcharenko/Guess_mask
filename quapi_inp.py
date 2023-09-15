#!/usr/local/bin/python3.8

import math
import numpy as np
from numpy import linalg as LA
import utils as utl

H2eV = 27.211396
H2cm = 219474.6305
eV2cm = H2cm/H2eV
H2kJmol = 2625.5
J2cm = 5.03412e22
fs2atu = 41.34137333656136
k_B_au = 3.1668114e-6 # [H/K] Boltzmann
k_B_cmK = 3.1668114e-6 * H2cm # [H/K] Boltzmann
##################################################################################################
# Create the .inp file for the quapi
##################################################################################################
def create_quapi_input(inp_file, ndk_mem, dt, tmax, T, kappa, sites, sd_file, 
        if_Hs_commute, if_mutual_commute, if_corr, trans_mx, rho_0, H, mask, overlp_mx, nbath, ndvr):

    with open(inp_file, "w") as file_:
        file_.write(" {:.16f}\n {:.16f}\n {:.16f}\n {:.1e}\n#\n".format(dt, tmax, T, kappa))

        for ibath in range(nbath):
            file_.write(" {:s}\n".format(sd_file[ibath]))

        file_.write("#If_baths_commute_with_Hamiltonian\n")
        strg = " "
        for ibath in range(nbath):
            strg += "{:d}   "
        strg += "\n"
        file_.write(strg.format(*if_Hs_commute))

        file_.write("#If_baths_commute_with_each_other\n")
        strg = " "
        for i in range(nbath):
            strg += "{:d}   "
        strg += "\n"
        for i in range(nbath):
            file_.write(strg.format(*if_mutual_commute[i]))

        file_.write("#If_baths_correlate_with_each_other\n")
        strg = " "
        for i in range(nbath):
            strg += "{:d}   "
        strg += "\n"
        for i in range(nbath):
            file_.write(strg.format(*if_mutual_correlate[i]))

        file_.write("#Sites\n")
        strg = " "
        for i in range(ndvr):
            strg += "{:.1f}   "
        strg += "\n"
        for ibath in range(nbath):
            file_.write(strg.format(*sites[ibath]))

        file_.write("#Overlap_matrix\n")
        strg = " "
        for i in range(ndvr):
            strg += "{:.16f}   "
        strg += "\n"
        for i in range(ndvr):
            file_.write(strg.format(*overlp_mx[i]))


        file_.write("#Transformation_matrix\n")
        strg = " "
        for i in range(ndvr):
            strg += "{:.16f}   "
        strg += "\n"
        for i in range(ndvr):
            file_.write(strg.format(*trans_mx[i]))

        file_.write("#rho_t=0\n")
        strg = " "
        for i in range(ndvr):
            strg += "{:.1f}   "
        strg += "\n"
        for i in range(ndvr):
            file_.write(strg.format(*rho_0[i]))

        file_.write("#Diabatic_Hamiltonian\n")
        strg = " "
        for i in range(ndvr):
            strg += "{:.16f}   "
        strg += "\n"
        for i in range(ndvr):
            file_.write(strg.format(*H[i]))

        file_.write("#Bath_memory_size\n")
        for ibath in range(nbath):
            file_.write(" {:d}\n".format(ndk_mem[ibath]))

        file_.write("#Mask\n")
        for ibath in range(nbath):

            strg = " "
            for imsk in range( len(mask[ ibath ]) ):
                strg += "{:d}   "
            strg += "\n"

            file_.write(strg.format(*mask[ ibath ]))

        file_.write("#")

    return
##################################################################################################
# Get spectral density
##################################################################################################
def get_spd_Ohmic(s, gamma, ww, wc):
    spd = [float( gamma * ww[iw]**s * np.exp(-np.abs(ww[iw])/wc)  ) for iw in range(nw)]
    return spd
##################################################################################################
def get_spd_exp_cutoff(s, gamma, ww, wc):
    nw = len(ww)
    spd = [float( gamma * ww[iw]**s * np.exp(-np.abs(ww[iw])/wc) / wc**(s-1)  ) for iw in range(nw)]
    return spd
##################################################################################################
def get_spd_Drude_lorentz(lambd, gamma, ww):
    nw = len(ww)
    spd = [float( float(2) * lambd * gamma * ww[iw] / ( ww[iw]**2 + gamma**2 )  ) for iw in range(nw)]
    return spd
##################################################################################################
# Plot spectral density
##################################################################################################
def plot_spectral_density(ww, spd, sd_file):
    nw = len(ww)
    str_ = "{:24.16e}        {:24.16e}\n"
    with open(sd_file, "w") as file_:
        for iw in range(nw):
            file_.write(str_.format(ww[iw], spd[iw]))

    return
#################################################################################################
if __name__ == '__main__':
    delta_cm = float(200) # [cm^-1]
    delta_au = delta_cm / H2cm
    dt = float( 0.15 ) / delta_au / fs2atu # [fs]
    T = float( 0.2 ) * delta_au / k_B_au # [K]

    nw = [ int(3000001) ]
    wmax = [ float(3000) * delta_cm ] # cm^-1
    wc = [float(10) * delta_cm ] # cm^-1
    s = [ float(1) ]
    gamma = [ float( 1 ) ] # no units
#    lambda_ = [ float(100) ] # cm^-1
    ndk_mem = [ int(24) ]
    ndk_mask = [ int(16) ]
###########################################################################################################################################################################
    ww = []
    spd = []
    sd_file = []

    sd_file.append( "spectral_density.dat" )

    ibath = int(0)

    ww.append( utl.get_ww(wmax[ ibath ], nw[ ibath ]) )
#    spd.append( get_spd_Drude_lorentz(lambda_[ ibath ], gamma[ ibath ], ww[ ibath ]) )
#    spd.append( get_spd_Ohmic(s, gamma_ncomm[ ibath ], ww_ncomm[ ibath ], wc_ncomm[ ibath ]) )
    spd.append( get_spd_exp_cutoff( s[ ibath ], gamma[ ibath ], ww[ ibath ], wc[ ibath ]) )
    plot_spectral_density(ww[ ibath ], spd[ ibath ], sd_file[ ibath ])

# Change to [a.u.]
    dt *= fs2atu
    for iw in range(nw[ ibath ]):
        ww[ ibath ][ iw ] /= H2cm
        spd[ ibath ][ iw ] /= H2cm

    Jtmp_0 = gamma[ ibath ]
#    Jtmp_0 = float(2) * lambda_[ ibath ] / gamma[ ibath ]
    Jtmp_pl, Jtmp_mn = utl.get_Jtmp(ww[ ibath ], spd[ ibath ], "Jtmp.dat", Jtmp_0)
    Ttmp_pl, Ttmp_mn, Ttmp_0 = utl.get_Ttmp(T, ww[ ibath ], "Ttmp.dat")
    Ktmp_pl, Ktmp_mn, Ktmp_0 = utl.get_Ktmp(dt, ww[ ibath ], ndk_mem[ ibath ], "re_Ktmp.dat", "im_Ktmp.dat", "abs_Ktmp.dat")
    eta, abs_eta, re_eta, im_eta = utl.get_eta_diag(ww[ ibath ], T, ndk_mem[ ibath ], Jtmp_pl, Jtmp_mn, Jtmp_0, Ttmp_pl, Ttmp_mn, Ttmp_0, Ktmp_pl, Ktmp_mn, Ktmp_0, "re_eta.dat", "im_eta.dat", "abs_eta.dat")

# Change to [cm]
    dt /= fs2atu
    for iw in range(nw[ ibath ]):
        ww[ ibath ][iw] *= H2cm
        spd[ ibath ][iw] *= H2cm
###########################################################################################################################################################################
    mask = utl.fit_mask(confidence_min=float(0.1), confidence_max=float(3.01), dconfidence=float(0.01), abs_eta=abs_eta, ndk_mem=ndk_mem[ibath], ndk_mask=ndk_mask[ibath])
###########################################################################################################################################################################



