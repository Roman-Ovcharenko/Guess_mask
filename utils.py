#!/usr/local/bin/python3.8

import cmath
import math
import numpy as np
import itertools

H2cm = 219474.6305
fs2atu = 41.34137333656136
k_B_au = 3.1668114e-6 # [H/K] Boltzmann
##################################################################################################
##################################################################################################
def fit_mask(confidence_min, confidence_max, dconfidence, abs_eta, ndk_mem, ndk_mask):
    Nconfidence = round( (confidence_max - confidence_min) / dconfidence )

    integral_eta_err = float(100000)
    for i in range(Nconfidence):
        confidence_tmp = confidence_min + i * dconfidence

        mask_tmp, abs_eta_err_tmp, integral_eta_err_tmp = generate_particular_mask(confidence_tmp, abs_eta, ndk_mem, ndk_mask)

        if integral_eta_err_tmp < integral_eta_err:
            confidence = confidence_tmp
            integral_eta_err = integral_eta_err_tmp
            abs_eta_err = abs_eta_err_tmp
            mask = [int(0)]*len(mask_tmp)
            for i in range(len(mask)):
                mask[i] = mask_tmp[i]

    print("ndk_mem: {:d}".format(ndk_mem))
    print("ndk_mask_ini: {:d}".format(ndk_mask))
    print("ndk_mask_act: {:d}".format(len(mask)))
    print("Confidence: {:f}".format(confidence))
    print("mask: {}".format(mask))
    print("Errors:")
    print("    abs_eta[mask[-1]] / abs_eta[0]: {:f}".format( abs_eta_err ))
#    print("    (int abs(abs_eta - abs_eta_mask): {:f}".format( integral_eta_err ))
    print("    (int abs(abs_eta**2 - abs_eta_mask**2): {:f}".format( integral_eta_err ))

    with open("abs_eta_on_mask.dat", "w") as file_:
        for imsk in range(len(mask)):
            file_.write("{:d}    {:f}\n".format(mask[imsk], abs_eta[ mask[imsk] ]))

    return mask
##################################################################################################
##################################################################################################
def generate_particular_mask(confidence, abs_eta, ndk_mem, ndk_mask):

    mask = [ 0, 1 ]
    istart = int(2)

    rho = [float(0)]*ndk_mem
    for i in range( istart, ndk_mem ):
        rho[i] = abs_eta[i]**2
#        rho[i] = abs_eta[i]

    for irest in range(ndk_mask-int(2), 1, -1):

        norm = integrate_eta(rho, range(ndk_mem))
        for i in range( ndk_mem ):
            rho[i] = irest * rho[i] / norm

        iend = find_iend(rho, istart, confidence)
        if iend == ndk_mem:
            break

        imsk = rho.index( max(rho[istart:iend+1]) )
        mask.append( imsk )

        for i in range(istart, iend+1):
            rho[i] = float(0)

        istart = iend+1
        if istart == ndk_mem:
            break

    if istart != ndk_mem:
        imsk = rho.index( max(rho[istart:ndk_mem]) )
        mask.append( imsk )

    abs_eta_err = abs_eta[ mask[-1] ] / abs_eta[ 0 ] 

    integral_eta_err = float(0)
#    norm = integrate_eta( abs_eta, range(ndk_mem) )
    rho_eta = [float(0)]*ndk_mem
    for i in range( ndk_mem ):
        rho_eta[i] = abs_eta[i]**2
    norm = integrate_eta( rho_eta, range(ndk_mem) )

    rho_eta = [float(0)]*ndk_mem
    for i in range( ndk_mem ):
#        rho_eta[i] = abs_eta[i] / norm
        rho_eta[i] = abs_eta[i]**2 / norm

    for imem in range(ndk_mem):

        for imsk in range(len(mask)):
            if mask[imsk] > imem:
                msk_cur = mask[imsk-1]
                break

        integral_eta_err += abs( rho_eta[msk_cur] - rho_eta[imem] )

    return mask, abs_eta_err, integral_eta_err
##################################################################################################
##################################################################################################
def find_iend(rho, istart, confidence):
    ndk_mem = len(rho)
    prob = float(0)
    for imem in range(istart, ndk_mem):
        prob += rho[imem]
        if prob >= confidence:
            return imem
    return ndk_mem
##################################################################################################
# Get frequency
##################################################################################################
def get_ww(wmax, nw):
    dw = wmax / nw 
    ww = [float((iw+0.5)*dw) for iw in range(nw)]
    return ww
##################################################################################################
# Get Jtmp(w) = J(w) / w  in [a.u.]
# and Jtmp_0 = lim Jtmp(w) at w \to 0
# *_pl for w > 0
# *_mn for w < 0
##################################################################################################
def get_Jtmp(ww, spd, Jtmp_file, Jtmp_0):
    nw = len(ww)
    Jtmp_pl = []
    Jtmp_mn = []
    for iw in range(nw):
        Jtmp_pl.append(float( spd[iw] / ww[iw] ))
        Jtmp_mn.append(Jtmp_pl[iw])

    with open(Jtmp_file, "w") as file_:
        for iw in range(nw-1, 1, -1):
            file_.write("{:.12f}    {:.12f}\n".format(-ww[iw], Jtmp_mn[iw]))
        file_.write("{:.12f}    {:.12f}\n".format(float(0), Jtmp_0))
        for iw in range(nw):
            file_.write("{:.12f}    {:.12f}\n".format(ww[iw], Jtmp_pl[iw]))

    return Jtmp_pl, Jtmp_mn
##################################################################################################
# Get Ttmp(w) = 0.5 * w * exp(0.5*beta*w) / sinh(0.5*beta*w) = w / (1 - exp(-beta*w))  in [a.u.]
# and Ttmp_0 = lim Ttmp(w) at w \to 0 = 1 / beta
# *_pl for w > 0
# *_mn for w < 0
##################################################################################################
def get_Ttmp(T, ww, Ttmp_file):
    beta = 1 / (k_B_au * T) # [1/H]
    Ttmp_0 = float(1) / beta

    nw = len(ww)
    Ttmp_pl = []
    Ttmp_mn = []
    for iw in range(nw): 
        Ttmp_pl.append( ww[iw] / (1.0 - math.exp(-beta*ww[iw])) )
        Ttmp_mn.append( Ttmp_pl[iw] * math.exp(-beta*ww[iw]) )

    with open(Ttmp_file, "w") as file_:
        for iw in range(nw-1, 1, -1):
            file_.write("{:.12f}    {:.12f}\n".format(-ww[iw], Ttmp_mn[iw]))
        file_.write("{:.12f}    {:.12f}\n".format(float(0), Ttmp_0))
        for iw in range(nw):
            file_.write("{:.12f}    {:.12f}\n".format(ww[iw], Ttmp_pl[iw]))

    return Ttmp_pl, Ttmp_mn, Ttmp_0
##################################################################################################
# Get Ktmp(w) = int int dt^prime * dt^prime^prime * exp(-i*w*(t^prime - t^prime^prime))  in [a.u.]
# and Ktmp_0 = lim Ktmp(w) at w \to 0
# *_pl for w > 0
# *_mn for w < 0
##################################################################################################
def get_Ktmp(dt, ww, ndk_mem, re_Ktmp_file, im_Ktmp_file, abs_Ktmp_file):
    Ktmp_0 = [ float(0.5) * dt**2 ]

    nw = len(ww)
    Ktmp_pl = [ [] ]
    Ktmp_mn = [ [] ]
    for iw in range(nw): 
        Ktmp_pl[0].append( (1.0 - cmath.exp(complex(0.0, -ww[iw]*dt))) / ww[iw]**2 )
        Ktmp_mn[0].append( Ktmp_pl[0][iw].conjugate() )

    for dk in range(1, ndk_mem):
        Ktmp_0.append( float(1) * dt**2 )

        Ktmp_pl.append([])
        Ktmp_mn.append([])
        for iw in range(nw): 
            Ktmp_pl[dk].append( 4.0 * math.sin(0.5*ww[iw]*dt)**2 * cmath.exp(complex(0.0, -ww[iw]*dt*dk)) / ww[iw]**2 )
            Ktmp_mn[dk].append( Ktmp_pl[dk][iw].conjugate() )

    with open(re_Ktmp_file, "w") as file_:
        dk = 0
        for iw in range(nw-1, 1, -1):
            file_.write("{:.12f}    {:.12f}\n".format(-ww[iw], Ktmp_mn[dk][iw].real))
        file_.write("{:.12f}    {:.12f}\n".format(float(0), Ktmp_0[dk].real))
        for iw in range(nw):
            file_.write("{:.12f}    {:.12f}\n".format(ww[iw], Ktmp_pl[dk][iw].real))

    with open(im_Ktmp_file, "w") as file_:
        dk = 0
        for iw in range(nw-1, 1, -1):
            file_.write("{:.12f}    {:.12f}\n".format(-ww[iw], Ktmp_mn[dk][iw].imag))
        file_.write("{:.12f}    {:.12f}\n".format(float(0), Ktmp_0[dk].imag))
        for iw in range(nw):
            file_.write("{:.12f}    {:.12f}\n".format(ww[iw], Ktmp_pl[dk][iw].imag))

    with open(abs_Ktmp_file, "w") as file_:
        dk = 0
        for iw in range(nw-1, 1, -1):
            file_.write("{:.12f}    {:.12f}\n".format(-ww[iw], abs(Ktmp_mn[dk][iw])))
        file_.write("{:.12f}    {:.12f}\n".format(float(0), abs(Ktmp_0[dk])))
        for iw in range(nw):
            file_.write("{:.12f}    {:.12f}\n".format(ww[iw], abs(Ktmp_pl[dk][iw])))

    return Ktmp_pl, Ktmp_mn, Ktmp_0
##################################################################################################
# Get eta(w) = (1/pi) int dw * Jtmp(w) * Ttmp(w) * Ktmp(w)  in [a.u.]
# and Ktmp_0 = lim Ktmp(w) at w \to 0
# *_pl for w > 0
# *_mn for w < 0
##################################################################################################
def get_eta_diag(ww, T, ndk_mem, Jtmp_pl, Jtmp_mn, Jtmp_0, Ttmp_pl, Ttmp_mn, Ttmp_0, Ktmp_pl, Ktmp_mn, Ktmp_0, re_eta_file, im_eta_file, abs_eta_file):
    nw = len(ww)
    dw = ww[2]-ww[1]
    beta = 1.0 / (k_B_au * T) # [1/H]

    eps = 0.5*dw
    eta = []
    for dk in range(ndk_mem):
        eta.append(complex(0.0, 0.0))
        for iw in range(nw):
            eta[dk] += Jtmp_pl[iw] * Ttmp_pl[iw] * Ktmp_pl[dk][iw]
            eta[dk] += Jtmp_mn[iw] * Ttmp_mn[iw] * Ktmp_mn[dk][iw]
        cnst = dw / np.pi
        eta[dk] *= cnst
        eta[dk] += 2 * Jtmp_0 * eps * Ktmp_0[dk] * Ttmp_0

    abs_eta = []
    re_eta = []
    im_eta = []
    for dk in range( ndk_mem ):
        abs_eta.append( abs( eta[dk] ) )
        re_eta.append( eta[dk].real )
        im_eta.append( eta[dk].imag )

    with open(abs_eta_file, "w") as file_:
        for dk in range(ndk_mem):
            file_.write("{:d}    {:.12f}\n".format(dk, abs_eta[dk]))

    with open(re_eta_file, "w") as file_:
        for dk in range(ndk_mem):
            file_.write("{:d}    {:.12f}\n".format(dk, re_eta[dk]))

    with open(im_eta_file, "w") as file_:
        for dk in range(ndk_mem):
            file_.write("{:d}    {:.12f}\n".format(dk, im_eta[dk]))

    return eta, abs_eta, re_eta, im_eta
##################################################################################################
# Get eta coefficients
##################################################################################################
def get_eta_jj_jjp(ww, spd, T, dt, ndk_mem, abs_eta_file, re_eta_file, im_eta_file, Jtmp_0, delta, Ttmp_file, Jtmp_file):
    nw = len(ww)

    beta = 1 / (k_B_au * T) # [1/H]

    cnst = 2.0 / np.pi
    Ttmp_pl = []
    Ttmp_mn = []
    for iw in range(nw):
        Ttmp_pl.append(float( cnst * (math.sin(0.5*ww[iw]*dt) / ww[iw]) * (2 * math.sin(0.5*ww[iw]*dt) / (1-np.exp(-beta*ww[iw])) ) ))
        Ttmp_mn.append(float( cnst * (math.sin(0.5*ww[iw]*dt) / ww[iw]) * (2 * math.sin(-0.5*ww[iw]*dt) / (1-np.exp(beta*ww[iw])) ) ))

    Ttmp_0 = pow(dt, 2) / beta / np.pi

    with open(Ttmp_file, "w") as file_:
        for iw in range(nw-1, 1, -1):
            file_.write("{:.6f}    {:.6f}\n".format(-ww[iw] / delta, Ttmp_mn[iw] * delta))
        for iw in range(nw):
            file_.write("{:.6f}    {:.6f}\n".format(ww[iw] / delta, Ttmp_pl[iw] * delta))

    Jtmp = []
    for iw in range(nw):
        Jtmp.append(float( spd[iw] / ww[iw] ))

    with open(Jtmp_file, "w") as file_:
        for iw in range(nw-1, 1, -1):
            file_.write("{:.6f}    {:.6f}\n".format(-ww[iw] / delta, Jtmp[iw]))
        for iw in range(nw):
            file_.write("{:.6f}    {:.6f}\n".format(ww[iw] / delta, Jtmp[iw]))

    eta_jj_jjp = [complex(0, 0)]
    for iw in range(1, nw-1):
        eta_jj_jjp[0] += Jtmp[iw] * Ttmp_pl[iw] * (ww[iw+1]-ww[iw]) / (1-cmath.exp(complex(0, ww[iw]*dt))) 
        eta_jj_jjp[0] += Jtmp[iw] * Ttmp_mn[iw] * (ww[iw+1]-ww[iw]) / (1-cmath.exp(complex(0, -ww[iw]*dt))) 

    print("eta_jj_jjp[0]: ", eta_jj_jjp[0])

    for dj in range(1, ndk_mem):
        eta_jj_jjp.append(complex(2.0 * (ww[1]-ww[0]) * Jtmp_0 * Ttmp_0, 0))
        for iw in range(1, nw-1):
            eta_jj_jjp[dj] += Jtmp[iw] * Ttmp_pl[iw] * cmath.exp(complex(0, -ww[iw]*dt*dj)) * (ww[iw+1]-ww[iw])
            eta_jj_jjp[dj] += Jtmp[iw] * Ttmp_mn[iw] * cmath.exp(complex(0, +ww[iw]*dt*dj)) * (ww[iw+1]-ww[iw])

#    const = eta_jj_jjp[0]
#    for dj in range(ndk_mem):
#        eta_jj_jjp[dj] /= abs(const)

    abs_eta = []
    re_eta = []
    im_eta = []
    for i in range( len(eta_jj_jjp) ):
        abs_eta.append( abs( eta_jj_jjp[i] ) )
        re_eta.append( eta_jj_jjp[i].real )
        im_eta.append( eta_jj_jjp[i].imag )

    with open(abs_eta_file, "w") as file_:
        for dj in range(ndk_mem):
            file_.write("{:d}    {:.6f}\n".format(dj, abs_eta[dj]))

    with open(re_eta_file, "w") as file_:
        for dj in range(ndk_mem):
            file_.write("{:d}    {:.6f}\n".format(dj, re_eta[dj]))

    with open(im_eta_file, "w") as file_:
        for dj in range(ndk_mem):
            file_.write("{:d}    {:.6f}\n".format(dj, im_eta[dj]))

    return eta_jj_jjp, abs_eta, re_eta, im_eta
##################################################################################################
# integrate eta
##################################################################################################
def integrate_eta(abs_eta, mask):
    KK = len(abs_eta)
    MM = len(mask)

    if KK == MM:
        int_eta_abs = float(0)
        for i in range(KK):
            int_eta_abs += abs_eta[i]
    else:
        int_eta_abs = abs_eta[ mask[MM-1] ] * (KK - mask[MM-1])
        for i in range(MM-1):
            di = mask[i+1] - mask[i] 
            int_eta_abs += abs_eta[ mask[i] ] * di

    return int_eta_abs
##################################################################################################
# find masks
##################################################################################################
def find_masks(MM, KK, to_include, abs_eta):

    if MM == 0:
        return [], 100.0

    int_eta_abs = integrate_eta(abs_eta, range(KK))

    masks = list( itertools.combinations(range(KK), MM) )
    print("Total number of trials: {}".format( len(masks) ))
    print()

    trial = [int(0)]*MM
    diff_eta_abs = float(1000000.0)
    imask_abs = -1
    for imask in range( len(masks) ):

        flg = False
        for i in range( len(to_include) ):
            if to_include[i] not in masks[imask]: 
                flg = True
                break

        if flg:
            continue
        else:
            trial[:] = masks[imask][:]

        int_eta_trial_abs = integrate_eta(abs_eta, trial)
        diff_eta_trial_abs = abs(int_eta_abs - int_eta_trial_abs)

        if diff_eta_trial_abs < diff_eta_abs:
            imask_abs = imask
            diff_eta_abs = diff_eta_trial_abs

    if imask_abs == -1:
        print("Wrong imask")

    diff_eta_abs = 100*diff_eta_abs/int_eta_abs

    return list(masks[imask_abs]), diff_eta_abs
##################################################################################################
# Plot spectral density
##################################################################################################
def get_spectral_density(sd_func, nw, wmax, sd_file):
    dw = wmax / nw 
    ww = [float((iw+0.5)*dw) for iw in range(nw)]
    spd = [float(sd_func(ww[iw])) for iw in range(nw)]
#    spd.insert(0, 0.0)

    str_ = "{:24.16e}        {:24.16e}\n"
    with open(sd_file, "w") as file_:
        for iw in range(nw):
            file_.write(str_.format(ww[iw], spd[iw]))

    return ww, spd
