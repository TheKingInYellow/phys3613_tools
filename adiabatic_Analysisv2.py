#!/usr/env/python2.7
#! encoding: utf-8

"""
Filename = adiabatic_Analysis.py
Author = Kody Crowell
Version = 2.0

Plots the pressure vs. the volume and determines gamma for air. Numerically integrates over the P-V data for each lever lifting and calculates the experimental work done. Computes the theoretical value.
"""

import sys, os
import numpy as np
import scipy.signal as ssig
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

PATH2DAT="data/files/"
DIAM_OF_CYLINDER=44.8 #mm

if __name__ == '__main__':

    path2dat=PATH2DAT
    files = [path2dat + fname for fname in os.listdir(path2dat)]

    gamma=[]; k=[]; err=[]; work=[]
    gammaT=[]; kT=[]; errT=[]
    for fname in files:
        print fname
        if "exp" in fname:
            disp, temp, pres = np.genfromtxt(fname, skip_header=1, \
                                             missing_values=["N/A"], \
                                             delimiter=",", unpack=True,
                                             dtype=float)
        elif "com" in fname:
            disp, pres, temp = np.genfromtxt(fname, skip_header=1, \
                                             missing_values=["N/A"], \
                                             delimiter=",", unpack=True,
                                             dtype=float)

        disp = disp[~np.isnan(disp)]
        pres = pres[~np.isnan(pres)]
        temp = temp[~np.isnan(temp)]

        if len(disp) > len(pres):
            disp = disp[0:len(pres)]
        
        print "\nlen (d,T,P): ", len(disp), len(temp), len(pres)

        # calibration of data from voltage according to manufacturer
        pres = 100*pres*1000                #Pa
        temp = 54.1*temp + 325              #K
        disp = (20*disp + 55)*0.001         #m

        # displacement to volume
        r = DIAM_OF_CYLINDER/2*0.001        #m
        vol = disp*np.pi*r*r

        lnV = np.log(vol)
        lnP = np.log(pres)
        lnT = np.log(temp)
        print "length of lnV, lnP, lnT: ", len(lnV), len(lnP), len(lnT)

        # linear model with numpy
        fit, cov = np.polyfit(lnV, lnP, 1, cov=True)
        fit_fn = np.poly1d(fit)
        lnP_model = np.polyval(fit, lnV)

        # for temp
        fitT = np.polyfit(lnP, lnT, 1)
        fit_T = np.poly1d(fitT)

        # statistics
        n = lnV.size #number of observations
        m = fit.size #number of parameters
        df = n - m # degrees of freedom
        t = stats.t.ppf(0.95, df) # t statistic for CI and PI bands
        # estimates of error in data / model
        resid = lnP - lnP_model
        chi2 = np.sum((resid/lnP_model)**2) # chi-squared statistic
        chi2_red = chi2/df  # reduced chi-sq measures goodness of fit
        s_err = np.sqrt(np.sum(resid**2)/df) # std. dev of the error

        # statistics from linear regression
        slope, intercept, r, p, std_err = stats.stats.linregress(lnV, lnP)
        print "gamma: ", -slope, "\nk: ", intercept
        print "r-value: ", r, "\nstd. err: ", std_err

        slope2, int2, r2, p2, std_err2 = stats.stats.linregress(lnP,lnT)
        print "----"
        print "gamma: ", 1/(1-slope2), "\nk: ", int2
        print "r-value: ", r2, "\nstd. err: ", std_err2

        if np.isnan(slope) or np.isnan(intercept):
            continue

        gam = -slope
        gam2 = 1/(1-slope2)
        icpt2 = int2
        icpt = intercept
        gamma.append(gam)
        k.append(icpt)
        err.append(std_err)
        gammaT.append(gam2)
        kT.append(icpt2)
        errT.append(std_err2)

        #plotting log log plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.loglog(lnV, lnP, 'o', color="#b9cfe7", alpha=0.1,\
        #          markersize=8, markeredgecolor='b', markerfacecolor='None',\
        #           basex=np.e, basey=np.e)
        # ax.plot(lnV, fit_fn(lnV), color="0.1", linewidth="2")
        # ax.set_yscale("log")
        # ax.set_xscale("log")
        # ax.grid(True)
        # # ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        # # ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
        # ax.set_xlim([np.amin(lnV)-.1, np.amax(lnV)+.1])
        # ax.set_ylim([np.amin(lnP)-.1, np.amax(lnP)+.1])
        # ax.set_xlabel("ln(V)")
        # ax.set_ylabel("ln(P)")
        # savename = "plots/plot_" + fname[-8:-4] + "_log.png"
        # plt.savefig(savename)
        
        # ploting raw data (works)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(vol, pres, 'o', color='#b9cffe7', alpha=0.1, \
        #          markersize=8, markeredgecolor='b', markerfacecolor='None')
        # ax.grid(True)
        # ax.set_xlabel("Volume (m^3)")
        # ax.set_ylabel("Pressure (Pa)")
        # ax.plot(vol, np.exp(icpt)/np.power(vol,gam), '-k', linewidth="2")
        # plt.show()
        # savename = "plots/plot_" + fname[-8:-4] + ".png"
        # plt.savefig(savename)

        #for temperature
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.loglog(lnP, lnT, 'o', color="#b9cfe7", alpha=0.1,\
        #          markersize=8, markeredgecolor='b', markerfacecolor='None',\
        #           basex=np.e, basey=np.e)
        # ax.plot(lnP, fit_T(lnP), color="0.1", linewidth="2")
        # ax.set_yscale("log")
        # ax.set_xscale("log")
        # ax.grid(True)
        # # ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        # # ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
        # ax.set_xlim([np.amin(lnP)-.1, np.amax(lnP)+.1])
        # ax.set_ylim([np.amin(lnT)-.1, np.amax(lnT)+.1])
        # ax.set_xlabel("ln(P)")
        # ax.set_ylabel("ln(T)")
        # savename = "plots/plot_" + fname[-8:-4] + "_temp_log.png"
        # plt.savefig(savename)
        
        #ploting raw data (works)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(pres, temp, 'o', color='#b9cffe7', alpha=0.1, \
        #          markersize=8, markeredgecolor='b', markerfacecolor='None')
        # ax.grid(True)
        # ax.set_xlabel("Pressure (Pa)")
        # ax.set_ylabel("Temperature (K)")
        # savename = "plots/plot_" + fname[-8:-4] + "_temp.png"
        # plt.savefig(savename)


        #numerical integration
        w = integrate.trapz(pres,vol) # J

        pi=np.min(pres)
        vi=np.min(vol)
        vf=np.max(vol)
        # note for theoretical compression, w_theo should be negative
        print pi,vi,vf
        w_theo = pi*np.power(vi,1.4)*(np.power(vf,.4)-np.power(vi,.4))/.4
        work.append(w)
        print "work: %.4g"%(w)
        print "work (theo): %.4g"%(w_theo)

        print '----------------'
        

    gamma_mean = np.mean(gamma)
    gamma_sem = stats.sem(gamma)
    k_mean = np.mean(k)
    k_sem = stats.sem(k)

    print "-----------------"
    print "mean(gamma): ", gamma_mean, "\nSE_gamma: ", gamma_sem, \
            "\nmean(k): ", k_mean, "\nSE_k: ", k_sem

    gamma2_mean = np.mean(gammaT)
    gamma2_sem = stats.sem(gammaT)
    k2_mean = np.mean(kT)
    k2_sem = stats.sem(kT)

    print "\nfor temperature: "
    print "mean(gamma): ", gamma2_mean, "\nSE_gamma: ", gamma2_sem, \
            "\nmean(k): ", k2_mean, "\nSE_k: ", k2_sem

