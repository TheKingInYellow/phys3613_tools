#!/usr/env/python2.7
#! encoding: utf-8

"""
Filename = adiabatic_Analysis.py
Author = Kody Crowell
Version = 1.0

Plots the pressure vs. the volume and determines gamma for air. Numerically integrates over the P-V data for each lever lifting and calculates the experimental work done. Computes the theoretical value.
"""

import sys, os
import numpy as np
import scipy.signal as ssig
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

PATH2DAT="data/part2/"
DIAM_OF_CYLINDER=.448 #mm

if __name__ == '__main__':

    path2dat=PATH2DAT
    files = [path2dat + file for file in os.listdir(path2dat)]
    t_data=[]; d_data=[]; p_data=[]
    for file in files:
        if "temperature_" in file:
            t_data.append(file)
        elif "pressure_" in file:
            p_data.append(file)
        elif "displacement_" in file:
            d_data.append(file)
    gamma=[];k=[];err=[]
    for i in xrange(1,len(t_data)):
        t, temp = np.loadtxt(path2dat+"temperature_"+str(i)+".txt", \
                skiprows=2, unpack=True)
        td, disp = np.loadtxt(path2dat+"displacement_"+str(i)+".txt", \
                skiprows=2, unpack=True)
        tp, pres = np.loadtxt(path2dat+"pressure_"+str(i)+".txt", \
                skiprows=2, unpack=True)

        # skip the bad trials
        if i in [1,2,6,7,8,9,10,20,22,23]:
            continue

        print "count: ", i, "\nlen (d,T,P): ", len(disp), len(temp), len(pres)

        # calibration of data from voltage according to manufacturer
        # convert Pa -> kPa and m -> mm for plotting
        pres = 100*pres*1000                #Pa
        temp = 54.1*temp + 325         #K
        disp = (20*disp + 55)*0.001          #m

        # displacement to volume
        r = DIAM_OF_CYLINDER/2*1000        #m
        vol = disp*np.pi*r*r

        #print vol
        #print pres
        lnV = np.log(vol)
        lnP = np.log(pres)
        print "length of lnV, lnP: ", len(lnV), len(lnP)

        # y*ln(V)+ln(P)=k
        # ln(P) = k - y*ln(V)
        # print min(vol), max(vol), min(pres), max(pres)

        # ignores trailing points
        inds = np.where(np.diff(lnV) < 0.0001)
        lnP = lnP[inds]
        lnV = lnV[inds]
        inds = np.squeeze(np.where(np.diff(lnV)>0.001))
        print "offending index: ", inds
        lnP = lnP[inds+1:]
        lnV = lnV[inds+1:]
        print "new lengths (lnV, lnP): ", len(lnV), len(lnP)
        #print lnV
        #print lnP
        # fig, ax = plt.subplots()
        # ax.loglog(lnV, lnP, 'ro', basex=np.e, basey=np.e)
        # ax.grid(True)
        # ax.set_title("Experimental log-log plot of pressure vs. volume")
        # ax.set_xlim([np.amin(lnV), np.amax(lnV)])
        # plt.show()
        # savename = "plots/new_plot_" + str(i) + ".png"
        # plt.savefig(savename)

        # statistics from linear regression
        # lnP = [np.amin(lnP), np.amax(lnP)]
        # lnV = [np.amin(lnV), np.amax(lnV)]
        slope, intercept, r, p, std_err = ss.stats.linregress(lnV, lnP)
        print "gamma: ", -slope, "\nk: ", intercept
        print "r-value: ", r, "\nstd. err: ", std_err
        print "---------"

        if np.isnan(slope) or np.isnan(intercept):
            continue

        gamma.append(-slope)
        k.append(intercept)
        err.append(std_err)

        #if i is 3:
        #    break

    gamma_mean = np.mean(gamma)
    gamma_sem = ss.sem(gamma)
    k_mean = np.mean(k)
    k_sem = ss.sem(k)

    print "mean(gamma): ", gamma_mean, "\nSE_gamma: ", gamma_sem, \
            "\nmean(k): ", k_mean, "\nSE_k: ", k_sem
