#!/usr/env/python2.7
#! encoding: utf-8

"""
Filename = heatEngineAnalysis.py
Author = Kody Crowell
Version = 1.0
"""

import sys, os
import numpy as np
import scipy.signal as ssig
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

PATH2DAT="data/heateng/"


def using_clump(a):
    return np.asarray([a[s] for s in \
                       np.ma.clump_unmasked(np.ma.masked_invalid(a))])


if __name__ == '__main__':

    path2dat=PATH2DAT
    files = [path2dat + fname for fname in os.listdir(path2dat)]

    diffsP=[]; diffsD=[]; err=[]; 

    for fname in files:
        print fname
        time, pres, temp, angle, vel, dist = np.genfromtxt(fname, \
                    skip_header=1, delimiter=",", unpack=True, dtype=float)

        nan = np.nan
        time = using_clump(time)
        pres = using_clump(pres)
        temp = using_clump(temp)
        angle = using_clump(angle)
        vel = using_clump(vel)
        dist = using_clump(dist)

        meanP = np.asarray([np.mean(i) for i in pres])
        meanT = np.asarray([np.mean(i) for i in temp])
        meanD = np.asarray([np.mean(i) for i in dist])

        print temp.size
        print meanT.size
