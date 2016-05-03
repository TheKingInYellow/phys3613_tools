#!/usr/env/python2.7
#! encoding: utf-8

"""
Filename = conductionAnalysis.py
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

PATH2DAT="data/partA/"


if __name__ == '__main__':

    path2dat=PATH2DAT
    files = [path2dat + fname for fname in os.listdir(path2dat)]

    time = {}
    temp = {}

    for fname in files:
        print fname

        T = str(fname[12])
        if "trial" in fname:
            trial = 2
        else:
            trial = 1
        ryan = str(fname[-5])

        print T, ryan, trial
            
        time["T%st%ir%s"%(T,trial,ryan)], temp["T%st%ir%s"%(T,trial,ryan)] \
        = np.genfromtxt(fname, skip_header=2, unpack=True, dtype=float)

    print time.keys()
    print len(time)

    diff_8722 = np.subtract(temp['T7t2r2'], temp['T8t2r2'])
    diff_8721 = np.subtract(temp['T7t2r1'], temp['T8t2r1'])
    diff_8712 = np.subtract(temp['T7t1r2'], temp['T8t1r2'])
    diff_8711 = np.subtract(temp['T7t1r1'], temp['T8t1r1'])
    diff_6522 = np.subtract(temp['T6t2r2'], temp['T5t2r2'])
    diff_6521 = np.subtract(temp['T6t2r1'], temp['T5t2r1'])
    diff_6512 = np.subtract(temp['T6t1r2'], temp['T5t1r2'])
    diff_6511 = np.subtract(temp['T6t1r1'], temp['T5t1r1'])
    diff_4322 = np.subtract(temp['T3t2r2'], temp['T4t2r2'])
    diff_4321 = np.subtract(temp['T3t2r1'], temp['T4t2r1'])
    diff_4312 = np.subtract(temp['T3t1r2'], temp['T4t1r2'])
    diff_4311 = np.subtract(temp['T3t1r1'], temp['T4t1r1'])
    diff_2122 = np.subtract(temp['T2t2r2'], temp['T1t2r2'])
    diff_2121 = np.subtract(temp['T2t2r1'], temp['T1t2r1'])
    diff_2112 = np.subtract(temp['T2t1r2'], temp['T1t1r2'])
    diff_2111 = np.subtract(temp['T2t1r1'], temp['T1t1r1'])

    print len(diff_8722), len(diff_8721), len(diff_8712), len(diff_8711)

    print np.mean([diff_8711[-1],diff_8712[-1],diff_8721[-1],diff_8722[-1]])
    print np.mean([diff_6511[-1],diff_6512[-1],diff_6521[-1],diff_6522[-1]])
    print np.mean([diff_4311[-1],diff_4312[-1],diff_4321[-1],diff_4322[-1]])
    print np.mean([diff_2111[-1],diff_2112[-1],diff_2121[-1],diff_2122[-1]])
    print np.std([diff_8711[-1],diff_8712[-1],diff_8721[-1],diff_8722[-1]])
    print np.std([diff_6511[-1],diff_6512[-1],diff_6521[-1],diff_6522[-1]])
    print np.std([diff_4311[-1],diff_4312[-1],diff_4321[-1],diff_4322[-1]])
    print np.std([diff_2111[-1],diff_2112[-1],diff_2121[-1],diff_2122[-1]])


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(time["T2t1r1"], diff_2111, 'o', color="darkred")
    # savename="plots/c_" + "T2t1r1.png"
    # ax.grid(True)
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("$\Delta$T ($^\circ$C)")

    # plt.savefig(savename)
     
