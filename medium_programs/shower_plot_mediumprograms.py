# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Plot only program.

import matplotlib.pyplot as plt
import numpy as np

    
def medium_analytical_comparison(filename, scale):
    filelocation = "data\\parton_shower_gluons_medium_data\\"
    destination = filelocation + filename + ".npz"
    file = np.load(destination)
    
    n          = file["n"]
    tauvalues  = file["tauvalues"]
    xlinrange     = file["xlinrange"]
    linsolutions  = file["linsolutions"]
    linbinlist    = file["linbinlist"]
    gluonlinlists = file["gluonlinlists"]
    gluonlinhards = file["gluonlinhards"]
    xlogrange     = file["xlogrange"]
    logsolutions  = file["logsolutions"]
    logbinlist    = file["logbinlist"]
    gluonloglists = file["gluonloglists"]
    gluonloghards = file["gluonloghards"]

    
    
    # Do the actual plotting. 
    plt.figure(dpi=300, figsize= (6,5)) #(w,h) figsize= (10,3)

    plt.rc('axes', titlesize="small" , labelsize="x-small")     # fontsize of the axes title and labels.
    plt.rc('xtick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('ytick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('legend',fontsize='xx-small')    # fontsize of the legend labels.
    plt.rc('lines', linewidth=0.8)

    ax1 = plt.subplot(221) #H B NR
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    axes = [ax1, ax2, ax3, ax4]

    for ax in axes:
        index = axes.index(ax)
        
        if scale == "lin":
            ax.plot(linbinlist, gluonlinlists[index], "--", label="MC")
            ax.plot(xlinrange, linsolutions[index], 'r', label="solution")
            ax.set_xscale("linear")
            ax.set_yscale("log")
            ax.set_xlim(0,1)

        elif scale == "log":
            ax.plot(logbinlist, gluonloglists[index], "--", label="MC")
            ax.plot(xlogrange, logsolutions[index], 'r', label="solution")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.001,1)        
            
        ax.set_title('tau = ' +str(tauvalues[index]))
        ax.set_ylim(0.01,10)
        ax.set_xlabel('$z$')
        ax.set_ylabel('$D(x,t)$')
        ax.grid(linestyle='dashed', linewidth=0.2)
        ax.legend(loc="lower left")
        
        textstring = '$n={%i}$'%n
        ax.text(0.05, 0.2, textstring, fontsize = "xx-small",
                horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
        
    plt.tight_layout()
    plt.show()
    print("\rDone!" + 10*" ")    