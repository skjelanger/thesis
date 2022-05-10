# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Plot only program.

import matplotlib.pyplot as plt
import numpy as np



def dasgupta_plot(filename):
    filelocation = "data\\parton_shower_vacuum_data\\"
    destination = filelocation + filename + ".npz"
    file = np.load(destination)
    
    n           = file["n"]
    tvalues     = file["tvalues"]
    binlist     = file["binlist"]
    gluonlists = [file["gluonlist1"], 
                  file["gluonlist2"], 
                  file["gluonlist3"],
                  file["gluonlist4"]]
    gluonhards = [file["gluonhard1"], 
                  file["gluonhard2"], 
                  file["gluonhard3"],
                  file["gluonhard4"]]
    quarklists = [file["quarklist1"], 
                  file["quarklist2"], 
                  file["quarklist3"],
                  file["quarklist4"]]
    quarkhards = [file["quarkhard1"], 
                  file["quarkhard2"], 
                  file["quarkhard3"],
                  file["quarkhard4"]]
    gluontzs   = [file["gluontz1"],
                  file["gluontz2"],
                  file["gluontz3"],
                  file["gluontz4"]]

    # Now starting the plotting.
    plt.figure(dpi=1000, figsize= (6,5)) #(w,h)

    #plt.suptitle(title)
    plt.rc('axes', titlesize="small" , labelsize="x-small")
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

        ax.plot(binlist, gluonlists[index], 'g-', label ="gluon")
        ax.plot(binlist, gluonhards[index], 'g--')
        ax.plot(binlist, quarklists[index], 'b-', label ="quark")
        ax.plot(binlist, quarkhards[index], 'b--')
        ax.set_yscale("log")

        ax.set_title("$t = $" + str(tvalues[index]))
        ax.set_xlim(0,1)
        ax.set_ylim(0.01,10)
        ax.set_xlabel('$z$')
        ax.set_ylabel('$f(x,t)$')
        ax.grid(linestyle='dashed', linewidth=0.2)
        ax.legend(loc='lower right')
        
        
        textstring = ("$n =$" + str(n) + "\n" + 
                      "$N_{z}= $" + str(round(gluontzs[index][0],index)) + 
                      "$\pm$" + str(round(gluontzs[index][1],index)))
        ax.text(0.96, 0.2, textstring, fontsize = "xx-small", #bbox=dict(facecolor='white', alpha=0.5),
                horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
                
    plt.tight_layout()
    plt.show()
    print("\rDone!")    