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
    plt.figure(dpi=500, figsize= (6,5)) #(w,h)

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
        
        
        textstring = ("$N_{z}= $" + str(round(gluontzs[index][0])) +
                      "\n$n =$" + str(n) + "\n" )
        ax.text(0.75, 0.15, textstring, fontsize = "xx-small", #bbox=dict(facecolor='white', alpha=0.5),
                horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
                
    plt.tight_layout()
    plt.show()
    print("\rDone!")    
    
    
def vacuum_analytical_comparison(filename, scale):
    filelocation = "data\\parton_shower_gluons_vacuum_data\\"
    destination = filelocation + filename + ".npz"
    file = np.load(destination)
    
    n          = file["n"]
    tvalues    = file["tvalues"]
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
    
    # Plot    
    plt.figure(dpi=300, figsize= (6,5)) #(w,h) figsize= (10,3)

    plt.rc('axes', titlesize="small" , labelsize="x-small")
    plt.rc('xtick', labelsize="x-small")
    plt.rc('ytick', labelsize="x-small")
    plt.rc('legend',fontsize='xx-small')
    plt.rc('lines', linewidth=0.8)

    ax1 = plt.subplot(221) #H B NR
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    axes = [ax1, ax2, ax3, ax4]
        
    for ax in axes:
        index = axes.index(ax)
        
        if scale == "lin":
            ax.plot(linbinlist, gluonlinlists[index], 'b--', label ="MC")
            #ax.plot(linbinlist, gluonlinhards[index], 'b:')
            ax.plot(xlinrange, linsolutions[index], 'r', label="solution")
            ax.set_xscale("linear")
            ax.set_yscale("log")
            ax.set_xlim(0,1)
            
        elif scale == "log":
            ax.plot(logbinlist, gluonloglists[index], 'b--', label ="MC")
            #ax.plot(logbinlist, gluonloghards[index], 'b:')
            ax.plot(xlogrange, logsolutions[index], 'r', label="solution")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.001,1)
        
        ax.set_title('t = ' + str(tvalues[index]))
        ax.set_xlabel('$z$')
        ax.set_ylabel('$D(x,t)$')
        ax.grid(linestyle='dashed', linewidth=0.2)
        ax.legend(loc='lower right')
        ax.set_ylim(0.01,10)

        textstring = '$n={%i}$'%n
        ax.text(0.85, 0.19, textstring, fontsize = "xx-small",
                horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        

    plt.tight_layout()
    plt.show()
    print("\rDone!" + 10*" ")    
    
    
def program_comparison(filename, scale):
    filelocation = "data\\vacuum_program_comparison_data\\"
    destination = filelocation + filename + ".npz"
    file = np.load(destination)    
    
    n        = file["n"]
    C_A      = file["C_A"]
    tvalues      = file["tvalues"]
    xlinrange    = file["xlinrange"]
    linsolutions = file["linsolutions"]
    linbinlist   = file["linbinlist"]
    gglinlists   = file["gglinlists"]
    gglinhards   = file["gglinhards"]
    gqlinlists   = file["gqlinlists"]
    gqlinhards   = file["gqlinhards"]
    
    xlogrange    = file["xlogrange"]
    logsolutions = file["logsolutions"]
    logbinlist   = file["logbinlist"]
    ggloglists   = file["ggloglists"]
    ggloghards   = file["ggloghards"]
    gqloglists   = file["gqloglists"]
    gqloghards   = file["gqloghards"]
    
    
    # Plot    
    plt.figure(dpi=300, figsize= (6,5)) #(w,h) figsize= (10,3)

    plt.rc('axes', titlesize="small" , labelsize="x-small")
    plt.rc('xtick', labelsize="x-small")
    plt.rc('ytick', labelsize="x-small")
    plt.rc('legend',fontsize='xx-small')
    plt.rc('lines', linewidth=0.8)

    ax1 = plt.subplot(221) #H B NR
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    axes = [ax1, ax2, ax3, ax4]
    
    print("\rPlotting..." + 10*" ", end="")
    
    for ax in axes:
        index = axes.index(ax)
        
        if scale == "lin":
            ax.plot(linbinlist, gglinlists[index], 'b--', label ="gluons")
            ax.plot(linbinlist, gqlinlists[index], 'g--', label="quarks & gluons")
            ax.plot(linbinlist, gglinhards[index], 'b:')
            ax.plot(linbinlist, gqlinhards[index], 'g:')
            ax.plot(xlinrange, linsolutions[index], 'r', label="solution gluons")
            ax.set_xscale("linear")
            ax.set_yscale("log")
            ax.set_xlim(0,1)

        elif scale == "log":
            ax.plot(logbinlist, ggloglists[index], 'b--', label ="gluons")
            ax.plot(logbinlist, gqloglists[index], 'g--', label="quarks & gluons")
            ax.plot(logbinlist, ggloghards[index], 'b:')
            ax.plot(logbinlist, gqloghards[index], 'g:')
            ax.plot(xlogrange, logsolutions[index], 'r', label="solution gluons")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.001,1)


        ax.set_title('t = ' + str(tvalues[index]))
        ax.set_ylim(0.01,10)
        ax.set_xlabel('z ')
        ax.set_ylabel('D(x,t)')
        ax.grid(linestyle='dashed', linewidth=0.2)
        ax.legend(loc="lower right")
        
        textstring = '$n={%i}$'%n + "\n$C_A={%f}$"%C_A
        ax.text(0.8, 0.3, textstring, fontsize = "xx-small",
                horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        
    plt.tight_layout()
    plt.show()
    print("\rDone!" + 10*" ")
