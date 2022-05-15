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
            
        ax.set_title('$\\tau = $' +str(tauvalues[index]))
        ax.set_ylim(0.01,10)
        ax.set_xlabel('$z$')
        ax.set_ylabel('$D(x,\\tau)$')
        ax.grid(linestyle='dashed', linewidth=0.2)
        ax.legend(loc="lower left")
        
        textstring = '$n={%i}$'%n
        ax.text(0.05, 0.2, textstring, fontsize = "xx-small",
                horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
        
    plt.tight_layout()
    plt.show()
    print("\rDone!" + 10*" ")    
    
def medium_scaling_comparison(filename):
    filelocation = "data\\parton_shower_gluons_medium_scaling_data\\"
    destination = filelocation + filename + ".npz"
    file = np.load(destination)
    
    n          = file["n"]
    tauvalues  = file["tauvalues"]
    logbinlist    = file["logbinlist"]
    gluonloglists = file["gluonloglists"]
    gluonloghards = file["gluonloghards"]
    hardestbranches = file["hardestbranches"].tolist()
    branchloghards      = file["branchloghards"]
    nonbranchloghards   = file["nonbranchloghards"]
    
    for hard in hardestbranches:
        index = hardestbranches.index(hard)
        percentage = round(hard*(100/n),2)
        print("For tau = ", tauvalues[index],
              ". Percentage of hardest on branch: ", percentage)
    
    # Do the actual plotting. 
    plt.figure(dpi=300, figsize= (6,5)) #(w,h) figsize= (10,3)

    plt.rc('axes', titlesize="small" , labelsize="x-small")
    plt.rc('xtick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('ytick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('legend',fontsize='xx-small')    # fontsize of the legend labels.
    plt.rc('lines', linewidth=0.8)

    ax = plt.subplot(111) #H B NR
    

    ax.plot(logbinlist, gluonloglists[0], "--", label="$\\tau = $"+str(tauvalues[0]))
    ax.plot(logbinlist, gluonloglists[1], "--", label="$\\tau = $"+str(tauvalues[1]))
    ax.plot(logbinlist, gluonloglists[2], "--", label="$\\tau = $"+str(tauvalues[2]))
    ax.plot(logbinlist, gluonloglists[3], "--", label="$\\tau = $"+str(tauvalues[3]))
    ax.plot(logbinlist, gluonloglists[4], "--", label="$\\tau = $"+str(tauvalues[4]))
    ax.plot(logbinlist, gluonloglists[5], "--", label="$\\tau = $"+str(tauvalues[5]))
    ax.plot(logbinlist, gluonloglists[6], "--", label="$\\tau = $"+str(tauvalues[6]))
    ax.plot(logbinlist, gluonloglists[7], "--", label="$\\tau = $"+str(tauvalues[7]))
    
    ax.set_xlim(0.001,1)
    ax.set_ylim(0.01,10)
    ax.set_xlabel('z ')
    ax.set_ylabel('$D(x,\\tau)$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(linestyle='dashed', linewidth=0.2)
    ax.legend(loc="lower left")
    
    textstring = '$n={%i}$'%n
    ax.text(0.02, 0.25, textstring, fontsize = "xx-small",
            horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
        
    
    plt.tight_layout()
    plt.show()
    print("\rDone!")   
    
    
    
def medium_leading_branches(filename, scale):
    filelocation = "data\\parton_shower_gluons_medium_data\\"
    destination = filelocation + filename + ".npz"
    file = np.load(destination)
    
    n          = file["n"]
    tauvalues  = file["tauvalues"].tolist()
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
    hardestbranches = file["hardestbranches"].tolist()
    branchlinhards      = file["branchlinhards"]
    nonbranchlinhards   = file["nonbranchlinhards"]
    branchloghards      = file["branchloghards"]
    nonbranchloghards   = file["nonbranchloghards"]
    
    for hard in hardestbranches:
        index = hardestbranches.index(hard)
        percentage = hard*(100/n)
        print("For tau = ", tauvalues[index],
              ". Percentage of hardest parton on branch: ", percentage)

    
    # Do the actual plotting. 
    plt.figure(dpi=300, figsize= (8,6)) #(w,h) figsize= (10,3)

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
            ax.plot(linbinlist, branchlinhards[index], ':', label="Hardest on branch")
            ax.plot(linbinlist, nonbranchlinhards[index], ':', label="Hardest not on branch")
            ax.set_xscale("linear")
            ax.set_yscale("log")
            ax.set_xlim(0,1)

        elif scale == "log":
            ax.plot(logbinlist, gluonloglists[index], "b--", label="MC")
            ax.plot(xlogrange, logsolutions[index], 'r', label="solution")
            ax.plot(logbinlist, branchloghards[index], ':', label="branchard")
            ax.plot(logbinlist, nonbranchloghards[index], ':', label="nonbranchhard")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.001,1)        
            
        ax.set_title('$\\tau = $' +str(tauvalues[index]))
        ax.set_ylim(0.01,10)
        ax.set_xlabel('$z$')
        ax.set_ylabel('$D(x,\\tau)$')
        ax.grid(linestyle='dashed', linewidth=0.2)
        ax.legend(loc="upper left")
        
        textstring = "Hardest on leading branch:${%i}$"%round(hardestbranches[index]*(100/n),2) +"%" + '\n$n={%i}$'%n
        ax.text(0.4, 0.95, textstring, fontsize = "xx-small",
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        
    plt.tight_layout()
    plt.show()
    print("\rDone!" + 10*" ")   
    
def medium_leading_parton_test(filename, scale):
    filelocation = "data\\parton_shower_gluons_medium_data\\"
    destination = filelocation + filename + ".npz"
    file = np.load(destination)
    
    n          = file["n"]
    tauvalues  = file["tauvalues"].tolist()
    xlinrange     = file["xlinrange"]
    linbinlist    = file["linbinlist"]
    gluonlinhards = file["gluonlinhards"]
    xlogrange     = file["xlogrange"]
    logbinlist    = file["logbinlist"]
    gluonloghards = file["gluonloghards"]
    BDMPSlinsolutions = file["linsolutions"]
    BDMPSlogsolutions = file["logsolutions"]

    
    leadingsolutions = [[],[],[],[]]
    BDMPSsolutions = [[],[],[],[]]


    if scale == "lin":
        xrange = xlinrange
    elif scale == "log":
        xrange = xlogrange
    else:
        print("ERROR - no scale")
        return
    
    for tau in tauvalues:
        index = tauvalues.index(tau)
        for x in xrange:
            leading = ((tau)/((1-x)**(3/2)) )* np.exp(-np.pi*((tau**2)/(1-x)))
            
            if x > 0.5:
                BDMPS = ((tau)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau**2)/(1-x)))
            else:
                BDMPS = None
            BDMPSsolutions[index].append(BDMPS)
            leadingsolutions[index].append(leading)

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
            ax.plot(linbinlist, gluonlinhards[index], "--", label="MC")
            ax.plot(xlinrange, BDMPSlinsolutions[index], label="BDMPS sol")
            ax.set_xscale("linear")
            ax.set_yscale("log")
            ax.set_xlim(0,1)

        elif scale == "log":
            ax.plot(logbinlist, gluonloghards[index], "--", label="MC")
            ax.plot(xlogrange, BDMPSlogsolutions[index], label="BDMPS sol")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.001,1)   
            
        ax.plot(xrange, leadingsolutions[index], 'r', label="Leading sol")
        ax.plot(xrange, BDMPSsolutions[index], 'c', label="BDMPS alt sol")

        ax.set_title('$\\tau = $' +str(tauvalues[index]))
        ax.set_ylim(0.01,10)
        ax.set_xlabel('$z$')
        ax.set_ylabel('$D(x,\\tau)$')
        ax.grid(linestyle='dashed', linewidth=0.2)
        ax.legend(loc="upper left")
        
        textstring = '$n={%i}$'%n
        ax.text(0.05, 0.75, textstring, fontsize = "xx-small",
                horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
        
    plt.tight_layout()
    plt.show()
    print("\rDone!" + 10*" ")  
    

    