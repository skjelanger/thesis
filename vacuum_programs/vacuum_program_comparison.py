# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Parton shower program for comparing the results of gluons in vacuum
# and quarks/gluons in vacuum.

import matplotlib.pyplot as plt
import numpy as np
import parton_shower_gluons_vacuum as ggv
import parton_shower_vacuum as gqv


#constants
epsilon = 10**(-3)
z_min = 10**(-5)
plot_lim = 10**(-3)
binnumber = 100


def vacuum_programs_comparison(n, opt_title, scale):
    """
    Runs n parton showers of both gluons in vacuum, and quarks/gluons in 
    vacuum, and compares the results for gluon initiated showers.
    
    Parameters: 
        n (int): Number of showers to simulate.
        opt_title (str): Additional title to add to final plot.
        scale (str): Set lin or log scale for the plot.

    Returns:
        A very nice plot. 
    """
    
    error, error_msg = error_message_several_showers(n, opt_title, scale)
    if error:
        print(error_msg)
        return
    
    R = 0.4 # Jet radius.    
    p_0 = 100 # Initial parton momentum.
    Q_0 = 1 # Hadronization scale.
    
    t1 = 0.04
    t2 = 0.1
    t3 = 0.2
    t4 = 0.3
    tvalues = (t1, t2, t3, t4)

    
    #Generating showers
    gglists = [[], [], [], []]
    gqlists = [[], [], [], []]
        
    for i in range(1,n):
        print("\rLooping... "+ str(round(100*i/(n),1)) + "%",end="")
        Showergg = ggv.generate_shower(tvalues, p_0, Q_0, R, i)
        Showergq = gqv.generate_shower("gluon", tvalues, p_0, Q_0, R, i)
        
        gglists[0].extend(Showergg.FinalFracList1)
        gglists[1].extend(Showergg.FinalFracList2)
        gglists[2].extend(Showergg.FinalFracList3)
        gglists[3].extend(Showergg.FinalFracList4)  
        gqlists[0].extend(Showergq.FinalFracList1)
        gqlists[1].extend(Showergq.FinalFracList2)
        gqlists[2].extend(Showergq.FinalFracList3)
        gqlists[3].extend(Showergq.FinalFracList4)  
        del Showergg
        del Showergq
    
    # Sets the different ranges required for the plots.
    if scale == "lin":
        #linbins1 = (np.linspace(plot_lim, 0.99, num=binnumber))
        #linbins2 = (np.linspace(0.991, 1, num= round((binnumber/4))))
        #bins = np.hstack((linbins1, linbins2))
        bins = np.linspace(plot_lim, 1, num=binnumber)
        xrange = np.linspace(plot_lim, 0.9999, num=(4*binnumber))

    elif scale == "log":
        logbins1 = np.logspace(-3, -0.1, num=binnumber)
        logbins2 = np.logspace(-0.09, 0, num = 10)
        bins = np.hstack((logbins1, logbins2))
        xrange = np.logspace(-3, -0.0001, num=(4*binnumber))
            
        
    # Normalizing showers
    binlist = []

    print("\rCalculating bins...", end="")
    
    ggbinlists = [[], [], [], []]
    gqbinlists = [[], [], [], []]
    
    for i in range(len(bins)-1):
        binwidth = bins[i+1]-bins[i]
        bincenter = bins[i+1] - (binwidth/2)
        binlist.append(bincenter)
        for gglist in gglists:
            index = gglists.index(gglist)
            frequencylist = []
            for initialfrac in gglist:
                if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                    frequencylist.append(initialfrac)
            density = len(frequencylist)*bincenter/(n*binwidth)
            ggbinlists[index].append(density)
        
        for gqlist in gqlists:
            index = gqlists.index(gqlist)
            frequencylist = []
            for initialfrac in gqlist:
                if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                    frequencylist.append(initialfrac)
            density = len(frequencylist)*bincenter/(n*binwidth)
            gqbinlists[index].append(density)
    
    # Calculating solutions
    solutions = [[], [], [], []]
    gamma = 0.57721566490153286
    
    for x in xrange:
        D1 = (1/2)*(t1/(np.pi**2 * np.log(1/x)**3))**(1/4) * np.exp(-gamma*t1+ 2*np.sqrt(t1*np.log(1/x)))
        D2 = (1/2)*(t2/(np.pi**2 * np.log(1/x)**3))**(1/4) * np.exp(-gamma*t2+ 2*np.sqrt(t2*np.log(1/x)))
        D3 = (1/2)*(t3/(np.pi**2 * np.log(1/x)**3))**(1/4) * np.exp(-gamma*t3+ 2*np.sqrt(t3*np.log(1/x)))
        D4 = (1/2)*(t4/(np.pi**2 * np.log(1/x)**3))**(1/4) * np.exp(-gamma*t4+ 2*np.sqrt(t4*np.log(1/x)))
        
        solutions[0].append(D1)
        solutions[1].append(D2)
        solutions[2].append(D3)
        solutions[3].append(D4)


    # Plot    
    plt.figure(dpi=1000, figsize= (6,5)) #(w,h) figsize= (10,3)
    title = ("Vaccum showers: " + str(n) + 
             ". epsilon: " + str(epsilon) + 
             "\n " + opt_title)    
    plt.suptitle(title)

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

        ax.plot(binlist, ggbinlists[index], 'b--', label ="gluons only")
        ax.plot(binlist, gqbinlists[index], 'g--', label="quarks & gluons")
        ax.plot(xrange, solutions[index], 'r:', label="solution gluons only")
        ax.set_title('t = ' + str(tvalues[index]))
        ax.set_xlim(plot_lim,1)
        ax.set_ylim(0.01,10)
        ax.set_xlabel('z ')
        ax.set_ylabel('D(x,t)')
        ax.grid(linestyle='dashed', linewidth=0.2)
        ax.legend()

        if scale == "lin":
            ax.set_xscale("linear")
            ax.set_yscale("log")

        elif scale == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")


    print("\rShowing..." + 10*" ", end="")

    plt.tight_layout()
    plt.show()
    print("\rDone!" + 10*" ")


def error_message_several_showers(n, opt_title, scale):
    """"Checks the input parameters for erros and generates merror_msg."""
    error = False
    msg = ""
    n_error = not isinstance(n, int)
    title_error = not  isinstance(opt_title, str)
    scale_error = not ((scale == "lin") or (scale == "log"))


    if n_error or title_error or scale_error:
        error = True
        if n_error:
            msg = msg + "\nERROR! - 'n' must be an integer."
        if title_error:
            msg = msg + "\nERROR! - 'opt_title' must be a str."
        if scale_error:
            msg = msg+ "\nERROR! - 'scale' must be 'lin' or 'log'."
    return error, msg    