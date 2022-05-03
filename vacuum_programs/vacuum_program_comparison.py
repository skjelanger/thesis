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
    gglist1 = []
    gqlist1 = []
    gglist2 = []
    gqlist2 = []
    gglist3 = []
    gqlist3 = []
    gglist4 = []
    gqlist4 = []
        
    for i in range(1,n):
        print("\rLooping... "+ str(round(100*i/(n),1)) + "%",end="")
        Showergg = ggv.generate_shower(tvalues, p_0, Q_0, R, i)
        Showergq = gqv.generate_shower("gluon", tvalues, p_0, Q_0, R, i)
        
        gglist1.extend(Showergg.FinalFracList1)
        gglist2.extend(Showergg.FinalFracList2)
        gglist3.extend(Showergg.FinalFracList3)
        gglist4.extend(Showergg.FinalFracList4)  
        gqlist1.extend(Showergq.FinalFracList1)
        gqlist2.extend(Showergq.FinalFracList2)
        gqlist3.extend(Showergq.FinalFracList3)
        gqlist4.extend(Showergq.FinalFracList4)  
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
    
    ggbinlist1 = []
    ggbinlist2 = []
    ggbinlist3 = []
    ggbinlist4 = []
    gqbinlist1 = []
    gqbinlist2 = []
    gqbinlist3 = []
    gqbinlist4 = []
    
    for i in range(len(bins)-1):
        binwidth = bins[i+1]-bins[i]
        bincenter = bins[i+1] - (binwidth/2)
        binlist.append(bincenter)
        
        # Calculating bins 1
        frequencylist1 = []
        frequencylist2 = []
        for initialfrac in gglist1:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist1.append(initialfrac)
        gluondensity = len(frequencylist1)*bincenter/(n*binwidth)
        ggbinlist1.append(gluondensity)
        
        for initialfrac in gqlist1:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist2.append(initialfrac)
        binharddensity = len(frequencylist2)*bincenter/(n*binwidth)
        gqbinlist1.append(binharddensity)
    
        # Calculating bins 2
        frequencylist1 = []
        frequencylist2 = []
        for initialfrac in gglist2:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist1.append(initialfrac)
        gluondensity = len(frequencylist1)*bincenter/(n*binwidth)
        ggbinlist2.append(gluondensity)

        
        for initialfrac in gqlist2:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist2.append(initialfrac)
        binharddensity = len(frequencylist2)*bincenter/(n*binwidth)
        gqbinlist2.append(binharddensity)
        
        # Calculating bins 3
        frequencylist1 = []
        frequencylist2 = []
        for initialfrac in gglist3:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist1.append(initialfrac)
        gluondensity = len(frequencylist1)*bincenter/(n*binwidth)
        ggbinlist3.append(gluondensity)

        for initialfrac in gqlist3:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist2.append(initialfrac)
        binharddensity = len(frequencylist2)*bincenter/(n*binwidth)
        gqbinlist3.append(binharddensity)
        
        # Calculating bins 4
        frequencylist1 = []
        frequencylist2 = []
        for initialfrac in gglist4:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist1.append(initialfrac)
        gluondensity = len(frequencylist1)*bincenter/(n*binwidth)
        ggbinlist4.append(gluondensity)

        for initialfrac in gqlist4:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist2.append(initialfrac)
        binharddensity = len(frequencylist2)*bincenter/(n*binwidth)
        gqbinlist4.append(binharddensity)
    
    
    # Calculating solutions
    solution1 = []
    solution2 = []
    solution3 = []
    solution4 = []
    gamma = 0.57721566490153286
    
    for x in xrange:
        D1 = (1/2)*(t1/(np.pi**2 * np.log(1/x)**3))**(1/4) * np.exp(-gamma*t1+ 2*np.sqrt(t1*np.log(1/x)))
        D2 = (1/2)*(t2/(np.pi**2 * np.log(1/x)**3))**(1/4) * np.exp(-gamma*t2+ 2*np.sqrt(t2*np.log(1/x)))
        D3 = (1/2)*(t3/(np.pi**2 * np.log(1/x)**3))**(1/4) * np.exp(-gamma*t3+ 2*np.sqrt(t3*np.log(1/x)))
        D4 = (1/2)*(t4/(np.pi**2 * np.log(1/x)**3))**(1/4) * np.exp(-gamma*t4+ 2*np.sqrt(t4*np.log(1/x)))
        
        solution1.append(D1)
        solution2.append(D2)
        solution3.append(D3)
        solution4.append(D4)


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
    
    print("\rPlotting 1...", end="")

    ax1.plot(binlist, ggbinlist1, 'b--', label ="gluons only")
    ax1.plot(binlist, gqbinlist1, 'g--', label="quarks & gluons")
    ax1.plot(xrange, solution1, 'r:', label="solution gluons only")
    ax1.set_title('t = ' + str(t1))
    ax1.set_xlim(plot_lim,1)
    ax1.set_ylim(0.01,10)
    ax1.set_xlabel('z ')
    ax1.set_ylabel('D(x,t)')
    ax1.grid(linestyle='dashed', linewidth=0.2)
    ax1.legend()
    
    
    print("\rPlotting 2...", end="")

    ax2.plot(binlist, ggbinlist2, 'b--', label ="gluons only")
    ax2.plot(binlist, gqbinlist2, 'g--', label="quarks & gluons")

    ax2.plot(xrange, solution2, 'r:', label="solution gluons only")
    ax2.set_title('t = ' + str(t2))
    ax2.set_xlim(plot_lim,1)
    ax2.set_ylim(0.01,10)
    ax2.set_xlabel('z')
    ax2.set_ylabel('D(x,t)')
    ax2.grid(linestyle='dashed', linewidth=0.2)
    ax2.legend()

    
    print("\rPlotting 3...", end="")

    ax3.plot(binlist, ggbinlist3, 'b--', label ="gluons only")
    ax3.plot(binlist, gqbinlist3, 'g--', label="quarks & gluons")
    ax3.plot(xrange, solution3, 'r:', label="solution gluons only")
    ax3.set_title('t = ' + str(t3))
    ax3.set_xlim(plot_lim,1)
    ax3.set_ylim(0.01,10)
    ax3.set_xlabel('z ')
    ax3.set_ylabel('D(x,t)')
    ax3.grid(linestyle='dashed', linewidth=0.2)
    ax3.legend()
    

    print("\rPlotting 4...", end="")

    ax4.plot(binlist, ggbinlist4, 'b--', label ="gluons only")
    ax4.plot(binlist, gqbinlist4, 'g--', label="quarks & gluons")
    ax4.plot(xrange, solution4, 'r:', label="solution gluons only")
    ax4.set_title('t = ' + str(t4))
    ax4.set_xlim(plot_lim,1)
    ax4.set_ylim(0.01,10)
    ax4.set_xlabel('z ')
    ax4.set_ylabel('D(x,t)')
    ax4.grid(linestyle='dashed', linewidth=0.2)
    ax4.legend()

    if scale == "lin":
        ax1.set_xscale("linear")
        ax1.set_yscale("log")
        ax2.set_xscale("linear")
        ax2.set_yscale("log")
        ax3.set_xscale("linear")
        ax3.set_yscale("log")
        ax4.set_xscale("linear")
        ax4.set_yscale("log")

    elif scale == "log":
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax4.set_xscale("log")
        ax4.set_yscale("log")

    print("\rShowing", end="")

    plt.tight_layout()
    plt.show()
    print("\rDone!")


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