# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Vacuum evolutionv variable program. 

import matplotlib.pyplot as plt
import numpy as np


def evolution_variable():
    Q_0 = 1
    p_values = [10, 20, 50, 100, 250, 500, 2000]
    Rrange = np.linspace(0.99999, 0.00001, 100000)
    tvalues = []
    
    alpha_S = 0.1184
    
    for p_t in p_values:
        tvalue = []
        for R in Rrange:
            if (R*p_t) > Q_0:
                t_max = (alpha_S / (np.pi)) * np.log((R*p_t)/Q_0)

                tvalue.append(t_max)
            else:
                tvalue.append(None)
        tvalues.append(tvalue)
        
    # Plot    

    plt.figure(dpi=1000) #figsize= (10,3)
    title = "Evolution variable vacuum showers. alpha_S = " + str(alpha_S)
    #plt.suptitle(title)

    plt.rc('axes', titlesize="small", labelsize="x-small")     # fontsize of the axes title and labels.
    plt.rc('xtick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('ytick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('legend',fontsize='xx-small')    # fontsize of the legend labels.
    plt.rc('lines', linewidth=0.8)

    ax1 = plt.subplot(111) #H B NR
    
    for i in range(len(p_values)):  
        print("Plotting... " + str(i))
        ax1.plot(Rrange, tvalues[i], label= "p_t: " + str(p_values[i]) + " GeV")
    
    ax1.set_title('')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 0.40)
    ax1.set_xscale("linear")
    ax1.set_xlabel('R ')
    ax1.set_ylabel('t')
    ax1.grid(linestyle='dashed', linewidth=0.2)

    print("Showing")

    plt.legend()
    plt.tight_layout()
    
    plt.show()
    print("Done!")    