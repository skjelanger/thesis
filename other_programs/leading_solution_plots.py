# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import betainc
from scipy.special import beta
from scipy.special import digamma

tvalues = [0.04, 0.1, 0.3, 0.4]
epsilon = 10**(-4)
em = 0.577216


def vacuum_leading_mellin(scale):
    
    if scale == "lin":
        nurange = np.linspace(epsilon,1,1000)
    elif scale == "log":
        nurange = np.logspace(-4,0-epsilon,500)


    inclsolutions = [[],[],[],[]]
    leadingsolutions = [[],[],[],[]]    
    polyfits    = [[],[],[],[]]
    polystrings = []
    testfunctions = [[],[],[],[]]
    
    pnufunc = lambda z, nu : (z**nu-1)/(z*(1-z))
    
    
    for t in tvalues:
        index = tvalues.index(t)
        for nu in nurange:
            pint, __ = quad(pnufunc, 0.5, 1, args=nu)
            leadingsol1 = ((2*betainc(nu, epsilon, 0.5)*beta(nu,epsilon) + 2*pint)/(2*pint))*np.exp(2*pint*t)
            leadingsol2= ((2*betainc(nu, epsilon, 0.5)*beta(nu,epsilon))/(2*pint))
            leadingsol = leadingsol1-leadingsol2
            leadingsolutions[index].append(leadingsol)
            inclsol = np.exp(-2*(digamma(nu)+em)*t)
            inclsolutions[index].append(inclsol)
            
            testfunc = np.exp(-nu)
            testfunctions[index].append(testfunc)
        
        #Calculate polyfits:
        z = np.polyfit((1/nurange) , leadingsolutions[index], 1)
        print("z: ", z)
        
        #f = np.poly1d(z)
        f = lambda nu : z[0]*(1/nu) + z[1]
        
        #f = ffunc(z[0], z[1])
        polyfit = f(nurange)
        polystring = "$f(\\nu)={%s}/\\nu$"%round(z[0],2) + "$+{%s}$"%round(z[1],2)  
    
        polyfits[index].extend(polyfit)
        polystrings.append(polystring)

    print("leadingsol: ", leadingsolutions[0])
    print("polyfit: ", polyfits[0])

    print("len nurange: ", len(nurange), ". len leadingsolutions index: ", len(leadingsolutions[0]))
            
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
        ax.plot(nurange, inclsolutions[index], "C3--", label="incl sol")
        ax.plot(nurange, leadingsolutions[index], "C1--", label="leading sol")
        #ax.plot(nurange, polyfits[index], "C2:", label = "fit")

        #ax.plot(nurange, testfunctions[index], "C6:", label = "testfunc")

        if scale == "lin":
            ax.set_xscale("linear")
            ax.set_xlim(0,1)
        
        if scale == "log":
            ax.set_xscale("log")
            ax.set_xlim(0.01,1)
            
        ax.set_yscale("log")
        ax.set_title('$t = $' +str(tvalues[index]))
        ax.set_ylim(0.1,100)
        ax.set_xlabel('$\\nu$')
        ax.set_ylabel('$\\tilde{D}(\\nu,t)$')
        ax.grid(linestyle='dashed', linewidth=0.2)
        ax.legend()
        
        #ax.text(0.05, 0.2, polystrings[index], fontsize = "x-small", horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.show()
    print("\rDone!" + 10*" ")    
            