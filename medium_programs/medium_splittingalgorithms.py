# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Contains the metropolis hastings algorithms, and programs for verifying 
# that they are reproducing the full splitting functions. 

import medium_splittingfunctions as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

epsilon = 10**(-3)


# MH algorithms intenteded for using in the actual shower programs.
def MH_gg():
    """Metropolis Hastings algorithm for the gg splitting function in 
    Medium. Returns a single splitting value."""
    simple_splitting_integral, __ = quad(sf.gg_simple, epsilon, 1-epsilon)
    while True:
        rnd1 = np.random.uniform(0,1)
        a = ((rnd1-0.5)*simple_splitting_integral)/6
        splittingvalue = 0.5 - (a/(2*(16+a**2)**(1/2)))
        acceptance = min(1, 
                    sf.gg_full(splittingvalue)/sf.gg_simple(splittingvalue))
        rnd2 = np.random.uniform(0,1) 
            
        if acceptance >= rnd2:
            break
        
    return splittingvalue
            
    
def MH_qq(): 
    """Metropolis Hastings algorithm for the qq splitting function in 
    Medium. Returns a single splitting value."""
    simple_splitting_integral, __ = quad(sf.qg_simple, epsilon, 1-epsilon)
    while True:
        rnd1 = np.random.uniform(0,1)
        a = ((rnd1/8)*simple_splitting_integral +
             np.sqrt(epsilon)/np.sqrt(1-epsilon))
        splittingvalue = (a**2)/(a**2 +1)
        acceptance = min(1, 
                    sf.qq_full(splittingvalue)/sf.qq_simple(splittingvalue))
        rnd2 = np.random.uniform(0,1)
        
        if acceptance >= rnd2:
            break
                
    return splittingvalue
    
    
def MH_qg():
    """Metropolis Hastings algorithm for the qg splitting function in 
    Medium. Returns a single splitting value."""
    simple_splitting_integral, __ = quad(sf.qg_simple, epsilon, 1-epsilon)
    while True:
        rnd1 = np.random.uniform(0,1)
        a = 2*np.arcsin(np.sqrt(1-epsilon))- rnd1*simple_splitting_integral
        splittingvalue = 1-(np.sin(a/2))**(2)
        acceptance = min(1, 
                    sf.qg_full(splittingvalue) / sf.qq_simple(splittingvalue))
        rnd2 = np.random.uniform(0,1) 
            
        if acceptance >= rnd2:
            break
                
    return splittingvalue
    

# MH validation programs intended for checking that the algorithms are working
# as intended. This is done by comparing MH results with the exact splitting
# functions.
def MH_comparison_gg(n):
    """This function samples n random samples from the simple gg splitting
    function, and then performs the MH algorithm to verify the results.
    Both the original samples, and MH corrected samples are then plotted"""
    
    simplefunction_values = []
    fullfunction_values = []
    xvalues = np.linspace(epsilon, 1-epsilon ,1000)
    simple_splitting_integral, __ = quad(sf.gg_simple, epsilon, 1-epsilon)
    full_splitting_integral, __ = quad(sf.gg_full, epsilon, 1-epsilon)
    
    simple_samples = []
    MH_samples = []
    MH_rejects = 0
    
    for i in range(1000): # Loop for plotting the exact splitting functions.
        simplevalue = sf.gg_simple(xvalues[i]) / simple_splitting_integral
        simplefunction_values.append(simplevalue)
        fullvalue = sf.gg_full(xvalues[i]) / full_splitting_integral
        fullfunction_values.append(fullvalue)
    
    for j in range(n): # Loop for random sampling and MH.
        rnd1 = np.random.uniform(0,1)
        a = ((rnd1-0.5)*simple_splitting_integral)/2
        dummy_y = 0.5 - (a/(2*(16+a**2)**(1/2)))
        simple_samples.append(dummy_y)
        
        acceptance = min(1, sf.gg_full(dummy_y)/sf.gg_simple(dummy_y))
        rnd2 = np.random.uniform(0,1)
            
        if acceptance >= rnd2: 
            MH_samples.append(dummy_y)
            
        else:
            MH_rejects += 1
            

    print("The number of MH_rejects was:", MH_rejects, 
          ". Successes are:", len(MH_samples), 
          ". Acceptance percentage: ", round(len(MH_samples)/
                                           (MH_rejects+len(MH_samples)), 2))
    
    plot_results(xvalues, simplefunction_values, fullfunction_values,
                 simple_samples, MH_samples, "gg", n)
    
    
def MH_comparison_qg(n):
    """This function samples n random samples from the simple gg splitting
    function, and then performs the MH algorithm to verify the results.
    Both the original samples, and MH corrected samples are then plotted"""
    
    simplefunction_values = []
    fullfunction_values = []
    xvalues = np.linspace(epsilon, 1-epsilon ,1000)
    simple_splitting_integral, __ = quad(sf.qg_simple, epsilon, 1-epsilon)
    full_splitting_integral, __ = quad(sf.qg_full, epsilon, 1-epsilon)
    
    simple_samples = []
    MH_samples = []
    MH_rejects = 1
    
    for i in range(1000): # Loop for plotting the exact splitting functions.
        simplevalue = sf.qg_simple(xvalues[i]) / simple_splitting_integral
        simplefunction_values.append(simplevalue)
        fullvalue = sf.qg_full(xvalues[i]) / full_splitting_integral
        fullfunction_values.append(fullvalue)
    
    for j in range(n): # Loop for random sampling and MH.
        rnd1 = np.random.uniform(0,1)
        a = (2*np.arcsin(np.sqrt(1-epsilon))- rnd1*simple_splitting_integral)
        dummy_y = 1-(np.sin(a/2))**(2)
        simple_samples.append(dummy_y)
        
        acceptance = min(1, sf.qg_full(dummy_y)/sf.qg_simple(dummy_y))
        rnd2 = np.random.uniform(0,1) 
            
        if acceptance >= rnd2:
            MH_samples.append(dummy_y)
            
        else:
            MH_rejects += 1
        
    print("The number of MH_rejects was:", MH_rejects, 
          ". Successes are:", len(MH_samples), 
          ". Acceptance percentage: ", round(len(MH_samples)/
                                             (MH_rejects+len(MH_samples)), 2))
    
    plot_results(xvalues, simplefunction_values, fullfunction_values,
                 simple_samples, MH_samples, "qg", n)
    

def MH_comparison_qq(n):
    """This function samples n random samples from the simple qq splitting
    function, and then performs the MH algorithm to verify the results.
    Both the original samples, and MH corrected samples are then plotted"""
    
    simplefunction_values = []
    fullfunction_values = []
    xvalues = np.linspace(epsilon, 1-epsilon ,1000)
    simple_splitting_integral, __ = quad(sf.qq_simple, epsilon, 1-epsilon)
    full_splitting_integral, __ = quad(sf.qq_full, epsilon, 1-epsilon)
    
    simple_samples = []
    MH_samples = []
    MH_rejects = 0
        
    for i in range(1000): # Loop for plotting the exact splitting functions.
        simplevalue = sf.qq_simple(xvalues[i]) / simple_splitting_integral
        simplefunction_values.append(simplevalue)
        fullvalue = sf.qq_full(xvalues[i]) / full_splitting_integral
        fullfunction_values.append(fullvalue)
    
    for j in range(n): # Loop for random sampling and MH.
        rnd1 = np.random.uniform(0,1)
        a = ((rnd1/8)*simple_splitting_integral +
             np.sqrt(epsilon)/np.sqrt(1-epsilon))
        dummy_y = (a**2)/(a**2 + 1)
        simple_samples.append(dummy_y)
            
        acceptance = min(1, sf.qq_full(dummy_y)/sf.qq_simple(dummy_y))
        rnd2 = np.random.uniform(0,1)
            
        if acceptance >= rnd2:
            MH_samples.append(dummy_y)
                
        else:
            MH_rejects += 1

    print("The number of MH_rejects was:", MH_rejects, 
          ". Successes are:", len(MH_samples), 
          ". Acceptance percentage: ", round(len(MH_samples)/
                                             (MH_rejects+len(MH_samples)), 2))
    
    plot_results(xvalues, simplefunction_values, fullfunction_values,
                 simple_samples, MH_samples, "qq", n)
    
    
def comparison_gg_simple(n):
    """This function samples n random samples from the simple gg splitting
    function, and then compares with the actual function."""
    
    fullfunction_values = []
    xvalues = np.linspace(epsilon, 1-epsilon ,1000)
    simple_splitting_integral, __ = quad(sf.gg_simple_analytical, epsilon, 1-epsilon)
    
    samples = []

    
    for i in range(1000): # Loop for plotting the exact splitting functions.
        fullvalue = sf.gg_simple_analytical(xvalues[i]) / simple_splitting_integral
        fullfunction_values.append(fullvalue)
    
    for j in range(n): # Loop for random sampling.
        rnd1 = np.random.uniform(0, 1)
        a = (rnd1-0.5)*simple_splitting_integral
        samplevalue = 0.5 + (a/( 2* np.sqrt((16 + a**2)))) 
        samples.append(samplevalue)
     
    plt.figure(figsize= (5,3), dpi=500)

    plt.rc('axes', titlesize="small" , labelsize="small")
    plt.rc('xtick', labelsize="small")
    plt.rc('ytick', labelsize="small")
    
    plt.plot(xvalues, fullfunction_values, 'r-', label='K(z)')
    plt.hist(samples, 200, density='true', label="samples")
    plt.title("Original histogram")
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('z')
    plt.legend(fontsize="x-small")
    plt.show()
    print("Done!")
    
    
def plot_results(xvalues, simple_function, full_function, simple_samples,
                 MH_samples, splitting, n):
    """This program plots the results from the MH_comparison programs"""
    plt.figure(figsize= (10,3), dpi=500)
    plt.rc('axes', titlesize="small" , labelsize="small")
    plt.rc('xtick', labelsize="small")
    plt.rc('ytick', labelsize="small")

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    ax1.plot(xvalues, simple_function, 'y-', label=splitting+'_simple(z)')
    ax1.plot(xvalues, full_function, 'r-', label=splitting+'_full(z)')
    ax2.plot(xvalues, simple_function, 'y-', label=splitting+'_simple(z)')
    ax2.plot(xvalues, full_function, 'r-', label=splitting+'_full(z)')
    ax1.hist(simple_samples, 200, density='true', label="samples")
    ax2.hist(MH_samples, 200, density='true', label="samples")
    
    ax1.set_title("Original histogram")
    ax2.set_title("MH corrected histogram")

    ax1.set_xlim(0,1)
    ax1.set_ylim(0,3)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,3)
    ax1.set_xlabel('z')
    ax2.set_xlabel('z')
    ax1.legend(fontsize="x-small")
    ax2.legend(fontsize="x-small")
    plt.show()
    print("Done!")