# -*- coding: utf-8 -*-
#Kristoffer spyder

#Subprogram.
#Contains the splitting functions and MH algorithms. 

import vacuum_splittingfunctions as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad # Integration package.

epsilon = 10**(-3)


# MH algorithms intenteded for using in the actual shower programs.
def MH_gg(): 
    """Metropolis Hastings algorithm for the gg splitting function in 
    Medium. Returns a single splitting value."""
    while True: 
        rnd1 = np.random.uniform(0,1)
        xi = ((1-epsilon)/epsilon)**((2*rnd1)-1)
        splittingvalue = xi/(1+xi)
        acceptance = min(1,
                    sf.gg_full(splittingvalue) / sf.gg_simple(splittingvalue)) 
        rnd2 = np.random.uniform(0,1)
            
        if acceptance >= rnd2: 
            break
        
    return splittingvalue
            
def MH_qq(): 
    """Metropolis Hastings algorithm for the qq splitting function in 
    Medium. Returns a single splitting value."""
    while True: 
        rnd1 = np.random.uniform(0,1)
        xi = ((1-epsilon)/epsilon)**(rnd1)
        splittingvalue = ((epsilon-1)/(xi))+1 
        acceptance = min(1, 
                    sf.qg_full(splittingvalue) / sf.qq_simple(splittingvalue))
        rnd2 = np.random.uniform(0,1)
            
        if acceptance >= rnd2:
            break
                
        return splittingvalue
    
    
def qg():
    """Calculates value for he qg splitting vertex."""
    rnd1 = np.random.uniform(0,1)
    d = (rnd1*(2/3))
    a = (36*(d**(2))-(24*d)+5)**(1/2)
    b = (a+(6*d)-2)**(1/3)
    splittingvalue = (0.5+0.5*b-(0.5/b))
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
        xi = ((1-epsilon)/epsilon)**((2*rnd1)-1)
        dummy_y = xi/(1+xi) 
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
                                             (MH_rejects+ len(MH_samples)),2))
    
    plot_results(xvalues, simplefunction_values, fullfunction_values,
                 simple_samples, MH_samples, "gg", n)
    
    
def comparison_qg(n):
    """The qg vertex can be calculated without using the MH algorithm. 
    This program gives acomparison between the randomly samples value, and the 
    exact function. """
    
    full_splitting_integral, __ = quad(sf.qg_full, 0, 1)

    xvalues = np.linspace(epsilon,1-epsilon, 1000)
    fullfunction_values = []
    
    random_samples = []

    for i in range(1000): # Loop for plotting the exact splitting function.
        fullvalue = sf.qg_full(xvalues[i]) / full_splitting_integral
        fullfunction_values.append(fullvalue)
    
    for i in range (n): #Loop for generating random momentum fractions
        rnd = np.random.uniform(0,1)
        d = (rnd*(2/3))
        a = (36*(d**(2))-(24*d)+5)**(1/2)
        b = (a+(6*d)-2)**(1/3)
        randomsample = (0.5+0.5*b-(0.5/b))
        random_samples.append(randomsample)
        
    plt.figure(figsize= (5,3), dpi=300)

    plt.rc('axes', titlesize="small" , labelsize="small")
    plt.rc('xtick', labelsize="small")
    plt.rc('ytick', labelsize="small")
    
    plt.plot(xvalues, fullfunction_values, 'r-', label='qg_full(z)')
    plt.hist(random_samples, 200, density='true', label="samples")
    plt.title("Original histogram")
    
    plt.xlim(0,1)
    plt.ylim(0,3)
    plt.xlabel('z')
    plt.legend(fontsize="x-small")
    plt.show()
    print("Done!")
    
    
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
        xi = ((1-epsilon)/epsilon)**(rnd1)
        dummy_y = ((epsilon-1)/(xi))+1 
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
    
    
def plot_results(xvalues, simple_function, full_function, simple_samples,
                 MH_samples, splitting, n):
    """This program plots the results from the MH_comparison programs"""
    plt.figure(figsize= (10,3), dpi=300)
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