# -*- coding: utf-8 -*-
#Kristoffer spyder

#Subprogram.
#Contains the splitting functions and MH algorithms. 


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad # Integration package.


epsilon = 10**(-3)
C_A = 3
C_F = 4/3
N_F = 5


def p_ggg_dummy(z):
    return ((2*C_A)*(1/(z*(1-z))))

def p_ggg(z):
    return ((2*C_A)*((1-z)/z + z/(1-z) + z*(1-z)))

def p_ggg_konrad(z):
    return (6*(1-z(1-z))**2/(z(1-z)))

def p_qqg_dummy(z):
    return (((-3*C_F)*2)/(z-1))

def p_qqg(z):
    return (((3*C_F)*(1+z**2))/(1-z))

def p_gqq(z):
    return ((N_F/2)*(z**2 + (1-z)**2))


def vacuum_ggg(): # vacuum ggg splitting function MH hastings
    while True: # loop for generating MH value
        rnd1 = np.random.uniform(epsilon, 1-epsilon)
        xi = ((1-epsilon)/epsilon)**(2*rnd1-1)
        splittingvalue = xi/(1+xi) #Randomly generated initial state from the distribution f(x)
        acceptance = min(1-epsilon, (p_ggg(splittingvalue) / p_ggg_dummy(splittingvalue))) #acceptance ratio determinedby probability density ratios.
        rnd2 = np.random.uniform(0,1) # random value for acceptance if statement.
            
        if acceptance >= rnd2: #Condition for accepting state.
            break
        
        return splittingvalue
            
    
def vacuum_qqg(): # vacuum qqg splitting function MH hastings
    while True: # loop for generating MH value
        rnd1 = np.random.uniform(epsilon, 1-epsilon)
        xi = ((1-epsilon)/epsilon)**(rnd1)
        splittingvalue = ((epsilon-1)/(xi))+1 #Randomly generated initial state from the distribution f(x)            
        acceptance = min(1-epsilon, (p_qqg(splittingvalue) / p_qqg_dummy(splittingvalue)))#acceptance ratio determinedby probability density ratios.
        rnd2 = np.random.uniform(0,1) # random value for acceptance if statement.
            
        if acceptance >= rnd2: #Condition for accepting state.
            break
                
        return splittingvalue
    
    
def vacuum_gqq(): # vacuum gqq splitting function
    rnd1 = np.random.uniform(epsilon,1-epsilon)
    d = rnd1*0.665+0.001
    a = (36 * (d**(2)) - (24 * d) + 5)**(1/2)
    b = ( a + (6* d) - 2)**(1/3)
    splittingvalue = (0.5 + 0.5*b - (0.5/b))
    return splittingvalue

        
    
    
def MH_comparison_ggg(n): #function for comparing the MH results from the original ones for ggg vertex
    dummy = []
    MHD = []
    P_dumy = []
    P_full = []
    x = np.linspace(epsilon, 1-epsilon ,1000)
    dumyint, error1 = quad(p_ggg_dummy, epsilon, 1-epsilon)
    fullint, error2 = quad(p_ggg, epsilon, 1-epsilon)
    
    rejects = 0
    
    for i in range(1000): # loop for generating precise functions
        P_dumyvalue = p_ggg_dummy(x[i]) / dumyint
        P_dumy.append(P_dumyvalue)
        P_fullvalue = p_ggg(x[i]) / fullint
        P_full.append(P_fullvalue)
    
    
    for j in range(n): # loop for generating MH
        rnd1 = np.random.uniform(epsilon, 1-epsilon)
        xi = ((1-epsilon)/epsilon)**(2*rnd1-1)
        dummyx = xi/(1+xi) #Randomly generated initial state from the distribution f(x)
        dummy.append(dummyx)
        
        acceptance = min(1-epsilon, (p_ggg(dummyx) / p_ggg_dummy(dummyx)))#acceptance ratio determinedby probability density ratios.
        rnd2 = np.random.uniform(0,1) # random value for acceptance if statement.
            
        if acceptance >= rnd2: #Condition for accepting state.
            MHD.append(dummyx)
            
        else:
            rejects += 1

    print("The number of rejects was:", rejects, "Successes are:", len(MHD), "Acceptance percentage: ", round(len(MHD)/(rejects + len(MHD)), 2) )
    
    plt.figure(figsize= (10,3), dpi=1000)
    plt.rc('axes', titlesize="small" , labelsize="small")     # fontsize of the axes title and labels.
    plt.rc('xtick', labelsize="small")    # fontsize of the tick labels.
    plt.rc('ytick', labelsize="small")    # fontsize of the tick labels.

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    ax1.plot(x, P_dumy, 'y-', label='P_d(z)')
    ax1.plot(x, P_full, 'r-', label='P_ggg(z)')
    ax2.plot(x, P_dumy, 'y-', label='P_d(z)')
    ax2.plot(x, P_full, 'r-', label='P_ggg(z)')
    ax1.hist(dummy, 200, density='true')
    ax2.hist(MHD, 200, density='true')
    
    ax1.set_title("Original histogram")
    ax2.set_title("MH corrected histogram")

    ax1.set_ylim(0,3)
    ax2.set_ylim(0,3)
    ax1.legend(fontsize="x-small")
    ax2.legend(fontsize="x-small")
    plt.show()
    print("Done!")
    
    
def MH_comparison_qqg(n): #function for comparing the MH results from the original ones for qqg vertex
    dummy = []
    MHD = []
    P_dumy = []
    P_full = []
    x = np.linspace(epsilon, 1-epsilon ,1000)
    dumyint, error1 = quad(p_qqg_dummy, epsilon, 1-epsilon)
    fullint, error2 = quad(p_qqg, epsilon, 1-epsilon)
    
    rejects = 0
    
    for i in range(1000): # loop for generating precise functions
        P_dumyvalue = p_qqg_dummy(x[i]) / dumyint
        P_dumy.append(P_dumyvalue)
        P_fullvalue = p_qqg(x[i]) / fullint
        P_full.append(P_fullvalue)
    
    for j in range(n): # loop for generating MH
        rnd1 = np.random.uniform(epsilon, 1-epsilon)
        xi = ((1-epsilon)/epsilon)**(rnd1)
        dummyx = ((epsilon-1)/(xi))+1 #Randomly generated initial state from the distribution f(x)
        dummy.append(dummyx)
            
        acceptance = min(1-epsilon, (p_qqg(dummyx) / p_qqg_dummy(dummyx)))#acceptance ratio determinedby probability density ratios.
        rnd2 = np.random.uniform(0,1) # random value for acceptance if statement.
            
        if acceptance >= rnd2: #Condition for accepting state.
            MHD.append(dummyx)
                
        else:
            rejects += 1

    print("The number of rejects was:", rejects, "Successes are:", len(MHD), "Acceptance percentage: ", round(len(MHD)/(rejects + len(MHD)), 2) )
    
    plt.figure(figsize= (10,3), dpi=1000)
    plt.rc('axes', titlesize="small" , labelsize="small")     # fontsize of the axes title and labels.
    plt.rc('xtick', labelsize="small")    # fontsize of the tick labels.
    plt.rc('ytick', labelsize="small")    # fontsize of the tick labels.

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    ax1.plot(x, P_dumy, 'y-', label='P_d(z)')
    ax1.plot(x, P_full, 'r-', label='P_qqg(z)')
    ax2.plot(x, P_dumy, 'y-', label='P_d(z)')
    ax2.plot(x, P_full, 'r-', label='P_qqg(z)')
    ax1.hist(dummy, 200, density='true')
    ax2.hist(MHD, 200, density='true')
    
    ax1.set_title("Original histogram")
    ax2.set_title("MH corrected histogram")

    ax1.set_ylim(0,3)
    ax2.set_ylim(0,3)
    ax1.legend(fontsize="x-small")
    ax2.legend(fontsize="x-small")
    plt.show()
    print("Done!")
    
    
def exact_gqq(n):
    x = np.linspace(epsilon,1-epsilon,n)
    z = []
    y2 = []
    value, error = quad(p_gqq, epsilon, 1-epsilon)

    for i in range (0,n): #Loop for generating random momentum fractions
        rnd = np.random.uniform(epsilon,1-epsilon)# A single value
        d = rnd*0.665+0.001
        
        a = (36 * (d**(2)) - (24 * d) + 5)**(1/2)
        b = ( a + (6* d) - 2)**(1/3)
        z1 = (0.5 + 0.5*b - (0.5/b))
        z.append(z1)
        
        z2 = (N_F/2*((x[i]**(2))+(1-x[i])**(2)))/value #Plotting exact function
        y2.append(z2)
        
        
    plt.figure(figsize= (5,3), dpi=1000)
    plt.rc('axes', titlesize="small" , labelsize="small")     # fontsize of the axes title and labels.
    plt.rc('xtick', labelsize="small")    # fontsize of the tick labels.
    plt.rc('ytick', labelsize="small")    # fontsize of the tick labels.
    plt.title("Original histogram")
    
    plt.plot(x, y2, 'r-', label='P_gqq(z)')
    plt.hist(z, 200, density='true', label='MC')
    #plt.xlabel('Momentum fraction (z)')
    #plt.ylabel('Probability density')
    #plt.ylim(0, 3)
    plt.legend()
    plt.show()
    
    
def normal(): # Function for checking the normal distribution used he
    rnd1 = round(np.random.uniform(epsilon, 1-epsilon), 4)
    xi = ((1-epsilon)/epsilon)**(2*rnd1-1)
    dummyx = xi/(1+xi) #Randomly generated initial state from the distribution f(x)
    stdd = 0.12
    samples2 = np.random.normal(loc=(dummyx), scale=stdd, size=1000000)
            
    print("Center at:", dummyx)
    plt.hist(samples2, 500, density='true')
    plt.xlim(0,1)
    plt.show()
    
    
def ggg_function_ratios():
    P_dumy = []
    P_full = []
    Ratio = []
    x = np.linspace(epsilon, 1-epsilon ,1000)
    dumyint, error1 = quad(p_ggg_dummy, epsilon, 1-epsilon)
    fullint, error2 = quad(p_ggg, epsilon, 1-epsilon)
    
    
    for i in range(1000): # loop for generating precise functions
        P_dumyvalue = p_ggg_dummy(x[i]) 
        P_dumy.append(P_dumyvalue)
        P_fullvalue = p_ggg(x[i]) 
        P_full.append(P_fullvalue)
        Ratio.append(P_fullvalue/P_dumyvalue)
    
    plt.axhline(y=1, color='r')
    plt.xlim(0,1)
    plt.ylim(0,1.2)
    plt.plot(x, Ratio )
    plt.show()
    
   
def qqg_function_ratios():
    P_dumy = []
    P_full = []
    Ratio = []
    x = np.linspace(epsilon, 1-epsilon ,1000)
    dumyint, error1 = quad(p_qqg_dummy, epsilon, 1-epsilon)
    fullint, error2 = quad(p_qqg, epsilon, 1-epsilon)
    
    
    for i in range(1000): # loop for generating precise functions
        P_dumyvalue = p_qqg_dummy(x[i]) 
        P_dumy.append(P_dumyvalue)
        P_fullvalue = p_qqg(x[i]) 
        P_full.append(P_fullvalue)
        Ratio.append(P_fullvalue/P_dumyvalue)
    
    plt.axhline(y=1, color='r')
    plt.xlim(0,1)
    plt.ylim(0,1.2)
    plt.plot(x, Ratio )
    plt.show()

