# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Parton shower program for gluons in medium.
# Using the simplified ggg splitting kernel.

import matplotlib.pyplot as plt
from medium_splittingfunctions import gg_simple
import numpy as np
import random
from scipy.integrate import quad


# Constants
epsilon = 10**(-3)
alpha_S = 0.1184

bins = 50
min_value = 0.001

gg_integral, __ = quad((gg_simple), epsilon, 1-epsilon)


# Define classes and class functions.
# These classes will be used for managing the different partons created in 
# different showers, throughout the rest of the program. 
# The Shower and Parton objecs will generally be deleted at the end of a given
# shower, and the required data will be written a list.
class Shower(object):
    """Class used for storing information about individual parton showers."""
    def __init__(self, showernumber):
        self.ShowerNumber = showernumber
        self.FinalList = [] # Contains all Final Partons 
        self.FinalFracList = [] # Contains all partons above min_value.
        self.SplittingGluons = []
        self.Hardest = None

class Parton(object):
    """Class used for storing information about individual partons."""
    def __init__(self, angle, initialfrac, parent, shower, showernumber):        
        self.Type = "gluon"
        self.Angle = angle
        self.InitialFrac = initialfrac
        self.Parent = parent
        self.Primary = None
        self.Secondary = None
        self.Shower = shower
        self.ShowerNumber = showernumber
        
    def split(self):
        """Picks random value from the medium ggg splitting function."""
        rnd1 = np.random.uniform(0, 1)
        a = ((rnd1*126.30129) - 63.15065)
        splittingvalue = 0.5 + (a/( 2* np.sqrt((16 + a**2)))) 
        return splittingvalue
    
    def advance_time(self):
        """Randomly generates a probably value for tau for this splitting. """
        rnd1 = np.random.uniform(epsilon,1-epsilon)
        delta_tau = -2*np.sqrt(self.InitialFrac)*np.log(rnd1)/(gg_integral)
        return delta_tau


# Main shower program. 
# This program generates a single parton shower, given the initial conditions. 
# When running n showers, this program is called n times, and the each 
# iteration returns a Shower-class object.
def generate_shower(tau_max , p_t, Q_0, R, showernumber, n):
    """Main shower program. Generates one shower, and returns the Shower-class
    object associated with the shower. """
    print("\rLooping... " + str(round(100*showernumber/(n))) +"%", end="")

    tau = 0
    Shower0 = Shower(showernumber)
    Parton0 = Parton(tau, 1, None, Shower0, showernumber) # Initial parton

    Shower0.SplittingGluons.append(Parton0)
    Shower0.FinalList.append(Parton0)
    
    while len(Shower0.SplittingGluons) > 0:  
        SplittingParton = random.choice(Shower0.SplittingGluons)
        delta_tau = SplittingParton.advance_time()
        tau = tau + delta_tau
        
        if tau < tau_max: # When condition is True, we can split.
            Shower0.SplittingGluons.remove(SplittingParton)
            Shower0.FinalList.remove(SplittingParton)
            momfrac = Parton.split(SplittingParton)            

            for j in range(0,2): # Loop for generating the new partons.
                if j==0: # Parton 1.
                    initialfrac = SplittingParton.InitialFrac * momfrac
                    NewParton = Parton(tau, initialfrac, SplittingParton,
                                       Shower0, showernumber)
                    SplittingParton.Primary = NewParton
                    
                elif j==1: # Parton 2.
                    initialfrac = SplittingParton.InitialFrac * (1-momfrac)
                    NewParton = Parton(tau, initialfrac, SplittingParton,
                                       Shower0, showernumber)
                    SplittingParton.Secondary = NewParton
                
                Shower0.FinalList.append(NewParton)

                if initialfrac > 0.001: # Limit on how soft gluons can split.
                    Shower0.SplittingGluons.append(NewParton)

        else: # Breaks when t value is too large for splitting.
            break                          
                
    for PartonObj in Shower0.FinalList:
        if PartonObj.InitialFrac > min_value:
            Shower0.FinalFracList.append(PartonObj.InitialFrac)
        del PartonObj
        
    Shower0.Hardest = max(Shower0.FinalFracList)
    
    return Shower0


# Program for comparing the shower program with analytical calculations.
# Four different values of tau are used, and n showers are generated for each
# of them. The results from the showers then compared to analytical results, 
# and plotted in the same figure.
def several_showers_analytical_comparison(n):
    """Runs n parton showers, and compares the result with the analytical."""
    R = 0.4 # Jet radius.    
    p_0 = 100 # Initial parton momentum.
    Q_0 = 1 # Hadronization scale.
    
    tau1 = 0.2
    tau2 = 0.5
    tau3 = 0.8
    tau4 = 1.2
    
    #Generating showers
    gluonlist1 = []
    gluonlist2 = []
    gluonlist3 = []
    gluonlist4 = []
    
    gluons1 = 0
    gluons2 = 0
    gluons3 = 0
    gluons4 = 0    

    for i in range (0,n):
        Shower0 = generate_shower(tau1, p_0, Q_0, R, i, 4*n)
        gluonlist1.extend(Shower0.FinalFracList)
        gluons1 += len(Shower0.FinalList)
        del Shower0

    for i in range (n,2*n):
        Shower0 = generate_shower(tau2, p_0, Q_0, R, i, 4*n)            
        gluonlist2.extend(Shower0.FinalFracList)
        gluons2 += len(Shower0.FinalList)
        del Shower0


    for i in range (2*n,3*n):
        Shower0 = generate_shower(tau3, p_0, Q_0, R, i, 4*n)
        gluonlist3.extend(Shower0.FinalFracList)
        gluons3 += len(Shower0.FinalList)
        del Shower0


    for i in range (3*n,4*n):
        Shower0 = generate_shower(tau4, p_0, Q_0, R, i, 4*n)
        gluonlist4.extend(Shower0.FinalFracList)
        gluons4 += len(Shower0.FinalList)
        del Shower0

        
    # Normalizing showers
    logbins = (np.logspace(-3, 0, num=bins))
    binlist = []

    print("\rCalculating bins 1...", end="")
    
    gluonbinlist1a = []
    gluonbinlist2a = []
    gluonbinlist3a = []
    gluonbinlist4a = []
    gluonbinlist1b = []
    gluonbinlist2b = []
    gluonbinlist3b = []
    gluonbinlist4b = []
    gluonbinlist1c = []
    gluonbinlist2c = []
    gluonbinlist3c = []
    gluonbinlist4c = []
    gluonbinlist1d = []
    gluonbinlist2d = []
    gluonbinlist3d = []
    gluonbinlist4d = []
    
    riemann1a = 0
    riemann2a = 0
    riemann3a = 0
    riemann4a = 0
    riemann1b = 0
    riemann2b = 0
    riemann3b = 0
    riemann4b = 0
    riemann1c = 0
    riemann2c = 0
    riemann3c = 0
    riemann4c = 0
    riemann1d = 0
    riemann2d = 0
    riemann3d = 0
    riemann4d = 0 
    
    for i in range(len(logbins)-1):
        binwidth = logbins[i+1]-logbins[i]
        bincenter = logbins[i+1] - (binwidth/2)
        binlist.append(bincenter)
                
        # Calculating bins 1
        frequencylist = []
        for initialfrac in gluonlist1:
            if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                frequencylist.append(initialfrac)
        
        gluonbinlist1a.append(len(frequencylist)/(n*binwidth))
        gluonbinlist1b.append(len(frequencylist)*bincenter/(n*binwidth))
        gluonbinlist1c.append(len(frequencylist)*bincenter/(len(gluonlist1)*binwidth))
        gluonbinlist1d.append(len(frequencylist)/(len(gluonlist1)*binwidth))

        riemann1a += len(frequencylist)/(n*binwidth)
        riemann1b += len(frequencylist)*bincenter/(n*binwidth)
        riemann1c += len(frequencylist)*bincenter/(len(gluonlist1)*binwidth)
        riemann1d += len(frequencylist)/(len(gluonlist1)*binwidth)

    
        # Calculating bins 2
        frequencylist = []
        for initialfrac in gluonlist2:
            if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                frequencylist.append(initialfrac)
        
        gluonbinlist2a.append(len(frequencylist)/(n*binwidth))
        gluonbinlist2b.append(len(frequencylist)*bincenter/(n*binwidth))
        gluonbinlist2c.append(len(frequencylist)*bincenter/(len(gluonlist2)*binwidth))
        gluonbinlist2d.append(len(frequencylist)/(len(gluonlist2)*binwidth))

        riemann2a += len(frequencylist)/(n*binwidth)
        riemann2b += len(frequencylist)*bincenter/(n*binwidth)
        riemann2c += len(frequencylist)*bincenter/(len(gluonlist2)*binwidth)
        riemann2d += len(frequencylist)/(len(gluonlist2)*binwidth) 


        # Calculating bins 3
        frequencylist = []
        for initialfrac in gluonlist3:
            if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                frequencylist.append(initialfrac)
        
        gluonbinlist3a.append(len(frequencylist)/(n*binwidth))
        gluonbinlist3b.append(len(frequencylist)*bincenter/(n*binwidth))
        gluonbinlist3c.append(len(frequencylist)*bincenter/(len(gluonlist3)*binwidth))
        gluonbinlist3d.append(len(frequencylist)/(len(gluonlist3)*binwidth))

        riemann3a += len(frequencylist)/(n*binwidth)
        riemann3b += len(frequencylist)*bincenter/(n*binwidth)
        riemann3c += len(frequencylist)*bincenter/(len(gluonlist3)*binwidth)
        riemann3d += len(frequencylist)/(len(gluonlist3)*binwidth)
        
        # Calculating bins 4
        frequencylist = []
        for initialfrac in gluonlist4:
            if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                frequencylist.append(initialfrac)
        
        gluonbinlist4a.append(len(frequencylist)/(n*binwidth))
        gluonbinlist4b.append(len(frequencylist)*bincenter/(n*binwidth))
        gluonbinlist4c.append(len(frequencylist)*bincenter/(len(gluonlist4)*binwidth))
        gluonbinlist4d.append(len(frequencylist)/(len(gluonlist4)*binwidth))

        riemann4a += len(frequencylist)/(n*binwidth)
        riemann4b += len(frequencylist)*bincenter/(n*binwidth)
        riemann4c += len(frequencylist)*bincenter/(len(gluonlist4)*binwidth)
        riemann4d += len(frequencylist)/(len(gluonlist4)*binwidth)
        
    
    # Calculating solutions
    xrange = np.logspace(-3, -0.001, num=(bins*2)) #range for plotting solution
    solution1 = []
    solution2 = []
    solution3 = []
    solution4 = []
    riemann1s = 0
    riemann2s = 0
    riemann3s = 0
    riemann4s = 0
    
    for x in xrange:
        D1 = ((tau1)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau1**2)/(1-x)))
        D2 = ((tau2)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau2**2)/(1-x)))
        D3 = ((tau3)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau3**2)/(1-x)))
        D4 = ((tau4)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau4**2)/(1-x)))
        
        
        if x > 0.001:
            riemann1s += D1
            riemann2s += D2
            riemann3s += D3
            riemann4s += D4

        solution1.append(D1)
        solution2.append(D2)
        solution3.append(D3)
        solution4.append(D4)
    
    
    # Printting data to find errors. 
    riemann1a = round(riemann1a,2)
    riemann2a = round(riemann2a,2)
    riemann3a = round(riemann3a,2)
    riemann4a = round(riemann4a,2)
    riemann1b = round(riemann1b,2)
    riemann2b = round(riemann2b,2)
    riemann3b = round(riemann3b,2)
    riemann4b = round(riemann4b,2)
    riemann1c = round(riemann1c,2)
    riemann2c = round(riemann2c,2)
    riemann3c = round(riemann3c,2)
    riemann4c = round(riemann4c,2)
    riemann1d = round(riemann1d,2)
    riemann2d = round(riemann2d,2)
    riemann3d = round(riemann3d,2)
    riemann4d = round(riemann4d,2)
    
    riemann1s = round(riemann1s,2)
    riemann2s = round(riemann2s,2)
    riemann3s = round(riemann3s,2)
    riemann4s = round(riemann4s,2)
    
    print("\nLength of gluons1: ", gluons1, ". len gluonlist1: ", len(gluonlist1))
    print("\nLength of gluons2: ", gluons2, ". len gluonlist2: ", len(gluonlist2))
    print("\nLength of gluons3: ", gluons3, ". len gluonlist3: ", len(gluonlist3))
    print("\nLength of gluons4: ", gluons4, ". len gluonlist4: ", len(gluonlist4))

    
    print("\n1a: ", riemann1a, ". 1b: ", riemann1b, ". 1c: ", riemann1c, ". 1d: ", riemann1d, ". Expected: ", riemann1s, ", exact: ", round(np.exp(-np.pi*tau1**2),2))
    print("2a: ", riemann2a, ". 2b: ", riemann2b, ". 2c: ", riemann2c, ". 2d: ", riemann2d, ". Expected: ", riemann2s, ", exact: ", round(np.exp(-np.pi*tau2**2),2))
    print("3a: ", riemann3a, ". 3b: ", riemann3b, ". 3c: ", riemann3c, ". 3d: ", riemann3d, ". Expected: ", riemann3s, ", exact: ", round(np.exp(-np.pi*tau3**2),2))
    print("4a: ", riemann4a, ". 4b: ", riemann4b, ". 4c: ", riemann4c, ". 4d: ", riemann4d, ". Expected: ", riemann4s, ", exact: ", round(np.exp(-np.pi*tau4**2),2))


    # Do the actual plotting. 
    plt.figure(dpi=1000, figsize= (6,5)) #(w,h) figsize= (10,3)
    title = "Gluons in Medium. showers: " + str(n) + ". epsilon: " + str(epsilon)  
    plt.suptitle(title)

    plt.rc('axes', titlesize="small" , labelsize="x-small")     # fontsize of the axes title and labels.
    plt.rc('xtick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('ytick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('legend',fontsize='xx-small')    # fontsize of the legend labels.
    plt.rc('lines', linewidth=0.8)

    ax1 = plt.subplot(221) #H B NR
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    print("\rPlotting 1...", end="")

    ax1.plot(binlist, gluonbinlist1a, "--", label ="a")
    ax1.plot(binlist, gluonbinlist1b, "--", label ="b")
    ax1.plot(binlist, gluonbinlist1c, "--", label ="c")
    ax1.plot(binlist, gluonbinlist1d, "--", label ="d")

    ax1.plot(xrange, solution1, 'r', label="solution")
    
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title('tau = ' +str(tau1))
    ax1.set_xlim(0.001,1)
    ax1.set_ylim(0.01,10)
    ax1.set_xlabel('z ')
    ax1.set_ylabel('D(x,t)')
    ax1.grid(linestyle='dashed', linewidth=0.2)
    ax1.legend()
    
    
    print("\rPlotting 2...", end="")

    ax2.plot(binlist, gluonbinlist2a, "--", label ="a")
    ax2.plot(binlist, gluonbinlist2b, "--", label ="b")
    ax2.plot(binlist, gluonbinlist2c, "--", label ="c")
    ax2.plot(binlist, gluonbinlist2d, "--", label ="d")

    ax2.plot(xrange, solution2, 'r', label="solution")
    
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title('tau = ' +str(tau2))
    ax2.set_xlim(0.001,1)
    ax2.set_ylim(0.01,10)
    ax2.set_xlabel('z')
    ax2.set_ylabel('D(x,t)')
    ax2.grid(linestyle='dashed', linewidth=0.2)
    ax2.legend()

    
    print("\rPlotting 3...", end="")

    ax3.plot(binlist, gluonbinlist3a, "--", label ="a")
    ax3.plot(binlist, gluonbinlist3b, "--", label ="b")
    ax3.plot(binlist, gluonbinlist3c, "--", label ="c")
    ax3.plot(binlist, gluonbinlist3d, "--", label ="d")

    ax3.plot(xrange, solution3, 'r', label="solution")
    
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_title('tau = ' +str(tau3))
    ax3.set_xlim(0.001,1)
    ax3.set_ylim(0.01,10)
    ax3.set_xlabel('z ')
    ax3.set_ylabel('D(x,t)')
    ax3.grid(linestyle='dashed', linewidth=0.2)
    ax3.legend()
    

    print("\rPlotting 4...", end="")

    ax4.plot(binlist, gluonbinlist4a, "--", label ="a")
    ax4.plot(binlist, gluonbinlist4b, "--", label ="b")
    ax4.plot(binlist, gluonbinlist4c, "--", label ="c")
    ax4.plot(binlist, gluonbinlist4d, "--", label ="d")

    ax4.plot(xrange, solution4, 'r', label="solution")
    
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_title('tau = ' +str(tau4))
    ax4.set_xlim(0.001,1)
    ax4.set_ylim(0.01,10)
    ax4.set_xlabel('z ')
    ax4.set_ylabel('D(x,t)')
    ax4.grid(linestyle='dashed', linewidth=0.2)
    ax4.legend()


    print("\rShowing", end="")

    plt.tight_layout()
    plt.show()
    print("\rDone!")    
