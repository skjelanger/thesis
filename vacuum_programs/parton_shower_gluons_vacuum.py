# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Parton shower program for gluons in vacuum.
# Using the simplified ggg splitting kernel.

import matplotlib.pyplot as plt
import vacuum_splittingfunctions as sf # Includes color factors.

import numpy as np
import random
from scipy.integrate import quad 
from treelib import Tree

#constants
epsilon = 10**(-3)

bins = 150
minimum_bin = 0.001

gg_integral, __ = quad((sf.gg_simple), epsilon, 1-epsilon)


# Define classes and class functions.
# These classes will be used for managing the different partons created in 
# different showers, throughout the rest of the program. 
# The Shower and Parton objecs will generally be deleted at the end of a given
# shower, and the required data will be written a list.
class IterShower(type):
    """This class is a metaclass for Shower, and serves to make the class 
    Shower iterable - meaning you can iterate a list of all the objects. 
    This is only used for the Treelib plotting, if desired. """
    def __iter__(cls):
        return iter(cls.allShowers)
    
class Shower(object, metaclass= IterShower):
    """
    Class used for storing information about individual partons.
    
    Attributes:
        ShowerNumber (int): Numer n of the shower.
        PartonList (list): All partons that are part of the shower.
        FinalList (list): All partons with no daughters.
        FinalFracList (list): All partons with initialfrac above z_min.
        SplittingGluons (list): All gluons that can branch.
        Hardest (float): Highest InitialFrac value in FinalList.
        
    Other:
        allShowers (list): Contains all shower objects.
    """

    allShowers = []
    def __init__(self, showernumber):
        self.allShowers.append(self)

        self.ShowerNumber = showernumber
        self.PartonList = [] # List for all Shower Partons. Used for treelib.
        self.FinalList = [] # Contains all Final Partons
        self.FinalFracList = [] # Contains all partons above min_value.
        self.SplittingGluons = []
        self.Hardest = None

class Parton(object):
    """
    Class used for storing information about individual partons.
    
    Attributes:
        Type (str): Flavour of the given parton "quark"/"gluon".
        Angle (float): Value of evolution variable at time of creation.
        InitialFrac (float): Momentum fraction of the initial parton momenta.
        Parent (object): Parent parton. 
        Primary (object): Daughter parton with momentum fraction (z).
        Secondary (object): Daughter parton with momentum fraction (1-z).
        Shower (object): Which shower the parton is a part of.
    """
    
    def __init__(self, angle, initialfrac, parent, shower):        
        """Constructs all the necessary attributes for the Parton object."""
        
        self.Type = "gluon"
        self.Angle = angle
        self.InitialFrac = initialfrac
        self.Parent = parent
        self.Primary = None
        self.Secondary = None
        self.Shower = shower

    def split(self):
        """Picks random value from the vacuum gg splitting function."""
        rnd1 = np.random.uniform(0,1)
        xi = ((1-epsilon)/epsilon)**((2*rnd1)-1)
        splittingvalue = xi/(1+xi) 
        return splittingvalue
    
    
# Program for calculating probably splitting interval from the Sudakov. 
# In the medium programs this value is dependent on what parton is splitting. 
# This is not the case here, so it is not a Parton-class function.
def advance_time():
    """Randomly generates a probably value t for this splitting. """
    rnd1 = np.random.uniform(0,1)
    delta_t = -(np.log(rnd1))/(gg_integral)
    return delta_t
                

# Main shower program. 
# This program generates a single parton shower, given the initial conditions. 
# When running n showers, this program is called n times, and the each 
# iteration returns a Shower-class object.
def generate_shower(t_max, p_t, Q_0, R, showernumber):
    """
    Main parton shower program for gluons in vacuum.
    
        Parameters: 
            t_max (int): Maximum value of evolution variable.
            p_t (int) - Initital parton momentum.
            Q_0 (int) - Hadronization scale.
            R (int) - Jet radius.
            showernumber (int) - Showernumber.
            
        Returns:
            Shower (object) - Object of the class Shower.
    """
        
    t = 0
    Shower0 = Shower(showernumber)
    Parton0 = Parton(t, 1, None, Shower0) # Initial parton
    
    Shower0.PartonList.append(Parton0)
    Shower0.FinalList.append(Parton0)
    Shower0.SplittingGluons.append(Parton0)
        
    while True:
        delta_t = advance_time() 
        t = t + delta_t
        
        no_branching = (t >= t_max or len(Shower0.SplittingGluons) == 0)
        if no_branching:
            break
            
        SplittingParton = random.choice(Shower0.SplittingGluons)
        
        
        Shower0.SplittingGluons.remove(SplittingParton)
        Shower0.FinalList.remove(SplittingParton)
        momfrac = Parton.split(SplittingParton)

        for j in range(0,2): # Loop for generating the new partons.
            if j==0: # Parton 1.
                initialfrac = SplittingParton.InitialFrac * momfrac
                NewParton = Parton(t, initialfrac, SplittingParton,
                                       Shower0)
                SplittingParton.Primary = NewParton
                    
            elif j==1: # Parton 2.
                initialfrac = SplittingParton.InitialFrac * (1-momfrac)
                NewParton = Parton(t, initialfrac, SplittingParton,
                                       Shower0)
                SplittingParton.Secondary = NewParton
                
            Shower0.FinalList.append(NewParton)

            if initialfrac > 0: # Limit on how soft gluons can split.
                Shower0.SplittingGluons.append(NewParton)

    
    for PartonObj in Shower0.FinalList:
        if PartonObj.InitialFrac > minimum_bin:
            Shower0.FinalFracList.append(PartonObj.InitialFrac)
        del PartonObj

    Shower0.Hardest = max(Shower0.FinalFracList)
    
    return Shower0


# Parton tree program.
def create_parton_tree(showernumber):
    """This program can draw a tree of the partons. It is really only practical
    for vacuum showers. Creates the tree fora given showernumber."""
    tree = Tree()
    for ShowerObj in Shower: 
        if ShowerObj.ShowerNumber == showernumber:
            print("ShowerNumber is:", ShowerObj.ShowerNumber, 
                  ". Number of partons is: ", len(ShowerObj.FinalList))
        
            for PartonObj in ShowerObj.PartonList: # Loop for creating tree.
                initialfrac = str(round(PartonObj.InitialFrac, 3))
                title = initialfrac + " - " + PartonObj.Type
                tree.create_node(title, PartonObj, parent= PartonObj.Parent)

    tree.show()
    tree.remove_node(tree.root)
    return
    


# Program for comparing the shower program with analytical calculations.
# Four different values of tau are used, and n showers are generated for each
# of them. The results from the showers then compared to analytical results, 
# and plotted in the same figure.
def several_showers_analytical_comparison(n, opt_title):
    """
    Runs n parton showers, and compares the result with the analytical results.
    
    Parameters: 
        n (int): Number of showers to simulate.
        opt_title (str): Additional title to add to final plot.
        
    Returns:
        A very nice plot. 
    """
    
    R = 0.4 # Jet radius.    
    p_0 = 100 # Initial parton momentum.
    Q_0 = 1 # Hadronization scale.
    
    t1 = 0.04
    t2 = 0.1
    t3 = 0.2
    t4 = 0.3
    
    #Generating showers
    gluonlist1 = []
    gluonhard1 = []
    gluonlist2 = []
    gluonhard2 = []
    gluonlist3 = []
    gluonhard3 = []
    gluonlist4 = []
    gluonhard4 = []
        
    
    for i in range(0,4*n):
        print("\rLooping... "+ str(round(100*i/(4*n))) + "%",end="")

        if (0 <= i and i < n):
            Shower0 = generate_shower(t1, p_0, Q_0, R, i)
            gluonhard1.append(Shower0.Hardest)
            gluonlist1.extend(Shower0.FinalFracList)
            del Shower0

        
        if (n <= i and i < 2*n):
            Shower0 = generate_shower(t2, p_0, Q_0, R, i)            
            gluonhard2.append(Shower0.Hardest)
            gluonlist2.extend(Shower0.FinalFracList)
            del Shower0
  
        if (2*n <= i and i < 3*n):
            Shower0 = generate_shower(t3, p_0, Q_0, R, i)
            gluonhard3.append(Shower0.Hardest)
            gluonlist3.extend(Shower0.FinalFracList)
            del Shower0
        
        if (3*n <= i and i < 4*n):
            Shower0 = generate_shower(t4, p_0, Q_0, R, i)
            gluonhard4.append(Shower0.Hardest)
            gluonlist4.extend(Shower0.FinalFracList)
            del Shower0

    # Normalizing showers
    logbins = np.logspace(-3, 0, num=bins)
    #logbins.extend(np.logspace(-3, 0, num=bins))
    binlist = []

    print("\rCalculating bins...", end="")
    
    gluonbinhardest1 = []
    gluonbinhardest2 = []
    gluonbinhardest3 = []
    gluonbinhardest4 = []
    
    gluonbinlistdensity1a = []
    gluonbinlistdensity1b = []
    
    gluonbinlistdensity2a = []
    gluonbinlistdensity2b = []
    
    gluonbinlistdensity3a = []
    gluonbinlistdensity3b = []

    gluonbinlistdensity4a = []
    gluonbinlistdensity4b = []

    
    for i in range(len(logbins)-1):
        binwidth = logbins[i+1]-logbins[i]
        bincenter = logbins[i+1] - (binwidth/2)
        binlist.append(bincenter)
        
        
        # Calculating bins 1
        frequencylist1 = []
        frequencylist2 = []
        for initialfrac in gluonlist1:
            if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                frequencylist1.append(initialfrac)
        gluonbinlistdensity1a.append(len(frequencylist1)*bincenter/(len(gluonlist1)*binwidth))
        gluonbinlistdensity1b.append(len(frequencylist1)*bincenter/(n*binwidth))

        
        for initialfrac in gluonhard1:
            if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                frequencylist2.append(initialfrac)
        binharddensity = len(frequencylist2)/(n*binwidth) # (len(gluonlist1)*(binwidth))
        gluonbinhardest1.append(binharddensity*bincenter)
    
        # Calculating bins 2
        frequencylist1 = []
        frequencylist2 = []
        for initialfrac in gluonlist2:
            if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                frequencylist1.append(initialfrac)
        gluonbinlistdensity2a.append(len(frequencylist1)*bincenter/(len(gluonlist2)*binwidth))
        gluonbinlistdensity2b.append(len(frequencylist1)*bincenter/(n*binwidth))

        
        for initialfrac in gluonhard2:
            if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                frequencylist2.append(initialfrac)
        binharddensity = len(frequencylist2)/(n*binwidth) # (len(gluonlist2)*(binwidth))
        gluonbinhardest2.append(binharddensity*bincenter)
        
        # Calculating bins 3
        frequencylist1 = []
        frequencylist2 = []
        for initialfrac in gluonlist3:
            if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                frequencylist1.append(initialfrac)
        gluonbinlistdensity3a.append(len(frequencylist1)*bincenter/(len(gluonlist3)*binwidth))
        gluonbinlistdensity3b.append(len(frequencylist1)*bincenter/(n*binwidth))

        for initialfrac in gluonhard3:
            if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                frequencylist2.append(initialfrac)
        binharddensity = len(frequencylist2)/(n*binwidth) # (len(gluonlist3)*(binwidth))
        gluonbinhardest3.append(binharddensity*bincenter)
        
        # Calculating bins 4
        frequencylist1 = []
        frequencylist2 = []
        for initialfrac in gluonlist4:
            if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                frequencylist1.append(initialfrac)
        gluonbinlistdensity4a.append(len(frequencylist1)*bincenter/(len(gluonlist4)*binwidth))
        gluonbinlistdensity4b.append(len(frequencylist1)*bincenter/(n*binwidth))

        for initialfrac in gluonhard4:
            if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                frequencylist2.append(initialfrac)
        binharddensity = len(frequencylist2)/(n*binwidth) # (len(gluonlist4)*(binwidth))
        gluonbinhardest4.append(binharddensity*bincenter)
    
    
    # Calculating solutions
    xrange = np.logspace(-3, -0.001, num=100) #range for plotting solution
    solution1 = []
    solution2 = []
    solution3 = []
    solution4 = []
    
    for x in xrange:
        D1 = (1/2)*(t1/((np.pi**2) * (np.log(1/x))**3))**(1/4) * np.exp(-t1*np.exp(1) + 2*np.sqrt(t1*(np.log(1/x))))
        D2 = (1/2)*(t2/((np.pi**2) * (np.log(1/x))**3))**(1/4) * np.exp(-t2*np.exp(1) + 2*np.sqrt(t2*(np.log(1/x))))
        D3 = (1/2)*(t3/((np.pi**2) * (np.log(1/x))**3))**(1/4) * np.exp(-t3*np.exp(1) + 2*np.sqrt(t3*(np.log(1/x))))
        D4 = (1/2)*(t4/((np.pi**2) * (np.log(1/x))**3))**(1/4) * np.exp(-t4*np.exp(1) + 2*np.sqrt(t4*(np.log(1/x))))
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

    ax1.plot(binlist, gluonbinlistdensity1a, '--', label ="/showers ")
    ax1.plot(binlist, gluonbinlistdensity1b, '--', label ="/gluons")
    
    #ax1.plot(binlist, gluonbinhardest1, 'g--')
    ax1.plot(xrange, solution1, 'r', label="solution")
    
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    ax1.set_title('t_0 = ' + str(t1))
    ax1.set_xlim(0.001,1)
    ax1.set_ylim(0.01,10)
    ax1.set_xlabel('z ')
    ax1.set_ylabel('D(x,t)')
    ax1.grid(linestyle='dashed', linewidth=0.2)
    ax1.legend()
    
    
    print("\rPlotting 2...", end="")

    ax2.plot(binlist, gluonbinlistdensity2a, '--', label ="/showers ")
    ax2.plot(binlist, gluonbinlistdensity2b, '--', label ="/gluons")

    #ax2.plot(binlist, gluonbinhardest2, 'g--')
    ax2.plot(xrange, solution2, 'r', label="solution")
    
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax2.set_title('t_0 = ' + str(t2))
    ax2.set_xlim(0.001,1)
    ax2.set_ylim(0.01,10)
    ax2.set_xlabel('z')
    ax2.set_ylabel('D(x,t)')
    ax2.grid(linestyle='dashed', linewidth=0.2)
    ax2.legend()

    
    print("\rPlotting 3...", end="")

    ax3.plot(binlist, gluonbinlistdensity3a, '--', label ="/showers ")
    ax3.plot(binlist, gluonbinlistdensity3b, '--', label ="/gluons")
    
    #ax3.plot(binlist, gluonbinhardest3, 'g--')
    ax3.plot(xrange, solution3, 'r', label="solution")
    
    ax3.set_xscale("log")
    ax3.set_yscale("log")

    ax3.set_title('t_0 = ' + str(t3))
    ax3.set_xlim(0.001,1)
    ax3.set_ylim(0.01,10)
    ax3.set_xlabel('z ')
    ax3.set_ylabel('D(x,t)')
    ax3.grid(linestyle='dashed', linewidth=0.2)
    ax3.legend()
    

    print("\rPlotting 4...", end="")

    ax4.plot(binlist, gluonbinlistdensity4a, '--', label ="/showers ")
    ax4.plot(binlist, gluonbinlistdensity4b, '--', label ="/gluons")

    #ax4.plot(binlist, gluonbinhardest4, 'g--')
    ax4.plot(xrange, solution4, 'r', label="solution")
    
    ax4.set_xscale("log")
    ax4.set_yscale("log")

    ax4.set_title('t_0 = ' + str(t4))
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
        