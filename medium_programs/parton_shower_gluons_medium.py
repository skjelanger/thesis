# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Parton shower program for gluons in medium.
# Using the simplified ggg splitting kernel.

import matplotlib.pyplot as plt
from medium_splittingfunctions import gg_simple # Includes color factors.
import numpy as np
from scipy.integrate import quad

# Constants
epsilon = 10**(-3)
z_min = 10**(-3)
plot_lim = 10**(-3) # Minimum value for plot
binnumber = 100

gg_integral, __ = quad((gg_simple), epsilon, 1-epsilon)


# Define classes and class functions.
# These classes will be used for managing the different partons created in 
# different showers, throughout the rest of the program. 
# The Shower and Parton objecs will generally be deleted at the end of a given
# shower, and the required data will be written a list.
class Shower(object):
    """
    Class used for storing information about individual partons.
    
    Attributes:
        ShowerNumber (int): Numer n of the shower.
        FinalList (list): All partons with no daughters.
        FinalFracList (list): All partons with initialfrac above z_min.
        SplittingGluons (list): All gluons that can branch.
        Hardest (float): Highest InitialFrac value in FinalList.
        
    Functions: 
        select_sudakov: Selects the most probable parton to split.
        
    """

    def __init__(self, showernumber):
        self.ShowerNumber = showernumber
        self.FinalList = [] # Contains all Final Partons 
        self.FinalFracList = [] # Contains all partons above z_min.
        self.SplittingGluons = []

        self.Hardest = None
        
        
    def select_sudakov(self):
        """
        Generates a random evolution interval from the sudakov form factor
        for each of the available gluons, and returns the gluon with the lowest
        expected interval for splitting, and its interval. 
        """
        
        gluons_deltatau = []
        for gluon in self.SplittingGluons:
            delta_tau_sample = gluon.advance_time(self)
            gluons_deltatau.append((gluon, delta_tau_sample))
        (min_gluon, min_tau) =  min(gluons_deltatau, key = lambda t: t[1])
        return min_gluon, min_tau
            

class Parton(object):
    """
    Class used for storing information about individual partons.
    
    Attributes:
        Type (str): Flavour of the given parton "quark"/"gluon".
        Angle (float): Value of evolution variable at time of creation.
        InitialFrac (float): Momentum fraction of the initial parton.
        Parent (object): Parent parton. 
        Primary (object): Daughter parton with momentum fraction (z).
        Secondary (object): Daughter parton with momentum fraction (1-z).
        Shower (object): Which shower the parton is a part of.
        
    Functions:
        split(self): Samples random value from the medium ggg splitting 
                    function. 
        advance_time(self): Generates random interval where one can expect
            a splitting for the parton. Depends on momentumfraction.
    """

    def __init__(self, angle, initialfrac, parent, shower):   
        self.Type = "gluon"
        self.Angle = angle
        self.InitialFrac = initialfrac
        self.Parent = parent
        self.Primary = None
        self.Secondary = None
        self.Shower = shower
        
    def split(self):
        """Picks random value from the medium ggg splitting function."""
        rnd1 = np.random.uniform(0, 1)
        a = (rnd1-0.5)*gg_integral
        splittingvalue = 0.5 + (a/( 2* np.sqrt((16 + a**2)))) 
        return splittingvalue
    
    def advance_time(self, Shower0):
        """Randomly generates a probably value for tau for this splitting. """
        rnd1 = np.random.uniform(0,1)
        delta_tau = -2*np.sqrt(self.InitialFrac)*np.log(rnd1)/(gg_integral)
        return delta_tau
    
        

# Main shower program. 
# This program generates a single parton shower, given the initial conditions. 
# When running n showers, this program is called n times, and the each 
# iteration returns a Shower-class object.
def generate_shower(tau_max, p_t, Q_0, R, showernumber):
    """
    Main parton shower program for gluons in medium.
    
        Parameters: 
            tau_max (int): Maximum value of evolution variable.
            p_t (int) - Initital parton momentum.
            Q_0 (int) - Hadronization scale.
            R (int) - Jet radius.
            showernumber (int) - Showernumber.
            
        Returns:
            Shower (object) - Object of the class Shower.
    """
    
    tau = 0
    Shower0 = Shower(showernumber)
    Parton0 = Parton(tau, 1, None, Shower0) # Initial parton

    Shower0.SplittingGluons.append(Parton0)
    Shower0.FinalList.append(Parton0)
    
    while len(Shower0.SplittingGluons) > 0:
        SplittingParton, delta_tau = Shower0.select_sudakov()
        tau = tau + delta_tau
        
        if tau >= tau_max:
            break
        
        Shower0.SplittingGluons.remove(SplittingParton)
        Shower0.FinalList.remove(SplittingParton)
        momfrac = Parton.split(SplittingParton)            

        for j in range(0,2): # Loop for generating the new partons.
            if j==0: # Parton 1.
                initialfrac = SplittingParton.InitialFrac * momfrac
                NewParton = Parton(tau, initialfrac, SplittingParton, Shower0)
                SplittingParton.Primary = NewParton
                    
            elif j==1: # Parton 2.
                initialfrac = SplittingParton.InitialFrac * (1-momfrac)
                NewParton = Parton(tau, initialfrac, SplittingParton, Shower0)
                SplittingParton.Secondary = NewParton
                
            Shower0.FinalList.append(NewParton)

            if initialfrac > z_min: # Limit on how soft gluons can split.
                Shower0.SplittingGluons.append(NewParton)  
                   
                
    for PartonObj in Shower0.FinalList:
        if PartonObj.InitialFrac > plot_lim:
            Shower0.FinalFracList.append(PartonObj.InitialFrac)
        del PartonObj
    
    try:
        Shower0.Hardest = max(Shower0.FinalFracList)
    except:
        print("ERROR, all partons radiated to soft.")

    return Shower0


# Program for comparing the shower program with analytical calculations.
# Four different values of tau are used, and n showers are generated for each
# of them. The results from the showers then compared to analytical results, 
# and plotted in the same figure.
def several_showers_analytical_comparison(n, opt_title, scale):
    """
    Runs n parton showers, and compares the analytical results.
    
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
    
    tau1 = 0.1
    tau2 = 0.2
    tau3 = 0.3
    tau4 = 0.4
    
    #Generating showers
    gluonlist1 = []
    gluonlist2 = []
    gluonlist3 = []
    gluonlist4 = []
    
    gluons1 = 0
    gluons2 = 0
    gluons3 = 0
    gluons4 = 0    

    for i in range (0,4*n):
        print("\rLooping... " + str(round(100*i/(4*n))) +"%", end="")

        if (0 <= i and i < n):
            Shower0 = generate_shower(tau1, p_0, Q_0, R, i)
            gluonlist1.extend(Shower0.FinalFracList)
            gluons1 += len(Shower0.FinalList)
            del Shower0

        if (n <= i and i < 2*n):
            Shower0 = generate_shower(tau2, p_0, Q_0, R, i)            
            gluonlist2.extend(Shower0.FinalFracList)
            gluons2 += len(Shower0.FinalList)
            del Shower0

        if (2*n <= i and i < 3*n):
            Shower0 = generate_shower(tau3, p_0, Q_0, R, i)
            gluonlist3.extend(Shower0.FinalFracList)
            gluons3 += len(Shower0.FinalList)
            del Shower0

        if (3*n <= i and i < 4*n):
            Shower0 = generate_shower(tau4, p_0, Q_0, R, i)
            gluonlist4.extend(Shower0.FinalFracList)
            gluons4 += len(Shower0.FinalList)
            del Shower0


    # Sets the different ranges required for the plots.
    
    if scale == "lin":
        linbins1 = (np.linspace(plot_lim, 0.99, num=binnumber))
        linbins2 = (np.linspace(0.991, 1, 10))
        bins = np.hstack((linbins1, linbins2))
        xrange = np.linspace(plot_lim, 0.9999, num=(4*binnumber))

    
    elif scale == "log":
        logbins1 = np.logspace(-3, -0.1, num=binnumber)
        logbins2 = np.logspace(-0.09, 0, num = 10)
        bins = np.hstack((logbins1, logbins2))
        xrange = np.logspace(-3, -0.0001, num=(4*binnumber))

    binlist = []

    # Normalizing the showers.
    print("\rCalculating bins...", end="")
    
    gluonbinlist1 = []
    gluonbinlist2 = []
    gluonbinlist3 = []
    gluonbinlist4 = []

    for i in range(len(bins)-1):
        binwidth = bins[i+1]-bins[i]
        bincenter = bins[i+1] - (binwidth/2)
        binlist.append(bincenter)
                
        # Calculating bins 1
        frequencylist = []
        for initialfrac in gluonlist1:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist.append(initialfrac)                
        binvalue = len(frequencylist)*bincenter/(n*binwidth)
        gluonbinlist1.append(binvalue)

        # Calculating bins 2
        frequencylist = []
        for initialfrac in gluonlist2:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist.append(initialfrac)                
        binvalue = len(frequencylist)*bincenter/(n*binwidth)
        gluonbinlist2.append(binvalue)

        # Calculating bins 3
        frequencylist = []
        for initialfrac in gluonlist3:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist.append(initialfrac)
        binvalue = len(frequencylist)*bincenter/(n*binwidth)
        gluonbinlist3.append(binvalue)

        # Calculating bins 4
        frequencylist = []
        for initialfrac in gluonlist4:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist.append(initialfrac)
        binvalue = len(frequencylist)*bincenter/(n*binwidth)
        gluonbinlist4.append(binvalue)
        
    
    # Calculating solutions
    solution1 = []
    solution2 = []
    solution3 = []
    solution4 = []
    
    for x in xrange:
        D1 = ((tau1)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau1**2)/(1-x)))
        D2 = ((tau2)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau2**2)/(1-x)))
        D3 = ((tau3)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau3**2)/(1-x)))
        D4 = ((tau4)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau4**2)/(1-x)))
        solution1.append(D1)
        solution2.append(D2)
        solution3.append(D3)
        solution4.append(D4)
    
    # Do the actual plotting. 
    plt.figure(dpi=1000, figsize= (6,5)) #(w,h) figsize= (10,3)
    title = ("Medium showers: " + str(n) + 
             ". epsilon: " + str(epsilon) + 
             ". z_min: " + str(z_min) +
             "\n " + opt_title)    
    #plt.suptitle(title)

    plt.rc('axes', titlesize="small" , labelsize="x-small")     # fontsize of the axes title and labels.
    plt.rc('xtick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('ytick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('legend',fontsize='xx-small')    # fontsize of the legend labels.
    plt.rc('lines', linewidth=0.8)

    ax1 = plt.subplot(221) #H B NR
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    print("\rPlotting...", end="")

    ax1.plot(binlist, gluonbinlist1, "--", label="MC")
    ax1.plot(xrange, solution1, 'r', label="solution")
    ax1.set_title('tau = ' +str(tau1))
    ax1.set_xlim(plot_lim,1)
    ax1.set_ylim(0.01,10)
    ax1.set_xlabel('z ')
    ax1.set_ylabel('D(x,t)')
    ax1.grid(linestyle='dashed', linewidth=0.2)
    ax1.legend()
    
    
    ax2.plot(binlist, gluonbinlist2, "--", label="MC")
    ax2.plot(xrange, solution2, 'r', label="solution")
    ax2.set_title('tau = ' +str(tau2))
    ax2.set_xlim(plot_lim,1)
    ax2.set_ylim(0.01,10)
    ax2.set_xlabel('z')
    ax2.set_ylabel('D(x,t)')
    ax2.grid(linestyle='dashed', linewidth=0.2)
    ax2.legend()

    
    ax3.plot(binlist, gluonbinlist3, "--", label="MC")
    ax3.plot(xrange, solution3, 'r', label="solution")
    ax3.set_title('tau = ' +str(tau3))
    ax3.set_xlim(plot_lim,1)
    ax3.set_ylim(0.01,10)
    ax3.set_xlabel('z ')
    ax3.set_ylabel('D(x,t)')
    ax3.grid(linestyle='dashed', linewidth=0.2)
    ax3.legend()
    

    ax4.plot(binlist, gluonbinlist4, "--", label="MC")
    ax4.plot(xrange, solution4, 'r', label="solution")
    ax4.set_title('tau = ' +str(tau4))
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

    print("\rShowing")

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

    if n_error or title_error or scale_error :
        error = True
        if n_error:
            msg = msg + "\nERROR! - 'n' must be an integer."
        if title_error:
            msg = msg + "\nERROR! - 'opt_title' must be a str."
        if scale_error:
            msg = msg+ "\nERROR! - 'scale' must be 'lin' or 'log'."
    return error, msg