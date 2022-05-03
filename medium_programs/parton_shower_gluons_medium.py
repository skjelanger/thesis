# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Parton shower program for gluons in medium.
# Using the simplified ggg splitting kernel.

import matplotlib.pyplot as plt
from medium_splittingfunctions import gg_simple_analytical
import numpy as np
from scipy.integrate import quad

# Constants
epsilon = 10**(-3)
z_min = 10**(-3)
plot_lim = 10**(-3)
binnumber = 100

gg_integral, __ = quad((gg_simple_analytical), epsilon, 1-epsilon)


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
        SplittingPartons (list): All gluons that can branch.
        Hardest (float): Highest InitialFrac value in FinalList.
        
    Functions: 
        select_splitting_parton: Selects the most probable parton to split.
        loop_status: Checks when to end the different tau showers.
        
    """

    def __init__(self, showernumber):
        self.ShowerNumber = showernumber
        self.FinalList = [] # Contains all Final Partons 
        self.FinalFracList1 = [] # Contains all partons above z_min.
        self.FinalFracList2 = [] # Contains all partons above z_min.
        self.FinalFracList3 = [] # Contains all partons above z_min.
        self.FinalFracList4 = [] # Contains all partons above z_min.
        self.SplittingPartons = []

        self.Loops = (True, True, True, True)
        self.Hardest = None
        
    def select_splitting_parton(self):
        """
        Generates a random evolution interval from the sudakov form factor
        for each of the available gluons, and returns the gluon with the lowest
        expected interval for splitting, and its interval. 
        """
        
        gluons_deltatau = []
        for gluon in self.SplittingPartons:
            delta_tau_sample = gluon.advance_time(self)
            gluons_deltatau.append((gluon, delta_tau_sample))
        (min_gluon, min_tau) =  min(gluons_deltatau, key = lambda t: t[1])
        return min_gluon, min_tau
    
 
    def loop_status(self, tau, tauvalues):
        """
        Loop conditions for generating showers for the four values of tau,
        without having to restart the shower every time.  
        """
        end_shower = False
        if tau > tauvalues[0] and self.Loops[0]: 
            self.Loops = (False, True, True, True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList1.append(PartonObj.InitialFrac) 
                    
        if tau > tauvalues[1] and self.Loops[1]:
            self.Loops = (False, False, True, True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList2.append(PartonObj.InitialFrac) 
                    
        if tau > tauvalues[2] and self.Loops[2]:
            self.Loops = (False, False, False, True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList3.append(PartonObj.InitialFrac) 
                    
        if tau > tauvalues[3] and self.Loops[3]:
            self.Loops = (False, False, False, False)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList4.append(PartonObj.InitialFrac)   
                del PartonObj
            end_shower = True
            
        return end_shower
            

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
def generate_shower(tauvalues, p_t, Q_0, R, showernumber):
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
    Shower0.SplittingPartons.append(Parton0)
    Shower0.FinalList.append(Parton0)
    
    while len(Shower0.SplittingPartons) > 0:
        SplittingParton, delta_tau = Shower0.select_splitting_parton()
        tau = tau + delta_tau
        
        end_shower = Shower0.loop_status(tau, tauvalues)
        if end_shower:
            break
        
        Shower0.SplittingPartons.remove(SplittingParton)
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
                Shower0.SplittingPartons.append(NewParton)  

    try:
        Shower0.Hardest = max(Shower0.FinalFracList4)
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
    tauvalues = (tau1, tau2, tau3, tau4)
    
    #Generating showers
    gluonlists = [[],[],[],[]]

    for i in range (1,n):
        print("\rLooping... " + str(round(100*i/(n),2)) +"%", end="")

        Shower0 = generate_shower(tauvalues, p_0, Q_0, R, i)
        gluonlists[0].extend(Shower0.FinalFracList1)
        gluonlists[1].extend(Shower0.FinalFracList2)
        gluonlists[2].extend(Shower0.FinalFracList3)
        gluonlists[3].extend(Shower0.FinalFracList4)        
        del Shower0

    # Sets the different ranges required for the plots.
    if scale == "lin":
        linbins1 = (np.linspace(plot_lim, 0.99, num=binnumber))
        linbins2 = (np.linspace(0.991, 1, num= round((binnumber/4))))
        bins = np.hstack((linbins1, linbins2))
        xrange = np.linspace(plot_lim, 0.9999, num=(4*binnumber))
        
    elif scale == "log":
        logbins1 = np.logspace(-3, -0.1, num=binnumber)
        logbins2 = np.logspace(-0.09, 0, num = 10)
        bins = np.hstack((logbins1, logbins2))
        xrange = np.logspace(-3, -0.0001, num=(4*binnumber))


    # Normalizing the showers.
    print("\rCalculating bins...", end="")
    binlist = []
    gluonbinlists = [[],[],[],[]]

    gluontzs = [0,0,0,0]

    for i in range(len(bins)-1):
        binwidth = bins[i+1]-bins[i]
        bincenter = bins[i+1] - (binwidth/2)
        binlist.append(bincenter)
        
        for gluonlist in gluonlists:
            index = gluonlists.index(gluonlist)
            frequencylist = []
            for initialfrac in gluonlist:
                if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                    frequencylist.append(initialfrac)       
                    if initialfrac ==1:
                        gluontzs[index] +=1             
            binvalue = len(frequencylist)*bincenter/(n*binwidth)
            gluonbinlists[index].append(binvalue)
    
    print("gluontzs = ", gluontzs)
    
    # Calculating solutions
    solutions = [[],[],[],[]]
    
    for x in xrange:
        D1 = ((tau1)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau1**2)/(1-x)))
        D2 = ((tau2)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau2**2)/(1-x)))
        D3 = ((tau3)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau3**2)/(1-x)))
        D4 = ((tau4)/(np.sqrt(x)*((1-x))**(3/2)) )* np.exp(-np.pi*((tau4**2)/(1-x)))
        solutions[0].append(D1)
        solutions[1].append(D2)
        solutions[2].append(D3)
        solutions[3].append(D4)
    
    # Do the actual plotting. 
    plt.figure(dpi=1000, figsize= (6,5)) #(w,h) figsize= (10,3)
    title = ("Medium showers: " + str(n) + 
             ". epsilon: " + str(epsilon) + 
             ". z_min: " + str(z_min) +
             "\n" + opt_title)    
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
    
    axes = [ax1, ax2, ax3, ax4]
    
    print("\rPlotting..." + 10*" ", end="")

    for ax in axes:
        index = axes.index(ax)
        
        ax.plot(binlist, gluonbinlists[index], "--", label="MC")
        ax.plot(xrange, solutions[index], 'r', label="solution")
        ax.set_title('tau = ' +str(tauvalues[index]))
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


    print("\rShowing" + 10*" ")

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

    if n_error or title_error or scale_error :
        error = True
        if n_error:
            msg = msg + "\nERROR! - 'n' must be an integer."
        if title_error:
            msg = msg + "\nERROR! - 'opt_title' must be a str."
        if scale_error:
            msg = msg+ "\nERROR! - 'scale' must be 'lin' or 'log'."
    return error, msg