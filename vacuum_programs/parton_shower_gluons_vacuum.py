# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Parton shower program for gluons in vacuum.
# Using the simplified ggg splitting kernel.

import matplotlib.pyplot as plt
import vacuum_splittingfunctions as sf # Includes color factors.
import numpy as np
from scipy.integrate import quad 
from treelib import Tree

#constants
epsilon = 10**(-3)
z_min = 10**(-3)
plot_lim = 10**(-3)
binnumber = 100

gg_integral, __ = quad((sf.gg_simple_analytical), epsilon, 1-epsilon)
print("epsilon: ", epsilon)
print("gg_integral: ", gg_integral)

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
        SplittingPartons (list): All gluons that can branch.
        Hardest (float): Highest InitialFrac value in FinalList.
        
    Functions: 
        select_splitting_parton: Selects the most probable parton to split.
        loop_status: Checks when to end the different tau showers.

    Other:
        allShowers (list): Contains all shower objects.
    """

    allShowers = []
    def __init__(self, showernumber):
        self.allShowers.append(self)

        self.ShowerNumber = showernumber
        self.PartonList = [] # List for all Shower Partons. Used for treelib.
        self.FinalList = [] # Contains all Final Partons
        self.FinalFracList1 = [] # Contains all partons above z_min.
        self.FinalFracList2 = [] # Contains all partons above z_min.
        self.FinalFracList3 = [] # Contains all partons above z_min.
        self.FinalFracList4 = [] # Contains all partons above z_min.
        self.Hardest1 = None
        self.Hardest2 = None
        self.Hardest3 = None
        self.Hardest4 = None
        self.SplittingPartons = []
        self.Loops = (True, True, True, True)
        
    def select_splitting_parton(self):
        """
        Generates a random evolution interval from the sudakov form factor
        for each of the available gluons, and returns the gluon with the lowest
        expected interval for splitting, and its interval. 
        """
        gluons_deltat = []
        for gluon in self.SplittingPartons:
            delta_t_sample = gluon.advance_time()
            gluons_deltat.append((gluon, delta_t_sample))
        (SplittingParton, delta_t) =  min(gluons_deltat, key = lambda t: t[1])
        return SplittingParton, delta_t

    def loop_status(self, t, tvalues):
        """
        Loop conditions for generating showers for the four values of tau,
        without having to restart the shower every time.  
        """
        end_shower = False
        if t > tvalues[0] and self.Loops[0]: 
            self.Loops = (False, True, True, True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList1.append(PartonObj.InitialFrac)  
            self.Hardest1 = max(self.FinalFracList1)

        if t > tvalues[1] and self.Loops[1]:
            self.Loops = (False, False, True, True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList2.append(PartonObj.InitialFrac) 
            self.Hardest2 = max(self.FinalFracList2)
                    
        if t > tvalues[2] and self.Loops[2]:
            self.Loops = (False, False, False, True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList3.append(PartonObj.InitialFrac) 
            self.Hardest3 = max(self.FinalFracList3)

        if t > tvalues[3] and self.Loops[3]:
            self.Loops = (False, False, False, False)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList4.append(PartonObj.InitialFrac)  
                del PartonObj
            self.Hardest4 = max(self.FinalFracList4)
            end_shower = True
            
        return end_shower

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
        
    Functions:
        split(self): Samples random value from the medium ggg splitting 
            function. 
        advance_time(self): Generates random interval where one can expect
            a splitting for the parton. Does not depend on momentumfraction.
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
        """Picks random value from the vacuum gg splitting function."""
        rnd1 = np.random.uniform(0,1)
        xi = ((1-epsilon)/epsilon)**((2*rnd1)-1)
        splittingvalue = xi/(1+xi) 
        return splittingvalue

    def advance_time(self):
        """Randomly generates a probably value t for this splitting. """
        rnd1 = np.random.uniform(0,1)
        delta_t = -(np.log(rnd1))/(gg_integral)
        return delta_t
                

# Main shower program. 
# This program generates a single parton shower, given the initial conditions. 
# When running n showers, this program is called n times, and the each 
# iteration returns a Shower-class object.
def generate_shower(tvalues, p_t, Q_0, R, showernumber):
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
    Shower0.SplittingPartons.append(Parton0)
        
    while len(Shower0.SplittingPartons) > 0:
        SplittingParton, delta_t = Shower0.select_splitting_parton()
        t = t + delta_t
        
        end_shower = Shower0.loop_status(t, tvalues)
        if end_shower:
            break
            
        Shower0.SplittingPartons.remove(SplittingParton)
        Shower0.FinalList.remove(SplittingParton)
        momfrac = SplittingParton.split()

        for j in range(0,2): # Loop for generating the new partons.
            if j==0: # Parton 1.
                initialfrac = SplittingParton.InitialFrac * momfrac
                NewParton = Parton(t, initialfrac, SplittingParton, Shower0)
                SplittingParton.Primary = NewParton
                    
            elif j==1: # Parton 2.
                initialfrac = SplittingParton.InitialFrac * (1-momfrac)
                NewParton = Parton(t, initialfrac, SplittingParton, Shower0)
                SplittingParton.Secondary = NewParton
            
            Shower0.FinalList.append(NewParton)
            if NewParton.InitialFrac > z_min:
                Shower0.SplittingPartons.append(NewParton)
    
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
def several_showers_vacuum_analytical_comparison(n, opt_title, scale):
    """
    Runs n parton showers, and compares the result with the analytical results.
    
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
    gluonlist1 = []
    gluonhard1 = []
    gluonlist2 = []
    gluonhard2 = []
    gluonlist3 = []
    gluonhard3 = []
    gluonlist4 = []
    gluonhard4 = []
        
    for i in range(1,n):
        print("\rLooping... "+ str(round(100*i/(4*n),1)) + "%",end="")
        Shower0 = generate_shower(tvalues, p_0, Q_0, R, i)
        gluonhard1.append(Shower0.Hardest1)
        gluonhard2.append(Shower0.Hardest2)
        gluonhard3.append(Shower0.Hardest3)
        gluonhard4.append(Shower0.Hardest4)
        gluonlist1.extend(Shower0.FinalFracList1)
        gluonlist2.extend(Shower0.FinalFracList2)
        gluonlist3.extend(Shower0.FinalFracList3)
        gluonlist4.extend(Shower0.FinalFracList4)  
        del Shower0
    
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
    
    gluonbinlist1 = []
    gluonbinlist2 = []
    gluonbinlist3 = []
    gluonbinlist4 = []
    gluonbinhardest1 = []
    gluonbinhardest2 = []
    gluonbinhardest3 = []
    gluonbinhardest4 = []
    
    for i in range(len(bins)-1):
        binwidth = bins[i+1]-bins[i]
        bincenter = bins[i+1] - (binwidth/2)
        binlist.append(bincenter)
        
        # Calculating bins 1
        frequencylist1 = []
        frequencylist2 = []
        for initialfrac in gluonlist1:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist1.append(initialfrac)
        gluondensity = len(frequencylist1)*bincenter/(n*binwidth)
        gluonbinlist1.append(gluondensity)
        
        for initialfrac in gluonhard1:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist2.append(initialfrac)
        binharddensity = len(frequencylist2)*bincenter/(n*binwidth)
        gluonbinhardest1.append(binharddensity)
    
        # Calculating bins 2
        frequencylist1 = []
        frequencylist2 = []
        for initialfrac in gluonlist2:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist1.append(initialfrac)
        gluondensity = len(frequencylist1)*bincenter/(n*binwidth)
        gluonbinlist2.append(gluondensity)

        
        for initialfrac in gluonhard2:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist2.append(initialfrac)
        binharddensity = len(frequencylist2)*bincenter/(n*binwidth)
        gluonbinhardest2.append(binharddensity)
        
        # Calculating bins 3
        frequencylist1 = []
        frequencylist2 = []
        for initialfrac in gluonlist3:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist1.append(initialfrac)
        gluondensity = len(frequencylist1)*bincenter/(n*binwidth)
        gluonbinlist3.append(gluondensity)

        for initialfrac in gluonhard3:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist2.append(initialfrac)
        binharddensity = len(frequencylist2)*bincenter/(n*binwidth)
        gluonbinhardest3.append(binharddensity)
        
        # Calculating bins 4
        frequencylist1 = []
        frequencylist2 = []
        for initialfrac in gluonlist4:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist1.append(initialfrac)
        gluondensity = len(frequencylist1)*bincenter/(n*binwidth)
        gluonbinlist4.append(gluondensity)

        for initialfrac in gluonhard4:
            if initialfrac > bins[i] and initialfrac <= bins[i+1]:
                frequencylist2.append(initialfrac)
        binharddensity = len(frequencylist2)*bincenter/(n*binwidth)
        gluonbinhardest4.append(binharddensity)
    
    
    # Calculating solutions
    solution1 = []
    solution2 = []
    solution3 = []
    solution4 = []
    
    for x in xrange:
        D1 = (1/2)*(t1/((np.pi**2) * (np.log(1/x))**3))**(1/4) * np.exp(-t1*np.exp(1) + 2*np.sqrt(t1*(np.log(1/x))))
        D2 = (1/2)*(t2/((np.pi**2) * (np.log(1/x))**3))**(1/4) * np.exp(-t2*np.exp(1) + 2*np.sqrt(t2*(np.log(1/x))))
        D3 = (1/2)*(t3/((np.pi**2) * (np.log(1/x))**3))**(1/4) * np.exp(-t3*np.exp(1) + 2*np.sqrt(t3*(np.log(1/x))))
        D4 = (1/2)*(t4/((np.pi**2) * (np.log(1/x))**3))**(1/4) * np.exp(-t4*np.exp(1) + 2*np.sqrt(t4*(np.log(1/x))))
        #D1L = (1/x)**(2*np.sqrt(t1/(np.log(1/x))))
        #D2L = (1/x)**(2*np.sqrt(t2/(np.log(1/x))))
        #D3L = (1/x)**(2*np.sqrt(t3/(np.log(1/x))))
        #D4L = (1/x)**(2*np.sqrt(t4/(np.log(1/x))))
        
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

    ax1.plot(binlist, gluonbinlist1, 'b--', label ="MC")
    ax1.plot(binlist, gluonbinhardest1, 'b:')
    ax1.plot(xrange, solution1, 'r', label="solution")
    ax1.set_title('t_0 = ' + str(t1))
    ax1.set_xlim(plot_lim,1)
    ax1.set_ylim(0.01,10)
    ax1.set_xlabel('z ')
    ax1.set_ylabel('D(x,t)')
    ax1.grid(linestyle='dashed', linewidth=0.2)
    ax1.legend()
    
    
    print("\rPlotting 2...", end="")

    ax2.plot(binlist, gluonbinlist2, 'b--', label ="MC")
    ax2.plot(binlist, gluonbinhardest2, 'b:')

    ax2.plot(xrange, solution2, 'r', label="solution")
    ax2.set_title('t_0 = ' + str(t2))
    ax2.set_xlim(plot_lim,1)
    ax2.set_ylim(0.01,10)
    ax2.set_xlabel('z')
    ax2.set_ylabel('D(x,t)')
    ax2.grid(linestyle='dashed', linewidth=0.2)
    ax2.legend()

    
    print("\rPlotting 3...", end="")

    ax3.plot(binlist, gluonbinlist3, 'b--', label ="MC")
    ax3.plot(binlist, gluonbinhardest3, 'b:')
    ax3.plot(xrange, solution3, 'r', label="solution")
    ax3.set_title('t_0 = ' + str(t3))
    ax3.set_xlim(plot_lim,1)
    ax3.set_ylim(0.01,10)
    ax3.set_xlabel('z ')
    ax3.set_ylabel('D(x,t)')
    ax3.grid(linestyle='dashed', linewidth=0.2)
    ax3.legend()
    

    print("\rPlotting 4...", end="")

    ax4.plot(binlist, gluonbinlist4, 'b--', label ="MC")
    ax4.plot(binlist, gluonbinhardest4, 'b:')
    ax4.plot(xrange, solution4, 'r', label="solution")
    ax4.set_title('t_0 = ' + str(t4))
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