# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Parton shower program for gluons in vacuum.
# Using the simplified ggg splitting kernel.

import matplotlib.pyplot as plt
import vacuum_splittingfunctions as sf # Includes color factors.
import numpy as np
from scipy.integrate import quad 
import datetime

#constants
epsilon = 10**(-3)
z_min = 10**(-3)
plot_lim = 10**(-3)
binnumber = 125

gg_integral, __ = quad((sf.gg_simple_analytical), epsilon, 1-epsilon)

print("epsilon: ", epsilon)
print("gluons only gg_integral: ", gg_integral)

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

    def __init__(self, showernumber):

        self.ShowerNumber = showernumber
        self.PartonList = [] # List for all Shower Partons. Used for treelib.
        self.FinalList = [] # Contains all Final Partons
        self.FinalFracList1 = [] # Contains all partons above z_min.
        self.FinalFracList2 = [] # Contains all partons above z_min.
        self.FinalFracList3 = [] # Contains all partons above z_min.
        self.FinalFracList4 = [] # Contains all partons above z_min.
        self.Hardest1 = 0
        self.Hardest2 = 0
        self.Hardest3 = 0
        self.Hardest4 = 0
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
def generate_shower(tvalues, showernumber):
    """
    Main parton shower program for gluons in vacuum.
    
        Parameters: 
            tvalues (list): Maximum values of the evolution variable.
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
            if NewParton.InitialFrac > 0.001:
                Shower0.SplittingPartons.append(NewParton)
    
    return Shower0


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
    Q_0 = 1 # Hadronization scale.
    alpha = 0.1184
    
    t1 = 0.04
    t2 = 0.1
    t3 = 0.2
    t4 = 0.3
    tvalues = [t1, t2, t3, t4]
    pvalues = []
    
    for t in tvalues:
        pvalue = Q_0/R * np.exp(t*np.pi/alpha)
        pvalues.append(pvalue)
        
    #Generating showers
    gluonlists = [[],[],[],[]]
    gluonhards = [[],[],[],[]]
        
    for i in range(1,n):
        print("\rLooping... "+ str(round(100*i/(n),1)) + "%",end="")
        Shower0 = generate_shower(tvalues, i)
        gluonhards[0].append(Shower0.Hardest1)
        gluonhards[1].append(Shower0.Hardest2)
        gluonhards[2].append(Shower0.Hardest3)
        gluonhards[3].append(Shower0.Hardest4)
        gluonlists[0].extend(Shower0.FinalFracList1)
        gluonlists[1].extend(Shower0.FinalFracList2)
        gluonlists[2].extend(Shower0.FinalFracList3)
        gluonlists[3].extend(Shower0.FinalFracList4)  
        del Shower0
    
    # Sets the different ranges required for the plots.
    linbins1 = (np.linspace(plot_lim, 0.99, num=binnumber))
    linbins2 = (np.linspace(0.991, 1, num= 10))
    linbins = np.hstack((linbins1, linbins2))
    #bins = np.linspace(plot_lim, 1, num=binnumber)
    xlinrange = np.linspace(plot_lim, 0.9999, num=(4*binnumber))

    logbins1 = np.logspace(-3, -0.1, num=binnumber)
    logbins2 = np.logspace(-0.09, 0, num = 10)
    logbins = np.hstack((logbins1, logbins2))
    xlogrange = np.logspace(-3, -0.0001, num=(4*binnumber))
            
        
    # Normalizing showers
    print("\rCalculating bins...", end="")

    linbinlist = []
    gluonlinlists = [[],[],[],[]]
    gluonlinhards = [[],[],[],[]]

    for i in range(len(linbins)-1):
        binwidth = linbins[i+1]-linbins[i]
        bincenter = linbins[i+1] - (binwidth/2)
        linbinlist.append(bincenter)
        
        # Calculating gluonlinbins
        for gluonlist in gluonlists:
            index = gluonlists.index(gluonlist)
            frequencylist = []
            
            for initialfrac in gluonlist:
                if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                    frequencylist.append(initialfrac)
            gluondensity = len(frequencylist)*bincenter/(n*binwidth)
            gluonlinlists[index].append(gluondensity)
        
        # Calculating hardlinbins.
        for gluonhard in gluonhards:
            index = gluonhards.index(gluonhard)
            frequencylist = []

            for initialfrac in gluonhard:
                if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                    frequencylist.append(initialfrac)
            binharddensity = len(frequencylist)*bincenter/(n*binwidth)
            gluonlinhards[index].append(binharddensity)
    
    logbinlist = []
    gluonloglists = [[],[],[],[]]
    gluonloghards = [[],[],[],[]]

    for i in range(len(logbins)-1):
        binwidth = logbins[i+1]-logbins[i]
        bincenter = logbins[i+1] - (binwidth/2)
        logbinlist.append(bincenter)
        
        # Calculating gluonlogbins
        for gluonlist in gluonlists:
            index = gluonlists.index(gluonlist)
            frequencylist = []
            
            for initialfrac in gluonlist:
                if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                    frequencylist.append(initialfrac)
            gluondensity = len(frequencylist)*bincenter/(n*binwidth)
            gluonloglists[index].append(gluondensity)
        
        # Calculating hardlogbins.
        for gluonhard in gluonhards:
            index = gluonhards.index(gluonhard)
            frequencylist = []

            for initialfrac in gluonhard:
                if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                    frequencylist.append(initialfrac)
            binharddensity = len(frequencylist)*bincenter/(n*binwidth)
            gluonloghards[index].append(binharddensity)
    
    # Calculate solutions
    linsolutions = DGLAP_solutions(tvalues, xlinrange)
    logsolutions = DGLAP_solutions(tvalues, xlogrange)

    
    # Save data to file
    fulldate = datetime.datetime.now()
    datetext = (fulldate.strftime("%y") +"_" + 
                fulldate.strftime("%m") +"_" + 
                fulldate.strftime("%d") +"_" )
    filename = "data" + datetext + str(n) + "showers"
    filenameloc = "data\\parton_shower_gluons_vacuum_data\\" + filename
    np.savez(filenameloc, 
             n = n,
             tvalues = tvalues,
             linbinlist = linbinlist,
             logbinlist = logbinlist, 
             gluonlinlists = gluonlinlists,
             gluonlinhards = gluonlinhards,
             gluonloglists = gluonloglists,
             gluonloghards = gluonloghards,
             xlinrange = xlinrange,
             xlogrange = xlogrange,
             linsolutions = linsolutions,
             logsolutions = logsolutions)
    
     
    # Plot    
    plt.figure(dpi=500, figsize= (6,5)) #(w,h) figsize= (10,3)
    title = ("Vaccum showers: " + str(n) + 
             ". epsilon: " + str(epsilon) + ". zmin: " + str(z_min) + 
             "\n " + opt_title)    
    #plt.suptitle(title)

    plt.rc('axes', titlesize="small" , labelsize="x-small")
    plt.rc('xtick', labelsize="x-small")
    plt.rc('ytick', labelsize="x-small")
    plt.rc('legend',fontsize='xx-small')
    plt.rc('lines', linewidth=0.8)

    ax1 = plt.subplot(221) #H B NR
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    axes = [ax1, ax2, ax3, ax4]
    
    print("\rPlotting..." + 10*" ", end="")
    
    for ax in axes:
        index = axes.index(ax)
        
        if scale == "lin":
            ax.plot(linbinlist, gluonlinlists[index], 'b--', label ="MC")
            #ax.plot(linbinlist, gluonlinhards[index], 'b:')
            ax.plot(xlinrange, linsolutions[index], 'r', label="solution")
            ax.set_xscale("linear")
            ax.set_yscale("log")
            ax.set_xlim(0,1)
            
        elif scale == "log":
            ax.plot(logbinlist, gluonloglists[index], 'b--', label ="MC")
            #ax.plot(logbinlist, gluonloghards[index], 'b:')
            ax.plot(xlogrange, logsolutions[index], 'r', label="solution")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.001,1)

        ax.set_ylim(0.01,10)
        ax.set_title('t = ' + str(tvalues[index]))
        ax.set_xlabel('$z$')
        ax.set_ylabel('$D(x,t)$')
        ax.grid(linestyle='dashed', linewidth=0.2)
        ax.legend(loc='lower right')
                
        textstring = '$n={%i}$'%n
        ax.text(0.85, 0.2, textstring, fontsize = "xx-small",
                horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        

    print("\rShowing..." + 10*" ", end="")

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


    if n_error or title_error or scale_error:
        error = True
        if n_error:
            msg = msg + "\nERROR! - 'n' must be an integer."
        if title_error:
            msg = msg + "\nERROR! - 'opt_title' must be a str."
        if scale_error:
            msg = msg+ "\nERROR! - 'scale' must be 'lin' or 'log'."
    return error, msg

def DGLAP_solutions(tvalues, xrange):
    """"
    Calculated the solutions of the DGLAP equation.
    
    Parameters: 
        tvalues (list): Vales of t to calculate for..
        xrange (list): xvalues for plot

    Returns:
        solutions (list): List of the caculated solutions for each x in xrange.
    """

    solutions = [[],[],[],[]]
    gamma = 0.57721566490153286060651209008240243104215933593992
    sqp = np.sqrt(np.pi)
    C = 1
    
    for t in tvalues:
        index = tvalues.index(t)
    
        for x in xrange:
            logx = np.log(1/x)
            logx3 = (np.log(1/x))**3
            D = np.exp(2*np.sqrt(2*C*t*logx)-2*C*gamma*t)* (1/(2*sqp))* ((2*C*t)/(logx3))**(1/4)
            solutions[index].append(D)
    return solutions