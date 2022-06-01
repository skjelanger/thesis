# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Parton shower program for gluons in medium.
# Using the simplified ggg splitting kernel.

import matplotlib.pyplot as plt
from medium_splittingfunctions import gg_simple_analytical
import numpy as np
from scipy.integrate import quad
import datetime
from operator import attrgetter


# Constants
epsilon = 10**(-3)
z_min = 10**(-3)
plot_lim = 10**(-3)
binnumber = 50

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
        self.FinalFracList5 = [] # Contains all partons above z_min.
        self.FinalFracList6 = [] # Contains all partons above z_min.
        self.FinalFracList7 = [] # Contains all partons above z_min.
        self.FinalFracList8 = [] # Contains all partons above z_min.
        self.Hardest1 = 0
        self.Hardest2 = 0
        self.Hardest3 = 0
        self.Hardest4 = 0
        self.Hardest5 = 0
        self.Hardest6 = 0
        self.Hardest7 = 0
        self.Hardest8 = 0
        self.Leadingbranch1 = None
        self.Leadingbranch2 = None
        self.Leadingbranch3 = None
        self.Leadingbranch4 = None
        self.Leadingbranch5 = None
        self.Leadingbranch6 = None
        self.Leadingbranch7 = None
        self.Leadingbranch8 = None
        self.SplittingPartons = []
        self.Loops = (True, True, True, True, True, True, True, True)
        
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
            self.Loops = (False, True, True, True, True, True, True, True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList1.append(PartonObj.InitialFrac)
            hardest = max(self.FinalList, key=attrgetter('InitialFrac'))
            self.Leadingbranch1 = hardest.find_branch()
            self.Hardest1 = hardest.InitialFrac
                    
        if tau > tauvalues[1] and self.Loops[1]:
            self.Loops = (False, False, True, True, True, True, True, True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList2.append(PartonObj.InitialFrac) 
            hardest = max(self.FinalList, key=attrgetter('InitialFrac'))
            self.Leadingbranch2 = hardest.find_branch()
            self.Hardest2 = hardest.InitialFrac
                    
        if tau > tauvalues[2] and self.Loops[2]:
            self.Loops = (False, False, False, True, True, True, True, True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList3.append(PartonObj.InitialFrac) 
            hardest = max(self.FinalList, key=attrgetter('InitialFrac'))
            self.Leadingbranch3 = hardest.find_branch()
            self.Hardest3 = hardest.InitialFrac
                    
        if tau > tauvalues[3] and self.Loops[3]:
            self.Loops = (False, False, False, False, True, True, True, True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList4.append(PartonObj.InitialFrac) 
            hardest = max(self.FinalList, key=attrgetter('InitialFrac'))
            self.Leadingbranch4 = hardest.find_branch()
            self.Hardest4 = hardest.InitialFrac
                    
        if tau > tauvalues[4] and self.Loops[4]:
            self.Loops = (False, False, False, False, False, True, True, True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList5.append(PartonObj.InitialFrac) 
            hardest = max(self.FinalList, key=attrgetter('InitialFrac'))
            self.Leadingbranch5 = hardest.find_branch()
            self.Hardest5 = hardest.InitialFrac
                    
        if tau > tauvalues[5] and self.Loops[5]:
            self.Loops = (False, False, False, False, False, False, True,True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList6.append(PartonObj.InitialFrac) 
            hardest = max(self.FinalList, key=attrgetter('InitialFrac'))
            self.Leadingbranch6 = hardest.find_branch()
            self.Hardest6 = hardest.InitialFrac
                    
        if tau > tauvalues[6] and self.Loops[6]:
            self.Loops = (False, False, False, False, False, False, False,True)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList7.append(PartonObj.InitialFrac) 
            hardest = max(self.FinalList, key=attrgetter('InitialFrac'))
            self.Leadingbranch7 = hardest.find_branch()
            self.Hardest7 = hardest.InitialFrac      
              
        if tau > tauvalues[7] and self.Loops[7]:
            self.Loops = (False, False, False, False, False, False,False,False)
            for PartonObj in self.FinalList:
                if PartonObj.InitialFrac > plot_lim:
                    self.FinalFracList8.append(PartonObj.InitialFrac)   
                del PartonObj
            hardest = max(self.FinalList, key=attrgetter('InitialFrac'))
            self.Leadingbranch8 = hardest.find_branch()
            self.Hardest8 = hardest.InitialFrac
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

    def find_branch(self):
        """Determines if a parton has been on every leading branch. """
        parton = self
        while True:
            parentparton = parton.Parent

            if parentparton == None:
                return True

            elif parentparton.Primary == parton:
                if parentparton.Secondary.InitialFrac <= parton.InitialFrac:
                    parton = parentparton
                elif parentparton.Secondary.InitialFrac > parton.InitialFrac:
                    return False
                
            elif parentparton.Secondary == parton:

                if parentparton.Primary.InitialFrac <= parton.InitialFrac:
                    parton = parentparton
                elif parentparton.Primary.InitialFrac > parton.InitialFrac:
                    return False
            
            else:
                print("ERROR - parton is not a daughterparton.")

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

    return Shower0


# Program for comparing the shower program with analytical calculations.
# Four different values of tau are used, and n showers are generated for each
# of them. The results from the showers then compared to analytical results, 
# and plotted in the same figure.
def several_showers_scaling_comparison(n, opt_title):
    """
    Runs n parton showers, and compares the analytical results.
    
    Parameters: 
        n (int): Number of showers to simulate.
        opt_title (str): Additional title to add to final plot.
        scale (str): Set lin or log scale for the plot.
        
    Returns:
        A very nice plot. 
    """
    
    error, error_msg = error_message_several_showers(n, opt_title)
    if error:
        print(error_msg)
        return
    
    R = 0.4 # Jet radius.    
    p_0 = 100 # Initial parton momentum.
    Q_0 = 1 # Hadronization scale.
    
    tau1 = 0.1
    tau2 = 0.2
    tau3 = 0.4
    tau4 = 0.6
    tau5 = 0.8
    tau6 = 1.0
    tau7 = 1.1
    tau8 = 1.2
    tauvalues = (tau1, tau2, tau3, tau4, tau5, tau6, tau7, tau8)
    
    #Generating showers
    gluonlists = [[],[],[],[],[],[],[],[]]
    gluonhards = [[],[],[],[],[],[],[],[]]
    hardestbranches = [0,0,0,0,0,0,0,0]
    branchhards = [[],[],[],[],[],[],[],[]]
    nonbranchhards = [[],[],[],[],[],[],[],[]]


    for i in range (1,n):
        print("\rLooping... " + str(round(100*i/(n),2)) +"%", end="")

        Shower0 = generate_shower(tauvalues, p_0, Q_0, R, i)
        gluonlists[0].extend(Shower0.FinalFracList1)
        gluonlists[1].extend(Shower0.FinalFracList2)
        gluonlists[2].extend(Shower0.FinalFracList3)
        gluonlists[3].extend(Shower0.FinalFracList4)     
        gluonlists[4].extend(Shower0.FinalFracList5)
        gluonlists[5].extend(Shower0.FinalFracList6)
        gluonlists[6].extend(Shower0.FinalFracList7)
        gluonlists[7].extend(Shower0.FinalFracList8)  
        gluonhards[0].append(Shower0.Hardest1)
        gluonhards[1].append(Shower0.Hardest2)
        gluonhards[2].append(Shower0.Hardest3)
        gluonhards[3].append(Shower0.Hardest4)
        gluonhards[4].append(Shower0.Hardest5)
        gluonhards[5].append(Shower0.Hardest6)
        gluonhards[6].append(Shower0.Hardest7)
        gluonhards[7].append(Shower0.Hardest8)
        
        if Shower0.Leadingbranch1 == True:
            hardestbranches[0] += 1
            branchhards[0].append(Shower0.Hardest1)
        elif Shower0.Leadingbranch1 == False:
            nonbranchhards[0].append(Shower0.Hardest1)
            
        if Shower0.Leadingbranch2 == True:
            hardestbranches[1] += 1
            branchhards[1].append(Shower0.Hardest2)
        elif Shower0.Leadingbranch2 == False:
            nonbranchhards[1].append(Shower0.Hardest2)
            
        if Shower0.Leadingbranch3 == True:
            hardestbranches[2] += 1
            branchhards[2].append(Shower0.Hardest3)
        elif Shower0.Leadingbranch3 == False:
            nonbranchhards[2].append(Shower0.Hardest3)
            
        if Shower0.Leadingbranch4 == True:
            hardestbranches[3] += 1
            branchhards[3].append(Shower0.Hardest4)
        elif Shower0.Leadingbranch4 == False:
            nonbranchhards[3].append(Shower0.Hardest4)
            
        if Shower0.Leadingbranch5 == True:
            hardestbranches[4] += 1
            branchhards[4].append(Shower0.Hardest5)
        elif Shower0.Leadingbranch5 == False:
            nonbranchhards[4].append(Shower0.Hardest5)
            
        if Shower0.Leadingbranch6 == True:
            hardestbranches[5] += 1
            branchhards[5].append(Shower0.Hardest6)
        elif Shower0.Leadingbranch6 == False:
            nonbranchhards[5].append(Shower0.Hardest6)
            
        if Shower0.Leadingbranch7 == True:
            hardestbranches[6] += 1
            branchhards[6].append(Shower0.Hardest7)
        elif Shower0.Leadingbranch7 == False:
            nonbranchhards[6].append(Shower0.Hardest7)
            
        if Shower0.Leadingbranch8 == True:
            hardestbranches[7] += 1
            branchhards[7].append(Shower0.Hardest8)
        elif Shower0.Leadingbranch8 == False:
            nonbranchhards[7].append(Shower0.Hardest8)
            
        del Shower0

    # Sets the different ranges required for the plots.
    logbins1 = np.logspace(-3, -0.1, num=binnumber)
    logbins2 = np.logspace(-0.09, 0, num = 10)
    logbins = np.hstack((logbins1, logbins2))


    # Normalizing the showers.
    print("\rCalculating bins...", end="")
    logbinlist = []
    
    gluonloglists = [[],[],[],[],[],[],[],[]]
    gluonloghards = [[],[],[],[],[],[],[],[]]
    branchloghards = [[],[],[],[],[],[],[],[]]
    nonbranchloghards = [[],[],[],[],[],[],[],[]]


    for i in range(len(logbins)-1):
        binwidth = logbins[i+1]-logbins[i]
        bincenter = logbins[i+1] - (binwidth/2)
        logbinlist.append(bincenter)
                
        for gluonlist in gluonlists:
            frequencylist = []
            index = gluonlists.index(gluonlist)
            for initialfrac in gluonlist:
                if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                    frequencylist.append(initialfrac)       
            binvalue = len(frequencylist)*bincenter/(n*binwidth)
            gluonloglists[index].append(binvalue)

        for gluonhard in gluonhards:
            frequencylist = []
            index = gluonhards.index(gluonhard)
            for initialfrac in gluonhard:
                if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                    frequencylist.append(initialfrac)       
            binvalue = len(frequencylist)*bincenter/(n*binwidth)
            gluonloghards[index].append(binvalue)
            
        for branchhard in branchhards:
            index = branchhards.index(branchhard)
            frequencylist = []
            for initialfrac in branchhard:
                if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                    frequencylist.append(initialfrac)                  
            binvalue = len(frequencylist)*bincenter/(n*binwidth)
            branchloghards[index].append(binvalue)
            
        for nonbranchhard in nonbranchhards:
            index = nonbranchhards.index(nonbranchhard)
            frequencylist = []
            for initialfrac in nonbranchhard:
                if initialfrac > logbins[i] and initialfrac <= logbins[i+1]:
                    frequencylist.append(initialfrac)                  
            binvalue = len(frequencylist)*bincenter/(n*binwidth)
            nonbranchloghards[index].append(binvalue)            
            

    # Save data to file
    fulldate = datetime.datetime.now()
    datetext = (fulldate.strftime("%y") +"_" + 
                fulldate.strftime("%m") +"_" + 
                fulldate.strftime("%d") +"_" )
    filename = "data" + datetext  + str(n) + "showers"
    filenameloc = "data\\parton_shower_gluons_medium_scaling_data\\" + filename
    
    np.savez(filenameloc, 
             n = n,
             tauvalues = tauvalues,
             logbinlist = logbinlist, 
             gluonloglists = gluonloglists,
             gluonloghards = gluonloghards,
             hardestbranches = hardestbranches,
             branchloghards = branchloghards,
             nonbranchloghards = nonbranchloghards)

    # Do the actual plotting. 
    plt.figure(dpi=300, figsize= (6,5)) #(w,h) figsize= (10,3)
    title = ("Medium showers: " + str(n) + 
             ". epsilon: " + str(epsilon) + 
             ". z_min: " + str(z_min) +
             "\n" + opt_title)    
    #plt.suptitle(title)

    plt.rc('axes', titlesize="small" , labelsize="x-small")
    plt.rc('xtick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('ytick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('legend',fontsize='xx-small')    # fontsize of the legend labels.
    plt.rc('lines', linewidth=0.8)

    ax = plt.subplot(111) #H B NR
    
    print("\rPlotting...", end="")

    ax.plot(logbinlist, gluonloglists[0], "--", label="tau="+str(tauvalues[0]))
    ax.plot(logbinlist, gluonloglists[1], "--", label="tau="+str(tauvalues[1]))
    ax.plot(logbinlist, gluonloglists[2], "--", label="tau="+str(tauvalues[2]))
    ax.plot(logbinlist, gluonloglists[3], "--", label="tau="+str(tauvalues[3]))
    ax.plot(logbinlist, gluonloglists[4], "--", label="tau="+str(tauvalues[4]))
    ax.plot(logbinlist, gluonloglists[5], "--", label="tau="+str(tauvalues[5]))
    ax.plot(logbinlist, gluonloglists[6], "--", label="tau="+str(tauvalues[6]))
    ax.plot(logbinlist, gluonloglists[7], "--", label="tau="+str(tauvalues[7]))
    
    ax.set_xlim(0.001,1)
    ax.set_ylim(0.01,10)
    ax.set_xlabel('z ')
    ax.set_ylabel('$D(x,\\tau)$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(linestyle='dashed', linewidth=0.2)
    ax.legend()
    


    print("\rShowing", end="\n")

    plt.tight_layout()
    plt.show()
    print("\rDone!")    


def error_message_several_showers(n, opt_title):
    """"Checks the input parameters for erros and generates merror_msg."""
    error = False
    msg = ""
    n_error = not isinstance(n, int)
    title_error = not  isinstance(opt_title, str)

    if n_error or title_error :
        error = True
        if n_error:
            msg = msg + "\nERROR! - 'n' must be an integer."
        if title_error:
            msg = msg + "\nERROR! - 'opt_title' must be a str."

    return error, msg