# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Parton shower program for quarks and gluons in vacuum.
# Using the full splitting functions.

import matplotlib.pyplot as plt
import vacuum_splittingfunctions as sf # Includes color factors.
import numpy as np
from scipy.integrate import quad 
from treelib import Tree 

#constants
epsilon = 10**(-3)
z_min = 10**(-3)
plot_lim = 10**(-3)

bins = 100
minimum_bin = 0.001

# Pre-calculations
gg_integral, __ = quad(sf.gg_full, epsilon, 1-epsilon)
qg_integral, __ = quad(sf.qg_full, 0, 1)
qq_integral, __ = quad(sf.qq_full, 0, 1-epsilon) 

gluon_contribution = (gg_integral+ qg_integral)/(gg_integral
                                                 + qq_integral + qg_integral)
gg_contribution = (gg_integral)/(gg_integral+ qg_integral)

print("Gluon contribution:", round(gluon_contribution,4))
print("ggg contribution:",  round(gg_contribution,4))


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
        InitialType (str): Flavour of the initial parton "quark"/"gluon".
        ShowerNumber (int): Numer n of the shower.
        PartonList (list): All partons that are part of the shower.
        FinalList (list): All partons with no daughters.
        FinalFracList (list): All partons with initialfrac above z_min.
        SplittingPartons (list): All partons available for branching
        ShowerGluons (int): Number of final gluons in the shower.
        ShowrQuarks (int): Number of final quarks in the shower.
        Hardest (float): Highest InitialFrac value in FinalList.
        
    Functions: 
        select_splitting_parton: Selects the most probable parton to split.
        loop_status: Checks when to end the different tau showers.
        
    Other:
        allShowers (list): Contains all shower objects.
    """

    allShowers = []

    def __init__(self, initialtype, showernumber):
        self.allShowers.append(self)

        self.InitialType = initialtype
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
        self.ShowerGluons = None
        self.ShowerQuarks = None
        self.Loops = (True, True, True, True)
        
    def select_splitting_parton(self):
        """
        Determines which parton to split, and which splittingfunction to use.
        
        Parameters: 
            Shower0 (object): Current parton shower.
            t (float): Current vale of the evolution variable.
            t_max (float): Maximum value of the evolution variable.
        
        Returns:
            SplittingParton (object): Which parton is selected for splitting.
            vertex (str): Which splitting vertex to use.
            t (float): Evolution variable after splitting.
        """
        quarkgluon_deltat = []
        for parton in self.SplittingPartons:
            delta_t_sample, vertex = parton.advance_time()
            quarkgluon_deltat.append((parton, delta_t_sample, vertex))

        (SplittingParton, delta_t, vertex) =  min(quarkgluon_deltat, key = lambda t: t[1])
        return SplittingParton, delta_t, vertex
        
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
    """
    def __init__(self, typ, angle, initialfrac, parent, shower):        
        self.Type = typ
        self.Angle = angle
        self.InitialFrac = initialfrac
        self.Parent = parent
        self.Primary = None
        self.Secondary = None
        self.Shower = shower
    
    def split(self, vertex): 
        """Randomly selects splitting vertex, and returns the new parton types,
        and momentumfractions - after splitting. """
        if vertex == "gg":
            splittingvalue = Parton.MH_gg(self)
            parton1_type = "gluon"
            parton2_type = "gluon"
        elif vertex == "qg":
            splittingvalue = Parton.qg(self)
            parton1_type = "quark"
            parton2_type = "quark"
            
        elif vertex == "qq":
            splittingvalue = Parton.MH_qq(self)
            parton1_type = "quark"
            parton2_type = "gluon"
            
        return splittingvalue, parton1_type, parton2_type
    
    def MH_gg(self): 
        """Performs the MH algorithm for the gg splitting vertex. """
        while True: 
            rnd1 = np.random.uniform(0,1)
            xi = ((1-epsilon)/epsilon)**((2*rnd1)-1)
            splittingvalue = xi/(1+xi) 
            acceptance = min(1, 
                       sf.gg_full(splittingvalue)/sf.gg_simple(splittingvalue))
            rnd2 = np.random.uniform(0,1) 
            if acceptance >= rnd2: # Condition for accepting MH value.
                break
        return splittingvalue
                
    def MH_qq(self): 
        """Performs the MH algorithm for the qq splitting vertex. """
        while True: 
            rnd1 = np.random.uniform(0,1)
            xi = ((1-epsilon)/epsilon)**(rnd1)
            splittingvalue = ((epsilon-1)/(xi))+1
            acceptance = min(1, 
                       sf.qq_full(splittingvalue)/sf.qq_simple(splittingvalue))
            rnd2 = np.random.uniform(0,1)
            if acceptance >= rnd2: # Condition for accepting MH value.
                break
        return splittingvalue
        
    def qg(self):
        """Calculates splitting value for the qg vertex.. """
        rnd1 = np.random.uniform(0,1)
        d = (rnd1*(2/3))
        a = (36*(d**(2))-(24*d)+5)**(1/2)
        b = (a+(6*d)-2)**(1/3)
        splittingvalue = (0.5+0.5*b-(0.5/b))
        return splittingvalue
    
    def advance_time(self):
        """Randomly generates a probably value t for this splitting. """
        rnd1 = np.random.uniform(0,1) 
        
        if self.Type == "gluon":
            rnd2 = np.random.uniform(0,1)
            delta_t = -(np.log(rnd1))/(gg_integral+qg_integral)
            if rnd2 < gg_contribution:
                vertex = "gg"
            elif rnd2 >= gg_contribution:
                vertex = "qg"
        
        elif self.Type == "quark":
            delta_t = -(np.log(rnd1))/(qq_integral)
            vertex = "qq"
            
        return delta_t, vertex


# Main shower program. Given the initial conditions, it performs splitting 
# of the initial parton, until the evolution interval t is too large for more
# branchings, or there are no more partons to split. 
def generate_shower(initialtype, tvalues , p_t, Q_0, R, showernumber):
    """Main parton shower program for quarks and gluons in vacuum.
    
        Parameters: 
            initialtype (str): Flavor of initial parton.
            t_max (int): Maximum value of evolution variable.
            p_t (int) - Initital parton momentum.
            Q_0 (int) - Hadronization scale.
            R (int) - Jet radius.
            showernumber (int) - Showernumber.
            
        Returns:
            Shower (object) - Object of the class Shower.
    """
    t = 0
    
    Shower0 = Shower(initialtype, showernumber) 
    Parton0 = Parton(initialtype, t, 1, None, Shower0) # Initial parton
    Shower0.PartonList.append(Parton0)
    Shower0.FinalList.append(Parton0)
    Shower0.SplittingPartons.append(Parton0)
        
    while len(Shower0.SplittingPartons) > 0:
        SplittingParton, delta_t, vertex = Shower0.select_splitting_parton()
        t = t+delta_t
        
        end_shower = Shower0.loop_status(t, tvalues)
        if end_shower:
            break
            
        Shower0.SplittingPartons.remove(SplittingParton)    
        Shower0.FinalList.remove(SplittingParton)
        momfrac, type1, type2 = Parton.split(SplittingParton, vertex)

        for j in range(0,2): #Loop for  generating the branched partons
            if j==0: # Parton 1.
                initialfrac = SplittingParton.InitialFrac * momfrac
                NewParton = Parton(type1, t, initialfrac, 
                                   SplittingParton, Shower0)
                SplittingParton.Primary = NewParton
                
            elif j==1: # Parton 2.
                initialfrac = SplittingParton.InitialFrac * (1-momfrac)
                NewParton = Parton(type2, t, initialfrac, 
                                   SplittingParton, Shower0)
                SplittingParton.Secondary = NewParton
                
            Shower0.PartonList.append(NewParton)
            Shower0.FinalList.append(NewParton)    
            
            if initialfrac > z_min: # Limit on how soft gluons can split.
                Shower0.SplittingPartons.append(NewParton)
            
    return Shower0


# Parton tree program.
def create_parton_tree(showernumber): # Treelib print parton tree.
    """This program can draw a tree of the partons. It is really only practical
    for vacuum showers. """
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
# of them, for both gluons and quarks.
# The results from the showers are then plotted in the style of Dasgupta.
def several_showers_dasgupta(n, opt_title):
    """
    Runs n parton showers, and compares the result with Dasgupta.
    
    Parameters: 
        n (int): Number of showers to simulate.
        opt_title (str): Additional title to add to final plot.
        
    Returns:
        A very nice plot. 
    """
    
    error, error_msg = error_message_several_showers(n, opt_title)
    if error:
        print(error_msg)
        return

    R = 0.4   
    p_0 = 100
    Q_0 = 1 

    t1 = 0.04
    t2 = 0.1
    t3 = 0.2
    t4 = 0.3
    tvalues = (t1, t2, t3, t4)
    
    gluonlist1 = []
    gluonhard1 = []
    gluonlist2 = []
    gluonhard2 = []
    gluonlist3 = []
    gluonhard3 = []
    gluonlist4 = []
    gluonhard4 = []
    quarklist1 = []
    quarkhard1 = []
    quarklist2 = []
    quarkhard2 = []
    quarklist3 = []
    quarkhard3 = []
    quarklist4 = []
    quarkhard4 = []
        
    
    for i in range(1,n):
        print("\rLooping... "+ str(round(100*i/(n),1)) + "%",end="")

        # Gluon showers
        Shower0 = generate_shower("gluon", tvalues, p_0, Q_0, R, i)
        gluonhard1.append(Shower0.Hardest1)
        gluonhard2.append(Shower0.Hardest2)
        gluonhard3.append(Shower0.Hardest3)
        gluonhard4.append(Shower0.Hardest4)
        gluonlist1.extend(Shower0.FinalFracList1)
        gluonlist2.extend(Shower0.FinalFracList2)
        gluonlist3.extend(Shower0.FinalFracList3)
        gluonlist4.extend(Shower0.FinalFracList4)  
        del Shower0
        
        # Quark showers.
        Shower0 = generate_shower("quark", tvalues, p_0, Q_0, R, i)
        quarkhard1.append(Shower0.Hardest1)
        quarkhard2.append(Shower0.Hardest2)
        quarkhard3.append(Shower0.Hardest3)
        quarkhard4.append(Shower0.Hardest4)
        quarklist1.extend(Shower0.FinalFracList1)
        quarklist2.extend(Shower0.FinalFracList2)
        quarklist3.extend(Shower0.FinalFracList3)
        quarklist4.extend(Shower0.FinalFracList4)  
        del Shower0

        
    # Now calculating normalizations for Gluons.
    print("\rCalculating bins 1...", end="")
    linbins1 = np.linspace(0, 0.99, num=bins)
    linbins2 = np.linspace(0.992, 0.999, num=10)
    linbins3 = np.linspace(0.9992, 1, num=10)

    linbins = np.hstack((linbins1, linbins2, linbins3))
    binlist = []
    
    gluon1tz = 0
    gluon2tz = 0
    gluon3tz = 0
    gluon4tz = 0
    
    gluonbinlist1 = []
    gluonbinhardest1 = []
    for i in range(len(linbins)-1):
        frequencylist1 = []
        frequencylist2 = []
        binwidth = linbins[i+1]-linbins[i]
        bincenter = linbins[i+1] - (binwidth/2)
        binlist.append(bincenter)
        
        for initialfrac in gluonlist1:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist1.append(initialfrac)
                if initialfrac ==1:
                    gluon1tz +=1
        density = len(frequencylist1)/(n*binwidth)
        gluonbinlist1.append(density)
        
        for initialfrac in gluonhard1:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(n*binwidth)
        gluonbinhardest1.append(density)
        if i >= len(linbins)-3:
            print("Plot nr 1. Gluons.", i, " Showers: ", n, "binwidth = ", binwidth)
            print("Members of final bin: ", linbins[i], "-", linbins[i+1]," are, ", len(frequencylist1))
    
    
    print("\rCalculating bins 2...", end="")

    gluonbinlist2 = []
    gluonbinhardest2 = []
    for i in range(len(linbins)-1):
        frequencylist1 = []
        frequencylist2 = []
        binwidth = linbins[i+1]-linbins[i]
        bincenter = linbins[i+1] - (binwidth/2)
        
        for initialfrac in gluonlist2:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist1.append(initialfrac)
                if initialfrac ==1:
                    gluon2tz +=1
        density = len(frequencylist1)/(n*binwidth)
        gluonbinlist2.append(density)
        
        for initialfrac in gluonhard2:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(n*binwidth)
        gluonbinhardest2.append(density)
        
        if i >=(len(linbins)-3):
            print("Plot nr 2. Gluons.", i, " Showers: ", n, "binwidth = ", binwidth)
            print("Members of final bin: ", linbins[i], "-", linbins[i+1]," are, ", len(frequencylist1))
        
    print("\rCalculating bins 3...", end="")
        
    gluonbinlist3 = []
    gluonbinhardest3 = []

    for i in range(len(linbins)-1):
        frequencylist1 = []
        frequencylist2 = []
        binwidth = linbins[i+1]-linbins[i]
        bincenter = linbins[i+1] - (binwidth/2)
        
        for initialfrac in gluonlist3:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist1.append(initialfrac)
                if initialfrac ==1:
                    gluon3tz +=1
        density = len(frequencylist1)/(n*binwidth)
        gluonbinlist3.append(density)
        
        for initialfrac in gluonhard3:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(n*binwidth)
        gluonbinhardest3.append(density)
        
        if i >=(len(linbins)-3):
            print("Plot nr 3. Gluons.", i, " Showers: ", n, "binwidth = ", binwidth)
            print("Members of final bin: ", linbins[i], "-", linbins[i+1]," are, ", len(frequencylist1))
        

    print("\rCalculating bins 4...", end="")

    gluonbinlist4 = []
    gluonbinhardest4 = []
    for i in range(len(linbins)-1):
        frequencylist1 = []
        frequencylist2 = []
        binwidth = linbins[i+1]-linbins[i]
        bincenter = linbins[i+1] - (binwidth/2)
        
        for initialfrac in gluonlist4:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist1.append(initialfrac)
                if initialfrac ==1:
                    gluon4tz +=1
        density = len(frequencylist1)/(n*binwidth)
        gluonbinlist4.append(density)
        #print("binrange: ", linbins[i],"-",linbins[i+1])
       # print("freqlist1 (n gluons): ", len(frequencylist1)," density: ", density)
        for initialfrac in gluonhard4:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(n*binwidth)
        gluonbinhardest4.append(density)
        if i >= len(linbins)-3:
            print("Plot nr 4. Gluons.", i, " Showers: ", n, "binwidth = ", binwidth)
            print("Members of final bin: ", linbins[i], "-", linbins[i+1]," are, ", len(frequencylist1))
    
    print("gluont1z = ", gluon1tz)
    print("gluont2z = ", gluon2tz)
    print("gluont3z = ", gluon3tz)
    print("gluont4z = ", gluon4tz)
    # Now calculating normalizations for Quarks.
    print("\rCalculating bins 5...", end="")
    
    quarkbinlist1 = []
    quarkbinhardest1 = []
    for i in range(len(linbins)-1):
        frequencylist1 = []
        frequencylist2 = []
        binwidth = linbins[i+1]-linbins[i]
        bincenter = linbins[i+1] - (binwidth/2)
        
        for initialfrac in quarklist1:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist1.append(initialfrac)
        quarkdensity = len(frequencylist1)/(n*binwidth)
        quarkbinlist1.append(quarkdensity)
        
        for initialfrac in quarkhard1:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(n*binwidth)
        quarkbinhardest1.append(density)
    
    
    print("\rCalculating bins 6...", end="")

    quarkbinlist2 = []
    quarkbinhardest2 = []
    for i in range(len(linbins)-1):
        frequencylist1 = []
        frequencylist2 = []
        binwidth = linbins[i+1]-linbins[i]
        bincenter = linbins[i+1] - (binwidth/2)
        
        for initialfrac in quarklist2:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist1.append(initialfrac)
        quarkdensity = len(frequencylist1)/(n*binwidth)
        quarkbinlist2.append(quarkdensity)
        
        for initialfrac in quarkhard2:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(n*binwidth)
        quarkbinhardest2.append(density)
        
        
    print("\rCalculating bins 7...", end="")
        
    quarkbinlist3 = []
    quarkbinhardest3 = []

    for i in range(len(linbins)-1):
        frequencylist1 = []
        frequencylist2 = []
        binwidth = linbins[i+1]-linbins[i]
        bincenter = linbins[i+1] - (binwidth/2)
        
        for initialfrac in quarklist3:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist1.append(initialfrac)
        quarkdensity = len(frequencylist1)/(n*binwidth)
        quarkbinlist3.append(quarkdensity)
        
        for initialfrac in quarkhard3:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(n*binwidth)
        quarkbinhardest3.append(density)
        
        
    print("\rCalculating bins 8...", end="")

    quarkbinlist4 = []
    quarkbinhardest4 = []
    for i in range(len(linbins)-1):
        frequencylist1 = []
        frequencylist2 = []
        binwidth = linbins[i+1]-linbins[i]
        bincenter = linbins[i+1] - (binwidth/2)
        
        for initialfrac in quarklist4:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist1.append(initialfrac)
        quarkdensity = len(frequencylist1)/(n*binwidth)
        quarkbinlist4.append(quarkdensity)
        
        for initialfrac in quarkhard4:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(n*binwidth)
        quarkbinhardest4.append(density)
    

    # Now starting the plotting.
    plt.figure(dpi=1000, figsize= (6,5)) #(w,h)

    title = ("Vaccum showers: " + str(n) + 
             ". epsilon: " + str(epsilon) + 
             ". gluon contr: " + str(round(gluon_contribution,3)) + 
             "\n " + opt_title)

    #plt.suptitle(title)

    plt.rc('axes', titlesize="small" , labelsize="x-small")
    plt.rc('xtick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('ytick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('legend',fontsize='xx-small')    # fontsize of the legend labels.
    plt.rc('lines', linewidth=0.8)

    ax1 = plt.subplot(221) #H B NR
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    print("\rPlotting 1...", end="")

    ax1.plot(binlist, gluonbinlist1, 'g-', label ="gluon")
    ax1.plot(binlist, gluonbinhardest1, 'g--')
    ax1.plot(binlist, quarkbinlist1, 'b-', label ="quark")
    ax1.plot(binlist, quarkbinhardest1, 'b--')
    ax1.set_yscale("log")

    ax1.set_title('t = ' + str(t1))
    ax1.set_xlim(0,1)
    ax1.set_ylim(0.01,10)
    ax1.set_xlabel('z ')
    ax1.set_ylabel('f(x,t)')
    ax1.grid(linestyle='dashed', linewidth=0.2)
    ax1.legend()
    
    
    print("\rPlotting 2...", end="")

    ax2.plot(binlist, gluonbinlist2, 'g-', label ="gluon")
    ax2.plot(binlist, gluonbinhardest2, 'g--')
    ax2.plot(binlist, quarkbinlist2, 'b-', label ="quark")
    ax2.plot(binlist, quarkbinhardest2, 'b--')
    ax2.set_yscale("log")

    ax2.set_title('t = ' + str(t2))
    ax2.set_xlim(0,1)
    ax2.set_ylim(0.01,10)
    ax2.set_xlabel('z')
    ax2.set_ylabel('f(x,t)')
    ax2.grid(linestyle='dashed', linewidth=0.2)
    ax2.legend()

    
    print("\rPlotting 3...", end="")

    ax3.plot(binlist, gluonbinlist3, 'g-', label ="gluon")
    ax3.plot(binlist, gluonbinhardest3, 'g--')
    ax3.plot(binlist, quarkbinlist3, 'b-', label ="quark")
    ax3.plot(binlist, quarkbinhardest3, 'b--')
    ax3.set_yscale("log")

    ax3.set_title('t = ' + str(t3))
    ax3.set_xlim(0,1)
    ax3.set_ylim(0.01,10)
    ax3.set_xlabel('z ')
    ax3.set_ylabel('f(x,t)')
    ax3.grid(linestyle='dashed', linewidth=0.2)
    ax3.legend()
    

    print("\rPlotting 4...", end="")

    ax4.plot(binlist, gluonbinlist4, 'g-', label ="gluon")
    ax4.plot(binlist, gluonbinhardest4, 'g--')
    ax4.plot(binlist, quarkbinlist4, 'b-', label ="quark")
    ax4.plot(binlist, quarkbinhardest4, 'b--')
    ax4.set_yscale("log")

    ax4.set_title('t = ' + str(t4))
    ax4.set_xlim(0,1)
    ax4.set_ylim(0.01,10)
    ax4.set_xlabel('z ')
    ax4.set_ylabel('f(x,t)')
    ax4.grid(linestyle='dashed', linewidth=0.2)
    ax4.legend()


    print("\rShowing", end="")

    plt.tight_layout()
    plt.show()
    print("\rDone!")    

def error_message_several_showers(n, opt_title):
    """"Checks the input parameters for erros and generates merror_msg."""
    error = False
    msg = ""
    n_error = not isinstance(n, int)
    title_error = not  isinstance(opt_title, str)

    if n_error or title_error:
        error = True
        if n_error:
            msg = msg + "\nERROR! - 'n' must be an integer."
        if title_error:
            msg = msg + "\nERROR! - 'opt_title' must be a str."
    return error, msg