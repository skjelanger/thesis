# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Parton shower program for quarks and gluons in vacuum.
# Using the full splitting functions.

import matplotlib.pyplot as plt
import vacuum_splittingfunctions as sf

import numpy as np
import random
from scipy.integrate import quad 
from treelib import Tree 

#constants
epsilon = 10**(-3)
#alpha_S = 0.12


bins = 50
minimum_bin = 0.001

#C_A = 3
#C_F = 4/3
#T_F = 1/2
#N_F = 5

# Pre-calculations
gg_integral, error1 = quad(sf.gg_full, epsilon, 1-epsilon)
qq_integral, error2 = quad(sf.qq_full, epsilon, 1-epsilon) 
qg_integral, error3 = quad(sf.qg_full, epsilon, 1-epsilon)
gluon_contribution = (gg_integral+ qg_integral)/(gg_integral
                                                 + qq_integral + qg_integral)
gg_contribution = (gg_integral)/(gg_integral+ qg_integral)

print("epsilon: ", epsilon)
print("Gluon contribution:", gluon_contribution)
print("ggg contribution:",  gg_contribution)


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
    """Class used for storing information about individual parton showers."""
    allShowers = []

    def __init__(self, initialtype, showernumber):
        self.allShowers.append(self)

        self.InitialType = initialtype
        self.ShowerNumber = showernumber
        self.PartonList = [] # List for all Shower Partons. Used for treelib.
        self.FinalList = [] # Contains all Final Partons
        self.FinalFracList = [] # Contains all partons above min_value.
        self.SplittingQuarks = []
        self.SplittingGluons = []
        self.ShowerGluons = None
        self.ShowerQuarks = None
        self.Hardest = None

class Parton(object):
    """Class used for storing information about individual partons."""
    def __init__(self, typ, angle, initialfrac, parent, shower):        
        self.Type = typ
        self.Angle = angle
        self.InitialFrac = initialfrac
        self.Parent = parent
        self.Primary = None
        self.Secondary = None
        self.Shower = shower
    
    def split(self): 
        """Randomly selects splitting vertex, and returns the new parton types,
        and momentumfractions - after splitting. """
        if self.Type == "gluon":
            rnd4 = np.random.uniform(0, 1)
            if rnd4 < gg_contribution:
                splittingvalue = Parton.MH_gg(self)
                parton1_type = "gluon"
                parton2_type = "gluon"
            elif rnd4 >= gg_contribution:
                splittingvalue = Parton.qg(self)
                parton1_type = "quark"
                parton2_type = "quark"
            
        elif self.Type == "quark":
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
                
            if acceptance >= rnd2: #Condition for accepting state.
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
                
            if acceptance >= rnd2: 
                break
                    
        return splittingvalue
        
        
    def qg(self):
        """Calculates splitting value for the qg vertex.. """
        rnd1 = np.random.uniform(0,1)
        d = (rnd1*0.665)+0.001
        a = (36*(d**(2))-(24*d)+5)**(1/2)
        b = (a+(6*d)-2)**(1/3)
        splittingvalue = (0.5+0.5*b-(0.5/b))
        return splittingvalue


# Othr shower related programs, which are unrelated to the class objets, are 
# written her. This is the advance_time program, and select_splitting program.
def advance_time():
    """Randomly generates a probably value t for this splitting. """
    rnd1 = np.random.uniform(0,1) 
    delta_t_quark = -(np.log(rnd1))/(qq_integral)
    delta_t_gluon = -(np.log(rnd1))/(gg_integral + qg_integral)
    return delta_t_quark, delta_t_gluon

# Program for selecting which splitting to perform. 
def select_splitting(Shower0, t, delta_t_quark, delta_t_gluon):
    """Determines which parton can split, based on the available interval t, 
    splittinggluons, and splittingquarks. Returns the selected parton for 
    splitting, and the evolved interval of t."""
    
    both_available = (len(Shower0.SplittingGluons)>0 and 
                        len(Shower0.SplittingQuarks)>0)
    gluons_available = (len(Shower0.SplittingGluons)>0 and 
                        len(Shower0.SplittingQuarks)==0)
    quarks_available = (len(Shower0.SplittingQuarks)>0 and 
                        len(Shower0.SplittingGluons)==0)
    
    if both_available:
        rnd3 = np.random.uniform(0, 1)

        if (gluon_contribution <= rnd3 ):
            SplittingParton = random.choice(Shower0.SplittingQuarks)
            Shower0.SplittingQuarks.remove(SplittingParton)
            t = t + delta_t_quark
                
        elif (gluon_contribution > rnd3 ):
            SplittingParton = random.choice(Shower0.SplittingGluons)
            Shower0.SplittingGluons.remove(SplittingParton)
            t = t + delta_t_gluon
                
    elif gluons_available: 
        SplittingParton = random.choice(Shower0.SplittingGluons)
        Shower0.SplittingGluons.remove(SplittingParton)
        t = t + delta_t_gluon
        
    elif quarks_available: 
        SplittingParton = random.choice(Shower0.SplittingQuarks)
        Shower0.SplittingQuarks.remove(SplittingParton)
        t = t + delta_t_quark
        
    else:
        print("\rERROR: no select_splitting criteria fulfilled.")
        return None, t
    
    return SplittingParton, t


def generate_shower(initialtype, t_max , p_t, Q_0, z_0, R, showernumber, n):
    print("\rLooping... "+ str(round(100*showernumber/(n))) + "%",end="")

    t = 0

    Shower0 = Shower(initialtype, showernumber) 
    Parton0 = Parton(initialtype, t, z_0, None, Shower0) # Initial parton
    Shower0.PartonList.append(Parton0)
    Shower0.FinalList.append(Parton0)

    if initialtype == "quark":
        Shower0.SplittingQuarks.append(Parton0)
    elif initialtype == "gluon":
        Shower0.SplittingGluons.append(Parton0)
        
    while True:
        delta_t_quark, delta_t_gluon = advance_time()
        
        t_int_both = (t+delta_t_quark < t_max and t+delta_t_gluon < t_max)
        t_int_gluons = (t+delta_t_quark >= t_max and t+delta_t_gluon < t_max)
            
        if t_int_both: 
            SplittingParton, t = select_splitting(Shower0, t, 
                                                  delta_t_quark, delta_t_gluon)
            if SplittingParton == None:
                break
            
        elif t_int_gluons:
            if len(Shower0.SplittingGluons) > 0:
                SplittingParton = random.choice(Shower0.SplittingGluons)
                Shower0.SplittingGluons.remove(SplittingParton)
                t = t + delta_t_gluon
            else: #can not split any gluons.
                break
            
        else: #can not split any t
            if delta_t_quark < delta_t_gluon:
                print("ERROR: delta_t_quark < delta_t_gluon." +
                      ". Showernumber is: ", showernumber)
            break
            
        Shower0.FinalList.remove(SplittingParton)
        momfrac, parton1_type, parton2_type = Parton.split(SplittingParton)

        for j in range(0,2): #Loop for  generating the branched partons
            if j==0: # Parton 1.
                initialfrac = SplittingParton.InitialFrac * momfrac
                NewParton = Parton(parton1_type, t, initialfrac, 
                                   SplittingParton, Shower0)
                SplittingParton.Primary = NewParton
                
            elif j==1: # Parton 2.
                initialfrac = SplittingParton.InitialFrac * (1-momfrac)
                NewParton = Parton(parton2_type, t, initialfrac, 
                                   SplittingParton, Shower0)
                SplittingParton.Secondary = NewParton
                
            Shower0.PartonList.append(NewParton)
            Shower0.FinalList.append(NewParton)    
            
            if initialfrac > 0.0001:
                if NewParton.Type =="gluon":
                    Shower0.SplittingGluons.append(NewParton)
                elif NewParton.Type =="quark":
                    Shower0.SplittingQuarks.append(NewParton)

    
    showergluons = 0
    showerquarks = 0

    for PartonObj in Shower0.FinalList:
        if PartonObj.InitialFrac > minimum_bin:
            Shower0.FinalFracList.append(PartonObj.InitialFrac)
        if PartonObj.Type =="gluon": 
            showergluons += 1
        elif PartonObj.Type == "quark":
            showerquarks += 1
            
    Shower0.ShowerGluons = showergluons
    Shower0.ShowerQuarks = showerquarks
    Shower0.Hardest = max(Shower0.FinalFracList)

    return Shower0


# Parton tree program.
def create_parton_tree(showernumber): # Treelib print parton tree.
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
# of them, for both gluons and quarks.
# The results from the showers are then plotted in the style of Dasgupta.
def several_showers_dasgupta(n):
    """Runs n parton showers, and compares the result with Dasgupta."""

    R = 0.4 # Define jet radius.    
    p_0 = 100
    Q_0 = 1 # Hadronization scale.

    t1 = 0.04
    t2 = 0.1
    t3 = 0.2
    t4 = 0.3
    
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
        
    
    # Gluon showers
    for i in range (0,n):
        Shower0 = generate_shower("gluon", t1, p_0, Q_0, 1, R, i, 4*n)
        gluonhard1.append(Shower0.Hardest)
        gluonlist1.extend(Shower0.FinalFracList)

    for i in range (n,2*n):
        Shower0 = generate_shower("gluon", t2, p_0, Q_0, 1, R, i, 4*n)            
        gluonhard2.append(Shower0.Hardest)
        gluonlist2.extend(Shower0.FinalFracList)

    for i in range (2*n,3*n):
        Shower0 = generate_shower("gluon", t3, p_0, Q_0, 1, R, i, 4*n)
        gluonhard3.append(Shower0.Hardest)
        gluonlist3.extend(Shower0.FinalFracList)

    for i in range (3*n,4*n):
        Shower0 = generate_shower("gluon", t4, p_0, Q_0, 1, R, i, 4*n)
        gluonhard4.append(Shower0.Hardest)
        gluonlist4.extend(Shower0.FinalFracList)
        
        
    # Quark showers.
    for i in range (4*n,5*n):
        Shower0 = generate_shower("quark", t1, p_0, Q_0, 1, R, i, 4*n)
        quarkhard1.append(Shower0.Hardest)
        quarklist1.extend(Shower0.FinalFracList)

    for i in range (5*n,6*n):
        Shower0 = generate_shower("quark", t2, p_0, Q_0, 1, R, i, 4*n)            
        quarkhard2.append(Shower0.Hardest)
        quarklist2.extend(Shower0.FinalFracList)

    for i in range (6*n,7*n):
        Shower0 = generate_shower("quark", t3, p_0, Q_0, 1, R, i, 4*n)
        quarkhard3.append(Shower0.Hardest)
        quarklist3.extend(Shower0.FinalFracList)
        
    for i in range (7*n,8*n):
        Shower0 = generate_shower("quark", t4, p_0, Q_0, 1, R, i, 4*n)
        quarkhard4.append(Shower0.Hardest)
        quarklist4.extend(Shower0.FinalFracList)

        
    # Now calculating normalizations for Gluons.
    print("\rCalculating bins 1...", end="")
    linbins = np.linspace(0, 1, num=bins)
    binlist = []
    
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
        density = len(frequencylist1)/(len(gluonhard1)*(binwidth))
        gluonbinlist1.append(density)
        
        for initialfrac in gluonhard1:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(len(gluonhard1)*(binwidth))
        gluonbinhardest1.append(density)
    
    
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
        density = len(frequencylist1)/(len(gluonhard2)*(binwidth))
        gluonbinlist2.append(density)
        
        for initialfrac in gluonhard2:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(len(gluonhard2)*(binwidth))
        gluonbinhardest2.append(density)
        
        
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
        density = len(frequencylist1)/(len(gluonhard3)*(binwidth))
        gluonbinlist3.append(density)
        
        for initialfrac in gluonhard3:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(len(gluonhard3)*(binwidth))
        gluonbinhardest3.append(density)
        

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
        density = len(frequencylist1)/(len(gluonhard4)*(binwidth))
        gluonbinlist4.append(density)
        
        for initialfrac in gluonhard4:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(len(gluonhard4)*(binwidth))
        gluonbinhardest4.append(density)
    
    
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
        quarkdensity = len(frequencylist1)/(len(quarkhard1)*(binwidth))
        quarkbinlist1.append(quarkdensity)
        
        for initialfrac in quarkhard1:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(len(quarkhard1)*(binwidth))
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
        quarkdensity = len(frequencylist1)/(len(quarkhard2)*(binwidth))
        quarkbinlist2.append(quarkdensity)
        
        for initialfrac in quarkhard2:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(len(quarkhard2)*(binwidth))
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
        quarkdensity = len(frequencylist1)/(len(quarkhard3)*(binwidth))
        quarkbinlist3.append(quarkdensity)
        
        for initialfrac in quarkhard3:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(len(quarkhard3)*(binwidth))
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
        quarkdensity = len(frequencylist1)/(len(quarkhard4)*(binwidth))
        quarkbinlist4.append(quarkdensity)
        
        for initialfrac in quarkhard4:
            if initialfrac > linbins[i] and initialfrac <= linbins[i+1]:
                frequencylist2.append(initialfrac)
        density = len(frequencylist2)/(len(quarkhard4)*(binwidth))
        quarkbinhardest4.append(density)
    

    # Now starting the plotting.
    plt.figure(dpi=1000, figsize= (6,5)) #(w,h) figsize= (10,3)
    title = ("Quarks and Gluons in Vaccum. showers: " + str(n) + 
             "\nepsilon: " + str(epsilon))
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

    ax1.plot(binlist, gluonbinlist1, 'g-', label ="gluon")
    ax1.plot(binlist, gluonbinhardest1, 'g--')
    ax1.plot(binlist, quarkbinlist1, 'b-', label ="quark")
    ax1.plot(binlist, quarkbinhardest1, 'b--')
    ax1.set_yscale("log")

    ax1.set_title('t_0 = ' + str(t1))
    ax1.set_xlim(0,1)
    ax1.set_ylim(0.01,10)
    ax1.set_xlabel('z ')
    ax1.set_ylabel('inclusive distribution')
    ax1.grid(linestyle='dashed', linewidth=0.2)
    ax1.legend()
    
    
    print("\rPlotting 2...", end="")

    ax2.plot(binlist, gluonbinlist2, 'g-', label ="gluon")
    ax2.plot(binlist, gluonbinhardest2, 'g--')
    ax2.plot(binlist, quarkbinlist2, 'b-', label ="quark")
    ax2.plot(binlist, quarkbinhardest2, 'b--')
    ax2.set_yscale("log")

    ax2.set_title('t_0 = ' + str(t2))
    ax2.set_xlim(0,1)
    ax2.set_ylim(0.01,10)
    ax2.set_xlabel('z')
    ax2.set_ylabel('inclusive distribution')
    ax2.grid(linestyle='dashed', linewidth=0.2)
    ax2.legend()

    
    print("\rPlotting 3...", end="")

    ax3.plot(binlist, gluonbinlist3, 'g-', label ="gluon")
    ax3.plot(binlist, gluonbinhardest3, 'g--')
    ax3.plot(binlist, quarkbinlist3, 'b-', label ="quark")
    ax3.plot(binlist, quarkbinhardest3, 'b--')
    ax3.set_yscale("log")

    ax3.set_title('t_0 = ' + str(t3))
    ax3.set_xlim(0,1)
    ax3.set_ylim(0.01,10)
    ax3.set_xlabel('z ')
    ax3.set_ylabel('inclusive distribution')
    ax3.grid(linestyle='dashed', linewidth=0.2)
    ax3.legend()
    

    print("\rPlotting 4...", end="")

    ax4.plot(binlist, gluonbinlist4, 'g-', label ="gluon")
    ax4.plot(binlist, gluonbinhardest4, 'g--')
    ax4.plot(binlist, quarkbinlist4, 'b-', label ="quark")
    ax4.plot(binlist, quarkbinhardest4, 'b--')
    ax4.set_yscale("log")

    ax4.set_title('t_0 = ' + str(t4))
    ax4.set_xlim(0,1)
    ax4.set_ylim(0.01,10)
    ax4.set_xlabel('z ')
    ax4.set_ylabel('inclusive distribution')
    ax4.grid(linestyle='dashed', linewidth=0.2)
    ax4.legend()


    print("\rShowing", end="")

    plt.tight_layout()
    plt.show()
    print("\rDone!")    
