# -*- coding: utf-8 -*-
#Kristoffer spyder

#Main parton shower program. V2.0
#Contains quarks and gluons in vacuum. 

import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.integrate import quad # Integration package.
from treelib import Tree # Library for plotting the tree.
from seaborn import histplot
from pandas import DataFrame, Series

#constants
epsilon = 10**(-3)

bins = 50
minimum_bin = 0.001

alpha_S = 0.12
C_A = 3
C_F = 4/3
T_F = 1/2
N_F = 5


# Define class functions
class IterParton(type): #Makes class iterable
    def __iter__(cls):
        return iter(cls.allPartons)
    
class IterShower(type): #Makes class iterable
    def __iter__(cls):
        return iter(cls.allShowers)
    
class Shower(metaclass= IterShower):
    allShowers = []

    def __init__(self, initialtype, showernumber):
        self.allShowers.append(self)

        self.InitialType = initialtype
        self.ShowerNumber = showernumber
        self.PartonList = [] # List for all Shower Partons.
        self.FinalList = [] # List for all Final Partons.
        self.FinalFracList = [] # List for Final Parton initialfracs.
        self.SplittingQuarks = []
        self.SplittingGluons = []
        self.ShowerGluons = None
        self.ShowerQuarks = None
        self.Hardest = None

class Parton(metaclass= IterParton):
    allPartons = []

    def __init__(self, typ, angle, initialfrac, parent, shower):
        self.allPartons.append(self)
        
        self.Type = typ
        self.Angle = angle
        self.InitialFrac = initialfrac
        self.Parent = parent
        self.Primary = None
        self.Secondary = None
        self.Shower = shower
        
    
    def split(self): #Self splitting.
        if self.Type == "gluon": #determine what splitting function to use.
            rnd4 = np.random.uniform(0, 1) #random selection of splitting
            if rnd4 < ggg_contribution:
                splittingvalue = Parton.vacuum_ggg(self)
                parton1_type = "gluon"
                parton2_type = "gluon"
            elif rnd4 >= ggg_contribution:
                splittingvalue = Parton.vacuum_gqq(self)
                parton1_type = "quark"
                parton2_type = "quark"
            
        elif self.Type == "quark":
            splittingvalue = Parton.vacuum_qqg(self)
            parton1_type = "quark"
            parton2_type = "gluon"
            
        return splittingvalue, parton1_type, parton2_type
    
        
    def vacuum_ggg(self): # vacuum ggg splitting function MH hastings
        while True: # loop for generating MH value
            rnd1 = np.random.uniform(epsilon, 1-epsilon)
            xi = ((1-epsilon)/epsilon)**(2*rnd1-1)
            splittingvalue = xi/(1+xi) #Randomly generated initial state from the distribution f(x)
            acceptance = min(1-epsilon, (p_ggg(splittingvalue) / p_ggg_dummy(splittingvalue)))#acceptance ratio determinedby inclusive distribution ratios.
            rnd2 = np.random.uniform(0,1) # random value for acceptance if statement.
                
            if acceptance >= rnd2: #Condition for accepting state.
                break

        return splittingvalue
                
        
    def vacuum_qqg(self): # vacuum qqg splitting function MH hastings
        while True: # loop for generating MH value
            rnd1 = np.random.uniform(epsilon, 1-epsilon)
            xi = ((1-epsilon)/epsilon)**(rnd1)
            splittingvalue = ((epsilon-1)/(xi))+1 #Randomly generated initial state from the distribution f(x)            
            acceptance = min(1-epsilon, (p_qqg(splittingvalue) / p_qqg_dummy(splittingvalue)))#acceptance ratio determinedby inclusive distribution ratios.
            rnd2 = np.random.uniform(0,1) # random value for acceptance if statement.
                
            if acceptance >= rnd2: #Condition for accepting state.
                break
                    
        return splittingvalue
        
        
    def vacuum_gqq(self): # vacuum gqq splitting function
        rnd1 = np.random.uniform(epsilon,1-epsilon)
        d = (rnd1*0.665)+0.001
        a = (36 * (d**(2)) - (24 * d) + 5)**(1/2)
        b = ( a + (6* d) - 2)**(1/3)
        splittingvalue = (0.5 + 0.5*b - (0.5/b))
        return splittingvalue


# Splitting functions.

def p_ggg_dummy(z):
    return ((C_A)*(1/(z*(1-z))))

def p_ggg(z):
    return ((C_A)*((1-z)/z + z/(1-z) + z*(1-z)))

def p_qqg_dummy(z):
    return ((C_F)*((-2)/(z-1)))

def p_qqg(z):
    return ((C_F)*((1+z**2))/(1-z))

def p_gqq(z):
    return ((N_F*T_F)*(z**2 + (1-z)**2))


# Pre-calculations
p_ggg_integral, error1 = quad(p_ggg, epsilon, 1-epsilon)
p_qqg_integral, error2 = quad(p_qqg, epsilon, 1-epsilon) 
p_gqq_integral, error3 = quad(p_gqq, epsilon, 1-epsilon)
gluon_contribution = (p_ggg_integral+ p_gqq_integral)/(p_ggg_integral+ p_qqg_integral + p_gqq_integral)
ggg_contribution = (p_ggg_integral)/(p_ggg_integral+ p_gqq_integral)

print("Gluon contribution:", gluon_contribution)
print("ggg contribution:",  ggg_contribution)

# Shower parameter functions

def advance_time(): #Time interval
    rnd1 = np.random.uniform(epsilon,1-epsilon) # A single value
    #delta_t = -(np.log(rnd1))/(p_ggg_integral+ p_qqg_integral + p_gqq_integral)
    delta_t_quark = -(np.log(rnd1))/(p_qqg_integral)
    delta_t_gluon = -(np.log(rnd1))/(p_ggg_integral + p_gqq_integral)

    return delta_t_quark, delta_t_gluon


def select_splitting(Shower0, t, delta_t_quark, delta_t_gluon):
    if len(Shower0.SplittingGluons)>0 and len(Shower0.SplittingQuarks)>0: #Split condition 1, can split either.
        rnd3 = np.random.uniform(0, 1)

        if (gluon_contribution <= rnd3 ):
            SplittingParton = random.choice(Shower0.SplittingQuarks)
            Shower0.SplittingQuarks.remove(SplittingParton)
            t = t + delta_t_quark
                
        elif (gluon_contribution > rnd3 ):
            SplittingParton = random.choice(Shower0.SplittingGluons)
            Shower0.SplittingGluons.remove(SplittingParton)
            t = t + delta_t_gluon
                
    elif len(Shower0.SplittingGluons)>0 and len(Shower0.SplittingQuarks) == 0 : #Split condition 2, can only split gluon.
        SplittingParton = random.choice(Shower0.SplittingGluons)
        Shower0.SplittingGluons.remove(SplittingParton)
        t = t + delta_t_gluon
        
    elif len(Shower0.SplittingGluons)==0 and len(Shower0.SplittingQuarks) > 0 : #Split condition 3, can only split quark.
        SplittingParton = random.choice(Shower0.SplittingQuarks)
        Shower0.SplittingQuarks.remove(SplittingParton)
        t = t + delta_t_quark
        
    else:
        #print("\rERROR: no select_splitting criteria fulfilled.")
        return None, t
    
    return SplittingParton, t


def generate_shower(initialtype, t_max , p_t, Q_0, z_0, R, showernumber, n):
    print("\rLooping... " + str(round(100*showernumber/(n))) +"%", end="", flush=True)

    #t_max = (alpha_S / (np.pi)) * np.log((p_t*R)/(p_t*R-Q_0)) 
    t = 0

    Shower0 = Shower(initialtype, showernumber) #creates shower object.
    Parton0 = Parton(initialtype, t, z_0, None, Shower0) # creates initial parton
    Shower0.PartonList.append(Parton0)
    Shower0.FinalList.append(Parton0)

    if initialtype == "quark":
        Shower0.SplittingQuarks.append(Parton0)
    elif initialtype == "gluon":
        Shower0.SplittingGluons.append(Parton0)
        
    while True:
        delta_t_quark, delta_t_gluon = advance_time() #advances time.
            
        if ((t+delta_t_quark) < t_max) and ((t+delta_t_gluon) < t_max): #  t small enough for both splittings.
            SplittingParton, t = select_splitting(Shower0, t, delta_t_quark, delta_t_gluon)
            if SplittingParton == None:
                break
            
        elif ((t+delta_t_quark) >= t_max) and ((t+delta_t_gluon) < t_max): # can only split gluon.
            if len(Shower0.SplittingGluons) > 0:
                SplittingParton = random.choice(Shower0.SplittingGluons)
                Shower0.SplittingGluons.remove(SplittingParton)
                t = t + delta_t_gluon
            else: #can not split any gluons.
                break
            
        else: #can not split any t
            if delta_t_quark < delta_t_gluon:
                print("ERROR: delta_t_quark < delta_t_gluon. Showernumber is: ", showernumber)
            break
            
        Shower0.FinalList.remove(SplittingParton)
        momfrac, parton1_type, parton2_type = Parton.split(SplittingParton) # Do the correct splitting for the object, and gives us momfrac.            

        for j in range(0,2): #Loop for  generating the branched partons
            if j==0: # Initialfrac for parton 1
                initialfrac = SplittingParton.InitialFrac * momfrac
                parton_type = parton1_type
                
            elif j==1: # Initialfrac for parton 1
                initialfrac = SplittingParton.InitialFrac * (1-momfrac)
                parton_type = parton2_type

            
            NewParton = Parton(parton1_type, t, initialfrac, SplittingParton, Shower0)
            SplittingParton.Primary = NewParton
            Shower0.PartonList.append(NewParton)
                   
            if initialfrac > 0.0001:
                Shower0.FinalList.append(NewParton)

                if parton_type == "quark" :
                    Shower0.SplittingQuarks.append(NewParton)
                            
                elif parton_type == "gluon" :
                    Shower0.SplittingGluons.append(NewParton)
            
            elif initialfrac <= 0.0001:
                del NewParton
                
                
    #create_parton_tree(showernumber)
    
    showergluons = 0
    showerquarks = 0
    fractionsum = 0
    for PartonObj in Shower0.FinalList:
        if PartonObj.InitialFrac > minimum_bin:
            Shower0.FinalFracList.append(PartonObj.InitialFrac)
        fractionsum += PartonObj.InitialFrac
        if PartonObj.Type =="gluon": 
            showergluons += 1
        
        elif PartonObj.Type == "quark":
            showerquarks += 1
            
    if fractionsum > 1+epsilon or fractionsum < 1-epsilon :
        print("Fractionsum is: ", fractionsum, ". Showernumber: ", showernumber)
   
    Shower0.ShowerGluons = showergluons
    Shower0.ShowerQuarks = showerquarks
    Shower0.Hardest = max(Shower0.FinalFracList)

    return Shower0


def create_parton_tree(showernumber): # Treelib print parton tree.
    tree = Tree()
    
    for ShowerObj in Shower: 
        if ShowerObj.ShowerNumber == showernumber:
            print("ShowerNumber is:", ShowerObj.ShowerNumber, ". Number of partons is: ", len(ShowerObj.FinalList))
        
            for PartonObj in ShowerObj.PartonList: # Loop for creating branching tree.
                title = str(round(PartonObj.InitialFrac, 3)) +" - " +  PartonObj.Type
                tree.create_node(title, PartonObj, parent= PartonObj.Parent)

    tree.show()
    tree.remove_node(tree.root)
    return
    

def several_showers(n, initialtype):
    z_0 = 1 # Define intial parton momfrac.
    R = 0.4 # Define jet radius.    
    p_0 = 100

    showersum = 0
    quarksum = 0
    gluonsum = 0
    List1 = []
    Hardest1 =[]
    
    for i in range (0,n):
        Shower0 = generate_shower(initialtype, 0.04, p_0, 1, z_0, R, i, 4*n)
        Hardest1.append(Shower0.Hardest)
        List1.extend(Shower0.FinalFracList)
        showersum += Shower0.ShowerPartons
        gluonsum += Shower0.ShowerGluons
        quarksum += Shower0.ShowerQuarks
        
    average = showersum/n
    gluonaverage = gluonsum/n
    quarkaverage = quarksum/n
    print("\rThe average number of shower partons when running ", n, "showers, is ", average)
    print("The average shower consisted of ", quarkaverage, " quarks, and ", gluonaverage, " gluons.")
    
    showersum = 0
    quarksum = 0
    gluonsum = 0
    List2 = []
    Hardest2 =[]
    for i in range (n,2*n):
        Shower0 = generate_shower(initialtype, 0.1, p_0, 1, z_0, R, i, 4*n)
        Hardest2.append(Shower0.Hardest)
        List2.extend(Shower0.FinalFracList)
        showersum += Shower0.ShowerPartons
        gluonsum += Shower0.ShowerGluons
        quarksum += Shower0.ShowerQuarks
        
    average = showersum/n
    gluonaverage = gluonsum/n
    quarkaverage = quarksum/n
    print("\rThe average number of shower partons when running ", n, "showers, is ", average)
    print("The average shower consisted of ", quarkaverage, " quarks, and ", gluonaverage, " gluons.")
    
    showersum = 0
    quarksum = 0
    gluonsum = 0
    List3 = []
    Hardest3 =[]
    for i in range (2*n,3*n):
        Shower0 = generate_shower(initialtype, 0.2, p_0, 1, z_0, R, i, 4*n)
        Hardest3.append(Shower0.Hardest)
        List3.extend(Shower0.FinalFracList)
        showersum += Shower0.ShowerPartons
        gluonsum += Shower0.ShowerGluons
        quarksum += Shower0.ShowerQuarks
        
    average = showersum/n
    gluonaverage = gluonsum/n
    quarkaverage = quarksum/n
    print("\rThe average number of shower partons when running ", n, "showers, is ", average)
    print("The average shower consisted of ", quarkaverage, " quarks, and ", gluonaverage, " gluons.")
    
    showersum = 0
    quarksum = 0
    gluonsum = 0
    List4 = []
    Hardest4 =[]
    for i in range (3*n,4*n):
        Shower0 = generate_shower(initialtype, 0.3, p_0, 1, z_0, R, i, 4*n)
        Hardest4.append(Shower0.Hardest)
        List4.extend(Shower0.FinalFracList)
        showersum += Shower0.ShowerPartons
        gluonsum += Shower0.ShowerGluons
        quarksum += Shower0.ShowerQuarks
        
    average = showersum/n
    gluonaverage = gluonsum/n
    quarkaverage = quarksum/n
    print("\rThe average number of shower partons when running ", n, "showers, is ", average)
    print("The average shower consisted of ", quarkaverage, " quarks, and ", gluonaverage, " gluons.")
    
    # Plot    

    plt.figure(dpi=1000) #figsize= (10,3)
    title = "Quarks and Gluons in Vaccum. Number of showers: " + str(n) + ".\n Initial parton: " + str(initialtype) + ". p_0 = " +str(p_0)
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
        
    print("\rPlotting 1...")
    
    
    mydict1 = {'Incl': List1, "Hard": Hardest1}
    df1 = DataFrame({ key: Series(value) for key, value in mydict1.items() })
    
    #kdeplot(ax = ax1, data=df1, bw_adjust=0.6, common_norm=True, cut=0)
    histplot(ax = ax1, data=df1, bins=200, kde=True, kde_kws={"bw_adjust": 0.2}, stat='density', common_norm=True, log_scale = (False,True))
 
    ax1.set_title('t_0 = 0.04')
    ax1.set_xlim(0,1)
    ax1.set_ylim(0.01,10)
    ax1.set_xlabel('z ')
    ax1.set_ylabel('inclusive distribution')
    ax1.grid(linestyle='dashed', linewidth=0.2)
    
    
    print("Plotting 2...")

    mydict2 = {'Incl': List2, "Hard": Hardest2}
    df2 = DataFrame({ key: Series(value) for key, value in mydict2.items() })
    
    #kdeplot(ax = ax2, data=df2, bw_adjust=0.6, common_norm=True, cut=0)
    histplot(ax = ax2, data=df2, bins=200, kde=True, kde_kws={"bw_adjust": 0.35}, stat='density', common_norm=True, log_scale = (False,True))
  
    ax2.set_title('t_0 = 0.1')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0.01,10)
    ax2.set_xlabel('z')
    ax2.set_ylabel('inclusive distribution')
    ax2.grid(linestyle='dashed', linewidth=0.2)

    
    print("Plotting 3...")

    mydict3 = {'Incl': List3, "Hard": Hardest3}
    df3 = DataFrame({ key: Series(value) for key, value in mydict3.items() })
    
    #kdeplot(ax = ax3, data=df3, bw_adjust=0.6, common_norm=True, cut=0)
    histplot(ax = ax3, data=df3, bins=200, kde=True, kde_kws={"bw_adjust": 0.65}, stat='density', common_norm=True, log_scale = (False,True))

    ax3.set_title('t_0 = 0.2')
    ax3.set_xlim(0,1)
    ax3.set_ylim(0.01,10)
    ax3.set_xlabel('z ')
    ax3.set_ylabel('inclusive distribution')
    ax3.grid(linestyle='dashed', linewidth=0.2)
    

    print("Plotting 4...")

    mydict4 = {'Incl': List4, "Hard": Hardest4}
    df4 = DataFrame({ key: Series(value) for key, value in mydict4.items() })
    
    #kdeplot(ax = ax4, data=df4, bw_adjust=0.6, common_norm=True, cut=0)
    histplot(ax = ax4, data=df4, bins=200, kde=True, kde_kws={"bw_adjust": 0.8}, stat='density', common_norm=True, log_scale = (False,True))
    
    ax4.set_title('t_0 = 0.3')
    ax4.set_xlim(0,1)
    ax4.set_ylim(0.01,10)
    ax4.set_xlabel('z ')
    ax4.set_ylabel('inclusive distribution')
    ax4.grid(linestyle='dashed', linewidth=0.2)


    print("Showing")

    plt.tight_layout()
    
    plt.show()
    print("Done!")    
    
    

def several_showers_dasgupta(n): #plots the gasgupta distribution.
    z_0 = 1 # Define intial parton momfrac.
    R = 0.4 # Define jet radius.    
    p_0 = 100
    
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
        
    
    for i in range (0,n):
        Shower0 = generate_shower("gluon", t1, p_0, 1, z_0, R, i, 4*n)
        gluonhard1.append(Shower0.Hardest)
        gluonlist1.extend(Shower0.FinalFracList)

        
    for i in range (n,2*n):
        Shower0 = generate_shower("gluon", t2, p_0, 1, z_0, R, i, 4*n)            
        gluonhard2.append(Shower0.Hardest)
        gluonlist2.extend(Shower0.FinalFracList)

  
    for i in range (2*n,3*n):
        Shower0 = generate_shower("gluon", t3, p_0, 1, z_0, R, i, 4*n)
        gluonhard3.append(Shower0.Hardest)
        gluonlist3.extend(Shower0.FinalFracList)

        
    for i in range (3*n,4*n):
        Shower0 = generate_shower("gluon", t4, p_0, 1, z_0, R, i, 4*n)
        gluonhard4.append(Shower0.Hardest)
        gluonlist4.extend(Shower0.FinalFracList)
        
        
    for i in range (4*n,5*n):
        Shower0 = generate_shower("quark", t1, p_0, 1, z_0, R, i, 4*n)
        quarkhard1.append(Shower0.Hardest)
        quarklist1.extend(Shower0.FinalFracList)

        
    for i in range (5*n,6*n):
        Shower0 = generate_shower("quark", t2, p_0, 1, z_0, R, i, 4*n)            
        quarkhard2.append(Shower0.Hardest)
        quarklist2.extend(Shower0.FinalFracList)

  
    for i in range (6*n,7*n):
        Shower0 = generate_shower("quark", t3, p_0, 1, z_0, R, i, 4*n)
        quarkhard3.append(Shower0.Hardest)
        quarklist3.extend(Shower0.FinalFracList)

        
    for i in range (7*n,8*n):
        Shower0 = generate_shower("quark", t4, p_0, 1, z_0, R, i, 4*n)
        quarkhard4.append(Shower0.Hardest)
        quarklist4.extend(Shower0.FinalFracList)

        
        
    #Now calculating normalizations
    
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
    
    
    #NOW STARTING QUARKS
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
    

    # Plot    
    plt.figure(dpi=1000, figsize= (6,5)) #(w,h) figsize= (10,3)
    title = "Quarks and Gluons in Vaccum. showers: " + str(n) + "\nminimum_bin: " + str(minimum_bin)
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
    
    

def several_showers_analytical_comparison(n, t_0, initialtype):
    z_0 = 1 # Define intial parton momfrac.
    R = 0.4 # Define jet radius.    
    p_0 = 100
    Q_0 = 1


    showersum = 0
    quarksum = 0
    gluonsum = 0
    List1 = []
    Hardest1 =[]
    
    for i in range (0,n):
        Shower0 = generate_shower(initialtype, t_0, p_0, Q_0, z_0, R, i)
        Hardest1.append(Shower0.Hardest)
        List1.extend(Shower0.FinalFracList)
        showersum += Shower0.ShowerPartons
        gluonsum += Shower0.ShowerGluons
        quarksum += Shower0.ShowerQuarks
                
    average = showersum/n
    gluonaverage = gluonsum/n
    quarkaverage = quarksum/n
    print("The average number of shower partons when running ", n, "showers, is ", average)
    print("The average shower consisted of ", quarkaverage, " quarks, and ", gluonaverage, " gluons.")
    
    
    xrange = np.linspace(epsilon, 1-epsilon, 2000)
    solution = []
    solution2 = []
    
    t=t_0
    
    for x in xrange:
        D = (1/x)**(2*np.sqrt(t/(np.log(1/x))))
        D2 = 0.5* (t/(np.pi**2 * np.log(1/x)**3))**(1/4)*np.exp(-t*np.e )*np.exp(2 * np.sqrt(t* np.log(1/x)))
        solution.append(D)
        solution2.append(D2)
    
    # Plot    

    plt.figure(dpi=1000) #figsize= (10,3)
    title = "Quarks and Gluons in Vaccum. Number of showers: " + str(n) + ".\n Initial parton: " + str(initialtype) + ". p_0 = " +str(p_0)
    plt.suptitle(title)

    plt.rc('axes', titlesize="small" , labelsize="x-small")     # fontsize of the axes title and labels.
    plt.rc('xtick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('ytick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('legend',fontsize='xx-small')    # fontsize of the legend labels.
    plt.rc('lines', linewidth=0.8)

    ax1 = plt.subplot(121) #H B NR
    ax2 = plt.subplot(122)
    
    print("Plotting...")
    
    
    mydict1 = {'Incl': List1, "Hard": Hardest1}
    df1 = DataFrame({ key: Series(value) for key, value in mydict1.items() })
    
    #kdeplot(ax = ax1, data=df1, bw_adjust=0.6, common_norm=True, cut=0)
    histplot(ax = ax1, data=df1, bins=100, kde=True, kde_kws={"bw_adjust": 0.2}, stat='density', common_norm=True, log_scale = (False,True))
    ax1.plot(xrange, solution, label="solution1")
    ax1.plot(xrange, solution2, label="solution2" )
    ax2.plot(xrange, solution, label="solution1")
    ax2.plot(xrange, solution2, label="solution2")
    ax1.set_title('')
    ax1.set_xlim(0,1)
    ax1.set_ylim(0.01,10)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0.01,10)
    ax2.set_yscale("log")
    ax1.set_xlabel('z ')
    ax1.set_ylabel('inclusive distribution')
    ax1.grid(linestyle='dashed', linewidth=0.2)
    


    print("Showing")

    plt.tight_layout()
    
    plt.show()
    print("Done!")    
    

def evolution_variable():
    Q_0 = 1
    p_values = [10, 20, 50, 100, 250, 500, 2000]
    Rrange = np.linspace(0.99999, 0.00001, 100000)
    tvalues = []
    
    alpha_S = 0.12
    
    for p_t in p_values:
        tvalue = []
        for R in Rrange:
            if (R*p_t) > Q_0:
                #t_old = (alpha_S / np.pi) * np.log((p_t*R)/(Q_0))
                #t_max = (alpha_S / (np.pi)) * np.log((p_t*R)/(p_t*R-Q_0))
                
                y = (Q_0)/(p_t*R)
                t_max = (alpha_S / (np.pi)) * np.log(1/(1-y))

                tvalue.append(t_max)
            else:
                tvalue.append(None)
        tvalues.append(tvalue)
        
    # Plot    

    plt.figure(dpi=1000) #figsize= (10,3)
    title = "Evolution variable vacuum showers. alpha_S = " + str(alpha_S)
    #plt.suptitle(title)

    plt.rc('axes', titlesize="small", labelsize="x-small")     # fontsize of the axes title and labels.
    plt.rc('xtick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('ytick', labelsize="x-small")    # fontsize of the tick labels.
    plt.rc('legend',fontsize='xx-small')    # fontsize of the legend labels.
    plt.rc('lines', linewidth=0.8)

    ax1 = plt.subplot(111) #H B NR
    
    for i in range(len(p_values)):  
        print("Plotting... " + str(i))
        ax1.plot(Rrange, tvalues[i], label= "p_t: " + str(p_values[i]) + " GeV")
    
    ax1.set_title('')
    ax1.set_xlim(0.0001, 1)
    ax1.set_ylim(0, 0.40)
    ax1.set_xscale("log")
    ax1.set_xlabel('R ')
    ax1.set_ylabel('t')
    ax1.grid(linestyle='dashed', linewidth=0.2)

    print("Showing")

    plt.legend()
    plt.tight_layout()
    
    plt.show()
    print("Done!")    
    
        