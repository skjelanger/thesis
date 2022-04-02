# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Module contains all the vacuum splitting functions.

# Color and flavor factors.
C_A = 3
C_F = 4/3
N_f = 5
T_f = 1/2

# Splitting functions.
def gg_simple(z):
    return C_A*(1/(z*(1-z)))

def gg_full(z):
    return C_A*((1-z)/z + z/(1-z) + z*(1-z))

def qg_full(z):
    return N_f*T_f*(z**2 + (1-z)**2)

def qq_simple(z):
    return (-2*C_F)/(z-1)

def qq_full(z):
    return ((C_F)*(1+z**2))/(1-z)
