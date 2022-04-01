# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Module contains all the medium splitting functions.

# Color and flavor factors.
C_A = 3
C_F = 4/3
N_f = 5
T_f = 1/2

def gg_simple(z):
    return C_A**(3/2)*(1/((z*(1-z))**(3/2)))

def gg_full(z):
    A = C_A
    B = (1-z)/z + z/(1-z) + z*(1-z)
    C = ((C_A*(1-z)+C_A*(z**2))/(z*(1-z)))**(1/2)
    return A*B*C

def qg_simple(z):
    return N_f*T_f*(1/(z*(1-z))**(1/2))
    
def qg_full(z):
    return N_f*T_f* (z**2+(1-z)**2)*((C_F-z*(1-z)*C_A)/(z*(1-z)))**(1/2)

def qq_simple(z):
    return C_F*(1/2)*((4)/(z**(1/2)*(1-z)**(3/2)))

def qq_full(z):
    return C_F*(1/2)*((1+z**2)/(1-z))*((z*C_A+C_F*(1-z)**2)/(z*(1-z)))**(1/2)