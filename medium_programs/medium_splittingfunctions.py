# -*- coding: utf-8 -*-
# Kristoffer Skjelanger, Master thesis, UiB.
# Module contains all the medium splitting functions.

# Color and flavor factors.
C_A = 3
C_F = 4/3
N_f = 5
T_f = 1/2

def gg_simple_analytical(z):
    return (1/(z*(1-z))**(3/2))

def gg_simple(z):
    return (2/(z*(1-z))**(3/2))

def gg_full(z):
    A = 1
    B = (1-z)/z + z/(1-z) + z*(1-z)
    C = ((C_A*(1-z)+C_A*(z**2))/(z*(1-z)))**(1/2)
    return A*B*C

def qg_simple(z):
    return (1/(z*(1-z))**(1/2))
    
def qg_full(z):
    return (z**2+(1-z)**2)*((C_F-z*(1-z)*C_A)/(z*(1-z)))**(1/2)

def qq_simple(z):
    return ((4)/(z**(1/2)*(1-z)**(3/2)))

def qq_full(z):
    return ((1+z**2)/(1-z))*((z*C_A+C_F*(1-z)**2)/(z*(1-z)))**(1/2)