from scipy import optimize
import numpy as np
import sympy as sm
from sympy.solvers import solve
from sympy import Symbol
import matplotlib.pyplot as plt
from types import SimpleNamespace
from tabulate import tabulate
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

class SolowModelClass:
    
    def __init__(self, do_print=True):
        """ create the model """

        # if do_print: print('initializing the model:')
        self.par = SimpleNamespace()
        self.val = SimpleNamespace()
        self.sim = SimpleNamespace()

        # if do_print: print('calling .setup()')
        self.setup()
    
    def setup(self):
        """ baseline parameters """

        val = self.val
        par = self.par
        sim = self.sim

        # model parameters for analytical solution
        par.k = sm.symbols('k')
        par.alpha = sm.symbols('alpha')
        par.delta = sm.symbols('delta')
        par.phi =  sm.symbols('phi')
        par.sK = sm.symbols('s_k')
        par.sH = sm.symbols('s_h')
        par.g = sm.symbols('g')
        par.n = sm.symbols('g')
        par.A = sm.symbols('A')
        par.K = sm.symbols('K')
        par.Y = sm.symbols('Y')
        par.L = sm.symbols('L')
        par.k_tilde = sm.symbols('k_tilde')
        par.h_tilde = sm.symbols('h_tilde')
        par.y_tilde = sm.symbols('y_tilde')
        par.k_tilde_ss = sm.symbols('k_tilde_ss')
        par.h_tilde_ss = sm.symbols('h_tilde_ss')

        # model parameter values for numerical solution
        val.sK = 0.1
        val.sH = 0.1
        val.g = 0.05
        val.n = 0.33
        val.alpha = 0.33
        val.delta = 0.3
        val.phi = 0.02

        # simulation parameters for further analysis
        par.simT = 100 #number of periods
        sim.K = np.zeros(par.simT)
        sim.L = np.zeros(par.simT)
        sim.A = np.zeros(par.simT)
        sim.Y = np.zeros(par.simT)
        sim.H = np.zeros(par.simT)

    def solve_analytical_ss(self):
        """ function that solves the model analytically and returns k_tilde in steady state """

        par = self.par
        # Define transistion equations
        trans_k = sm.Eq(par.k_tilde, 1/((1+par.n)*(1+par.g))*(par.sK*par.k_tilde**par.alpha*par.h_tilde**par.phi+(1-par.delta)*par.k_tilde))
        trans_h = sm.Eq(par.h_tilde, 1/((1+par.n)*(1+par.g))*(par.sH*par.k_tilde**par.alpha*par.h_tilde**par.phi+(1-par.delta)*par.h_tilde))

        # solve the equation for k_tilde and h_tilde
        k_tilde_ss = sm.solve(trans_k, par.k_tilde)[0]
        h_tilde_ss = sm.solve(trans_h, par.h_tilde)[0]

        # Print the solutions
        sm.pprint(k_tilde_ss)
        sm.pprint(h_tilde_ss)

        return k_tilde_ss, h_tilde_ss
    
    def solve_numerical_ss(self):
        """ function that solves the model numerically and returns k_tilde in steady state """

        par = self.val

        # define the steady state equation for k_tilde
        def k_steady_state_eq(k_tilde):
            return k_tilde - (1/((1+par.n)*(1+par.g))*(par.sK*par.k_tilde**par.alpha*par.h_tilde**par.phi)+(1-par.delta)*par.k_tilde)
        
         # define the steady state equation for h_tilde
        def h_steady_state_eq(h_tilde):
            return h_tilde - (1/((1+par.n)*(1+par.g))*(par.sH*par.k_tilde**par.alpha*par.h_tilde**par.phi)+(1-par.delta)*par.h_tilde)

        # make an initial guess for the solution
        initial_guess = 0.5

        # solve the equation numerically
        k_tilde_ss = optimize.root(k_steady_state_eq, initial_guess).x[0]
        h_tilde_ss = optimize.root(h_steady_state_eq, initial_guess).x[0]
        sm.pprint(k_tilde_ss, h_tilde_ss)

        return k_tilde_ss, h_tilde_ss