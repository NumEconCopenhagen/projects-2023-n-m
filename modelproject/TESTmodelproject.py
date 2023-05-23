from scipy import optimize
import numpy as np
import time
from types import SimpleNamespace
import matplotlib
from matplotlib import pyplot as plt 
import ipywidgets as widgets
from ipywidgets import Output, SelectionSlider
from IPython.display import display, clear_output


class Solow():

    def __init__(self, do_ = True):
        """ define the model """

        if do_: ('initialising the model')

        self.par = SimpleNamespace()

        if do_: ('calling .setup()')
        self.setup()


    def setup(self):

        par = self.par

        par.production_function = 'cobb-douglas'
        par.alpha = 1/3
        par.phi = 1/3

        par.n = 0.01    
        par.g = 0.02
        par.delta = 0.05
        par.sH = 0.15
        par.sK = 0.2
        
        par.A_init = 1
        par.K_init = 1
        par.H_init = 1
        par.L_init = 1

        par.simT = 1000

    
    def find_steady_state(self, sK=0.2, sH=0.15, tol=1e-6, do_print=False):
        """
        Returns: 
        sim_out: namespace, contains simulated variables, used parameters, and index of when 
                            in steady state for all periods from 0 to T-1.  
            
        Args: 
        sK: float, savings rate for physical capital.
        sH: float, savings rate for human capital.
        tol: float, tolerance for when in steady state.
        do_print: bool, print what period steady state is reached. 
        """

        par = self.par
        sim_out = self.sim_out = SimpleNamespace() # make empty simulation

        # a. pre-allocate memory
        A = np.empty(par.simT)
        K = np.empty(par.simT)
        H = np.empty(par.simT)
        L = np.empty(par.simT)
        Y = np.empty(par.simT)
        y_tilde = np.empty(par.simT)
        k_tilde = np.empty(par.simT)
        h_tilde = np.empty(par.simT) 
        y_t = np.empty(par.simT)
        
        steady_state_periods = []

        # b. allocate initial values for A,L,K,H.
        for i,j in zip([A,L,K,H], [par.A_init, par.L_init, par.K_init, par.H_init]):
            i[0] = j
        
        
        # c. do simulation for all periods
        t = 0
        while t < par.simT - 1:

            # i. calculate values for period t that are reliant on A,L,K,H.   
            if par.production_function == 'cobb-douglas':
                Y[t] = (K[t]**par.alpha)*(H[t]**par.phi)*(A[t]*L[t])**(1-par.alpha-par.phi)
            else:
                Y[t] = np.nan

            y_tilde[t] = Y[t]/(A[t]*L[t])
            k_tilde[t] = K[t]/(A[t]*L[t])
            h_tilde[t] = H[t]/(A[t]*L[t])
            y_t[t] = Y[t]/L[t]

            # ii. check if in steady state.  
            if (t>1) and (abs(k_tilde[t]-k_tilde[t-1]) < tol) and (abs(h_tilde[t]-h_tilde[t-1]) < tol):
                steady_state_periods += [t] # all instances in t, when in steady state. 
                if (do_print == True) and (t == steady_state_periods[0]):
                    print(f"Steady state reached in period {t}") 
            
            # iii. calculate values for next period of A,L,K,H. 
            A[t+1] = A[t]*(1+par.g)     
            L[t+1] = L[t]*(1+par.n) 
            K[t+1] = Y[t]*sK + (1-par.delta)*K[t]
            H[t+1] = Y[t]*sH + (1-par.delta)*H[t]

            t += 1

        # d. insert in namespace simulation
        # i. variables 
        sim_out.y_tilde = y_tilde[:t-1]
        sim_out.k_tilde = k_tilde[:t-1]
        sim_out.h_tilde = h_tilde[:t-1]
        sim_out.y_t = y_t[:t-1]
        sim_out.A = A[:t-1]
        sim_out.K = K[:t-1]
        sim_out.H = H[:t-1]
        sim_out.L = L[:t-1]

        # ii. used parameters and index
        sim_out.sK = sK
        sim_out.sH = sH 
        sim_out.steadystate_t = steady_state_periods[0]

        return sim_out
    
    def anal_steady_state(self):
        """
        Returns: 
        anal_sol: namespace, contains analytical steady state solutions for all relevant tilde-variables.    

        """
        anal_sol = self.anal_sol = SimpleNamespace() #empty 
        par = self.par 

        u = par.n + par.g + par.delta + par.g*par.n # to ease the length of the analytic calculations
    
        anal_sol.k_tilde = (par.sK/u)**((1-par.phi)/(1-par.phi-par.alpha))*(par.sH/u)**(par.phi/(1-par.phi-par.alpha))
        anal_sol.h_tilde = (par.sH/u)**((1-par.alpha)/(1-par.phi-par.alpha))*(par.sK/u)**(par.alpha/(1-par.phi-par.alpha))
        anal_sol.y_tilde = anal_sol.k_tilde**par.alpha*anal_sol.h_tilde**par.phi

        return anal_sol 
    
    def null_k_func_anal(self, ktilde_t, alpha, delta, g, n, phi, s_K):
        # analytical nullcline for k  
        return (ktilde_t**(1 - alpha)*(delta + g*n + g + n)/s_K)**(phi**(-1.0))
    
    def null_h_func_anal(self, ktilde_t, alpha, delta, g, n, phi, s_H): 
        # analytical nullcline for h
        return (ktilde_t**(-alpha)*(delta + g*n + g + n)/s_H)**((phi - 1)**(-1.0))


    def plot_convergence(self, H_init, K_init):

        """ 
        Returns: graph of null clines from analytical solution and simulated convergence
        
        Args: 
        discrete, float, initial values for K and H
        
        """
        
        par = self.par 

        # a. define initial value
        par.H_init = H_init
        par.K_init = K_init

        # b. extract simulation & unpack 
        sim_out = self.find_steady_state() 
        k_t = sim_out.k_tilde
        h_t = sim_out.h_tilde

        # c. insert parameter values from simulation in nullclines  
        # i. define values 
        alpha_val = par.alpha
        delta_val = par.delta
        g_val = par.g
        n_val = par.n
        phi_val = par.phi
        
        sK_val = sim_out.sK
        sH_val = sim_out.sH

        # ii. find range of k_tilde for plot
        k_tilde_vec = np.linspace(1e-10, max(k_t)+5, 100)

        # iii. insert in lamdified nullclines
        # Values for analytical null clines
        null_k_val = self.null_k_func_anal(k_tilde_vec,alpha_val,delta_val,g_val, n_val, phi_val, sK_val)
        null_h_val = self.null_h_func_anal(k_tilde_vec,alpha_val,delta_val,g_val, n_val, phi_val, sH_val)

        # d. plot results
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.clear()
        ax.plot(k_tilde_vec, null_k_val, label = r'$ \Delta \tilde{k}_t = 0$')
        ax.plot(k_tilde_vec, null_h_val, label = r'$ \Delta \tilde{h}_t = 0$')
        ax.plot(k_t, h_t, label='simulation', linestyle = "dotted", linewidth = 2)
        ax.set_xlabel(r'$\tilde{k}_t$',)
        ax.set_ylabel(r'$\tilde{h}_t$',)
        ax.set_title('Fig 4. Path to steady state', size=12)

        ax.legend(loc='upper left')
        plt.plot()
    
    def plot_convergence_interactive(self):
        out2=widgets.interact(self.plot_convergence, H_init = widgets.SelectionSlider(options=np.linspace(0,50,51), value=3),
                            K_init = widgets.SelectionSlider(options=np.linspace(0,50,51), value=2))
        return display(out2)

    def cons_t(self, sK=0.2, sH=0.15):
        """
        Returns: consumption vector from simulation 

        """

        sim_out= self.find_steady_state(sK=sK, sH=sH)
        y_t = sim_out.y_t
        consumption_t_vector = (1-sK-sH)*y_t
        return consumption_t_vector
    
    def negative_cons(self, x):
        """
        Returns: negative value of consumption from last period in simulation.
        """

        sim_result = self.cons_t(sK=x[0], sH=x[1])
        ct = -1 * sim_result[-1]
        return ct

    def find_opt_s(self, discrete=True, discrete_sqrt_iter=50):
        """
        Returns: optimal savings rate for human and non human capital and consumption in period T
            
        Args: 
        discrete: bool, if True: returns discrete solution from a grid search.
                        if False: returns solution using scipy optimize. 
        discrete_sqrt_iter: square root of iterations in grid search for discrete solution.

        """
        sol_save = self.sol_save = SimpleNamespace() #initialize namespace for solution

        if discrete == True: 
            # a. vector of possible savings rates (physical- and human capital)
            s_K = np.linspace(1e-8, 1, discrete_sqrt_iter)
            s_H = np.linspace(1e-8, 1, discrete_sqrt_iter)  

            # b. loop throug possible combinations
            sk_res = []  # result vectors
            sh_res = []
            cons_res = []            
            for i in s_K:
                for x in s_H:
                    if i + x >= 1: # constraint on saving rates
                        pass
                    else:
                        sk_res += [i]
                        sh_res += [x]
                        cons = self.cons_t(sK=i, sH=x); # calculate consumption from simulation
                        cons_res += [cons[-1]] # extract last period

            # c. extract solution from simulation 
            # i. find optimal solution and index 
            optimal_cons_t = np.max(cons_res) 
            index = cons_res.index(optimal_cons_t)

            # ii. find optimal values for sK and sH
            sk_opt = sk_res[index]
            sh_opt = sk_res[index]

            # d. insert solution values in namespace 
            sol_save.sK_opt = sk_opt 
            sol_save.sH_opt = sh_opt 
            sol_save.cons_T_opt = optimal_cons_t 

            return  sol_save 
        
        else: 
            ("Optimal savings rate continoues solution")
            
            # a. objective function (to minimize)
            def penalty(x):

                # i. unpack
                sK = x[0]
                sH = x[1]
                
                # ii. penalty
                penalty = 0
                S = sK+sH # total savings share 
                if S > 1: # savings share > possible income -> not allowed (loan not possible) 
                    fac = 1/S # fac < 1 if too high expenses
                    penalty += 1000*(S-1) # calculate penalty        
                    sK *= fac # force S = 1
                    sH *= fac # force S = 1
                    
                return self.negative_cons(x) + penalty

            # b. set initial values and solver 
            x0 = [0.2, 0.2] # [sK, sH]
            bnds = ((1e-3,1), (1e-3, 1))

            # c. call solver
            solcont = optimize.minimize(penalty, 
                                        x0=x0, 
                                        bounds=bnds, 
                                        method='Nelder-Mead') # call optimizer

            # d. insert solutions in namespace 
            sol_save.sK_opt = solcont.x[0]
            sol_save.sH_opt = solcont.x[1]
            sol_save.cons_t = -solcont.fun

            return sol_save
        
    def plotbaseline_vs_new_sh(self, new_sH):
        """
        Returns: Plot comparing baseline with the post shock (change in sH)
        
        Args: New sH value not larger than 1 or smaller than 0.
        """

        par = self.par

        # a. genereate baseline results 
        # i. set starting values to baseline
        par.A_init = 1
        par.K_init = 1
        par.H_init = 1
        par.L_init = 1

        # ii. find steady state 
        baseline_result = self.find_steady_state(sK=0.2, sH=0.15)
        
        # iii. unpack solution for baseline 
        ss_t = baseline_result.steadystate_t # period when in ss. 
        baseline_result.y_tilde = baseline_result.y_tilde[ss_t:] # values in steady state
        baseline_result.k_tilde = baseline_result.k_tilde[ss_t:]
        baseline_result.h_tilde = baseline_result.h_tilde[ss_t:]

        # b. generate shock after steady state change 
        # i. set starting values to SS-values from baseline. 
        self.par.A_init = baseline_result.A[ss_t]
        self.par.K_init = baseline_result.K[ss_t]
        self.par.L_init = baseline_result.L[ss_t]
        self.par.H_init = baseline_result.H[ss_t]

        # ii. find steady state with new sH.
        post_shock = self.find_steady_state(sK=0.2, sH=new_sH, do_print=False)

        # iii. extract only the amount of periods as in baseline_result.
        post_shock_periods_index = int(par.simT) - len(baseline_result.k_tilde) 
        post_shock.y_tilde  =  post_shock.y_tilde[:-post_shock_periods_index]
        post_shock.k_tilde  =  post_shock.k_tilde[:-post_shock_periods_index]
        post_shock.h_tilde  =  post_shock.h_tilde[:-post_shock_periods_index]

        # c. plot results from a. and b. 
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=120)

        colors = ['red', 'black', 'orange']  # Set different colors for each variable
        ax.plot(baseline_result.y_tilde, label='y_tilde baseline', linestyle='--', color=colors[0])
        ax.plot(post_shock.y_tilde, label='y_tilde post shock', color=colors[0])
        ax.plot(baseline_result.k_tilde, label='k_tilde baseline', linestyle='--', color=colors[1])
        ax.plot(post_shock.k_tilde, label='k_tilde post shock', color=colors[1])
        ax.plot(baseline_result.h_tilde, label='h_tilde baseline', linestyle='--', color=colors[2])
        ax.plot(post_shock.h_tilde, label='h_tilde post shock', color=colors[2])
        ax.set_title('Fig 5. Effect of a shock to savings', size = 12)

        ax.legend(loc='center left', bbox_to_anchor = (1, 0.5))
        plt.plot()

    def plotbaseline_vs_new_sh_intactive(self):
        out=widgets.interact(self.plotbaseline_vs_new_sh, new_sH=widgets.SelectionSlider(options=np.linspace(0,0.3,31), value=0.12))
        return display(out)