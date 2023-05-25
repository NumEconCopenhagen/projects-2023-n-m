import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import sympy as sm
import numpy as np
import matplotlib.pyplot as plt

class UtilityPlotter:
    def __init__(self, kappa, alpha, nu, tau, w):
        self.kappa = kappa
        self.alpha = alpha
        self.nu = nu
        self.tau = tau
        self.w = w

    def _calculate_optimal_L(self, w_tilde_values):
        # Defining the parameter symbols
        L, w_tilde, kappa, alpha, nu, C, G, tau = sm.symbols('L w_tilde kappa alpha nu C G tau')

        # Defining utility & consumption constraint (we have already substituted the expression for w_tilde in the consumption constraint)
        utility = sm.log(C**alpha * G**(1-alpha)) - (nu * (L**2 / 2))
        consumption_constraint = sm.Eq(C, kappa + w_tilde * L)

        consumption_constraint_solved = sm.solve(consumption_constraint, C)

        # Inserting expression for C in the utility function
        utility_subs = utility.subs(C, consumption_constraint_solved[0])

        # Different values of G to test
        G_values = [1.0, 2.0]

        # Calculate the optimal L for each G value using list comprehension
        optimal_L = [sm.solve(sm.Eq(sm.diff(utility_subs.subs(G, G_val), L), 0), L)[0] for G_val in G_values]

        # saving the function
        optimal_L_func = sm.lambdify(args=(kappa, nu, alpha, w_tilde), expr=optimal_L[0])

        L_values = optimal_L_func(self.kappa, self.nu, self.alpha, w_tilde_values)  # Replace with actual parameter values
        return L_values

    def plot_utility(self):
        w_tilde_values = np.linspace(0.01, 2, 100)
        L_values = self._calculate_optimal_L(w_tilde_values)

        # Plot the graph
        plt.plot(w_tilde_values, L_values)
        plt.xlabel('w_tilde')
        plt.ylabel('L^star')
        plt.title('Plot of L^star(w_tilde)')
        plt.grid(True)
        plt.show()

class RefinedGlobalOptimizer:
    def __init__(self, warm_up_iterations):
        self.warm_up_iterations = warm_up_iterations
        self.bounds = [-600, 600]
        self.tau = 1e-8
        self.max_iterations = 1000
        self.x_opt = np.nan
        self.f_opt = np.inf
        self.x_k0_values = []
        self.iteration_numbers = []

    def griewank(self, x):
        return self.griewank_(x[0], x[1])

    def griewank_(self, x1, x2):
        A = x1**2 / 4000 + x2**2 / 4000
        B = np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))
        return A - B + 1

    def optimize(self):
        for k in range(self.max_iterations):
            x_k = np.random.uniform(self.bounds[0], self.bounds[1], size=2)

            if k >= self.warm_up_iterations:
                chi_k = 0.50 * 2 / (1 + np.exp((k - self.warm_up_iterations) / 100))
                x_k0 = chi_k * x_k + (1 - chi_k) * self.x_opt
            else:
                x_k0 = x_k

            result = optimize.minimize(self.griewank, x_k0, method='BFGS', tol=self.tau)
            x_k_opt = result.x
            f_k_opt = result.fun

            if k == 0 or f_k_opt < self.f_opt:
                self.x_opt = x_k_opt
                self.f_opt = f_k_opt

            self.x_k0_values.append(x_k0)
            self.iteration_numbers.append(k+1)

            if k < 10 or self.f_opt < self.tau:
                print(f'{k:4d}: x^k = ({x_k[0]:7.2f}, {x_k[1]:7.2f}) -> x^k0 = ({x_k0[0]:7.2f}, {x_k0[1]:7.2f})')

            if self.f_opt < self.tau:
                break

    def print_convergence_iteration(self):
        print(f'Converged at iteration: {len(self.iteration_numbers)}')

    def plot_effective_initial_guesses(self):
        x_k0_values = np.array(self.x_k0_values)
        iteration_numbers = np.array(self.iteration_numbers)

        fig, ax = plt.subplots()
        plt.scatter(x_k0_values[:, 0], x_k0_values[:, 1], c=iteration_numbers, cmap='viridis')
        plt.colorbar(label='Iteration Number')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Effective Initial Guesses x^k0 vs. Iteration Number')
        plt.show()

