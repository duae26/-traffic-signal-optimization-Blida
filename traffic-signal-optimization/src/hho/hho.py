import numpy as np
import math


class HHO:
    def __init__(self, fitness_fn, n_vars, lb, ub, pop_size=6, max_iter=10):
        self.fitness_fn = fitness_fn
        self.n_vars     = n_vars
        self.lb         = np.array(lb, dtype=float)
        self.ub         = np.array(ub, dtype=float)
        self.pop_size   = pop_size
        self.max_iter   = max_iter

    def _levy(self, n):
        beta  = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta *
                  2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(n) * sigma
        v = np.random.randn(n)
        return u / (np.abs(v) ** (1 / beta))

    def run(self):
        # Initialize uniformly across full range
        pop = self.lb + np.random.rand(self.pop_size, self.n_vars) * (
              self.ub - self.lb)

        fitness  = np.array([self.fitness_fn(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        rabbit   = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        print(f"\n[HHO] Initial best fitness: {best_fit:.2f}")

        for t in range(self.max_iter):
            E1 = 2.0 * (1.0 - t / self.max_iter)

            for i in range(self.pop_size):
                E0 = 2.0 * np.random.rand() - 1.0
                E  = E1 * E0

                if abs(E) >= 1:
                    # Exploration
                    q = np.random.rand()
                    if q >= 0.5:
                        rand_idx = np.random.randint(0, self.pop_size)
                        X_rand   = pop[rand_idx].copy()
                        pop[i]   = X_rand - np.random.rand(self.n_vars) * np.abs(
                                   X_rand - 2.0 * np.random.rand(self.n_vars) * pop[i])
                    else:
                        X_mean = np.mean(pop, axis=0)
                        # Random position in search space
                        X_rand = self.lb + np.random.rand(self.n_vars) * (self.ub - self.lb)
                        pop[i] = X_rand - np.random.rand(self.n_vars) * np.abs(
                                 X_rand - pop[i])

                else:
                    r = np.random.rand()
                    J = 2.0 * (1.0 - np.random.rand(self.n_vars))

                    if r >= 0.5 and abs(E) >= 0.5:
                        # Soft besiege — can move away from rabbit
                        D = rabbit - pop[i]
                        pop[i] = rabbit - E * np.abs(J * rabbit - pop[i])

                    elif r >= 0.5 and abs(E) < 0.5:
                        # Hard besiege — use signed difference not absolute
                        D = rabbit - pop[i]
                        pop[i] = rabbit - E * D

                    elif r < 0.5 and abs(E) >= 0.5:
                        # Soft besiege + Levy
                        LF = self._levy(self.n_vars)
                        X1 = np.clip(rabbit - E * np.abs(J * rabbit - pop[i]),
                                     self.lb, self.ub)
                        if self.fitness_fn(X1) < fitness[i]:
                            pop[i] = X1
                        else:
                            X2 = np.clip(X1 + np.random.rand(self.n_vars) * LF,
                                         self.lb, self.ub)
                            if self.fitness_fn(X2) < fitness[i]:
                                pop[i] = X2

                    else:
                        # Hard besiege + Levy — use signed difference
                        LF     = self._levy(self.n_vars)
                        X_mean = np.mean(pop, axis=0)
                        D      = rabbit - X_mean
                        X1     = np.clip(rabbit - E * D,
                                         self.lb, self.ub)
                        if self.fitness_fn(X1) < fitness[i]:
                            pop[i] = X1
                        else:
                            X2 = np.clip(X1 + np.random.rand(self.n_vars) * LF,
                                         self.lb, self.ub)
                            if self.fitness_fn(X2) < fitness[i]:
                                pop[i] = X2

                pop[i] = np.clip(pop[i], self.lb, self.ub)

            fitness  = np.array([self.fitness_fn(ind) for ind in pop])
            idx      = np.argmin(fitness)
            if fitness[idx] < best_fit:
                best_fit = fitness[idx]
                rabbit   = pop[idx].copy()

        print(f"[HHO] Optimization complete!")
        print(f"[HHO] Best fitness: {best_fit:.2f}")
        return rabbit, best_fit