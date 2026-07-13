import numpy as np

class WOA:
    def __init__(self, fitness_fn, n_vars, lb, ub, pop_size=6, max_iter=10):
        self.fitness_fn = fitness_fn
        self.n_vars     = n_vars
        self.lb         = np.array(lb, dtype=float)
        self.ub         = np.array(ub, dtype=float)
        self.pop_size   = pop_size
        self.max_iter   = max_iter

    def run(self):
        pop     = self.lb + np.random.rand(self.pop_size, self.n_vars) * (self.ub - self.lb)
        fitness = np.array([self.fitness_fn(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx].copy()
        best_fit = fitness[best_idx]
        print(f"\n[WOA] Initial best fitness: {best_fit:.2f}")

        for t in range(self.max_iter):
            a  = 2.0 - t * (2.0 / self.max_iter)
            a2 = -1.0 - t * (1.0 / self.max_iter)
            for i in range(self.pop_size):
                r1 = np.random.rand(self.n_vars)
                r2 = np.random.rand(self.n_vars)
                A  = 2.0 * a * r1 - a
                C  = 2.0 * r2
                p  = np.random.rand()
                if p < 0.5:
                    if np.linalg.norm(A) < 1:
                        D      = np.abs(C * best_pos - pop[i])
                        pop[i] = best_pos - A * D
                    else:
                        rand_idx = np.random.randint(0, self.pop_size)
                        D        = np.abs(C * pop[rand_idx] - pop[i])
                        pop[i]   = pop[rand_idx] - A * D
                else:
                    D_star = np.abs(best_pos - pop[i])
                    l      = (a2 - 1.0) * np.random.rand(self.n_vars) + 1.0
                    pop[i] = D_star * np.exp(1.0 * l) * np.cos(2 * np.pi * l) + best_pos
                pop[i] = np.clip(pop[i], self.lb, self.ub)

            fitness  = np.array([self.fitness_fn(ind) for ind in pop])
            idx      = np.argmin(fitness)
            if fitness[idx] < best_fit:
                best_fit = fitness[idx]
                best_pos = pop[idx].copy()

        print(f"[WOA] Optimization complete!")
        print(f"[WOA] Best fitness: {best_fit:.2f}")
        return best_pos, best_fit