import numpy as np


class GA:
    """
    Genetic Algorithm for traffic signal optimization.
    Reactive only — no LSTM prediction.
    Uses current traffic state via fitness function.
    """

    def __init__(self, fitness_fn, n_vars, lb, ub,
                 pop_size=6, max_iter=10,
                 crossover_rate=0.8, mutation_rate=0.2):
        self.fitness_fn    = fitness_fn
        self.n_vars        = n_vars
        self.lb            = np.array(lb, dtype=float)
        self.ub            = np.array(ub, dtype=float)
        self.pop_size      = pop_size
        self.max_iter      = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate  = mutation_rate

    def run(self):
        # Initialize population
        pop = self.lb + np.random.rand(self.pop_size, self.n_vars) * (
              self.ub - self.lb)

        fitness  = np.array([self.fitness_fn(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        print(f"\n[GA] Initial best fitness: {best_fit:.2f}")

        for t in range(self.max_iter):
            new_pop = []

            for i in range(self.pop_size):
                # ── Tournament selection
                a, b   = np.random.choice(self.pop_size, 2, replace=False)
                parent1 = pop[a] if fitness[a] < fitness[b] else pop[b]
                c, d   = np.random.choice(self.pop_size, 2, replace=False)
                parent2 = pop[c] if fitness[c] < fitness[d] else pop[d]

                # ── Crossover
                if np.random.rand() < self.crossover_rate:
                    alpha  = np.random.rand(self.n_vars)
                    child  = alpha * parent1 + (1 - alpha) * parent2
                else:
                    child  = parent1.copy()

                # ── Mutation
                for j in range(self.n_vars):
                    if np.random.rand() < self.mutation_rate:
                        child[j] = self.lb[j] + np.random.rand() * (
                                   self.ub[j] - self.lb[j])

                child = np.clip(child, self.lb, self.ub)
                new_pop.append(child)

            pop     = np.array(new_pop)
            fitness = np.array([self.fitness_fn(ind) for ind in pop])
            idx     = np.argmin(fitness)

            if fitness[idx] < best_fit:
                best_fit = fitness[idx]
                best_pos = pop[idx].copy()

        print(f"[GA] Optimization complete!")
        print(f"[GA] Best fitness: {best_fit:.2f}")
        return best_pos, best_fit