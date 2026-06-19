import numpy as np
from sklearn.cluster import KMeans

class QLDE:
    """
    Q-Learning based Differential Evolution (QLDE) Algorithm
    Optimizes K-means initial centroids to minimize SSE.
    """
    def __init__(self, n_clusters=6, pop_size=30, max_iter=100,
                 F_init=0.5, Cr=0.9, mu=3.9,
                 alpha=0.1, gamma=0.9, epsilon=0.2,
                 random_state=42):
        self.n_clusters   = n_clusters
        self.pop_size     = pop_size
        self.max_iter     = max_iter
        self.F_init       = F_init
        self.Cr           = Cr
        self.mu           = mu
        self.alpha        = alpha
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.random_state = random_state
        
        self.actions = np.array([-0.01, 0.0, 0.01])
        self.n_states = 2
        self.n_actions = len(self.actions)

    def _logistic_chaotic(self, n, d):
        np.random.seed(self.random_state)
        phi = np.random.rand(d)
        pop = np.zeros((n, d))
        for i in range(n):
            phi = self.mu * phi * (1 - phi)
            pop[i] = phi
        return pop

    def _fitness(self, centers, X):
        centers_reshaped = centers.reshape(self.n_clusters, -1)
        dists = np.linalg.norm(X[:, np.newaxis, :] - centers_reshaped[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        sse = np.sum(min_dists ** 2)
        return sse

    def _softmax(self, q_values):
        q_exp = np.exp(q_values - np.max(q_values))
        return q_exp / q_exp.sum()

    def _q_learning_action(self, Q_table, state, k, k_max):
        threshold = 1 - self.epsilon * (k / k_max)
        if np.random.rand() < threshold:
            probs = self._softmax(Q_table[state])
            action_idx = np.random.choice(self.n_actions, p=probs)
        else:
            action_idx = np.random.randint(self.n_actions)
        return action_idx

    def fit(self, X):
        n_samples, n_features = X.shape
        dim = self.n_clusters * n_features
        
        LB = np.tile(X.min(axis=0), self.n_clusters)
        UB = np.tile(X.max(axis=0), self.n_clusters)
        
        pop_phi = self._logistic_chaotic(self.pop_size, dim)
        population = LB + pop_phi * (UB - LB)
        
        F_arr = np.full(self.pop_size, self.F_init)
        Q_table = np.zeros((self.n_states, self.n_actions))
        fitness = np.array([self._fitness(pop, X) for pop in population])
        
        best_idx  = np.argmin(fitness)
        best_sol  = population[best_idx].copy()
        best_fit  = fitness[best_idx]
        
        self.convergence_curve_ = [best_fit]
        
        for k in range(1, self.max_iter + 1):
            sorted_idx = np.argsort(fitness)
            top50_idx  = sorted_idx[:self.pop_size // 2]
            
            for i in range(self.pop_size):
                Lk_idx = np.random.choice(top50_idx)
                candidates = [j for j in range(self.pop_size) if j != i and j != Lk_idx]
                i1, i2    = np.random.choice(candidates, size=2, replace=False)
                
                v = population[Lk_idx] + F_arr[i] * (population[i1] - population[i2])
                
                j_rand = np.random.randint(dim)
                r2     = np.random.rand(dim)
                u      = np.where((r2 <= self.Cr) | (np.arange(dim) == j_rand),
                                  v, population[i])
                
                u = np.clip(u, LB, UB)
                fit_u = self._fitness(u, X)
                fit_z = fitness[i]
                
                if fit_u < fit_z:
                    population[i] = u
                    fitness[i]    = fit_u
                    reward = 1
                    state  = 0
                else:
                    reward = 0
                    state  = 1
                
                action_idx = self._q_learning_action(Q_table, state, k, self.max_iter)
                lambda_i   = self.actions[action_idx]
                
                F_arr[i] = np.clip(F_arr[i] + lambda_i, 0.1, 1.0)
                
                next_state = 0 if reward == 1 else 1
                k_ratio    = k / self.max_iter
                if np.random.rand() < (1 - self.epsilon * k_ratio):
                    exp_Q = np.max(Q_table[next_state])
                else:
                    exp_Q = Q_table[next_state, np.random.randint(self.n_actions)]
                
                Q_table[state, action_idx] += self.alpha * (
                    reward + self.gamma * exp_Q - Q_table[state, action_idx]
                )
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fit:
                best_fit = fitness[best_idx]
                best_sol = population[best_idx].copy()
            
            self.convergence_curve_.append(best_fit)
        
        init_centers = best_sol.reshape(self.n_clusters, n_features)
        km = KMeans(n_clusters=self.n_clusters, init=init_centers,
                    n_init=1, max_iter=300, random_state=self.random_state)
        km.fit(X)
        
        self.labels_          = km.labels_
        self.cluster_centers_ = km.cluster_centers_
        self.inertia_         = km.inertia_
        self.best_qlde_sse_   = best_fit
        self.Q_table_         = Q_table
        self.F_final_         = F_arr
        
        return self


class KMeansDE:
    """
    Differential Evolution (DE) clustering with constant scaling factor F.
    """
    def __init__(self, n_clusters=6, pop_size=30, max_iter=100,
                 F=0.5, Cr=0.9, mu=3.9, random_state=42):
        self.n_clusters   = n_clusters
        self.pop_size     = pop_size
        self.max_iter     = max_iter
        self.F            = F
        self.Cr           = Cr
        self.mu           = mu
        self.random_state = random_state

    def _logistic_chaotic(self, n_pop, dim):
        np.random.seed(self.random_state)
        phi_0  = np.random.rand(dim)
        pop    = np.zeros((n_pop, dim))
        pop[0] = phi_0
        for i in range(1, n_pop):
            phi    = pop[i-1]
            pop[i] = self.mu * phi * (1 - phi)
        return pop

    def _fitness(self, centers, X):
        cr = centers.reshape(self.n_clusters, -1)
        d  = np.linalg.norm(X[:,np.newaxis,:] - cr[np.newaxis,:,:], axis=2)
        lb = np.argmin(d, axis=1)
        sse = 0.0
        for l in range(self.n_clusters):
            mask = lb == l
            if mask.sum() > 0:
                sse += np.sum((X[mask] - X[mask].mean(axis=0))**2)
        return sse

    def fit(self, X):
        n, d   = X.shape
        dim    = self.n_clusters * d
        LB     = np.tile(X.min(axis=0), self.n_clusters)
        UB     = np.tile(X.max(axis=0), self.n_clusters)
        
        pop    = LB + self._logistic_chaotic(self.pop_size, dim) * (UB - LB)
        fit    = np.array([self._fitness(p, X) for p in pop])
        
        bi     = np.argmin(fit)
        best_s = pop[bi].copy()
        best_f = fit[bi]
        self.convergence_curve_ = [best_f]

        for k in range(1, self.max_iter + 1):
            top50 = np.argsort(fit)[:self.pop_size//2]
            for i in range(self.pop_size):
                Lk = np.random.choice(top50)
                cands = [j for j in range(self.pop_size) if j != i and j != Lk]
                i1, i2 = np.random.choice(cands, size=2, replace=False)
                
                v  = pop[Lk] + self.F * (pop[i1] - pop[i2])
                jr = np.random.randint(dim)
                r2 = np.random.rand(dim)
                u  = np.where((r2 <= self.Cr) | (np.arange(dim) == jr), v, pop[i])
                u  = np.clip(u, LB, UB)
                fu = self._fitness(u, X)
                if fu < fit[i]:
                    pop[i], fit[i] = u, fu
            bi = np.argmin(fit)
            if fit[bi] < best_f:
                best_f, best_s = fit[bi], pop[bi].copy()
            self.convergence_curve_.append(best_f)

        init_c = best_s.reshape(self.n_clusters, d)
        km = KMeans(n_clusters=self.n_clusters, init=init_c, n_init=1,
                    max_iter=300, random_state=self.random_state)
        km.fit(X)
        self.labels_          = km.labels_
        self.cluster_centers_ = km.cluster_centers_
        self.inertia_         = km.inertia_
        self.best_de_sse_     = best_f
        return self


class KMeansPSO:
    """
    Particle Swarm Optimization (PSO) clustering.
    """
    def __init__(self, n_clusters=6, pop_size=30, max_iter=100,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, mu=3.9, random_state=42):
        self.n_clusters   = n_clusters
        self.pop_size     = pop_size
        self.max_iter     = max_iter
        self.w_max        = w_max
        self.w_min        = w_min
        self.c1           = c1
        self.c2           = c2
        self.mu           = mu
        self.random_state = random_state

    def _logistic_chaotic(self, n_pop, dim):
        np.random.seed(self.random_state)
        phi_0  = np.random.rand(dim)
        pop    = np.zeros((n_pop, dim))
        pop[0] = phi_0
        for i in range(1, n_pop):
            phi    = pop[i-1]
            pop[i] = self.mu * phi * (1 - phi)
        return pop

    def _fitness(self, centers, X):
        cr = centers.reshape(self.n_clusters, -1)
        d  = np.linalg.norm(X[:,np.newaxis,:] - cr[np.newaxis,:,:], axis=2)
        lb = np.argmin(d, axis=1)
        sse = 0.0
        for l in range(self.n_clusters):
            mask = lb == l
            if mask.sum() > 0:
                sse += np.sum((X[mask] - X[mask].mean(axis=0))**2)
        return sse

    def fit(self, X):
        n, d   = X.shape
        dim    = self.n_clusters * d
        LB     = np.tile(X.min(axis=0), self.n_clusters)
        UB     = np.tile(X.max(axis=0), self.n_clusters)
        
        pop    = LB + self._logistic_chaotic(self.pop_size, dim) * (UB - LB)
        vel    = np.zeros_like(pop)
        
        pbest_pos = pop.copy()
        pbest_fit = np.array([self._fitness(p, X) for p in pop])
        
        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]
        
        self.convergence_curve_ = [gbest_fit]

        for k in range(1, self.max_iter + 1):
            w = self.w_max - k * ((self.w_max - self.w_min) / self.max_iter)
            
            for i in range(self.pop_size):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                
                vel[i] = (w * vel[i] + 
                          self.c1 * r1 * (pbest_pos[i] - pop[i]) + 
                          self.c2 * r2 * (gbest_pos - pop[i]))
                
                pop[i] = pop[i] + vel[i]
                pop[i] = np.clip(pop[i], LB, UB)
                
                fit_i = self._fitness(pop[i], X)
                
                if fit_i < pbest_fit[i]:
                    pbest_fit[i] = fit_i
                    pbest_pos[i] = pop[i].copy()
                    
                    if fit_i < gbest_fit:
                        gbest_fit = fit_i
                        gbest_pos = pop[i].copy()
                        
            self.convergence_curve_.append(gbest_fit)

        init_c = gbest_pos.reshape(self.n_clusters, d)
        km = KMeans(n_clusters=self.n_clusters, init=init_c, n_init=1,
                    max_iter=300, random_state=self.random_state)
        km.fit(X)
        self.labels_          = km.labels_
        self.cluster_centers_ = km.cluster_centers_
        self.inertia_         = km.inertia_
        self.best_pso_sse_    = gbest_fit
        return self


class KMeansEOA:
    """
    Equilibrium Optimization Algorithm (EOA) clustering.
    """
    def __init__(self, n_clusters=6, pop_size=30, max_iter=100,
                 a1=2, a2=1, GP=0.5, mu=3.9, random_state=42):
        self.n_clusters   = n_clusters
        self.pop_size     = pop_size
        self.max_iter     = max_iter
        self.a1           = a1
        self.a2           = a2
        self.GP           = GP
        self.mu           = mu
        self.random_state = random_state

    def _logistic_chaotic(self, n_pop, dim):
        np.random.seed(self.random_state)
        phi_0  = np.random.rand(dim)
        pop    = np.zeros((n_pop, dim))
        pop[0] = phi_0
        for i in range(1, n_pop):
            phi    = pop[i-1]
            pop[i] = self.mu * phi * (1 - phi)
        return pop

    def _fitness(self, centers, X):
        cr = centers.reshape(self.n_clusters, -1)
        d  = np.linalg.norm(X[:,np.newaxis,:] - cr[np.newaxis,:,:], axis=2)
        lb = np.argmin(d, axis=1)
        sse = 0.0
        for l in range(self.n_clusters):
            mask = lb == l
            if mask.sum() > 0:
                sse += np.sum((X[mask] - X[mask].mean(axis=0))**2)
        return sse

    def fit(self, X):
        n, d   = X.shape
        dim    = self.n_clusters * d
        LB     = np.tile(X.min(axis=0), self.n_clusters)
        UB     = np.tile(X.max(axis=0), self.n_clusters)
        
        pop = LB + self._logistic_chaotic(self.pop_size, dim) * (UB - LB)
        fit = np.array([self._fitness(p, X) for p in pop])
        
        Ceq1, Ceq2, Ceq3, Ceq4 = np.zeros(dim), np.zeros(dim), np.zeros(dim), np.zeros(dim)
        fit1, fit2, fit3, fit4 = float('inf'), float('inf'), float('inf'), float('inf')
        
        for i in range(self.pop_size):
            if fit[i] < fit1:
                fit4, Ceq4 = fit3, Ceq3.copy()
                fit3, Ceq3 = fit2, Ceq2.copy()
                fit2, Ceq2 = fit1, Ceq1.copy()
                fit1, Ceq1 = fit[i], pop[i].copy()
            elif fit[i] < fit2:
                fit4, Ceq4 = fit3, Ceq3.copy()
                fit3, Ceq3 = fit2, Ceq2.copy()
                fit2, Ceq2 = fit[i], pop[i].copy()
            elif fit[i] < fit3:
                fit4, Ceq4 = fit3, Ceq3.copy()
                fit3, Ceq3 = fit[i], pop[i].copy()
            elif fit[i] < fit4:
                fit4, Ceq4 = fit[i], pop[i].copy()
                
        Ceq_ave = (Ceq1 + Ceq2 + Ceq3 + Ceq4) / 4
        Ceq_pool = [Ceq1, Ceq2, Ceq3, Ceq4, Ceq_ave]
        
        self.convergence_curve_ = [fit1]
        
        for k in range(1, self.max_iter + 1):
            t = (1 - k/self.max_iter)**(self.a2 * k/self.max_iter)
            
            for i in range(self.pop_size):
                Ceq = Ceq_pool[np.random.randint(0, 5)]
                
                lam = np.random.rand(dim)
                r   = np.random.rand(dim)
                r1  = np.random.rand()
                r2  = np.random.rand()
                
                F = self.a1 * np.sign(r - 0.5) * (np.exp(-lam * t) - 1)
                GCP = 0.5 * r1
                G0  = GCP * (Ceq - lam * pop[i])
                G   = G0 * F
                
                pop[i] = Ceq + (pop[i] - Ceq) * F + (G / (lam * np.ones(dim))) * (1 - F)
                pop[i] = np.clip(pop[i], LB, UB)
                
                fit_i = self._fitness(pop[i], X)
                fit[i] = fit_i
                
                if fit_i < fit1:
                    fit4, Ceq4 = fit3, Ceq3.copy()
                    fit3, Ceq3 = fit2, Ceq2.copy()
                    fit2, Ceq2 = fit1, Ceq1.copy()
                    fit1, Ceq1 = fit_i, pop[i].copy()
                elif fit_i < fit2:
                    fit4, Ceq4 = fit3, Ceq3.copy()
                    fit3, Ceq3 = fit2, Ceq2.copy()
                    fit2, Ceq2 = fit_i, pop[i].copy()
                elif fit_i < fit3:
                    fit4, Ceq4 = fit3, Ceq3.copy()
                    fit3, Ceq3 = fit_i, pop[i].copy()
                elif fit_i < fit4:
                    fit4, Ceq4 = fit_i, pop[i].copy()
                    
            Ceq_ave = (Ceq1 + Ceq2 + Ceq3 + Ceq4) / 4
            Ceq_pool = [Ceq1, Ceq2, Ceq3, Ceq4, Ceq_ave]
            
            self.convergence_curve_.append(fit1)
            
        init_c = Ceq1.reshape(self.n_clusters, d)
        km = KMeans(n_clusters=self.n_clusters, init=init_c, n_init=1,
                    max_iter=300, random_state=self.random_state)
        km.fit(X)
        self.labels_          = km.labels_
        self.cluster_centers_ = km.cluster_centers_
        self.inertia_         = km.inertia_
        self.best_eoa_sse_    = fit1
        return self
