import numpy as np
import cvxpy as cp
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import math
import warnings
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Suppress CVXPY UserWarning about solving a problem with a non-DCP objective
warnings.filterwarnings("ignore", category=UserWarning)

class LinearBanditEnvironment:
    """
    A simple simulation environment for a stochastic linear bandit problem.
    Rewards are generated as Bernoulli random variables.
    """
    def __init__(self, X, theta_star):
        """
        Initializes the environment.

        Args:
            X (np.ndarray): The set of available arms, shape (num_arms, d).
            theta_star (np.ndarray): The true, unknown parameter vector, shape (d,).
        """
        self.X = X
        self.theta_star = theta_star
        self.num_arms, self.d = X.shape
        # Ensure mean rewards are in [0, 1] for Bernoulli rewards
        # print(self.X.shape, self.theta_star.shape)
        self.mean_rewards = self.X @ self.theta_star
        self.optimal_arm_index = np.argmax(self.mean_rewards)
        self.optimal_reward = self.mean_rewards[self.optimal_arm_index]

    def pull_arm(self, arm_index):
        """
        Pulls an arm and returns a stochastic reward.

        Args:
            arm_index (int): The index of the arm to pull.

        Returns:
            float: A reward (0 or 1).
        """
        mean = self.mean_rewards[arm_index]
        return np.random.normal(loc=mean, scale=1) 

class LinNash:
    """
    Implementation of the LINNASH algorithm for finite sets of arms.
    """
    def __init__(self, X, T, nu=1.0):
        """
        Initializes the LINNASH algorithm.

        Args:
            X (np.ndarray): The set of available arms, shape (num_arms, d).
            T (int): The total time horizon.
            nu (float): The sub-Poisson parameter of the rewards.
        """
        self.X = X
        self.num_arms, self.d = X.shape
        self.T = T
        self.nu = nu
        self.arm_indices = np.arange(self.num_arms)
        self.log_term = np.log(self.T * self.num_arms)

    def _solve_d_optimal_design(self, arm_set_indices, support_limit=None):
        """
        Solve D-optimal design maximize logdet(U) subject to sum(lambda)=1, lambda>=0.
        If support_limit is provided, we will post-process to keep only top-k entries (practical).
        Returns (lambdas_full, indices_full) where lambdas_full is aligned with arm_set_indices.
        """
        n_active = len(arm_set_indices)
        if n_active == 0:
            return None, None

        active_arms = self.X[arm_set_indices]  # shape (n_active, d)
        lambda_vars = cp.Variable(n_active, nonneg=True)
        # Build U = sum_i lambda_i x_i x_i^T
        # CVXPY prefers expression building as sum of small matrices
        U = sum([lambda_vars[i] * np.outer(active_arms[i], active_arms[i]) for i in range(n_active)])
        constraints = [cp.sum(lambda_vars) == 1]
        objective = cp.Maximize(cp.log_det(U + 1e-9 * np.eye(self.d)))  # small regularizer for stability

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.SCS, verbose=False)  # SCS is robust; change if you prefer
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                lambdas = np.array(lambda_vars.value).flatten()
                lambdas[lambdas < 1e-9] = 0.0
                if lambdas.sum() <= 0:
                    return None, None
                # Optionally enforce support size constraint by keeping top-k masses
                if support_limit is not None and support_limit < n_active:
                    k = min(n_active, support_limit)
                    sorted_idx = np.argsort(-lambdas)
                    keep = sorted_idx[:k]
                    new_lambdas = np.zeros_like(lambdas)
                    new_lambdas[keep] = lambdas[keep]
                    if new_lambdas.sum() == 0:
                        return None, None
                    new_lambdas /= new_lambdas.sum()
                    lambdas = new_lambdas
                else:
                    lambdas /= lambdas.sum()
                return lambdas, arm_set_indices
            else:
                return None, None
        except Exception:
            return None, None
        
    def _get_john_ellipsoid_dist(self, active_indices):
        """
        Approximate distribution U described in paper (Section 3.2) using the Chebyshev center
        of the convex hull of the active arms. If geometry fails, fallback to uniform over active_indices.
        """
        try:
            X_active = self.X[active_indices]
            if len(X_active) == 0:
                return None, None
            # If too few points or not full-dim, fallback to uniform
            if len(X_active) <= self.d:
                alphas = np.ones(len(active_indices)) / len(active_indices)
                return alphas, active_indices

            hull = ConvexHull(X_active)
            hull_vertices = X_active[hull.vertices]
            A = hull.equations[:, :-1]
            b = -hull.equations[:, -1]

            # Chebyshev center LP: maximize r s.t. A c + r ||a_i|| <= b_i
            c_obj = np.zeros(self.d + 1)
            c_obj[-1] = -1  # minimize -r
            norms = np.linalg.norm(A, axis=1, keepdims=True)
            A_ub = np.hstack([A, norms])
            b_ub = b

            res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub)
            if not res.success:
                # fallback uniform
                alphas = np.ones(len(active_indices)) / len(active_indices)
                return alphas, active_indices
            center = res.x[:-1]

            # express center as convex comb. of hull vertices
            Y = hull_vertices.T  # d x m
            c_feasibility = np.zeros(len(hull.vertices))
            A_eq = np.vstack([Y, np.ones(len(hull.vertices))])
            b_eq = np.hstack([center, 1])
            res_alpha = linprog(c=c_feasibility, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
            if not res_alpha.success:
                alphas = np.ones(len(active_indices)) / len(active_indices)
                return alphas, active_indices
            alphas = res_alpha.x
            alphas[alphas < 1e-9] = 0
            alphas /= max(1e-12, alphas.sum())

            # Need to map hull.vertices (indices wrt X_active) to global indices
            global_indices = np.array(active_indices)[hull.vertices]
            return alphas, global_indices
        except Exception:
            # fallback uniform
            alphas = np.ones(len(active_indices)) / len(active_indices)
            return alphas, active_indices


    def _generate_arm_sequence(self, tilde_T):
        """
        Implements Algorithm 1 to generate the initial arm sequence for Part I.
        """
        # 1. D-optimal design distribution

        support_limit = min(self.num_arms, self.d * (self.d + 1) // 2)

        arms = []

        lambdas0, lam_indices = self._solve_d_optimal_design(self.arm_indices, support_limit=support_limit)
        if lambdas0 is None:
            # fallback to uniform over all arms
            lam_indices = self.arm_indices
            lambdas0 = np.ones(len(lam_indices)) / len(lam_indices)


        supp_mask = lambdas0 > 1e-9
        A = np.array(lam_indices)[supp_mask]
        lambda_on_A = np.array(lambdas0)[supp_mask]
        if len(A) == 0:
            # fallback: pick top-1 arm
            A = np.array([int(lam_indices[np.argmax(lambdas0)])])
            lambda_on_A = np.array([1.0])


        U_alphas, U_indices = self._get_john_ellipsoid_dist(self.arm_indices)
        if U_alphas is None:
            U_indices = self.arm_indices
            U_alphas = np.ones(len(U_indices)) / len(U_indices)


        tilde_T = int(tilde_T)

        # --- build lambda map aligned to global arm indices in A ---
        if isinstance(lambda_on_A, dict):
            lam_map = dict(lambda_on_A)
        else:
            lam_arr = np.array(lambda_on_A, dtype=float)
            if lam_arr.ndim == 0:  # scalar
                lam_map = {int(a): float(lam_arr) for a in A}
            elif len(lam_arr) == len(A):
                lam_map = {int(a): float(lam_arr[i]) for i, a in enumerate(A)}
            else:
                # fallback: uniform over A
                lam_map = {int(a): 1.0 / max(1, len(A)) for a in A}


        alphas, u_indices = U_alphas, U_indices
        alphas = np.array(alphas, dtype=float)
        u_indices = np.array(u_indices, dtype=int)
        if alphas.sum() <= 0:
            alphas = np.ones_like(alphas) / len(alphas)
        else:
            alphas = alphas / alphas.sum()

        rr_order = list(A)            # dynamic RR list
        counts = {int(a): 0 for a in A}
        rr_ptr = 0


        for _ in range(tilde_T):
            # stop early if horizon reached

            # coin flip
            if random.random() < 0.5 or len(rr_order) == 0:
                # SAMPLE from U
                arm = int(np.random.choice(u_indices, p=alphas))
            else:
                # D: pick next in round-robin
                arm = int(rr_order[rr_ptr % len(rr_order)])
                rr_ptr = (rr_ptr + 1) % max(1, len(rr_order))
                counts[arm] = counts.get(arm, 0) + 1

                # threshold check: ceil(lambda_arm * T_tilde / 3)
                lambda_arm = lam_map.get(arm, 0.0)
                thresh = math.ceil(lambda_arm * float(tilde_T) / 3.0)
                if counts[arm] >= thresh:
                    # remove from rr_order if present
                    try:
                        pos = rr_order.index(arm)
                        rr_order.pop(pos)
                        # fix pointer to remain valid
                        if len(rr_order) == 0:
                            rr_ptr = 0
                        else:
                            rr_ptr = rr_ptr % len(rr_order)
                    except ValueError:
                        pass

            # pull, update stats
            arms.append(arm)

        return arms

    def _LCB(self, x_vec, phi, t):
        inner = float(x_vec @ phi)
        if inner <= 0:
            return -np.inf
        width = 6.0 * math.sqrt((inner * self.nu * self.d * self.log_term) / max(1, t))
        return inner - width

    def _UCB(self, x_vec, phi, t):
        inner = float(x_vec @ phi)
        if inner <= 0:
            return np.inf
        width = 6.0 * math.sqrt((inner * self.nu * self.d * self.log_term) / max(1, t))
        return inner + width


    def run(self, environment):
        """
        Executes the full LINNASH algorithm.

        Returns:
            list: A list of arm indices pulled in each round.
        """
        log_term = np.log(self.T * self.num_arms)
        history = []
        total_rounds_played = 0
        V = np.zeros((self.d, self.d))
        sum_rX = np.zeros(self.d)          # maintain cumulative sum_rX across phases
        tilde_T = int(3 * np.sqrt(self.T * self.d * self.nu * log_term))

        # --- Part I ---


        arms = self._generate_arm_sequence(tilde_T)        
        
        for t in range(tilde_T):
            arm_idx = arms[t]
            arm_vec = self.X[arm_idx]
            reward = environment.pull_arm(arm_idx)
            
            V += np.outer(arm_vec, arm_vec)
            sum_rX += reward * arm_vec
            history.append(arm_idx)
        
        total_rounds_played += tilde_T
        
        t_prime_part1 = tilde_T / 3

        theta_hat = np.linalg.inv(V) @ sum_rX
        est_rewards = self.X @ theta_hat
        lcb_vals = np.array([self._LCB(self.X[i], theta_hat, t_prime_part1) for i in range(self.num_arms)])
        ucb_vals = np.array([self._UCB(self.X[i], theta_hat, t_prime_part1) for i in range(self.num_arms)])
        max_lcb = np.max(lcb_vals)
        X_tilde_indices = np.where(ucb_vals >= max_lcb)[0]

        T_prime = max(1, int(round((2.0 / 3.0) * tilde_T)))


        # print(total_rounds_played, len(history))
        while total_rounds_played < self.T:
            if len(X_tilde_indices) == 0:
                # nothing left: pull best estimated arm for remaining budget
                best = int(np.argmax(est_rewards))
                remaining = self.T - total_rounds_played
                for _ in range(remaining):
                    r = environment.pull_arm(best)
                    history.append(best)
                    total_rounds_played += 1
                    V += np.outer(self.X[best], self.X[best])
                    sum_rX += r * self.X[best]
                break

            # print(len(ghiuhisto)
            # Solve D-opt on X_tilde_indices
            lambdas_phase, indices_phase = self._solve_d_optimal_design(
                X_tilde_indices,
                support_limit=min(len(X_tilde_indices), self.d*(self.d+1)//2)
            )
            # print(total_rounds_played)
            if lambdas_phase is None:
                # fallback: uniformly sample active arms for remaining rounds
                while total_rounds_played < self.T:
                    chosen = int(np.random.choice(X_tilde_indices))
                    r = environment.pull_arm(chosen)
                    history.append(chosen)
                    total_rounds_played += 1
                    V += np.outer(self.X[chosen], self.X[chosen])
                    sum_rX += r * self.X[chosen]
                break
            # print(t)
            # ensure normalization and convert indices_phase to array of ints
            lambdas_phase = np.array(lambdas_phase, dtype=float)
            if lambdas_phase.sum() <= 0:
                lambdas_phase = np.ones_like(lambdas_phase) / len(lambdas_phase)
            else:
                lambdas_phase = lambdas_phase / lambdas_phase.sum()
            indices_phase = np.array(indices_phase, dtype=int)

            # For each arm in support, pull ceil(lambda_i * T') times (capped by remaining budget)
            remaining_budget = self.T - total_rounds_played
            # compute planned pulls per support index, but we will cap as we go
            planned_pulls = [math.ceil(lambdas_phase[idx] * T_prime) for idx in range(len(indices_phase))]

            for idx_local, arm_global in enumerate(indices_phase):
                # print(total_rounds_played)
                if remaining_budget <= 0:
                    break
                pulls = min(planned_pulls[idx_local], remaining_budget)
                if pulls <= 0:
                    continue
                xvec = self.X[arm_global]
                for _ in range(pulls):
                    r = environment.pull_arm(arm_global)
                    history.append(int(arm_global))
                    total_rounds_played += 1
                    remaining_budget -= 1
                    V += np.outer(xvec, xvec)
                    sum_rX += r * xvec
                    if total_rounds_played >= self.T:
                        break

            # Re-estimate theta_hat using cumulative V and sum_rX
            V_inv = np.linalg.pinv(V + 1e-9 * np.eye(self.d))
            theta_hat = V_inv @ sum_rX
            est_rewards = self.X @ theta_hat

            # recompute LCB/UCB and shrink X_tilde
            lcb_vals = np.array([self._LCB(self.X[i], theta_hat, T_prime) for i in range(self.num_arms)])
            ucb_vals = np.array([self._UCB(self.X[i], theta_hat, T_prime) for i in range(self.num_arms)])
            max_lcb = np.max(lcb_vals)
            X_tilde_indices = np.where(ucb_vals >= max_lcb)[0]

            # double T_prime for next phase (but not beyond remaining budget if you prefer)
            T_prime = min(self.T, int(2 * T_prime))

        # print(t)
        # done Phase II; return history (plus optional stats)
        return history
    

def simulate_linnash(env, X, T, num_trials=10, sigma2=1.0, regret_type="Nash"):
    mu_star = np.max(env.mean_rewards)
    total_rewards = []
    try:
        for _ in tqdm(range(num_trials), desc="LINNASH Trials"):
            algo = LinNash(X, T, nu=2)
            history = algo.run(env)
            rewards = env.mean_rewards[history]
            total_rewards.append(rewards)
    except KeyboardInterrupt:
        print("Interrupted by user.")


    total_rewards = np.array(total_rewards)
    expected_means = np.mean(total_rewards, axis=0)

    if regret_type == "Nash":
        cumsum_log = np.cumsum(np.log(expected_means))
        inv_t = 1.0 / np.arange(1, T+1)
        geom_mean = np.exp(cumsum_log * inv_t)
        avg_regret = mu_star - geom_mean
        return avg_regret
    else:
        cum_rewards = np.cumsum(expected_means)
        inv_t = 1.0 / np.arange(1, T+1)
        arith_mean = cum_rewards * inv_t
        avg_regret = mu_star - arith_mean
        return avg_regret



if __name__ == '__main__':
    # Small sanity run (reduce T for quick run)
    d = 10
    num_arms = 50
    T = 1000000

    np.random.seed(42)
    theta_star = np.ones(d)
    # theta_star /= np.linalg.norm(theta_star)
    X = np.random.randn(num_arms, d)

    # Ensure acute angle: project into positive half-space
    for i in range(num_arms):
        if np.dot(X[i], theta_star) <= 0:
            X[i] *= -1   #
    # for i in range(d):

    # X /= np.linalg.norm(X, axis=1, keepdims=True)

    env = LinearBanditEnvironment(X, theta_star)
    print("Running LINNASH (modified) ...")
    regret_curve = simulate_linnash(env, X, T, num_trials=50, sigma2=1.0, regret_type="Nash")
    np.save("hhb.npy", regret_curve)
    plt.plot(regret_curve)
    plt.xlabel("Rounds")
    plt.ylabel("Nash regret")
    plt.show()
