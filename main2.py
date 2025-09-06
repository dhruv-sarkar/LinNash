import numpy as np
import cvxpy as cp
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import math
import warnings
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

class LinearBanditEnvironment:
    def __init__(self, X, theta_star):
        self.X = X
        self.theta_star = theta_star
        self.num_arms, self.d = X.shape
        # self.mean_rewards = np.clip(self.X @ self.theta_star, 0, 1)
        self.mean_rewards = self.X @ self.theta_star
        self.optimal_arm_index = np.argmax(self.mean_rewards)
        self.optimal_reward = self.mean_rewards[self.optimal_arm_index]

    def pull_arm(self, arm_index):
        mean = self.mean_rewards[arm_index]
        return np.random.normal(loc=mean, scale=1)  # Gaussian noise with stddev 0.1

class LinNash:
    def __init__(self, X, T, sigma2=1.0):
        """
        X: (num_arms, d) matrix of arm feature vectors (row-per-arm)
        T: horizon
        sigma2: variance parameter (maps to nu in your earlier code)
        """
        self.X = X
        self.num_arms, self.d = X.shape
        self.T = T
        self.sigma2 = sigma2
        self.arm_indices = np.arange(self.num_arms)
        self.log_term = math.log(max(2, self.T * self.num_arms))

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
        
    def pull_arms(self, A, U, lam, V, T_tilde, env, sum_rX=None, history=None, total_rounds=0):
        """
        Minimal implementation of Algorithm 1 (PullArms).

        Args:
            A: list/array of active arm indices (global indices)
            U: tuple (alphas, indices) describing distribution U (alphas sums to 1)
            lam: mapping or array of lambda masses aligned with A.
                - if dict: lam[arm] -> lambda mass
                - if array-like of same length as A: elementwise mapping
                - if scalar: treated uniformly
            V: current covariance matrix (d x d) updated in-place
            T_tilde: number of pulls to perform in this call (int)
            env: environment with env.pull_arm(arm_index) -> scalar reward
            sum_rX: (optional) cumulative sum_rX; if None, initialized to zero
            history: (optional) list to append pulled arm indices to; if None, new list created
            total_rounds: (optional) starting total rounds (int)

        Returns:
            theta_hat, V, sum_rX, history, total_rounds
        """
        # --- init ---
        A = list(A)
        # if history is None:
        #     history = []
        # if sum_rX is None:
        #     sum_rX = np.zeros(self.d)

        T_tilde = int(T_tilde)

        # --- build lambda map aligned to global arm indices in A ---
        if isinstance(lam, dict):
            lam_map = dict(lam)
        else:
            lam_arr = np.array(lam, dtype=float)
            if lam_arr.ndim == 0:  # scalar
                lam_map = {int(a): float(lam_arr) for a in A}
            elif len(lam_arr) == len(A):
                lam_map = {int(a): float(lam_arr[i]) for i, a in enumerate(A)}
            else:
                # fallback: uniform over A
                lam_map = {int(a): 1.0 / max(1, len(A)) for a in A}

        # --- prepare U as (alphas, indices) ---
        alphas, u_indices = U
        alphas = np.array(alphas, dtype=float)
        u_indices = np.array(u_indices, dtype=int)
        if alphas.sum() <= 0:
            alphas = np.ones_like(alphas) / len(alphas)
        else:
            alphas = alphas / alphas.sum()

        # --- round-robin state & counts ---
        rr_order = list(A)            # dynamic RR list
        counts = {int(a): 0 for a in A}
        rr_ptr = 0

        # --- main loop: perform T_tilde pulls ---
        for _ in range(T_tilde):
            # stop early if horizon reached
            if total_rounds >= self.T:
                break

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
                thresh = math.ceil(lambda_arm * float(T_tilde) / 3.0)
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
            r = env.pull_arm(int(arm))
            x = self.X[int(arm)]
            V += np.outer(x, x)
            sum_rX += float(r) * x
            history.append(int(arm))
            total_rounds += 1

        # estimate theta_hat from cumulative stats (use pseudo-inverse for stability)
        theta_hat = np.linalg.inv(V) @ sum_rX

        return theta_hat, V, sum_rX, history

    def _check_condition(self, t, theta_hat):
        log_term = np.log(self.T * self.num_arms)
        for i in range(self.num_arms):
            x_vec = self.X[i]
            inner = float(x_vec @ theta_hat)
            if inner <= 0:
                continue  # skip invalid arms
            
            lhs = (t * inner) / (3 * self.d)
            
            denom = inner - np.sqrt((6 * self.d * self.sigma2 * log_term) / max(1, t))
            if denom <= 0:
                continue  # condition automatically satisfied (avoid div by 0)
            
            rhs = (100 * self.d *  self.sigma2 * log_term) / denom \
                + np.sqrt((t * self.sigma2 * log_term) / self.d)
            
            # print(i, lhs, rhs)

            if lhs > rhs:
                return False  # condition violated for some x
        return True

    def _LCB(self, x_vec, phi, V_inv):
        xtVx = max(0.0, x_vec @ V_inv @ x_vec)
        width = 2.0 * math.sqrt(2.0 * self.sigma2 * xtVx * self.log_term)
        return (x_vec @ phi) - width

    def _UCB(self, x_vec, phi, V_inv):
        xtVx = max(0.0, x_vec @ V_inv @ x_vec)
        width = 2.0 * math.sqrt(2.0 * self.sigma2 * xtVx * self.log_term)
        return (x_vec @ phi) + width

    def run(self, environment):
        
        # """
        # Main algorithm following Algorithm 2 in the image.
        # Returns: history list of pulled arm indices (length T)
        # """
        history = []
        total_rounds = 0
        V = np.zeros((self.d, self.d))
        sum_rX = np.zeros(self.d)          # maintain cumulative sum_rX across phases
        tilde_T = int(36 * np.log(max(2, self.T)))  # use self.T and ensure positive
        t = 1


        support_limit = min(self.num_arms, self.d * (self.d + 1) // 2)
        lambdas0, lam_indices = self._solve_d_optimal_design(self.arm_indices, support_limit=support_limit)
        if lambdas0 is None:
            # fallback to uniform over all arms
            lam_indices = self.arm_indices
            lambdas0 = np.ones(len(lam_indices)) / len(lam_indices)

        # A = supp(lambda0)
        supp_mask = lambdas0 > 1e-9
        A = np.array(lam_indices)[supp_mask]
        lambda_on_A = np.array(lambdas0)[supp_mask]
        if len(A) == 0:
            # fallback: pick top-1 arm
            A = np.array([int(lam_indices[np.argmax(lambdas0)])])
            lambda_on_A = np.array([1.0])

        theta_hat = np.zeros(self.d)

        # Compute distribution U (using John ellipsoid/chebyshev approach)
        U_alphas, U_indices = self._get_john_ellipsoid_dist(self.arm_indices)
        if U_alphas is None:
            U_indices = self.arm_indices
            U_alphas = np.ones(len(U_indices)) / len(U_indices)

        while (t == 1 or self._check_condition(t, theta_hat)):

            # print(f"yes {t}")
            theta_hat, V, sum_rX, history = self.pull_arms( A.tolist(), (U_alphas, U_indices), lambda_on_A, V, tilde_T, environment, sum_rX, history)
            vals = self.X @ theta_hat   # shape (num_arms,)
            # print("max(x^T theta_hat * tilde_T/3d) =", tilde_T*np.max(vals)/(3*self.d))
            # print("second term: ",np.sqrt(t*self.sigma2*self.log_term/self.d))
            # print("fienf : ", np.max(vals)-np.sqrt((6*self.d*self.sigma2*self.log_term)/max(1,t)))
            t += tilde_T
            tilde_T *= 2

        V_inv = np.linalg.pinv(V + 1e-9 * np.eye(self.d))

        # print("Phase 1 here", t, len(history))
        est_rewards = self.X @ theta_hat
        lcb_vals = np.array([self._LCB(self.X[i], theta_hat, V_inv) for i in range(self.num_arms)])
        ucb_vals = np.array([self._UCB(self.X[i], theta_hat, V_inv) for i in range(self.num_arms)])
        max_lcb = np.max(lcb_vals)
        X_tilde_indices = np.where(ucb_vals >= max_lcb)[0]

        # # Phase II
        T_prime = max(1, int(round((2.0 / 3.0) * tilde_T)))

        # print(total_rounds)

        while t < self.T + 1:
            if len(X_tilde_indices) == 0:
                # nothing left: pull best estimated arm for remaining budget
                best = int(np.argmax(est_rewards))
                remaining = self.T - total_rounds
                for _ in range(remaining):
                    r = environment.pull_arm(best)
                    history.append(best)
                    t += 1
                    V += np.outer(self.X[best], self.X[best])
                    sum_rX += r * self.X[best]
                break

            # print(len(ghiuhisto)
            # Solve D-opt on X_tilde_indices
            lambdas_phase, indices_phase = self._solve_d_optimal_design(
                X_tilde_indices,
                support_limit=min(len(X_tilde_indices), self.d*(self.d+1)//2)
            )
            if lambdas_phase is None:
                # fallback: uniformly sample active arms for remaining rounds
                while t < self.T+1:
                    chosen = int(np.random.choice(X_tilde_indices))
                    r = environment.pull_arm(chosen)
                    history.append(chosen)
                    t += 1
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
            remaining_budget = self.T+1 - t
            # compute planned pulls per support index, but we will cap as we go
            planned_pulls = [math.ceil(lambdas_phase[idx] * T_prime) for idx in range(len(indices_phase))]

            for idx_local, arm_global in enumerate(indices_phase):
                if remaining_budget <= 0:
                    break
                pulls = min(planned_pulls[idx_local], remaining_budget)
                if pulls <= 0:
                    continue
                xvec = self.X[arm_global]
                for _ in range(pulls):
                    r = environment.pull_arm(arm_global)
                    history.append(int(arm_global))
                    t += 1
                    remaining_budget -= 1
                    V += np.outer(xvec, xvec)
                    sum_rX += r * xvec
                    if t >= self.T:
                        break

            # Re-estimate theta_hat using cumulative V and sum_rX
            V_inv = np.linalg.pinv(V + 1e-9 * np.eye(self.d))
            theta_hat = V_inv @ sum_rX
            est_rewards = self.X @ theta_hat

            # recompute LCB/UCB and shrink X_tilde
            lcb_vals = np.array([self._LCB(self.X[i], theta_hat, V_inv) for i in range(self.num_arms)])
            ucb_vals = np.array([self._UCB(self.X[i], theta_hat, V_inv) for i in range(self.num_arms)])
            max_lcb = np.max(lcb_vals)
            X_tilde_indices = np.where(ucb_vals >= max_lcb)[0]

            # double T_prime for next phase (but not beyond remaining budget if you prefer)
            T_prime = min(self.T, int(2 * T_prime))

        # print(t)
        # done Phase II; return history (plus optional stats)
        return history
        # T_prime = int(round((2.0 / 3.0) * tilde_T))
        # if T_prime <= 0:
        #     T_prime = 1

        # # loop phases until we exhaust horizon
        # while total_rounds < self.T:

        #     V = np.zeros((self.d, self.d))
        #     sum_rX = np.zeros(self.d)

        #     # if len(X_tilde_indices) == 0:
        #     #     # nothing left, pull arbitrary arm (best estimate)
        #     #     best = np.argmax(est_rewards)
        #     #     remaining = self.T - total_rounds
        #     #     for _ in range(remaining):
        #     #         history.append(best)
        #     #         environment.pull_arm(best)  # we don't track further stats here
        #     #     break

        #     # Solve D-opt over X_tilde_indices with support constraint
        #     lambdas_phase, indices_phase = self._solve_d_optimal_design(X_tilde_indices, support_limit=min(len(X_tilde_indices), self.d*(self.d+1)//2))
        #     if lambdas_phase is None:
        #         # fallback: uniformly sample active arms for remaining rounds
        #         while total_rounds < self.T:
        #             chosen = int(np.random.choice(X_tilde_indices))
        #             environment.pull_arm(chosen)
        #             history.append(chosen)
        #             total_rounds += 1
        #         break

        #     # For each arm in support, pull ceil(lambda_i * T') times (or until horizon)
        #     # indices_phase aligned with lambdas_phase and holds global arm indices
        #     for idx_local, arm_global in enumerate(indices_phase):
        #         pulls = math.ceil(lambdas_phase[idx_local] * T_prime)
        #         # ensure not to exceed horizon
        #         pulls = min(pulls, self.T - total_rounds)
        #         if pulls <= 0:
        #             continue
        #         xvec = self.X[arm_global]
        #         sum_rewards_local = 0
        #         for _ in range(pulls):
        #             r = environment.pull_arm(arm_global)
        #             history.append(arm_global)
        #             total_rounds += 1
        #             sum_rewards_local += r
        #             V += np.outer(xvec, xvec)
        #             sum_rX += r * xvec
        #             if total_rounds >= self.T:
        #                 break
        #         if total_rounds >= self.T:
        #             break

        #     # Re-estimate theta_hat using new V and sum_rX
        #     try:
        #         V_inv = np.linalg.pinv(V + 1e-9 * np.eye(self.d))
        #         theta_hat = V_inv @ sum_rX
        #     except Exception:
        #         theta_hat = np.linalg.pinv(V) @ sum_rX

        #     # recompute LCB/UCB and shrink X_tilde
        #     V_inv = np.linalg.pinv(V + 1e-9 * np.eye(self.d))
        #     est_rewards = self.X @ theta_hat
        #     lcb_vals = np.array([self._LCB(self.X[i], theta_hat, V_inv) for i in range(self.num_arms)])
        #     ucb_vals = np.array([self._UCB(self.X[i], theta_hat, V_inv) for i in range(self.num_arms)])
        #     max_lcb = np.max(lcb_vals)
        #     X_tilde_indices = np.where(ucb_vals >= max_lcb)[0]

        #     # double T_prime for next phase
        #     T_prime = min(self.T, int(2 * T_prime))
        #     # if T_prime becomes huge relative to remaining budget, next loop will cap pulls at horizon

        # return history


def simulate_linnash(env, X, T, num_trials=10, sigma2=1.0, regret_type="Nash"):
    mu_star = np.max(env.mean_rewards)
    total_rewards = []
    for _ in tqdm(range(num_trials), desc="LINNASH Trials"):
        algo = LinNash(X, T, sigma2=sigma2)
        history = algo.run(env)
        rewards = env.mean_rewards[history]
        total_rewards.append(rewards)
    total_rewards = np.array(total_rewards)
    expected_means = np.mean(total_rewards, axis=0)

    if regret_type == "Nash":
        cumsum_log = np.cumsum(np.log(np.maximum(expected_means, 1e-300)))
        print(cumsum_log)
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

    # X /= np.linalg.norm(X, axis=1, keepdims=True)

    env = LinearBanditEnvironment(X, theta_star)
    print("Running LINNASH (modified) ...")
    # regret_curve = simulate_linnash(env, X, T, num_trials=1, sigma2=1.0, regret_type="Avg")
    cr = simulate_linnash(env, X, T, num_trials=10, sigma2=1.0, regret_type="Nash")
    np.save("hhs.npy", cr)
    regret_curve = np.load("hhb.npy")
    # cr = np.load("hha.npy")
    plt.plot(regret_curve, label= "Barman")
    plt.plot(cr, label='Ours')
    plt.xlabel("Rounds")
    plt.ylabel("Nash regret")
    plt.legend()
    plt.show()

    