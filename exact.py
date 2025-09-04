import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from math import log, sqrt

class LinearBanditEnvironment:
    """
    Simulates a stochastic linear bandit environment.
    """
    def _init_(self, X, theta_star):
        self.X = X
        self.theta_star = theta_star
        self.d = theta_star.shape[0]
        self.n_arms = X.shape[0]
        self.best_arm_idx = np.argmax(self.X @ self.theta_star)
        self.max_reward = np.max(self.X @ self.theta_star)

    def pull_arm(self, arm_idx):
        """
        Pulls an arm and returns a stochastic reward.
        Reward is Bernoulli distributed for simplicity.
        """
        mean_reward = self.X[arm_idx] @ self.theta_star
        # Ensure mean_reward is in [0, 1] for Bernoulli
        mean_reward = np.clip(mean_reward, 0, 1)
        return np.random.binomial(1, mean_reward)

class LinNash:
    """
    Implementation of the LINNASH algorithm from the paper
    "Nash Regret Guarantees for Linear Bandits".
    """
    def _init_(self, X, T):
        self.X = X
        self.T = T
        self.n_arms, self.d = X.shape
        self.nu = 1.0  # sub-Poisson parameter, assume 1 for Bernoulli rewards
        self.beta = 1.0 # From Theorem 1, depends on max reward, assume 1

        self.history = []
        self.arm_indices = np.arange(self.n_arms)

    def run(self):
        """
        Executes the full LINNASH algorithm.
        """
        # Part I: Initial exploration and elimination
        t_tilde, initial_arm_pulls, rewards = self.run_part_one()
        
        # Part II: Phased elimination
        self.run_part_two(t_tilde, initial_arm_pulls, rewards)

        return self.history

    def run_part_one(self):
        """
        Implements Part I of the algorithm: John Ellipsoid sampling and D-optimal design.
        """
        t_tilde = int(3 * sqrt(self.T * self.d * self.nu * log(self.T * self.n_arms)))
        if t_tilde > self.T:
            t_tilde = self.T

        print(f"Running Part I for t_tilde = {t_tilde} rounds...")

        # Get D-optimal design distribution for all arms
        lambda_d_opt, _ = self._solve_d_optimal_design(self.arm_indices)
        d_opt_support = {idx: p for idx, p in zip(self.arm_indices, lambda_d_opt) if p > 0}
        
        # Get John Ellipsoid sampling distribution
        # This now calls the exact method
        alpha_john, john_support_indices = self._get_john_ellipsoid_dist_exact()
        john_dist = {idx: p for idx, p in zip(john_support_indices, alpha_john) if p > 0}

        # Generate arm sequence S as per Algorithm 1 (simplified)
        sequence_s = []
        for t in range(t_tilde):
            if np.random.rand() < 0.5: # Flip a coin
                # Sample from John Ellipsoid distribution
                chosen_arm = np.random.choice(list(john_dist.keys()), p=list(john_dist.values()))
            else:
                # Sample from D-optimal design distribution
                chosen_arm = np.random.choice(list(d_opt_support.keys()), p=list(d_opt_support.values()))
            sequence_s.append(chosen_arm)
        
        # Pull arms and collect data
        V = np.zeros((self.d, self.d))
        sum_rX = np.zeros(self.d)
        
        initial_arm_pulls = []
        rewards = []

        for t in range(t_tilde):
            arm_idx = sequence_s[t]
            reward = env.pull_arm(arm_idx)
            self.history.append({'t': t, 'arm_idx': arm_idx, 'reward': reward})
            initial_arm_pulls.append(arm_idx)
            rewards.append(reward)
            
            x_t = self.X[arm_idx]
            V += np.outer(x_t, x_t)
            sum_rX += reward * x_t

        # Initial estimate and elimination
        V_inv = np.linalg.inv(V + np.eye(self.d) * 1e-6)
        theta_hat = V_inv @ sum_rX
        
        self._eliminate_arms(theta_hat, t_tilde)
        
        return t_tilde, initial_arm_pulls, rewards

    def run_part_two(self, t_start, initial_arm_pulls, rewards):
        """
        Implements Part II of the algorithm: phased elimination.
        """
        t = t_start
        phase = 0
        t_prime = (2/3) * t_start

        while t < self.T:
            print(f"\n--- Phase {phase}, t = {t}, active arms = {len(self.arm_indices)} ---")
            
            # 1. Find D-optimal design on surviving arms
            if len(self.arm_indices) <= 1:
                print("Only one arm left, pulling it for the remainder.")
                rem_pulls = self.T - t
                for _ in range(rem_pulls):
                     arm_idx = self.arm_indices[0]
                     reward = env.pull_arm(arm_idx)
                     self.history.append({'t': t, 'arm_idx': arm_idx, 'reward': reward})
                     t += 1
                break

            lambdas, active_indices = self._solve_d_optimal_design(self.arm_indices)
            support = {idx: p for idx, p in zip(active_indices, lambdas) if p > 0}

            # 2. Pull arms according to the design
            V_phase = np.zeros((self.d, self.d))
            sum_rX_phase = np.zeros(self.d)
            
            pulls_in_phase = 0
            for arm_idx, p in support.items():
                num_pulls = int(np.ceil(p * t_prime))
                if t + num_pulls > self.T:
                    num_pulls = self.T - t

                for _ in range(num_pulls):
                    reward = env.pull_arm(arm_idx)
                    self.history.append({'t': t, 'arm_idx': arm_idx, 'reward': reward})
                    x_t = self.X[arm_idx]
                    V_phase += np.outer(x_t, x_t)
                    sum_rX_phase += reward * x_t
                    t += 1
                
                pulls_in_phase += num_pulls
                if t >= self.T: break
            
            # 3. Re-estimate and eliminate
            if pulls_in_phase > 0:
                V_inv_phase = np.linalg.inv(V_phase + np.eye(self.d) * 1e-6)
                theta_hat_phase = V_inv_phase @ sum_rX_phase
                self._eliminate_arms(theta_hat_phase, t_prime)

            # 4. Update for next phase
            t_prime *= 2
            phase += 1
            if t >= self.T: break

    def _eliminate_arms(self, theta_hat, t_prime_val):
        """
        Calculates LNCB and UNCB for surviving arms and eliminates suboptimal ones.
        """
        if len(self.arm_indices) <= 1:
            return

        active_arms = self.X[self.arm_indices]
        estimated_rewards = active_arms @ theta_hat

        # Use a small positive constant to prevent sqrt of zero or negative
        safe_estimated_rewards = np.maximum(estimated_rewards, 1e-9)

        confidence_widths = 6 * np.sqrt(
            (safe_estimated_rewards * self.nu * self.d * np.log(self.T * self.n_arms)) / t_prime_val
        )
        
        uncb = estimated_rewards + confidence_widths
        lncb = estimated_rewards - confidence_widths

        max_lncb = np.max(lncb)
        
        survivor_mask = uncb >= max_lncb
        
        print(f"Elimination phase: {len(self.arm_indices)} -> {np.sum(survivor_mask)} arms.")
        self.arm_indices = self.arm_indices[survivor_mask]

    def _solve_d_optimal_design(self, arm_set_indices):
        """
        Solves the D-optimal design problem for a given set of arms using cvxpy.
        """
        n_active = len(arm_set_indices)
        active_arms = self.X[arm_set_indices]
        
        lambda_vars = cp.Variable(n_active, nonneg=True)
        
        U = cp.sum([lambda_vars[i] * np.outer(active_arms[i], active_arms[i]) for i in range(n_active)])
        
        constraints = [cp.sum(lambda_vars) == 1]
        objective = cp.Maximize(cp.log_det(U))
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS) # Using SCS solver which is good for this type
        
        if lambda_vars.value is None: # Handle solver failure
            print("D-optimal design solver failed, returning uniform distribution.")
            lambdas = np.ones(n_active) / n_active
        else:
            lambdas = lambda_vars.value
            lambdas[lambdas < 1e-6] = 0
            lambdas /= np.sum(lambdas)
            
        return lambdas, arm_set_indices

    def _get_john_ellipsoid_dist_exact(self):
        """
        Finds the exact John Ellipsoid center and the corresponding sampling distribution.
        This is a more precise but computationally intensive method.
        """
        print("Computing exact John Ellipsoid center...")
        
        # 1. Compute Convex Hull of all arms
        try:
            hull = ConvexHull(self.X)
        except:
             # Fallback for low-dimensional or degenerate cases
            print("ConvexHull failed, falling back to uniform sampling over all arms.")
            return np.ones(self.n_arms) / self.n_arms, np.arange(self.n_arms)

        # Hull equations are of the form A*x + b <= 0, which is a.T*x <= -b
        A_hull = hull.equations[:, :-1] # Normals 'a'
        b_hull = -hull.equations[:, -1] # Offsets '-b'

        # 2. Solve for the John Ellipsoid using CVXPY (SDP)
        # An ellipsoid is defined by {x | (x-c).T * E_inv * (x-c) <= 1}
        # We model it as {Bu+c | ||u||_2 <= 1}, where B is psd.
        # Maximize log_det(B) subject to ellipsoid being inside the hull.
        B = cp.Variable((self.d, self.d), PSD=True)
        c = cp.Variable(self.d)
        
        # Constraint: a_i.T * c + ||B @ a_i||_2 <= b_i for each facet
        constraints = [
            A_hull[i] @ c + cp.norm(B @ A_hull[i], 2) <= b_hull[i] for i in range(len(A_hull))
        ]
        
        objective = cp.Maximize(cp.log_det(B))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if c.value is None:
            print("John Ellipsoid solver failed, falling back to uniform on hull vertices.")
            center_c = np.mean(self.X[hull.vertices], axis=0)
        else:
            center_c = c.value

        # 3. Find sampling distribution alpha using Caratheodory's theorem (via LP)
        # We want to find alpha_i such that sum(alpha_i * y_i) = center_c,
        # where y_i are the vertices of the hull.
        vertices = self.X[hull.vertices]
        n_vertices = len(vertices)
        
        # Objective: find any feasible solution, so it can be 0.
        c_obj = np.zeros(n_vertices)
        
        # Equality constraints: A_eq @ alpha = b_eq
        A_eq = np.vstack([vertices.T, np.ones((1, n_vertices))])
        b_eq = np.hstack([center_c, 1])
        
        # Bounds: 0 <= alpha_i <= 1
        bounds = (0, 1)
        
        res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            alphas = res.x
            alphas[alphas < 1e-6] = 0
            alphas /= np.sum(alphas)
            return alphas, hull.vertices
        else:
            print("LP for John Ellipsoid distribution failed. Falling back to uniform on vertices.")
            alphas = np.ones(n_vertices) / n_vertices
            return alphas, hull.vertices
            
if _name_ == '_main_':
    # --- Experiment Setup ---
    d = 10
    n_arms = 100
    T = 20000

    print(f"Setting up experiment with d={d}, |X|={n_arms}, T={T}")

    # Generate random arms and an unknown theta_star
    np.random.seed(42)
    X = np.random.randn(n_arms, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True) # Normalize arms to unit sphere
    theta_star = np.random.randn(d)
    theta_star /= np.linalg.norm(theta_star) # Normalize for predictable reward scale

    # Shift rewards to be mostly positive for Nash Regret calculation
    mean_rewards = X @ theta_star
    min_mean = np.min(mean_rewards)
    theta_star = theta_star - X[np.argmin(mean_rewards)] * min_mean * 1.1
    mean_rewards = X @ theta_star

    # --- Run Algorithm ---
    env = LinearBanditEnvironment(X, theta_star)
    print(f"Optimal arm has expected reward: {env.max_reward:.4f}\n")

    linnash_algo = LinNash(X, T)
    history = linnash_algo.run()
    
    print("\nLINNASH finished.")

    # --- Analysis ---
    print("\n--- Results Analysis ---")
    df = pd.DataFrame(history)
    
    # Calculate Cumulative Regret
    cumulative_reward = df['reward'].sum()
    optimal_reward = env.max_reward * T
    regret = optimal_reward - cumulative_reward
    print(f"Cumulative Regret: {regret:.2f}")

    # Calculate Nash Regret
    expected_rewards = X @ theta_star
    df['expected_reward'] = df['arm_idx'].apply(lambda idx: expected_rewards[idx])
    
    # Use log-sum-exp for numerical stability
    log_geo_mean = df['expected_reward'].apply(lambda r: np.log(max(r, 1e-9))).mean()
    geo_mean = np.exp(log_geo_mean)
    nash_regret = env.max_reward - geo_mean
    
    print(f"Final Geometric Mean of Rewards: {geo_mean:.4f}")
    print(f"Nash Regret: {nash_regret:.4f}")

    # --- Visualization ---
    try:
        print("\nPlotting moving average of rewards...")
        window_size = 500
        df['moving_average'] = df['reward'].rolling(window=window_size).mean()

        plt.figure(figsize=(12, 7))
        plt.plot(df['t'], df['moving_average'], label=f'Moving Average (window={window_size})')
        plt.axhline(y=env.max_reward, color='r', linestyle='--', label='Optimal Reward')
        plt.title('LINNASH Performance: Moving Average of Rewards')
        plt.xlabel('Rounds (t)')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping plot.")
