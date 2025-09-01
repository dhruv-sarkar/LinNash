import numpy as np
import cvxpy as cp
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import math
import warnings

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
        self.mean_rewards = np.clip(self.X @ self.theta_star, 0, 1)
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
        return np.random.binomial(1, mean)

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

    def _solve_d_optimal_design(self, arm_set_indices):
        """
        Solves the D-optimal design problem using CVXPY.
        Maximizes log(det(U)), where U = sum(lambda_x * x * x^T).
        """
        n_active = len(arm_set_indices)
        if n_active == 0:
            return None, None
        
        active_arms = self.X[arm_set_indices]
        lambda_vars = cp.Variable(n_active, nonneg=True)
        
        U = cp.sum([lambda_vars[i] * np.outer(active_arms[i], active_arms[i]) for i in range(n_active)])
        
        constraints = [cp.sum(lambda_vars) == 1]
        objective = cp.Maximize(cp.log_det(U))
        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Filter out negligible probabilities
                lambdas = lambda_vars.value
                lambdas[lambdas < 1e-6] = 0
                lambdas /= np.sum(lambdas)
                return lambdas, arm_set_indices
            else:
                return None, None
        except (cp.error.SolverError, ValueError):
            return None, None

    def _get_john_ellipsoid_dist(self):
        """
        Calculates a sampling distribution 'U' based on the center of the
        John Ellipsoid of the convex hull of the arms.
        This implementation finds the Chebyshev center as a practical approximation.
        """
        try:
            hull = ConvexHull(self.X)
            hull_vertices = self.X[hull.vertices]
            
            # Find Chebyshev center (c, r) of the convex hull
            # max r s.t. a_i^T * c + r * ||a_i|| <= b_i for each facet
            A = hull.equations[:, :-1]
            b = -hull.equations[:, -1]
            
            c_obj = np.zeros(self.d + 1)
            c_obj[-1] = -1  # Maximize r by minimizing -r

            A_ub = np.hstack([A, np.linalg.norm(A, axis=1, keepdims=True)])
            b_ub = b
            
            res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
            
            if not res.success: return None, None
            center = res.x[:-1]

            # Express the center as a convex combination of hull vertices
            # (Carathéodory's theorem). Solved via a feasibility LP.
            Y = hull_vertices.T
            c_feasibility = np.zeros(len(hull.vertices))
            A_eq = np.vstack([Y, np.ones(len(hull.vertices))])
            b_eq = np.hstack([center, 1])
            
            res_alpha = linprog(c=c_feasibility, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
            
            if not res_alpha.success: return None, None
            
            alphas = res_alpha.x
            alphas[alphas < 1e-6] = 0
            alphas /= np.sum(alphas)
            
            return alphas, hull.vertices

        except Exception:
            # Fallback for geometric errors (e.g., if arms are not full-dimensional)
            return None, None


    def _generate_arm_sequence(self, tilde_T):
        """
        Implements Algorithm 1 to generate the initial arm sequence for Part I.
        """
        # 1. D-optimal design distribution
        lambda_d_opt, indices_d_opt = self._solve_d_optimal_design(self.arm_indices)
        
        # 2. John Ellipsoid center distribution
        alphas_john, indices_john = self._get_john_ellipsoid_dist()
        
        # Fallback to uniform if geometric calculations fail
        if alphas_john is None:
            indices_john = self.arm_indices
            alphas_john = np.ones(self.num_arms) / self.num_arms

        # 3. Generate the sequence
        arm_sequence = []
        
        support_d_opt = indices_d_opt[lambda_d_opt > 0]
        lambda_support = lambda_d_opt[lambda_d_opt > 0]
        
        counts_d_opt = {arm_idx: 0 for arm_idx in support_d_opt}
        min_pulls_d_opt = {
            arm_idx: math.ceil(lam * tilde_T / 3) 
            for arm_idx, lam in zip(support_d_opt, lambda_support)
        }
        
        d_opt_arm_pool = list(support_d_opt)
        d_opt_robin_idx = 0

        for _ in range(tilde_T):
            if np.random.rand() < 0.5 and len(d_opt_arm_pool) > 0:
                # D/G-OPT arm selection (round-robin)
                arm_idx = d_opt_arm_pool[d_opt_robin_idx]
                arm_sequence.append(arm_idx)
                counts_d_opt[arm_idx] += 1
                
                if counts_d_opt[arm_idx] >= min_pulls_d_opt[arm_idx]:
                    d_opt_arm_pool.pop(d_opt_robin_idx)
                
                if len(d_opt_arm_pool) > 0:
                    d_opt_robin_idx = (d_opt_robin_idx + 1) % len(d_opt_arm_pool)
                else: # Reset if pool gets exhausted early
                    d_opt_robin_idx = 0

            else:
                # SAMPLE-U (John Ellipsoid) arm selection
                arm_idx = np.random.choice(indices_john, p=alphas_john)
                arm_sequence.append(arm_idx)
                
        return arm_sequence

    def run(self, environment):
        """
        Executes the full LINNASH algorithm.

        Returns:
            list: A list of arm indices pulled in each round.
        """
        history = []
        total_rounds_played = 0

        # --- Part I ---
        log_term = np.log(self.T * self.num_arms)
        tilde_T = int(3 * np.sqrt(self.T * self.d * self.nu * log_term))
        tilde_T = min(tilde_T, self.T)
        
        V = np.zeros((self.d, self.d))
        sum_rX = np.zeros(self.d)

        # Generate arm sequence for the first tilde_T rounds
        arm_sequence_part1 = self._generate_arm_sequence(tilde_T)
        
        for t in range(tilde_T):
            arm_idx = arm_sequence_part1[t]
            arm_vec = self.X[arm_idx]
            reward = environment.pull_arm(arm_idx)
            
            V += np.outer(arm_vec, arm_vec)
            sum_rX += reward * arm_vec
            history.append(arm_idx)
        
        total_rounds_played += tilde_T
        
        # Estimate and eliminate
        try:
            V_inv = np.linalg.inv(V)
            theta_hat = V_inv @ sum_rX
        except np.linalg.LinAlgError:
            # Fallback if V is singular
            theta_hat = np.linalg.pinv(V) @ sum_rX

        # Confidence bound calculation
        t_prime_part1 = tilde_T / 3
        
        est_rewards = self.X @ theta_hat
        # Clip estimated rewards at a small positive value to avoid math errors in sqrt
        est_rewards_clipped = np.maximum(1e-9, est_rewards)

        confidence_widths = 6 * np.sqrt(
            (est_rewards_clipped * self.nu * self.d * log_term) / t_prime_part1
        )
        lncb = est_rewards - confidence_widths
        uncb = est_rewards + confidence_widths

        # Elimination
        max_lncb = np.max(lncb)
        active_arm_indices = self.arm_indices[uncb >= max_lncb]
        
        # --- Part II ---
        T_prime = (2 / 3) * tilde_T

        while total_rounds_played < self.T:
            if len(active_arm_indices) <= 1:
                # If one or zero arms are left, play the best one
                if len(active_arm_indices) == 1:
                    arm_idx_to_play = active_arm_indices[0]
                else: # Fallback if all arms are eliminated
                    arm_idx_to_play = np.random.choice(self.arm_indices)
                
                remaining_rounds = self.T - total_rounds_played
                for _ in range(remaining_rounds):
                    history.append(arm_idx_to_play)
                break
            
            # New phase
            V_phase = np.zeros((self.d, self.d))
            s_phase = np.zeros(self.d)
            
            lambdas, indices = self._solve_d_optimal_design(active_arm_indices)
            
            if lambdas is None: # Fallback if solver fails
                arm_idx = np.random.choice(active_arm_indices)
                reward = environment.pull_arm(arm_idx)
                history.append(arm_idx)
                total_rounds_played += 1
                continue

            # Pull arms according to D-optimal design
            rounds_this_phase = 0
            for i, arm_idx in enumerate(indices):
                num_pulls = math.ceil(lambdas[i] * T_prime)
                if total_rounds_played + num_pulls > self.T:
                    num_pulls = self.T - total_rounds_played
                
                arm_vec = self.X[arm_idx]
                rewards_sum = 0
                for _ in range(num_pulls):
                    rewards_sum += environment.pull_arm(arm_idx)
                    history.append(arm_idx)

                V_phase += num_pulls * np.outer(arm_vec, arm_vec)
                s_phase += rewards_sum * arm_vec
                rounds_this_phase += num_pulls
                total_rounds_played += num_pulls
                
                if total_rounds_played >= self.T:
                    break

            # Estimate and eliminate again
            try:
                theta_hat = np.linalg.pinv(V_phase) @ s_phase
            except np.linalg.LinAlgError:
                theta_hat = np.zeros(self.d) # Should not happen if design is good
            
            est_rewards = self.X @ theta_hat
            est_rewards_clipped = np.maximum(1e-9, est_rewards)
            
            confidence_widths = 6 * np.sqrt(
                (est_rewards_clipped * self.nu * self.d * log_term) / T_prime
            )
            lncb = est_rewards - confidence_widths
            uncb = est_rewards + confidence_widths

            max_lncb = np.max(lncb[active_arm_indices])
            active_arm_indices = active_arm_indices[uncb[active_arm_indices] >= max_lncb]
            
            # Update for next phase
            T_prime *= 2

        return history

if __name__ == '__main__':
    # --- Experimental Setup ---
    # This setup is inspired by the paper's experiments section.
    d = 10         # Ambient dimension
    num_arms = 100 # Number of arms
    T = 20000      # Time horizon

    print(f"Setting up experiment with d={d}, |X|={num_arms}, T={T}")

    # Generate synthetic data
    np.random.seed(42)
    theta_star = np.random.randn(d)
    theta_star /= np.linalg.norm(theta_star) # Normalize for stability

    X = np.random.randn(num_arms, d)
    # Normalize each arm vector
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    # Create environment
    env = LinearBanditEnvironment(X, theta_star)
    print(f"Optimal arm has expected reward: {env.optimal_reward:.4f}")

    # Initialize and run LINNASH
    linnash_algo = LinNash(X, T, nu=1.0)
    print("\nRunning LINNASH...")
    history = linnash_algo.run(env)
    print("LINNASH finished.")

    # --- Analysis ---
    print("\n--- Results Analysis ---")
    
    # Calculate cumulative and Nash regret
    expected_rewards_pulled = env.mean_rewards[history]
    
    cumulative_regret = np.sum(env.optimal_reward - expected_rewards_pulled)
    print(f"Cumulative Regret: {cumulative_regret:.2f}")

    # To avoid log(0), we use a small epsilon for zero-reward arms if any
    log_rewards = np.log(np.maximum(1e-9, expected_rewards_pulled))
    geometric_mean_reward = np.exp(np.mean(log_rewards))
    
    nash_regret = env.optimal_reward - geometric_mean_reward
    print(f"Final Geometric Mean of Rewards: {geometric_mean_reward:.4f}")
    print(f"Nash Regret: {nash_regret:.4f}")
    
    # You can use a library like matplotlib to plot the results
    # For example, to plot the moving average of rewards:
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        print("\nPlotting moving average of rewards...")
        rewards_s = pd.Series(expected_rewards_pulled)
        moving_avg = rewards_s.rolling(window=T//50).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(moving_avg)
        plt.axhline(y=env.optimal_reward, color='r', linestyle='--', label=f'Optimal Reward ({env.optimal_reward:.2f})')
        plt.title('1000-Round Moving Average of Expected Rewards (LINNASH)')
        plt.xlabel('Rounds')
        plt.ylabel('Expected Reward')
        plt.legend()
        plt.grid(True)
        # plt.savefig("linnash_performance.png") # Uncomment to save the plot
        plt.show()

    except ImportError:
        print("\nPlease install pandas and matplotlib (`pip install pandas matplotlib`) to see the plot.")
