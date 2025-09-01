# LINNASH: Nash Regret Guarantees for Linear Bandits

This repository contains a Python implementation of the **LINNASH** algorithm, as described in the paper  
[*"Nash Regret Guarantees for Linear Bandits"*](https://arxiv.org/abs/2310.02023) by Sawarni, Pal, and Barman.

The project provides a self-contained, runnable script (`linnash.py`) that simulates a linear bandit environment and demonstrates the effectiveness of the LINNASH algorithm in minimizing **Nash Regret**, a fairness-oriented alternative to traditional regret measures.

---

## About the Algorithm

Standard bandit algorithms aim to minimize **average regret**, which focuses on maximizing the *sum of rewards*. However, this can lead to unfair outcomes where some rounds receive very low rewards.

**LINNASH** addresses this by minimizing **Nash Regret**, which uses the **geometric mean of rewards**.  
This incentivizes the algorithm to maintain a consistently high reward in every round, providing a stronger fairness guarantee.

The algorithm operates in **two main phases**:

### Part I: Initial Exploration
- Prevents the geometric mean from collapsing due to low initial rewards.
- Uses a novel sampling strategy involving the **John Ellipsoid** and **D-optimal design**.
- Ensures expected rewards in the early rounds are bounded and sufficiently high.
- After this phase, clearly suboptimal arms are eliminated.

### Part II: Phased Elimination
- Proceeds in **phases of exponentially increasing length**.
- In each phase:
  - Uses a **D-optimal design** to select arms for exploration.
  - Refines its estimate of the reward parameters.
  - Eliminates more suboptimal arms using **estimate-dependent confidence bounds**.

---

## Key Concepts Implemented

This implementation faithfully reproduces the two most critical components of the LINNASH algorithm.

### 1. D-Optimal Design for Exploration
The goal of D-optimal design is to find a probability distribution **λ** over the arms that maximizes the information gathered.  
Formally, this maximizes the determinant of the information matrix:

$$
U(\lambda) = \sum_i \lambda_i x_i x_i^T
$$

This is a convex optimization problem.

**Implementation with `cvxpy`:**

```python
lambda_vars = cp.Variable(n_active, nonneg=True)
U = cp.sum([lambda_vars[i] * np.outer(arms[i], arms[i]) for i in range(n_active)])
objective = cp.Maximize(cp.log_det(U))
problem = cp.Problem(objective, [cp.sum(lambda_vars) == 1])
problem.solve()
```
### 2. John Ellipsoid Sampling for Fair Initial Rewards

To guarantee a high baseline reward in Part I, LINNASH uses a sampling distribution whose expectation is the **center** of the John Ellipsoid (the largest ellipsoid inscribed in the convex hull of the arms).

---

#### Practical Implementation with `scipy`

Computing the exact John Ellipsoid is complex, so we use a **two-step approximation**:

1. **Chebyshev Center**  
   - Compute the center of the largest sphere inside the convex hull.  
   - This serves as a proxy for the John Ellipsoid’s center.  
   - The convex hull is found using `scipy.spatial.ConvexHull`.  
   - A Linear Program is solved with `scipy.optimize.linprog` to find the center \(c\) and radius \(r\).

2. **Sampling Distribution**  
   - By **Carathéodory's theorem**, the center \(c\) can be expressed as a convex combination of the hull’s vertices: $c = \sum_{i} \alpha_i y_i$
   - The coefficients $\(\alpha_i \geq 0\)$ with $\(\sum_i \alpha_i = 1\)$ give the desired **sampling probabilities**.  
   - A second LP is solved to compute a feasible set of coefficients $\(\{\alpha_i\}\)$.

---

This method provides a robust and computationally feasible approximation of the paper’s **John Ellipsoid Sampling** strategy.
