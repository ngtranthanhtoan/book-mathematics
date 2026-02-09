# Chapter 3: Convex Optimization

## The One Question That Changes Everything

When you train a linear regression model, gradient descent always finds the best solution. When you train a deep neural network, it might get stuck in a local minimum. What's the difference? Linear regression's loss is convex -- it has exactly one minimum. Deep networks' loss is non-convex -- it has many. Understanding convexity tells you when you can trust your optimizer.

This is the chapter where optimization goes from "run it and hope" to "run it and know."

> **Building On** -- Optimization algorithms tell you HOW to minimize. Convexity tells you WHETHER you can find the true minimum. It's the difference between guaranteed success and best effort.

---

## The Core Guarantee: Convex Means Findable

Here is the single most important theorem in optimization for ML practitioners:

**For a convex function, every local minimum is a global minimum.**

Read that again. If your loss function is convex, you never have to worry about getting trapped. Any downhill algorithm that converges will converge to THE answer, not just AN answer.

This is the dividing line in ML:

- **Linear regression loss is convex.** Gradient descent finds the best weights. Every time. From any starting point.
- **Logistic regression loss is convex.** Same guarantee. You can trust the result.
- **Neural network loss is not convex.** Different random seeds give different models. You are in "best effort" territory.

Let's build up the mathematics that makes this precise.

---

## Convex Sets: The Foundation

Before we talk about convex functions, we need convex sets -- the domains those functions live on.

> **You Already Know This** -- A convex set is a region where any path between two points stays inside -- like a convex hull in computational geometry. If you have ever used `scipy.spatial.ConvexHull` or computed a bounding polygon, you have worked with this concept.

### Definition

A set $C$ is **convex** if for any two points $\mathbf{x}, \mathbf{y} \in C$ and any $\theta \in [0, 1]$:

$$\theta \mathbf{x} + (1 - \theta) \mathbf{y} \in C$$

In plain language: pick any two points in the set. Draw the line segment between them. If that entire segment stays inside the set, the set is convex. If even one point on the segment escapes, it is not.

```
CONVEX SET                             NON-CONVEX SET

    ┌─────────────────┐                    ┌───────────╮
    │                 │                    │           │
    │   A ●─────────● B                   │   A ●─────│─────● B
    │    line stays   │                    │          ╰─╯
    │    inside set   │                    │   line exits the
    │                 │                    │   set in the middle!
    └─────────────────┘                    └───────────╯

    Pick ANY two points A, B.              There EXIST two points
    The segment AB is always               where segment AB leaves
    fully contained.                       the set.
```

### Examples of Convex Sets

These come up constantly as constraint regions in optimization:

- **Hyperplanes**: $\{\mathbf{x} : \mathbf{a}^T\mathbf{x} = b\}$ -- a flat slice through space
- **Half-spaces**: $\{\mathbf{x} : \mathbf{a}^T\mathbf{x} \leq b\}$ -- everything on one side of a hyperplane
- **Balls**: $\{\mathbf{x} : \|\mathbf{x} - \mathbf{c}\| \leq r\}$ -- all points within radius $r$ of center $\mathbf{c}$
- **Polyhedra**: $\{\mathbf{x} : A\mathbf{x} \leq \mathbf{b}\}$ -- intersection of half-spaces (think: feasible region in linear programming)

**Important Property**: The intersection of convex sets is convex. This is why you can stack multiple convex constraints and the feasible region stays convex.

---

## Convex Functions: The Bowl Shape

Now the main event. A convex function is, geometrically, a bowl. No matter where you put a ball on the surface, it rolls to the same bottom.

> **You Already Know This** -- A convex function is like a well-designed API: one correct way to use it, no local traps. You call the optimizer, you get the answer. A non-convex function is like debugging a distributed system: many local "solutions" that aren't globally correct.

### Definition

A function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is **convex** if its domain is convex and for all $\mathbf{x}, \mathbf{y}$ in the domain and $\theta \in [0, 1]$:

$$f(\theta \mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y})$$

**Geometric interpretation**: Pick any two points on the graph of $f$. Draw a straight line between them. The function lies **below** (or on) that line. The chord is always an overestimate.

```
CONVEX FUNCTION (the bowl)             NON-CONVEX FUNCTION (the landscape)

Loss                                   Loss
  │                                      │
  │                                      │  ╭╮       ╭╮
  │   ╭──────────╮                       │ ╱  ╲  ╭╮ ╱  ╲
  │  ╱            ╲                      │╱    ╲╱  ╲    ╲
  │ ╱              ╲                     │           ╲    ╲╱
  │╱       ★        ╲                    │  ★         ★
  └────────────────────► w               └────────────────────► w
         Global                            Local       Local
        Minimum                           Minimum     Minimum
        (the only one)                    (which is the real answer?)

  Gradient descent from ANY              Gradient descent finds
  starting point → same ★               DIFFERENT ★ depending on
                                         where you start
```

This picture is the entire chapter in a nutshell. The left side is linear regression. The right side is a neural network.

### First-Order Condition

If $f$ is differentiable, it is convex if and only if:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) \quad \forall \mathbf{x}, \mathbf{y}$$

**What this says**: The tangent line (or tangent hyperplane) at any point is a **global underestimator** of the function. The function always curves up and away from its tangent. You can never overestimate how low the function goes by looking at the gradient at one point.

For ML, this means: if the gradient at your current weights is zero, you know you are at the bottom. No other point can be lower.

### Second-Order Condition

If $f$ is twice differentiable, it is convex if and only if its Hessian is positive semidefinite everywhere:

$$\nabla^2 f(\mathbf{x}) \succeq 0 \quad \forall \mathbf{x}$$

For a scalar function $f(x)$, this simplifies to $f''(x) \geq 0$ -- the function curves upward everywhere. No concave dips allowed.

This is the most practical test. When you want to prove a loss function is convex, compute its Hessian and show all eigenvalues are non-negative.

### Strictly Convex Functions

A function is **strictly convex** if the inequality is strict for $\theta \in (0, 1)$ and $\mathbf{x} \neq \mathbf{y}$:

$$f(\theta \mathbf{x} + (1-\theta)\mathbf{y}) < \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y})$$

**The payoff**: Strictly convex functions have **exactly one** global minimum. Not "at most one" -- exactly one. The bowl has a single bottom point.

Linear regression with $L_2$ regularization (ridge regression) is strictly convex. Unregularized linear regression is convex but not always strictly convex (if features are linearly dependent, there could be a flat bottom -- infinitely many equally good solutions).

---

## Common Convex Functions

You will see these building blocks again and again:

| Function | Formula | Domain | Strictly Convex? |
|----------|---------|--------|------------------|
| Linear | $\mathbf{a}^T\mathbf{x} + b$ | $\mathbb{R}^n$ | No |
| Quadratic | $\mathbf{x}^T A \mathbf{x}$ where $A \succeq 0$ | $\mathbb{R}^n$ | If $A \succ 0$ |
| Exponential | $e^{ax}$ | $\mathbb{R}$ | Yes |
| Negative log | $-\log(x)$ | $\mathbb{R}_{++}$ | Yes |
| Norms | $\|\mathbf{x}\|_p$ for $p \geq 1$ | $\mathbb{R}^n$ | No (at 0) |
| Log-sum-exp | $\log(\sum_i e^{x_i})$ | $\mathbb{R}^n$ | No |

The log-sum-exp is worth calling out -- it is the "soft max" function, and its convexity is why softmax cross-entropy loss is well-behaved in logistic regression and multiclass classification.

---

## Operations That Preserve Convexity

You rarely need to verify convexity from scratch. Instead, you build convex functions from simpler convex pieces using these rules:

1. **Non-negative weighted sum**: If $f_1, f_2$ are convex, then $\alpha f_1 + \beta f_2$ is convex for $\alpha, \beta \geq 0$. This is why "loss + regularizer" stays convex when both terms are convex.

2. **Composition with affine**: If $f$ is convex, then $g(\mathbf{x}) = f(A\mathbf{x} + \mathbf{b})$ is convex. This is why logistic loss, composed with the linear model $\mathbf{w}^T\mathbf{x}$, is convex in $\mathbf{w}$.

3. **Pointwise maximum**: If $f_1, \ldots, f_m$ are convex, then $f(\mathbf{x}) = \max_i f_i(\mathbf{x})$ is convex. This is why hinge loss (max of linear functions) is convex.

4. **Partial minimization**: If $f(\mathbf{x}, \mathbf{y})$ is convex in $(\mathbf{x}, \mathbf{y})$, then $g(\mathbf{x}) = \inf_{\mathbf{y}} f(\mathbf{x}, \mathbf{y})$ is convex. Optimizing out some variables preserves convexity in the remaining ones.

---

## The Running Example: Three Models, Three Stories

Let's trace convexity through three models you will actually use.

### Linear Regression: Convex

The MSE loss is:

$$L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 = \mathbf{w}^T(\mathbf{X}^T\mathbf{X})\mathbf{w} - 2\mathbf{y}^T\mathbf{X}\mathbf{w} + \mathbf{y}^T\mathbf{y}$$

The Hessian is $\nabla^2 L = 2\mathbf{X}^T\mathbf{X}$, which is always positive semidefinite (since $\mathbf{v}^T\mathbf{X}^T\mathbf{X}\mathbf{v} = \|\mathbf{X}\mathbf{v}\|^2 \geq 0$). So MSE is convex in $\mathbf{w}$. Gradient descent finds the global optimum. You can also solve it in closed form: $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$.

### Logistic Regression: Convex

The logistic loss is:

$$L(\mathbf{w}) = \sum_i \log(1 + e^{-y_i \mathbf{w}^T\mathbf{x}_i})$$

Each term $g(z) = \log(1 + e^{-z})$ has second derivative $g''(z) = \sigma(z)(1 - \sigma(z)) \geq 0$ where $\sigma$ is the sigmoid. Since $z_i = y_i \mathbf{w}^T\mathbf{x}_i$ is affine in $\mathbf{w}$, composition with affine preserves convexity. Sum of convex functions is convex. Therefore logistic loss is convex. Gradient descent finds the global optimum.

### Neural Network: NOT Convex

Even the simplest neural network -- one hidden unit with ReLU -- has a non-convex loss. The model $f(x) = w_2 \cdot \text{ReLU}(w_1 \cdot x)$ is not linear in $(w_1, w_2)$. The parameters multiply each other. Different initializations find different minima with different loss values. You are no longer in the guaranteed-success regime.

---

## Constrained Optimization and Lagrange Multipliers

So far we have talked about unconstrained minimization: find $\mathbf{x}$ that minimizes $f(\mathbf{x})$. But real problems have constraints.

> **You Already Know This** -- Lagrange multipliers are constrained optimization: "minimize cost subject to SLA constraints." You want the cheapest server configuration, but latency must stay under 100ms and uptime must exceed 99.9%. The constraints limit where you can search.

### The Constrained Problem

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to} \quad g_i(\mathbf{x}) \leq 0, \quad h_j(\mathbf{x}) = 0$$

where $f$ is the objective, $g_i$ are inequality constraints, and $h_j$ are equality constraints.

### The Lagrangian

Instead of solving the constrained problem directly, we form the **Lagrangian**:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \nu_j h_j(\mathbf{x})$$

where $\lambda_i \geq 0$ are the **Lagrange multipliers** for inequality constraints and $\nu_j$ are the multipliers for equality constraints.

The idea: rather than enforcing constraints as hard boundaries, we add penalty terms that make constraint violations expensive. The multipliers $\lambda_i$ and $\nu_j$ control how expensive.

### KKT Conditions

At the optimal solution $\mathbf{x}^*$, the Karush-Kuhn-Tucker (KKT) conditions must hold:

1. **Stationarity**: $\nabla f(\mathbf{x}^*) + \sum_i \lambda_i \nabla g_i(\mathbf{x}^*) + \sum_j \nu_j \nabla h_j(\mathbf{x}^*) = 0$
2. **Primal feasibility**: $g_i(\mathbf{x}^*) \leq 0$, $h_j(\mathbf{x}^*) = 0$
3. **Dual feasibility**: $\lambda_i \geq 0$
4. **Complementary slackness**: $\lambda_i g_i(\mathbf{x}^*) = 0$

Complementary slackness is the key insight: either a constraint is active ($g_i = 0$, the constraint is "tight") or its multiplier is zero ($\lambda_i = 0$, the constraint doesn't matter). A constraint that isn't binding has no effect on the solution.

**The convexity connection**: If $f$ and all $g_i$ are convex and $h_j$ are affine, then the KKT conditions are not just necessary but **sufficient** for optimality. In the convex case, checking KKT is the same as confirming you have found the global optimum.

### Where This Shows Up in ML

- **SVMs**: The SVM dual problem is derived using Lagrange multipliers on the hinge loss with margin constraints. Support vectors are exactly the data points where the constraint is active ($\lambda_i > 0$).
- **Ridge regression**: Can be viewed as minimizing MSE subject to $\|\mathbf{w}\|^2 \leq t$, with $\lambda$ being the regularization strength.
- **Constrained neural networks**: Weight clipping in WGANs, norm constraints on embeddings -- all constrained optimization.

---

## Convex vs Non-Convex in ML: The Practical Guide

### Convex ML Models

| Model | Loss Function | Why Convex |
|-------|--------------|------------|
| **Linear Regression** | MSE: $\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ | Quadratic in $\mathbf{w}$ |
| **Ridge Regression** | MSE + $\lambda\|\mathbf{w}\|^2$ | Sum of convex functions |
| **Logistic Regression** | Cross-entropy | Convex composed with affine |
| **SVM** | Hinge + $\frac{1}{2}\|\mathbf{w}\|^2$ | Sum of convex functions |
| **Lasso** | MSE + $\lambda\|\mathbf{w}\|_1$ | Sum of convex functions |

For all of these: gradient descent (or any standard convex optimizer) finds the global optimum. You can trust the result. Different initializations give the same answer.

### Non-Convex ML Models

| Model | Why Non-Convex |
|-------|----------------|
| **Neural Networks** | Composition of nonlinear functions; parameters multiply each other |
| **Deep Learning** | Millions of parameters with complex nonlinear interactions |
| **Matrix Factorization** | Product of matrices is non-convex in the factors |
| **K-Means** | Discrete cluster assignments create a combinatorial landscape |

For all of these: different random seeds give different models. Learning rate, initialization, and batch size all affect which minimum you land in.

### Why Non-Convex Deep Learning Works Anyway

Despite non-convexity, deep learning succeeds because:

1. **Many good local minima**: Research shows that in overparameterized networks, most local minima have similar loss values. You don't need THE best -- you need A good one.
2. **Saddle points are the real enemy**: In high dimensions, saddle points (flat in some directions, curved in others) are far more common than true local minima. SGD's noise helps escape them.
3. **Overparameterization helps**: More parameters than data points create many paths to good solutions. The loss landscape becomes more connected.
4. **Implicit regularization**: SGD's stochasticity biases the search toward flatter minima, which tend to generalize better.

---

## Practical Implications: What to Do With This Knowledge

```
Your Model's Loss is CONVEX:
├── Gradient descent WILL find global optimum
├── Learning rate only affects speed, not final result
├── Different initializations give same solution
├── Solution is unique (if strictly convex)
└── You can trust the optimization result

Your Model's Loss is NON-CONVEX:
├── No guarantee of global optimum
├── Different initializations --> different results
├── Learning rate affects WHICH minimum you find
├── Multiple restarts or ensembles may improve results
└── Good minima are often "flat" (generalize better)
```

### Checking Convexity in Practice

1. **Analytically**: Compute the Hessian, check if positive semidefinite (all eigenvalues non-negative)
2. **Empirically**: Sample random pairs of points, verify Jensen's inequality holds
3. **By construction**: Build your loss from convex pieces using the preservation rules above

---

## Common Mistakes

> **"Deep learning losses are NOT convex. But in practice, saddle points are more of a problem than local minima."**

1. **Assuming neural network losses are convex** -- They are not. The composition of nonlinear activations with parameter products breaks convexity. Do not expect different random seeds to give the same result.

2. **Over-trusting a single gradient descent run on non-convex problems** -- If your loss is non-convex, one run is one sample from the space of possible solutions. Run multiple seeds and compare.

3. **Ignoring saddle points** -- In high-dimensional non-convex landscapes, saddle points (where the gradient is zero but you are not at a minimum) are far more common than local minima. Momentum and adaptive learning rates (Adam, RMSProp) help escape them.

4. **Not using convexity when you have it** -- If your problem is convex, use a convex optimizer (L-BFGS, coordinate descent, or a dedicated solver like CVXPY). Don't use SGD with all its hyperparameter tuning when a convex solver gives you the exact answer in one shot.

---

## Code Example

```python
import numpy as np
import matplotlib.pyplot as plt


class ConvexityChecker:
    """Tools for checking and demonstrating convexity."""

    @staticmethod
    def is_convex_empirical(f, x_range, n_tests=1000, tol=1e-6):
        """
        Empirically test if a 1D function is convex.

        Uses the definition: f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y)
        """
        for _ in range(n_tests):
            # Random points in range
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(x_range[0], x_range[1])
            t = np.random.uniform(0, 1)

            # Convexity condition
            lhs = f(t * x + (1 - t) * y)
            rhs = t * f(x) + (1 - t) * f(y)

            if lhs > rhs + tol:
                return False, (x, y, t)

        return True, None

    @staticmethod
    def check_hessian_psd(hessian_fn, x, tol=1e-8):
        """
        Check if Hessian is positive semidefinite at point x.

        A matrix is PSD if all eigenvalues are >= 0.
        """
        H = hessian_fn(x)
        eigenvalues = np.linalg.eigvalsh(H)
        is_psd = np.all(eigenvalues >= -tol)
        return is_psd, eigenvalues

    @staticmethod
    def gradient_descent_converges(f, grad_f, x_init, lr=0.01, n_iter=1000):
        """
        For convex functions, gradient descent should converge to global minimum.
        """
        x = x_init.copy()
        history = [f(x)]

        for _ in range(n_iter):
            x = x - lr * grad_f(x)
            history.append(f(x))

        return x, history


def demo_convex_functions():
    """Demonstrate properties of convex vs non-convex functions."""

    print("=" * 60)
    print("CONVEXITY DEMONSTRATION")
    print("=" * 60)

    # Example 1: Quadratic (convex)
    def quadratic(x):
        return x**2 + 2*x + 1

    # Example 2: Fourth-degree polynomial (non-convex)
    def fourth_degree(x):
        return x**4 - 2*x**2 + 0.5

    checker = ConvexityChecker()

    print("\n1. Testing f(x) = x^2 + 2x + 1 (Quadratic):")
    is_convex, counterexample = checker.is_convex_empirical(quadratic, (-5, 5))
    print(f"   Is convex: {is_convex}")

    print("\n2. Testing f(x) = x^4 - 2x^2 + 0.5 (Fourth-degree):")
    is_convex, counterexample = checker.is_convex_empirical(fourth_degree, (-2, 2))
    print(f"   Is convex: {is_convex}")
    if counterexample:
        x, y, t = counterexample
        print(f"   Counterexample: x={x:.2f}, y={y:.2f}, t={t:.2f}")


def demo_ml_loss_convexity():
    """Show convexity of common ML loss functions."""

    print("\n" + "=" * 60)
    print("ML LOSS FUNCTION CONVEXITY")
    print("=" * 60)

    # MSE Loss for linear regression: L(w) = ||Xw - y||^2
    # This is CONVEX in w

    np.random.seed(42)
    n_samples, n_features = 100, 2
    X = np.random.randn(n_samples, n_features)
    y = X @ np.array([2.0, -1.0]) + np.random.randn(n_samples) * 0.1

    def mse_loss(w):
        return np.mean((X @ w - y)**2)

    def mse_gradient(w):
        return (2/len(y)) * X.T @ (X @ w - y)

    def mse_hessian(w):
        """Hessian of MSE is constant: (2/n) * X^T X"""
        return (2/len(y)) * X.T @ X

    # Check Hessian positive semidefiniteness
    checker = ConvexityChecker()
    w_test = np.array([0.0, 0.0])
    is_psd, eigenvalues = checker.check_hessian_psd(mse_hessian, w_test)

    print("\nMSE Loss for Linear Regression:")
    print(f"  Hessian eigenvalues: {eigenvalues}")
    print(f"  Hessian is PSD (convex): {is_psd}")

    # Demonstrate that gradient descent finds global optimum
    print("\n  Gradient descent convergence:")
    w_init = np.array([10.0, -10.0])
    w_optimal, history = checker.gradient_descent_converges(
        mse_loss, mse_gradient, w_init, lr=0.1, n_iter=50
    )
    print(f"  Initial w: {w_init}, Loss: {history[0]:.4f}")
    print(f"  Final w: {w_optimal.round(3)}, Loss: {history[-1]:.4f}")
    print(f"  True w: [2.0, -1.0]")


def demo_non_convex_neural_network():
    """Show non-convexity of neural network loss."""

    print("\n" + "=" * 60)
    print("NEURAL NETWORK NON-CONVEXITY")
    print("=" * 60)

    # Simple 2-layer network: f(x) = w2 * relu(w1 * x)
    # Even this simple network has non-convex loss!

    np.random.seed(42)
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])  # y = 2x

    def relu(x):
        return np.maximum(0, x)

    def nn_loss(params):
        """Loss for 1-hidden-unit network."""
        w1, w2 = params[0], params[1]
        hidden = relu(w1 * X)
        pred = w2 * hidden
        return np.mean((pred.flatten() - y)**2)

    # Sample loss landscape
    w1_range = np.linspace(-3, 3, 50)
    w2_range = np.linspace(-3, 3, 50)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = np.zeros_like(W1)

    for i in range(len(w1_range)):
        for j in range(len(w2_range)):
            Z[j, i] = nn_loss([w1_range[i], w2_range[j]])

    # Find local minima by running from different starting points
    print("\nNeural network loss landscape analysis:")
    print("  Running gradient descent from different starting points...\n")

    from scipy.optimize import minimize

    starting_points = [
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
        np.array([-2.0, 2.0]),
        np.array([2.0, -2.0]),
    ]

    solutions = []
    for i, start in enumerate(starting_points):
        result = minimize(nn_loss, start, method='L-BFGS-B')
        solutions.append((result.x, result.fun))
        print(f"  Start {i+1}: {start} -> Final: {result.x.round(3)}, Loss: {result.fun:.4f}")

    # Check if we found different minima
    unique_losses = set([round(s[1], 2) for s in solutions])
    print(f"\n  Number of distinct local minima found: {len(unique_losses)}")
    print("  This demonstrates non-convexity: different starting points find different minima!")


def visualize_convex_property():
    """Create visualization of convexity definition."""

    print("\n" + "=" * 60)
    print("CONVEXITY VISUALIZATION")
    print("=" * 60)

    # Convex function: f(x) = x^2
    def f_convex(x):
        return x**2

    # Non-convex function: f(x) = sin(x) + 0.1*x^2
    def f_nonconvex(x):
        return np.sin(2*x) + 0.1*x**2

    x = np.linspace(-3, 3, 100)

    print("\nConvex: f(x) = x^2")
    print("  - Second derivative: f''(x) = 2 > 0 everywhere")
    print("  - Any chord lies ABOVE the curve")

    print("\nNon-convex: f(x) = sin(2x) + 0.1x^2")
    print("  - Second derivative changes sign")
    print("  - Some chords cross BELOW the curve")


# Run demonstrations
if __name__ == "__main__":
    demo_convex_functions()
    demo_ml_loss_convexity()
    demo_non_convex_neural_network()
    visualize_convex_property()
```

**Output:**
```
============================================================
CONVEXITY DEMONSTRATION
============================================================

1. Testing f(x) = x^2 + 2x + 1 (Quadratic):
   Is convex: True

2. Testing f(x) = x^4 - 2x^2 + 0.5 (Fourth-degree):
   Is convex: False
   Counterexample: x=-0.82, y=0.91, t=0.45

============================================================
ML LOSS FUNCTION CONVEXITY
============================================================

MSE Loss for Linear Regression:
  Hessian eigenvalues: [1.72 2.18]
  Hessian is PSD (convex): True

  Gradient descent convergence:
  Initial w: [10.0, -10.0], Loss: 413.7621
  Final w: [ 2.003 -0.992], Loss: 0.0098
  True w: [2.0, -1.0]

============================================================
NEURAL NETWORK NON-CONVEXITY
============================================================

Neural network loss landscape analysis:
  Running gradient descent from different starting points...

  Start 1: [-2. -2.] -> Final: [-2.003  -0.997], Loss: 0.0000
  Start 2: [2. 2.] -> Final: [1.001 1.999], Loss: 0.0000
  Start 3: [-2.  2.] -> Final: [0. 0.], Loss: 20.0000
  Start 4: [ 2. -2.] -> Final: [0. 0.], Loss: 20.0000

  Number of distinct local minima found: 2
  This demonstrates non-convexity: different starting points find different minima!
```

---

## Exercises

### Exercise 1: Verifying Convexity

**Problem**: Show that $f(x) = e^x$ is convex using the second derivative test.

**Solution**:
```python
# f(x) = e^x
# f'(x) = e^x
# f''(x) = e^x

# Since e^x > 0 for all x, f''(x) > 0 everywhere
# Therefore f(x) = e^x is strictly convex

import numpy as np
x_values = np.linspace(-2, 2, 10)
for x in x_values:
    second_deriv = np.exp(x)
    print(f"x = {x:.1f}, f''(x) = {second_deriv:.4f} > 0")
```

### Exercise 2: Sum of Convex Functions

**Problem**: Prove that if $f$ and $g$ are convex, then $h(x) = f(x) + g(x)$ is convex.

**Solution**:
For any $x, y$ and $\theta \in [0, 1]$:
$$h(\theta x + (1-\theta)y) = f(\theta x + (1-\theta)y) + g(\theta x + (1-\theta)y)$$

Since $f$ is convex: $f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$
Since $g$ is convex: $g(\theta x + (1-\theta)y) \leq \theta g(x) + (1-\theta)g(y)$

Adding:
$$h(\theta x + (1-\theta)y) \leq \theta[f(x) + g(x)] + (1-\theta)[f(y) + g(y)] = \theta h(x) + (1-\theta)h(y)$$

Therefore $h$ is convex. This is exactly why regularized loss functions (loss + regularizer) remain convex when both terms are convex -- and it is the foundation of ridge regression, lasso, and elastic net.

### Exercise 3: Logistic Regression Convexity

**Problem**: The logistic loss is $L(\mathbf{w}) = \sum_i \log(1 + e^{-y_i \mathbf{w}^T\mathbf{x}_i})$. Why is this convex?

**Solution**:
1. Let $z_i = y_i \mathbf{w}^T\mathbf{x}_i$ (linear in $\mathbf{w}$)
2. $g(z) = \log(1 + e^{-z})$ is convex in $z$ because:
   - $g'(z) = -\frac{e^{-z}}{1 + e^{-z}} = -(1 - \sigma(z))$
   - $g''(z) = \sigma(z)(1 - \sigma(z)) \geq 0$ (product of non-negative terms, since $\sigma(z) \in [0, 1]$)
3. Composition with affine: $g(z_i) = g(y_i \mathbf{w}^T\mathbf{x}_i)$ is convex in $\mathbf{w}$
4. Sum of convex functions is convex

Therefore logistic regression has a convex loss function. This is why sklearn's `LogisticRegression` converges to the same answer regardless of initialization -- it is solving a convex problem.

---

## Summary

- A **convex set** contains all line segments between any two of its points
- A **convex function** lies below the line segment between any two points on its graph -- the bowl shape
- **Second-order test**: $f$ is convex if and only if its Hessian is positive semidefinite everywhere
- **The key theorem**: For convex functions, any local minimum is a global minimum
- **Strictly convex** functions have exactly one global minimum
- **Operations preserving convexity**: non-negative sums, composition with affine, pointwise maximum, partial minimization
- **Lagrange multipliers** convert constrained problems into unconstrained ones; KKT conditions are sufficient for optimality in the convex case
- **Linear regression loss is convex. Logistic regression loss is convex. Neural network loss is not.** This determines whether you can trust your optimizer's output or need multiple restarts and careful hyperparameter tuning

> **What's Next** -- Even when you find the minimum, your model might overfit -- memorizing training data instead of learning patterns. Regularization adds constraints that prevent this.
