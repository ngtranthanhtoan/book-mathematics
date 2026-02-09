# Chapter 5: Inequalities

## Building On

Equations ask "when are things equal?" Inequalities ask "when is one thing bigger or smaller?" In ML, constraints on parameters, outputs, and margins are all inequalities. You spent the last four chapters mastering the language of equality. Now you learn the language of bounds, limits, and constraints --- the language your optimizer actually speaks.

---

Your SVM needs a margin of at least 1. Your learning rate must be positive. Your regularization strength must be non-negative. Constraints are everywhere in ML --- and constraints are expressed as inequalities.

Think about the first thing you do when a training run explodes. You check bounds. Is the learning rate too large? Did the gradients exceed a threshold? Are the probabilities still between 0 and 1? Every one of those checks is an inequality in disguise.

Here is the inequality that will run through this entire chapter --- the SVM margin constraint:

$$y_i(w \cdot x_i + b) \geq 1 \quad \text{for all } i$$

This single inequality defines the SVM optimization problem. Every training point must be on the correct side of the decision boundary, and not just barely --- it must clear the margin by at least 1. By the end of this chapter, you will understand every piece of that statement.

---

## You Already Know This

You write inequalities in code every day. You just do not call them that.

```python
# These are all inequalities
assert learning_rate > 0, "Learning rate must be positive"
assert 0 <= probability <= 1, "Probability must be in [0, 1]"
assert batch_size >= 1, "Batch size must be at least 1"
assert weight_decay >= 0, "Weight decay must be non-negative"

# Interval notation = value ranges in your config
config = {
    "learning_rate": {"min": 1e-6, "max": 1.0},       # (1e-6, 1.0]
    "dropout":       {"min": 0.0, "max": 1.0},         # [0.0, 1.0)
    "temperature":   {"min": 0.0, "max": float("inf")} # (0, inf)
}

# Feasible region = valid parameter space
# Just like an API that validates input ranges
def validate_hyperparameters(lr, dropout, epochs):
    """Every constraint here is an inequality."""
    assert 0 < lr <= 1.0
    assert 0.0 <= dropout < 1.0
    assert epochs >= 1
```

That `validate_hyperparameters` function? It defines a **feasible region** --- the set of all valid hyperparameter combinations. Any point inside the region is a legal configuration. Any point outside it crashes your training run (or worse, silently produces garbage).

---

## The Problem: What Constraints Does an ML Model Have?

Before we get into rules and notation, let us enumerate the constraints you actually encounter:

| Constraint | Inequality | Where It Shows Up |
|-----------|-----------|-------------------|
| Probabilities are bounded | $0 \leq p \leq 1$ | Softmax output, sigmoid output |
| Learning rate is positive | $\alpha > 0$ | Every optimizer |
| SVM margin | $y_i(w \cdot x_i + b) \geq 1$ | Support Vector Machines |
| ReLU activation | $x \geq 0$ determines output | Every deep network |
| Weight clipping | $\|w_i\| \leq c$ | WGAN, gradient clipping |
| Regularization | $\|\theta\|_p \leq t$ | L1/L2 regularization |
| KL divergence | $D_{KL}(P \| Q) \geq 0$ | VAEs, information theory |
| Triangle inequality | $d(x, z) \leq d(x, y) + d(y, z)$ | Metric learning, embeddings |

Every single one of these is an inequality. To work with them, you need the rules.

---

## Inequality Notation and Symbols

| Symbol | Meaning | Example | On a Number Line |
|--------|---------|---------|------------------|
| $<$ | Strictly less than | $x < 3$ | Open circle at 3, arrow left |
| $\leq$ | Less than or equal | $x \leq 3$ | Filled circle at 3, arrow left |
| $>$ | Strictly greater than | $x > 3$ | Open circle at 3, arrow right |
| $\geq$ | Greater than or equal | $x \geq 3$ | Filled circle at 3, arrow right |

Here is $x > -2$ on a number line:

```
  ──────────────┬───────────────────────────>
 -5  -4  -3  -2  -1   0   1   2   3   4   5
                 (
          xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx>>>

  ( = open circle (does NOT include -2)
  x = all values in the solution set
```

And $x \leq 3$:

```
  <─────────────────────────────┬───────────
 -5  -4  -3  -2  -1   0   1   2   3   4   5
                                ]
  <<<xxxxxxxxxxxxxxxxxxxxxxxxxxx

  ] = closed circle (INCLUDES 3)
  x = all values in the solution set
```

---

## Interval Notation

You already use this concept when defining value ranges in configs. Here is the formal notation:

| Notation | Meaning | Code Equivalent |
|----------|---------|----------------|
| $(a, b)$ | $a < x < b$ | `a < x < b` (exclusive both ends) |
| $[a, b]$ | $a \leq x \leq b$ | `a <= x <= b` (inclusive both ends) |
| $[a, b)$ | $a \leq x < b$ | `a <= x < b` (inclusive left, exclusive right) |
| $(a, b]$ | $a < x \leq b$ | `a < x <= b` (exclusive left, inclusive right) |
| $(-\infty, b)$ | $x < b$ | `x < b` (unbounded left) |
| $(a, \infty)$ | $x > a$ | `x > a` (unbounded right) |

**Key insight**: Parentheses `()` mean "does not include the endpoint" --- like a strict inequality. Brackets `[]` mean "includes the endpoint" --- like a non-strict inequality. Infinity always gets parentheses because you can never actually reach it.

```
  Interval: [1, 5)

  ──────┬───────────────────────┬───────────
  -1  0  1   2   3   4   5   6   7   8
        [                       )
        xxxxxxxxxxxxxxxxxxxxxxx

  Includes 1, does NOT include 5
  Valid learning rate range? config["lr"] = {"min": 1e-4, "max": 1e-1}
```

---

## Solving Linear Inequalities

Solving inequalities works like solving equations, with one critical rule.

### The Critical Rule

> **When you multiply or divide both sides by a negative number, you FLIP the inequality sign.**

This is the rule that catches everyone at least once. Here is why it works: if $3 > 2$, then $-3 < -2$. Negation reverses order.

**Example**: Solve $-2x + 5 > 11$

**Step 1**: Subtract 5 from both sides (no sign flip --- subtraction is fine):
$$-2x > 6$$

**Step 2**: Divide by $-2$ (FLIP the sign --- dividing by a negative):
$$x < -3$$

**Solution**: $(-\infty, -3)$

```
  ──────────────────────┬───────────────────>
 -6  -5  -4  -3  -2  -1   0   1   2   3
                        (
  <<<xxxxxxxxxxxxxxxxxx

  All values strictly less than -3
```

### Common Mistake

```python
# WRONG: Forgot to flip the sign
# Solving: -2x > 6
# x > -3   <-- WRONG!

# RIGHT: Divided by negative, so flip
# x < -3   <-- CORRECT!
```

Think of it this way: in Python, you would never confuse `-1 * 5 > -1 * 3` with `5 > 3`. Multiplying by $-1$ reverses everything.

---

## Compound Inequalities

### AND (Intersection) --- Both Conditions Must Hold

$$1 < x \leq 5$$

This means $x > 1$ AND $x \leq 5$. The solution is the **intersection**: $(1, 5]$.

```
  ──────────┬───────────────────┬───────────
 -1   0   1   2   3   4   5   6   7
            (                   ]
            xxxxxxxxxxxxxxxxxxxxx

  x must satisfy BOTH conditions
```

**ML example**: Your dropout rate must satisfy $0 \leq d < 1$. It is non-negative AND strictly less than 1. The interval $[0, 1)$ is the valid range.

### OR (Union) --- At Least One Condition Holds

$$x < -2 \quad \text{OR} \quad x > 3$$

The solution is the **union**: $(-\infty, -2) \cup (3, \infty)$.

```
  ──────────────┬───────────────────┬───────>
 -5  -4  -3  -2  -1   0   1   2   3   4   5
  <<<xxxxxxxxxx(                    (xxx>>>

  x must satisfy AT LEAST ONE condition
```

**ML example**: An anomaly detector flags data points where the z-score is $|z| > 3$, meaning $z < -3$ or $z > 3$.

---

## Absolute Value Inequalities

Two rules to memorize:

$$|x| < a \iff -a < x < a$$
$$|x| > a \iff x < -a \text{ OR } x > a$$

The first says "distance from zero is less than $a$" --- you stay inside the interval. The second says "distance from zero is greater than $a$" --- you go outside the interval.

**Example**: Solve $|2x - 3| \leq 5$

$$-5 \leq 2x - 3 \leq 5$$
$$-2 \leq 2x \leq 8$$
$$-1 \leq x \leq 4$$

**Solution**: $[-1, 4]$

```
  ──────────┬───────────────────────┬───────>
 -3  -2  -1   0   1   2   3   4   5   6
            [                       ]
            xxxxxxxxxxxxxxxxxxxxxxxxx

  All x within distance 5 of 3 (after scaling by 2)
```

**ML connection**: Gradient clipping is an absolute value inequality. When $\|g\| > \text{threshold}$, you rescale the gradient. The constraint $\|g\| \leq \text{threshold}$ keeps your gradients inside a bounded region.

---

## Quadratic Inequalities

When your inequality involves $x^2$, factor first, then test intervals.

**Example**: Solve $x^2 - 5x + 6 < 0$

**Step 1**: Factor.
$$(x - 2)(x - 3) < 0$$

**Step 2**: Find roots: $x = 2$ and $x = 3$.

**Step 3**: The roots divide the number line into three intervals. Test a point from each:

| Interval | Test Point | $(x-2)$ | $(x-3)$ | Product | Satisfies $< 0$? |
|----------|-----------|---------|---------|---------|-------------------|
| $x < 2$ | $x = 0$ | $-$ | $-$ | $+$ | No |
| $2 < x < 3$ | $x = 2.5$ | $+$ | $-$ | $-$ | Yes |
| $x > 3$ | $x = 4$ | $+$ | $+$ | $+$ | No |

**Solution**: $(2, 3)$

```
  ──────────────────┬───────┬───────────────>
 -1   0   1   2   2.5   3   4   5
                    (       )
                    xxxxxxxxx

  Product is negative only between the roots
```

---

## Systems of Inequalities and Feasible Regions

This is where things get directly relevant to ML. A **system** of inequalities is a set of constraints that must all hold simultaneously.

**Example**:
$$\begin{cases}
x + y \leq 10 \\
x \geq 0 \\
y \geq 0 \\
2x + y \leq 14
\end{cases}$$

The solution is the **feasible region** --- the set of all $(x, y)$ that satisfy ALL constraints.

```
  y
  10 +.
     | '.
   8 +   '.
     |     '.
   6 +   *---+. (4, 6)
     |   |    '.
   4 +   |      '.
     |   |   Feasible
   2 +   |   Region    '.
     |   |               '.
   0 +---+--------+--------+----> x
     0   2    4    6    7   10

  Boundaries:
    x = 0       (left edge)
    y = 0       (bottom edge)
    x + y = 10  (diagonal, top-right)
    2x + y = 14 (steeper diagonal)

  Vertices: (0,0), (0,10), (4,6), (7,0)
```

Think of this as the "valid hyperparameter space." Every point inside the region is a legal configuration. The optimizer searches within this region for the best solution.

### Finding Vertices

Vertices occur where constraint boundaries intersect. To find them, solve pairs of equations:

1. $x = 0$ and $y = 0$ gives $(0, 0)$
2. $x = 0$ and $x + y = 10$ gives $(0, 10)$
3. $x + y = 10$ and $2x + y = 14$: subtract to get $x = 4$, so $y = 6$ gives $(4, 6)$
4. $y = 0$ and $2x + y = 14$ gives $(7, 0)$

Why do vertices matter? In linear programming, the optimal solution always occurs at a vertex. Your optimizer does not need to search the entire region --- just the corners.

---

## Linear Programming

**Linear programming** optimizes a linear objective function subject to linear inequality constraints.

**Standard form**:
$$\text{Maximize } c_1 x_1 + c_2 x_2 + \ldots + c_n x_n$$
$$\text{Subject to: } a_{i1}x_1 + a_{i2}x_2 + \ldots + a_{in}x_n \leq b_i$$

**Key theorem**: If an optimal solution exists, it occurs at a vertex of the feasible region.

This principle extends far beyond textbook examples. Feature selection with budget constraints, optimal transport problems, and sparse optimization all reduce to linear programs under the hood.

---

## The Running Example: SVM Margin Constraint

Let us return to the inequality we started with:

$$y_i(w \cdot x_i + b) \geq 1 \quad \text{for all } i$$

Now you have the tools to understand it:

- $y_i \in \{-1, +1\}$ is the class label (positive or negative class)
- $w \cdot x_i + b$ is the decision function (a linear equation from Chapter 1)
- The product $y_i(w \cdot x_i + b)$ is positive when the point is correctly classified
- The $\geq 1$ part is the margin constraint --- not just correct, but correct by at least 1 unit

This is a **system of inequalities** --- one constraint per training point. The feasible region is the set of all $(w, b)$ that correctly classify every point with sufficient margin. The SVM optimization problem is:

$$\text{Minimize } \frac{1}{2}\|w\|^2 \quad \text{subject to } y_i(w \cdot x_i + b) \geq 1 \quad \forall i$$

You are minimizing a quadratic objective (from Chapter 3) subject to linear inequality constraints (this chapter). The entire SVM is an exercise in inequalities.

```
  Margin constraint visualized (1D slice):

  Negative class          |  margin  |          Positive class
  y_i = -1                | >= 1     |          y_i = +1
                           |          |
  <-------- w*x+b < -1 ---|--- 0 ----|--- w*x+b > 1 -------->
                           |          |
                    decision boundary
                      (w*x + b = 0)

  Points MUST be outside the margin to satisfy the constraint.
  Points inside the margin violate y_i(w*x_i + b) >= 1.
```

---

## The Triangle Inequality

One more inequality that shows up constantly in ML:

$$d(x, z) \leq d(x, y) + d(y, z)$$

This says: "the direct distance is never more than a detour." It is the foundation of **metric spaces** --- any valid distance function must satisfy it.

Why do you care? Because distance functions are everywhere in ML:

- **k-NN** relies on distances being well-behaved
- **Metric learning** (e.g., triplet loss) explicitly enforces triangle-inequality-compatible embeddings
- **Clustering algorithms** assume the triangle inequality holds for their distance metric

If your "distance function" violates the triangle inequality, it is not a real metric, and algorithms that assume it is one will produce unreliable results.

---

## ML Applications in Depth

### ReLU: An Inequality-Defined Activation

The most common activation function in deep learning is defined entirely by an inequality:

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

The inequality $x \geq 0$ splits the input space into two regions: one where the neuron is "active" (passes the signal through) and one where it is "dead" (outputs zero). The derivative is also inequality-defined:

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$$

### Regularization as Inequality Constraints

L1 and L2 regularization are typically written as penalty terms added to the loss. But they are mathematically equivalent to inequality constraints:

$$\min \mathcal{L}(\theta) + \lambda\|\theta\|_p \quad \Longleftrightarrow \quad \min \mathcal{L}(\theta) \quad \text{subject to} \quad \|\theta\|_p \leq t$$

For every $\lambda > 0$, there exists a $t > 0$ that makes these two formulations equivalent. The constraint form makes the geometry clear: you are restricting $\theta$ to lie within a ball of radius $t$.

### Constrained Optimization (General Form)

Most ML training involves implicit or explicit constraints:

$$\min_{\theta} \mathcal{L}(\theta) \quad \text{subject to} \quad g_i(\theta) \leq 0$$

Examples:
- **Weight clipping**: $|w_i| \leq c$ (used in WGANs)
- **Probability constraints**: $0 \leq p \leq 1$ (softmax outputs)
- **Norm constraints**: $\|\theta\|_2 \leq \lambda$ (weight decay)

### Gradient Clipping

When gradients explode during training, you clip them. This is an inequality check followed by projection:

$$g_{\text{clipped}} = \begin{cases} g & \text{if } \|g\| \leq \text{threshold} \\ \text{threshold} \cdot \frac{g}{\|g\|} & \text{otherwise} \end{cases}$$

In code, this is `torch.nn.utils.clip_grad_norm_`. The constraint $\|g\| \leq \text{threshold}$ defines the feasible region for gradients.

---

## Code: Inequalities in Practice

```python
import numpy as np
from scipy.optimize import linprog
from typing import List, Tuple

# ── 1. The inequalities you already write ──────────────────────

def validate_training_config(lr: float, dropout: float,
                              weight_decay: float, temperature: float):
    """
    Every assert here is an inequality.
    This function defines a feasible region in hyperparameter space.
    """
    assert lr > 0,              f"learning_rate > 0, got {lr}"
    assert dropout >= 0,        f"dropout >= 0, got {dropout}"
    assert dropout < 1,         f"dropout < 1, got {dropout}"
    assert weight_decay >= 0,   f"weight_decay >= 0, got {weight_decay}"
    assert temperature > 0,     f"temperature > 0, got {temperature}"
    print("All constraints satisfied. Config is in the feasible region.")

validate_training_config(lr=0.001, dropout=0.5, weight_decay=1e-4, temperature=1.0)


# ── 2. Checking if a point satisfies a system of inequalities ──

def is_feasible(point: np.ndarray, A: np.ndarray, b: np.ndarray) -> bool:
    """
    Check if point satisfies ALL constraints Ax <= b.

    This is the core question in constrained optimization:
    is this point inside the feasible region?
    """
    return bool(np.all(A @ point <= b))

# System: x >= 0, y >= 0, x + y <= 10, 2x + y <= 14
# Rewritten in standard form Ax <= b:
A = np.array([
    [-1,  0],   # -x  <= 0   (i.e., x >= 0)
    [ 0, -1],   # -y  <= 0   (i.e., y >= 0)
    [ 1,  1],   #  x + y <= 10
    [ 2,  1],   # 2x + y <= 14
])
b = np.array([0, 0, 10, 14])

print("\n=== Feasibility Check ===")
test_points = [
    (np.array([3, 4]),  "Should be feasible"),
    (np.array([5, 5]),  "Should be feasible"),
    (np.array([8, 4]),  "Violates 2x + y <= 14"),
    (np.array([-1, 3]), "Violates x >= 0"),
]
for point, description in test_points:
    result = "Feasible" if is_feasible(point, A, b) else "NOT feasible"
    print(f"  ({point[0]:2.0f}, {point[1]:2.0f}): {result:14s} | {description}")


# ── 3. Finding vertices of the feasible region ─────────────────

def find_vertices_2d(A: np.ndarray, b: np.ndarray) -> List[np.ndarray]:
    """
    Vertices occur where constraint boundaries intersect.
    In linear programming, the optimum is always at a vertex.
    """
    n_constraints = len(b)
    vertices = []

    for i in range(n_constraints):
        for j in range(i + 1, n_constraints):
            A_pair = np.array([A[i], A[j]])
            b_pair = np.array([b[i], b[j]])
            try:
                if np.linalg.matrix_rank(A_pair) == 2:
                    vertex = np.linalg.solve(A_pair, b_pair)
                    if is_feasible(vertex, A, b + 1e-10):
                        vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue
    return vertices

vertices = find_vertices_2d(A, b)
print("\n=== Vertices of Feasible Region ===")
for v in vertices:
    print(f"  ({v[0]:.1f}, {v[1]:.1f})")


# ── 4. Linear programming with scipy ──────────────────────────

def solve_linear_program():
    """
    Maximize: 3x + 2y
    Subject to: x >= 0, y >= 0, x + y <= 10, 2x + y <= 14

    scipy.linprog minimizes, so negate coefficients.
    """
    c = [-3, -2]  # Negate to convert max -> min
    A_ub = np.array([[1, 1], [2, 1]])
    b_ub = np.array([10, 14])
    bounds = [(0, None), (0, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return result

result = solve_linear_program()
print("\n=== Linear Programming ===")
print(f"  Optimal point: ({result.x[0]:.1f}, {result.x[1]:.1f})")
print(f"  Maximum value of 3x + 2y: {-result.fun:.1f}")


# ── 5. Solving inequalities programmatically ──────────────────

def solve_linear_inequality(a: float, b: float, c: float,
                             inequality: str) -> str:
    """
    Solve ax + b (inequality) c for x.
    Returns solution in interval notation.

    The key: if a < 0, dividing flips the inequality sign.
    """
    rhs = c - b

    if a == 0:
        return "All real numbers" if eval(f"0 {inequality} {rhs}") else "No solution"

    boundary = rhs / a
    flip = a < 0

    if inequality in ["<", "<="]:
        op = (">=" if inequality == "<=" else ">") if flip else inequality
    else:
        op = ("<=" if inequality == ">=" else "<") if flip else inequality

    formats = {
        "<":  f"(-inf, {boundary})",
        "<=": f"(-inf, {boundary}]",
        ">":  f"({boundary}, inf)",
        ">=": f"[{boundary}, inf)",
    }
    return formats[op]

print("\n=== Solving 1D Inequalities ===")
print(f"  3x + 2 < 11  =>  x in {solve_linear_inequality(3, 2, 11, '<')}")
print(f"  -2x + 5 >= 1 =>  x in {solve_linear_inequality(-2, 5, 1, '>=')}")
print(f"  -2x + 5 > 11 =>  x in {solve_linear_inequality(-2, 5, 11, '>')}")


# ── 6. Quadratic inequality solver ────────────────────────────

def solve_quadratic_inequality(a: float, b: float, c: float,
                                inequality: str) -> str:
    """
    Solve ax^2 + bx + c (inequality) 0.
    Factor, find roots, test intervals.
    """
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        if a > 0:
            return "No solution" if inequality in ["<", "<="] else "All real numbers"
        else:
            return "All real numbers" if inequality in ["<", "<="] else "No solution"

    sqrt_d = np.sqrt(discriminant)
    x1 = (-b - sqrt_d) / (2 * a)
    x2 = (-b + sqrt_d) / (2 * a)
    if x1 > x2:
        x1, x2 = x2, x1

    if a > 0:  # Parabola opens up: negative between roots
        templates = {
            "<":  f"({x1:.4f}, {x2:.4f})",
            "<=": f"[{x1:.4f}, {x2:.4f}]",
            ">":  f"(-inf, {x1:.4f}) U ({x2:.4f}, inf)",
            ">=": f"(-inf, {x1:.4f}] U [{x2:.4f}, inf)",
        }
    else:  # Parabola opens down: positive between roots
        templates = {
            "<":  f"(-inf, {x1:.4f}) U ({x2:.4f}, inf)",
            "<=": f"(-inf, {x1:.4f}] U [{x2:.4f}, inf)",
            ">":  f"({x1:.4f}, {x2:.4f})",
            ">=": f"[{x1:.4f}, {x2:.4f}]",
        }
    return templates[inequality]

print("\n=== Quadratic Inequalities ===")
print(f"  x^2 - 5x + 6 < 0 =>  x in {solve_quadratic_inequality(1, -5, 6, '<')}")
print(f"  x^2 - 4 >= 0     =>  x in {solve_quadratic_inequality(1, 0, -4, '>=')}")


# ── 7. ReLU: the inequality-defined activation ────────────────

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU(x) = max(0, x). The inequality x >= 0 determines the output."""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """ReLU'(x) = 1 if x > 0, else 0. Another inequality."""
    return (x > 0).astype(float)

print("\n=== ReLU (Inequality-Based Activation) ===")
x_test = np.array([-2, -1, 0, 1, 2])
print(f"  Input:      {x_test}")
print(f"  ReLU:       {relu(x_test)}")
print(f"  Derivative: {relu_derivative(x_test)}")


# ── 8. Constraint projection (gradient clipping, weight clipping) ──

def project_onto_box(x: np.ndarray, lower: np.ndarray,
                      upper: np.ndarray) -> np.ndarray:
    """
    Project x onto box constraints: lower <= x <= upper.
    This is what np.clip does -- and it is the simplest form
    of projecting onto a feasible region.
    """
    return np.clip(x, lower, upper)

print("\n=== Constraint Projection (Box Constraints) ===")
x = np.array([1.5, -0.5, 2.5, 0.3])
lower = np.array([0, 0, 0, 0])
upper = np.array([1, 1, 1, 1])
projected = project_onto_box(x, lower, upper)
print(f"  Original:  {x}")
print(f"  Bounds:    [{lower} , {upper}]")
print(f"  Projected: {projected}")
print(f"  (Values clipped to satisfy 0 <= x_i <= 1)")


# ── 9. SVM margin check ───────────────────────────────────────

def check_svm_margin(X: np.ndarray, y: np.ndarray,
                      w: np.ndarray, b: float) -> dict:
    """
    Check the SVM constraint: y_i(w . x_i + b) >= 1 for all i.
    This is our running example inequality.
    """
    margins = y * (X @ w + b)
    return {
        "margins": margins,
        "all_satisfied": bool(np.all(margins >= 1)),
        "min_margin": float(np.min(margins)),
        "violations": int(np.sum(margins < 1)),
    }

print("\n=== SVM Margin Constraint: y_i(w*x_i + b) >= 1 ===")
X = np.array([[2, 3], [1, 1], [3, 2], [0, 1]])
y = np.array([1, -1, 1, -1])
w = np.array([1.0, 0.5])
b = -2.0

result = check_svm_margin(X, y, w, b)
for i in range(len(X)):
    status = "OK" if result["margins"][i] >= 1 else "VIOLATED"
    print(f"  Point {X[i]}, label {y[i]:+d}: "
          f"margin = {result['margins'][i]:+.2f}  [{status}]")
print(f"  All constraints satisfied: {result['all_satisfied']}")
print(f"  Minimum margin: {result['min_margin']:.2f}")


# ── 10. Softmax output constraints ────────────────────────────

def check_softmax_constraints(output: np.ndarray,
                               tolerance: float = 1e-6) -> dict:
    """
    Softmax output must satisfy three inequalities:
      1. p_i >= 0  for all i   (non-negativity)
      2. p_i <= 1  for all i   (bounded above)
      3. sum(p_i) = 1           (normalization -- technically an equality)
    """
    results = {
        "all_non_negative": bool(np.all(output >= -tolerance)),
        "all_at_most_one":  bool(np.all(output <= 1 + tolerance)),
        "sum_equals_one":   bool(np.abs(np.sum(output) - 1) < tolerance),
    }
    results["all_satisfied"] = all(results.values())
    return results

print("\n=== Softmax Output Constraints ===")
valid = np.array([0.7, 0.2, 0.1])
invalid = np.array([0.5, 0.3, 0.3])  # Sum = 1.1
print(f"  {valid} -> {check_softmax_constraints(valid)}")
print(f"  {invalid} -> {check_softmax_constraints(invalid)}")
```

---

## Common Mistakes

1. **Forgetting to flip the sign when dividing by a negative.** This is the single most common mistake in inequality manipulation. Multiplying both sides of $-x > 3$ by $-1$ gives $x < -3$, not $x > -3$.

2. **Confusing strict and non-strict inequalities.** $x < 3$ does not include 3. $x \leq 3$ does. In code: `range(n)` goes up to but does not include $n$. Same idea.

3. **Empty feasible regions.** If your constraints are contradictory, no solution exists. For example, $x > 5$ AND $x < 3$ has no solution. In ML, overly aggressive constraints can make optimization infeasible.

4. **Unbounded regions.** Some constraint sets do not limit solutions sufficiently. Without regularization ($\|\theta\| \leq t$), your parameter space is unbounded and the optimizer may diverge.

5. **Numerical precision at boundaries.** Is $x = 0.9999999$ less than 1? In exact math, yes. In floating point, it depends on your tolerance. Always use an epsilon: `abs(x - boundary) < 1e-8`.

---

## Exercises

### Exercise 1: Solve the Inequality
Solve: $4 - 3x \geq 10$

**Solution**:
$$4 - 3x \geq 10$$
$$-3x \geq 6$$
$$x \leq -2 \quad \text{(flip the sign --- dividing by } -3\text{)}$$

Answer: $(-\infty, -2]$

### Exercise 2: Find the Feasible Region
For the constraints:
- $x \geq 0$
- $y \geq 0$
- $x + 2y \leq 8$
- $3x + y \leq 9$

Find all corner points (vertices) of the feasible region.

**Solution**:
1. $x = 0$ and $y = 0$: $(0, 0)$
2. $x = 0$ and $x + 2y = 8$: $(0, 4)$
3. $y = 0$ and $3x + y = 9$: $(3, 0)$
4. $x + 2y = 8$ and $3x + y = 9$:
   - From the first: $x = 8 - 2y$
   - Substitute: $3(8 - 2y) + y = 9 \Rightarrow 24 - 6y + y = 9 \Rightarrow y = 3, x = 2$
   - Point: $(2, 3)$

Vertices: $(0, 0)$, $(0, 4)$, $(3, 0)$, $(2, 3)$

### Exercise 3: SVM Margin Check
Given data points $x_1 = [2, 1]$ with $y_1 = +1$ and $x_2 = [0, -1]$ with $y_2 = -1$, and a candidate hyperplane $w = [1, 1], b = -1$:

(a) Compute $y_i(w \cdot x_i + b)$ for each point.
(b) Does this hyperplane satisfy the margin constraint $y_i(w \cdot x_i + b) \geq 1$?

**Solution**:
(a)
- Point 1: $y_1(w \cdot x_1 + b) = (+1)([1,1] \cdot [2,1] - 1) = (+1)(3 - 1) = 2$
- Point 2: $y_2(w \cdot x_2 + b) = (-1)([1,1] \cdot [0,-1] - 1) = (-1)(-1 - 1) = 2$

(b) Both margins are $2 \geq 1$. Yes, the constraint is satisfied for both points.

### Exercise 4: Implement Constraint Checking
Write a function that checks if a neural network output satisfies softmax constraints.

**Solution**:
```python
import numpy as np

def check_softmax_constraints(output: np.ndarray, tolerance: float = 1e-6) -> dict:
    """
    Check if output satisfies softmax constraints:
    1. All values >= 0       (inequality)
    2. All values <= 1       (inequality)
    3. Sum equals 1          (equality)
    """
    results = {
        'all_non_negative': bool(np.all(output >= -tolerance)),
        'all_at_most_one':  bool(np.all(output <= 1 + tolerance)),
        'sum_equals_one':   bool(np.abs(np.sum(output) - 1) < tolerance),
    }
    results['all_satisfied'] = all(results.values())
    return results

# Test
valid_output = np.array([0.7, 0.2, 0.1])
invalid_output = np.array([0.5, 0.3, 0.3])  # Sum = 1.1

print("Valid output:", check_softmax_constraints(valid_output))
print("Invalid output:", check_softmax_constraints(invalid_output))
```

---

## Summary

- **Inequalities** compare expressions using $<, \leq, >, \geq$ --- they define regions, not points.

- **The critical rule**: Flip the inequality sign when multiplying or dividing by a negative number. Period.

- **Interval notation** maps directly to value ranges in your configs: $(a, b)$ means exclusive, $[a, b]$ means inclusive. You already use this concept in `min_value <= x <= max_value`.

- **Compound inequalities**: AND gives you intersections (both must hold), OR gives you unions (at least one holds).

- **Absolute value inequalities**: $|x| < a$ means stay inside $(-a, a)$. $|x| > a$ means go outside.

- **Feasible regions**: Systems of inequalities carve out the valid parameter space. Every point inside is a legal solution. This is exactly how constrained optimization works in ML.

- **Linear programming**: Optimize over feasible regions. The optimum is always at a vertex.

- **The SVM margin constraint** $y_i(w \cdot x_i + b) \geq 1$ ties together everything in this chapter: linear inequalities, systems of constraints, and feasible regions. It is the single inequality that defines how SVMs work.

- **The triangle inequality** $d(x, z) \leq d(x, y) + d(y, z)$ is the foundation of metric spaces. Any valid distance function must satisfy it.

---

## What's Next

Level 2 is complete. You have gone from variables through equations, polynomials, exponentials, and constraints. Now Level 3: functions --- the core abstraction of ML. Every model IS a function. You will learn how to describe, compose, and transform functions, which is precisely what a neural network does at every layer.

---

[Back to Level 2 Overview](./README.md)
