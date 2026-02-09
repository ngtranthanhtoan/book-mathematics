# Chapter 2: Linear Equations

> **Building On** -- Expressions describe relationships. Equations add an equals sign and ask: *for what values is this true?* Solving equations is the core skill of algebra -- and it's what your model does every training step.

## The Problem That Starts Everything

Your linear regression model predicts `y = 3.2x + 1.7`. When does the prediction equal the actual value? When `3.2x + 1.7 = y_actual`. Solving that equation gives you the breakeven point. Linear equations are the simplest case of "find the parameters" -- and the foundation for all of ML.

Here is the running example we will carry through this entire chapter:

**Linear regression's normal equation**: $X^T X \, w = X^T y$. This is a system of linear equations, and solving it gives you the optimal weights. Every concept below -- isolating a variable, substitution, elimination, matrix form -- feeds directly into understanding that single line of math.

## What Is a Linear Equation?

A linear equation is a statement where variables appear only to the first power. No squares, no cubes, no roots of variables. Think of it as a constraint: it narrows the universe of possible values down to the ones that make the statement true.

**Single variable** -- the standard form:

$$ax + b = c \quad \text{where } a \neq 0$$

**Two variables** -- standard form and slope-intercept form:

$$ax + by = c \qquad \text{or equivalently} \qquad y = mx + b$$

where $m$ is the slope (rate of change) and $b$ is the y-intercept (the output when input is zero).

> **You Already Know This** -- Solving an equation is *root finding*, like binary search for where `f(x) = 0`. You are looking for the input that zeroes out the residual. In code: you have `f(x) = ax + b - c` and you want `f(x) == 0`. The algebra just gives you the closed-form answer instead of iterating.

## Solving Single-Variable Equations

The goal is always the same: isolate the variable on one side. Every operation you apply to one side, you apply to the other. The equation stays balanced.

**Example**: Solve $3x + 7 = 22$

**Step 1** -- Subtract 7 from both sides:

$$3x + 7 - 7 = 22 - 7$$
$$3x = 15$$

**Step 2** -- Divide both sides by 3:

$$\frac{3x}{3} = \frac{15}{3}$$
$$x = 5$$

**Verification**: $3(5) + 7 = 15 + 7 = 22$ -- correct.

That is the entire algorithm: undo addition/subtraction first, then undo multiplication/division. You peel back operations in reverse order, exactly like unwinding a call stack.

### Visualizing on a Number Line

A single-variable linear equation pins down exactly one point:

$$2x + 3 = 7 \implies x = 2$$

```
  Number Line
  ----+----+----+----+----+----+---->
      0    1   [2]   3    4    5
                ^
            solution
```

## From One Equation to Two: Systems

One equation with one unknown gives you a point. One equation with *two* unknowns gives you a line -- infinitely many $(x, y)$ pairs that satisfy it. To pin down a single point, you need a second equation: a second constraint.

This is a **system of linear equations**:

$$\begin{cases} 2x + y = 5 \\ x - y = 1 \end{cases}$$

Geometrically, each equation is a line. The solution is where the lines cross.

```
    y
    |
  5 +                         .  (Line 1: y = -2x + 5)
    |                      .
  4 +                   .
    |          .     .
  3 +       .     x
    |    .  /  .              (Line 2: y = x - 1)
  2 +  . /
    | /
  1 + (2,1) <-- solution
    |/
  0 +--+--+--+--+--+--+----> x
    0  1  2  3  4  5
    |
 -1 +
```

The intersection point $(2, 1)$ is the unique pair that satisfies *both* equations simultaneously.

> **You Already Know This** -- A system of equations is just *solving constraints*, like a dependency resolver. Each equation eliminates one degree of freedom. Two equations with two unknowns: fully determined (usually). Think of it like type constraints narrowing down the set of valid programs.

### Three Possible Outcomes

A system of two equations with two unknowns can have:

| Outcome | Geometry | Example |
|---|---|---|
| **One solution** | Lines intersect at a point | $y = 2x + 1$ and $y = -x + 4$ |
| **No solution** | Lines are parallel | $y = 2x + 1$ and $y = 2x + 3$ |
| **Infinite solutions** | Lines are identical (coincident) | $y = 2x + 1$ and $2y = 4x + 2$ |

```
  ONE SOLUTION         NO SOLUTION          INFINITE SOLUTIONS

    \  /                  /   /               /
     \/                  /   /               /  (lines overlap
     /\                 /   /               /    completely)
    /  \               /   /               /

  Lines cross        Lines parallel       Same line
```

> **Common Mistake** -- A system of 2 equations with 2 unknowns *usually* has exactly 1 solution. But parallel lines have 0 solutions, and coincident lines have infinitely many. Always check before assuming a unique answer exists. In code terms: `np.linalg.solve` will throw a `LinAlgError` for singular systems.

## Method 1: Substitution

**Idea**: Solve one equation for a variable, then plug that expression into the other equation.

> **You Already Know This** -- Substitution is *variable replacement*, like inlining a function. You take `x = f(y)`, then everywhere you see `x` in the other equation, you replace it with `f(y)`. Same value, fewer unknowns.

**Example**: Solve the system:

$$\begin{cases} 2x + y = 5 \\ x - y = 1 \end{cases}$$

**Step 1** -- Solve equation 2 for $x$:

$$x = y + 1$$

**Step 2** -- Substitute into equation 1:

$$2(y + 1) + y = 5$$
$$2y + 2 + y = 5$$
$$3y = 3$$
$$y = 1$$

**Step 3** -- Back-substitute to find $x$:

$$x = 1 + 1 = 2$$

**Solution**: $(x, y) = (2, 1)$

This is exactly like inlining. You had two functions of two variables. You inlined one into the other, reducing the problem to one function of one variable.

## Method 2: Elimination

**Idea**: Add or subtract equations so that one variable cancels out.

> **You Already Know This** -- Elimination is *reducing variables*, like simplifying a boolean expression. You combine constraints to remove a term, leaving a simpler problem. If substitution is inlining, elimination is algebraic dead-code elimination.

**Example** (same system):

$$\begin{cases} 2x + y = 5 \\ x - y = 1 \end{cases}$$

**Step 1** -- Add the two equations (the $y$ terms cancel):

$$(2x + y) + (x - y) = 5 + 1$$
$$3x = 6$$
$$x = 2$$

**Step 2** -- Substitute back into either equation:

$$2(2) + y = 5$$
$$y = 1$$

**Solution**: $(x, y) = (2, 1)$

Both methods give the same answer. Substitution is often easier when one variable has coefficient 1. Elimination is often easier when coefficients line up for clean cancellation.

## Generalizing: The Matrix Form

Here is where it gets powerful. Write the system as a single matrix equation:

$$A\mathbf{x} = \mathbf{b}$$

For our running example:

$$\begin{bmatrix} 2 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 5 \\ 1 \end{bmatrix}$$

This compact notation works for 2 equations or 2 million. The solving algorithms (Gaussian elimination, LU decomposition, etc.) are systematic versions of the elimination method you just learned.

### Back to the Running Example

The normal equation for linear regression is:

$$X^T X \, w = X^T y$$

This has *exactly* the form $Aw = b$, where $A = X^T X$ and $b = X^T y$. Solving it gives you the optimal weight vector $w$. Every concept in this chapter -- isolating variables, substitution, elimination -- is what `np.linalg.solve` does under the hood when you call it on this system.

For a simple linear regression $\hat{y} = w_0 + w_1 x$, the design matrix $X$ has a column of ones and a column of your feature values. The normal equation becomes a 2x2 system -- two equations, two unknowns ($w_0$ and $w_1$). For a model with 100 features, it becomes a 100x100 system. Same method. Same math. Just bigger matrices.

## Code: From Algebra to Implementation

```python
import numpy as np
from typing import Tuple, Optional

# ─── Solving single-variable linear equations ───────────────────────

def solve_linear_1var(a: float, b: float, c: float) -> Optional[float]:
    """
    Solve ax + b = c for x.
    Returns None if a = 0 (not a valid linear equation in x).
    """
    if a == 0:
        return None
    return (c - b) / a

# Example: 3x + 7 = 22
x = solve_linear_1var(a=3, b=7, c=22)
print(f"Solution to 3x + 7 = 22: x = {x}")  # x = 5.0


# ─── Solving systems: substitution (manual approach) ────────────────

def solve_system_substitution() -> Tuple[float, float]:
    """
    Solve:
      2x + y = 5
      x - y = 1
    Using substitution method.
    """
    # From equation 2: x = y + 1
    # Substitute into equation 1: 2(y + 1) + y = 5
    # 3y + 2 = 5
    # y = 1
    y = (5 - 2) / 3

    # Back-substitute: x = y + 1
    x = y + 1

    return x, y

x, y = solve_system_substitution()
print(f"Substitution method: x = {x}, y = {y}")  # x = 2.0, y = 1.0


# ─── Solving systems: matrix method (how you actually do it) ────────

def solve_system_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b using NumPy's linear algebra solver.
    This is the production approach for any non-trivial system.
    """
    return np.linalg.solve(A, b)

# Same system in matrix form
A = np.array([
    [2, 1],   # 2x + y = 5
    [1, -1]   # x - y = 1
])
b = np.array([5, 1])

solution = solve_system_matrix(A, b)
print(f"Matrix method: x = {solution[0]}, y = {solution[1]}")  # x = 2.0, y = 1.0


# ─── Verifying solutions ───────────────────────────────────────────

def verify_solution(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> bool:
    """Check if Ax = b (within floating point tolerance)."""
    result = A @ x  # Matrix multiplication
    return np.allclose(result, b)

print(f"Solution verified: {verify_solution(A, solution, b)}")  # True


# ─── The running example: linear regression via normal equations ────

def linear_regression_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Find weights w that minimize ||Xw - y||^2.

    Closed-form solution: w = (X^T X)^(-1) X^T y
    Implemented as solving the system: (X^T X) w = X^T y

    This IS the normal equation -- a system of linear equations
    where the matrix is X^T X and the right-hand side is X^T y.
    """
    # X^T X
    XtX = X.T @ X

    # X^T y
    Xty = X.T @ y

    # Solve (X^T X) w = X^T y  <-- this is Aw = b
    w = np.linalg.solve(XtX, Xty)

    return w

# Example: Simple linear regression
# Data points: (1, 2), (2, 4), (3, 5), (4, 4), (5, 5)
X = np.array([
    [1, 1],  # [1, x] format for y = w0 + w1*x
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5]
])
y = np.array([2, 4, 5, 4, 5])

weights = linear_regression_closed_form(X, y)
print(f"Linear regression weights: w0 = {weights[0]:.3f}, w1 = {weights[1]:.3f}")
# This gives us y_hat = w0 + w1*x -- the best-fit line


# ─── Checking system type before solving ────────────────────────────

def check_system_type(A: np.ndarray, b: np.ndarray) -> str:
    """
    Determine if a system has:
    - One solution (consistent, independent)
    - No solution (inconsistent)
    - Infinite solutions (consistent, dependent)
    """
    # Augmented matrix [A|b]
    augmented = np.column_stack([A, b])

    rank_A = np.linalg.matrix_rank(A)
    rank_aug = np.linalg.matrix_rank(augmented)
    n_vars = A.shape[1]

    if rank_A < rank_aug:
        return "No solution (inconsistent)"
    elif rank_A == rank_aug == n_vars:
        return "One unique solution"
    else:
        return "Infinite solutions (dependent)"

# Test: parallel lines (no solution)
A1 = np.array([[1, 1], [2, 2]])
b1 = np.array([1, 3])
print(f"Parallel lines: {check_system_type(A1, b1)}")

# Test: intersecting lines (one solution)
A2 = np.array([[1, 1], [1, -1]])
b2 = np.array([2, 0])
print(f"Intersecting lines: {check_system_type(A2, b2)}")


# ─── Iterative solving (preview of gradient descent) ────────────────

def solve_iteratively(A: np.ndarray, b: np.ndarray,
                      learning_rate: float = 0.1,
                      iterations: int = 100) -> np.ndarray:
    """
    Solve Ax = b iteratively by minimizing ||Ax - b||^2.

    This is conceptually the same thing gradient descent does
    during neural network training -- just on a simpler problem.
    """
    x = np.zeros(A.shape[1])  # Initial guess

    for i in range(iterations):
        # Gradient of ||Ax - b||^2 with respect to x is 2*A^T(Ax - b)
        gradient = 2 * A.T @ (A @ x - b)
        x = x - learning_rate * gradient

        if i % 20 == 0:
            error = np.linalg.norm(A @ x - b)
            print(f"  Iteration {i}: error = {error:.6f}")

    return x

print("\nIterative solution (gradient descent on a linear system):")
A = np.array([[2, 1], [1, -1]], dtype=float)
b = np.array([5, 1], dtype=float)
x_iterative = solve_iteratively(A, b, learning_rate=0.1, iterations=100)
print(f"  Final solution: x = {x_iterative[0]:.4f}, y = {x_iterative[1]:.4f}")
```

## Why This Matters for ML

Let's connect every piece back to the work you actually do.

### Linear Regression (the direct connection)

The most direct application. Given data points, find the best-fit hyperplane:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_n x_n$$

The normal equation $X^T X \, w = X^T y$ solves for optimal weights in one shot. This is a system of linear equations -- the exact thing this chapter teaches you to solve.

### Neural Network Training

At each training step, the weight update rule is:

$$W_{\text{new}} = W_{\text{old}} - \alpha \nabla L$$

The gradient $\nabla L$ involves linear operations at each layer during backpropagation. The forward pass is a chain of linear transformations (followed by nonlinear activations). Understanding linear equations is understanding the backbone of every neural network layer.

### Feature Transformations

Standardization is a linear equation:

$$z = \frac{x - \mu}{\sigma}$$

You use this every time you call `StandardScaler`. Knowing what it does algebraically helps you debug when your model behaves unexpectedly after preprocessing.

### Support Vector Machines

The dual formulation of SVMs requires solving a system of linear equations with inequality constraints.

### Principal Component Analysis

Finding principal components involves an eigenvalue problem, which reduces to solving systems of linear equations of the form $(A - \lambda I)\mathbf{v} = \mathbf{0}$.

## When to Use Direct vs. Iterative Methods

| Criterion | Direct Solve (`np.linalg.solve`) | Iterative (gradient-based) |
|---|---|---|
| System size | Small to moderate (< 10K variables) | Large (10K+ variables) |
| Matrix structure | Dense | Sparse (mostly zeros) |
| Precision needed | Exact | Approximate is fine |
| Repeated solving | One-time computation | Fits into training loop |
| Your use case | Normal equations, small models | Neural networks, large-scale ML |

### Common Pitfalls

1. **Singular matrices** -- If $\det(A) = 0$, the system has no unique solution. `np.linalg.solve` will raise `LinAlgError`. Always check, or use `np.linalg.lstsq` for a least-squares fallback.

2. **Numerical instability** -- Very large or very small coefficients cause floating-point precision issues. Scale your data first. This is why feature normalization matters.

3. **Overfitting in regression** -- Just because you CAN fit a line through points does not mean you SHOULD. A perfect fit on training data often means poor generalization. More on this when we cover polynomials.

4. **Assuming linearity** -- Real-world relationships are often nonlinear. Linear models are the starting point, not always the answer. But understanding them deeply makes nonlinear methods far easier to learn.

## Exercises

### Exercise 1: Solve by Substitution

Solve the system:

$$\begin{cases} 3x + 2y = 12 \\ x - y = 1 \end{cases}$$

**Solution**:

From equation 2: $x = y + 1$

Substitute into equation 1:

$$3(y + 1) + 2y = 12$$
$$3y + 3 + 2y = 12$$
$$5y = 9$$
$$y = 1.8$$

Back-substitute:

$$x = 1.8 + 1 = 2.8$$

**Answer**: $(x, y) = (2.8, 1.8)$

### Exercise 2: Solve a 3x3 System Using NumPy

Write code to solve:

$$\begin{cases} 4x + 3y - z = 1 \\ 2x - y + 2z = 8 \\ x + y + z = 6 \end{cases}$$

**Solution**:

```python
import numpy as np

A = np.array([
    [4, 3, -1],
    [2, -1, 2],
    [1, 1, 1]
])
b = np.array([1, 8, 6])

solution = np.linalg.solve(A, b)
print(f"x = {solution[0]:.2f}, y = {solution[1]:.2f}, z = {solution[2]:.2f}")
# x = 1.00, y = 2.00, z = 3.00
```

### Exercise 3: Identify System Type Without Solving

Determine if this system has one, zero, or infinite solutions:

$$\begin{cases} 2x + 4y = 8 \\ x + 2y = 4 \end{cases}$$

**Solution**:

Equation 1 is exactly 2 times equation 2:

$$2(x + 2y) = 2(4) \implies 2x + 4y = 8$$

The equations represent the same line, so there are **infinite solutions**. Any point $(x, y)$ satisfying $x + 2y = 4$ works. In matrix terms, the coefficient matrix is singular ($\det = 0$), confirming this.

### Exercise 4: Normal Equation by Hand

For the data points $(1, 1)$ and $(2, 3)$, set up the normal equation $X^T X \, w = X^T y$ for the model $\hat{y} = w_0 + w_1 x$. Solve it by elimination.

**Solution**:

Design matrix and target:

$$X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}, \quad y = \begin{bmatrix} 1 \\ 3 \end{bmatrix}$$

Compute $X^T X$ and $X^T y$:

$$X^T X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}^T \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 2 & 3 \\ 3 & 5 \end{bmatrix}$$

$$X^T y = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}^T \begin{bmatrix} 1 \\ 3 \end{bmatrix} = \begin{bmatrix} 4 \\ 7 \end{bmatrix}$$

The system to solve:

$$\begin{cases} 2w_0 + 3w_1 = 4 \\ 3w_0 + 5w_1 = 7 \end{cases}$$

Multiply equation 1 by 3 and equation 2 by 2:

$$\begin{cases} 6w_0 + 9w_1 = 12 \\ 6w_0 + 10w_1 = 14 \end{cases}$$

Subtract:

$$w_1 = 2$$

Back-substitute: $2w_0 + 3(2) = 4 \implies w_0 = -1$

**Answer**: $\hat{y} = -1 + 2x$. Verify: $\hat{y}(1) = 1$ and $\hat{y}(2) = 3$ -- both match the data exactly.

## Summary

- **Linear equations** are statements where variables appear only to the first power. Solving them means finding the values that make the statement true.

- **Single-variable equations** are solved by isolating the variable through inverse operations -- unwinding the expression like unwinding a call stack.

- **Systems of equations** are solved by **substitution** (inline a variable) or **elimination** (cancel a variable by combining equations). Both reduce the number of unknowns until you can solve directly.

- **Three outcomes** for a 2x2 system: one solution (lines intersect), no solution (parallel lines), infinite solutions (same line).

- **Matrix form** $A\mathbf{x} = \mathbf{b}$ generalizes everything. The same notation works for 2 unknowns or 2 million, and `np.linalg.solve` handles both.

- **The normal equation** $X^T X \, w = X^T y$ is the direct link to ML: solving a system of linear equations gives you the optimal weights for linear regression.

- **Key insight**: The substitution and elimination you do by hand are exactly what Gaussian elimination does inside `np.linalg.solve`. Understanding the small case gives you intuition for the large case.

---

> **What's Next** -- Linear equations are degree 1. What about $x^2$? Polynomials introduce curves, roots, and the foundation for activation functions.

Next: [Chapter 3: Polynomials](./03-polynomials.md) -->
