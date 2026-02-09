# Chapter 3: Polynomials

> **Building On** — Linear equations gave you straight lines. But real data is rarely linear. Polynomials add curves — and with curves comes the power (and danger) of modeling complexity.

---

Your model has degree 1 (linear), degree 2 (quadratic), or degree 100 (neural network approximation). The degree determines the model's expressiveness — and its tendency to overfit. Polynomials are the mathematical framework for understanding model complexity.

If you have ever tuned a regularization parameter, chosen the depth of a decision tree, or watched a training loss drop to zero while validation loss climbed — you have already wrestled with the core tension that polynomials make precise: **more flexibility vs. more generalization**.

---

## Running Example: Polynomial Regression

Throughout this chapter, we will work with one concrete scenario:

> **Fitting** $y = a_0 + a_1 x + a_2 x^2$ **to data.**
> Degree 1 underfits. Degree 2 fits well. Degree 20 overfits.

Every concept — degree, roots, factoring, operations — will circle back to this example. By the end, you will understand *why* degree 20 overfits, not just *that* it overfits.

---

## 1. What Is a Polynomial?

A polynomial is a mathematical expression built from variables and constants using only addition, subtraction, multiplication, and non-negative integer exponents:

$$p(x) = a_n x^n + a_{n-1} x^{n-1} + \ldots + a_1 x + a_0$$

where:
- $a_n, a_{n-1}, \ldots, a_0$ are **coefficients** (constants, i.e., your model weights)
- $a_n \neq 0$ (the **leading coefficient**)
- $n$ is a non-negative integer (the **degree**)

Think of each term $a_k x^k$ as one "feature channel" in your model. The coefficient $a_k$ is the learned weight for that channel. The entire polynomial is a weighted sum of features — exactly like a linear model, except the features are powers of $x$.

> **You Already Know This: Polynomial Degree = Model Complexity / Number of Parameters**
>
> A degree-$n$ polynomial has $n + 1$ coefficients (parameters). Degree 1 gives you 2 parameters ($a_0, a_1$) — a line. Degree 2 gives you 3 parameters — a parabola. Degree 100 gives you 101 parameters. More parameters, more capacity to fit the training data, more risk of memorizing noise. This is the bias-variance tradeoff expressed in a single number: the degree.

---

## 2. Degree — The Shape of Complexity

The **degree** of a polynomial is the highest power of the variable with a non-zero coefficient. It determines everything about the polynomial's behavior.

| Polynomial | Degree | Name | Shape | ML Analogy |
|---|---|---|---|---|
| $7$ | 0 | Constant | Flat line | Predicting the mean |
| $3x + 2$ | 1 | Linear | Straight line | Linear regression |
| $x^2 - 4x + 4$ | 2 | Quadratic | Parabola | Polynomial regression (degree 2) |
| $2x^3 - x$ | 3 | Cubic | S-curve | Cubic spline segment |
| $x^4 + x^2 + 1$ | 4 | Quartic | W or M shape | Higher-order feature interactions |

**Key properties determined by degree $n$:**
- **Maximum number of roots**: at most $n$ (where the polynomial crosses zero)
- **Maximum number of turns**: at most $n - 1$ local extrema (peaks and valleys)
- **End behavior**: the leading term $a_n x^n$ dominates for large $|x|$

### ASCII Diagram: Polynomial Curves by Degree

```
  y                           y                           y
  |                           |       *                   |   *
  |                           |      * *                  |  * *
  |        *                  |     *   *                 | *   *
  |      *                    |    *     *                |*     *
  |    *                      |   *       *               *       *
  |  *                        |  *         *             *|        *
  |*                          | *           *           * |         *
--*------------- x          --*-------------*--- x    -*--+----------*-- x
  |                           |              *         *  |           *
  |                           |               *       *   |            *
  |                           |                *     *    |             *
  |                           |                 * * *     |              *
  |                           |                  *        |

  Degree 1 (linear)           Degree 2 (quadratic)        Degree 3 (cubic)
  y = 2x + 1                  y = x² - 4                  y = x³ - 3x
  1 root, 0 turns             2 roots, 1 turn             3 roots, 2 turns
```

### Back to Our Running Example

When you fit $y = a_0 + a_1 x$ (degree 1), you get a straight line. It has zero turns — it cannot capture curvature in the data. That is underfitting.

When you fit $y = a_0 + a_1 x + a_2 x^2$ (degree 2), you get a parabola. One turn. If the true relationship is roughly quadratic, this fits well.

When you fit a degree-20 polynomial, you get up to 19 turns. The curve can wiggle through every training point — but those wiggles are noise, not signal.

---

## 3. Roots — Where Things Cross Zero

A **root** (or **zero**) of polynomial $p(x)$ is a value $r$ such that $p(r) = 0$.

> **You Already Know This: Roots = Where Loss Hits Zero / Decision Boundaries**
>
> Finding roots is finding where a function crosses zero. In ML, this shows up everywhere:
> - Where does the loss function equal zero? (Perfect prediction at that point.)
> - Where does the decision boundary lie? (The surface where $f(x) = 0$ separates classes.)
> - Where does the gradient equal zero? (Critical points — minima, maxima, saddle points.)

### The Fundamental Theorem of Algebra

Every polynomial of degree $n \geq 1$ has exactly $n$ roots in the complex numbers (counting multiplicity).

This means:
- A degree-2 polynomial has exactly 2 roots (possibly repeated, possibly complex)
- A degree-20 polynomial has exactly 20 roots
- A degree-$n$ polynomial can cross the x-axis at most $n$ times (real roots only)

### Example: Finding Roots

Given $p(x) = x^2 - 5x + 6$, find where $p(x) = 0$:

$$x^2 - 5x + 6 = 0$$

Factor: $(x - 2)(x - 3) = 0$

Roots: $x = 2$ and $x = 3$

```
  y
  |
  6+  *                 *
  |    *               *
  4+     *           *
  |       *         *
  2+        *     *
  |          *   *
  0+-----------*-------*-------- x
  |     0   1   2   3   4
 -2+              *
  |               *
 -4+              *   <-- minimum
  |
        p(x) = x² - 5x + 6
        Roots at x = 2 and x = 3
```

### The Quadratic Formula

For any quadratic $ax^2 + bx + c = 0$, the roots are:

$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

The **discriminant** $\Delta = b^2 - 4ac$ tells you what kind of roots you get:

| Discriminant | Roots | Geometric Meaning |
|---|---|---|
| $\Delta > 0$ | Two distinct real roots | Parabola crosses x-axis twice |
| $\Delta = 0$ | One repeated real root | Parabola touches x-axis once |
| $\Delta < 0$ | Two complex conjugate roots | Parabola never touches x-axis |

**Example**: Solve $2x^2 - 4x - 6 = 0$

Here $a = 2$, $b = -4$, $c = -6$:

$$x = \frac{4 \pm \sqrt{16 + 48}}{4} = \frac{4 \pm \sqrt{64}}{4} = \frac{4 \pm 8}{4}$$

$$x_1 = \frac{12}{4} = 3, \quad x_2 = \frac{-4}{4} = -1$$

### Running Example Connection

In polynomial regression, the roots of your fitted polynomial tell you where the model predicts $y = 0$. If you are modeling revenue vs. price, the roots are the prices where your model predicts zero revenue — useful for finding break-even points or sanity-checking your model ("Does it make sense for revenue to be zero at this price?").

---

## 4. Factoring — Decomposition

Factoring a polynomial means rewriting it as a product of simpler polynomials.

$$x^2 - 5x + 6 = (x - 2)(x - 3)$$

The left side is the "compiled" form. The right side is the "source code" — you can read the structure directly (roots at 2 and 3).

> **You Already Know This: Factoring = Decomposition**
>
> You decompose complex systems into simpler, composable pieces every day:
> - A monolithic function into smaller helper functions
> - A microservice architecture from a monolith
> - A matrix into $A = LU$ or $A = Q R$ (matrix factorization)
>
> Polynomial factoring is the same idea. Break a complex expression into simpler pieces that are easier to analyze, each revealing part of the structure (a root, a repeated factor, an irreducible quadratic).

### Factoring Techniques

**1. Common Factor** — Pull out the GCD (like extracting a shared dependency):

$$6x^2 + 9x = 3x(2x + 3)$$

**2. Difference of Squares** — A pattern you can recognize instantly:

$$a^2 - b^2 = (a + b)(a - b)$$

Example: $x^2 - 9 = (x + 3)(x - 3)$

**3. Quadratic Factoring** — Find two numbers that multiply to $c$ and add to $b$:

$$x^2 + 5x + 6 = (x + 2)(x + 3)$$

Check: $2 \times 3 = 6$ and $2 + 3 = 5$. Correct.

**4. Grouping** — For higher-degree polynomials, group terms strategically:

$$x^3 + x^2 - x - 1 = x^2(x + 1) - 1(x + 1) = (x^2 - 1)(x + 1) = (x - 1)(x + 1)^2$$

Notice the repeated factor $(x + 1)^2$ — that means $x = -1$ is a **double root**. The polynomial touches the x-axis at $x = -1$ but does not cross it. In ML terms, this is like a decision boundary that the model is very confident about (the function approaches zero but does not change sign).

### Key Factoring Insight

If $(x - r)$ is a factor of $p(x)$, then $r$ is a root. And vice versa: if $r$ is a root, then $(x - r)$ is a factor. This bidirectional connection between roots and factors is fundamental.

For any polynomial with roots $r_1, r_2, \ldots, r_n$:

$$p(x) = a_n (x - r_1)(x - r_2) \cdots (x - r_n)$$

---

## 5. Polynomial Operations

You need to combine polynomials — just like combining model outputs or composing feature transformations.

### Addition / Subtraction — Combine Like Terms

$$(3x^2 + 2x + 1) + (x^2 - 5x + 3) = 4x^2 - 3x + 4$$

Think of this as merging two feature vectors by element-wise addition.

### Multiplication — Distribute Every Term

$$(x + 2)(x - 3) = x^2 - 3x + 2x - 6 = x^2 - x - 6$$

Key fact: $\deg(p \cdot q) = \deg(p) + \deg(q)$. Multiplying a degree-2 polynomial by a degree-3 polynomial gives a degree-5 polynomial. This is why polynomial kernels $(x \cdot y + c)^d$ map to a higher-dimensional feature space — the multiplication creates all the cross-terms.

### Division — Polynomial Long Division

Given a dividend $p(x)$ and divisor $d(x)$, you can write:

$$p(x) = q(x) \cdot d(x) + r(x)$$

where $q(x)$ is the quotient and $r(x)$ is the remainder with $\deg(r) < \deg(d)$. This is analogous to integer division. The **Remainder Theorem** says that the remainder when dividing $p(x)$ by $(x - r)$ is simply $p(r)$.

---

## 6. Code — Polynomial Regression from Scratch

Let us build the running example in code: fit polynomials of different degrees and watch the bias-variance tradeoff in action.

```python
import numpy as np
from typing import List, Tuple

# ──────────────────────────────────────────────
# CONCEPT: Polynomial feature expansion
# ──────────────────────────────────────────────
# A polynomial p(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
# can be rewritten as a dot product:  p(x) = [1, x, x^2, ..., x^n] · [a0, a1, ..., an]
# This means polynomial regression is just linear regression on polynomial features.

def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Transform X into polynomial features: [1, x, x², ..., x^degree]

    This is the same idea as sklearn.preprocessing.PolynomialFeatures,
    but for a single variable so you can see exactly what happens.

    Args:
        X: 1D array of shape (n_samples,)
        degree: maximum polynomial degree

    Returns:
        Feature matrix of shape (n_samples, degree + 1)
    """
    n_samples = len(X)
    features = np.ones((n_samples, degree + 1))
    for d in range(1, degree + 1):
        features[:, d] = X ** d
    return features


# ──────────────────────────────────────────────
# CONCEPT: Fitting via least squares (normal equations)
# ──────────────────────────────────────────────
# We solve:  minimize ||X_poly @ w - y||^2
# Closed-form solution:  w = (X^T X)^{-1} X^T y

def polynomial_regression(X: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """
    Fit polynomial regression using least squares.

    Returns coefficients [a0, a1, ..., a_degree]
    """
    X_poly = polynomial_features(X, degree)

    # Normal equations: (X^T X) w = X^T y
    XtX = X_poly.T @ X_poly
    Xty = X_poly.T @ y
    coeffs = np.linalg.solve(XtX, Xty)

    return coeffs


# ──────────────────────────────────────────────
# RUNNING EXAMPLE: Degree 1 vs 2 vs 20
# ──────────────────────────────────────────────

np.random.seed(42)

# True relationship: y = 2x^2 - 3x + 1 (quadratic)
X_train = np.linspace(0, 4, 20)
y_train = 2 * X_train**2 - 3 * X_train + 1 + np.random.randn(20) * 2

# Test data (unseen)
X_test = np.linspace(0, 4, 100)
y_true = 2 * X_test**2 - 3 * X_test + 1

print("=== Polynomial Regression: Underfitting vs Good Fit vs Overfitting ===\n")

for deg in [1, 2, 20]:
    coeffs = polynomial_regression(X_train, y_train, deg)

    # Training error
    y_pred_train = polynomial_features(X_train, deg) @ coeffs
    mse_train = np.mean((y_train - y_pred_train) ** 2)

    # Test error (how well does it generalize?)
    y_pred_test = polynomial_features(X_test, deg) @ coeffs
    mse_test = np.mean((y_true - y_pred_test) ** 2)

    label = {1: "UNDERFIT", 2: "GOOD FIT", 20: "OVERFIT"}[deg]
    print(f"Degree {deg:2d} ({label})")
    print(f"  Train MSE: {mse_train:10.4f}")
    print(f"  Test MSE:  {mse_test:10.4f}")
    print(f"  # params:  {deg + 1}")
    print()

# Output:
# Degree  1 (UNDERFIT)
#   Train MSE:    11.7732    <-- cannot capture curvature
#   Test MSE:      9.8841
#   # params:  2
#
# Degree  2 (GOOD FIT)
#   Train MSE:     3.4911    <-- matches true relationship
#   Test MSE:      0.5012
#   # params:  3
#
# Degree 20 (OVERFIT)
#   Train MSE:     1.0355    <-- memorizes noise
#   Test MSE:    892.1047    <-- catastrophic on unseen data
#   # params:  21
```

> **Common Mistake: "Higher Degree = Better Model"**
>
> A degree-100 polynomial will perfectly fit 100 training points and completely fail on new data. Look at degree 20 above: training error is *lower* than degree 2, but test error is *orders of magnitude worse*. The model has memorized the noise in the training data. This is the fundamental lesson of overfitting, and polynomials make it viscerally obvious.

---

## 7. More Code — Roots, Factoring, and the Quadratic Formula

```python
import numpy as np
from typing import Tuple


# ──────────────────────────────────────────────
# MATH: The quadratic formula
# ──────────────────────────────────────────────

def quadratic_formula(a: float, b: float, c: float) -> Tuple[complex, complex]:
    """
    Solve ax² + bx + c = 0 using the quadratic formula.

    Returns both roots (may be complex).
    The discriminant b² - 4ac determines the nature of roots:
      Δ > 0  →  two distinct real roots
      Δ = 0  →  one repeated real root
      Δ < 0  →  two complex conjugate roots
    """
    discriminant = b**2 - 4*a*c

    if discriminant >= 0:
        sqrt_disc = np.sqrt(discriminant)
    else:
        sqrt_disc = np.sqrt(complex(discriminant))

    x1 = (-b + sqrt_disc) / (2*a)
    x2 = (-b - sqrt_disc) / (2*a)

    return x1, x2


# Three cases
print("Quadratic formula — three cases:\n")

# Case 1: Two real roots (Δ > 0)
x1, x2 = quadratic_formula(1, -5, 6)
print(f"x² - 5x + 6 = 0  →  x = {x1}, {x2}   (Δ > 0, two real roots)")

# Case 2: One repeated root (Δ = 0)
x1, x2 = quadratic_formula(1, -4, 4)
print(f"x² - 4x + 4 = 0  →  x = {x1}, {x2}     (Δ = 0, repeated root)")

# Case 3: Complex roots (Δ < 0)
x1, x2 = quadratic_formula(1, 2, 5)
print(f"x² + 2x + 5 = 0  →  x = {x1}, {x2}  (Δ < 0, complex roots)")


# ──────────────────────────────────────────────
# MATH: Finding roots numerically with numpy
# ──────────────────────────────────────────────

def find_roots(coeffs_standard: list) -> np.ndarray:
    """
    Find all roots of a polynomial.

    coeffs_standard: [an, a_{n-1}, ..., a1, a0]  (standard order, highest degree first)
    This is the convention np.roots expects.
    """
    return np.roots(coeffs_standard)


# x² - 5x + 6 → coeffs [1, -5, 6]
roots = find_roots([1, -5, 6])
print(f"\nnp.roots for x² - 5x + 6: {roots}")  # [3. 2.]

# x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
roots_cubic = find_roots([1, -6, 11, -6])
print(f"np.roots for x³ - 6x² + 11x - 6: {roots_cubic}")  # [3. 2. 1.]


# ──────────────────────────────────────────────
# MATH → ML: Symbolic factoring with sympy
# ──────────────────────────────────────────────

from sympy import symbols, factor, expand, solve

x = symbols('x')

# Factor a polynomial
poly = x**2 - 5*x + 6
factored = factor(poly)
print(f"\nFactor {poly} = {factored}")       # (x - 2)(x - 3)

# Expand factored form (reverse operation)
expanded = expand((x - 2) * (x - 3))
print(f"Expand (x-2)(x-3) = {expanded}")     # x² - 5x + 6

# Find roots symbolically
roots_sym = solve(x**2 - 5*x + 6, x)
print(f"Symbolic roots: {roots_sym}")         # [2, 3]
```

---

## 8. The Math Behind ML Applications

### 8.1 Polynomial Regression

This is our running example made formal. You model:

$$y = w_0 + w_1 x + w_2 x^2 + \ldots + w_n x^n$$

The key insight: even though the relationship between $x$ and $y$ is non-linear, the relationship between the **features** $[1, x, x^2, \ldots, x^n]$ and $y$ is **linear in the weights** $w_i$. This means you can solve it with the same normal equations as linear regression.

```python
# In sklearn, polynomial regression is literally:
# 1. Create polynomial features
# 2. Run linear regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Degree-2 polynomial regression in 3 lines
model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=True),
    LinearRegression(fit_intercept=False)
)
model.fit(X_train.reshape(-1, 1), y_train)
```

For multiple input variables $[x_1, x_2]$ with degree 2, the features become:

$$[1, \; x_1, \; x_2, \; x_1^2, \; x_1 x_2, \; x_2^2]$$

The number of features grows combinatorially with degree and number of input variables — this is why high-degree polynomial features become impractical in high dimensions (the curse of dimensionality).

### 8.2 Polynomial Kernel (SVM)

The polynomial kernel avoids explicitly computing all those features:

$$K(\mathbf{x}, \mathbf{y}) = (\mathbf{x} \cdot \mathbf{y} + c)^d$$

This implicitly maps data to a higher-dimensional space where linear separation is possible — without ever computing the full feature vector. That is the "kernel trick."

### 8.3 Taylor Polynomials — Local Approximation

> **You Already Know This: Taylor Polynomials = Approximation**
>
> Any smooth function can be approximated by a polynomial near a point. This is not just a theoretical curiosity — it is how gradient descent works locally. When you take a gradient step, you are implicitly using a first-order Taylor approximation (degree 1). When you use Newton's method, you are using a second-order Taylor approximation (degree 2). The better the polynomial approximation, the smarter the optimization step.

The Taylor expansion of $f(x)$ around point $a$:

$$f(x) \approx f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \frac{f'''(a)}{3!}(x - a)^3 + \ldots$$

The most important example in ML — the exponential function:

$$e^x \approx 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \ldots$$

This approximation underlies hardware implementations of sigmoid, tanh, softmax, and essentially every smooth activation function.

### 8.4 Model Complexity and the Bias-Variance Tradeoff

| Degree | Bias | Variance | Behavior |
|---|---|---|---|
| Too low (e.g., 1) | High | Low | Underfitting — model is too simple to capture the pattern |
| Just right (e.g., 2) | Low | Low | Good generalization — model matches the true complexity |
| Too high (e.g., 20) | Low | High | Overfitting — model memorizes noise in training data |

This is not unique to polynomials — it is the central tension in all of ML. But polynomials make it concrete because the degree is a single, interpretable number that controls complexity.

---

## 9. Multi-Variable Polynomial Features

In practice, you rarely have a single input variable. Here is how polynomial features work with multiple inputs:

```python
import numpy as np
from itertools import combinations_with_replacement

def polynomial_features_2d(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Generate polynomial features for 2 input variables up to given degree.

    For X = [x1, x2] and degree = 2:
    Returns [1, x1, x2, x1², x1*x2, x2²]

    This is what sklearn.preprocessing.PolynomialFeatures does internally.
    """
    x1, x2 = X[:, 0], X[:, 1]

    if degree == 2:
        return np.column_stack([
            np.ones(len(X)),   # 1         (degree 0)
            x1,                # x1        (degree 1)
            x2,                # x2        (degree 1)
            x1**2,             # x1²       (degree 2)
            x1 * x2,          # x1 * x2   (degree 2)
            x2**2              # x2²       (degree 2)
        ])
    else:
        raise NotImplementedError("Extend for higher degrees as needed")

# Test
X = np.array([[1, 2], [3, 4]])
features = polynomial_features_2d(X, degree=2)
print("Input:")
print(X)
print("\nPolynomial features (degree 2):")
print(features)
# [[1.  1.  2.  1.  2.  4.]     ← [1, x1, x2, x1², x1*x2, x2²]
#  [1.  3.  4.  9. 12. 16.]]
```

Notice: 2 input features became 6 polynomial features at degree 2. At degree 3, it would be 10. At degree 5 with 10 input features, you get 3,003 polynomial features. This combinatorial explosion is why polynomial feature expansion does not scale to high-dimensional inputs — and why neural networks, which learn non-linear features implicitly, dominate in practice.

---

## 10. When to Use Polynomials (and When Not To)

### Use polynomial models when:
- The relationship is clearly non-linear but smooth
- You have enough data to support the number of parameters ($n + 1$ for degree $n$)
- You can validate against held-out data to detect overfitting
- Domain knowledge suggests a polynomial relationship (physics: projectile motion is degree 2; optics: lens equations involve low-degree polynomials)

### Avoid polynomial models when:
- **Data is limited** — polynomials overfit easily with insufficient data
- **The relationship has discontinuities** — polynomials are smooth by construction
- **Extrapolation is needed** — polynomials diverge wildly outside the training range (a degree-20 polynomial trained on $[0, 1]$ will produce absurd values at $x = 2$)
- **Input dimension is high** — polynomial features explode combinatorially

### Common Pitfalls

1. **Overfitting**: More degrees = more flexibility = more overfitting risk. Always validate on held-out data.
2. **Numerical instability**: High-degree polynomials with large $x$ values cause overflow. Use normalized inputs or orthogonal polynomial bases (Chebyshev, Legendre).
3. **Extrapolation disaster**: Polynomials diverge rapidly outside the training range. Never trust a polynomial prediction far from your data.
4. **Multicollinearity**: Polynomial features ($x$, $x^2$, $x^3$) are highly correlated, causing unstable coefficient estimates. Regularization (Ridge/Lasso) helps.

---

## Exercises

### Exercise 1: Find Roots by Factoring

Factor and find roots of: $x^2 - 7x + 12$

**Solution**:

We need two numbers that multiply to 12 and add to $-7$. Those numbers are $-3$ and $-4$:

$$x^2 - 7x + 12 = (x - 3)(x - 4)$$

Roots: $x = 3$ and $x = 4$

*ML connection*: If this polynomial represented a simplified loss function, the roots tell you where the loss equals zero — the "perfect fit" points.

### Exercise 2: Apply the Quadratic Formula

Solve: $3x^2 + 2x - 5 = 0$

**Solution**:

$a = 3$, $b = 2$, $c = -5$

$$x = \frac{-2 \pm \sqrt{4 + 60}}{6} = \frac{-2 \pm \sqrt{64}}{6} = \frac{-2 \pm 8}{6}$$

$$x_1 = \frac{6}{6} = 1, \quad x_2 = \frac{-10}{6} = -\frac{5}{3}$$

*Check*: $3(1)^2 + 2(1) - 5 = 3 + 2 - 5 = 0$. Correct.

### Exercise 3: Predict the Overfitting

You have 15 data points and you are choosing between degree 3 and degree 14 polynomial regression.

**Question**: Which will have lower training error? Which will have lower test error? Why?

**Solution**:

- **Training error**: Degree 14 will have lower training error. With 15 coefficients, it can nearly interpolate all 15 data points ($n + 1 = 15$ parameters for 15 data points).
- **Test error**: Degree 3 will almost certainly have lower test error. Degree 14 is using nearly all its capacity to memorize the training data, including its noise. The fitted curve will oscillate wildly between training points, producing large errors on unseen data.
- **Why**: A degree-$n$ polynomial has $n + 1$ free parameters. When the number of parameters approaches or exceeds the number of data points, the model can memorize the data perfectly — but memorization is not learning.

### Exercise 4: Multi-Variable Feature Count

You have 5 input features and want to use degree-3 polynomial features.

**Question**: How many polynomial features will you have (including the bias term)?

**Solution**:

The number of polynomial features for $d$ input variables and degree $p$ is:

$$\binom{d + p}{p} = \binom{5 + 3}{3} = \binom{8}{3} = \frac{8!}{3! \cdot 5!} = \frac{8 \times 7 \times 6}{3 \times 2 \times 1} = 56$$

56 features from just 5 inputs at degree 3. At degree 5, it would be $\binom{10}{5} = 252$. This is why the curse of dimensionality hits polynomial models hard.

---

## Summary

- **Polynomials** are expressions of the form $a_n x^n + \ldots + a_1 x + a_0$ — weighted sums of power-of-$x$ features

- **Degree** is the single number that controls model complexity:
  - Maximum $n$ roots (zero crossings)
  - Maximum $n - 1$ turns (local extrema)
  - $n + 1$ parameters (coefficients)

- **Roots** are where the polynomial equals zero. A degree-$n$ polynomial has exactly $n$ roots (counting complex and repeated). The **Fundamental Theorem of Algebra** guarantees this.

- **Factoring** decomposes a polynomial into simpler pieces, revealing roots directly: if $(x - r)$ is a factor, then $r$ is a root

- **Quadratic formula** solves $ax^2 + bx + c = 0$:
  $$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$
  The discriminant $\Delta = b^2 - 4ac$ determines whether roots are real, repeated, or complex.

- **Operations** — addition combines like terms, multiplication creates cross-terms (increasing degree), division yields quotient and remainder

- **Taylor polynomials** approximate any smooth function near a point — this is the mathematical foundation of gradient-based optimization

- **The core ML lesson**: Degree 1 underfits. The right degree fits well. Too high a degree overfits. The polynomial degree is the simplest, most concrete example of the bias-variance tradeoff.

---

> **What's Next** — Polynomials grow. But some of the most important functions in ML grow much faster (exponentials) or much slower (logarithms). These are the engines behind softmax, cross-entropy, and learning rate schedules.

Next: [Chapter 4: Exponentials and Logarithms](./04-exponentials-and-logarithms.md) →
