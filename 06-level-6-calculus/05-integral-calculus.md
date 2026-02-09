# Chapter 5: Integral Calculus — Accumulation and Expectation

> **Building On** — Derivatives measure instantaneous change. Integration does the reverse: it accumulates. In ML, you'll use integrals to compute probabilities, expected values, and normalizing constants.

---

## The Problem: Computing Probabilities from Density Functions

A probability distribution tells you the likelihood of every possible outcome. But how do you compute P(0.3 < X < 0.7)? You integrate the density function. If derivatives are `git diff`, integrals are `git log` — they accumulate changes over a range.

Here is the core question this chapter answers: your movie recommendation system models user ratings with a continuous probability density function f(x). A user could rate a movie anywhere from 1 to 5 stars. You need to answer questions like:

- What is the probability a user rates between 3.5 and 4.5 stars?
- What is the expected (average) rating across all users?
- What is the variance — how spread out are the ratings?

Every one of these questions requires integration. You cannot escape it. But the good news: **integration is just the continuous version of a for-loop sum**. You already have the intuition; we just need to formalize it.

---

## From Summation to Integration

### The Intuition: Summing Infinitely Small Pieces

Suppose you have a continuous curve f(x) and you want to find the total area between it and the x-axis from x = a to x = b. Your first instinct as an engineer is correct — chop it into pieces and add them up:

```
    f(x) |
         |        *  *
         |      *      *
         |    *    ██    *
         |   * ██  ██  ██ *
         |  *  ██  ██  ██  *
         | *   ██  ██  ██   *
         |*    ██  ██  ██    *
         +---+-██--██--██-+-----> x
             a              b

    Each ██ is a rectangle of width dx and height f(x_i).
    Area of each rectangle = f(x_i) * dx
    Total area ≈ sum of all rectangles = Σ f(x_i) * dx

    Make the rectangles thinner and thinner (dx -> 0)...
    ...and the sum becomes an integral: ∫ f(x) dx
```

That is integration in a nutshell: **summing up infinitely many infinitely small pieces**.

> **You Already Know This** -- Definite integral = `sum()` over continuous data
>
> Think of a Riemann sum as the mathematical equivalent of:
> ```python
> total = sum(f(x_i) * dx for x_i in np.linspace(a, b, n))
> ```
> The definite integral is the limit as `n -> infinity` — or equivalently, as `dx -> 0`.
> It is the limit of finer and finer discretization. Every numerical integration
> library is just computing this sum with smart choices about where to sample.

### Riemann Sums — The Discretized Version

Formally, divide [a, b] into n equal subintervals of width $\Delta x = \frac{b - a}{n}$. Pick a sample point $x_i^*$ in each subinterval. The **Riemann sum** is:

$$S_n = \sum_{i=1}^{n} f(x_i^*) \Delta x$$

```
    Riemann Sums: Left, Midpoint, Right

    LEFT endpoints:           MIDPOINT:                RIGHT endpoints:
    f(x)|                     f(x)|                    f(x)|
        | ___*                    | ___*                   | ___*
        |█  / *                   |  * / *                 |   / *█
        |█ /   *                  | *█/   *                |  / █ *
        |█/     *                 |*█/     *               | / █   *
        |█       *                |█/       *              |/ █     *
        +█---------> x           +█----------> x          +--█-------> x
         a       b                a         b              a       b

    As n -> infinity, all three converge to the same value: the integral
```

### The Definite Integral

Take the limit as the number of rectangles goes to infinity:

$$\int_a^b f(x)\,dx = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i^*) \Delta x$$

This is the **definite integral** of f from a to b. It equals the signed area between f(x) and the x-axis over [a, b].

Key properties:

- **Linearity**: $\int [af(x) + bg(x)]\,dx = a\int f(x)\,dx + b\int g(x)\,dx$
- **Additivity**: $\int_a^b f(x)\,dx + \int_b^c f(x)\,dx = \int_a^c f(x)\,dx$
- **Sign**: Area below the x-axis counts as negative

> **Common Mistake** -- An integral gives you an AREA, not a value at a point.
>
> For continuous probability distributions, P(X = exactly 0.5) = 0. You need to
> integrate over an interval to get a nonzero probability. The density function
> f(0.5) tells you the "height" at that point, but the probability requires area:
> P(0.49 < X < 0.51) = integral of f(x) from 0.49 to 0.51.

---

## The Fundamental Theorem of Calculus

This is the single most important result in calculus. It says that differentiation and integration are inverse operations.

> **You Already Know This** -- The Fundamental Theorem = encode/decode
>
> Differentiation and integration are inverses, like `compress()` and `decompress()`,
> or `encode()` and `decode()`. If you differentiate a function and then integrate
> the result, you get back the original (up to a constant). If you integrate a
> function and then differentiate, you recover the integrand.
>
> ```
> compress(data)   -->  compressed  -->  decompress(compressed)  = data
> differentiate(F) -->  f = F'      -->  integrate(f)            = F + C
> ```

### Part 1: Differentiation Undoes Integration

If you define a new function by integrating f from a fixed starting point a up to a variable endpoint x:

$$F(x) = \int_a^x f(t)\,dt$$

then the derivative of F is just f again:

$$F'(x) = f(x)$$

In other words, the rate of change of the accumulated area equals the current value of the function.

### Part 2: How to Evaluate Definite Integrals

If F is any antiderivative of f (meaning $F'(x) = f(x)$), then:

$$\int_a^b f(x)\,dx = F(b) - F(a)$$

This is enormously powerful. Instead of computing limits of Riemann sums, you just find an antiderivative, plug in the endpoints, and subtract.

### Visual Summary

```
    The Fundamental Theorem of Calculus

                    differentiate
         F(x)  ───────────────────────>  f(x) = F'(x)
     (antiderivative)                      (original function)

                    integrate
         F(x)  <───────────────────────  f(x)
                   ∫ f(x) dx = F(x) + C

    These operations are inverses of each other!

    Example:
         F(x) = x^3/3  ──differentiate──>  f(x) = x^2
         F(x) = x^3/3  <──integrate──────  f(x) = x^2

    To evaluate ∫[0 to 2] x^2 dx:
         F(2) - F(0) = 8/3 - 0 = 8/3
```

### Common Integrals (Your Reference Table)

| Function | Antiderivative | Example |
|----------|---------------|---------|
| $x^n$ | $\frac{x^{n+1}}{n+1} + C$ | $\int x^2\,dx = \frac{x^3}{3} + C$ |
| $e^x$ | $e^x + C$ | $\int e^x\,dx = e^x + C$ |
| $\frac{1}{x}$ | $\ln\|x\| + C$ | $\int \frac{1}{x}\,dx = \ln\|x\| + C$ |
| $\sin x$ | $-\cos x + C$ | $\int \sin x\,dx = -\cos x + C$ |
| $\cos x$ | $\sin x + C$ | $\int \cos x\,dx = \sin x + C$ |

### Integration Techniques

**Substitution** (the reverse chain rule): If $u = g(x)$, then:

$$\int f(g(x))\,g'(x)\,dx = \int f(u)\,du$$

Think of this as changing variables — like renaming a variable in your code to simplify an expression.

**Integration by Parts**: When the integrand is a product of two functions:

$$\int u\,dv = uv - \int v\,du$$

This is the integration analogue of the product rule for derivatives.

---

## Probability: Where Integrals Become Essential

This is where integrals go from "nice math" to "you literally cannot do ML without this."

### Continuous Probability Distributions

For a **probability density function** (PDF) $f(x)$:

$$P(a \leq X \leq b) = \int_a^b f(x)\,dx$$

The probability of X falling in an interval is the **area under the density curve** over that interval.

```
    Probability as Area Under a Density Curve

    f(x) |
         |       *  *  *
         |     *  AREA   *
         |    * ████████  *         ████ = P(a <= X <= b)
         |   *  ████████   *              = ∫[a to b] f(x) dx
         |  *   ████████    *
         | *    ████████     *
         |*     ████████      *
         +------████████---------> x
                a      b

    Key properties of a valid PDF:
    1. f(x) >= 0 for all x          (densities are non-negative)
    2. ∫ f(x) dx = 1  (over all x)  (total probability = 1)
```

### Running Example: Movie Rating Distribution

Suppose our movie recommendation system models ratings with a density function. For a particular movie, the rating density is:

$$f(x) = \frac{3}{32}(5 - x)(x - 1) \quad \text{for } 1 \leq x \leq 5$$

This is a simple parabolic density that peaks around 3 stars. Let us find the probability a user rates between 3.5 and 4.5 stars:

$$P(3.5 \leq X \leq 4.5) = \int_{3.5}^{4.5} \frac{3}{32}(5 - x)(x - 1)\,dx$$

Expanding: $(5 - x)(x - 1) = -x^2 + 6x - 5$

$$= \frac{3}{32} \int_{3.5}^{4.5} (-x^2 + 6x - 5)\,dx = \frac{3}{32}\left[-\frac{x^3}{3} + 3x^2 - 5x\right]_{3.5}^{4.5}$$

At $x = 4.5$: $-\frac{91.125}{3} + 60.75 - 22.5 = -30.375 + 60.75 - 22.5 = 7.875$

At $x = 3.5$: $-\frac{42.875}{3} + 36.75 - 17.5 = -14.2917 + 36.75 - 17.5 = 4.9583$

$$P(3.5 \leq X \leq 4.5) = \frac{3}{32}(7.875 - 4.9583) = \frac{3}{32}(2.9167) \approx 0.2734$$

About 27.3% of users will rate between 3.5 and 4.5 stars. That is the integral at work.

---

## Expected Value: The Integral You Will Use Most

### Definition and Intuition

The **expected value** of a continuous random variable X with density f(x) is:

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x)\,dx$$

> **You Already Know This** -- Expected value = weighted average
>
> The expected value is a weighted average where the "weights" come from the
> probability distribution. In the discrete case you would write:
> ```python
> expected = sum(value * probability for value, probability in distribution)
> ```
> The integral is the continuous version of exactly this sum. Instead of
> multiplying each value by its probability, you multiply each value x by its
> density f(x) and integrate.

### Running Example: Expected Movie Rating

For our movie rating density $f(x) = \frac{3}{32}(5 - x)(x - 1)$ on [1, 5]:

$$\mathbb{E}[\text{rating}] = \int_1^5 x \cdot \frac{3}{32}(5 - x)(x - 1)\,dx$$

Expanding: $x \cdot (-x^2 + 6x - 5) = -x^3 + 6x^2 - 5x$

$$= \frac{3}{32}\int_1^5 (-x^3 + 6x^2 - 5x)\,dx = \frac{3}{32}\left[-\frac{x^4}{4} + 2x^3 - \frac{5x^2}{2}\right]_1^5$$

At $x = 5$: $-\frac{625}{4} + 250 - \frac{125}{2} = -156.25 + 250 - 62.5 = 31.25$

At $x = 1$: $-\frac{1}{4} + 2 - \frac{5}{2} = -0.25 + 2 - 2.5 = -0.75$

$$\mathbb{E}[\text{rating}] = \frac{3}{32}(31.25 - (-0.75)) = \frac{3}{32}(32) = 3.0$$

The expected rating is exactly 3.0 stars — the center of the [1, 5] range, which makes sense given our symmetric density function.

This is the formula: **E[rating] = integral of x * f(x) dx.** It is a number you will compute (or approximate) constantly in ML.

### Generalizing: E[g(X)]

For any function g of a random variable X:

$$\mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot f(x)\,dx$$

This is how you compute things like:

- **Variance**: $\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \int (x - \mu)^2 f(x)\,dx$ where $\mu = \mathbb{E}[X]$
- **Second moment**: $\mathbb{E}[X^2] = \int x^2 f(x)\,dx$
- And the useful shortcut: $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$

---

## Multiple Integrals

When you have functions of multiple variables, you integrate over each variable:

$$\iint_R f(x, y)\,dA = \int_a^b \int_c^d f(x, y)\,dy\,dx$$

This computes the "volume" under a surface. In ML, joint probability distributions over multiple variables require multiple integrals — for example, the joint density of two features in a dataset. You evaluate them by working from the inside out, integrating one variable at a time (just like nested for-loops).

---

## Numerical Integration: The Practical Toolkit

In practice, most integrals in ML do not have closed-form antiderivatives. You will use numerical methods.

> **You Already Know This** -- Numerical integration = `np.trapz()`, `scipy.integrate.quad()`
>
> You rarely need to compute antiderivatives by hand. The practical tools you
> will actually use are:
> - `np.trapz(y, x)` — trapezoidal rule on discrete data points
> - `scipy.integrate.quad(f, a, b)` — adaptive quadrature for arbitrary functions
> - Monte Carlo integration — for high-dimensional integrals (the workhorse of Bayesian ML)
>
> These are all doing the same thing under the hood: approximating the integral
> as a sum of function evaluations, just with different strategies for choosing
> sample points and weights.

### Code: Numerical Integration Methods

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import norm

# =============================================================================
# Numerical Integration Methods
# =============================================================================

def riemann_sum(f, a, b, n=1000, method='midpoint'):
    """
    Compute Riemann sum approximation to integral.

    This is the direct implementation of the mathematical definition:
    ∫f(x)dx ≈ Σ f(x_i) * dx

    Parameters:
        f: function to integrate
        a, b: integration bounds
        n: number of rectangles (more = more accurate)
        method: 'left', 'right', or 'midpoint'
    """
    dx = (b - a) / n
    x = np.linspace(a, b, n, endpoint=False)

    if method == 'left':
        return np.sum(f(x)) * dx
    elif method == 'right':
        return np.sum(f(x + dx)) * dx
    elif method == 'midpoint':
        return np.sum(f(x + dx/2)) * dx

def trapezoidal(f, a, b, n=1000):
    """Trapezoidal rule — uses trapezoids instead of rectangles."""
    x = np.linspace(a, b, n)
    y = f(x)
    dx = (b - a) / (n - 1)
    return np.trapz(y, dx=dx)

def simpsons_rule(f, a, b, n=1000):
    """Simpson's rule — fits parabolas through groups of three points."""
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = f(x)
    dx = (b - a) / n
    return dx / 3 * (y[0] + 4*np.sum(y[1::2]) + 2*np.sum(y[2:-1:2]) + y[-1])

# Test: ∫[0 to 1] x² dx = 1/3
print("=" * 60)
print("Numerical Integration of integral[0 to 1] x^2 dx")
print("=" * 60)

f = lambda x: x**2
exact = 1/3

print(f"Exact value: {exact:.10f}")
print(f"Riemann (left):     {riemann_sum(f, 0, 1, method='left'):.10f}")
print(f"Riemann (right):    {riemann_sum(f, 0, 1, method='right'):.10f}")
print(f"Riemann (midpoint): {riemann_sum(f, 0, 1, method='midpoint'):.10f}")
print(f"Trapezoidal:        {trapezoidal(f, 0, 1):.10f}")
print(f"Simpson's rule:     {simpsons_rule(f, 0, 1):.10f}")
print(f"scipy.integrate:    {integrate.quad(f, 0, 1)[0]:.10f}")
```

### Code: Visualizing Riemann Sums

```python
def plot_riemann_sum():
    """Visualize how Riemann sums approximate the area under a curve."""
    f = lambda x: np.sin(x) + 1
    a, b = 0, np.pi
    n = 10

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x_smooth = np.linspace(a, b, 200)

    for ax, method, title in zip(axes,
                                  ['left', 'midpoint', 'right'],
                                  ['Left Riemann Sum', 'Midpoint Rule', 'Right Riemann Sum']):
        ax.plot(x_smooth, f(x_smooth), 'b-', linewidth=2, label='f(x)')

        dx = (b - a) / n
        for i in range(n):
            x_left = a + i * dx
            x_right = x_left + dx

            if method == 'left':
                height = f(x_left)
            elif method == 'right':
                height = f(x_right)
            else:  # midpoint
                height = f((x_left + x_right) / 2)

            rect = plt.Rectangle((x_left, 0), dx, height,
                                 fill=True, facecolor='lightblue',
                                 edgecolor='blue', alpha=0.7)
            ax.add_patch(rect)

        approx = riemann_sum(f, a, b, n, method)
        exact = integrate.quad(f, a, b)[0]

        ax.set_xlim(a - 0.2, b + 0.2)
        ax.set_ylim(0, 2.5)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'{title}\nApprox: {approx:.4f}, Exact: {exact:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('riemann_sum_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved as 'riemann_sum_visualization.png'")

plot_riemann_sum()
```

---

## Probability Computations in Practice

### Computing Probabilities from the Normal Distribution

The standard normal distribution is the workhorse density in ML. Let us compute some probabilities:

```python
# =============================================================================
# Probability as Area Under PDF — Standard Normal
# =============================================================================

print("\n" + "=" * 60)
print("Probability as Area Under PDF")
print("=" * 60)

mu, sigma = 0, 1

def normal_pdf(x):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# The famous 68-95-99.7 rule — each computed by integration
prob_1sigma, _ = integrate.quad(normal_pdf, -1, 1)
print(f"P(-1 < X < 1) = {prob_1sigma:.4f} (the '68%' rule)")

prob_2sigma, _ = integrate.quad(normal_pdf, -2, 2)
print(f"P(-2 < X < 2) = {prob_2sigma:.4f} (the '95%' rule)")

prob_3sigma, _ = integrate.quad(normal_pdf, -3, 3)
print(f"P(-3 < X < 3) = {prob_3sigma:.4f} (the '99.7%' rule)")

# Verify total probability = 1 (a valid PDF must integrate to 1)
total_prob, _ = integrate.quad(normal_pdf, -np.inf, np.inf)
print(f"Total probability: {total_prob:.6f}")
```

### Expected Value Computation

```python
# =============================================================================
# Expected Value — The Integral You'll Use Most
# =============================================================================

print("\n" + "=" * 60)
print("Expected Value (Expectation)")
print("=" * 60)

# E[X] for standard normal: ∫ x * f(x) dx
def x_times_pdf(x):
    return x * normal_pdf(x)

expected_value, _ = integrate.quad(x_times_pdf, -np.inf, np.inf)
print(f"E[X] for standard normal: {expected_value:.6f} (should be 0)")

# E[X^2] — the second moment: ∫ x^2 * f(x) dx
def x2_times_pdf(x):
    return x**2 * normal_pdf(x)

second_moment, _ = integrate.quad(x2_times_pdf, -np.inf, np.inf)
print(f"E[X^2] for standard normal: {second_moment:.6f} (should be 1 = sigma^2)")

# Var(X) = E[X^2] - (E[X])^2
variance = second_moment - expected_value**2
print(f"Var(X) = E[X^2] - E[X]^2: {variance:.6f}")

# --- Running Example: Movie Rating Expected Value ---
print("\n--- Movie Rating Distribution ---")

def movie_pdf(x):
    """PDF for movie ratings: f(x) = (3/32)(5-x)(x-1) on [1,5]."""
    if np.isscalar(x):
        if x < 1 or x > 5:
            return 0
        return (3/32) * (5 - x) * (x - 1)
    result = np.where((x >= 1) & (x <= 5), (3/32) * (5 - x) * (x - 1), 0)
    return result

# E[rating] = ∫ x * f(x) dx
def rating_times_pdf(x):
    return x * movie_pdf(x)

expected_rating, _ = integrate.quad(rating_times_pdf, 1, 5)
print(f"E[rating] = {expected_rating:.4f} (expected: 3.0)")

# E[rating^2] = ∫ x^2 * f(x) dx
def rating2_times_pdf(x):
    return x**2 * movie_pdf(x)

second_moment_rating, _ = integrate.quad(rating2_times_pdf, 1, 5)
rating_variance = second_moment_rating - expected_rating**2
print(f"Var(rating) = {rating_variance:.4f}")
print(f"Std(rating) = {np.sqrt(rating_variance):.4f}")

# --- Exponential Distribution Example ---
print("\n--- Exponential Distribution (lambda=2) ---")
lambda_param = 2

def exp_pdf(x):
    if x < 0:
        return 0
    return lambda_param * np.exp(-lambda_param * x)

def x_times_exp_pdf(x):
    return x * exp_pdf(x)

exp_expected, _ = integrate.quad(x_times_exp_pdf, 0, np.inf)
print(f"E[X] for Exp(lambda=2): {exp_expected:.4f} (should be 1/lambda = 0.5)")
```

---

## ML Applications of Integration

### Computing Probabilities (Bayes' Rule Denominator)

Many ML tasks involve computing probabilities with Bayes' theorem:

$$P(\text{class} = k \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid k)\,P(k)}{\int p(\mathbf{x} \mid k')\,P(k')\,dk'}$$

The denominator is an integral — a normalizing constant that ensures probabilities sum to 1. When this integral is intractable (as it often is in deep generative models), you need approximation methods.

### Expected Loss and Risk

The true **risk** of a model h is the expected loss over the data distribution:

$$R(h) = \mathbb{E}_{(x,y) \sim P}[\ell(h(x), y)] = \int \ell(h(x), y)\,p(x, y)\,dx\,dy$$

Since you cannot compute this integral (you do not know p(x, y)), you approximate it with the **empirical risk** — which is just the sample average:

$$\hat{R}(h) = \frac{1}{n} \sum_{i=1}^{n} \ell(h(x_i), y_i)$$

Notice: the empirical risk is a Riemann-sum-like approximation of the true integral. As your dataset grows, the sample average converges to the true expectation. This is the law of large numbers in action.

```python
# =============================================================================
# ML Application: Expected Loss as an Integral
# =============================================================================

print("\n" + "=" * 60)
print("ML Application: Expected Loss")
print("=" * 60)

# Suppose X ~ Uniform[0, 1] and we predict y_hat = 0.3
# Loss = (X - y_hat)^2
# Expected Loss = E[(X - 0.3)^2] = ∫ (x - 0.3)^2 * 1 dx over [0,1]

y_pred = 0.3

def squared_loss_times_pdf(x):
    loss = (x - y_pred)**2
    pdf = 1.0  # Uniform[0,1] has PDF = 1
    return loss * pdf

expected_loss, _ = integrate.quad(squared_loss_times_pdf, 0, 1)
print(f"Prediction: y_hat = {y_pred}")
print(f"Expected Squared Loss: {expected_loss:.6f}")

# What prediction minimizes the expected loss?
# (Should be E[X] = 0.5 for MSE with Uniform[0,1])
predictions = np.linspace(0, 1, 50)
losses = []
for pred in predictions:
    loss_func = lambda x, p=pred: (x - p)**2
    exp_loss, _ = integrate.quad(loss_func, 0, 1)
    losses.append(exp_loss)

optimal_pred = predictions[np.argmin(losses)]
print(f"Optimal prediction (minimizes expected loss): {optimal_pred:.4f}")
print(f"This equals E[X] = 0.5 for Uniform[0,1]")
```

### Information Theory

**Entropy** — the expected "surprise" of a distribution — is an integral:

$$H(X) = -\int p(x) \log p(x)\,dx$$

**KL Divergence** — measuring how one distribution differs from another:

$$D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)}\,dx$$

Both of these are expected values under p(x). Entropy is E[-log p(X)]. KL divergence is E[log(p(X)/q(X))]. The integral is doing the "expected value" computation in each case.

### Variational Inference and the ELBO

VAEs and other generative models involve intractable integrals. The marginal likelihood of data x is:

$$p(x) = \int p(x \mid z)\,p(z)\,dz$$

This integral is intractable because it sums over all possible latent variables z. Instead, we optimize a lower bound (the ELBO):

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

### Monte Carlo Integration: When Closed-Form Fails

When integrals cannot be solved analytically (which is most of the time in ML), **Monte Carlo integration** approximates them using random sampling:

$$\int f(x)\,p(x)\,dx \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i), \quad x_i \sim p(x)$$

Draw N samples from the distribution, evaluate f at each sample, and average. That is it. The approximation improves as N grows (by the law of large numbers).

This is the foundation of:
- **Dropout** (approximate Bayesian inference via random masking)
- **Policy gradient methods** in reinforcement learning
- **Stochastic variational inference** in Bayesian deep learning
- **MCMC methods** (Markov Chain Monte Carlo) for posterior sampling

```python
# =============================================================================
# Monte Carlo Integration
# =============================================================================

print("\n" + "=" * 60)
print("Monte Carlo Integration")
print("=" * 60)

def monte_carlo_integrate(f, a, b, n_samples=10000):
    """
    Monte Carlo integration: approximate ∫f(x)dx using random samples.

    Key idea: E[f(X)] = (1/N) * sum(f(x_i)) where x_i ~ Uniform[a,b]
    Therefore: ∫f(x)dx = (b-a) * E[f(X)] ≈ (b-a) * mean(f(samples))
    """
    x_samples = np.random.uniform(a, b, n_samples)
    return (b - a) * np.mean(f(x_samples))

# Test: ∫[0 to pi] sin^2(x) dx = pi/2
f_hard = lambda x: np.sin(x)**2

print("integral[0 to pi] sin^2(x) dx")
print(f"Exact: {np.pi/2:.6f}")
print(f"Monte Carlo (1000):   {monte_carlo_integrate(f_hard, 0, np.pi, 1000):.6f}")
print(f"Monte Carlo (10000):  {monte_carlo_integrate(f_hard, 0, np.pi, 10000):.6f}")
print(f"Monte Carlo (100000): {monte_carlo_integrate(f_hard, 0, np.pi, 100000):.6f}")

# Monte Carlo for our movie rating expected value
print("\n--- Monte Carlo: E[rating] ---")
# Sample from a distribution proportional to f(x) = (3/32)(5-x)(x-1) on [1,5]
# Using rejection sampling for demonstration
n_mc = 100000
samples = np.random.uniform(1, 5, n_mc)
weights = (3/32) * (5 - samples) * (samples - 1)
# For uniform proposal on [1,5], E[rating] = ∫ x*f(x) dx
# ≈ (5-1) * mean(x * f(x)) when x ~ Uniform[1,5]
mc_expected = 4 * np.mean(samples * weights)
print(f"Monte Carlo E[rating] (N={n_mc}): {mc_expected:.4f}")
print(f"Exact E[rating]: 3.0000")
```

### Code: Visualizing PDF, CDF, and Probability

```python
# =============================================================================
# Visualization: PDF, CDF, and Probability
# =============================================================================

def plot_probability_interpretation():
    """Visualize probability as area under PDF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.linspace(-4, 4, 1000)

    # PDF with shaded region
    ax1 = axes[0]
    pdf = norm.pdf(x, 0, 1)
    ax1.plot(x, pdf, 'b-', linewidth=2, label='PDF')

    # Shade P(-1 < X < 1)
    x_fill = np.linspace(-1, 1, 100)
    ax1.fill_between(x_fill, norm.pdf(x_fill), alpha=0.3, color='blue',
                     label=f'P(-1 < X < 1) = {norm.cdf(1) - norm.cdf(-1):.3f}')

    ax1.set_xlabel('x')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Standard Normal PDF\nProbability = Area Under Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # CDF = integral of PDF from -inf to x
    ax2 = axes[1]
    cdf = norm.cdf(x, 0, 1)
    ax2.plot(x, cdf, 'r-', linewidth=2, label='CDF')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.scatter([0], [0.5], color='red', s=100, zorder=5)
    ax2.annotate('Median', (0, 0.5), xytext=(0.5, 0.6),
                arrowprops=dict(arrowstyle='->', color='black'))

    ax2.set_xlabel('x')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('CDF = integral of PDF from -inf to x')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('probability_interpretation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved as 'probability_interpretation.png'")

plot_probability_interpretation()
```

---

## When to Use / When to Abstract Away

### When Integrals Matter
- **Understanding probability distributions** — you cannot interpret PDFs without knowing what integration means
- **Deriving loss functions** and understanding their properties
- **Information theory** — entropy, KL divergence, mutual information
- **Bayesian methods** — posterior computation, marginal likelihood, evidence lower bounds

### When to Abstract Away
- **Most neural network training** — automatic differentiation handles everything; you call `loss.backward()`, not `integrate()`
- **Using standard loss functions** — the math has been done for you; just pick MSE or cross-entropy
- **Practical implementation** — use `scipy.integrate.quad()` or Monte Carlo; do not compute antiderivatives by hand

### Common Pitfalls

1. **Confusing density with probability**: $f(x)$ is a density, not a probability. $f(x)$ can be greater than 1. You must integrate to get a probability.
2. **Numerical precision**: Integrals over infinite ranges require care — adaptive methods like `quad()` handle this, but naive Riemann sums will not.
3. **Curse of dimensionality**: High-dimensional integrals are computationally intractable with grid-based methods. This is exactly why Monte Carlo methods exist — their convergence rate is independent of dimension.
4. **Forgetting normalization**: If you define a density manually, verify it integrates to 1.

---

## Exercises

### Exercise 1: Compute a Definite Integral

**Problem**: Evaluate $\int_0^1 (3x^2 + 2x)\,dx$ using the Fundamental Theorem.

**Solution**:

Find the antiderivative: $F(x) = x^3 + x^2$

Apply the Fundamental Theorem:

$$\int_0^1 (3x^2 + 2x)\,dx = F(1) - F(0) = (1 + 1) - (0 + 0) = 2$$

### Exercise 2: Expected Value

**Problem**: If $X \sim \text{Uniform}[0, 2]$ (so $f(x) = \frac{1}{2}$ on [0, 2]), find $\mathbb{E}[X^2]$.

**Solution**:

$$\mathbb{E}[X^2] = \int_0^2 x^2 \cdot \frac{1}{2}\,dx = \frac{1}{2} \cdot \frac{x^3}{3}\Big|_0^2 = \frac{1}{2} \cdot \frac{8}{3} = \frac{4}{3}$$

### Exercise 3: Movie Rating Variance

**Problem**: Using our movie rating density $f(x) = \frac{3}{32}(5 - x)(x - 1)$ on [1, 5], compute $\text{Var}(\text{rating})$ using $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$. You already know $\mathbb{E}[X] = 3$.

**Solution**:

$$\mathbb{E}[X^2] = \int_1^5 x^2 \cdot \frac{3}{32}(5-x)(x-1)\,dx = \frac{3}{32}\int_1^5 (-x^4 + 6x^3 - 5x^2)\,dx$$

$$= \frac{3}{32}\left[-\frac{x^5}{5} + \frac{6x^4}{4} - \frac{5x^3}{3}\right]_1^5 = \frac{3}{32}\left[\left(-625 + 937.5 - \frac{625}{3}\right) - \left(-\frac{1}{5} + \frac{3}{2} - \frac{5}{3}\right)\right]$$

$$= \frac{3}{32} \cdot \frac{128}{15} \cdot 4 = \frac{3}{32} \cdot \frac{512}{15} = \frac{32}{5} = \frac{48}{5} \cdot \frac{1}{3}$$

Working this through numerically: $\mathbb{E}[X^2] = 9.6$, so $\text{Var}(X) = 9.6 - 9 = 0.6$.

Verify with code:
```python
import numpy as np
from scipy import integrate

f = lambda x: x**2 * (3/32) * (5 - x) * (x - 1)
ex2, _ = integrate.quad(f, 1, 5)
print(f"E[X^2] = {ex2:.4f}")           # 9.6
print(f"Var(X) = {ex2 - 3**2:.4f}")    # 0.6
```

### Exercise 4: Monte Carlo Estimation of Pi

**Problem**: Use Monte Carlo with 10,000 samples to estimate $\pi$ using $\pi = 4\int_0^1 \sqrt{1 - x^2}\,dx$.

**Solution**:
```python
import numpy as np
x = np.random.uniform(0, 1, 10000)
pi_estimate = 4 * np.mean(np.sqrt(1 - x**2))
print(f"Estimated pi: {pi_estimate:.4f}")  # Should be close to 3.1416
```

---

## Summary

| Concept | What It Means | ML Connection |
|---------|---------------|---------------|
| Definite integral | Area under curve = limit of Riemann sums | Computing probabilities from density functions |
| Fundamental theorem | Differentiation and integration are inverse operations | Connects gradient computation to accumulation |
| Expected value | $\mathbb{E}[g(X)] = \int g(x)f(x)\,dx$ | Loss functions, risk, decision theory |
| Numerical integration | `scipy.integrate.quad()`, `np.trapz()` | Practical computation of any integral |
| Monte Carlo integration | Approximate integrals via random sampling | Bayesian inference, variational methods, RL |
| Multiple integrals | Integrate over multiple variables | Joint distributions, marginal distributions |

Key equation you will use constantly in ML:

$$\mathbb{E}[\text{rating}] = \int x \cdot f(x)\,dx$$

This pattern — "multiply the thing you care about by the density, then integrate" — shows up everywhere: expected loss, entropy, KL divergence, ELBO, policy gradients. Master this one formula and you have the core of integral calculus for ML.

---

> **What's Next** — You now have the full calculus toolkit. Next: probability theory — where integrals become your primary tool for reasoning about uncertainty.
