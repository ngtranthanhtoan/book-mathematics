# Chapter 4: Expectation and Moments

Your model predicts a distribution over outcomes. But you need a single number -- the expected revenue, the average error, the mean prediction. Expected value distills an entire distribution into one number. It is the most important summary statistic in all of ML.

---

## Building On

Random variables give you distributions. But distributions have infinitely many values. How do you summarize them? With moments: mean, variance, and higher-order statistics. If Chapter 3 gave you the "shape" of uncertainty, this chapter gives you the tools to compress that shape into a handful of actionable numbers.

---

## Running Example: Movie Ratings

Throughout this chapter, we will work with a concrete scenario. You are building a recommendation system. Users rate movies on a 1-to-5 scale. You want to answer questions like:

- What is the expected rating for a given movie? (Expected value)
- How much do ratings vary? (Variance)
- If a user rates action movies highly, do they also rate comedies highly? (Covariance)

These are the exact computations happening inside collaborative filtering, matrix factorization, and every recommendation engine you have ever used.

---

## 1. Expected Value (Mean)

### The Idea

> **You Already Know This**: Expected value is a weighted average. You have done this a thousand times: `sum(value * weight for value, weight in zip(values, weights))`. If you have ever computed a weighted score -- say, averaging code review scores where senior reviewers count double -- you have computed an expected value.

The **expected value** of a random variable X is the probability-weighted average of all its possible values.

**Discrete case** (finite or countable outcomes):

$$\mathbb{E}[X] = \sum_x x \cdot P(X = x) = \sum_x x \cdot p(x)$$

**Continuous case** (uncountable outcomes):

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

**Notation you will encounter**: $\mathbb{E}[X]$, $\mu$, $\mu_X$, or $\langle X \rangle$.

### Running Example: Expected Movie Rating

Suppose a movie has the following rating distribution:

| Rating (x) | 1    | 2    | 3    | 4    | 5    |
|-------------|------|------|------|------|------|
| P(X = x)    | 0.05 | 0.10 | 0.20 | 0.35 | 0.30 |

```python
import numpy as np

# Movie rating distribution
ratings = np.array([1, 2, 3, 4, 5])
probs   = np.array([0.05, 0.10, 0.20, 0.35, 0.30])

# E[X] = sum(x * P(X=x))  -- this is just a weighted average
expected_rating = np.sum(ratings * probs)
print(f"Expected rating E[X] = {expected_rating:.2f}")
# Expected rating E[X] = 3.75
```

That 3.75 is the number your recommendation engine shows as the "average rating." It is a single scalar that summarizes the entire distribution.

### ASCII Diagram: Distribution With Mean Marked

```
  P(X=x)
  0.35 |            #
  0.30 |            #  #
  0.20 |         #  #  #
  0.10 |      #  #  #  #
  0.05 |   #  #  #  #  #
       +---+--+--+--+--+--
           1  2  3  4  5
                 |
                3.75
              E[X] = mu
```

The mean is the "center of mass" of the distribution. If you cut out the histogram from cardboard, $\mu$ is the balance point.

---

## 2. Properties of Expectation

### Linearity -- The Most Powerful Property

$$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$

This holds **even if X and Y are dependent**. Read that again. It does not matter how tangled the relationship between X and Y is. Linearity of expectation always holds. This is why it shows up everywhere in ML proofs.

### Expectation of a Function of X

If you apply a function $g$ to a random variable:

$$\mathbb{E}[g(X)] = \sum_x g(x) \cdot p(x) \quad \text{(discrete)}$$

$$\mathbb{E}[g(X)] = \int g(x) \cdot f(x) \, dx \quad \text{(continuous)}$$

> **Common Mistake -- Jensen's Inequality**: $\mathbb{E}[f(X)] \neq f(\mathbb{E}[X])$ in general. The expected value of $X^2$ is NOT the square of the expected value of $X$. Concretely: `E[X**2] != E[X]**2`. This is Jensen's inequality, and ignoring it is one of the most common bugs in probabilistic reasoning. The only exception is when $f$ is linear (affine).

```python
# Demonstrating Jensen's inequality with our movie ratings
E_X = np.sum(ratings * probs)           # E[X]
E_X_squared = np.sum(ratings**2 * probs) # E[X^2]

print(f"E[X]         = {E_X:.4f}")
print(f"E[X]^2       = {E_X**2:.4f}")
print(f"E[X^2]       = {E_X_squared:.4f}")
print(f"E[X^2] != E[X]^2 ? {E_X_squared != E_X**2}")
# E[X]         = 3.7500
# E[X]^2       = 14.0625
# E[X^2]       = 15.2500
# E[X^2] != E[X]^2 ? True
```

### Linearity in Action

```python
np.random.seed(42)
n = 100_000

# Simulate two user rating streams (possibly dependent)
X = np.random.normal(loc=3.75, scale=1.0, size=n)  # action movie ratings
Y = np.random.normal(loc=3.20, scale=1.2, size=n)  # comedy ratings

# E[X + Y] = E[X] + E[Y], always
print(f"E[X]     = {np.mean(X):.4f}")
print(f"E[Y]     = {np.mean(Y):.4f}")
print(f"E[X + Y] = {np.mean(X + Y):.4f}  (should be ~{3.75 + 3.20})")

# E[2X + 3] = 2*E[X] + 3
a, b = 2, 3
print(f"\nE[{a}X + {b}] = {np.mean(a*X + b):.4f}  (should be ~{a*3.75 + b})")

# Even works with DEPENDENT variables
Z = X + 0.5 * Y  # Z depends on both X and Y
print(f"\nZ = X + 0.5*Y (dependent on both)")
print(f"E[X + Z] = {np.mean(X + Z):.4f}  (should be ~{3.75 + 3.75 + 0.5*3.20})")
```

---

## 3. Variance

### The Idea

> **You Already Know This**: Variance measures spread, like the "jitter" in your API response times. If your p50 latency is 200ms but p99 is 2000ms, you have high variance. If p50 is 200ms and p99 is 210ms, you have low variance. Same mean, very different reliability.

Variance measures how spread out the distribution is around its mean:

$$\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Notation**: $\text{Var}(X)$, $\sigma^2$, or $\sigma_X^2$.

The second form -- $\mathbb{E}[X^2] - (\mathbb{E}[X])^2$ -- is the computational shortcut. It is why Jensen's inequality matters: the gap between $\mathbb{E}[X^2]$ and $(\mathbb{E}[X])^2$ IS the variance.

### Running Example: Variance of Movie Ratings

```python
# Continuing with our movie rating distribution
ratings = np.array([1, 2, 3, 4, 5])
probs   = np.array([0.05, 0.10, 0.20, 0.35, 0.30])

E_X = np.sum(ratings * probs)             # 3.75
E_X2 = np.sum(ratings**2 * probs)          # 15.25

# Var(X) = E[X^2] - (E[X])^2
var_X = E_X2 - E_X**2
print(f"E[X]    = {E_X:.4f}")
print(f"E[X^2]  = {E_X2:.4f}")
print(f"Var(X)  = {var_X:.4f}")
# Var(X) = 15.25 - 14.0625 = 1.1875

# This tells you: ratings are spread about 1.19 "squared-rating-units"
# around the mean. For interpretability, take the square root.
```

A variance of 1.1875 for a 1-to-5 scale means ratings are moderately spread. A movie where everyone gives it a 4 would have variance near zero. A polarizing movie (lots of 1s and 5s) would have high variance.

### ASCII Diagram: Same Mean, Different Variance

```
Low Variance (tight consensus):     High Variance (polarizing movie):

  P(x)                                P(x)
  0.6 |      #                        0.3 | #              #
  0.4 |   #  #  #                     0.2 | #  #     #  #  #
  0.2 |   #  #  #                     0.1 | #  #  #  #  #  #
      +--+--+--+--+--                     +--+--+--+--+--+--
         1  2  3  4  5                       1  2  3  4  5
              |                                   |
             mu                                  mu
         sigma^2 small                      sigma^2 large
```

Both distributions can have the same mean, but very different variances. In ML, this distinction is everything: a confident wrong prediction and an uncertain correct prediction look the same if you only check the mean.

### Properties of Variance

$$\text{Var}(aX + b) = a^2 \text{Var}(X)$$

Notice: adding a constant $b$ does not change variance (shifting a distribution left or right does not change its spread). Scaling by $a$ scales variance by $a^2$.

For **independent** X and Y:

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

For **dependent** X and Y, you need the covariance term:

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

Variance is always non-negative: $\text{Var}(X) \geq 0$, and it equals zero only when X is a constant.

```python
np.random.seed(42)
n = 100_000

X = np.random.normal(loc=0, scale=3, size=n)  # Var(X) ~ 9
Y = np.random.normal(loc=0, scale=2, size=n)  # Var(Y) ~ 4

# Var(aX + b) = a^2 * Var(X)
a, b = 2, 5
print(f"Var(X) = {np.var(X):.4f}  (should be ~9)")
print(f"Var({a}X + {b}) = {np.var(a*X + b):.4f}  (should be ~{a**2 * 9})")
print("Adding constant b=5 does NOT change variance.\n")

# For independent X, Y: Var(X + Y) = Var(X) + Var(Y)
print(f"Var(X) + Var(Y) = {np.var(X) + np.var(Y):.4f}  (should be ~13)")
print(f"Var(X + Y)      = {np.var(X + Y):.4f}\n")

# For dependent variables: need covariance
Z = X + 0.5 * Y  # Z depends on X
cov_XZ = np.cov(X, Z, ddof=0)[0, 1]
print(f"Var(X + Z) direct          = {np.var(X + Z):.4f}")
print(f"Var(X) + Var(Z) + 2Cov(X,Z) = {np.var(X) + np.var(Z) + 2*cov_XZ:.4f}")
```

---

## 4. Standard Deviation

$$\sigma = \sqrt{\text{Var}(X)}$$

> **You Already Know This**: Standard deviation is the "typical distance from the mean." Your p50 plus or minus one standard deviation gives you the range where most values fall. If you have ever looked at a monitoring dashboard and seen "mean +/- std," that is exactly this.

Standard deviation has the **same units as X**. Variance has squared units, which makes it hard to interpret. If ratings are on a 1-to-5 scale, variance is in "squared ratings" (meaningless), but standard deviation is in ratings (meaningful).

```python
ratings = np.array([1, 2, 3, 4, 5])
probs   = np.array([0.05, 0.10, 0.20, 0.35, 0.30])

E_X = np.sum(ratings * probs)
var_X = np.sum(ratings**2 * probs) - E_X**2
std_X = np.sqrt(var_X)

print(f"Mean rating:       {E_X:.2f}")
print(f"Variance:          {var_X:.4f} (squared rating units -- not intuitive)")
print(f"Standard deviation: {std_X:.4f} (rating units -- interpretable)")
print(f"\nTypical rating range: [{E_X - std_X:.2f}, {E_X + std_X:.2f}]")
# Typical rating range: [2.66, 4.84]
```

---

## 5. Covariance

### The Idea

Covariance measures how two random variables move together.

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

- **Positive covariance**: when X is above its mean, Y tends to be above its mean too.
- **Negative covariance**: when X is above its mean, Y tends to be below its mean.
- **Zero covariance**: no linear relationship.

### Running Example: Action vs. Comedy Ratings

In your recommendation system, you want to know: if a user rates action movies highly, do they also rate comedies highly?

```python
np.random.seed(42)
n_users = 10_000

# Simulate user ratings
# Some users love everything (positive correlation between genres)
# Some users are genre-specific
base_taste = np.random.normal(3.5, 0.8, n_users)  # general movie enjoyment
action_ratings = base_taste + np.random.normal(0.3, 0.7, n_users)  # action bonus
comedy_ratings = base_taste + np.random.normal(-0.1, 0.9, n_users)  # comedy offset

# Covariance: do action and comedy ratings co-move?
E_action = np.mean(action_ratings)
E_comedy = np.mean(comedy_ratings)
E_product = np.mean(action_ratings * comedy_ratings)

cov_manual = E_product - E_action * E_comedy
cov_numpy  = np.cov(action_ratings, comedy_ratings, ddof=0)[0, 1]

print(f"E[action]  = {E_action:.4f}")
print(f"E[comedy]  = {E_comedy:.4f}")
print(f"E[action * comedy] = {E_product:.4f}")
print(f"Cov(action, comedy) manual = {cov_manual:.4f}")
print(f"Cov(action, comedy) numpy  = {cov_numpy:.4f}")
print(f"\nPositive covariance: users who like action tend to also like comedy.")
print("This is because both share the 'base_taste' component.")
```

### Key Properties of Covariance

- $\text{Cov}(X, X) = \text{Var}(X)$ -- covariance of a variable with itself is its variance
- $\text{Cov}(X, Y) = \text{Cov}(Y, X)$ -- symmetric
- If X and Y are independent: $\text{Cov}(X, Y) = 0$
- **The converse is NOT true**: zero covariance does NOT imply independence

That last point is critical. Here is the classic counterexample:

```python
np.random.seed(42)
n = 10_000

# U is uniform on [-1, 1]
U = np.random.uniform(-1, 1, n)
# V = U^2 -- V is completely determined by U (maximally dependent!)
V = U**2

print(f"Cov(U, V)  = {np.cov(U, V, ddof=0)[0, 1]:.4f}  (near zero)")
print(f"Corr(U, V) = {np.corrcoef(U, V)[0, 1]:.4f}  (near zero)")
print(f"But V is a DETERMINISTIC function of U!")
print("Zero covariance != independence. It only means no LINEAR relationship.")
```

---

## 6. Correlation

**Correlation** is covariance normalized to the [-1, 1] range:

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

This is the number you actually want to look at, because raw covariance depends on the scales of X and Y.

- $\rho = 1$: perfect positive linear relationship
- $\rho = -1$: perfect negative linear relationship
- $\rho = 0$: no linear relationship (but there may be a nonlinear one!)

```python
np.random.seed(42)
n = 10_000

# Independent
X = np.random.normal(0, 1, n)
Y_indep = np.random.normal(0, 1, n)

# Positive linear relationship
Y_pos = X + np.random.normal(0, 0.5, n)

# Negative linear relationship
Y_neg = -X + np.random.normal(0, 0.5, n)

print("Independent:             rho =", f"{np.corrcoef(X, Y_indep)[0,1]:.4f}")
print("Positive relationship:   rho =", f"{np.corrcoef(X, Y_pos)[0,1]:.4f}")
print("Negative relationship:   rho =", f"{np.corrcoef(X, Y_neg)[0,1]:.4f}")
```

---

## 7. Higher Moments

> **You Already Know This**: Moments are summary statistics at different "resolutions." Mean tells you the center. Variance tells you the spread. Skewness tells you the asymmetry. Kurtosis tells you the tail heaviness. Each moment gives you a higher-resolution view of the distribution's shape.

### The Moment Hierarchy

```
Moment          Formula                              What it tells you
------          -------                              -----------------
1st (Mean)      E[X]                                 Center / location
2nd (Variance)  E[(X - mu)^2]                        Spread / volatility
3rd (Skewness)  E[((X - mu) / sigma)^3]              Asymmetry / lopsidedness
4th (Kurtosis)  E[((X - mu) / sigma)^4]              Tail heaviness / outlier-proneness
```

### Raw vs. Central vs. Standardized Moments

- **Raw moments**: $\mathbb{E}[X^n]$ -- moments about zero
- **Central moments**: $\mathbb{E}[(X - \mu)^n]$ -- moments about the mean
- **Standardized moments**: $\mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^n\right]$ -- unitless, scale-invariant

### Skewness (3rd Standardized Moment)

Measures asymmetry of the distribution:

$$\gamma_1 = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^3\right]$$

- $\gamma_1 > 0$: right-skewed (long right tail) -- e.g., income distributions, API latencies
- $\gamma_1 < 0$: left-skewed (long left tail) -- e.g., exam scores with a hard ceiling
- $\gamma_1 = 0$: symmetric -- e.g., normal distribution

### Kurtosis (4th Standardized Moment)

Measures how heavy the tails are:

$$\gamma_2 = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^4\right]$$

The normal distribution has kurtosis = 3. "Excess kurtosis" subtracts 3, so the normal has excess kurtosis = 0. Heavy-tailed distributions (more outlier-prone) have positive excess kurtosis.

```python
from scipy import stats

np.random.seed(42)
n = 100_000

# Different distributions with distinct moment profiles
normal_samples    = stats.norm.rvs(size=n)
right_skewed      = stats.expon.rvs(size=n)        # like API latency
left_skewed       = -stats.expon.rvs(size=n)        # reflected
heavy_tailed      = stats.t.rvs(df=3, size=n)       # like financial returns

distributions = [
    ("Normal (baseline)",       normal_samples),
    ("Right-skewed (Exp)",      right_skewed),
    ("Left-skewed (-Exp)",      left_skewed),
    ("Heavy-tailed (t, df=3)",  heavy_tailed),
]

print(f"{'Distribution':<28} {'Skewness':>10} {'Excess Kurtosis':>17}")
print("-" * 57)
for name, samples in distributions:
    skew = stats.skew(samples)
    kurt = stats.kurtosis(samples)  # scipy gives excess kurtosis by default
    print(f"{name:<28} {skew:>10.4f} {kurt:>17.4f}")

# Interpretation:
# Normal: skew ~ 0, kurtosis ~ 0  (the reference)
# Exponential: positive skew, positive kurtosis (right tail, more outliers)
# t-distribution: skew ~ 0 but HIGH kurtosis (extreme outliers)
```

---

## 8. ML Applications: Where This All Shows Up

### Loss Functions Are Expected Values

Every standard loss function is an expectation:

- **MSE**: $\mathbb{E}[(y - \hat{y})^2]$
- **Cross-entropy**: $\mathbb{E}[-\log p(\hat{y})]$
- **Hinge loss**: $\mathbb{E}[\max(0, 1 - y \cdot \hat{y})]$

When you call `loss.backward()` in PyTorch, you are computing the gradient of an expected value.

### Variance in ML

- **Bias-Variance Tradeoff**: $\mathbb{E}[\text{Error}] = \text{Bias}^2 + \text{Variance} + \text{Noise}$
- **Batch Normalization**: Uses batch estimates of $\mathbb{E}[X]$ and $\text{Var}(X)$
- **Feature Scaling**: Standardization divides by $\sigma$
- **PCA**: Principal components are ordered by variance explained
- **Dropout**: Injects variance as approximate regularization

### The ADAM Optimizer: Moments in Action

ADAM literally uses the first and second moments of gradients:

- First moment estimate (mean of gradients): $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
- Second moment estimate (mean of squared gradients): $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$

The update is $\theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$. This is adaptive learning rate by normalizing the gradient (first moment) by its volatility (square root of second moment).

### Running Example: Movie Rating Prediction Loss

```python
np.random.seed(42)

# Your recommendation model predicts ratings for 1000 users
n_users = 1000
true_ratings = np.random.choice([1, 2, 3, 4, 5], size=n_users,
                                 p=[0.05, 0.10, 0.20, 0.35, 0.30])
true_ratings = true_ratings.astype(float)

# Model A: predicts the mean for everyone (baseline)
pred_A = np.full(n_users, 3.75)

# Model B: a trained model with some noise
pred_B = true_ratings + np.random.normal(0, 0.5, n_users)

# MSE = E[(y - y_hat)^2]
mse_A = np.mean((true_ratings - pred_A)**2)
mse_B = np.mean((true_ratings - pred_B)**2)

print("Movie Rating Prediction -- MSE as Expected Value")
print(f"Model A (always predict mean): MSE = {mse_A:.4f}")
print(f"Model B (trained model):       MSE = {mse_B:.4f}")

# Bias-Variance decomposition intuition
print(f"\nModel A: high bias (ignores user), zero variance in predictions")
print(f"Model B: low bias (tracks truth), some variance from noise")
print(f"\nBias^2 + Variance + Noise = Total Error")
bias_A = np.mean(pred_A) - np.mean(true_ratings)
bias_B = np.mean(pred_B) - np.mean(true_ratings)
print(f"Model A bias: {bias_A:.4f}, prediction variance: {np.var(pred_A):.4f}")
print(f"Model B bias: {bias_B:.4f}, prediction variance: {np.var(pred_B):.4f}")
```

---

## 9. Putting It All Together: Full Exploration

Here is a complete data exploration of our movie rating example, computing every moment and relationship we have discussed.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# ==========================================================================
# Full movie rating analysis
# ==========================================================================

# Rating distributions for two genres
action_probs = np.array([0.05, 0.10, 0.20, 0.35, 0.30])  # action movies
comedy_probs = np.array([0.10, 0.15, 0.30, 0.25, 0.20])  # comedies
ratings = np.array([1, 2, 3, 4, 5])

print("=" * 60)
print("MOVIE RATING ANALYSIS: Expectation and Moments")
print("=" * 60)

# --- Expected Values ---
E_action = np.sum(ratings * action_probs)
E_comedy = np.sum(ratings * comedy_probs)
print(f"\n--- Expected Ratings ---")
print(f"E[action]  = {E_action:.4f}")
print(f"E[comedy]  = {E_comedy:.4f}")

# --- Variances ---
var_action = np.sum(ratings**2 * action_probs) - E_action**2
var_comedy = np.sum(ratings**2 * comedy_probs) - E_comedy**2
std_action = np.sqrt(var_action)
std_comedy = np.sqrt(var_comedy)

print(f"\n--- Variance and Std Dev ---")
print(f"Var(action) = {var_action:.4f},  Std(action) = {std_action:.4f}")
print(f"Var(comedy) = {var_comedy:.4f},  Std(comedy) = {std_comedy:.4f}")
print(f"Action range (mu +/- sigma): [{E_action - std_action:.2f}, {E_action + std_action:.2f}]")
print(f"Comedy range (mu +/- sigma): [{E_comedy - std_comedy:.2f}, {E_comedy + std_comedy:.2f}]")

# --- Simulate paired user ratings for covariance ---
n_users = 50_000
base_taste = np.random.normal(0, 0.8, n_users)
action_samples = np.random.choice(ratings, size=n_users, p=action_probs).astype(float)
action_samples += base_taste  # shared taste component
comedy_samples = np.random.choice(ratings, size=n_users, p=comedy_probs).astype(float)
comedy_samples += base_taste  # same shared taste

cov_ac = np.cov(action_samples, comedy_samples, ddof=0)[0, 1]
corr_ac = np.corrcoef(action_samples, comedy_samples)[0, 1]

print(f"\n--- Covariance and Correlation (Action vs. Comedy) ---")
print(f"Cov(action, comedy)  = {cov_ac:.4f}")
print(f"Corr(action, comedy) = {corr_ac:.4f}")
print(f"Positive correlation: users with good taste rate both genres higher.")

# --- Higher Moments ---
print(f"\n--- Higher Moments (from simulated samples) ---")
print(f"{'Genre':<10} {'Skewness':>10} {'Excess Kurtosis':>17}")
print("-" * 40)
for name, samples in [("Action", action_samples), ("Comedy", comedy_samples)]:
    skew = stats.skew(samples)
    kurt = stats.kurtosis(samples)
    print(f"{name:<10} {skew:>10.4f} {kurt:>17.4f}")

# --- Jensen's Inequality Reminder ---
print(f"\n--- Jensen's Inequality Check ---")
E_X = np.mean(action_samples)
E_X2 = np.mean(action_samples**2)
print(f"E[X]^2  = {E_X**2:.4f}")
print(f"E[X^2]  = {E_X2:.4f}")
print(f"Gap     = {E_X2 - E_X**2:.4f}  (this gap IS the variance)")
```

---

## 10. Common Mistakes

Here are the mistakes that trip up even experienced engineers.

**1. E[f(X)] != f(E[X]) -- Jensen's Inequality**

This is the single most common error. If your loss function is nonlinear (and it almost always is), you cannot just plug the mean into the function. You must compute the expectation over the full distribution.

**2. Confusing sample and population statistics**

- Population mean: $\mu = \mathbb{E}[X]$ (theoretical, usually unknown)
- Sample mean: $\bar{x} = \frac{1}{n}\sum x_i$ (what you compute from data)
- For variance: use $\frac{1}{n-1}$ (Bessel's correction) for an unbiased estimator of population variance. NumPy's `np.var()` uses $\frac{1}{n}$ by default -- pass `ddof=1` for the unbiased version.

**3. Var(X + Y) = Var(X) + Var(Y) -- only if independent**

If X and Y are dependent, you must include the covariance term: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$. Forgetting this is a classic bug in error propagation.

**4. Zero covariance implies independence -- FALSE**

Zero covariance means no linear relationship. There can be a perfect nonlinear relationship (like $V = U^2$) and covariance will still be zero.

**5. Ignoring units**

Variance has squared units. If X is in meters, $\text{Var}(X)$ is in meters-squared. Use standard deviation for interpretable results.

---

## Exercises

### Exercise 1: Expected Value of Movie Ratings

A new movie has the following rating distribution: P(1)=0.15, P(2)=0.20, P(3)=0.30, P(4)=0.25, P(5)=0.10. What is the expected rating? Is this movie above or below the platform average of 3.75?

**Solution**:
```python
import numpy as np

ratings = np.array([1, 2, 3, 4, 5])
probs   = np.array([0.15, 0.20, 0.30, 0.25, 0.10])

E_X = np.sum(ratings * probs)
print(f"E[X] = {E_X:.2f}")  # 2.95
print(f"Below platform average of 3.75 by {3.75 - E_X:.2f} points")
```

### Exercise 2: Variance From Definition

For a movie where P(1)=0.25, P(3)=0.50, P(5)=0.25, compute the variance and standard deviation. Is this a polarizing movie?

**Solution**:
```python
import numpy as np

values = np.array([1, 3, 5])
probs  = np.array([0.25, 0.50, 0.25])

E_X = np.sum(values * probs)          # 3.0
E_X2 = np.sum(values**2 * probs)       # 0.25*1 + 0.50*9 + 0.25*25 = 11.0
var_X = E_X2 - E_X**2                  # 11.0 - 9.0 = 2.0
std_X = np.sqrt(var_X)

print(f"E[X]    = {E_X:.2f}")          # 3.00
print(f"E[X^2]  = {E_X2:.2f}")         # 11.00
print(f"Var(X)  = {var_X:.2f}")         # 2.00
print(f"Std(X)  = {std_X:.4f}")         # 1.4142

# Variance of 2.0 on a 1-5 scale is fairly high.
# The movie has no 2s or 4s -- people either hate it or love it.
# This is a polarizing movie.
```

### Exercise 3: Covariance Between Genres

Given the following joint distribution of action (X) and comedy (Y) ratings (simplified to Low=0, High=1):

| | Y=0 (low comedy) | Y=1 (high comedy) |
|---|---|---|
| X=0 (low action) | 0.20 | 0.30 |
| X=1 (high action) | 0.40 | 0.10 |

Compute Cov(X, Y). What does the sign tell you?

**Solution**:
```python
import numpy as np

# Joint distribution
joint = np.array([[0.20, 0.30],   # X=0
                  [0.40, 0.10]])  # X=1

X_vals = np.array([0, 1])
Y_vals = np.array([0, 1])

# Marginals
P_X = joint.sum(axis=1)  # [0.50, 0.50]
P_Y = joint.sum(axis=0)  # [0.60, 0.40]

# Expected values
E_X = np.sum(X_vals * P_X)  # 0.5
E_Y = np.sum(Y_vals * P_Y)  # 0.4

# E[XY]
E_XY = 0
for i, x in enumerate(X_vals):
    for j, y in enumerate(Y_vals):
        E_XY += x * y * joint[i, j]
# Only (1,1) contributes: 1*1*0.10 = 0.10

# Cov(X,Y) = E[XY] - E[X]*E[Y]
cov = E_XY - E_X * E_Y
print(f"E[X]      = {E_X}")       # 0.5
print(f"E[Y]      = {E_Y}")       # 0.4
print(f"E[XY]     = {E_XY}")      # 0.1
print(f"Cov(X,Y)  = {cov}")       # 0.1 - 0.2 = -0.1

# Negative covariance: users who rate action highly tend to rate comedy LOW.
# These are genre-specific users, not "love everything" users.
```

---

## Summary

| Concept | Formula | One-liner |
|---------|---------|-----------|
| **Expected Value** | $\mathbb{E}[X] = \sum x \cdot p(x)$ or $\int x \cdot f(x) \, dx$ | Center of the distribution |
| **Linearity** | $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$ | Always holds, even for dependent variables |
| **Variance** | $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | Spread around the mean |
| **Standard Deviation** | $\sigma = \sqrt{\text{Var}(X)}$ | Same units as X; interpretable spread |
| **Covariance** | $\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$ | Linear co-movement |
| **Correlation** | $\rho = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} \in [-1, 1]$ | Normalized covariance |
| **Skewness** | $\mathbb{E}\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]$ | Asymmetry (positive = right tail) |
| **Kurtosis** | $\mathbb{E}\left[\left(\frac{X-\mu}{\sigma}\right)^4\right]$ | Tail heaviness (normal = 3) |

**The key identities to internalize**:
- Linearity of expectation: always works, never needs independence.
- Variance of a sum: needs independence OR explicit covariance terms.
- Jensen's inequality: $\mathbb{E}[f(X)] \neq f(\mathbb{E}[X])$ unless $f$ is linear.
- Zero covariance does not imply independence.

---

## What's Next

You can compute expectations and variances. But which distributions actually show up in practice? Gaussian, Bernoulli, Poisson -- the common distributions are the building blocks of every ML model. In the next chapter, we catalog each one, show you its moments, and explain where it appears in ML.

**Next**: [Chapter 5: Common Distributions](05-common-distributions.md)
