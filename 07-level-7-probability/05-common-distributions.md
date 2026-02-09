# Chapter 5: Common Probability Distributions

Every ML model makes assumptions about how data is distributed. Logistic regression assumes Bernoulli outputs. Linear regression assumes Gaussian noise. VAEs use the "reparameterization trick" on Gaussians. Knowing the common distributions isn't trivia -- it's knowing which Lego bricks your models are built from.

---

**Building On** -- You can compute expectations and variances for any distribution. Now meet the specific distributions that power ML: Bernoulli, Gaussian, Poisson, and friends.

---

## Running Example: A Streaming Platform

Throughout this chapter, we will ground every distribution in a concrete scenario you can picture:

- **Movie ratings** follow a **Beta distribution** (bounded between 0 and 1, often skewed).
- **Click-through rates** on the "Play" button are **Bernoulli** -- each user either clicks or doesn't.
- **Daily active users** might be **Poisson** -- a count of events in a time window.
- **Time between server crashes** is **Exponential** -- memoryless waiting.
- **Average session length** across thousands of users converges to **Gaussian** -- thanks to the Central Limit Theorem.

Keep these in mind as we work through each distribution.

---

## Distribution Family Map

Before diving in, here is how these distributions relate to each other:

```
                   ┌──────────────────────────────────────────┐
                   │         DISCRETE DISTRIBUTIONS           │
                   │                                          │
                   │  Bernoulli ──(repeat n times)──> Binomial│
                   │     │                              │     │
                   │     │                              │     │
                   │     │                     (n→∞,p→0)│     │
                   │     │                              v     │
                   │     │                           Poisson  │
                   └─────┼────────────────────────────┼───────┘
                         │                            │
                         │       (CLT / limits)       │
                         v                            v
                   ┌──────────────────────────────────────────┐
                   │        CONTINUOUS DISTRIBUTIONS          │
                   │                                          │
                   │  Uniform    Exponential    Gaussian       │
                   │             (time between   (bell curve)  │
                   │              Poisson events)              │
                   └──────────────────────────────────────────┘
```

---

## 1. Bernoulli Distribution

### ML Use Case

Logistic regression, the workhorse of binary classification, models each label as a Bernoulli random variable. When your model outputs `P(spam) = 0.87`, it is saying "this email is drawn from a Bernoulli with p = 0.87." Dropout layers? Each neuron's keep/drop decision is an independent Bernoulli trial.

> **You Already Know This** -- A Bernoulli is just a boolean: `True` with probability `p`, `False` with probability `1 - p`. Think of a coin flip, a feature flag that is randomly enabled for `p` fraction of users, or a single click/no-click event.

### ASCII Visualization

```
  Bernoulli(p = 0.3)            Bernoulli(p = 0.7)

  P(X=k)                        P(X=k)
  |                              |
  |                              |
 0.7 ████                       0.7      ████
  |  ████                        |       ████
 0.3      ████                  0.3 ████
  |       ████                   |  ████
  +---+------+---                +---+------+---
      0      1                       0      1
      (fail) (success)               (fail) (success)
```

### Code Exploration

```python
import numpy as np
from scipy import stats

# ----- Bernoulli: Click-through on "Play" button -----
p_click = 0.12  # 12% click-through rate
bernoulli = stats.bernoulli(p_click)

# Simulate 10,000 users visiting the page
np.random.seed(42)
clicks = bernoulli.rvs(size=10_000)

print(f"Bernoulli(p={p_click})")
print(f"P(click)   = {bernoulli.pmf(1):.4f}")
print(f"P(no click)= {bernoulli.pmf(0):.4f}")
print(f"Simulated CTR: {clicks.mean():.4f}")
print(f"Mean (theory): {bernoulli.mean():.4f}")
print(f"Var  (theory): {bernoulli.var():.4f}")
```

### Math Formalization

**PMF (Probability Mass Function)**:

$$P(X = k) = p^k (1-p)^{1-k}, \quad k \in \{0, 1\}$$

Or more plainly: $P(X=1) = p$, $P(X=0) = 1-p$.

**Parameter**: $p \in [0, 1]$ -- the probability of success.

### Properties

- **Mean**: $\mathbb{E}[X] = p$
- **Variance**: $\text{Var}(X) = p(1-p)$
- Maximum variance occurs at $p = 0.5$ (maximum uncertainty).
- The Bernoulli is the building block: every other discrete distribution in this chapter is constructed from Bernoulli trials.

---

## 2. Binomial Distribution

### ML Use Case

A/B testing is built on the Binomial. You show a new UI to 1,000 users, and you count how many convert. That count is Binomial(n=1000, p). Ensemble methods that take a majority vote among `n` classifiers? The vote count follows a Binomial distribution.

> **You Already Know This** -- Binomial = counting successes in `n` independent trials. Like asking "how many of 100 HTTP requests succeed?" or "how many of 50 deploys trigger a rollback this quarter?"

### ASCII Visualization

```
  Binomial(n=10, p=0.3)                 Binomial(n=20, p=0.5)

  P(X=k)                                P(X=k)
  |                                      |
 0.27 ██                                0.18       ██
  |   ██ ██                              |      ██ ██ ██
 0.20 ██ ██                             0.12   ██ ██ ██ ██
  |   ██ ██ ██                           |  ██ ██ ██ ██ ██ ██
 0.10 ██ ██ ██ ██                       0.06 ██ ██ ██ ██ ██ ██ ██
  |██ ██ ██ ██ ██ ██                     |██ ██ ██ ██ ██ ██ ██ ██ ██
  +--+--+--+--+--+--+--+--+--+--+--     +--+--+--+--+--+--+--+--+--+--+--
     0  1  2  3  4  5  6  7  8  9 10       5  6  7  8  9 10 11 12 13 14 15
                  k                                      k
```

### Code Exploration

```python
from scipy import stats

# ----- Binomial: A/B Testing -----
# 100 users see a new recommendation banner. Baseline conversion = 10%.
n_users = 100
p_baseline = 0.10
binomial = stats.binom(n=n_users, p=p_baseline)

print(f"Binomial(n={n_users}, p={p_baseline})")
print(f"Expected conversions: {binomial.mean():.1f}")
print(f"Std dev: {binomial.std():.2f}")

# We observe 15 conversions. Is this surprisingly high?
observed = 15
p_value = 1 - binomial.cdf(observed - 1)  # P(X >= 15)
print(f"\nObserved: {observed} conversions")
print(f"P(X >= {observed} | p={p_baseline}) = {p_value:.4f}")
print(f"Significant at alpha=0.05? {'Yes' if p_value < 0.05 else 'No'}")

# PMF for key values
print("\nPMF for selected k:")
for k in [5, 8, 10, 12, 15, 20]:
    print(f"  P(X={k:2d}) = {binomial.pmf(k):.4f}")

# Cumulative probabilities
print(f"\nP(X <= 12) = {binomial.cdf(12):.4f}")
print(f"P(8 <= X <= 15) = {binomial.cdf(15) - binomial.cdf(7):.4f}")
```

### Math Formalization

**PMF**:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k \in \{0, 1, \ldots, n\}$$

**Parameters**: $n$ (number of trials), $p \in [0,1]$ (success probability per trial).

**Relationship to Bernoulli**: If $X_1, X_2, \ldots, X_n$ are independent $\text{Bernoulli}(p)$ random variables, then $X = \sum_{i=1}^n X_i \sim \text{Binomial}(n, p)$.

### Properties

- **Mean**: $\mathbb{E}[X] = np$
- **Variance**: $\text{Var}(X) = np(1-p)$
- As $n \to \infty$ with $p$ fixed, the Binomial approaches a Gaussian (this is why the normal approximation works for large A/B tests).
- As $n \to \infty$ and $p \to 0$ with $np = \lambda$ held constant, the Binomial approaches a Poisson (rare events regime).

---

## 3. Poisson Distribution

### ML Use Case

Poisson regression models count data -- how many words appear in a document, how many items a user purchases, how many anomalies a monitoring system flags per hour. In NLP, word frequencies in a fixed-length text window are often modeled as Poisson. Any time your target variable is a non-negative integer with no hard upper bound, Poisson is your first candidate.

> **You Already Know This** -- Poisson counts events in a time window. Think "requests per second hitting your load balancer," "errors per hour in your logging dashboard," or "Slack messages per day in #incidents."

### ASCII Visualization

```
  Poisson(lambda=2)              Poisson(lambda=5)

  P(X=k)                        P(X=k)
  |                              |
 0.27 ██                        0.18          ██
  |   ██ ██                      |         ██ ██
 0.18 ██ ██                     0.14      ██ ██ ██
  |██ ██ ██                      |     ██ ██ ██ ██
 0.09 ██ ██ ██ ██               0.07   ██ ██ ██ ██ ██
  |██ ██ ██ ██ ██ ██             |  ██ ██ ██ ██ ██ ██ ██
  +--+--+--+--+--+--+--+--      +--+--+--+--+--+--+--+--+--+--+--
     0  1  2  3  4  5  6  7        0  1  2  3  4  5  6  7  8  9 10
                k                                 k

  Note: mean = variance = lambda (a unique Poisson signature)
```

### Code Exploration

```python
from scipy import stats
import numpy as np

# ----- Poisson: Errors per hour in production -----
lambda_errors = 3  # average 3 errors per hour
poisson = stats.poisson(mu=lambda_errors)

print(f"Poisson(lambda={lambda_errors})")
print(f"Mean = {poisson.mean():.4f}")
print(f"Var  = {poisson.var():.4f}")
print("Notice: mean == variance -- the Poisson signature!\n")

# PMF
print("PMF values:")
for k in range(10):
    bar = "█" * int(poisson.pmf(k) * 100)
    print(f"  P(X={k}) = {poisson.pmf(k):.4f}  {bar}")

# Operational questions
print(f"\nP(zero errors this hour)     = {poisson.pmf(0):.4f}")
print(f"P(more than 5 errors)        = {1 - poisson.cdf(5):.4f}")

# Scaling: errors per day (24 hours) ~ Poisson(3 * 24 = 72)
daily = stats.poisson(mu=lambda_errors * 24)
print(f"\nDaily errors: Poisson(lambda={lambda_errors * 24})")
print(f"P(daily errors > 80) = {1 - daily.cdf(80):.4f}")
```

### Math Formalization

**PMF**:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k \in \{0, 1, 2, \ldots\}$$

**Parameter**: $\lambda > 0$ (rate parameter = expected count).

**Key identity**: Mean equals variance. If your observed data has variance much larger than its mean, Poisson is the wrong model -- consider Negative Binomial instead.

### Properties

- **Mean**: $\mathbb{E}[X] = \lambda$
- **Variance**: $\text{Var}(X) = \lambda$
- **Additivity**: If $X \sim \text{Poisson}(\lambda_1)$ and $Y \sim \text{Poisson}(\lambda_2)$ are independent, then $X + Y \sim \text{Poisson}(\lambda_1 + \lambda_2)$. This is why you can scale from "errors per hour" to "errors per day."
- **Poisson limit of Binomial**: When $n$ is large and $p$ is small, $\text{Binomial}(n, p) \approx \text{Poisson}(np)$.

---

## 4. Uniform Distribution

### ML Use Case

Before Xavier/He initialization existed, neural network weights were initialized from a Uniform distribution. Random search for hyperparameter tuning draws candidates uniformly. In Bayesian inference, a Uniform prior says "I have no idea which value is more likely" -- the principle of maximum ignorance. The `random()` function you call every day returns $\text{Uniform}(0, 1)$.

### ASCII Visualization

```
  Uniform(a=0, b=1)             Uniform(a=2, b=8)

  f(x)                          f(x)
  |                              |
 1.0 ████████████████████       0.167 ████████████████████
  |  ████████████████████        |    ████████████████████
  |  ████████████████████        |    ████████████████████
  |  ████████████████████        |    ████████████████████
  +--+------------------+--      +----+------------------+--
     0                  1             2                  8

  Height = 1/(b - a)    <-- must integrate to 1
```

### Code Exploration

```python
from scipy import stats

# ----- Uniform: Random hyperparameter search -----
# Search learning rate in [0.0001, 0.01]
a, b = 0.0001, 0.01
uniform = stats.uniform(loc=a, scale=b - a)  # scipy parametrization!

print(f"Uniform(a={a}, b={b})")
print(f"Mean = {uniform.mean():.6f}  (should be {(a+b)/2:.6f})")
print(f"Var  = {uniform.var():.10f}  (should be {(b-a)**2/12:.10f})")

# PDF is constant
print(f"\nPDF (constant) = {uniform.pdf(0.005):.4f} = 1/(b-a) = {1/(b-a):.4f}")

# Probability of landing in a sub-interval
print(f"\nP(0.001 < lr < 0.005) = {uniform.cdf(0.005) - uniform.cdf(0.001):.4f}")

# Generate 5 candidate learning rates
np.random.seed(42)
candidates = uniform.rvs(size=5)
print(f"\n5 random learning rates: {[f'{x:.6f}' for x in candidates]}")
```

### Math Formalization

**PDF**:

$$f(x) = \frac{1}{b-a}, \quad x \in [a, b]$$

**CDF**:

$$F(x) = \frac{x-a}{b-a}, \quad x \in [a, b]$$

**Parameters**: $a$ (minimum), $b$ (maximum).

### Properties

- **Mean**: $\mathbb{E}[X] = \frac{a+b}{2}$
- **Variance**: $\text{Var}(X) = \frac{(b-a)^2}{12}$
- Maximum entropy distribution on a bounded interval -- the "most uncertain" you can be when you only know the range.
- **Quantiles**: The $q$-th quantile is simply $a + q(b - a)$. Linear and predictable.

---

## 5. Exponential Distribution

### ML Use Case

Survival analysis models (used in churn prediction, medical AI, and predictive maintenance) are built on the Exponential distribution. "How long until a user churns?" "How long until this hard drive fails?" If events arrive as a Poisson process, the waiting time between events is Exponential. It is the continuous-time cousin of the geometric distribution.

### ASCII Visualization

```
  Exponential(lambda=1)          Exponential(lambda=0.5)

  f(x)                           f(x)
  |                               |
 1.0 █                           0.5 ████
  |  ██                           |  ██████
  |  ████                         |  ████████
  |  ██████                       |  ████████████
  |  ████████                     |  ████████████████
  |  ██████████████               |  ████████████████████████
  |  ████████████████████____     |  ████████████████████████████████____
  +--+--+--+--+--+--+--+--+--    +--+--+--+--+--+--+--+--+--+--+--+--
     0  1  2  3  4  5  6  7  8       0  1  2  3  4  5  6  7  8  9 10 11

  Rapid decay (high lambda)       Slow decay (low lambda)
  Mean = 1/lambda                 Mean = 1/lambda
```

### Code Exploration

```python
from scipy import stats

# ----- Exponential: Time between server crashes -----
# Average 1 crash per 30 days => lambda = 1/30, mean = 30 days
mean_days = 30
lambda_rate = 1 / mean_days
exponential = stats.expon(scale=mean_days)  # scipy uses scale = 1/lambda

print(f"Exponential(lambda={lambda_rate:.4f}, mean={mean_days} days)")
print(f"Mean = {exponential.mean():.2f} days")
print(f"Var  = {exponential.var():.2f} days^2")
print(f"Std  = {exponential.std():.2f} days")

# Operational questions
print(f"\nP(crash within 7 days)  = {exponential.cdf(7):.4f}")
print(f"P(survive > 60 days)   = {1 - exponential.cdf(60):.4f}")

# Memoryless property -- this is the key insight
print("\n--- Memoryless Property ---")
print("You've survived 20 days. P(crash before day 27 | survived 20):")
p_cond = (exponential.cdf(27) - exponential.cdf(20)) / (1 - exponential.cdf(20))
print(f"  P(X < 27 | X > 20) = {p_cond:.4f}")
print(f"  P(X < 7)           = {exponential.cdf(7):.4f}")
print("  They're equal! The past doesn't matter.")

# Connection to Poisson
print(f"\n--- Poisson Connection ---")
print(f"If crashes ~ Poisson(rate={lambda_rate:.4f}/day),")
print(f"then time between crashes ~ Exponential(mean={mean_days} days)")
```

### Math Formalization

**PDF**:

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

**CDF**:

$$F(x) = 1 - e^{-\lambda x}, \quad x \geq 0$$

**Parameter**: $\lambda > 0$ (rate parameter). Equivalently, mean $= 1/\lambda$.

**Memoryless property** (unique to the Exponential among continuous distributions):

$$P(X > s + t \mid X > s) = P(X > t)$$

### Properties

- **Mean**: $\mathbb{E}[X] = \frac{1}{\lambda}$
- **Variance**: $\text{Var}(X) = \frac{1}{\lambda^2}$
- The only continuous memoryless distribution.
- Dual of Poisson: if events arrive at rate $\lambda$ (Poisson process), the inter-arrival times are $\text{Exponential}(\lambda)$.

> **Watch out for scipy's parametrization**: `stats.expon(scale=...)` takes `scale = 1/lambda`, not `lambda` directly. This bites everyone at least once.

---

## 6. Gaussian (Normal) Distribution

### ML Use Case

The Gaussian is everywhere in ML. Linear regression assumes the residuals are Gaussian. Gaussian Processes place a Gaussian prior over entire functions. Batch normalization pushes activations toward a Gaussian. VAEs encode data into a Gaussian latent space and use the "reparameterization trick" ($z = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0,1)$) to backpropagate through the sampling step. Weight initialization (Xavier, He) draws from scaled Gaussians.

> **You Already Know This** -- The Central Limit Theorem says that averages of *anything* (response times, request sizes, error counts) converge to a Gaussian as sample size grows. Your system's load average is approximately Gaussian. So are the means in your A/B test dashboards.

### ASCII Visualization

```
  Gaussian(mu=0, sigma=1)  -- Standard Normal

  f(x)
  |
 0.40          ████
  |          ████████
 0.30      ████████████
  |      ████████████████
 0.20   ████████████████████
  |   ████████████████████████
 0.10 ██████████████████████████████
  |████████████████████████████████████
  +--+--+--+--+--+--+--+--+--+--+--+--
    -3    -2    -1     0     1     2     3

  |-------- 68% --------|     (mu +/- 1*sigma)
  |------------ 95% -----------|  (mu +/- 2*sigma)
  |--------------- 99.7% --------------|  (mu +/- 3*sigma)
```

### Code Exploration

```python
from scipy import stats
import numpy as np

# ----- Gaussian: Session length on our streaming platform -----
# Average session = 45 min, std dev = 12 min
mu, sigma = 45, 12
normal = stats.norm(loc=mu, scale=sigma)

print(f"Normal(mu={mu}, sigma={sigma})")
print(f"Mean = {normal.mean():.2f}")
print(f"Var  = {normal.var():.2f}")
print(f"Std  = {normal.std():.2f}")

# The 68-95-99.7 rule
print("\n--- 68-95-99.7 Rule ---")
for n_sigma, label in [(1, "68%"), (2, "95%"), (3, "99.7%")]:
    lo, hi = mu - n_sigma * sigma, mu + n_sigma * sigma
    prob = normal.cdf(hi) - normal.cdf(lo)
    print(f"  P({lo} < X < {hi}) = {prob:.4f}  ({label})")

# Percentiles -- useful for alerting thresholds
print("\n--- Percentiles (for SLO thresholds) ---")
for p in [0.50, 0.90, 0.95, 0.99]:
    print(f"  p{int(p*100):2d} = {normal.ppf(p):.1f} min")

# Z-scores: standardization
print("\n--- Z-scores ---")
x_val = 70
z = (x_val - mu) / sigma
print(f"  Session of {x_val} min => Z = {z:.2f}")
print(f"  P(session > {x_val} min) = {1 - normal.cdf(x_val):.4f}")

# Central Limit Theorem in action
print("\n--- CLT Demo ---")
np.random.seed(42)
# Exponential (heavily skewed) session data
raw_sessions = stats.expon(scale=45).rvs(size=100_000)
# Take means of groups of 30
means_of_30 = [raw_sessions[i:i+30].mean() for i in range(0, 90_000, 30)]
print(f"  Raw data (exponential): skewness = {stats.skew(raw_sessions):.2f}")
print(f"  Means of 30:            skewness = {stats.skew(means_of_30):.2f}")
print(f"  (Much closer to 0 = Gaussian-like)")
```

### Math Formalization

**PDF**:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Parameters**: $\mu \in \mathbb{R}$ (mean), $\sigma > 0$ (standard deviation).

**Standard Normal**: When $\mu = 0$ and $\sigma = 1$, we write $Z \sim \mathcal{N}(0, 1)$.

**Standardization**: Any Gaussian can be converted to the standard normal via $Z = \frac{X - \mu}{\sigma}$.

**The 68-95-99.7 Rule**:
- 68% of data within $\mu \pm \sigma$
- 95% within $\mu \pm 2\sigma$
- 99.7% within $\mu \pm 3\sigma$

### Properties

- **Mean**: $\mathbb{E}[X] = \mu$
- **Variance**: $\text{Var}(X) = \sigma^2$
- **Closed under linear transformations**: If $X \sim \mathcal{N}(\mu, \sigma^2)$, then $aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$.
- **Sum of independent Gaussians**: If $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are independent, then $X + Y \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$.
- **Maximum entropy**: Among all distributions with a given mean and variance, the Gaussian has the highest entropy -- it is the "least informative" choice.

**Why the Gaussian dominates ML**:

1. **Central Limit Theorem** -- The sum (or average) of many independent random variables approaches a Gaussian, regardless of their original distribution.
2. **Maximum entropy** -- It makes the fewest assumptions beyond mean and variance.
3. **Mathematical convenience** -- Products of Gaussian PDFs are Gaussian. Marginals and conditionals of multivariate Gaussians are Gaussian. This makes Bayesian updates tractable.

---

## Common Mistakes

> **Not everything is Gaussian!** Heavy-tailed data -- like income distributions, network traffic bursts, stock returns, and outlier-heavy sensor readings -- breaks Gaussian assumptions badly. A Gaussian model assigns negligible probability to extreme events that happen regularly in heavy-tailed data. If your data has frequent outliers, consider Student's t-distribution, log-normal, or Pareto distributions instead. Always plot your data before assuming a distribution.

Other pitfalls to watch for:

| Mistake | Why It Fails | What to Use Instead |
|---------|-------------|-------------------|
| Assuming Gaussian for bounded data | Gaussian extends to $\pm\infty$; heights, probabilities, ratings cannot be negative | Beta (for [0,1] data), Truncated Normal |
| Using Poisson when variance >> mean | Poisson requires mean $=$ variance | Negative Binomial |
| Confusing Exponential rate vs. scale | scipy uses `scale = 1/lambda` | Always double-check parametrization |
| Applying CLT with tiny samples | CLT is asymptotic; $n=5$ is not enough | Use $n \geq 30$ as a rough guideline |
| Using Bernoulli for multi-class | Bernoulli is strictly binary | Categorical / Multinomial |

---

## Distribution-Algorithm Cheat Sheet

Here is a quick reference for which distributions power which ML algorithms:

| Distribution | ML Applications |
|--------------|----------------|
| **Bernoulli** | Binary classification labels, dropout masks, binary features, logistic regression output |
| **Binomial** | A/B testing, multi-trial experiments, ensemble voting, acceptance sampling |
| **Poisson** | Count regression, rare event modeling, NLP word counts, anomaly rates |
| **Uniform** | Weight initialization (pre-Xavier), random hyperparameter search, non-informative priors |
| **Exponential** | Survival analysis, churn modeling, time-to-event, Poisson process inter-arrivals |
| **Gaussian** | Regression noise, Gaussian processes, VAE latent space, batch normalization, weight init |

### Specific Algorithms and Their Distributions

1. **Logistic Regression**: Models $P(Y=1 \mid X)$ as Bernoulli with parameter $\sigma(w^T x)$.
2. **Naive Bayes**:
   - Gaussian NB: Features are Gaussian given the class.
   - Multinomial NB: Word counts are Multinomial.
   - Bernoulli NB: Features are binary.
3. **Gaussian Mixture Models**: Data is a weighted mixture of Gaussians.
4. **Neural Network Initialization**:
   - Xavier: Uniform or Normal scaled by $\sqrt{1/n_{\text{in}}}$.
   - He: Normal scaled by $\sqrt{2/n_{\text{in}}}$ for ReLU.
5. **Variational Autoencoders**: Latent space is Gaussian; the reparameterization trick enables backprop through sampling.
6. **Gaussian Processes**: Prior over functions defined by a multivariate Gaussian.
7. **Poisson Regression**: Models count data via $\log(\mathbb{E}[Y]) = w^T x$.

---

## Choosing the Right Distribution

When you encounter new data, ask yourself these questions:

| Data Characteristic | Distribution to Consider |
|--------------------|--------------------------|
| Binary (0 or 1) | Bernoulli |
| Count of successes in fixed trials | Binomial |
| Count with no upper bound | Poisson |
| Any value equally likely in a range | Uniform |
| Waiting time / duration (memoryless) | Exponential |
| Sum of many independent factors | Gaussian |
| Proportion or probability (bounded [0,1]) | Beta |
| Count with overdispersion (var >> mean) | Negative Binomial |

---

## Quick Reference Table

| Distribution | Type | Parameters | Mean | Variance |
|--------------|------|------------|------|----------|
| Bernoulli | Discrete | $p$ | $p$ | $p(1-p)$ |
| Binomial | Discrete | $n, p$ | $np$ | $np(1-p)$ |
| Poisson | Discrete | $\lambda$ | $\lambda$ | $\lambda$ |
| Uniform | Continuous | $a, b$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| Exponential | Continuous | $\lambda$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |
| Gaussian | Continuous | $\mu, \sigma$ | $\mu$ | $\sigma^2$ |

---

## Exercises

### Exercise 1: Choosing the Right Distribution

For each scenario, identify the most appropriate distribution and explain why:

a) Number of users signing up per hour on your platform
b) Whether a user clicks on a recommendation
c) Average response latency across thousands of requests
d) Time until the next server failure
e) Number of successful API calls out of 50 attempts

**Solution**:
```python
# a) Poisson -- counting events (signups) in a fixed time interval,
#    no hard upper bound, events roughly independent.

# b) Bernoulli -- binary outcome (click / no click) for a single trial.

# c) Gaussian -- by the Central Limit Theorem, the average of many
#    independent latency measurements converges to Gaussian.

# d) Exponential -- time until next event, memoryless waiting.

# e) Binomial -- fixed number of trials (50), each with two outcomes
#    (success / failure), counting total successes.
```

### Exercise 2: Poisson in Production

Your error-logging system shows an average of 4 errors per hour. Calculate:
- The probability of exactly 6 errors in an hour.
- The probability of more than 10 errors in an hour (should you page on-call?).

**Solution**:
```python
from scipy import stats

poisson = stats.poisson(mu=4)

# P(X = 6)
p_exactly_6 = poisson.pmf(6)
print(f"P(X = 6) = {p_exactly_6:.4f}")  # ~0.1042

# P(X > 10)
p_more_than_10 = 1 - poisson.cdf(10)
print(f"P(X > 10) = {p_more_than_10:.4f}")  # ~0.0028
# Very unlikely under normal conditions -- definitely page on-call!
```

### Exercise 3: Gaussian Thresholds for Alerting

Your platform's session lengths are approximately Normal with mean 45 minutes and standard deviation 12 minutes. What session length is at the 95th percentile? What fraction of sessions are longer than 70 minutes?

**Solution**:
```python
from scipy import stats

sessions = stats.norm(loc=45, scale=12)

# 95th percentile
p95 = sessions.ppf(0.95)
print(f"95th percentile: {p95:.1f} min")  # ~64.7 min

# Fraction above 70 minutes
frac_above_70 = 1 - sessions.cdf(70)
print(f"P(session > 70 min) = {frac_above_70:.4f}")  # ~0.0186 (~1.9%)
```

---

## Summary

Here are the key takeaways:

- **Bernoulli** -- the boolean of probability. Single binary trial. Building block for everything else.
- **Binomial** -- sum of Bernoulli trials. Use it for counting successes in fixed experiments and A/B tests.
- **Poisson** -- counting events in time or space. Mean equals variance is its fingerprint.
- **Uniform** -- maximum ignorance. All values in a range equally likely.
- **Exponential** -- memoryless waiting times. The continuous dual of the Poisson process.
- **Gaussian** -- the "universal" distribution via the Central Limit Theorem. Mathematically convenient, but don't use it blindly.
- Choose distributions based on your data's characteristics, not convenience. Always plot first.
- `scipy.stats` provides a unified interface (`pmf`/`pdf`, `cdf`, `ppf`, `rvs`) for all of them.

---

**What's Next** -- Probability theory gives you the language of uncertainty. Now: statistics -- learning the parameters of these distributions from actual data.
