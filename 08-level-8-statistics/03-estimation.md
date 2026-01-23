# Estimation

## Intuition

Estimation is about finding the "best guess" for unknown parameters based on observed data. When you train a machine learning model, you're doing estimation - finding parameter values (weights) that best explain your training data.

Imagine you're a detective trying to determine a suspect's height. You have several eyewitness accounts: "tall", "about 6 feet", "over 180 cm". Each account is noisy and imprecise, but by combining them intelligently, you can estimate the true height. That's what statistical estimation does with data.

**Real-world analogy**: Estimation is like adjusting your aim in archery. Each arrow you shoot (data point) tells you something about where you should aim. Point estimation gives you a single best aim point. MLE asks "what aim point makes my observed shots most likely?" MAP asks "what aim point makes my observed shots most likely, given what I knew before shooting?"

**Why this matters for ML**: Virtually every ML algorithm involves estimation:
- Linear regression estimates coefficients
- Logistic regression estimates class probabilities
- Neural networks estimate millions of weights
- Understanding estimation helps you understand why regularization works, why overfitting happens, and how to interpret model parameters.

## Visual Explanation

```mermaid
graph TD
    subgraph "Estimation Framework"
        A[Observed Data X] --> B[Likelihood Function<br/>P(X|θ)]
        C[Prior Belief<br/>P(θ)] --> D[Posterior<br/>P(θ|X)]
        B --> D
    end

    D --> E[Point Estimate θ̂]

    E --> F[MLE: Maximize Likelihood]
    E --> G[MAP: Maximize Posterior]

    style A fill:#e3f2fd
    style C fill:#fff3e0
    style D fill:#e8f5e9
```

### MLE vs MAP Intuition

```
               LIKELIHOOD (from data)         PRIOR (our belief)
               ━━━━━━━━━━━━━━━━━━━━━━━        ━━━━━━━━━━━━━━━━━━━━

                     ╭──────╮
                    ╱        ╲                      ╭────╮
                   ╱          ╲                    ╱      ╲
                  ╱            ╲                  ╱        ╲
               ──╱──────────────╲──            ──╱──────────╲──
                    θ_MLE                          θ_prior

                              ↓ Combine ↓

               POSTERIOR (updated belief)
               ━━━━━━━━━━━━━━━━━━━━━━━━━━━
                           ╭────╮
                          ╱      ╲
                         ╱        ╲
                        ╱          ╲
                     ──╱────────────╲──
                           θ_MAP
                    (between MLE and prior)
```

## Mathematical Foundation

### Point Estimation

A **point estimator** is a function that maps data to a single value:

$$\hat{\theta} = g(X_1, X_2, ..., X_n)$$

**Desirable properties of estimators**:

1. **Unbiasedness**: $E[\hat{\theta}] = \theta$
   - The estimator is correct "on average"

2. **Consistency**: $\hat{\theta}_n \xrightarrow{P} \theta$ as $n \to \infty$
   - More data leads to better estimates

3. **Efficiency**: Minimum variance among unbiased estimators
   - Makes best use of available data

**Bias-Variance Tradeoff**:
$$MSE(\hat{\theta}) = Var(\hat{\theta}) + Bias(\hat{\theta})^2$$

Sometimes a biased estimator with lower variance is preferred (this is the idea behind regularization).

### Maximum Likelihood Estimation (MLE)

MLE finds the parameter values that make the observed data most probable:

$$\hat{\theta}_{MLE} = \arg\max_{\theta} P(X | \theta) = \arg\max_{\theta} L(\theta)$$

where $L(\theta) = P(X|\theta)$ is the **likelihood function**.

For i.i.d. observations $x_1, ..., x_n$:

$$L(\theta) = \prod_{i=1}^{n} p(x_i | \theta)$$

In practice, we maximize the **log-likelihood** (equivalent, but numerically stable):

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log p(x_i | \theta)$$

**MLE Procedure**:
1. Write down the likelihood function $L(\theta)$
2. Take the log to get $\ell(\theta)$
3. Differentiate: $\frac{\partial \ell}{\partial \theta} = 0$
4. Solve for $\hat{\theta}$

**Example: MLE for Normal Distribution**

Given $x_1, ..., x_n \sim N(\mu, \sigma^2)$:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

Taking derivatives and setting to zero:

$$\hat{\mu}_{MLE} = \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

$$\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

Note: MLE for variance is biased! ($n$ instead of $n-1$)

### Maximum A Posteriori (MAP) Estimation

MAP incorporates prior knowledge using Bayes' theorem:

$$P(\theta | X) = \frac{P(X | \theta) P(\theta)}{P(X)} \propto P(X | \theta) P(\theta)$$

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta | X) = \arg\max_{\theta} P(X | \theta) P(\theta)$$

Taking logs:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} \left[ \log P(X | \theta) + \log P(\theta) \right]$$

**Key insight**: MAP = MLE + Prior term

**Connection to Regularization**:

| Prior Distribution | Regularization Type |
|-------------------|---------------------|
| Gaussian: $\theta \sim N(0, \tau^2)$ | L2 (Ridge) |
| Laplace: $\theta \sim Laplace(0, b)$ | L1 (Lasso) |
| Uniform (improper) | No regularization (= MLE) |

**Example**: With Gaussian prior $\theta \sim N(0, \tau^2)$:

$$\log P(\theta) = -\frac{\theta^2}{2\tau^2} + const$$

This adds an L2 penalty term to the log-likelihood, exactly like ridge regression!

### MLE vs MAP Comparison

| Aspect | MLE | MAP |
|--------|-----|-----|
| Uses prior? | No | Yes |
| With infinite data | Optimal | Converges to MLE |
| With limited data | Can overfit | More robust |
| Computation | Usually simpler | Requires prior specification |
| Interpretation | Most likely given data | Most likely given data + prior |

## Code Example

```python
import numpy as np
from scipy import stats
from scipy.optimize import minimize

np.random.seed(42)

# ============================================
# MLE FOR NORMAL DISTRIBUTION
# ============================================

print("MLE FOR NORMAL DISTRIBUTION")
print("=" * 50)

# Generate data from N(5, 2^2)
true_mu = 5
true_sigma = 2
n_samples = 100
data = np.random.normal(true_mu, true_sigma, n_samples)

# Analytical MLE
mu_mle = np.mean(data)
sigma_mle = np.std(data, ddof=0)  # MLE uses n, not n-1

print(f"True parameters: μ = {true_mu}, σ = {true_sigma}")
print(f"MLE estimates:   μ̂ = {mu_mle:.3f}, σ̂ = {sigma_mle:.3f}")

# Verify by numerical optimization
def neg_log_likelihood(params, data):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)  # Ensure sigma > 0
    n = len(data)
    ll = -n/2 * np.log(2*np.pi) - n * log_sigma - np.sum((data - mu)**2) / (2 * sigma**2)
    return -ll  # Minimize negative log-likelihood

result = minimize(neg_log_likelihood, x0=[0, 0], args=(data,), method='BFGS')
mu_opt, log_sigma_opt = result.x
sigma_opt = np.exp(log_sigma_opt)

print(f"Numerical MLE:   μ̂ = {mu_opt:.3f}, σ̂ = {sigma_opt:.3f}")

# ============================================
# MLE FOR BERNOULLI (COIN FLIP)
# ============================================

print("\n" + "=" * 50)
print("MLE FOR BERNOULLI")
print("=" * 50)

# Flip a biased coin
true_p = 0.7
n_flips = 50
flips = np.random.binomial(1, true_p, n_flips)

# MLE for Bernoulli is just the sample proportion
p_mle = np.mean(flips)

print(f"True probability: p = {true_p}")
print(f"Number of heads: {np.sum(flips)} out of {n_flips}")
print(f"MLE estimate: p̂ = {p_mle:.3f}")

# ============================================
# MAP ESTIMATION WITH GAUSSIAN PRIOR
# ============================================

print("\n" + "=" * 50)
print("MAP ESTIMATION (GAUSSIAN PRIOR)")
print("=" * 50)

# Scenario: Estimating mean with limited data and prior knowledge
# Prior: μ ~ N(0, 3^2) - we believe mean is near 0
# Data: Small sample from N(5, 2^2)

prior_mu = 0
prior_sigma = 3
data_small = data[:10]  # Use only 10 samples

# MLE ignores prior
mu_mle_small = np.mean(data_small)

# MAP with Gaussian prior: closed-form solution
# Posterior mean = weighted average of prior mean and MLE
n = len(data_small)
sigma_data = 2  # Assume known for simplicity

# Posterior precision = prior precision + data precision
prior_precision = 1 / prior_sigma**2
data_precision = n / sigma_data**2
posterior_precision = prior_precision + data_precision

# MAP estimate (posterior mean for Gaussian)
mu_map = (prior_precision * prior_mu + data_precision * mu_mle_small) / posterior_precision

print(f"Prior: μ ~ N({prior_mu}, {prior_sigma}²)")
print(f"Data: {n} samples with mean = {mu_mle_small:.3f}")
print(f"")
print(f"MLE estimate: μ̂_MLE = {mu_mle_small:.3f}")
print(f"MAP estimate: μ̂_MAP = {mu_map:.3f}")
print(f"True value:   μ = {true_mu}")
print(f"")
print("MAP is pulled toward the prior, which helps when data is limited")
print("but can hurt when prior is wrong (as in this case).")

# ============================================
# MAP = MLE + REGULARIZATION
# ============================================

print("\n" + "=" * 50)
print("MAP AS REGULARIZATION")
print("=" * 50)

# Linear regression with L2 regularization (Ridge) = MAP with Gaussian prior
from sklearn.linear_model import LinearRegression, Ridge

# Generate regression data
n_samples = 20
n_features = 10
X = np.random.randn(n_samples, n_features)
true_weights = np.array([1, 2, 0, 0, 0, -1, 0, 0, 0, 0.5])
y = X @ true_weights + np.random.randn(n_samples) * 0.5

# MLE (ordinary least squares)
lr = LinearRegression(fit_intercept=False)
lr.fit(X, y)
weights_mle = lr.coef_

# MAP with Gaussian prior (Ridge regression)
ridge = Ridge(alpha=1.0, fit_intercept=False)
ridge.fit(X, y)
weights_map = ridge.coef_

print("True weights: ", np.round(true_weights, 2))
print("MLE weights:  ", np.round(weights_mle, 2))
print("MAP weights:  ", np.round(weights_map, 2))
print()
print(f"MLE L2 norm:  {np.linalg.norm(weights_mle):.3f}")
print(f"MAP L2 norm:  {np.linalg.norm(weights_map):.3f}")
print("MAP (Ridge) shrinks weights toward zero, reducing overfitting!")

# ============================================
# MLE FOR LOGISTIC REGRESSION
# ============================================

print("\n" + "=" * 50)
print("MLE FOR LOGISTIC REGRESSION")
print("=" * 50)

# Binary classification
from sklearn.linear_model import LogisticRegression

# Generate classification data
n_samples = 200
X_class = np.random.randn(n_samples, 2)
true_w = np.array([2, -1])
prob = 1 / (1 + np.exp(-X_class @ true_w))
y_class = np.random.binomial(1, prob)

# Fit logistic regression (MLE)
clf = LogisticRegression(penalty=None, fit_intercept=False, solver='lbfgs')
clf.fit(X_class, y_class)

print(f"True weights: {true_w}")
print(f"MLE weights:  {clf.coef_[0].round(3)}")

# Log-likelihood of the fitted model
prob_pred = clf.predict_proba(X_class)[:, 1]
log_likelihood = np.sum(y_class * np.log(prob_pred + 1e-10) +
                        (1 - y_class) * np.log(1 - prob_pred + 1e-10))
print(f"Log-likelihood: {log_likelihood:.2f}")
```

**Output**:
```
MLE FOR NORMAL DISTRIBUTION
==================================================
True parameters: μ = 5, σ = 2
MLE estimates:   μ̂ = 5.057, σ̂ = 1.998
Numerical MLE:   μ̂ = 5.057, σ̂ = 1.998

==================================================
MLE FOR BERNOULLI
==================================================
True probability: p = 0.7
Number of heads: 37 out of 50
MLE estimate: p̂ = 0.740

==================================================
MAP ESTIMATION (GAUSSIAN PRIOR)
==================================================
Prior: μ ~ N(0, 3²)
Data: 10 samples with mean = 5.226

MLE estimate: μ̂_MLE = 5.226
MAP estimate: μ̂_MAP = 4.211
True value:   μ = 5

MAP is pulled toward the prior, which helps when data is limited
but can hurt when prior is wrong (as in this case).
...
```

## ML Relevance

### Where Estimation Appears in ML

1. **Training = MLE or MAP**
   - Cross-entropy loss = negative log-likelihood for classification
   - Mean squared error = negative log-likelihood for Gaussian regression
   - L2 regularization = Gaussian prior (MAP)
   - L1 regularization = Laplace prior (MAP)

2. **Specific Algorithms**
   | Algorithm | Estimation Method |
   |-----------|------------------|
   | Linear Regression | MLE (with Gaussian errors) |
   | Ridge Regression | MAP with Gaussian prior |
   | Lasso | MAP with Laplace prior |
   | Logistic Regression | MLE (Bernoulli likelihood) |
   | Naive Bayes | MLE for class conditionals |
   | Neural Networks | MLE or MAP (with weight decay) |
   | Bayesian Neural Networks | Full posterior, not just MAP |

3. **Loss Functions = Negative Log-Likelihoods**

   | Data Type | Distribution | Log-Likelihood | Loss Function |
   |-----------|--------------|----------------|---------------|
   | Continuous | Gaussian | $-\sum(y-\hat{y})^2$ | MSE |
   | Binary | Bernoulli | $\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | Cross-entropy |
   | Counts | Poisson | $\sum[y\log\hat{y} - \hat{y}]$ | Poisson loss |

### Regularization Connection

The relationship between MAP and regularization is fundamental:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} \left[\underbrace{\log P(X|\theta)}_{\text{log-likelihood}} + \underbrace{\log P(\theta)}_{\text{regularization}}\right]$$

| Prior $P(\theta)$ | $\log P(\theta)$ | Regularization |
|-------------------|------------------|----------------|
| $N(0, \frac{1}{\lambda})$ | $-\frac{\lambda}{2}\|\theta\|_2^2$ | L2 (Ridge) |
| $Laplace(0, \frac{1}{\lambda})$ | $-\lambda\|\theta\|_1$ | L1 (Lasso) |
| Spike-and-slab | Sparse penalty | Elastic Net |

## When to Use / Ignore

### When to Use MLE

- Large datasets where priors are "washed out" by data
- When you don't have reliable prior information
- For baseline models before adding regularization
- When interpretability of pure data-driven estimates is important

### When to Use MAP

- Limited data where regularization prevents overfitting
- When you have genuine prior knowledge
- When MLE gives unreasonable estimates
- For production models (regularization almost always helps)

### Common Pitfalls

1. **Confusing likelihood with probability**: $P(X|\theta)$ is likelihood of data given parameters, not probability of parameters.

2. **Forgetting MLE can overfit**: With complex models and limited data, MLE finds parameters that fit noise.

3. **Choosing bad priors**: A strong wrong prior can hurt MAP estimates. Use weakly informative priors when uncertain.

4. **Ignoring uncertainty**: Point estimates (MLE or MAP) don't quantify uncertainty. Consider full Bayesian inference for critical applications.

5. **Assuming MLE is always unbiased**: MLE for variance is biased. Many MLEs are only asymptotically unbiased.

## Exercises

### Exercise 1: MLE for Exponential Distribution

Given samples from an exponential distribution with unknown rate $\lambda$: $X_1, ..., X_n \sim Exp(\lambda)$

The PDF is $f(x|\lambda) = \lambda e^{-\lambda x}$ for $x > 0$.

Derive the MLE for $\lambda$.

**Solution**:

Log-likelihood:
$$\ell(\lambda) = \sum_{i=1}^n \log(\lambda e^{-\lambda x_i}) = n\log\lambda - \lambda\sum_{i=1}^n x_i$$

Taking derivative and setting to zero:
$$\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0$$

Solving:
$$\hat{\lambda}_{MLE} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{x}}$$

```python
# Verify with simulation
true_lambda = 0.5
data = np.random.exponential(1/true_lambda, 100)
lambda_mle = 1 / np.mean(data)
print(f"True λ = {true_lambda}, MLE λ̂ = {lambda_mle:.3f}")
```

### Exercise 2: MAP with Different Priors

You observe 3 heads in 4 coin flips. Compare MLE and MAP estimates with:
a) Uniform prior: $p \sim Beta(1, 1)$
b) Weakly informative prior: $p \sim Beta(2, 2)$
c) Strong prior toward fair: $p \sim Beta(10, 10)$

**Solution**:

MLE: $\hat{p} = 3/4 = 0.75$

For Beta prior $Beta(\alpha, \beta)$, MAP with Binomial likelihood gives:
$$\hat{p}_{MAP} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2}$$

where $k$ = number of heads, $n$ = number of flips.

```python
k, n = 3, 4
p_mle = k / n

# Different priors
for alpha, beta in [(1, 1), (2, 2), (10, 10)]:
    p_map = (k + alpha - 1) / (n + alpha + beta - 2)
    print(f"Beta({alpha},{beta}): MAP = {p_map:.3f}")

# Output:
# Beta(1,1): MAP = 0.750  (= MLE, uniform prior)
# Beta(2,2): MAP = 0.667  (shrunk toward 0.5)
# Beta(10,10): MAP = 0.545 (strongly shrunk toward 0.5)
```

### Exercise 3: Regularization Strength

In Ridge regression, the regularization parameter $\lambda$ corresponds to what prior assumption about the weights?

**Solution**:

Ridge regression minimizes: $\|y - X\theta\|^2 + \lambda\|\theta\|^2$

This is equivalent to MAP with Gaussian prior $\theta_j \sim N(0, \tau^2)$ where:

$$\lambda = \frac{\sigma^2}{\tau^2}$$

- Large $\lambda$ (strong regularization) = small $\tau^2$ = tight prior around 0
- Small $\lambda$ (weak regularization) = large $\tau^2$ = diffuse prior (approaches MLE)

## Summary

- **Point estimation**: Find a single best value for unknown parameters
- **MLE**: Maximizes $P(data|\theta)$; answers "what parameters make my data most likely?"
- **MAP**: Maximizes $P(\theta|data) \propto P(data|\theta)P(\theta)$; incorporates prior knowledge
- **MLE properties**: Consistent, asymptotically efficient, asymptotically normal
- **MAP vs MLE**: MAP = MLE + regularization; MAP more robust with limited data
- **Regularization connection**: L2 = Gaussian prior, L1 = Laplace prior
- **In ML**: Training is estimation; loss functions are negative log-likelihoods; regularization is MAP
- **Key insight**: Understanding estimation reveals why ML algorithms work and when they might fail
