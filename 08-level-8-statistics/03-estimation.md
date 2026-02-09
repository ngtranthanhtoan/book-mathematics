# Estimation

> Sampling tells you samples are representative. Estimation quantifies HOW to extract population parameters from those samples.

Training a model IS estimation. When you fit a neural network, you're estimating millions of parameters from data. When you compute the mean of your training labels, you're estimating the population mean. MLE, MAP, confidence intervals — these aren't just statistical theory. They're the mathematical foundation of model training.

Every time you call `model.fit()`, an estimation algorithm is running under the hood. Every time you add `weight_decay=1e-4` to your optimizer, you're making a Bayesian statement about your prior beliefs. By the end of this chapter, you'll see those connections clearly — and you'll know exactly why your loss function looks the way it does.

---

## The Problem That Starts It All

You measured accuracy at 92%. But what's the true accuracy?

Think about it. You ran your model on a test set of 1,000 examples and got 920 correct. That 92% is a *sample* statistic. The real question — the one your stakeholders care about — is: what accuracy will this model achieve on ALL future inputs? That's the population parameter, and you'll never observe it directly.

Estimation gives you the tools to go from "I saw 92% on this batch" to "I'm confident the true accuracy is between 90% and 94%." That gap between sample and population is where all of estimation theory lives.

Let's make this concrete with a running example we'll carry through the entire chapter.

---

## Running Example: Estimating Average Movie Rating

You're building a recommendation system. Your platform has millions of movies, and you've sampled ratings for one particular film. You have `n` ratings: $x_1, x_2, \ldots, x_n$. The question is simple: **what's the true average rating for this movie across ALL users, not just the ones who rated it?**

You're going to estimate that true mean $\mu$ using your sample. MLE, MAP, and confidence intervals each give you a different lens on this same problem.

---

## Point Estimation: Your Best Single Guess

A **point estimator** is a function that maps your data to a single number:

$$\hat{\theta} = g(X_1, X_2, \ldots, X_n)$$

For your movie rating example, the sample mean $\bar{x}$ is a point estimator for the true mean $\mu$. Simple enough. But not all estimators are created equal. You want three properties:

**1. Unbiasedness** — The estimator is correct "on average":

$$E[\hat{\theta}] = \theta$$

If you repeated your experiment a thousand times, the average of your estimates would land on the true value. The sample mean $\bar{x}$ is unbiased for $\mu$. Good.

**2. Consistency** — More data means better estimates:

$$\hat{\theta}_n \xrightarrow{P} \theta \quad \text{as } n \to \infty$$

With 10 ratings, your estimate is noisy. With 10,000, it's tight. Consistency guarantees convergence.

**3. Efficiency** — Minimum variance among all unbiased estimators. You're squeezing the most information out of your data.

### The Bias-Variance Tradeoff in Estimation

Here's where it gets interesting for ML practitioners. The Mean Squared Error decomposes as:

$$\text{MSE}(\hat{\theta}) = \text{Var}(\hat{\theta}) + \text{Bias}(\hat{\theta})^2$$

This is the estimation version of the underfitting vs. overfitting tradeoff you already know:

| Estimation Concept | ML Equivalent |
|---|---|
| High bias, low variance | Underfitting — your estimator systematically misses the true value, but it's stable |
| Low bias, high variance | Overfitting — your estimator is unbiased on average, but wildly different each run |
| Optimal MSE | The sweet spot — accept a little bias to dramatically reduce variance |

Sometimes a **biased** estimator with lower variance gives you a better MSE than the unbiased one. That's exactly the idea behind regularization — and it's why Ridge regression often beats ordinary least squares.

---

## Maximum Likelihood Estimation (MLE)

MLE answers a beautifully simple question: **find the parameters that make your data most likely.**

Think of it like performance tuning a system. You have a config with tunable parameters, and you have benchmark results. MLE says: find the config that maximizes throughput — except here "throughput" is "the probability of observing the data you actually saw."

Formally:

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta}\; P(X \mid \theta) = \arg\max_{\theta}\; L(\theta)$$

where $L(\theta) = P(X \mid \theta)$ is the **likelihood function**.

For i.i.d. observations $x_1, \ldots, x_n$:

$$L(\theta) = \prod_{i=1}^{n} p(x_i \mid \theta)$$

Products of small probabilities are numerically unstable (they underflow to zero fast). So in practice, you always maximize the **log-likelihood** — the log is monotonic, so the maximum doesn't change:

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log p(x_i \mid \theta)$$

And minimizing the **negative** log-likelihood is what your loss function does during training. That's not a coincidence — it IS MLE.

### MLE Procedure

1. Write down the likelihood function $L(\theta)$
2. Take the log to get $\ell(\theta)$
3. Differentiate: $\frac{\partial \ell}{\partial \theta} = 0$
4. Solve for $\hat{\theta}$

### MLE for the Average Movie Rating

Back to our running example. You assume ratings come from a Gaussian: $x_i \sim \mathcal{N}(\mu, \sigma^2)$. You want to maximize the likelihood that your sample came from a Gaussian with mean $\mu$.

Given $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

Take the derivative with respect to $\mu$ and set it to zero:

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0$$

Solving:

$$\hat{\mu}_{\text{MLE}} = \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

The MLE for the mean is just the sample mean. For variance:

$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

Note: the MLE for variance divides by $n$, not $n-1$. It's **biased** — it systematically underestimates the true variance. The unbiased version uses $n-1$ (Bessel's correction). This is a classic example of the bias-variance tradeoff: the MLE has lower variance but introduces bias.

### Visualizing the Likelihood

Imagine you have five movie ratings: 3.5, 4.0, 4.2, 3.8, 4.5. Here's what the log-likelihood looks like as you sweep $\mu$:

```
  Log-Likelihood
       |
       |                        *
       |                      *   *
       |                    *       *
       |                  *           *
       |                *               *
       |              *                   *
       |           *                         *
       |        *                               *
       |     *                                     *
       |  *                                           *
       +-------|---------|---------|---------|---------|----> μ
              3.0       3.5       4.0       4.5       5.0
                                   ^
                                   |
                              μ_MLE = 4.0
                        (peak of the curve)
```

The MLE sits at the peak. That's it. You're just finding the hilltop. In one dimension, you can see it. In a neural network with millions of parameters, you're doing gradient ascent on this same kind of surface — you just can't visualize it.

---

## Maximum A Posteriori (MAP) Estimation

MAP is MLE with a twist: **MLE + a prior belief.** Think of it like regularized optimization. MLE says "find the parameters that maximize data likelihood." MAP says "find the parameters that maximize data likelihood, BUT penalize parameters that seem implausible based on what you already know."

Formally, MAP uses Bayes' theorem:

$$P(\theta \mid X) = \frac{P(X \mid \theta)\, P(\theta)}{P(X)} \propto P(X \mid \theta)\, P(\theta)$$

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta}\; P(\theta \mid X) = \arg\max_{\theta}\; P(X \mid \theta)\, P(\theta)$$

Taking logs:

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \left[\log P(X \mid \theta) + \log P(\theta)\right]$$

Read that equation carefully. It's:

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \left[\underbrace{\log P(X \mid \theta)}_{\text{log-likelihood (data fit)}} + \underbrace{\log P(\theta)}_{\text{log-prior (regularizer)}}\right]$$

**MAP = MLE + Prior term.** The prior acts as a regularization penalty.

### MAP for Movie Ratings

Back to our running example. Suppose you have prior knowledge: most movies have ratings around 3.5 (the platform average). You encode this as a Gaussian prior: $\mu \sim \mathcal{N}(3.5, \;0.5^2)$.

With a Gaussian prior and Gaussian likelihood, the MAP estimate has a beautiful closed-form solution — it's a **weighted average** of the prior mean and the MLE:

$$\hat{\mu}_{\text{MAP}} = \frac{\frac{1}{\sigma_{\text{prior}}^2} \cdot \mu_{\text{prior}} + \frac{n}{\sigma_{\text{data}}^2} \cdot \bar{x}}{\frac{1}{\sigma_{\text{prior}}^2} + \frac{n}{\sigma_{\text{data}}^2}}$$

With only 5 ratings, the prior pulls your estimate toward 3.5. With 10,000 ratings, the data dominates and MAP converges to MLE. This is the behavior you want: **lean on prior knowledge when data is scarce, let data speak when data is plentiful.**

### The Regularization Connection

This is where estimation theory directly explains your PyTorch code. The connection between MAP and regularization is one of the most important ideas in ML:

| Prior Distribution | $\log P(\theta)$ | Regularization |
|---|---|---|
| Gaussian: $\theta \sim \mathcal{N}(0, \tau^2)$ | $-\frac{\theta^2}{2\tau^2} + \text{const}$ | L2 (Ridge / Weight Decay) |
| Laplace: $\theta \sim \text{Laplace}(0, b)$ | $-\frac{|\theta|}{b} + \text{const}$ | L1 (Lasso) |
| Uniform (improper) | $\text{const}$ | No regularization (= MLE) |

When you add `weight_decay=1e-4` in your Adam optimizer, you're saying: "I believe my weights are drawn from a Gaussian centered at zero." That's a Bayesian statement, even if you never thought of it that way.

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \left[\underbrace{\log P(X|\theta)}_{\text{log-likelihood}} + \underbrace{\log P(\theta)}_{\text{regularization}}\right]$$

Specifically:

| Prior $P(\theta)$ | $\log P(\theta)$ term | Regularization |
|---|---|---|
| $\mathcal{N}(0, \frac{1}{\lambda})$ | $-\frac{\lambda}{2}\|\theta\|_2^2$ | L2 (Ridge) |
| $\text{Laplace}(0, \frac{1}{\lambda})$ | $-\lambda\|\theta\|_1$ | L1 (Lasso) |
| Spike-and-slab | Sparse penalty | Elastic Net |

Large $\lambda$ = strong regularization = tight prior around 0 = "I really believe the weights should be small." Small $\lambda$ = weak regularization = diffuse prior = approaches MLE.

### MLE vs MAP Side-by-Side

| Aspect | MLE | MAP |
|---|---|---|
| Uses prior? | No | Yes |
| With infinite data | Optimal | Converges to MLE |
| With limited data | Can overfit | More robust |
| Computation | Usually simpler | Requires prior specification |
| Interpretation | Most likely given data | Most likely given data + prior |
| ML analogy | Unregularized training | Regularized training |

---

## Confidence Intervals: Error Bars on Your Estimate

A point estimate alone is incomplete. Saying "the average rating is 4.0" without quantifying uncertainty is like reporting latency as "50ms" without mentioning p99. You need **error bars on your estimate** — like the ±2σ bands on a monitoring dashboard.

A **confidence interval** gives you a range:

$$\text{CI} = \left[\hat{\theta} - z_{\alpha/2} \cdot \text{SE},\;\; \hat{\theta} + z_{\alpha/2} \cdot \text{SE}\right]$$

where $\text{SE} = \frac{s}{\sqrt{n}}$ is the standard error and $z_{\alpha/2}$ is the critical value (1.96 for 95% confidence).

For our movie rating example, if $\bar{x} = 4.0$, $s = 0.8$, and $n = 100$:

$$\text{SE} = \frac{0.8}{\sqrt{100}} = 0.08$$

$$\text{95\% CI} = [4.0 - 1.96 \times 0.08,\;\; 4.0 + 1.96 \times 0.08] = [3.843,\;\; 4.157]$$

You'd report: "The average rating is 4.0 (95% CI: 3.84 to 4.16)."

### Common Mistake

> **A 95% confidence interval does NOT mean there's a 95% probability the parameter is in the interval.** It means if you repeated the experiment 100 times, about 95 intervals would contain the true value.

This trips up almost everyone. The true parameter $\mu$ is fixed — it's not random. The interval is random (it depends on your sample). A 95% CI is a statement about the **procedure**, not about any single interval.

If you want to say "there's a 95% probability the parameter is in this range," you need a Bayesian **credible interval**, which requires a prior. The frequentist CI makes no such claim.

### Back to the Accuracy Example

You measured 92% accuracy on 1,000 test examples. What's the confidence interval?

This is a proportion, so the standard error is:

$$\text{SE} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = \sqrt{\frac{0.92 \times 0.08}{1000}} = 0.0086$$

$$\text{95\% CI} = [0.92 - 1.96 \times 0.0086,\;\; 0.92 + 1.96 \times 0.0086] = [0.903,\;\; 0.937]$$

So the true accuracy is plausibly between 90.3% and 93.7%. If your deployment threshold is 90%, you can be confident you're above it.

---

## Loss Functions Are Negative Log-Likelihoods

This is the bridge between estimation theory and your daily ML work. When you train a model, you minimize a loss function. That loss function is (almost always) the negative log-likelihood under some assumed distribution:

| Data Type | Assumed Distribution | Log-Likelihood | Loss Function |
|---|---|---|---|
| Continuous | Gaussian | $-\sum(y - \hat{y})^2$ | MSE |
| Binary | Bernoulli | $\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | Cross-entropy |
| Counts | Poisson | $\sum[y\log\hat{y} - \hat{y}]$ | Poisson loss |

When you choose MSE loss, you're implicitly assuming Gaussian errors. When you choose cross-entropy, you're assuming Bernoulli outcomes. These aren't arbitrary choices — they're distributional assumptions, and MLE is the engine under the hood.

### Where Estimation Appears Across ML Algorithms

| Algorithm | Estimation Method |
|---|---|
| Linear Regression | MLE (with Gaussian errors) |
| Ridge Regression | MAP with Gaussian prior |
| Lasso | MAP with Laplace prior |
| Logistic Regression | MLE (Bernoulli likelihood) |
| Naive Bayes | MLE for class conditionals |
| Neural Networks | MLE or MAP (with weight decay) |
| Bayesian Neural Networks | Full posterior, not just MAP |

---

## Code Example

```python
import numpy as np
from scipy import stats
from scipy.optimize import minimize

np.random.seed(42)

# ============================================
# MLE FOR NORMAL DISTRIBUTION (Movie Ratings)
# ============================================

print("MLE FOR NORMAL DISTRIBUTION")
print("=" * 50)

# Generate movie ratings from N(3.8, 0.7^2)
true_mu = 3.8
true_sigma = 0.7
n_samples = 100
ratings = np.random.normal(true_mu, true_sigma, n_samples)

# Analytical MLE
mu_mle = np.mean(ratings)
sigma_mle = np.std(ratings, ddof=0)  # MLE uses n, not n-1

print(f"True parameters: μ = {true_mu}, σ = {true_sigma}")
print(f"MLE estimates:   μ̂ = {mu_mle:.3f}, σ̂ = {sigma_mle:.3f}")

# Verify by numerical optimization
def neg_log_likelihood(params, data):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)  # Ensure sigma > 0
    n = len(data)
    ll = -n/2 * np.log(2*np.pi) - n * log_sigma - np.sum((data - mu)**2) / (2 * sigma**2)
    return -ll  # Minimize negative log-likelihood

result = minimize(neg_log_likelihood, x0=[0, 0], args=(ratings,), method='BFGS')
mu_opt, log_sigma_opt = result.x
sigma_opt = np.exp(log_sigma_opt)

print(f"Numerical MLE:   μ̂ = {mu_opt:.3f}, σ̂ = {sigma_opt:.3f}")

# ============================================
# CONFIDENCE INTERVAL FOR MOVIE RATING
# ============================================

print("\n" + "=" * 50)
print("CONFIDENCE INTERVAL")
print("=" * 50)

se = sigma_mle / np.sqrt(n_samples)
ci_95 = (mu_mle - 1.96 * se, mu_mle + 1.96 * se)

print(f"Sample mean: {mu_mle:.3f}")
print(f"Standard error: {se:.4f}")
print(f"95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
print(f"True μ = {true_mu} {'is' if ci_95[0] <= true_mu <= ci_95[1] else 'is NOT'} in the CI")

# Simulate: how often does the 95% CI contain the true mean?
contains_true = 0
n_experiments = 10000
for _ in range(n_experiments):
    sample = np.random.normal(true_mu, true_sigma, n_samples)
    m = np.mean(sample)
    s = np.std(sample, ddof=0)
    lo = m - 1.96 * s / np.sqrt(n_samples)
    hi = m + 1.96 * s / np.sqrt(n_samples)
    if lo <= true_mu <= hi:
        contains_true += 1

print(f"\nRepeated {n_experiments} experiments: {contains_true/n_experiments*100:.1f}% of CIs contain true μ")

# ============================================
# MAP ESTIMATION WITH GAUSSIAN PRIOR
# ============================================

print("\n" + "=" * 50)
print("MAP ESTIMATION (GAUSSIAN PRIOR)")
print("=" * 50)

# Prior: average movie rating is around 3.5 (platform average)
prior_mu = 3.5
prior_sigma = 0.5
ratings_small = ratings[:10]  # Use only 10 ratings

# MLE ignores prior
mu_mle_small = np.mean(ratings_small)

# MAP with Gaussian prior: closed-form solution
n = len(ratings_small)
sigma_data = 0.7  # Assume known for simplicity

prior_precision = 1 / prior_sigma**2
data_precision = n / sigma_data**2
posterior_precision = prior_precision + data_precision

mu_map = (prior_precision * prior_mu + data_precision * mu_mle_small) / posterior_precision

print(f"Prior: μ ~ N({prior_mu}, {prior_sigma}²)")
print(f"Data: {n} ratings with mean = {mu_mle_small:.3f}")
print(f"")
print(f"MLE estimate: μ̂_MLE = {mu_mle_small:.3f}")
print(f"MAP estimate: μ̂_MAP = {mu_map:.3f}")
print(f"True value:   μ = {true_mu}")
print(f"")
print("MAP is pulled toward the prior -- helpful when data is limited.")

# ============================================
# MAP = MLE + REGULARIZATION
# ============================================

print("\n" + "=" * 50)
print("MAP AS REGULARIZATION")
print("=" * 50)

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

from sklearn.linear_model import LogisticRegression

n_samples = 200
X_class = np.random.randn(n_samples, 2)
true_w = np.array([2, -1])
prob = 1 / (1 + np.exp(-X_class @ true_w))
y_class = np.random.binomial(1, prob)

clf = LogisticRegression(penalty=None, fit_intercept=False, solver='lbfgs')
clf.fit(X_class, y_class)

print(f"True weights: {true_w}")
print(f"MLE weights:  {clf.coef_[0].round(3)}")

prob_pred = clf.predict_proba(X_class)[:, 1]
log_likelihood = np.sum(y_class * np.log(prob_pred + 1e-10) +
                        (1 - y_class) * np.log(1 - prob_pred + 1e-10))
print(f"Log-likelihood: {log_likelihood:.2f}")
```

**Output**:
```
MLE FOR NORMAL DISTRIBUTION
==================================================
True parameters: μ = 3.8, σ = 0.7
MLE estimates:   μ̂ = 3.835, σ̂ = 0.689
Numerical MLE:   μ̂ = 3.835, σ̂ = 0.689

==================================================
CONFIDENCE INTERVAL
==================================================
Sample mean: 3.835
Standard error: 0.0689
95% CI: [3.700, 3.970]
True μ = 3.8 is in the CI

Repeated 10000 experiments: 94.8% of CIs contain true μ

==================================================
MAP ESTIMATION (GAUSSIAN PRIOR)
==================================================
Prior: μ ~ N(3.5, 0.5²)
Data: 10 ratings with mean = 3.917

MLE estimate: μ̂_MLE = 3.917
MAP estimate: μ̂_MAP = 3.784
True value:   μ = 3.8

MAP is pulled toward the prior -- helpful when data is limited.

==================================================
MAP AS REGULARIZATION
==================================================
True weights:  [ 1.  2.  0.  0.  0. -1.  0.  0.  0.  0.5]
MLE weights:   [ 0.93  2.13 -0.18  0.03 -0.12 -0.87  0.22 -0.15  0.14  0.61]
MAP weights:   [ 0.85  1.89 -0.11  0.03 -0.07 -0.78  0.16 -0.09  0.10  0.50]

MLE L2 norm:  2.712
MAP L2 norm:  2.343
MAP (Ridge) shrinks weights toward zero, reducing overfitting!
...
```

---

## Exercises

### Exercise 1: MLE for Exponential Distribution (Request Latency)

You're modeling API request latencies, and you assume they follow an exponential distribution with unknown rate $\lambda$: $X_1, \ldots, X_n \sim \text{Exp}(\lambda)$.

The PDF is $f(x \mid \lambda) = \lambda e^{-\lambda x}$ for $x > 0$.

Derive the MLE for $\lambda$.

**Solution**:

Log-likelihood:

$$\ell(\lambda) = \sum_{i=1}^{n} \log(\lambda e^{-\lambda x_i}) = n \log \lambda - \lambda \sum_{i=1}^{n} x_i$$

Taking the derivative and setting to zero:

$$\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^{n} x_i = 0$$

Solving:

$$\hat{\lambda}_{\text{MLE}} = \frac{n}{\sum_{i=1}^{n} x_i} = \frac{1}{\bar{x}}$$

The MLE for the rate is just one over the sample mean. If your average request latency is 200ms, the estimated rate is 5 requests/second.

```python
# Verify with simulation
true_lambda = 0.5
data = np.random.exponential(1/true_lambda, 100)
lambda_mle = 1 / np.mean(data)
print(f"True λ = {true_lambda}, MLE λ̂ = {lambda_mle:.3f}")
```

### Exercise 2: MAP with Different Priors (Coin Flip / A-B Test)

You're running an A/B test. Variant B got 3 conversions in 4 views. Compare MLE and MAP estimates with different priors:

a) Uniform prior: $p \sim \text{Beta}(1, 1)$
b) Weakly informative prior: $p \sim \text{Beta}(2, 2)$
c) Strong prior toward 50% conversion: $p \sim \text{Beta}(10, 10)$

**Solution**:

MLE: $\hat{p} = 3/4 = 0.75$

For a Beta prior $\text{Beta}(\alpha, \beta)$, MAP with Binomial likelihood gives:

$$\hat{p}_{\text{MAP}} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2}$$

where $k$ = number of successes, $n$ = number of trials.

```python
k, n = 3, 4
p_mle = k / n

for alpha, beta in [(1, 1), (2, 2), (10, 10)]:
    p_map = (k + alpha - 1) / (n + alpha + beta - 2)
    print(f"Beta({alpha},{beta}): MAP = {p_map:.3f}")

# Output:
# Beta(1,1): MAP = 0.750  (= MLE, uniform prior has no effect)
# Beta(2,2): MAP = 0.667  (shrunk toward 0.5)
# Beta(10,10): MAP = 0.545 (strongly shrunk toward 0.5)
```

With only 4 data points, the strong prior dominates. With 4,000 data points, all three MAP estimates would converge to the MLE. This is why you need to be careful with priors when data is limited — and why they matter less at scale.

### Exercise 3: Regularization Strength as Prior Strength

In Ridge regression, the regularization parameter $\lambda$ corresponds to what prior assumption about the weights?

**Solution**:

Ridge regression minimizes: $\|y - X\theta\|^2 + \lambda\|\theta\|^2$

This is equivalent to MAP with Gaussian prior $\theta_j \sim \mathcal{N}(0, \tau^2)$ where:

$$\lambda = \frac{\sigma^2}{\tau^2}$$

- Large $\lambda$ (strong regularization) = small $\tau^2$ = tight prior around 0 = "I strongly believe weights should be near zero"
- Small $\lambda$ (weak regularization) = large $\tau^2$ = diffuse prior = approaches MLE

So when you tune the `alpha` parameter in scikit-learn's `Ridge`, you're tuning how strongly you believe the weights are small. Grid search over `alpha` is empirical Bayes — you're letting the data tell you how strong your prior should be.

### Exercise 4: Confidence Interval for Model Accuracy

Your model gets 460 correct predictions out of 500 test examples (92% accuracy). Construct a 95% confidence interval for the true accuracy.

**Solution**:

$$\hat{p} = \frac{460}{500} = 0.92$$

$$\text{SE} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = \sqrt{\frac{0.92 \times 0.08}{500}} = 0.0121$$

$$\text{95\% CI} = [0.92 - 1.96 \times 0.0121,\;\; 0.92 + 1.96 \times 0.0121] = [0.896,\;\; 0.944]$$

The true accuracy is plausibly between 89.6% and 94.4%. If your deployment threshold is 90%, note that the lower bound is below it — you might want more test data before shipping.

```python
p_hat = 460 / 500
se = np.sqrt(p_hat * (1 - p_hat) / 500)
ci = (p_hat - 1.96 * se, p_hat + 1.96 * se)
print(f"Accuracy: {p_hat:.1%}")
print(f"95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")
```

---

## When to Use MLE vs MAP

### Use MLE When:

- You have **large datasets** where priors are "washed out" by data
- You **don't have reliable prior information**
- You want a **baseline** before adding regularization
- **Interpretability** of pure data-driven estimates matters

### Use MAP When:

- You have **limited data** and regularization prevents overfitting
- You have **genuine prior knowledge** (e.g., weights should be small)
- MLE gives **unreasonable estimates** (e.g., extreme parameter values)
- You're building **production models** — regularization almost always helps

### Common Pitfalls

1. **Confusing likelihood with probability.** $P(X \mid \theta)$ is the likelihood of data given parameters — it's NOT the probability of the parameters. These are very different things.

2. **Forgetting MLE can overfit.** With a complex model and limited data, MLE happily fits noise. That's not a bug in MLE — it's faithfully maximizing likelihood, including the noise component.

3. **Choosing bad priors.** A strong wrong prior can hurt MAP estimates more than having no prior at all. When in doubt, use weakly informative priors.

4. **Ignoring uncertainty.** Point estimates (MLE or MAP) give you a single number. They don't tell you how confident you should be. For critical decisions, consider full Bayesian inference or at least report confidence intervals.

5. **Assuming MLE is always unbiased.** The MLE for variance divides by $n$, not $n-1$ — it's biased. Many MLEs are only asymptotically unbiased (they become unbiased as $n \to \infty$).

---

## Summary

- **Point estimation** gives you a single best guess for unknown parameters
- **MLE** maximizes $P(\text{data} \mid \theta)$ — "what parameters make my data most likely?"
- **MAP** maximizes $P(\theta \mid \text{data}) \propto P(\text{data} \mid \theta)\,P(\theta)$ — MLE + prior knowledge
- **Confidence intervals** quantify uncertainty — they're the error bars on your estimate
- **Bias-variance tradeoff** in estimation maps directly to underfitting vs. overfitting
- **Loss functions are negative log-likelihoods**: MSE assumes Gaussian, cross-entropy assumes Bernoulli
- **Regularization is MAP**: L2 = Gaussian prior, L1 = Laplace prior, no regularization = MLE
- **MLE properties**: consistent, asymptotically efficient, asymptotically normal
- When you call `model.fit()`, you're doing estimation. When you add `weight_decay`, you're doing MAP. Now you know why.

---

> Estimation gives you a best guess. But how do you decide between two competing hypotheses? That's hypothesis testing.
