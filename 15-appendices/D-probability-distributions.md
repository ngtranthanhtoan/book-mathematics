# Appendix D: Probability Distributions Reference

> One card per distribution. For each: the formula, parameters, mean, variance, shape, and where you will encounter it in ML.

---

## Discrete Distributions

### Bernoulli

$$P(X = k) = p^k(1-p)^{1-k}, \quad k \in \{0, 1\}$$

| Property | Value |
|----------|-------|
| Parameters | $p \in [0, 1]$ (success probability) |
| Mean | $p$ |
| Variance | $p(1-p)$ |
| ML usage | Binary classification output, coin flip, single yes/no trial |

---

### Binomial

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

| Property | Value |
|----------|-------|
| Parameters | $n$ (trials), $p$ (success probability) |
| Mean | $np$ |
| Variance | $np(1-p)$ |
| ML usage | Number of successes in $n$ independent trials, batch accuracy |

---

### Poisson

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

| Property | Value |
|----------|-------|
| Parameters | $\lambda > 0$ (rate) |
| Mean | $\lambda$ |
| Variance | $\lambda$ |
| ML usage | Count data, rare events, request rates, word frequencies |

---

### Categorical (Generalized Bernoulli)

$$P(X = k) = p_k, \quad \sum_{k=1}^{K} p_k = 1$$

| Property | Value |
|----------|-------|
| Parameters | $p_1, \ldots, p_K$ (class probabilities) |
| Mean | $\arg\max_k p_k$ (mode) |
| ML usage | Multiclass classification output (softmax output) |

---

## Continuous Distributions

### Uniform

$$f(x) = \frac{1}{b-a}, \quad x \in [a, b]$$

| Property | Value |
|----------|-------|
| Parameters | $a$ (min), $b$ (max) |
| Mean | $\frac{a+b}{2}$ |
| Variance | $\frac{(b-a)^2}{12}$ |
| ML usage | Random initialization, random sampling, baseline model |

---

### Normal (Gaussian)

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

| Property | Value |
|----------|-------|
| Parameters | $\mu$ (mean), $\sigma^2$ (variance) |
| Mean | $\mu$ |
| Variance | $\sigma^2$ |
| ML usage | Weight initialization, noise modeling, Central Limit Theorem, Gaussian processes, VAE latent space |

**Standard Normal**: $\mu = 0, \sigma = 1$, denoted $\mathcal{N}(0, 1)$.

---

### Multivariate Normal

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\mathbf{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

| Property | Value |
|----------|-------|
| Parameters | $\boldsymbol{\mu} \in \mathbb{R}^d$ (mean vector), $\mathbf{\Sigma} \in \mathbb{R}^{d \times d}$ (covariance matrix) |
| Mean | $\boldsymbol{\mu}$ |
| Covariance | $\mathbf{\Sigma}$ |
| ML usage | Gaussian mixture models, Gaussian processes, Bayesian inference, VAE |

---

### Exponential

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

| Property | Value |
|----------|-------|
| Parameters | $\lambda > 0$ (rate) |
| Mean | $1/\lambda$ |
| Variance | $1/\lambda^2$ |
| ML usage | Time between events, exponential backoff, survival analysis |

---

### Beta

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0, 1]$$

| Property | Value |
|----------|-------|
| Parameters | $\alpha > 0$, $\beta > 0$ (shape parameters) |
| Mean | $\frac{\alpha}{\alpha + \beta}$ |
| Variance | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |
| ML usage | Prior for probabilities (Bayesian), A/B testing, Thompson sampling |

---

### Gamma

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x \geq 0$$

| Property | Value |
|----------|-------|
| Parameters | $\alpha > 0$ (shape), $\beta > 0$ (rate) |
| Mean | $\alpha/\beta$ |
| Variance | $\alpha/\beta^2$ |
| ML usage | Prior for precision (inverse variance), waiting times, Bayesian inference |

---

### Chi-Squared ($\chi^2$)

$$f(x) = \frac{x^{k/2-1}e^{-x/2}}{2^{k/2}\Gamma(k/2)}, \quad x \geq 0$$

| Property | Value |
|----------|-------|
| Parameters | $k$ (degrees of freedom) |
| Mean | $k$ |
| Variance | $2k$ |
| ML usage | Hypothesis testing, goodness-of-fit tests, feature selection |

Special case of Gamma with $\alpha = k/2$, $\beta = 1/2$.

---

### Student's t

| Property | Value |
|----------|-------|
| Parameters | $\nu$ (degrees of freedom) |
| Mean | $0$ (for $\nu > 1$) |
| Variance | $\frac{\nu}{\nu - 2}$ (for $\nu > 2$) |
| ML usage | Small-sample hypothesis testing, robust regression, confidence intervals |

Approaches Normal as $\nu \to \infty$. Heavier tails than Normal (more robust to outliers).

---

## Quick Selection Guide

| Scenario | Distribution |
|----------|-------------|
| Yes/no outcome | Bernoulli |
| Count of successes in $n$ trials | Binomial |
| Count of rare events | Poisson |
| Multi-class label | Categorical |
| Continuous, symmetric, "default" | Normal |
| Strictly positive, right-skewed | Gamma or Exponential |
| Probability value $\in [0,1]$ | Beta |
| Small sample, unknown variance | Student's t |
| No prior knowledge, bounded range | Uniform |

---

*Back to [Appendices](README.md)*
