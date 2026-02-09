# Level 7: Probability Theory

Your model says "85% cat." What does that mean? Is it calibrated? Would you bet money on it? If you feed it ten similar images, will exactly 8.5 be cats? (Spoiler: probably not.) Probability theory is the framework for answering these questions — and it's the backbone of every ML system you'll build.

## You Already Know This

You've been doing probability all along:

- **Hash collisions?** That's the birthday paradox. You calculated the probability that two keys hash to the same bucket.
- **Load balancers?** You model traffic as a probability distribution. Poisson arrivals, exponential service times.
- **Retries with exponential backoff?** Memoryless exponential distribution.
- **A/B testing?** Bayesian updating. You start with a prior and update it with data.

The difference is that now we're making it rigorous. The intuition you've built debugging distributed systems maps directly to the mathematics.

## What This Level Covers

| Chapter | What You'll Learn | ML Application |
|---------|-------------------|----------------|
| **[01 - Probability Foundations](01-probability-foundations.md)** | Axioms, sample spaces, events, counting, combinatorics | Defining valid probability distributions, understanding independence |
| **[02 - Conditional Probability](02-conditional-probability.md)** | Bayes' theorem, independence, marginalization | Bayesian ML, posterior inference, naive Bayes classifiers |
| **[03 - Random Variables](03-random-variables.md)** | Discrete vs continuous, PMF, PDF, CDF, transformations | Modeling data, understanding distributions, feature engineering |
| **[04 - Expectation and Moments](04-expectation-and-moments.md)** | Mean, variance, covariance, moment generating functions | Loss functions, gradient expectations, variance reduction |
| **[05 - Common Distributions](05-common-distributions.md)** | Bernoulli, Binomial, Poisson, Gaussian, Exponential, Beta | Recognizing which distribution models your data, choosing priors |

## Building On What You Know

This level connects directly to **Level 4 (Linear Algebra)**:

- Vectors become **random vectors**. Now they have distributions.
- Matrices become **covariance matrices**. They capture correlations between dimensions.
- Linear transformations become **affine transformations of random variables**. If `X ~ N(μ, Σ)`, then `Y = AX + b ~ N(Aμ + b, AΣA^T)`.
- Eigenvalues tell you about the **principal directions of variance** (PCA is just applied probability).

You've been manipulating deterministic vectors. Now those vectors are stochastic, and everything you learned still applies — with probability distributions along for the ride.

## What Comes Next

**Level 8 (Statistics)** reverses the direction. Here, you're generating data from known distributions: "If `X ~ N(0, 1)`, what's `P(X < 1.96)`?" In statistics, you observe data and infer the distribution: "Given these samples, what are `μ` and `σ`?"

**Level 10 (Information Theory)** takes probability and asks information-theoretic questions: How much uncertainty does this distribution have? How efficiently can we encode samples from it? How different are two distributions?

Together, these three levels form the core of modern ML: probability models the world, statistics learns from data, information theory quantifies uncertainty.

## Prerequisites

You'll need:

- **Level 4 (Linear Algebra)**: Vectors, matrices, eigenvalues. Random variables live in vector spaces.
- **Level 6 (Calculus)**: Integrals for computing expectations. Derivatives for maximum likelihood estimation.
- **Set theory basics**: Unions, intersections, complements (from Level 0).

## The Concrete Picture

Think of probability as answering questions about noisy systems:

- **Sensor readings**: Your GPS says you're at `(lat, lon)`. What's the distribution of your true location?
- **API response times**: Most requests are fast. Some tail at 2 seconds. What distribution models this?
- **Model outputs**: Your classifier outputs a softmax. Are those probabilities calibrated? Would you use them to make decisions under uncertainty?

Every time you write `np.random.randn()`, you're sampling from a distribution. Every time you compute `cross_entropy`, you're measuring the fit between two distributions. Probability makes this intuition precise.

## SWE Bridges to ML Concepts

Here's how your existing knowledge maps to probability theory:

| You Already Know | Probability Concept | ML Application |
|------------------|---------------------|----------------|
| Hash collision probability | Birthday paradox, combinatorics | Understanding rare event probabilities |
| Load balancer traffic patterns | Poisson distribution, queueing theory | Modeling arrival processes, event data |
| Exponential backoff | Exponential distribution (memoryless) | Time-to-event modeling, survival analysis |
| A/B test "statistical significance" | Bayesian updating, hypothesis testing | Bayesian ML, online learning |
| Noisy sensor readings | Random variables, measurement error | Data augmentation, robustness |
| Weighted random sampling | Categorical distribution | Sampling from softmax, MCMC |

## Key Notation

| Symbol | Meaning |
|--------|---------|
| `P(A)` | Probability of event A |
| `P(A│B)` | Probability of A given B (conditional probability) |
| `X` | Random variable |
| `p(x)` or `P(X=x)` | Probability mass function (discrete) |
| `f(x)` | Probability density function (continuous) |
| `F(x)` | Cumulative distribution function: `P(X ≤ x)` |
| `E[X]` | Expected value (mean) of X |
| `Var(X)` | Variance of X |
| `Cov(X, Y)` | Covariance of X and Y |

## The Probabilistic Mindset

As an engineer, you're used to deterministic reasoning: "This function returns 42." In ML, everything is probabilistic: "This function returns 42 with probability 0.3."

| Deterministic Thinking | Probabilistic Thinking |
|------------------------|------------------------|
| "The user will click" | "P(click) = 0.05" |
| "This image is a cat" | "P(cat│image) = 0.85" |
| "The system will fail" | "P(failure) = 10^-6" |
| "The algorithm returns the correct answer" | "The algorithm succeeds with probability ≥ 1 - δ" |

This shift is uncomfortable at first. You'll want certainty. But real systems are noisy, and probability is how we reason about them rigorously.

## Study Strategy

1. **Read with examples in mind.** Every theorem should connect to a concrete system you've debugged.
2. **Compute by hand first.** Work through the derivations. Don't skip the math.
3. **Then code it.** Use `scipy.stats` to verify your calculations. Visualize distributions.
4. **Ask "Where does this show up in ML?"** Every distribution, every theorem has applications. We'll point them out.
5. **Expect discomfort.** Probability is slippery. Conditional probability is counterintuitive. Bayes' theorem takes practice. Push through — it clicks eventually.

## Tools We'll Use

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Example: Create a Gaussian distribution
mu, sigma = 0, 1
dist = stats.norm(mu, sigma)

# Sample from it
samples = dist.rvs(size=1000)

# Compute probabilities
prob = dist.cdf(1.96)  # P(X < 1.96) ≈ 0.975
print(f"P(X < 1.96) = {prob:.4f}")

# Compute the PDF at a point
density = dist.pdf(0)  # f(0) = 1/√(2π) ≈ 0.399
print(f"f(0) = {density:.4f}")
```

You'll use `scipy.stats` constantly. It implements every distribution we'll cover. Learn it well.

## Real-World ML Applications

Here's what you'll be able to reason about:

- **Model calibration**: Is `P(cat│image) = 0.9` actually correct, or is your model overconfident?
- **Bayesian neural networks**: Instead of a single set of weights, maintain a distribution over weights.
- **Generative models**: VAEs, diffusion models, normalizing flows — all probability distributions.
- **Uncertainty quantification**: "I'm 95% confident the parameter is in this interval."
- **Anomaly detection**: "This event has probability 10^-7 under our model. Investigate."
- **Loss functions**: Cross-entropy is just negative log-likelihood. KL divergence measures distribution distance.

## Let's Get Started

Probability is one of the most practical mathematical tools you'll learn. Every ML system is probabilistic at its core. Start with [**Chapter 1: Probability Foundations**](01-probability-foundations.md) to build the axioms from scratch.

Then [**Chapter 2: Conditional Probability**](02-conditional-probability.md) introduces Bayes' theorem — the single most important tool in probabilistic ML.

---

*"In God we trust. All others must bring data."* — Attributed to W. Edwards Deming (and every Bayesian ever since)
