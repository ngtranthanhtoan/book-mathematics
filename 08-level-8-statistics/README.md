# Level 8: Statistics - Learning From Data

You shipped Model B. Accuracy went up 0.3%. Your PM asks: "Is that real, or just noise?" You've run A/B tests before, you know how to read monitoring dashboards with error bars, but now you need to understand the math that makes those decisions rigorous. That's statistics.

Here's the difference from probability: Probability says "I have a fair coin, what's the chance of 7 heads in 10 flips?" Statistics says "I flipped a coin 10 times and got 7 heads, is this coin fair?" Probability generates data from known distributions. Statistics does the reverse—it infers the distribution from observed data. That's exactly what you do in ML: you have training data, you need to learn the underlying patterns.

## Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Level 7: Probability](../07-level-7-probability/README.md) | **Level 8: Statistics** | [Level 9: Optimization Theory](../09-level-9-optimization-theory/README.md) |

**Contents:**
- [Descriptive Statistics](./01-descriptive-statistics.md) - Summarizing data
- [Sampling Theory](./02-sampling-theory.md) - Populations, samples, and the CLT
- [Estimation](./03-estimation.md) - MLE, MAP, and confidence intervals
- [Hypothesis Testing](./04-hypothesis-testing.md) - Making decisions from data

---

## You Already Know This

When you run an A/B test, you're doing hypothesis testing. The control group is your null hypothesis ("no change"), the treatment group is your alternative hypothesis ("there's an effect"). When you say "95% confidence interval" on a dashboard, you're using estimation theory. When you check if your training set is representative of production traffic, you're worried about sampling bias. Statistics formalizes what you already do intuitively.

## What You'll Learn

### 1. Descriptive Statistics
**File:** `01-descriptive-statistics.md`

Before you can infer anything, you need to describe what you have. This chapter covers:
- Mean, median, mode (when to use which)
- Variance and standard deviation (spread matters)
- Covariance and correlation (are features related?)
- Visualization techniques

**SWE Bridge:** Feature normalization uses mean and variance. Correlation analysis helps you detect multicollinearity before it breaks your regression. When you see `scaler.fit(X_train)`, it's computing mean and standard deviation under the hood.

### 2. Sampling Theory
**File:** `02-sampling-theory.md`

You never have all the data. You sample. This chapter explains why that's okay:
- Populations vs samples (what you want vs what you have)
- Law of Large Numbers (larger samples are better)
- Central Limit Theorem (why sample means are normally distributed)
- Sampling distributions (the distribution of statistics themselves)

**SWE Bridge:** Your training set is a sample from the real-world distribution. Sampling bias is just selection bias in telemetry—if you only log successful requests, your error rate estimates are wrong. The CLT is why you can use normal distributions for confidence intervals even when your data isn't normal.

### 3. Estimation
**File:** `03-estimation.md`

How do you estimate parameters from data? This chapter covers:
- Point estimation (giving a single best guess)
- Maximum Likelihood Estimation (MLE) (find the parameters that make the data most probable)
- Maximum A Posteriori (MAP) estimation (MLE + prior knowledge)
- Confidence intervals (quantifying uncertainty)
- Bias and variance of estimators (yes, bias-variance is everywhere)

**SWE Bridge:** When you call `model.fit()`, you're running MLE (or a variant). Logistic regression? MLE. Neural networks with cross-entropy loss? MLE. L2 regularization? That's MAP with a Gaussian prior. This chapter shows you what's happening inside `.fit()`.

### 4. Hypothesis Testing
**File:** `04-hypothesis-testing.md`

How do you decide if an effect is real or just random noise? This chapter covers:
- Null and alternative hypotheses (the claim you're testing)
- p-values (how surprising is this data, if the null hypothesis were true?)
- Type I and Type II errors (false positives vs false negatives)
- t-tests and other common tests
- A/B testing in practice

**SWE Bridge:** A/B testing frameworks are hypothesis testing. Type I error = shipping a change that doesn't work. Type II error = not shipping a change that does work. p-value < 0.05 is just a convention—understand the tradeoffs.

## Building On Level 7 (Probability)

In Level 7, you learned about random variables, distributions, and how to compute probabilities. That was the forward direction: "Given a distribution, what's the probability of this data?"

Statistics runs it backward: "Given this data, what's the distribution?" You're using the same distributions (normal, binomial, etc.), but now you're estimating their parameters from samples instead of assuming them.

## What Comes Next: Level 9 (Optimization Theory)

MLE from this level is an optimization problem: find the parameters that maximize the likelihood function. Level 9 will show you how to solve those optimization problems efficiently, which leads directly to:
- Gradient descent (maximizing likelihood = minimizing negative log-likelihood)
- Loss functions (cross-entropy loss is negative log-likelihood)
- Regularization (MAP estimation is MLE + regularization terms)

Then in Level 13 (ML Models Math), you'll see how all the pieces connect: logistic regression is MLE for a Bernoulli distribution, neural networks are MLE for complex distributions, and everything is just maximum likelihood estimation at scale.

## How To Use This Level

1. **Read in order.** Descriptive statistics → Sampling theory → Estimation → Hypothesis testing. Each builds on the previous.

2. **Run the code.** Every concept has code examples. Don't just read them—run them, break them, modify them. Statistical intuition comes from seeing distributions change as you tweak parameters.

3. **Connect to your work.** After each section, pause and think: "Where have I seen this at work?" MLE in model training? Hypothesis testing in A/B tests? Sampling bias in user data?

4. **Don't skip the theory.** It's tempting to jump straight to "just run the t-test," but understanding why the t-test works (sampling distributions, CLT) helps you know when it doesn't work.

## Time Investment

- **Chapter 1 (Descriptive Statistics):** 2-3 hours
- **Chapter 2 (Sampling Theory):** 2-3 hours
- **Chapter 3 (Estimation):** 3-4 hours
- **Chapter 4 (Hypothesis Testing):** 3-4 hours
- **Exercises and practice:** 4-6 hours

**Total:** 14-20 hours for solid understanding.

## Prerequisites

You should be comfortable with:
- **Level 7 (Probability):** Random variables, distributions, expectation
- **Calculus:** Derivatives (for MLE optimization)
- **Linear algebra basics:** Vectors, matrices (for multivariate statistics)
- **Python:** NumPy, basic data manipulation

## Tools

```python
import numpy as np              # Numerical computations
import scipy.stats as stats     # Statistical distributions and tests
import matplotlib.pyplot as plt # Visualization (critical for intuition)
```

---

**Bottom Line:** You're already using statistics every day—A/B tests, confidence intervals, monitoring dashboards. This level shows you the math underneath those tools, so you can use them correctly, interpret them confidently, and know when they break.

Let's go.
