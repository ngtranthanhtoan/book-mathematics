# Sampling Theory

## Opening: The Part-to-Whole Problem

Your training set is 50,000 images from a universe of billions. Your A/B test ran on 10,000 users out of 50 million. How do you draw conclusions about the whole from a part? Sampling theory tells you when (and how confidently) you can generalize.

*Descriptive stats summarize your sample. But is your sample representative? Can you trust the mean you computed?*

This chapter answers those questions. You will work through the math that lets you take a slice of reality--a sample--and make rigorous statements about everything you did not observe. Along the way, you will see why your model's test accuracy is trustworthy (or not), why mini-batch gradients work at all, and how to put error bars on any metric you report.

---

## Problem First: Can You Trust Your Test Set Accuracy?

You trained an image classifier. You held out 5,000 images as a test set. Your model scores 92.4% accuracy on those 5,000 images. Your manager asks: "What's the *real* accuracy?"

You cannot test on every image in the world. You have a *sample* of 5,000 from a vast *population*. Your 92.4% is a *statistic*--a number computed from the sample. The true accuracy is a *parameter*--a fixed but unknown property of the population. Sampling theory is the bridge between the two.

### Running Example: Movie Ratings

Throughout this chapter, you will follow a single scenario. A streaming platform has 1,000,000 reviews for a movie. You cannot process all of them in real time, so you pull a random sample of 1,000 reviews. Each review includes a rating from 1 to 10. You want to estimate the true average rating across all 1M reviews.

Everything you learn--LLN, CLT, standard error, bootstrap--you will apply to this one problem. By the end, you will know exactly how much to trust your sample mean.

---

## Population vs. Sample: The Vocabulary

Before the math, nail down the terminology. You will see these pairs everywhere:

| Concept | Population (what you want) | Sample (what you have) |
|---------|---------------------------|------------------------|
| Size | $N$ (often huge or infinite) | $n$ (manageable) |
| Mean | $\mu$ (parameter, fixed, unknown) | $\bar{x}$ (statistic, computed, varies) |
| Variance | $\sigma^2$ (parameter) | $s^2$ (statistic) |
| Std Dev | $\sigma$ | $s$ |

**The core insight**: parameters are fixed but unknown. Statistics are calculated from data and *vary between samples*. Every time you draw a different sample of 1,000 reviews, you get a slightly different $\bar{x}$. Sampling theory describes *how* it varies.

### SWE Bridge: Config vs. Measurement

Think of $\mu$ as a hard-coded constant buried deep in a system you cannot inspect--like the true p99 latency of a service across all requests ever made. Think of $\bar{x}$ as the value your monitoring dashboard shows for the last 1,000 requests. The dashboard number fluctuates. The true value does not. Your job is to figure out how close the dashboard number is to reality.

---

## Sampling Bias: When Your Sample Lies

Random sampling is the foundation. When it breaks, everything downstream breaks. Here are the failure modes you will encounter as an engineer:

**Selection bias.** Your training data comes from users who opted in. But the users who opted in are not representative of all users. Your model learns the biases of the willing.

**Survivorship bias.** You study successful ML projects to find patterns. But you never see the failed projects that had the same patterns. You are looking at a filtered sample.

**Response bias.** You measure user satisfaction through a feedback button. Angry users click it 5x more often than happy users. Your "sample" of feedback over-represents dissatisfaction.

**Temporal bias.** You train on data from 2023 and deploy in 2025. The world changed. Your sample is from the wrong time period.

**In ML, this is distribution shift:**

$$P_{\text{train}}(X, Y) \neq P_{\text{deploy}}(X, Y)$$

Types of shift:
- **Covariate shift**: $P_{\text{train}}(X) \neq P_{\text{deploy}}(X)$ -- the inputs change
- **Label shift**: $P_{\text{train}}(Y) \neq P_{\text{deploy}}(Y)$ -- the class balance changes
- **Concept drift**: $P_{\text{train}}(Y|X) \neq P_{\text{deploy}}(Y|X)$ -- the relationship itself changes

No amount of mathematical machinery fixes a biased sample. Get the sampling right first. Then apply the theorems below.

---

## The Law of Large Numbers (LLN)

### The Theorem

The LLN says: as your sample grows, your sample mean locks onto the true mean.

**Formal statement (Weak LLN).** For i.i.d. random variables $X_1, X_2, \ldots, X_n$ with mean $\mu$:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \xrightarrow{P} \mu \quad \text{as } n \to \infty$$

This is convergence in probability: for any tolerance $\epsilon > 0$, the chance that $\bar{X}_n$ is more than $\epsilon$ away from $\mu$ goes to zero.

**Strong LLN** (the stronger guarantee):

$$P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$

Almost surely, the running average converges to the true mean.

### SWE Bridge: Running Averages Stabilize

You have seen LLN in action every time you look at a monitoring dashboard. Your system's 5-minute load average is noisy. The 1-hour load average is smoother. The 24-hour average barely moves. That is LLN at work--as $n$ increases, the running average converges to the true underlying rate.

The same thing happens with your movie rating sample. Pull 10 reviews and the average might be 8.3. Pull 100 and it is 7.1. Pull 1,000 and it settles at 7.24. Pull 10,000 and it barely budges from 7.23. The noise averages out.

### How Fast Does It Converge? The Standard Error

LLN tells you the average converges. But how fast? The answer is the **standard error**:

$$SE(\bar{X}) = \frac{\sigma}{\sqrt{n}}$$

This is the standard deviation of the sampling distribution of the mean. It tells you how much your sample mean typically deviates from $\mu$.

**The square-root law.** To cut your error in half, you need 4x the data. To cut it by 10x, you need 100x the data. This is the fundamental diminishing-returns law of sampling.

| Sample size $n$ | $SE = \sigma / \sqrt{n}$ (if $\sigma = 2.0$) |
|----------------|----------------------------------------------|
| 10 | 0.632 |
| 100 | 0.200 |
| 1,000 | 0.063 |
| 10,000 | 0.020 |

For your movie rating example: if the population standard deviation of ratings is $\sigma = 2.0$ and you sample $n = 1{,}000$ reviews, your standard error is $2.0 / \sqrt{1000} \approx 0.063$. Your sample mean is typically within about 0.063 of the true mean. That is precise enough to distinguish a 7.2-rated movie from a 7.4-rated movie.

---

## The Central Limit Theorem (CLT)

### The Theorem

The CLT is the reason most of statistical inference works. It says: the distribution of sample means is approximately Gaussian, **no matter what the original data looks like**.

**Formal statement.** For i.i.d. random variables $X_1, \ldots, X_n$ with mean $\mu$ and finite variance $\sigma^2$:

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1) \quad \text{as } n \to \infty$$

Or equivalently:

$$\bar{X}_n \dot{\sim} \; N\!\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{for large } n$$

### What This Means in Plain English

Individual movie ratings follow some weird, lumpy distribution--lots of 1s and 10s, a cluster around 7, a dip at 5. It is not Gaussian at all. But if you take 1,000 ratings and compute the mean, then do that again, and again, and again--the *distribution of those means* is Gaussian. Always. That is the CLT.

### SWE Bridge: Why Batch Averages Are Gaussian

In mini-batch SGD, each mini-batch is a random sample from your training set. The gradient you compute is an average over that batch. Even though individual per-example gradients can be wildly non-Gaussian (sparse, heavy-tailed, skewed), the *batch average gradient* is approximately Gaussian by the CLT. This is why Gaussian noise assumptions in optimizer theory work reasonably well--not because individual gradients are Gaussian, but because their *averages* are.

### Common Mistake

> **The CLT applies to averages, not individual observations.** Individual data points can follow ANY distribution. The CLT says nothing about them. It says: take enough of them, compute their mean, and *that mean* is approximately Gaussian. Confusing these two statements is one of the most common errors in applied statistics.

### ASCII Visualization: Sampling Distribution Narrowing as n Increases

Here is the key visual insight. As you increase your sample size $n$, the sampling distribution of $\bar{X}$ gets taller and narrower--you become more and more certain about where $\mu$ is:

```
  Sampling Distribution of x̄  (true μ = 7.2)

  n = 10  (SE = 0.63)               Wide, uncertain
                  ·  ·
                ·      ·
              ·          ·
            ·              ·
        · ·                  · ·
  ─────────────────────────────────────
       5.5   6.0   6.5   7.0   7.5   8.0   8.5   9.0


  n = 100  (SE = 0.20)              Narrower
                    ···
                  ··   ··
                ··       ··
              ··           ··
           ···               ···
  ─────────────────────────────────────
       5.5   6.0   6.5   7.0   7.5   8.0   8.5   9.0


  n = 1000  (SE = 0.063)            Tight, precise
                     |
                    |||
                   |||||
                  |||||||
                |||||||||
  ─────────────────────────────────────
       5.5   6.0   6.5   7.0   7.5   8.0   8.5   9.0
                          ↑
                     True μ = 7.2

  As n increases: SE = σ/√n shrinks → distribution narrows
  More data = more certainty about the population mean
```

Three distributions, one truth. With $n = 10$, your sample mean could land anywhere from 5.5 to 9.0. With $n = 1{,}000$, it is squeezed into a narrow band around 7.2. That is the power of sampling: you do not need all million reviews, just enough to shrink that distribution to an acceptable width.

### Rule of Thumb

The CLT approximation works well for $n \geq 30$ in most cases. But if your data is heavily skewed (like income distributions, or file size distributions), you may need $n \geq 100$ or more. The more skewed or heavy-tailed the underlying distribution, the more samples you need before the Gaussian approximation kicks in.

---

## Bootstrap: The Mock Server Approach to Uncertainty

### The Problem

The standard error formula $SE = \sigma / \sqrt{n}$ requires you to know $\sigma$. You can estimate it with $s$ (the sample standard deviation), and for means that works fine. But what if you want the standard error of the *median*? Or the *90th percentile*? Or some complicated custom metric? There is no neat formula for those.

### The Bootstrap Idea

The bootstrap is elegantly simple. You have one sample of $n$ data points. You cannot go back and collect more samples from the population. But you *can* resample from your existing sample--draw $n$ values *with replacement*--and treat each resample as if it were a new experiment.

**SWE Bridge: the "mock server" approach.** Imagine you ran a load test against production and collected 1,000 latency measurements. You want to know the uncertainty on the p99 latency. You cannot re-run the load test 1,000 times--that would take weeks and the conditions would change. Instead, you *simulate* re-running it: you resample your 1,000 measurements (with replacement) to create a "mock" dataset, compute p99, and repeat 10,000 times. The spread of those 10,000 p99 values is your uncertainty estimate. You are resampling your data to simulate running the experiment 1,000 times.

### The Algorithm

```
Bootstrap Confidence Interval:

1. You have a sample of n data points: [x₁, x₂, ..., xₙ]
2. Repeat B times (B = 10,000 is common):
   a. Draw n values WITH REPLACEMENT from your sample
   b. Compute the statistic of interest (mean, median, p99, ...)
   c. Store the result
3. Sort the B bootstrap statistics
4. The middle 95% is your 95% confidence interval
   (i.e., the 2.5th percentile to the 97.5th percentile)
```

### Movie Rating Example

You have 1,000 movie ratings. The sample mean is 7.24. How uncertain are you?

```python
import numpy as np

np.random.seed(42)

# Your sample of 1000 ratings (simulated here)
ratings = np.random.choice(range(1, 11), size=1000, p=[
    0.03, 0.02, 0.04, 0.05, 0.08, 0.12, 0.22, 0.24, 0.12, 0.08
])

sample_mean = np.mean(ratings)
print(f"Sample mean: {sample_mean:.3f}")

# Bootstrap: resample 10,000 times
n_bootstrap = 10_000
bootstrap_means = np.array([
    np.mean(np.random.choice(ratings, size=len(ratings), replace=True))
    for _ in range(n_bootstrap)
])

# 95% confidence interval
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)

print(f"Bootstrap SE: {np.std(bootstrap_means):.4f}")
print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
```

The bootstrap gives you a confidence interval without any distributional assumptions. It works for means, medians, ratios, custom metrics--anything you can compute from data.

### Where Bootstrap Shows Up in ML

- **Random Forests (bagging)**: each tree is trained on a bootstrap sample of the training data. The ensemble's variance reduction comes directly from the bootstrap principle.
- **Confidence intervals on model metrics**: bootstrap your test set predictions to get error bars on accuracy, AUC, F1.
- **Bayesian bootstrap**: a weighted variant used in uncertainty-aware ML.

---

## Putting It All Together: Code

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# ============================================
# SETUP: MOVIE RATING POPULATION AND SAMPLE
# ============================================

# Simulate the full population of 1M reviews
# Ratings 1-10, with a realistic skewed distribution
population_size = 1_000_000
rating_probs = [0.03, 0.02, 0.04, 0.05, 0.08, 0.12, 0.22, 0.24, 0.12, 0.08]
population = np.random.choice(range(1, 11), size=population_size, p=rating_probs)
true_mean = np.mean(population)
true_std = np.std(population)

print("MOVIE RATING POPULATION")
print("=" * 55)
print(f"Population size:   {population_size:,} reviews")
print(f"True mean (mu):    {true_mean:.4f}")
print(f"True std (sigma):  {true_std:.4f}")

# Draw your sample of 1,000 reviews
sample_size = 1000
sample = np.random.choice(population, size=sample_size, replace=False)
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)

print(f"\nSample size:       {sample_size}")
print(f"Sample mean (x̄):  {sample_mean:.4f}")
print(f"Sample std (s):    {sample_std:.4f}")
print(f"Estimation error:  {abs(sample_mean - true_mean):.4f}")

# ============================================
# LAW OF LARGE NUMBERS: RUNNING AVERAGES CONVERGE
# ============================================

print("\n" + "=" * 55)
print("LAW OF LARGE NUMBERS: CONVERGENCE")
print("=" * 55)

sample_sizes = [10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000]

for n in sample_sizes:
    s = np.random.choice(population, size=n, replace=False)
    error = abs(np.mean(s) - true_mean)
    se_theoretical = true_std / np.sqrt(n)
    print(f"n = {n:>6,}: x̄ = {np.mean(s):.4f},  "
          f"error = {error:.4f},  SE(theory) = {se_theoretical:.4f}")

print(f"\nTrue mu = {true_mean:.4f}")
print("Notice: error shrinks roughly as 1/sqrt(n)")

# ============================================
# CENTRAL LIMIT THEOREM: MEANS BECOME GAUSSIAN
# ============================================

print("\n" + "=" * 55)
print("CENTRAL LIMIT THEOREM: DISTRIBUTION OF SAMPLE MEANS")
print("=" * 55)

n_experiments = 10_000
sample_n = 1000

sample_means = np.array([
    np.mean(np.random.choice(population, size=sample_n, replace=False))
    for _ in range(n_experiments)
])

theoretical_mean = true_mean
theoretical_se = true_std / np.sqrt(sample_n)

print(f"Drew {n_experiments} samples, each of size {sample_n}")
print(f"\nEmpirical distribution of sample means:")
print(f"  Mean of means:   {np.mean(sample_means):.4f}  (theory: {theoretical_mean:.4f})")
print(f"  Std of means:    {np.std(sample_means):.4f}  (theory: {theoretical_se:.4f})")

_, p_value = stats.normaltest(sample_means)
print(f"\nNormality test p-value: {p_value:.4f}")
if p_value > 0.05:
    print("Sample means are approximately normally distributed (CLT confirmed)")

# ============================================
# BOOTSTRAP: UNCERTAINTY WITHOUT FORMULAS
# ============================================

print("\n" + "=" * 55)
print("BOOTSTRAP CONFIDENCE INTERVAL")
print("=" * 55)

n_bootstrap = 10_000
bootstrap_means = np.array([
    np.mean(np.random.choice(sample, size=len(sample), replace=True))
    for _ in range(n_bootstrap)
])

ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)

print(f"Original sample mean:  {sample_mean:.4f}")
print(f"Bootstrap SE:          {np.std(bootstrap_means):.4f}")
print(f"Formula SE (s/√n):     {sample_std / np.sqrt(sample_size):.4f}")
print(f"95% Bootstrap CI:      ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"True mean falls in CI: {ci_lower <= true_mean <= ci_upper}")

# Also bootstrap the median (no formula for this!)
bootstrap_medians = np.array([
    np.median(np.random.choice(sample, size=len(sample), replace=True))
    for _ in range(n_bootstrap)
])
med_ci_lower = np.percentile(bootstrap_medians, 2.5)
med_ci_upper = np.percentile(bootstrap_medians, 97.5)
print(f"\nBootstrap 95% CI for median: ({med_ci_lower:.1f}, {med_ci_upper:.1f})")
print("(No analytic formula needed--that is the power of bootstrap)")

# ============================================
# STANDARD ERROR AND SAMPLE SIZE PLANNING
# ============================================

print("\n" + "=" * 55)
print("SAMPLE SIZE PLANNING")
print("=" * 55)

target_se = 0.05  # want SE < 0.05 rating points
sigma_est = sample_std

print(f"Estimated sigma: {sigma_est:.4f}")
print(f"Target SE:       {target_se}")

required_n = int(np.ceil((sigma_est / target_se) ** 2))
print(f"Required n:      {required_n:,}")
print(f"\nTo halve SE from {target_se} to {target_se/2}:")
print(f"  Need n = {int(np.ceil((sigma_est / (target_se/2)) ** 2)):,}  (4x as many!)")
```

**Output** (approximate):
```
MOVIE RATING POPULATION
=======================================================
Population size:   1,000,000 reviews
True mean (mu):    6.9587
True std (sigma):  2.1034

Sample size:       1000
Sample mean (x̄):  6.9810
Sample std (s):    2.1198
Estimation error:  0.0223

=======================================================
LAW OF LARGE NUMBERS: CONVERGENCE
=======================================================
n =     10: x̄ = 7.4000,  error = 0.4413,  SE(theory) = 0.6652
n =     50: x̄ = 6.7000,  error = 0.2587,  SE(theory) = 0.2975
n =    100: x̄ = 7.1200,  error = 0.1613,  SE(theory) = 0.2103
n =    500: x̄ = 6.9120,  error = 0.0467,  SE(theory) = 0.0941
n =  1,000: x̄ = 6.9750,  error = 0.0163,  SE(theory) = 0.0665
n =  5,000: x̄ = 6.9630,  error = 0.0043,  SE(theory) = 0.0297
n = 10,000: x̄ = 6.9559,  error = 0.0028,  SE(theory) = 0.0210
n = 50,000: x̄ = 6.9601,  error = 0.0014,  SE(theory) = 0.0094

True mu = 6.9587
Notice: error shrinks roughly as 1/sqrt(n)

=======================================================
CENTRAL LIMIT THEOREM: DISTRIBUTION OF SAMPLE MEANS
=======================================================
Drew 10000 samples, each of size 1000

Empirical distribution of sample means:
  Mean of means:   6.9589  (theory: 6.9587)
  Std of means:    0.0662  (theory: 0.0665)

Normality test p-value: 0.2341
Sample means are approximately normally distributed (CLT confirmed)

=======================================================
BOOTSTRAP CONFIDENCE INTERVAL
=======================================================
Original sample mean:  6.9810
Bootstrap SE:          0.0668
Formula SE (s/√n):     0.0670
95% Bootstrap CI:      (6.8490, 7.1120)
True mean falls in CI: True

Bootstrap 95% CI for median: (7.0, 8.0)
(No analytic formula needed--that is the power of bootstrap)

=======================================================
SAMPLE SIZE PLANNING
=======================================================
Estimated sigma: 2.1198
Target SE:       0.05
Required n:      1,797

To halve SE from 0.05 to 0.025:
  Need n = 7,188  (4x as many!)
```

---

## ML Applications: Where Sampling Theory Runs Your Stack

### 1. Train/Test/Validation Splits

Your test set is a sample from the true data distribution. CLT tells you the test accuracy is approximately Gaussian around the true accuracy. Standard error tells you how wide the uncertainty band is. If your test set has $n = 5{,}000$ examples and accuracy is $p = 0.92$:

$$SE(p) = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0.92 \times 0.08}{5000}} \approx 0.0038$$

Your 95% confidence interval on accuracy is roughly $0.92 \pm 0.008$, or $(0.912, 0.928)$. That is what you should report, not a bare "92%".

### 2. Mini-Batch Gradient Descent

Each mini-batch is a random sample of size $B$ from your training set. By LLN, the mini-batch gradient converges to the true gradient as $B$ increases. By CLT, the noise in the gradient estimate is approximately Gaussian. The standard error of the gradient scales as $1/\sqrt{B}$--so doubling the batch size only reduces gradient noise by $\sqrt{2} \approx 1.41$x.

### 3. A/B Testing

You are testing a new recommendation algorithm. You randomly assign 10,000 users to treatment and 10,000 to control. The CLT justifies modeling the difference in means as Gaussian. Sample size formulas use $SE = \sigma / \sqrt{n}$ to determine how many users you need to detect a given effect size.

### 4. Cross-Validation

$k$-fold cross-validation gives you $k$ estimates of model performance. Averaging them leverages LLN for a more stable estimate. The variance across folds gives you a sense of uncertainty.

### 5. Bootstrap in Practice

Bootstrap your test predictions to get confidence intervals on any metric: accuracy, F1, AUC, custom business metrics. No formulas needed--just resample and recompute.

---

## Sample Size Guidelines

| Goal | Rule of Thumb |
|------|---------------|
| CLT to apply | $n \geq 30$ (more for skewed data) |
| Reliable proportions | $n \geq 100$, $np \geq 10$, $n(1-p) \geq 10$ |
| Detect small effects | Power analysis needed (see hypothesis testing chapter) |
| Margin of error $d$ | $n \geq (z \cdot \sigma / d)^2$ |
| Standard error target | $n \geq (\sigma / SE_{\text{target}})^2$ |

---

## Common Pitfalls

**1. Assuming random sampling when it is not.**
Your production logs are not a random sample. They over-represent high-traffic times, active users, and successful requests. Convenience samples are everywhere in ML--social media data, web scrapes, volunteer-response surveys. Treat them with suspicion.

**2. Ignoring sampling bias in benchmark datasets.**
ImageNet, COCO, and other benchmarks have known demographic and geographic biases. A model that scores 95% on the benchmark may score 80% on a more representative sample.

**3. Confusing sample size with population size.**
The standard error depends on $n$ (your sample size), not $N$ (the population size). Whether the population is 100,000 or 100 billion, if your sample is 1,000, your SE is the same (assuming $N \gg n$). This surprises many engineers.

**4. Applying CLT to individual observations.**
The CLT says *averages* are Gaussian. It says nothing about individual data points. If you are modeling individual user behavior, do not assume normality just because "CLT."

**5. Expecting linear improvement with more data.**
Because $SE \propto 1/\sqrt{n}$, going from 1,000 to 2,000 samples helps much more than going from 100,000 to 101,000. Diminishing returns are baked into the math.

**6. Forgetting the i.i.d. assumption.**
LLN and CLT require independent, identically distributed samples. Time-series data, clustered data, and data with dependencies violate this. You need different tools (block bootstrap, clustered standard errors) in those cases.

---

## Exercises

### Exercise 1: Standard Error Calculation (Movie Ratings)

You sample 1,000 movie reviews. The sample standard deviation is $s = 2.1$. Compute:
1. The standard error of the sample mean.
2. An approximate 95% confidence interval for the true mean if $\bar{x} = 7.24$.
3. How many reviews would you need to get $SE \leq 0.03$?

**Solution**:

$$SE = \frac{s}{\sqrt{n}} = \frac{2.1}{\sqrt{1000}} \approx 0.0664$$

$$\text{95% CI} = \bar{x} \pm 1.96 \cdot SE = 7.24 \pm 0.130 = (7.110, 7.370)$$

$$n \geq \left(\frac{s}{SE_{\text{target}}}\right)^2 = \left(\frac{2.1}{0.03}\right)^2 = 4{,}900$$

```python
import numpy as np

s = 2.1
n = 1000
x_bar = 7.24

se = s / np.sqrt(n)
ci_lower = x_bar - 1.96 * se
ci_upper = x_bar + 1.96 * se
required_n = int(np.ceil((s / 0.03) ** 2))

print(f"SE: {se:.4f}")             # 0.0664
print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")  # (7.110, 7.370)
print(f"Required n for SE=0.03: {required_n}")        # 4900
```

### Exercise 2: Identifying Sampling Bias

You train a movie recommendation model on reviews from your platform. It performs well on a held-out test set but poorly when you license it to a different platform. List three possible sources of sampling bias and explain how each would cause this failure.

**Solution**:

1. **Selection bias**: your platform's users skew younger/more tech-savvy. The other platform has a broader demographic. Your model learned preferences of a non-representative subset.

2. **Temporal bias**: your training data reflects current trends on your platform. The other platform may have different content, or users who are at a different stage of engagement.

3. **Response bias**: on your platform, only users who feel strongly leave reviews. On the other platform, ratings might be solicited from all users, changing the distribution of observed ratings.

All three are instances of $P_{\text{train}}(X, Y) \neq P_{\text{deploy}}(X, Y)$.

### Exercise 3: CLT Application

Movie ratings on your platform range from 1 to 10 (discrete, skewed). The true mean is 7.2 and standard deviation is 2.1. You sample 500 reviews.

a) What is the expected value of the sample mean?
b) What is the standard error?
c) What range contains approximately 95% of possible sample means?
d) Why can you use the normal distribution here even though ratings are discrete and skewed?

**Solution**:

```python
import numpy as np

mu = 7.2
sigma = 2.1
n = 500

# a) Expected value of the sample mean
print(f"a) E[x̄] = {mu}")                            # 7.2

# b) Standard error
se = sigma / np.sqrt(n)
print(f"b) SE = {se:.4f}")                            # 0.0939

# c) 95% interval
lower = mu - 1.96 * se
upper = mu + 1.96 * se
print(f"c) 95% interval: ({lower:.3f}, {upper:.3f})") # (7.016, 7.384)

# d) CLT: the mean of 500 observations is approximately Gaussian
#    regardless of the shape of the underlying distribution.
#    n=500 >> 30, so the normal approximation is excellent.
print("d) CLT applies to the MEAN of 500 observations,")
print("   not to individual observations. n=500 is large enough.")
```

### Exercise 4: Bootstrap in Practice

You have 200 test-set predictions with their true labels. You want a 95% confidence interval on the F1 score. Describe the bootstrap procedure step by step.

**Solution**:

1. You have 200 (prediction, label) pairs. Compute F1 on the full set: this is your point estimate.
2. Repeat $B = 10{,}000$ times:
   - Sample 200 pairs *with replacement* from your 200 pairs.
   - Compute F1 on this bootstrap sample.
   - Store the result.
3. Sort the 10,000 bootstrap F1 values.
4. Take the 2.5th percentile and 97.5th percentile. That is your 95% confidence interval.

```python
from sklearn.metrics import f1_score
import numpy as np

# y_true, y_pred are arrays of length 200
point_estimate = f1_score(y_true, y_pred)

bootstrap_f1s = []
for _ in range(10_000):
    idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
    bootstrap_f1s.append(f1_score(y_true[idx], y_pred[idx]))

bootstrap_f1s = np.array(bootstrap_f1s)
ci_lower = np.percentile(bootstrap_f1s, 2.5)
ci_upper = np.percentile(bootstrap_f1s, 97.5)

print(f"F1: {point_estimate:.3f}, 95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
```

No closed-form SE formula for F1 exists. Bootstrap handles it effortlessly.

---

## Summary

| Concept | What It Says | SWE Analogy |
|---------|-------------|-------------|
| **LLN** | Sample mean converges to true mean as $n$ grows | Your 5-min load average converges to the true rate |
| **CLT** | Sample means are approximately Gaussian for large $n$ | Batch gradients are Gaussian even when individual gradients are not |
| **Standard Error** | $SE = \sigma / \sqrt{n}$; uncertainty shrinks slowly | To halve your error bar, quadruple your data |
| **Bootstrap** | Resample your data to simulate running the experiment many times | The "mock server" approach: replay your data to estimate uncertainty |
| **Sampling Bias** | Non-random samples yield biased estimates at any size | Production data is not a random sample of all possible inputs |

**Key formulas preserved**:

$$\bar{X}_n \xrightarrow{P} \mu \quad \text{(LLN)}$$

$$\bar{X}_n \dot{\sim} \; N\!\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{(CLT)}$$

$$SE(\bar{X}) = \frac{\sigma}{\sqrt{n}}$$

The running example: from 1,000 movie reviews out of 1M, you computed a sample mean of 7.24 with $SE \approx 0.066$. LLN told you the average would converge. CLT told you the sampling distribution is Gaussian. Standard error told you how wide it is. And bootstrap gave you a confidence interval without assumptions.

---

*You can estimate population parameters from samples. But how precise are those estimates? That is estimation theory--the subject of the next chapter.*
