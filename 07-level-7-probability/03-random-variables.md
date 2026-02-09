# Chapter 3: Random Variables

> **Building On** -- Events give you "something happened or didn't." But ML needs numbers -- predictions, scores, errors. Random variables bridge the gap: they assign numbers to random outcomes.

---

Your model outputs a number: 0.73. Is that a probability? A score? How confident should you be? To answer these questions, you need random variables -- the bridge between probability events and actual numbers that your code computes.

Here is the thing most courses bury: every time you call `model.predict(x)`, the result is a random variable. Each training run produces different weights (different random seed, different shuffle order, different dropout masks) which produce different predictions. That 0.73 is one *realization* of a random variable -- one sample from a distribution of possible outputs. Understanding this is the key to reasoning about model uncertainty, calibration, and everything probabilistic in ML.

---

## Running Example: Movie Ratings

We will use one example throughout this chapter to ground every concept:

> A user's movie rating is a **discrete** random variable: $X \in \{1, 2, 3, 4, 5\}$.
> The model's predicted rating is a **continuous** random variable: $\hat{X} \in [1.0, 5.0]$.

The user clicks 4 stars -- that is a discrete outcome. Your recommender system predicts 3.72 -- that is a continuous outcome. Both are random variables because they carry uncertainty: you do not know what the user will rate before they rate, and you do not know what the model will predict before you train it.

---

## What Is a Random Variable?

### The Formal Definition

A **random variable** $X$ is a function that maps outcomes in a sample space $\Omega$ to real numbers:

$$X: \Omega \rightarrow \mathbb{R}$$

That is the textbook line. Now let's make it concrete.

**Example**: A coin flip has $\Omega = \{H, T\}$. Define:
- $X(H) = 1$
- $X(T) = 0$

$X$ takes the abstract event "heads" and turns it into the number 1. That is it. A random variable is a function from outcomes to numbers.

> **You Already Know This** -- A random variable is a function that maps outcomes to numbers -- like a callback that converts events to metrics. You have written `lambda event: 1 if event == "success" else 0` a hundred times. That is literally a random variable.

### Code-First: Generate Random Data

Let's start where you would start -- by generating data and looking at it.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# --- Discrete: simulate 10,000 user movie ratings ---
# Suppose users rate movies with these probabilities:
#   1 star: 5%, 2 stars: 10%, 3 stars: 20%, 4 stars: 35%, 5 stars: 30%
rating_values = [1, 2, 3, 4, 5]
rating_probs  = [0.05, 0.10, 0.20, 0.35, 0.30]

user_ratings = np.random.choice(rating_values, size=10_000, p=rating_probs)

print("First 20 user ratings:", user_ratings[:20])
print(f"Mean rating: {user_ratings.mean():.2f}")
# First 20 user ratings: [4 5 3 4 5 4 5 4 4 3 4 5 5 5 4 3 5 4 4 2]
# Mean rating: 3.75

# --- Continuous: simulate 10,000 model predicted ratings ---
# Model predictions cluster around 3.7 with some noise
predicted_ratings = np.clip(np.random.normal(loc=3.7, scale=0.6, size=10_000), 1.0, 5.0)

print(f"\nFirst 5 predicted ratings: {predicted_ratings[:5].round(3)}")
print(f"Mean predicted: {predicted_ratings.mean():.2f}")
# First 5 predicted ratings: [3.898 3.617 3.988 4.614 3.56 ]
# Mean predicted: 3.70
```

You now have two arrays: one discrete, one continuous. Both are samples from random variables. The next step is to *describe* their distributions.

---

## Discrete Random Variables and the PMF

### Plot the Histogram First

```python
import matplotlib.pyplot as plt

# Count how often each rating appears
values, counts = np.unique(user_ratings, return_counts=True)
frequencies = counts / len(user_ratings)

for v, f in zip(values, frequencies):
    print(f"Rating {v}: {f:.4f} (observed) vs {rating_probs[v-1]:.4f} (true)")
# Rating 1: 0.0497 (observed) vs 0.0500 (true)
# Rating 2: 0.0982 (observed) vs 0.1000 (true)
# Rating 3: 0.2018 (observed) vs 0.2000 (true)
# Rating 4: 0.3494 (observed) vs 0.3500 (true)
# Rating 5: 0.3009 (observed) vs 0.3000 (true)
```

That table *is* the PMF -- you just computed it empirically.

### ASCII Diagram: PMF as a Bar Chart

```
PMF: P(X = k)  -- User Movie Ratings

0.35 |              ########
0.30 |              ########  ########
0.25 |              ########  ########
0.20 |    ########  ########  ########
0.15 |    ########  ########  ########
0.10 |  ########    ########  ########
0.05 |  ########  ########  ########  ########
     +----+----+----+----+----+----->
          1    2    3    4    5       rating (k)

Each bar height = P(X = k)
All bar heights sum to 1.0
```

> **You Already Know This** -- PMF is a histogram of a discrete variable. You have plotted these a thousand times. The only rule: the bars must sum to 1.

### Formal Definition: PMF

The **Probability Mass Function (PMF)** gives the probability that a discrete random variable takes each value:

$$p_X(x) = P(X = x)$$

Properties of PMF:
1. $p_X(x) \geq 0$ for all $x$
2. $\sum_x p_X(x) = 1$
3. $P(X \in A) = \sum_{x \in A} p_X(x)$

### PMF in Action: Movie Ratings

```python
# Define the PMF explicitly
pmf = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.35, 5: 0.30}

# Verify it is a valid PMF
assert all(p >= 0 for p in pmf.values()), "PMF values must be non-negative"
assert abs(sum(pmf.values()) - 1.0) < 1e-10, "PMF must sum to 1"

# P(user rates 4 or 5) -- "the user liked the movie"
p_liked = pmf[4] + pmf[5]
print(f"P(X >= 4) = {p_liked:.2f}")  # 0.65

# P(user rates below 3) -- "the user disliked the movie"
p_disliked = pmf[1] + pmf[2]
print(f"P(X < 3) = {p_disliked:.2f}")  # 0.15
```

The PMF lets you answer any question about discrete probabilities by summing the right bars.

---

## Continuous Random Variables and the PDF

### From Histogram to Density

Now look at the model's *predicted* ratings. These are continuous -- the model can output 3.712 or 3.713 or anything in between.

```python
# Plot a histogram of predicted ratings
# As you increase the number of bins, the histogram approaches a smooth curve
for n_bins in [10, 50, 200]:
    counts, bin_edges = np.histogram(predicted_ratings, bins=n_bins, density=True)
    print(f"Bins: {n_bins:>3d} | max density: {counts.max():.3f}")
# Bins:  10 | max density: 0.666
# Bins:  50 | max density: 0.726
# Bins: 200 | max density: 0.867
```

As the number of bins approaches infinity, the histogram becomes a smooth curve. That curve is the **Probability Density Function (PDF)**.

### ASCII Diagram: PDF as a Continuous Curve

```
PDF: f(x)  -- Model's Predicted Rating

0.70 |            *****
0.60 |          **     **
0.50 |        **         **
0.40 |       *             *
0.30 |      *               *
0.20 |    **                 **
0.10 |  **                     **
     +--*--+----+----+----+----*---->
        1  1.5  2.5  3.5  4.5  5     predicted rating (x)

                |<-------->|
               P(3 < X < 4) = shaded AREA, not the curve height

The total area under the curve = 1.0
```

> **You Already Know This** -- PDF is the continuous version of a histogram: "probability per unit" rather than "probability at a point." The curve height is *density*, not probability. Probability lives in areas.

### Formal Definition: PDF

The **Probability Density Function (PDF)** defines probabilities for continuous random variables via integration:

$$P(a \leq X \leq b) = \int_a^b f_X(x) \, dx$$

Properties of PDF:
1. $f_X(x) \geq 0$ for all $x$
2. $\int_{-\infty}^{\infty} f_X(x) \, dx = 1$
3. **Important**: $f_X(x)$ is NOT a probability! It can be greater than 1.
4. $P(X = x) = 0$ for any specific value $x$

> **Common Mistake** -- `P(X = exactly 3.7) = 0` for continuous random variables. Probability lives in INTERVALS, not points. This trips people up constantly. The PDF value at 3.7 tells you the *density* -- how concentrated the probability is near 3.7 -- not the probability of landing on 3.7 exactly. Think of it like population density: "10,000 people per square mile" does not mean 10,000 people live at a single point.

### PDF in Action: Model Predictions

```python
# Model predictions ~ Normal(3.7, 0.6), clipped to [1, 5]
# For simplicity, let's use an unclipped normal to demonstrate the math

mu, sigma = 3.7, 0.6
model_dist = stats.norm(loc=mu, scale=sigma)

# PDF values -- these are DENSITIES, not probabilities
print("PDF values (densities, NOT probabilities):")
for x in [2.0, 3.0, 3.7, 4.0, 5.0]:
    print(f"  f({x}) = {model_dist.pdf(x):.4f}")
# f(2.0) = 0.0087
# f(3.0) = 0.3187
# f(3.7) = 0.6650  <-- peak of the bell curve
# f(4.0) = 0.5467
# f(5.0) = 0.0305

# Probability that prediction is between 3.0 and 4.0
p_3_to_4 = model_dist.cdf(4.0) - model_dist.cdf(3.0)
print(f"\nP(3.0 < X < 4.0) = {p_3_to_4:.4f}")  # 0.6892

# Verify via integration
from scipy.integrate import quad
integral_result, _ = quad(model_dist.pdf, 3.0, 4.0)
print(f"Integral of PDF from 3.0 to 4.0 = {integral_result:.4f}")  # same

# Verify total area = 1
total_area, _ = quad(model_dist.pdf, -np.inf, np.inf)
print(f"Total area under PDF = {total_area:.4f}")  # 1.0000
```

---

## The CDF: The Universal Descriptor

The **Cumulative Distribution Function (CDF)** answers one question: "What fraction of values are at or below this threshold?"

$$F_X(x) = P(X \leq x)$$

> **You Already Know This** -- CDF is the percentile: "what fraction of values are below this threshold?" -- like p99 latency. When you say "our p99 latency is 250ms," you mean $F(250) = 0.99$ -- 99% of requests complete in 250ms or less. That is a CDF value.

### ASCII Diagram: CDF

```
CDF: F(x) -- Discrete (User Ratings)     CDF: F(x) -- Continuous (Model Predictions)

1.00 |                    *----          1.00 |                          *********
     |              *-----               0.80 |                     ****
0.70 |              *                    0.60 |                  ***
     |        *-----                     0.40 |               ***
0.35 |        *                          0.20 |           ****
     |  *-----                           0.05 |     ******
0.05 |  *                                     +--*--+----+----+----+----+----->
     +--+----+----+----+----+----->            1    2    3    4    5
        1    2    3    4    5
                                         Smooth S-shaped curve
     Staircase: jumps at each rating     (always non-decreasing, 0 to 1)
```

### Formal Properties of CDF

1. $F_X(-\infty) = 0$ and $F_X(\infty) = 1$
2. $F_X$ is non-decreasing
3. $F_X$ is right-continuous
4. $P(a < X \leq b) = F_X(b) - F_X(a)$

### CDF in Action: Both Discrete and Continuous

```python
# --- Discrete CDF: User ratings ---
# Build CDF from PMF
pmf_values = [0.05, 0.10, 0.20, 0.35, 0.30]
cdf_values = np.cumsum(pmf_values)

print("Discrete CDF (User Ratings):")
for k, f in zip(rating_values, cdf_values):
    print(f"  F({k}) = P(X <= {k}) = {f:.2f}")
# F(1) = 0.05
# F(2) = 0.15
# F(3) = 0.35
# F(4) = 0.70
# F(5) = 1.00

# P(user rates between 2 and 4 inclusive)
# P(2 <= X <= 4) = F(4) - F(1) = 0.70 - 0.05 = 0.65
print(f"\nP(2 <= X <= 4) = F(4) - F(1) = {cdf_values[3] - cdf_values[0]:.2f}")

# --- Continuous CDF: Model predictions ---
model_dist = stats.norm(loc=3.7, scale=0.6)

print("\nContinuous CDF (Model Predictions):")
for x in [1.0, 2.0, 3.0, 3.7, 4.0, 5.0]:
    print(f"  F({x}) = P(X <= {x}) = {model_dist.cdf(x):.4f}")

# Inverse CDF: what predicted rating is at the 90th percentile?
p90 = model_dist.ppf(0.90)
print(f"\n90th percentile prediction: {p90:.3f}")
# This is like asking: "90% of the time, the model predicts below what value?"

# More percentiles -- familiar territory if you've worked with latency metrics
print("\nPercentiles (like latency p50/p90/p95/p99):")
for p in [0.50, 0.90, 0.95, 0.99]:
    print(f"  p{int(p*100):>2d} = {model_dist.ppf(p):.3f}")
```

---

## Relationships: PMF, PDF, and CDF

These three functions are different views of the same distribution. Here is how they connect:

### From PMF to CDF (discrete)

$$F_X(x) = \sum_{t \leq x} p_X(t)$$

Sum up all the PMF bars from the left up to $x$.

### From PDF to CDF (continuous)

$$F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt$$

Accumulate the area under the PDF curve from the left up to $x$.

### From CDF to PDF (continuous)

$$f_X(x) = \frac{d}{dx} F_X(x)$$

The PDF is the derivative of the CDF. Where the CDF rises steeply, the PDF is tall (high density). Where the CDF is flat, the PDF is near zero.

### From CDF to PMF (discrete)

$$p_X(k) = F_X(k) - F_X(k-1)$$

The PMF at each point equals the jump in the CDF at that point.

```python
# Demonstrate all relationships in code
print("=== Relationships Demo ===\n")

# --- Discrete: PMF <-> CDF ---
pmf_dict = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.35, 5: 0.30}

# PMF -> CDF: cumulative sum
cdf_from_pmf = {}
running_sum = 0
for k in sorted(pmf_dict.keys()):
    running_sum += pmf_dict[k]
    cdf_from_pmf[k] = running_sum
print("PMF -> CDF:")
for k in sorted(cdf_from_pmf.keys()):
    print(f"  F({k}) = {cdf_from_pmf[k]:.2f}")

# CDF -> PMF: differences
print("\nCDF -> PMF:")
prev = 0
for k in sorted(cdf_from_pmf.keys()):
    pmf_recovered = cdf_from_pmf[k] - prev
    prev = cdf_from_pmf[k]
    print(f"  P(X={k}) = {pmf_recovered:.2f}")

# --- Continuous: PDF <-> CDF ---
dist = stats.norm(loc=3.7, scale=0.6)

# PDF -> CDF via integration
x_point = 4.0
cdf_via_integral, _ = quad(dist.pdf, -np.inf, x_point)
cdf_direct = dist.cdf(x_point)
print(f"\nPDF -> CDF at x={x_point}:")
print(f"  Integral of PDF from -inf to {x_point}: {cdf_via_integral:.6f}")
print(f"  Direct CDF call:                        {cdf_direct:.6f}")

# CDF -> PDF via numerical derivative
dx = 1e-7
pdf_via_derivative = (dist.cdf(x_point + dx) - dist.cdf(x_point - dx)) / (2 * dx)
pdf_direct = dist.pdf(x_point)
print(f"\nCDF -> PDF at x={x_point}:")
print(f"  Numerical derivative of CDF: {pdf_via_derivative:.6f}")
print(f"  Direct PDF call:             {pdf_direct:.6f}")
```

---

## Connecting to ML: Why This Matters

Now that you have the formal tools, let's connect them back to the systems you build.

### Model Predictions Are Random Variables

Every time you train a neural network, randomness enters through:
- Weight initialization (random seed)
- Data shuffling (random batch order)
- Dropout (random neuron masking)
- Data augmentation (random transforms)

This means the trained weights are random variables, and therefore the predictions are random variables too. When your model says $\hat{y} = 0.73$, that is one sample from the distribution of predictions across all possible training runs.

```python
# Simulate: train a model 1000 times with different seeds
# (simplified -- just showing the concept)
np.random.seed(0)

# Imagine these are the predicted probabilities for one test sample
# across 1000 different training runs
predictions_across_runs = np.random.normal(loc=0.73, scale=0.04, size=1000)
predictions_across_runs = np.clip(predictions_across_runs, 0, 1)

print("Model prediction for one sample across 1000 training runs:")
print(f"  Mean: {predictions_across_runs.mean():.4f}")
print(f"  Std:  {predictions_across_runs.std():.4f}")
print(f"  Min:  {predictions_across_runs.min():.4f}")
print(f"  Max:  {predictions_across_runs.max():.4f}")

# What fraction of runs give a prediction > 0.5?
p_positive = np.mean(predictions_across_runs > 0.5)
print(f"  P(prediction > 0.5) = {p_positive:.4f}")
# If this is close to 1.0, the model is "confident" across runs
```

### Where Random Variables Appear in ML

| ML Concept | Random Variable Type | What You Use |
|------------|---------------------|--------------|
| User ratings | Discrete (PMF) | $P(X = k)$ for each star rating |
| Model predictions | Continuous (PDF) | Density of predictions around a value |
| Feature values | Continuous (PDF) | Distribution of each feature column |
| Class probabilities | Continuous on [0,1] (PDF) | Output of softmax / sigmoid |
| Loss across batches | Continuous (PDF) | Varies each batch due to sampling |
| Dropout mask | Discrete Bernoulli (PMF) | Each neuron: on/off with probability $p$ |

### Cross-Entropy Loss Is Defined on PMFs

The cross-entropy loss -- the one you use in almost every classification task -- is directly defined in terms of PMFs:

$$L = -\sum_k y_k \log \hat{p}_k$$

where $y_k$ is the true PMF (one-hot: all mass on the correct class) and $\hat{p}_k$ is the model's predicted PMF over classes. Without understanding PMFs, this formula is just symbols.

### scipy.stats: Your Toolkit for Random Variables

`scipy.stats` gives you a unified interface for both discrete and continuous distributions. Every distribution object provides the same methods:

```python
# --- Discrete: Binomial(n=10, p=0.3) ---
binomial = stats.binom(n=10, p=0.3)
print("Binomial(n=10, p=0.3):")
print(f"  PMF at k=3:  P(X=3)  = {binomial.pmf(3):.4f}")
print(f"  CDF at k=3:  P(X<=3) = {binomial.cdf(3):.4f}")
print(f"  Mean:    {binomial.mean():.2f}")
print(f"  Variance: {binomial.var():.2f}")
print(f"  Samples: {binomial.rvs(size=5)}")

# --- Continuous: Normal(mu=0, sigma=1) ---
normal = stats.norm(loc=0, scale=1)
print("\nNormal(mu=0, sigma=1):")
print(f"  PDF at x=0:  f(0)    = {normal.pdf(0):.4f}")
print(f"  CDF at x=0:  P(X<=0) = {normal.cdf(0):.4f}")
print(f"  Mean:    {normal.mean():.2f}")
print(f"  Variance: {normal.var():.2f}")
print(f"  Samples: {normal.rvs(size=5).round(3)}")

# Key methods (same for ALL distributions):
# .pmf(k) / .pdf(x)  -- probability mass / density
# .cdf(x)             -- cumulative distribution function
# .ppf(q)             -- inverse CDF (quantile function)
# .rvs(size)          -- random samples
# .mean(), .var()     -- moments
# .interval(alpha)    -- confidence interval
```

---

## Discrete vs Continuous: How to Choose

| Use Discrete | Use Continuous |
|---|---|
| Countable outcomes (star ratings, word counts) | Measurements on a scale (predicted scores) |
| Categories or labels (spam/not-spam) | Physical quantities (temperature, distance) |
| Counts: 0, 1, 2, ... (click counts) | Anything fractional (model weights, gradients) |
| Binary decisions (user clicked or didn't) | Time, probabilities, loss values |

**Rule of thumb**: if you would store it as an `int`, it is probably discrete. If you would store it as a `float`, it is probably continuous.

---

## Exercises

### Exercise 1: PMF Verification

Given P(X=1) = 0.2, P(X=2) = 0.3, P(X=3) = 0.4, P(X=4) = 0.1. Verify this is a valid PMF and calculate P(X > 2).

**Solution**:
```python
pmf = {1: 0.2, 2: 0.3, 3: 0.4, 4: 0.1}

# Check validity
all_non_negative = all(p >= 0 for p in pmf.values())
sums_to_one = abs(sum(pmf.values()) - 1.0) < 1e-10

print(f"All non-negative: {all_non_negative}")  # True
print(f"Sums to 1: {sums_to_one}")  # True
print(f"Valid PMF: {all_non_negative and sums_to_one}")  # True

# P(X > 2) = P(X=3) + P(X=4)
P_X_gt_2 = pmf[3] + pmf[4]
print(f"P(X > 2) = {P_X_gt_2}")  # 0.5
```

### Exercise 2: CDF to PMF

Given CDF: F(1) = 0.1, F(2) = 0.4, F(3) = 0.7, F(4) = 1.0. Find the PMF.

**Solution**:
```python
cdf = {1: 0.1, 2: 0.4, 3: 0.7, 4: 1.0}

# PMF: P(X=k) = F(k) - F(k-1)
# For k=1, P(X=1) = F(1) - 0 = F(1)
pmf = {}
prev = 0
for k in sorted(cdf.keys()):
    pmf[k] = cdf[k] - prev
    prev = cdf[k]
    print(f"P(X = {k}) = {pmf[k]}")

# Output: P(X=1)=0.1, P(X=2)=0.3, P(X=3)=0.3, P(X=4)=0.3
```

### Exercise 3: PDF Integration

For a continuous RV with PDF f(x) = 2x for 0 <= x <= 1, find P(0.5 < X < 0.8).

**Solution**:
```python
from scipy.integrate import quad

def pdf(x):
    if 0 <= x <= 1:
        return 2 * x
    return 0

# Verify it integrates to 1
total, _ = quad(pdf, 0, 1)
print(f"Total area: {total}")  # 1.0

# P(0.5 < X < 0.8)
prob, _ = quad(pdf, 0.5, 0.8)
print(f"P(0.5 < X < 0.8) = {prob}")  # 0.39

# Analytical: integral of 2x from 0.5 to 0.8 = x^2 evaluated
# = 0.8^2 - 0.5^2 = 0.64 - 0.25 = 0.39
```

### Exercise 4: ML Application -- Prediction Confidence

A binary classifier outputs a score $S \sim \text{Normal}(0.8, 0.1)$ for positive samples. You classify as positive if $S > 0.5$. What fraction of positive samples get classified correctly?

**Solution**:
```python
# Score distribution for positive samples
score_dist = stats.norm(loc=0.8, scale=0.1)

# P(S > 0.5) = 1 - P(S <= 0.5) = 1 - F(0.5)
recall = 1 - score_dist.cdf(0.5)
print(f"P(S > 0.5) = {recall:.4f}")  # 0.9987

# What threshold gives us exactly 95% recall?
threshold_95 = score_dist.ppf(0.05)  # 5th percentile = 95% above
print(f"Threshold for 95% recall: {threshold_95:.4f}")  # 0.6355
```

---

## Summary

- **Random Variable**: A function mapping outcomes to numbers -- the bridge between abstract events and the numerical computations your code performs
- **Discrete Random Variable**: Takes countable values; described by the PMF $p_X(x) = P(X = x)$
- **Continuous Random Variable**: Takes values from a continuum; described by the PDF $f_X(x)$
- **PMF Properties**: Non-negative, sums to 1
- **PDF Properties**: Non-negative, integrates to 1. The density value $f(x)$ is NOT a probability -- it can exceed 1. Probability lives in intervals (areas under the curve), not at points.
- **CDF**: $F_X(x) = P(X \leq x)$, works for both discrete and continuous, always goes from 0 to 1
- **Relationships**:
  - CDF = integral of PDF (continuous) or cumulative sum of PMF (discrete)
  - PDF = derivative of CDF
  - PMF at a point = jump in CDF at that point
- **In ML**: inputs, outputs, loss values, and even model weights are all random variables. Understanding their distributions is how you reason about uncertainty, calibration, and model behavior.
- **`scipy.stats`**: Unified interface with `.pmf()/.pdf()`, `.cdf()`, `.ppf()`, `.rvs()` for any distribution

---

> **What's Next** -- You can describe distributions with PMFs and PDFs. But how do you summarize a distribution with a single number? Expected value (the mean) and variance -- the key summary statistics that compress an entire distribution into the numbers you actually use in loss functions, optimization, and model evaluation.

**Next**: [Chapter 4: Expectation and Moments](04-expectation-and-moments.md)
