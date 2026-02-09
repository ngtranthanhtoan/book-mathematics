# Descriptive Statistics

> Probability gives you the theory of randomness. Statistics does the reverse: given actual data, what can you infer about the underlying distribution?

Before you train a model, you need to understand your data. What's the average? How spread out is it? Are there outliers? Are features correlated? Descriptive statistics is your data's medical checkup — and skipping it is how you end up debugging models when you should be debugging data.

You know that feeling when a model's loss won't converge and you spend three days tuning hyperparameters, only to discover that one feature column is 80% NaN and another has a standard deviation of zero? That's what happens when you skip EDA. Descriptive statistics gives you a small set of numbers — means, medians, variances, correlations — that tell you what your data actually looks like before you feed it into anything.

---

## The Problem

You got a CSV with 10M rows. What do you look at first?

You can't eyeball 10 million rows. You can't scroll through them. You need a handful of summary numbers that compress all that data into something your brain can act on. That's the entire job of descriptive statistics: take a dataset of arbitrary size and reduce it to a few numbers that capture its shape, center, spread, and relationships.

Throughout this chapter, you're going to work through a running example: **EDA on a movie ratings dataset**. You have a table with columns like `rating` (1-10), `budget` (in dollars), and `revenue` (in dollars) for thousands of films. Your goal: understand the average rating, how ratings are distributed, and whether budget and revenue are correlated. Every concept below gets applied to this dataset.

---

## Measures of Center: Where Is the "Typical" Value?

### Mean (Arithmetic Average)

The **mean** is the sum of all values divided by the count:

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

For a probability distribution with PDF $f(x)$, the **expected value** (population mean) is:

$$\mu = E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

**SWE bridge.** The mean is the "average" on your monitoring dashboard. When your Grafana panel says "average response time: 120ms," that's the arithmetic mean of every request duration in the window. It's the number everyone reaches for first.

**Properties of the mean:**

- It's the "balance point" of the data — if you placed weights on a number line, the mean is where the line balances.
- It's sensitive to outliers. One 30-second timeout in a thousand 100ms requests will yank the average up.
- It minimizes the sum of squared deviations: $\arg\min_c \sum_i (x_i - c)^2 = \bar{x}$

**Running example.** You compute the average movie rating across your dataset and get $\bar{x} = 6.1$. That tells you the "typical" film lands slightly above the midpoint of the 1-10 scale. But is that the whole story? Not even close.

### Median

The **median** is the middle value when you sort the data. For $n$ observations:

$$\text{Median} = \begin{cases} x_{(n+1)/2} & \text{if } n \text{ is odd} \\ \frac{x_{n/2} + x_{n/2+1}}{2} & \text{if } n \text{ is even} \end{cases}$$

**SWE bridge.** The median is the **p50 latency** — the value where half the requests are faster and half are slower. You use p50 (and p99) instead of the average precisely because they're robust to spikes. A single 10-second outlier doesn't move the p50 at all. That's why SRE dashboards lean on percentiles rather than averages.

**Properties of the median:**

- Robust to outliers. Add a billionaire to a room of teachers and the median salary barely moves.
- Minimizes the sum of absolute deviations: $\arg\min_c \sum_i |x_i - c| = \text{Median}$

**Running example.** The median movie rating is 6.3. That's close to the mean of 6.1, which tells you the rating distribution isn't terribly skewed. If the mean were 6.1 but the median were 7.5, you'd know a long left tail of terrible movies was dragging the average down.

### Mode

The **mode** is the most frequently occurring value. A distribution can be:

- **Unimodal**: One peak (most common)
- **Bimodal**: Two peaks (e.g., ratings that cluster at 1-star and 5-stars)
- **Multimodal**: Multiple peaks

The mode is most useful for categorical data. For continuous data, you typically look at the histogram shape rather than a single mode value.

**Running example.** The mode of movie ratings is 7. That's the single most common rating. This makes sense — audiences tend to rate films they liked, and 7 is a comfortable "good, not great" score.

---

## Measures of Spread: How Dispersed Is the Data?

Knowing the center isn't enough. Two datasets can have the same mean but completely different shapes. You need to know how spread out the values are.

### Variance

**Variance** measures the average squared deviation from the mean:

$$\sigma^2 = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

**Sample variance** (the unbiased estimator you should always use on real data):

$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

Why $n - 1$ instead of $n$? This is **Bessel's correction**. When you estimate the mean from the same sample, you've "used up" one degree of freedom. Using $n$ would systematically underestimate the true variance. In NumPy, this means you always pass `ddof=1` for sample data.

### Standard Deviation

**Standard deviation** is the square root of variance:

$$\sigma = \sqrt{\sigma^2}$$

The reason you care about standard deviation more than variance in practice is units. If your data is in dollars, the variance is in dollars-squared (which is meaningless to a human). The standard deviation is back in dollars.

**SWE bridge.** Standard deviation is the **"jitter"** in your response times. If your average latency is 100ms with a standard deviation of 5ms, your service is rock-solid. If the standard deviation is 200ms, the average is meaningless — some requests are instant and others are glacial. That's exactly the kind of thing you need to know before using latency as an ML feature.

**Running example.** The standard deviation of movie ratings is $s = 1.4$. Combined with the mean of 6.1, that tells you most ratings fall between roughly 4.7 and 7.5 (one standard deviation on either side). The data isn't wildly spread out — most movies get middling-to-good ratings.

---

## Visualizing the Distribution

Numbers alone don't tell you everything. You need to look at the shape. Here are the two most important plots for a single variable.

### Histogram: The Shape of Your Data

A histogram bins your values and counts how many fall in each bin. Here's what the movie rating distribution might look like:

```
  Movie Rating Distribution (n = 5,000 films)

  Count
  900 |
  800 |              #####
  700 |           ########
  600 |        ###########
  500 |      #############
  400 |    ################
  300 |   ##################
  200 |  ####################
  100 | #######################
    0 +---+---+---+---+---+---+---+---+---+---
      1   2   3   4   5   6   7   8   9  10
                      Rating

  Mean = 6.1  |  Median = 6.3  |  Std Dev = 1.4
```

You can immediately see this is **left-skewed** — there's a long tail of low-rated films dragging the mean slightly below the median. That's something a mean alone would never tell you. If you were building a recommendation model and assumed ratings were symmetric, your predictions would be biased.

### Box Plot: Center, Spread, and Outliers in One Picture

A box plot gives you five numbers at a glance: the minimum, Q1 (25th percentile), median (50th percentile), Q3 (75th percentile), and maximum. Anything beyond the "whiskers" (typically 1.5 * IQR from the quartiles) is flagged as an outlier.

```
  Movie Ratings Box Plot

         Outliers           Q1   Median   Q3            Outliers
           *     *          |      |      |                *
  |--------+-----+---------[======|======]----------+-----|
  1        2     3          5     6.3     7.5        9    10

                    IQR = Q3 - Q1 = 2.5

  Lower fence = Q1 - 1.5 * IQR = 5 - 3.75 = 1.25
  Upper fence = Q3 + 1.5 * IQR = 7.5 + 3.75 = 11.25

  Outliers: Ratings at 1 and 2 (below lower fence ... barely)
```

The box plot tells you that the middle 50% of movies are rated between 5.0 and 7.5, and the few films rated 1 or 2 are statistical outliers. If you're training a model to predict ratings, those outliers might deserve special treatment — or removal.

---

## Measures of Relationship: How Do Variables Move Together?

So far you've looked at one variable at a time. But ML is about relationships between variables. Does a bigger budget lead to higher revenue? Let's find out.

### Covariance

**Covariance** measures how two variables change together:

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

**Sample covariance:**

$$s_{xy} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

**How to read it:**

- $\text{Cov}(X, Y) > 0$: When X goes up, Y tends to go up
- $\text{Cov}(X, Y) < 0$: When X goes up, Y tends to go down
- $\text{Cov}(X, Y) = 0$: No linear relationship

The problem with covariance is that its magnitude depends on the scale of X and Y. A covariance of 50,000,000 between budget (in dollars) and revenue (in dollars) sounds huge, but it tells you nothing about the strength of the relationship. You need to normalize it.

### Correlation (Pearson's r)

**Correlation** is normalized covariance, bounded between -1 and 1:

$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

**Sample correlation:**

$$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

**Interpretation:**

- $r = 1$: Perfect positive linear relationship
- $r = -1$: Perfect negative linear relationship
- $r = 0$: No linear relationship (but there could still be a strong nonlinear relationship!)

**SWE bridge.** Correlation measures **feature co-movement**, the same way you'd look at correlated service failures. If Service A and Service B always go down together, you suspect a shared dependency (a load balancer, a database). In ML features, if two columns have $r = 0.98$, they're carrying almost the same information — you can drop one without losing predictive power. This is exactly what you check for when diagnosing multicollinearity before fitting a linear model.

**Running example.** You compute the correlation between movie budget and revenue and get $r = 0.73$. That's a strong positive relationship — higher-budget films tend to earn more. But $r = 0.73$ is not $r = 1.0$. Plenty of big-budget films flop, and some indie films become hits. The correlation tells you there's a trend, not a guarantee.

---

### Common Mistake

> **Correlation does not imply causation.** Two features can be perfectly correlated because they both depend on a third variable.

Classic example: ice cream sales and drowning deaths are highly correlated ($r \approx 0.85$). Does ice cream cause drowning? Obviously not. Both are caused by a **confounding variable**: hot weather. Summer heat drives people to buy ice cream *and* to go swimming (where drownings happen).

In your ML pipeline, this matters. If you find that `feature_A` and `feature_B` are highly correlated with your target, don't assume both are useful predictors. They might both be proxies for some unmeasured third variable, and including both could introduce multicollinearity without improving your model.

---

## The Covariance Matrix

When you have multiple features (and you almost always do), you organize all pairwise relationships into a single matrix:

$$\Sigma = \begin{bmatrix} \text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots \\ \text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

This matrix shows up everywhere in ML:

- **PCA**: You eigendecompose the covariance matrix to find the directions of maximum variance.
- **Gaussian Mixture Models**: Each cluster has its own covariance matrix describing its shape.
- **Linear Discriminant Analysis**: Uses between-class and within-class covariance matrices.
- **Mahalanobis distance**: Measures distance accounting for feature correlations via $\Sigma^{-1}$.

**Running example.** For the movie dataset with three numeric columns (rating, budget, revenue), the covariance matrix is 3x3. The diagonal entries are the variances of each column. The off-diagonal entries tell you how each pair co-moves. Budget and revenue have a large positive covariance; rating and budget have a smaller one.

---

## Where Descriptive Statistics Appear in ML

You might think this is "just EDA" and has nothing to do with modeling. You'd be wrong. These concepts are baked into the algorithms themselves.

**1. Feature Scaling (Z-Score Normalization)**

When you standardize features, you're directly using the mean and standard deviation:

$$z = \frac{x - \mu}{\sigma}$$

Gradient-based models (neural networks, logistic regression) converge faster when features are on comparable scales. You literally can't do this without computing $\mu$ and $\sigma$ first.

**2. Batch Normalization**

Inside a neural network, batch norm computes the mean and variance of each layer's activations across the current mini-batch, then normalizes. It's descriptive statistics running on every forward pass.

**3. Feature Selection**

- **Variance threshold**: Remove features with near-zero variance. A feature that's constant carries no information.
- **Correlation filtering**: If two features have $|r| > 0.95$, drop one. They're redundant.

**4. Algorithms That Directly Use These Statistics**

- **Naive Bayes**: Assumes features follow a Gaussian distribution and uses per-class mean and variance to compute likelihoods.
- **K-Means**: Cluster centers are the means of assigned points. The algorithm literally minimizes within-cluster variance.
- **PCA**: Finds the eigenvectors of the covariance matrix.

**5. Model Diagnostics**

- Check that residuals have mean $\approx 0$ and constant variance (homoscedasticity).
- Monitor the mean and variance of training loss across epochs to detect convergence issues.

---

## When to Use Which Measure

| Situation | Best Measure | Why |
|---|---|---|
| Symmetric data, no outliers | Mean | It uses all the data |
| Skewed data or outliers present | Median | Robust to extreme values |
| Categorical data | Mode | Mean/median are meaningless for categories |
| Need interpretable spread (same units as data) | Standard deviation | Variance is in squared units |
| Mathematical convenience / ML algorithms | Variance | Easier to work with algebraically |
| Strength of linear relationship between two variables | Correlation (Pearson's r) | Scale-free, bounded [-1, 1] |
| Raw co-movement (preserving scale) | Covariance | Needed for matrix computations (PCA, etc.) |

---

## Code: Full EDA on the Movie Ratings Dataset

```python
import numpy as np
from scipy import stats

# ============================================================
# SIMULATE THE MOVIE RATINGS DATASET
# ============================================================
np.random.seed(42)
n_movies = 5000

# Ratings: left-skewed distribution (most movies get 5-8)
ratings = np.clip(np.random.normal(loc=6.1, scale=1.4, size=n_movies), 1, 10)

# Budget: log-normal (most films are modest, a few are blockbusters)
budget = np.random.lognormal(mean=17.5, sigma=1.2, size=n_movies)  # in dollars

# Revenue: correlated with budget, but noisy
revenue = 1.5 * budget + np.random.lognormal(mean=17, sigma=1.5, size=n_movies)

# ============================================================
# MEASURES OF CENTER
# ============================================================
print("=" * 55)
print("MEASURES OF CENTER")
print("=" * 55)

mean_rating = np.mean(ratings)
median_rating = np.median(ratings)
mode_result = stats.mode(np.round(ratings).astype(int), keepdims=True)

print(f"Mean rating:    {mean_rating:.2f}")
print(f"Median rating:  {median_rating:.2f}")
print(f"Mode rating:    {mode_result.mode[0]} (appears {mode_result.count[0]} times)")

# ============================================================
# MEASURES OF SPREAD
# ============================================================
print(f"\n{'=' * 55}")
print("MEASURES OF SPREAD")
print("=" * 55)

# Always use ddof=1 for sample variance (Bessel's correction)
var_rating = np.var(ratings, ddof=1)
std_rating = np.std(ratings, ddof=1)

print(f"Sample variance:  {var_rating:.4f}")
print(f"Sample std dev:   {std_rating:.4f}")
print(f"Range:            [{np.min(ratings):.2f}, {np.max(ratings):.2f}]")
print(f"IQR:              {np.percentile(ratings, 75) - np.percentile(ratings, 25):.2f}")

# ============================================================
# MEASURES OF RELATIONSHIP
# ============================================================
print(f"\n{'=' * 55}")
print("MEASURES OF RELATIONSHIP: BUDGET vs. REVENUE")
print("=" * 55)

# Covariance
covariance = np.cov(budget, revenue, ddof=1)[0, 1]
print(f"Covariance (budget, revenue): {covariance:.2e}")

# Correlation
correlation = np.corrcoef(budget, revenue)[0, 1]
print(f"Pearson r (budget, revenue):  {correlation:.3f}")

# scipy gives you the p-value for free
r, p_value = stats.pearsonr(budget, revenue)
print(f"Pearson r: {r:.3f}, p-value: {p_value:.2e}")

# ============================================================
# FULL CORRELATION MATRIX
# ============================================================
print(f"\n{'=' * 55}")
print("CORRELATION MATRIX")
print("=" * 55)

features = np.column_stack([ratings, budget, revenue])
corr_matrix = np.corrcoef(features, rowvar=False)
labels = ["Rating", "Budget", "Revenue"]

print(f"{'':>10}", end="")
for label in labels:
    print(f"{label:>10}", end="")
print()
for i, label in enumerate(labels):
    print(f"{label:>10}", end="")
    for j in range(len(labels)):
        print(f"{corr_matrix[i, j]:>10.3f}", end="")
    print()

# Flag highly correlated pairs
print("\nHighly correlated pairs (|r| > 0.5):")
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        if abs(corr_matrix[i, j]) > 0.5:
            print(f"  {labels[i]} - {labels[j]}: r = {corr_matrix[i, j]:.3f}")

# ============================================================
# FEATURE ANALYSIS SUMMARY
# ============================================================
print(f"\n{'=' * 55}")
print("PER-FEATURE SUMMARY (for ML preprocessing)")
print("=" * 55)

for name, data in [("Rating", ratings), ("Budget", budget), ("Revenue", revenue)]:
    print(f"\n  {name}:")
    print(f"    Mean:     {np.mean(data):.3f}")
    print(f"    Median:   {np.median(data):.3f}")
    print(f"    Std:      {np.std(data, ddof=1):.3f}")
    print(f"    Skewness: {stats.skew(data):.3f}")
    print(f"    Kurtosis: {stats.kurtosis(data):.3f}")
```

**What to look for in the output:**

- If the mean and median are close, the distribution is roughly symmetric. If they diverge, you have skew.
- A skewness near 0 means symmetric. Budget and revenue will be heavily right-skewed (long tail of blockbusters), which tells you to consider a log transform before modeling.
- The correlation matrix instantly shows you which features are redundant and which pairs have predictive relationships.

---

## Exercises

### Exercise 1: Outlier Impact

Given response times (in ms): `[102, 98, 105, 110, 101, 99, 97, 103, 5000]`

One request timed out at 5000ms. Compute the mean and median. Which would you report on a dashboard, and why?

**Solution:**

```python
data = np.array([102, 98, 105, 110, 101, 99, 97, 103, 5000])
print(f"Mean:   {np.mean(data):.2f} ms")    # Mean:   646.11 ms
print(f"Median: {np.median(data):.2f} ms")   # Median: 102.00 ms
```

The mean (646ms) is dragged up by the single 5000ms outlier and misrepresents the typical experience. The median (102ms) is unaffected and accurately reflects what most users experienced. This is exactly why SRE dashboards report p50/p99 rather than averages — a single timeout shouldn't make your service look six times slower than it is.

### Exercise 2: Correlation vs. Causation

Your movie dataset shows that `number_of_theaters` and `revenue` have $r = 0.88$. A colleague says: "If we just release the film in more theaters, revenue will go up proportionally." What's wrong with this reasoning?

**Solution:**

This is correlation-as-causation. The number of theaters a film opens in is not randomly assigned — studios give more theaters to films they expect to perform well (based on star power, marketing budget, franchise recognition, etc.). Those same factors independently drive revenue. The confounding variables (marketing spend, franchise value, star power) cause both wide theatrical release *and* high revenue. Simply dumping a low-budget indie film into 4,000 theaters won't magically give it blockbuster revenue.

### Exercise 3: Manual Covariance Calculation

Given: X = [1, 2, 3, 4, 5], Y = [2, 4, 5, 4, 5]

Calculate the sample covariance step by step, then verify with NumPy.

**Solution:**

```python
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# Step 1: Compute means
mean_x = np.mean(X)  # 3.0
mean_y = np.mean(Y)  # 4.0

# Step 2: Compute deviations from mean
# X - mean_x: [-2, -1, 0, 1, 2]
# Y - mean_y: [-2,  0, 1, 0, 1]

# Step 3: Multiply pairwise and sum
# Products:    [4,   0, 0, 0, 2]  -> Sum = 6

# Step 4: Divide by (n - 1)
cov_manual = 6 / (5 - 1)  # 1.50

# Verify
cov_numpy = np.cov(X, Y, ddof=1)[0, 1]
print(f"Manual:  {cov_manual:.2f}")  # 1.50
print(f"NumPy:   {cov_numpy:.2f}")   # 1.50
```

Positive covariance (1.50) tells you X and Y tend to increase together, though the relationship isn't perfectly linear.

### Exercise 4: When the Mean Lies

You have two ML models. Both report a mean test accuracy of 85%. But Model A has a standard deviation of 1% across cross-validation folds, while Model B has a standard deviation of 12%. Which do you deploy, and why?

**Solution:**

Deploy Model A. Both models have the same average accuracy, but Model A is consistent (std = 1% means it performs between roughly 83-87% on every fold), while Model B is wildly unstable (std = 12% means some folds hit 97% and others hit 73%). Model B might catastrophically fail on certain data distributions. The mean alone hid this — you needed the standard deviation to see it.

---

## Summary

Here's what you now have in your toolkit:

| Concept | What It Tells You | Formula |
|---|---|---|
| **Mean** | Center of the data; sensitive to outliers | $\bar{x} = \frac{1}{n}\sum x_i$ |
| **Median** | Robust center; the p50 | Middle value when sorted |
| **Mode** | Most common value; best for categorical data | Most frequent $x_i$ |
| **Variance** | Average squared spread; units are squared | $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ |
| **Standard deviation** | Spread in original units; the "jitter" | $s = \sqrt{s^2}$ |
| **Covariance** | Direction of co-movement; scale-dependent | $s_{xy} = \frac{1}{n-1}\sum(x_i-\bar{x})(y_i-\bar{y})$ |
| **Correlation** | Strength of linear relationship; [-1, 1] | $r = \frac{s_{xy}}{s_x \cdot s_y}$ |

The key lessons:

- **Always visualize.** A histogram and a box plot will catch things that summary statistics hide (bimodality, skew, outliers).
- **Mean vs. median.** If they disagree, your data is skewed. Use the median for reporting and consider log transforms for modeling.
- **Correlation is not causation.** Two features can be perfectly correlated because they both depend on a third variable.
- **These statistics are not just EDA — they're embedded in the algorithms.** Z-score normalization, batch norm, PCA, Naive Bayes, K-Means — they all compute means, variances, and covariances under the hood.

---

Descriptive statistics summarizes what you have. But your data is a sample from a larger population. How do you generalize from sample to population? That's sampling theory.
