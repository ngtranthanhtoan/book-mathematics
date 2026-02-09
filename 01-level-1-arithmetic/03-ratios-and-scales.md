# Ratios and Scales

> **Building On** — You understand numbers and operations. Now: comparing and scaling numbers. In ML, this isn't academic — feature scaling directly determines whether your model converges, how fast it trains, and whether features are treated fairly.

---

Your features range from 0.001 (gene expression) to 1,000,000 (annual salary). If you feed these raw into a neural network, the salary feature will dominate everything — gradient descent will mostly optimize for salary and ignore gene expression. Feature scaling isn't optional; it's the difference between a model that trains and one that doesn't.

This chapter covers the math behind ratios, proportions, percentages, growth rates, and normalization techniques. Every one of these shows up in your ML pipeline, often in places you don't expect.

---

## Why Won't My Model Converge?

Let's make this concrete. You're building a movie recommendation model with three features:

| Feature | Range | Example Values |
|---------|-------|----------------|
| Budget | $10M -- $300M | 10,000,000 ... 300,000,000 |
| Runtime | 80 -- 200 min | 80 ... 200 |
| Rating Count | 100 -- 1M | 100 ... 1,000,000 |

Here's what the raw feature space looks like:

```
BEFORE SCALING — Raw Feature Magnitudes
(each * = ~10M in budget scale)

Budget ($):    |******************************.| 10M ─────────────── 300M
Runtime (min): |.                              | 80 ── 200
Rating Count:  |**********.                    | 100 ────────── 1,000,000

Problem: Budget values are 1,000,000x larger than runtime values.
         Gradient descent sees a wildly elongated loss surface.
         The model zigzags instead of descending smoothly.
```

Without scaling, budget dominates. The loss surface is a long, narrow valley. Gradient descent bounces off the walls instead of rolling down the center. Your learning rate is either too big for the budget dimension (overshooting) or too small for runtime (barely moving). Training takes forever or diverges entirely.

```
AFTER MIN-MAX SCALING — All Features in [0, 1]

Budget (scaled):       |==============================| 0.0 ──── 1.0
Runtime (scaled):      |==============================| 0.0 ──── 1.0
Rating Count (scaled): |==============================| 0.0 ──── 1.0

Result: All features contribute equally to gradient updates.
        The loss surface is roughly circular.
        Gradient descent converges smoothly.
```

Now every feature lives on the same playing field. The loss surface is approximately circular, and gradient descent heads straight for the minimum.

---

## Ratios — You Already Think in Ratios

> **You Already Know This**: Conversion rates, click-through rates, precision/recall — you already work in ratios every day. A ratio is just one quantity compared to another.

A **ratio** compares two quantities $a$ and $b$:

$$a:b = \frac{a}{b}$$

**Properties:**
- Ratios can be simplified: $6:4 = 3:2$
- Equivalent ratios: $\frac{a}{b} = \frac{ka}{kb}$ for any $k \neq 0$

**Part-to-whole vs Part-to-part:**
- If red:blue = 3:2, then red:total = 3:5 and blue:total = 2:5

### Ratios in Your ML Work

You use ratios constantly:

| Ratio | What It Means |
|-------|---------------|
| Train/test split (80:20) | 4 training samples for every 1 test sample |
| Class imbalance (100:1) | 100 negatives for every positive — you need class weights |
| Precision: $\frac{TP}{TP + FP}$ | Of everything you predicted positive, how many were right? |
| Recall: $\frac{TP}{TP + FN}$ | Of everything actually positive, how many did you find? |
| F1 Score: $\frac{2 \cdot P \cdot R}{P + R}$ | Harmonic mean of precision and recall |

```python
# Ratios you already compute
TP, FP, FN, TN = 80, 20, 10, 90

precision = TP / (TP + FP)   # 0.80 — "of predicted positives, how many correct?"
recall    = TP / (TP + FN)   # 0.89 — "of actual positives, how many found?"
f1        = 2 * precision * recall / (precision + recall)  # harmonic mean
accuracy  = (TP + TN) / (TP + FP + FN + TN)

print(f"Precision: {precision:.2%}")  # 80.00%
print(f"Recall:    {recall:.2%}")     # 88.89%
print(f"F1 Score:  {f1:.2%}")         # 84.21%
print(f"Accuracy:  {accuracy:.2%}")   # 85.00%
```

### Proportions

A **proportion** states that two ratios are equal:

$$\frac{a}{b} = \frac{c}{d}$$

**Cross-multiplication property:**
$$a \cdot d = b \cdot c$$

This is useful for solving unknown values:
If $\frac{x}{12} = \frac{3}{4}$, then $4x = 36$, so $x = 9$.

In ML, proportions show up in softmax — converting raw logits into probabilities that sum to 1:

```python
import numpy as np

def softmax(x):
    """Convert raw scores to probabilities (normalized ratios)."""
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / exp_x.sum()

logits = np.array([2.0, 1.0, 0.5])
probs = softmax(logits)
print(f"Logits: {logits}")
print(f"Probabilities: {probs.round(4)}")  # [0.5761, 0.2119, 0.212]
print(f"Sum: {probs.sum():.4f}")           # 1.0000
```

---

## Percentages and Growth Rates

A **percentage** is a ratio per 100:

$$p\% = \frac{p}{100}$$

**Key conversions:**
- Decimal to percentage: multiply by 100 ($0.75 \rightarrow 75\%$)
- Percentage to decimal: divide by 100 ($75\% \rightarrow 0.75$)
- Fraction to percentage: $\frac{a}{b} \times 100\%$

**Percentage change:**
$$\text{Change} = \frac{\text{New} - \text{Old}}{\text{Old}} \times 100\%$$

### Percentage vs Percentage Points — Get This Right

This trips up data scientists in reports all the time:

```
Your model's accuracy:
  Before: 90%
  After:  95%

CORRECT ways to describe the improvement:
  - "5 percentage point improvement"        (95% - 90% = 5pp)
  - "5.6% relative improvement"             ((95-90)/90 = 5.6%)

WRONG:
  - "5% improvement"  <-- ambiguous and misleading
```

### Growth Rates

**Simple growth rate:**
$$r = \frac{V_{\text{final}} - V_{\text{initial}}}{V_{\text{initial}}}$$

**Compound growth** over $n$ periods at rate $r$:
$$V_n = V_0 (1 + r)^n$$

**Continuous growth** at rate $r$ over time $t$:
$$V(t) = V_0 e^{rt}$$

**Compound Annual Growth Rate (CAGR):**
$$\text{CAGR} = \left(\frac{V_{\text{final}}}{V_{\text{initial}}}\right)^{1/n} - 1$$

Growth rates appear in ML as learning rate decay, loss curves, and user growth projections:

```python
import numpy as np

# Learning rate decay — compound shrinkage each epoch
initial_lr = 0.01
decay_rate = 0.1
epochs = 10

lr = initial_lr
print("Learning rate decay (10% per epoch):")
for epoch in range(epochs):
    print(f"  Epoch {epoch}: lr = {lr:.6f}")
    lr *= (1 - decay_rate)

# CAGR from time series
start_revenue = 100_000
end_revenue = 200_000
num_years = 3
cagr = (end_revenue / start_revenue) ** (1 / num_years) - 1
print(f"\nCAGR from ${start_revenue:,} to ${end_revenue:,} over {num_years} years: {cagr:.2%}")
```

---

## Normalization Techniques — The Core of Feature Scaling

This is where ratios and scales become directly load-bearing in your ML pipeline. Every technique below is a specific way of re-scaling your features so that no single feature dominates training.

### Min-Max Normalization (scales to $[0, 1]$)

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

> **You Already Know This**: You map pixel values from [0, 255] to [0.0, 1.0] for image models. Min-max scaling does the same thing for any feature range — it linearly maps [min, max] to [0, 1].

**Running example — Movie features:**

```python
import numpy as np

# Movie features: [budget ($), runtime (min), rating_count]
movies = np.array([
    [10_000_000,  120,    500],    # Low-budget indie
    [50_000_000,  95,   15_000],   # Mid-range
    [150_000_000, 180,  200_000],  # Blockbuster
    [300_000_000, 140, 1_000_000], # Mega-blockbuster
    [25_000_000,  88,    2_000],   # Small film
])

def min_max_scale(x):
    """Scale each feature to [0, 1]."""
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

scaled = min_max_scale(movies.astype(float))

print("Original data:")
print(movies)
print(f"\nOriginal ranges: {movies.min(axis=0)} to {movies.max(axis=0)}")
print(f"\nMin-Max scaled (each feature now in [0, 1]):")
print(np.round(scaled, 4))
```

Min-max scaling is sensitive to outliers — one extreme value stretches the entire range.

### Z-Score Standardization (mean=0, std=1)

$$z = \frac{x - \mu}{\sigma}$$

> **You Already Know This**: "How many standard deviations from the mean?" — it's the same z-score from a z-test or statistical process control chart. You center the data at zero and measure everything in units of spread.

```python
def z_score_scale(x):
    """Standardize to mean=0, std=1."""
    return (x - x.mean(axis=0)) / x.std(axis=0)

standardized = z_score_scale(movies.astype(float))
print("Z-Score standardized:")
print(np.round(standardized, 3))
print(f"  Means: {standardized.mean(axis=0).round(10)}")   # ~0
print(f"  Stds:  {standardized.std(axis=0).round(3)}")      # ~1
```

Z-score is the default choice for most neural networks. It handles unbounded features well and is less sensitive to outliers than min-max.

### Max-Abs Scaling (scales to $[-1, 1]$)

$$x' = \frac{x}{\max(|x|)}$$

Useful for sparse data because it preserves zeros — if a feature value is 0, it stays 0 after scaling.

### Robust Scaling (uses median and IQR)

$$x' = \frac{x - \text{median}}{\text{IQR}}$$

When your data has outliers, robust scaling ignores the extremes and focuses on the middle 50% of values.

```python
def robust_scale(x):
    """Scale using median and IQR — robust to outliers."""
    median = np.median(x, axis=0)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr == 0, 1, iqr)  # handle zero IQR
    return (x - median) / iqr

# Demonstrate robustness to outliers
data_with_outlier = np.array([[1], [2], [3], [4], [5], [100]])
print(f"Data: {data_with_outlier.flatten()}")
print(f"Z-Score: {z_score_scale(data_with_outlier).flatten().round(2)}")
print(f"Robust:  {robust_scale(data_with_outlier).flatten().round(2)}")
# Robust scaling keeps the main cluster tightly grouped
```

### L2 Normalization (unit vector)

$$\hat{x} = \frac{x}{\|x\|_2} = \frac{x}{\sqrt{\sum_i x_i^2}}$$

This normalizes each sample (row) to have unit length. Used in text embeddings, cosine similarity, and anywhere direction matters more than magnitude.

```python
def l2_normalize(x):
    """Normalize each row to unit length."""
    norms = np.sqrt((x ** 2).sum(axis=1, keepdims=True))
    return x / norms

normalized_l2 = l2_normalize(movies.astype(float))
print("L2 normalized (unit vectors):")
print(np.round(normalized_l2, 6))
print(f"Row norms: {np.sqrt((normalized_l2**2).sum(axis=1)).round(3)}")  # all 1.0
```

### Logarithmic Scale

> **You Already Know This**: You use log scales on monitoring dashboards when values span orders of magnitude — request latencies from 1ms to 10s, for instance. Log transforms compress wide ranges.

Log transforms are not a "scaler" in the sklearn sense, but they're essential preprocessing for features with exponential distributions (income, population, page views):

$$x' = \log(x + 1)$$

The "+1" handles zeros. After log-transforming, you often follow up with z-score standardization.

```python
# Rating counts span 100 to 1,000,000 — log compresses this
rating_counts = np.array([100, 500, 2000, 15000, 200000, 1000000])
log_counts = np.log1p(rating_counts)  # log(x + 1)

print("Rating counts — raw vs log-transformed:")
for raw, logged in zip(rating_counts, log_counts):
    bar = "#" * int(logged * 2)
    print(f"  {raw:>10,} -> {logged:6.2f} {bar}")
```

---

## Using scikit-learn — The Production Way

You'll implement these from scratch once to understand them, then use sklearn forever after:

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# Movie features: [budget, runtime, rating_count]
movies = np.array([
    [10_000_000,  120,    500],
    [50_000_000,  95,   15_000],
    [150_000_000, 180,  200_000],
    [300_000_000, 140, 1_000_000],
    [25_000_000,  88,    2_000],
], dtype=float)

# Fit on training data, transform both train and test
scaler = StandardScaler()
movies_scaled = scaler.fit_transform(movies)

print("Standardized movie features:")
print(np.round(movies_scaled, 3))
print(f"Means: {movies_scaled.mean(axis=0).round(10)}")
print(f"Stds:  {movies_scaled.std(axis=0).round(3)}")
```

---

## When to Use Which Technique

| Data Characteristic | Recommended Technique | Why |
|--------------------|----------------------|-----|
| Bounded features (e.g., pixel values) | Min-Max (0 to 1) | Maps to a fixed, interpretable range |
| Gaussian-like distribution | Z-Score | Centers data, works well with most algorithms |
| Many outliers | Robust scaling (median/IQR) | Outliers don't distort the scale |
| Sparse data (many zeros) | MaxAbs | Preserves sparsity (zeros stay zero) |
| Comparing sample directions | L2 normalization | Makes magnitude irrelevant |
| Values spanning orders of magnitude | Log transform + Z-Score | Compresses exponential spread |
| Neural networks | Z-Score or Min-Max | Keeps gradients balanced |
| Tree-based models (RF, XGBoost) | Often not needed | Splits are scale-invariant |

---

## Common Mistakes

### Mistake 1: Fitting the Scaler on Test Data (Data Leakage)

Always fit your scaler on TRAINING data only, then transform test data with the same parameters. Otherwise you leak information from the test set.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# CORRECT: fit on train, transform both
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # uses train's mean and std

# WRONG: fitting on test data = data leakage
X_test_scaled = scaler.fit_transform(X_test)  # NO! This uses test statistics
```

Why does this matter? If you fit on test data, your scaler "knows" the test distribution. Your model's performance estimate becomes optimistically biased — it looks better in evaluation than it performs in production.

### Mistake 2: Confusing Percentage and Percentage Points

```python
old_rate = 0.20  # 20%
new_rate = 0.25  # 25%

pp_change  = (new_rate - old_rate) * 100        # 5 percentage points
pct_change = ((new_rate - old_rate) / old_rate) * 100  # 25% relative increase

# "Increased by 5 percentage points" -- correct
# "Increased by 25% (relative)" -- correct
# "Increased by 5%" -- WRONG (ambiguous, usually misinterpreted)
```

### Mistake 3: Not Handling Division by Zero

If a feature has zero variance (constant), z-score divides by zero:

```python
std = x.std()
if std == 0:
    return np.zeros_like(x)  # constant feature — no information
return (x - x.mean()) / std
```

### Mistake 4: Forgetting to Normalize at Inference Time

Your scaler is part of your model. Save it and apply it in production:

```python
import joblib

# At training time: save the scaler alongside the model
joblib.dump(scaler, 'scaler.pkl')

# At inference time: load and apply
scaler = joblib.load('scaler.pkl')
X_new_scaled = scaler.transform(X_new)
```

### Mistake 5: Normalizing the Target Variable Incorrectly

For regression, you can normalize the target — but you must denormalize predictions before reporting them. For classification, never normalize class labels.

---

## Putting It All Together — Full Pipeline with Movie Data

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ====================================================
# Our running example: movie recommendation features
# ====================================================

np.random.seed(42)

# Simulate a dataset: budget, runtime, rating_count
n_movies = 100
budgets      = np.random.uniform(10e6, 300e6, n_movies)
runtimes     = np.random.uniform(80, 200, n_movies)
rating_counts = np.random.uniform(100, 1e6, n_movies)

movies = np.column_stack([budgets, runtimes, rating_counts])

# Split into train/test (80:20 ratio)
split = int(0.8 * n_movies)
X_train, X_test = movies[:split], movies[split:]

print("=== Before Scaling ===")
print(f"Train feature ranges:")
print(f"  Budget:      {X_train[:,0].min():>15,.0f} to {X_train[:,0].max():>15,.0f}")
print(f"  Runtime:     {X_train[:,1].min():>15.0f} to {X_train[:,1].max():>15.0f}")
print(f"  Rating Count:{X_train[:,2].min():>15,.0f} to {X_train[:,2].max():>15,.0f}")

# Standardize: fit on train only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # same parameters

print("\n=== After Z-Score Scaling ===")
print(f"Train means:  {X_train_scaled.mean(axis=0).round(10)}")  # ~0
print(f"Train stds:   {X_train_scaled.std(axis=0).round(3)}")     # ~1
print(f"Test means:   {X_test_scaled.mean(axis=0).round(3)}")     # close to 0
print(f"Test stds:    {X_test_scaled.std(axis=0).round(3)}")      # close to 1

# Demonstrate gradient balance
grad_unscaled = X_train.mean(axis=0)
grad_scaled   = X_train_scaled.mean(axis=0)

print("\n=== Why Scaling Matters for Gradients ===")
print(f"Mean gradient magnitudes (unscaled): {np.abs(grad_unscaled).round(1)}")
print(f"Mean gradient magnitudes (scaled):   {np.abs(grad_scaled).round(6)}")
print("Scaled gradients are balanced — no feature dominates optimization.")
```

---

## Class Weights — Ratios in Action

When your dataset has class imbalance (e.g., 900 negatives, 100 positives), you use inverse frequency ratios to weight your loss function:

```python
import numpy as np

def compute_class_weights(y):
    """Compute inverse frequency weights to handle class imbalance."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {cls: total / (len(classes) * count) for cls, count in zip(classes, counts)}
    return weights

y_imbalanced = np.array([0]*900 + [1]*100)
weights = compute_class_weights(y_imbalanced)
print(f"Class weights for 900:100 imbalance:")
print(f"  Class 0 (900 samples): weight = {weights[0]:.3f}")
print(f"  Class 1 (100 samples): weight = {weights[1]:.3f}")
# Class 1 gets 9x the weight — the loss penalizes missing a positive 9x more
```

---

## Image Normalization — A Special Case You'll See Everywhere

```python
import numpy as np

def normalize_imagenet(images):
    """Normalize images using ImageNet mean and std.

    Args:
        images: Array of shape (N, H, W, C) with values in [0, 255]

    Returns:
        Normalized images with approximate mean=0, std=1
    """
    # ImageNet channel statistics (RGB)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    # Step 1: Min-max to [0, 1]
    images = images / 255.0

    # Step 2: Standardize per channel
    normalized = (images - mean) / std

    return normalized

# Demo
batch = np.random.randint(0, 256, (4, 224, 224, 3))
print(f"Original — range: [{batch.min()}, {batch.max()}]")
normed = normalize_imagenet(batch.astype(np.float32))
print(f"Normalized — range: [{normed.min():.2f}, {normed.max():.2f}]")
print(f"Per-channel means: {normed.mean(axis=(0,1,2)).round(3)}")
```

---

## Exercises

### Exercise 1: Implement Robust Scaling

**Problem**: Implement robust scaling using median and interquartile range (IQR). Show that it handles outliers better than z-score.

**Solution**:
```python
import numpy as np

def robust_scale(x):
    """Scale using median and IQR (robust to outliers)."""
    median = np.median(x, axis=0)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr == 0, 1, iqr)
    return (x - median) / iqr

# Test: the outlier (100) should not distort the main cluster
data = np.array([[1], [2], [3], [4], [5], [100]])
print(f"Original: {data.flatten()}")
print(f"Z-Score:  {((data - data.mean()) / data.std()).flatten().round(2)}")
print(f"Robust:   {robust_scale(data).flatten().round(2)}")
# With z-score, the outlier compresses the main cluster.
# With robust scaling, values 1-5 stay spread out; 100 is just "very far."
```

### Exercise 2: Compute CAGR from Time Series

**Problem**: Given monthly revenue data, compute the compound annual growth rate.

**Solution**:
```python
import numpy as np

def compute_cagr(values, periods_per_year=12):
    """Compute CAGR from a time series."""
    start_value = values[0]
    end_value = values[-1]
    num_periods = len(values) - 1
    num_years = num_periods / periods_per_year

    cagr = (end_value / start_value) ** (1 / num_years) - 1
    return cagr

# Monthly revenue over 2 years with ~2% monthly growth + noise
np.random.seed(42)
months = 24
revenue = [100_000]
for _ in range(months - 1):
    revenue.append(revenue[-1] * 1.02 * (0.95 + 0.1 * np.random.random()))

revenue = np.array(revenue)
cagr = compute_cagr(revenue)

print(f"Starting revenue: ${revenue[0]:,.0f}")
print(f"Ending revenue:   ${revenue[-1]:,.0f}")
print(f"CAGR: {cagr:.2%}")
print(f"Expected ~26.8% (1.02^12 - 1)")
```

### Exercise 3: Normalize Movie Features End-to-End

**Problem**: Take the movie dataset, apply log transform to rating_count (which spans orders of magnitude), then standardize all features. Verify that the scaler fitted on training data works on test data.

**Solution**:
```python
import numpy as np
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Generate movie data
n = 200
budgets       = np.random.uniform(10e6, 300e6, n)
runtimes      = np.random.uniform(80, 200, n)
rating_counts = np.random.exponential(50000, n).clip(100, 1e6)

movies = np.column_stack([budgets, runtimes, rating_counts])

# Log-transform the rating_count column (index 2)
movies[:, 2] = np.log1p(movies[:, 2])

# Train/test split (80:20)
split = int(0.8 * n)
X_train, X_test = movies[:split], movies[split:]

# Fit scaler on train, transform both
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("Train — means:", X_train_s.mean(axis=0).round(6))
print("Train — stds: ", X_train_s.std(axis=0).round(6))
print("Test  — means:", X_test_s.mean(axis=0).round(3))
print("Test  — stds: ", X_test_s.std(axis=0).round(3))
print("\nAll features are now on comparable scales.")
print("No data leakage: test was transformed using train statistics only.")
```

---

## Summary

| Concept | Formula / Key Idea | ML Application |
|---------|-------------------|----------------|
| **Ratios** | $a:b = \frac{a}{b}$ | Train/test splits, class imbalance, precision/recall |
| **Proportions** | $\frac{a}{b} = \frac{c}{d}$ | Softmax outputs, probability calibration |
| **Percentages** | $p\% = \frac{p}{100}$ | Accuracy, dropout rate — distinguish from percentage points |
| **Growth rates** | Simple: $\frac{V_f - V_i}{V_i}$, Compound: $V_0(1+r)^n$, CAGR: $(V_f/V_i)^{1/n}-1$ | Learning rate decay, loss curves |
| **Min-Max** | $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | Bounded features, pixel values |
| **Z-Score** | $z = \frac{x - \mu}{\sigma}$ | Default for neural networks |
| **Max-Abs** | $x' = \frac{x}{\max(\|x\|)}$ | Sparse data |
| **Robust** | $x' = \frac{x - \text{median}}{\text{IQR}}$ | Data with outliers |
| **L2 Norm** | $\hat{x} = \frac{x}{\|x\|_2}$ | Text embeddings, cosine similarity |
| **Log Transform** | $x' = \log(x+1)$ | Features spanning orders of magnitude |

**Key rules:**
- Fit scalers on training data only. Transform test data with the same parameters. Otherwise you leak test information.
- Save your scaler alongside your model. It's part of the inference pipeline.
- Tree-based models (Random Forest, XGBoost) usually don't need scaling. Neural networks almost always do.
- When features span many orders of magnitude, log-transform first, then standardize.

---

> **What's Next** — Numbers, operations, and scaling are your arithmetic toolkit. Now Level 2: algebra — using symbols to represent unknown quantities and expressing relationships as equations.

---

*Proceed to [Level 2: Algebra](../02-level-2-algebra/README.md) to continue your journey.*
