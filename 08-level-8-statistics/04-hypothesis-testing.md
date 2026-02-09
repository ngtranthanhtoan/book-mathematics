# Hypothesis Testing

## From Estimation to Decisions

Estimation gives you numbers. Hypothesis testing helps you make decisions: is this effect real, or could it be chance?

Your A/B test shows the new recommendation algorithm has a 2.3% higher click-through rate. Is that real, or just noise? Hypothesis testing gives you a rigorous framework for answering "is this difference significant?" -- the question you face in every experiment and model comparison.

If you have ever stared at two metrics and wondered "should I ship this?" -- that is exactly the problem hypothesis testing solves. It takes your gut feeling of "this looks better, maybe" and replaces it with a principled, repeatable decision process.

## The Core Idea: Assume Nothing Changed, Then Check

Here is the logic in one sentence: **assume nothing changed, then ask how surprising your data is under that assumption.**

Think of it like a smoke detector. You set a threshold for how sensitive you want it. If the signal exceeds that threshold, you sound the alarm. Hypothesis testing works the same way -- you set a sensitivity level, and if your data exceeds it, you conclude that something real happened.

### The Setup

You need four things:

1. **Null hypothesis ($H_0$)** -- "nothing changed." In SWE terms, this is the current production system. It is your default assumption: the new code does not help, the new algorithm is no better, the feature has no effect.

2. **Alternative hypothesis ($H_1$)** -- "something did change." This is the claim you are trying to support with evidence: the new algorithm IS better.

3. **Test statistic** -- a single number that measures how far your observed data is from what you would expect under $H_0$.

4. **Significance level ($\alpha$)** -- your tolerance for false alarms. Think of it like setting an alert threshold in your monitoring system. Typical value: 0.05 (you are willing to accept a 5% false-alarm rate).

### Running Example: A/B Test for a Recommendation Algorithm

You will carry this example through the entire chapter:

```
Scenario: Old vs. new recommendation algorithm
- Control group (old algorithm): n = 10,000 users
- Treatment group (new algorithm): n = 10,000 users
- Control click-through rate (CTR): 4.5%
- Treatment CTR: 6.8%  (a 2.3 percentage-point lift)

H₀: There is no difference in CTR. (μ_treatment - μ_control = 0)
H₁: The new algorithm has higher CTR. (μ_treatment - μ_control > 0)

Question: Should you ship the new algorithm?
```

## Null and Alternative Hypotheses

### Formal Definition

The **null hypothesis** $H_0$ is the default claim -- typically "no effect" or "no difference":

- $H_0: \mu_{\text{treatment}} - \mu_{\text{control}} = 0$  (no difference between groups)
- $H_0: \mu = \mu_0$  (population mean equals a specific value)
- $H_0: \rho = 0$  (no correlation)

The **alternative hypothesis** $H_1$ is what you want to demonstrate:

- **One-tailed (directional)**: $H_1: \mu_{\text{treatment}} > \mu_{\text{control}}$ -- "new is better"
- **Two-tailed (non-directional)**: $H_1: \mu_{\text{treatment}} \neq \mu_{\text{control}}$ -- "they are different"

### SWE Bridge: One-Tailed vs. Two-Tailed

Use a **one-tailed** test when you only care about improvement in one direction. In the recommendation algorithm example, you only want to ship if the new algorithm is *better*, not just *different*. So one-tailed makes sense.

Use a **two-tailed** test when a change in either direction matters. For example, if you are checking whether a code refactor changed latency at all (faster or slower), you want two tails.

### The Logic Flow

```
Step 1: State H₀ and H₁
           │
           v
Step 2: Collect data (run your experiment)
           │
           v
Step 3: Calculate a test statistic
        "How far is my result from what H₀ predicts?"
           │
           v
Step 4: Compute the p-value
        "If H₀ were true, how often would I see a result this extreme?"
           │
           v
Step 5: Compare p-value to α
           │
     ┌─────┴─────┐
     v           v
  p < α       p >= α
  Reject H₀   Fail to reject H₀
  (ship it)   (keep current system)
```

## P-Value: The Core Metric

### What It Is

The **p-value** is the probability of seeing your result (or something more extreme) IF nothing changed. Think of it as a false-alarm rate -- how often would random chance alone produce data this impressive?

$$p\text{-value} = P(\text{test statistic} \geq \text{observed value} \mid H_0 \text{ is true})$$

### P-Value as a Tail Area

Picture the distribution of your test statistic under $H_0$. The p-value is the area in the tail beyond your observed value:

```
  Distribution of test statistic under H₀
  (what you'd see if nothing changed)

                        |
                       /|\
                      / | \
                     /  |  \
                    /   |   \
                   /    |    \
                  /     |     \
                 /      |      \
                /       |       \
  ────────────/────────-+────────\──────────────────
             /          |     ▓▓▓▓\▓▓▓▓▓▓▓▓
            /           |     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                       μ₀          ↑
                   (expected       Your observed
                    if H₀ true)    test statistic

                              ◄──────────────►
                               p-value = this
                                shaded area

  Small p-value = your result is way out in the tail
                = unlikely under H₀
                = evidence against H₀
```

For a two-tailed test, you shade both tails:

```
  Two-tailed p-value

           |
          /|\
         / | \
        /  |  \
       /   |   \
  ▓▓▓/    |    \▓▓▓
  ▓▓/     |     \▓▓▓
  ────────-+──────────
     ↑    μ₀     ↑
   -|t|         +|t|

  p-value = left shaded area + right shaded area
```

### Back to the Running Example

For the recommendation algorithm A/B test:

```
Control CTR: p̂_c = 0.045    (450 clicks out of 10,000)
Treatment CTR: p̂_t = 0.068  (680 clicks out of 10,000)
Difference: 0.023

Under H₀, the expected difference is 0.
Our observed difference is 0.023.

Standard error of the difference:
  SE = sqrt(p̂(1-p̂) * (1/n₁ + 1/n₂))
  where p̂ = (450 + 680) / 20,000 = 0.0565

  SE = sqrt(0.0565 * 0.9435 * (1/10000 + 1/10000))
     = sqrt(0.0565 * 0.9435 * 0.0002)
     = sqrt(0.00001066)
     ≈ 0.00326

Test statistic (z):
  z = (p̂_t - p̂_c) / SE
    = 0.023 / 0.00326
    ≈ 7.06

p-value ≈ 0.0000000000008  (essentially zero)
```

A z-score of 7.06 is extremely far in the tail. The p-value is astronomically small. If there were truly no difference, you would essentially never see a result this extreme by chance. You reject $H_0$ and ship the new algorithm.

## The Significance Level ($\alpha$): Your False-Alarm Tolerance

### What It Is

The significance level $\alpha$ is the threshold you set *before* looking at the data. It is your tolerance for false alarms -- how often you are willing to incorrectly declare "something changed!" when nothing actually did.

$$\alpha = P(\text{reject } H_0 \mid H_0 \text{ is true})$$

This is exactly like setting an alert threshold in your monitoring system:

- Set it too low ($\alpha = 0.001$): you rarely get false alarms, but you miss real problems
- Set it too high ($\alpha = 0.10$): you catch more real problems, but you get too many false alarms
- Common default ($\alpha = 0.05$): a reasonable balance for most situations

### The Decision Rule

- If $p < \alpha$: **Reject $H_0$** -- the evidence is strong enough to conclude something changed
- If $p \geq \alpha$: **Fail to reject $H_0$** -- you do not have enough evidence (this is NOT the same as proving nothing changed)

## Type I and Type II Errors

Every decision has two ways to be wrong. This is the error matrix you need to internalize:

```
                         REALITY
                  ┌──────────────┬──────────────┐
                  │   H₀ True    │   H₀ False   │
                  │  (no effect) │(effect exists)│
   ┌──────────────┼──────────────┼──────────────┤
   │  Reject H₀   │  TYPE I      │  CORRECT     │
   │  ("ship it") │  ERROR (α)   │  DECISION    │
 D │              │              │  (Power)     │
 E │              │  False       │  True        │
 C │              │  Positive    │  Positive    │
 I ├──────────────┼──────────────┼──────────────┤
 S │  Fail to     │  CORRECT     │  TYPE II     │
 I │  reject H₀   │  DECISION    │  ERROR (β)   │
 O │  ("don't     │              │              │
 N │   ship")     │  True        │  False       │
   │              │  Negative    │  Negative    │
   └──────────────┴──────────────┴──────────────┘

α = P(Type I Error) = P(Reject H₀ | H₀ True)      [significance level]
β = P(Type II Error) = P(Fail to reject H₀ | H₀ False)
Power = 1 - β = P(Reject H₀ | H₀ False)           [detecting a real effect]
```

### SWE Bridge: Type I and Type II Errors

| Statistical Term | SWE Translation | In the A/B Test Example |
|---|---|---|
| Type I Error (false positive) | Deploying a change that does not actually help | You ship the new algorithm, but it is not really better. You waste engineering resources and possibly hurt the user experience. |
| Type II Error (false negative) | Rejecting a change that would have helped | You kill the new algorithm even though it truly is better. You leave value on the table. |
| Significance level ($\alpha$) | Your tolerance for false alarms (like setting an alert threshold) | Setting $\alpha = 0.05$ means you accept a 5% chance of shipping something that does not help. |
| Power ($1 - \beta$) | Your ability to detect real improvements | With power = 0.80, you have an 80% chance of detecting a genuinely better algorithm. |

### The Tradeoff

You cannot minimize both errors simultaneously (with a fixed sample size). Decreasing $\alpha$ (fewer false positives) increases $\beta$ (more false negatives), and vice versa. It is the same tradeoff you face with precision vs. recall in classification.

The way out? **Increase your sample size.** More data lets you reduce both errors.

## Common Mistake: What P-Values Do NOT Mean

> **p < 0.05 does NOT mean there is a 95% chance the effect is real. It means there is a 5% chance you would see this result if there were NO effect.**

Read that again. This is the single most common misinterpretation in all of statistics.

Here is what p-values do NOT tell you:

| Wrong Interpretation | Why It Is Wrong |
|---|---|
| "P = 0.03 means there's a 3% chance $H_0$ is true." | The p-value says nothing about the probability of $H_0$. It conditions ON $H_0$ being true. |
| "P = 0.03 means there's a 97% chance the treatment works." | That is not how conditional probability works. You would need Bayes' theorem for that. |
| "P = 0.03 means the effect size is large." | P-values mix up effect size and sample size. With n = 1,000,000, even a tiny, meaningless difference can give p < 0.001. |

**Correct interpretation**: "If there were truly no difference between the old and new algorithm, we would see data this extreme only 3% of the time."

## Test Statistics: The Formulas

### Z-Test (Known Variance or Large Sample)

When you know the population variance or have a large sample (n > 30):

$$z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$

This is the most common test in A/B testing because you typically have thousands of observations.

### One-Sample T-Test (Unknown Variance, Small Sample)

When the population variance is unknown and your sample is small:

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

where $s$ is the sample standard deviation and the test statistic follows a $t$-distribution with $n-1$ degrees of freedom.

### Two-Sample T-Test (Comparing Two Groups)

For comparing the means of two independent groups (the bread and butter of A/B testing):

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

### Two-Proportion Z-Test (Comparing Rates)

For comparing click-through rates, conversion rates, or any binary outcome -- this is what you use in the running example:

$$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}$$

where $\hat{p} = \frac{x_1 + x_2}{n_1 + n_2}$ is the pooled proportion.

### Chi-Squared Test (Categorical Data)

For testing independence between categorical variables:

$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

where $O_i$ are observed frequencies and $E_i$ are expected frequencies under $H_0$.

## Confidence Intervals: The Dual View

Confidence intervals and hypothesis tests are two sides of the same coin.

A **$(1-\alpha)$ confidence interval** gives you a range of plausible values for the true parameter:

For a mean with known variance:
$$CI = \bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

For a mean with unknown variance (t-distribution):
$$CI = \bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

**The connection**: If a 95% confidence interval for $\mu_{\text{treatment}} - \mu_{\text{control}}$ does not contain 0, then you would reject $H_0: \mu_{\text{treatment}} = \mu_{\text{control}}$ at $\alpha = 0.05$.

In the running example, the 95% CI for the CTR difference is approximately $(0.017, 0.029)$. Zero is not in this interval, which confirms the significant result. And crucially, the interval tells you *how big* the effect is -- something the p-value alone does not.

## Power: Can You Even Detect the Effect?

### What Power Is

**Power** = $1 - \beta$ = the probability of correctly detecting a real effect.

If you run an A/B test with power = 0.80, that means: if the new algorithm truly is better, you have an 80% chance of your test correctly saying so.

### Why Power Matters

An underpowered test is a waste of time. If your power is only 0.20, there is an 80% chance you will miss a real improvement. You will conclude "no significant difference" and kill a change that would have helped.

### What Affects Power

Four levers control power:

1. **Sample size (n)** -- more data = more power. This is usually the lever you pull.
2. **Effect size** -- larger effects are easier to detect.
3. **Significance level ($\alpha$)** -- higher $\alpha$ = more power (but more false positives).
4. **Variance** -- less noise = more power.

### Power Analysis: How Many Users Do You Need?

Before running an A/B test, do a power analysis to figure out the required sample size. The formula for a two-sample test:

$$n = 2 \cdot \left(\frac{z_{\alpha/2} + z_{\text{power}}}{d}\right)^2$$

where $d$ is the effect size (Cohen's d).

```
Effect Size (Cohen's d) | Description | Required n per group (power=0.80, α=0.05)
─────────────────────────────────────────────────────────────────────────────────
d = 0.2  (small)        | Subtle      | ~393 per group
d = 0.5  (medium)       | Noticeable  | ~63 per group
d = 0.8  (large)        | Obvious     | ~25 per group
```

For the running example, the recommendation algorithm A/B test: you chose n = 10,000 per group, which gives you very high power to detect even small differences. That is why the result was so decisive.

### Power Curve

```
Power vs. Sample Size (effect size d = 0.5, α = 0.05)

Power
1.0 │                                    ●─────────────
    │                              ●
0.8 │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ●─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ← target power
    │                   ●
0.6 │              ●
    │          ●
0.4 │      ●
    │   ●
0.2 │ ●
    │●
0.0 └──┬──┬──┬───┬───┬───┬───┬───┬───┬──
       10 20 30  50  70  100 150 200 300
                Sample size (n per group)
```

## Code Example

```python
import numpy as np
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# RUNNING EXAMPLE: A/B TEST FOR RECOMMENDATION ALGORITHM
# ============================================================
# H₀: No difference in CTR between old and new algorithm
# H₁: New algorithm has higher CTR
# n = 10,000 per group

print("A/B TEST: OLD vs NEW RECOMMENDATION ALGORITHM")
print("=" * 60)

n_control = 10000
n_treatment = 10000

# Simulate: old algorithm has 4.5% CTR, new has 6.8% CTR
control_clicks = np.random.binomial(1, 0.045, n_control)
treatment_clicks = np.random.binomial(1, 0.068, n_treatment)

ctr_control = control_clicks.mean()
ctr_treatment = treatment_clicks.mean()
diff = ctr_treatment - ctr_control

print(f"Control CTR:   {ctr_control:.4f}  ({ctr_control*100:.2f}%)")
print(f"Treatment CTR: {ctr_treatment:.4f}  ({ctr_treatment*100:.2f}%)")
print(f"Difference:    {diff:.4f}  ({diff*100:.2f} percentage points)")

# Two-proportion z-test (one-tailed: is treatment > control?)
p_pooled = (control_clicks.sum() + treatment_clicks.sum()) / (n_control + n_treatment)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
z_stat = diff / se
p_value_one_tail = 1 - norm.cdf(z_stat)

print(f"\nPooled proportion: {p_pooled:.4f}")
print(f"Standard error:    {se:.6f}")
print(f"Z-statistic:       {z_stat:.3f}")
print(f"P-value (one-tail): {p_value_one_tail:.2e}")

# 95% confidence interval for the difference
ci_low = diff - 1.96 * se
ci_high = diff + 1.96 * se
print(f"\n95% CI for difference: ({ci_low:.4f}, {ci_high:.4f})")

if p_value_one_tail < 0.05:
    print("\nDecision: REJECT H₀ — Ship the new algorithm!")
    print(f"The {diff*100:.2f} pp lift is statistically significant.")
else:
    print("\nDecision: FAIL TO REJECT H₀ — Keep the current algorithm.")

# ============================================================
# ONE-SAMPLE T-TEST: IS LATENCY DIFFERENT FROM TARGET?
# ============================================================

print("\n" + "=" * 60)
print("ONE-SAMPLE T-TEST: Latency Check")
print("=" * 60)
print("Question: Is the mean response time different from 200ms?")

# Sample data: response times (true mean = 215ms)
response_times = np.random.normal(loc=215, scale=50, size=30)

mu_null = 200
t_statistic, p_value = stats.ttest_1samp(response_times, mu_null)

print(f"\nSample mean: {np.mean(response_times):.2f}ms")
print(f"Sample std:  {np.std(response_times, ddof=1):.2f}ms")
print(f"H₀: μ = {mu_null}ms")
print(f"t-statistic: {t_statistic:.3f}")
print(f"p-value:     {p_value:.4f}")

if p_value < 0.05:
    print("Result: Reject H₀ — Response time is significantly different from 200ms")
else:
    print("Result: Fail to reject H₀")

# Confidence interval
sem = stats.sem(response_times)
ci = stats.t.interval(0.95, len(response_times)-1,
                      loc=np.mean(response_times), scale=sem)
print(f"95% CI for mean: ({ci[0]:.2f}, {ci[1]:.2f})ms")

# ============================================================
# TWO-SAMPLE T-TEST: COMPARING MODEL ACCURACY
# ============================================================

print("\n" + "=" * 60)
print("TWO-SAMPLE T-TEST: Model A vs Model B Accuracy")
print("=" * 60)

model_a_scores = np.random.normal(loc=0.82, scale=0.05, size=50)
model_b_scores = np.random.normal(loc=0.85, scale=0.05, size=50)

t_stat, p_val_two = stats.ttest_ind(model_a_scores, model_b_scores)
p_val_one = p_val_two / 2 if t_stat < 0 else 1 - p_val_two / 2

print(f"\nModel A mean accuracy: {np.mean(model_a_scores):.4f}")
print(f"Model B mean accuracy: {np.mean(model_b_scores):.4f}")
print(f"Difference:            {np.mean(model_b_scores) - np.mean(model_a_scores):.4f}")
print(f"t-statistic:           {t_stat:.3f}")
print(f"p-value (two-tail):    {p_val_two:.4f}")
print(f"p-value (one-tail):    {p_val_one:.4f}")

if p_val_one < 0.05:
    print("Result: Model B is significantly better (α=0.05, one-tailed)")
else:
    print("Result: Insufficient evidence that B is better")

# ============================================================
# MULTIPLE COMPARISONS: THE P-HACKING TRAP
# ============================================================

print("\n" + "=" * 60)
print("MULTIPLE COMPARISONS: The P-Hacking Trap")
print("=" * 60)

n_tests = 20
p_values = []

for i in range(n_tests):
    group1 = np.random.normal(0, 1, 30)
    group2 = np.random.normal(0, 1, 30)
    _, p = stats.ttest_ind(group1, group2)
    p_values.append(p)

significant_uncorrected = sum(p < 0.05 for p in p_values)
print(f"\nTests run: {n_tests} (ALL nulls are TRUE — no real effects)")
print(f"Significant at α=0.05: {significant_uncorrected}")
print(f"Expected false positives: {n_tests * 0.05:.1f}")
print("\nThis is why testing 20 metrics and cherry-picking the")
print("significant one is bad science (and bad engineering).")

# Bonferroni correction
alpha_corrected = 0.05 / n_tests
significant_bonferroni = sum(p < alpha_corrected for p in p_values)
print(f"\nWith Bonferroni correction (α={alpha_corrected:.4f}):")
print(f"Significant: {significant_bonferroni}")

# ============================================================
# POWER ANALYSIS: HOW MANY USERS DO YOU NEED?
# ============================================================

print("\n" + "=" * 60)
print("POWER ANALYSIS: How Many Users Do You Need?")
print("=" * 60)

def calculate_power(n, effect_size, alpha=0.05):
    """Calculate power for two-sample z-test."""
    z_alpha = norm.ppf(1 - alpha/2)
    z_power = effect_size * np.sqrt(n/2) - z_alpha
    power = norm.cdf(z_power)
    return power

def required_sample_size(effect_size, power=0.8, alpha=0.05):
    """Calculate required sample size per group."""
    z_alpha = norm.ppf(1 - alpha/2)
    z_power = norm.ppf(power)
    n = 2 * ((z_alpha + z_power) / effect_size) ** 2
    return int(np.ceil(n))

effect_sizes = [0.2, 0.5, 0.8]  # small, medium, large
labels = ['small', 'medium', 'large']
print("\nEffect Size | Required n per group (power=0.8, α=0.05)")
print("-" * 55)
for d, label in zip(effect_sizes, labels):
    n_required = required_sample_size(d)
    print(f"d = {d} ({label:6s}) | n = {n_required}")

print("\nPower vs. Sample Size (effect_size = 0.5):")
print(f"{'n':>6s}  {'Power':>7s}")
print("-" * 15)
for n in [10, 20, 50, 100, 200]:
    power = calculate_power(n, effect_size=0.5)
    print(f"{n:>6d}  {power:>7.3f}")

# ============================================================
# CHI-SQUARED TEST: IS BEHAVIOR SEGMENT-DEPENDENT?
# ============================================================

print("\n" + "=" * 60)
print("CHI-SQUARED TEST: Algorithm Preference by User Segment")
print("=" * 60)

observed = np.array([
    [120, 80],   # Segment A: clicked recs from Algo X vs Algo Y
    [90, 110]    # Segment B: clicked recs from Algo X vs Algo Y
])

chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print("\nObserved frequencies:")
print(f"          Algo X   Algo Y")
print(f"Seg A:     {observed[0,0]}       {observed[0,1]}")
print(f"Seg B:      {observed[1,0]}      {observed[1,1]}")

print(f"\nExpected frequencies (if independent):")
print(f"          Algo X   Algo Y")
print(f"Seg A:    {expected[0,0]:.1f}     {expected[0,1]:.1f}")
print(f"Seg B:    {expected[1,0]:.1f}     {expected[1,1]:.1f}")

print(f"\nchi-squared = {chi2:.3f}")
print(f"p-value = {p_value:.4f}")
print(f"df = {dof}")

if p_value < 0.05:
    print("Result: Algorithm preference depends on user segment")
else:
    print("Result: No significant association")

# ============================================================
# FULL A/B TEST FRAMEWORK (reusable)
# ============================================================

print("\n" + "=" * 60)
print("REUSABLE A/B TEST FRAMEWORK")
print("=" * 60)

def ab_test(control, treatment, alpha=0.05):
    """
    Perform a complete A/B test with full reporting.
    Returns a dict with all key metrics.
    """
    n_c = len(control)
    n_t = len(treatment)
    mean_c = np.mean(control)
    mean_t = np.mean(treatment)
    lift = (mean_t - mean_c) / mean_c * 100

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(control, treatment)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((n_c-1)*np.var(control, ddof=1) +
                          (n_t-1)*np.var(treatment, ddof=1)) /
                         (n_c + n_t - 2))
    cohens_d = (mean_t - mean_c) / pooled_std

    # Confidence interval for the difference
    se_diff = np.sqrt(np.var(control, ddof=1)/n_c +
                      np.var(treatment, ddof=1)/n_t)
    ci_low = (mean_t - mean_c) - 1.96 * se_diff
    ci_high = (mean_t - mean_c) + 1.96 * se_diff

    return {
        'control_mean': mean_c,
        'treatment_mean': mean_t,
        'lift_percent': lift,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'ci_95': (ci_low, ci_high),
        'significant': p_value < alpha
    }

# Use the framework on our running example data
results = ab_test(control_clicks, treatment_clicks)

print("\nA/B Test Results (Recommendation Algorithm):")
print(f"Control CTR:   {results['control_mean']:.4f}")
print(f"Treatment CTR: {results['treatment_mean']:.4f}")
print(f"Relative lift: {results['lift_percent']:+.1f}%")
print(f"p-value:       {results['p_value']:.2e}")
print(f"Cohen's d:     {results['cohens_d']:.3f}")
print(f"95% CI for difference: ({results['ci_95'][0]:.4f}, {results['ci_95'][1]:.4f})")
print(f"Significant:   {'Yes' if results['significant'] else 'No'}")
```

## Where Hypothesis Testing Shows Up in ML

### 1. A/B Testing for Models

This is the most direct application. You deploy two models, split traffic, and use a hypothesis test to decide which is better. The running example in this chapter is exactly this scenario.

### 2. Feature Selection

Before training, you can use statistical tests to filter out irrelevant features:
- **T-test / ANOVA**: Does this numeric feature differ across classes?
- **Chi-squared test**: Is this categorical feature associated with the target?
- **F-test**: Does this group of features jointly have predictive power?

### 3. Model Diagnostics

After training, tests tell you whether your model's assumptions hold:
- **Shapiro-Wilk test**: Are the residuals normally distributed?
- **Breusch-Pagan test**: Is the variance constant (homoscedasticity)?
- **Durbin-Watson test**: Are the residuals independent (no autocorrelation)?

### 4. Research and Model Comparison

When publishing results or deciding between architectures:

| Scenario | Recommended Test |
|---|---|
| Compare 2 models on same test set | Paired t-test |
| Compare accuracy across multiple datasets | Wilcoxon signed-rank |
| Compare multiple models | ANOVA + post-hoc tests |
| Classification accuracy | McNemar's test |
| Feature importance | Permutation test |

## Pitfalls and Practical Wisdom

### Pitfall 1: Confusing Statistical and Practical Significance

With n = 10,000,000 users, even a 0.01% improvement in CTR will be "statistically significant." But is it worth the engineering cost to maintain the new system? Always report **effect sizes** and **confidence intervals**, not just p-values.

### Pitfall 2: P-Hacking

You test 20 different metrics. One of them has p < 0.05. You report that one. Congratulations, you just found noise.

At $\alpha = 0.05$ with 20 tests, you *expect* 1 false positive even when all nulls are true. Solutions:
- **Pre-register** your primary metric before the experiment
- **Bonferroni correction**: use $\alpha / k$ where $k$ is the number of tests
- **False Discovery Rate (FDR)** control for exploratory analysis

### Pitfall 3: Ignoring Power (Underpowered Tests)

"We ran an A/B test for 2 days, saw no significant difference, so we killed the feature." Maybe you just did not have enough data. Failing to reject $H_0$ does NOT mean $H_0$ is true. Always do a **power analysis** before running the experiment.

### Pitfall 4: Peeking at Results

You check your A/B test every day and stop as soon as you see p < 0.05. This inflates your false positive rate far above 5%. If you want to check early, use **sequential testing** methods designed for this.

### Pitfall 5: One Metric, Many Dimensions

Your overall A/B test is not significant, so you slice by country, device type, and user age. In one segment, you find p < 0.05. This is the same multiple comparisons problem -- you just hid it behind subgroup analysis.

## Exercises

### Exercise 1: Interpreting P-Values

You run an A/B test on the recommendation algorithm and get p = 0.03. Which interpretations are correct?

a) There is a 3% chance the null hypothesis is true.
b) If there were truly no difference, there is a 3% chance of seeing a result this extreme.
c) You are 97% confident the new algorithm is better.
d) If you repeated this experiment many times and there were truly no effect, about 3% of experiments would show a difference this large or larger.

**Solution**:
Only **(b)** and **(d)** are correct -- they say the same thing in different words.

(a) is wrong: the p-value is not $P(H_0 \text{ is true})$. It is $P(\text{data} \mid H_0)$, not $P(H_0 \mid \text{data})$.
(c) is wrong: this confuses the p-value with a confidence level. You would need Bayes' theorem to make a statement like this.

### Exercise 2: Sample Size Calculation

You want to detect a "small" effect (Cohen's d = 0.2) with 80% power at $\alpha = 0.05$. How many users do you need per group?

**Solution**:
Using the formula: $n = 2 \cdot \left(\frac{z_{\alpha/2} + z_{\text{power}}}{d}\right)^2$

```python
from scipy.stats import norm
import numpy as np

z_alpha = norm.ppf(0.975)  # 1.96
z_power = norm.ppf(0.8)    # 0.84
d = 0.2

n = 2 * ((z_alpha + z_power) / d) ** 2
print(f"Required n per group: {int(np.ceil(n))}")  # ~393
```

You need approximately **393 users per group** (786 total). For the running example with n = 10,000 per group, you are massively overpowered for even small effects -- which is exactly why the result was so clear.

### Exercise 3: Type I vs Type II Error Tradeoff

Your anomaly detection system has:
- $\alpha = 0.01$ (1% false positive rate)
- Power = 0.95 (95% detection rate, so $\beta = 0.05$)

a) If 1% of requests are anomalous and you process 10,000 requests, how many false positives and false negatives do you expect?

b) Despite the low false positive rate, why might you still have a problem?

**Solution**:

a) In 10,000 requests:
- Anomalous: 100 (1% of 10,000)
- Normal: 9,900 (99% of 10,000)

False positives = 9,900 * 0.01 = **99**
False negatives = 100 * 0.05 = **5**

b) Even with a 1% false positive rate, you get **99 false positives** vs. only **5 false negatives**. That is because there are so many more normal requests than anomalous ones. This is the **base rate problem** -- the same issue that makes rare-event detection hard in ML (class imbalance). Your precision is only 100 / (100 + 99 - 5) = roughly 50%, even with seemingly good error rates.

## Summary

Here is what you need to remember:

- **Null hypothesis ($H_0$)**: "Nothing changed" -- the current production system is your default assumption.
- **P-value**: The probability of seeing your result if nothing changed. It is a false-alarm rate, not the probability that $H_0$ is true.
- **Significance level ($\alpha$)**: Your tolerance for false alarms. Set it before you look at the data.
- **Type I error (false positive)**: Deploying a change that does not actually help. Controlled by $\alpha$.
- **Type II error (false negative)**: Rejecting a change that would have helped. Controlled by sample size and power.
- **Power ($1 - \beta$)**: Your ability to detect real effects. Do a power analysis before running any experiment.
- **Confidence interval**: Tells you both significance AND effect size. Report it alongside p-values.
- **Key pitfall**: p < 0.05 does NOT mean 95% chance the effect is real.
- **In practice**: Pre-register your hypothesis, choose your metric before the experiment, correct for multiple comparisons, and always report effect sizes.

## What Comes Next

You have completed statistics. Descriptive stats, sampling, estimation, and hypothesis testing -- these tools let you evaluate models, compare algorithms, and make data-driven decisions. When someone asks "is model B better than model A?", you now know how to answer rigorously instead of eyeballing metrics and hoping for the best. These foundations will serve you every time you run an experiment, validate a model, or make a ship/no-ship decision.
