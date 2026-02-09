# Chapter 2: Shannon Entropy

## From Single Surprise to Average Uncertainty

Single-event information measures surprise at ONE outcome. Entropy averages this over ALL outcomes — giving you the overall uncertainty of a distribution.

In the last chapter, you saw that a single event carries $-\log P(x)$ bits of surprise. But in practice, you don't care about just one outcome. You want to know: across all possible outcomes, how much surprise should you *expect* on average? That's entropy.

A fair coin has maximum uncertainty — you genuinely don't know what's coming. A biased coin (99% heads) has almost no uncertainty. Entropy quantifies this: $H(X) = -\sum p(x) \log p(x)$. It's the average surprise, the expected information content, and — in ML — the baseline for how good a classifier CAN be.

## Intuition: Why You Already Know This

You've encountered entropy your entire engineering career, even if you didn't call it that.

**Compression lower bound.** When you gzip a file, the algorithm exploits patterns to shrink it. Entropy is the theoretical floor — you literally cannot compress data below $H$ bits per symbol on average. If your source has 2 bits of entropy per character, no lossless compression scheme in the universe will average below 2 bits per character. That's Shannon's source coding theorem, and it's why entropy matters to anyone who's ever shipped bytes over a wire.

**Decision tree splits.** Every time scikit-learn's `DecisionTreeClassifier` picks a feature to split on, it's maximizing *information gain* — which is just the reduction in entropy. Before the split, the labels have some entropy. After the split, each branch has lower entropy. The feature that drops entropy the most wins. You've been using entropy every time you trained a tree.

**Password strength.** When your security library rates a password, it's estimating entropy. A password drawn uniformly from 8 lowercase letters has $8 \times \log_2(26) \approx 37.6$ bits of entropy. A password drawn from a dictionary of 10,000 words has $\log_2(10000) \approx 13.3$ bits. High entropy = hard to guess. Low entropy = brute-forceable.

**Random number generation.** Your OS collects entropy from hardware events (mouse movements, disk timing, network jitter) to seed `/dev/urandom`. High entropy = good randomness source. Low entropy = predictable output. The kernel literally tracks an "entropy pool" measured in bits.

## Running Example: Genre Prediction

Throughout this chapter, you'll follow one concrete example. You have a music genre classifier with four classes: Rock, Jazz, Pop, Classical. Two models make predictions for the same song:

**Model A (clueless):** outputs uniform $[0.25, 0.25, 0.25, 0.25]$

**Model B (confident):** outputs $[0.9, 0.05, 0.03, 0.02]$

Intuitively, Model A has learned nothing — it's guessing randomly. Model B has strong opinions. Entropy will make this precise.

## The Formula

For a discrete random variable $X$ with probability distribution $P$, the **Shannon entropy** is:

$$H(X) = H(P) = -\sum_{x} P(x) \log P(x) = \mathbb{E}[-\log P(X)]$$

That's it. You take every possible outcome, compute its self-information $-\log P(x)$, weight it by how likely it is $P(x)$, and sum. The result is the *expected surprise* — how uncertain you are before observing $X$.

**Units depend on the log base:**
- $\log_2$: bits (most common in ML and information theory)
- $\ln$: nats (natural units, common in optimization because gradients are cleaner)
- $\log_{10}$: hartleys (rare, but you'll see it in older textbooks)

**Convention for zero probabilities:** When $P(x) = 0$, you define $0 \cdot \log(0) = 0$ using the limit:

$$\lim_{p \to 0^+} p \log p = 0$$

This makes sense: an event that never happens contributes zero average surprise.

### Back to Genre Prediction

Let's compute entropy for both models using base-2 (bits):

**Model A** (uniform $[0.25, 0.25, 0.25, 0.25]$):

$$H_A = -4 \times (0.25 \times \log_2 0.25) = -4 \times (0.25 \times (-2)) = 2.0 \text{ bits}$$

**Model B** (confident $[0.9, 0.05, 0.03, 0.02]$):

$$H_B = -(0.9 \log_2 0.9 + 0.05 \log_2 0.05 + 0.03 \log_2 0.03 + 0.02 \log_2 0.02)$$
$$\approx -(−0.137 + (−0.216) + (−0.152) + (−0.113)) \approx 0.618 \text{ bits}$$

Model A: 2.0 bits (maximum for 4 classes). Model B: ~0.6 bits. The numbers confirm your intuition — the uniform model carries maximum uncertainty, the confident model carries very little.

### Binary Entropy Function

For the special case of a Bernoulli random variable with $P(X=1) = p$:

$$H_b(p) = -p \log_2 p - (1-p) \log_2(1-p)$$

This function peaks at $p = 0.5$ with $H_b(0.5) = 1$ bit and reaches zero at both extremes.

```
Entropy H(p)
    ^
  1 |        ****
    |      **    **
0.8 |     *        *
    |    *          *
0.6 |   *            *
    |  *              *
0.4 | *                *
    |*                  *
0.2 |
    |
  0 +--------------------> p
    0   0.2  0.4  0.6  0.8  1.0
```

The binary entropy curve is the most important shape in information theory. Every time you see a classification problem with two outcomes, this curve tells you the uncertainty story.

## Entropy Across Distributions: An ASCII Bar Chart

Here's what entropy looks like for several distributions, all in bits:

```
Distribution                      Entropy (bits)    Bar
─────────────────────────────────────────────────────────────────
Certain [1.0, 0, 0, 0]           0.000             |
99/1 coin [0.99, 0.01]           0.081             |=
90/10 coin [0.90, 0.10]          0.469             |=====
Genre model B [.9,.05,.03,.02]   0.618             |======
70/30 coin [0.70, 0.30]          0.881             |=========
Fair coin [0.50, 0.50]           1.000             |==========
Loaded die [.5,.1,.1,.1,.1,.1]   2.161             |======================
Fair 4-sided [.25,.25,.25,.25]   2.000             |====================
Fair 6-sided die                 2.585             |==========================
Fair 8-sided die                 3.000             |==============================
Uniform over 256 (1 byte)        8.000             |============== ... (80) ==============
─────────────────────────────────────────────────────────────────
                                                   0         1         2         3
                                                   Each '=' ≈ 0.1 bits
```

Notice the pattern: more equally-spread outcomes = higher entropy. The uniform distribution over $n$ outcomes always hits the ceiling at $\log_2(n)$ bits. Anything non-uniform falls below.

## Properties of Entropy

These aren't abstract theorems — each one has a direct engineering consequence.

### 1. Non-negativity: $H(X) \geq 0$

Entropy is always non-negative. You can't have negative average surprise. This follows directly from the fact that probabilities are in $[0, 1]$, so $-\log P(x) \geq 0$.

*Engineering consequence:* Any time your entropy calculation returns a negative number, you have a bug.

### 2. Maximum for uniform distribution: $H(X) \leq \log |X|$

Entropy is maximized when all outcomes are equally likely. The maximum value is $\log n$ where $n$ is the number of possible outcomes.

*Engineering consequence:* This gives you a normalization constant. You can compute $H(X) / \log n$ to get a "normalized entropy" between 0 and 1 that's comparable across distributions with different support sizes.

> **Common Mistake:** Entropy is maximized by the uniform distribution. If your model's output is uniform, it has learned NOTHING. When you see a softmax layer outputting near-uniform probabilities after training, that's not "being fair" — that's a model that failed to learn any signal from the data. Don't confuse maximum entropy with good performance.

### 3. Zero for deterministic distributions: $H(X) = 0$ iff $X$ is constant

If you know the outcome with certainty, there's zero uncertainty. Conversely, any entropy above zero means there's genuine uncertainty.

*Engineering consequence:* In a perfectly separable classification problem, a decision tree can achieve zero entropy in its leaf nodes. In practice, you'll never see exactly zero — and that's fine.

### 4. Additivity for independent variables: $H(X, Y) = H(X) + H(Y)$

When $X$ and $Y$ are independent, the joint entropy is the sum of the individual entropies.

*Engineering consequence:* This is why you can think of entropy as a "measure of information." Independent information sources add up, just like you'd expect from a well-behaved measure.

### 5. Chain rule: $H(X, Y) = H(X) + H(Y|X)$

The joint entropy of two variables equals the entropy of the first plus the conditional entropy of the second given the first. This holds whether or not $X$ and $Y$ are independent.

*Engineering consequence:* This is the mathematical backbone of decision trees. Each split conditions on a feature, and the chain rule tells you exactly how much entropy remains.

### Conditional Entropy

The entropy of $Y$ given $X$ is:

$$H(Y|X) = \sum_x P(x) H(Y|X=x) = -\sum_{x,y} P(x,y) \log P(y|x)$$

This measures the remaining uncertainty in $Y$ after you observe $X$. It's always less than or equal to $H(Y)$ — knowing something never increases uncertainty on average.

### Mutual Information (Preview)

The reduction in entropy from knowing another variable:

$$I(X; Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)$$

This quantifies how much information $X$ and $Y$ share. You'll see this again when we cover KL divergence.

## ML Applications

### Decision Trees and Information Gain

This is where entropy earns its keep in day-to-day ML. The information gain for splitting on feature $X$ is:

$$\text{Information Gain} = H(Y) - H(Y|X) = H(\text{parent}) - \sum_{\text{children}} \frac{N_{\text{child}}}{N_{\text{parent}}} H(\text{child})$$

Trees split on the feature that maximizes information gain — equivalently, minimizes the weighted conditional entropy of the children. Every `criterion='entropy'` you've passed to scikit-learn triggers exactly this computation.

Let's trace through a concrete split. You're predicting whether users churn (binary: yes/no). Before splitting, 70% stay and 30% churn:

$$H(\text{before}) = -0.7 \log_2 0.7 - 0.3 \log_2 0.3 \approx 0.881 \text{ bits}$$

You split on "subscription tier." The Premium branch has 90% stay / 10% churn. The Free branch has 30% stay / 70% churn. Branches are equal size:

$$H(\text{after}) = 0.5 \times H_b(0.9) + 0.5 \times H_b(0.3) = 0.5 \times 0.469 + 0.5 \times 0.881 = 0.675 \text{ bits}$$

$$\text{Information Gain} = 0.881 - 0.675 = 0.206 \text{ bits}$$

That 0.206 bits is how much uncertainty the "subscription tier" feature resolves. If another feature gave 0.4 bits, you'd split on that one instead.

### Neural Network Confidence

The entropy of a softmax output tells you how confident your model is:

- **Low entropy** = peaked distribution = confident prediction
- **High entropy** = flat distribution = uncertain prediction

You'll use this for:

- **Active learning:** Query the oracle on samples where your model has highest entropy — those are the ones where a label would be most informative.
- **Uncertainty quantification:** Flag predictions with entropy above a threshold for human review.
- **Out-of-distribution detection:** OOD inputs often produce higher-entropy softmax outputs because the model hasn't seen anything like them.

Back to genre prediction: if you threshold at 1.0 bits, Model A (2.0 bits) gets flagged as uncertain, Model B (0.6 bits) passes as confident. That's a simple but effective production-grade uncertainty filter.

### Maximum Entropy Principle

When you have constraints but no other information, the least biased distribution is the one with maximum entropy subject to those constraints. This principle underlies:

- **Logistic regression** (yes, really — it's a maximum entropy classifier)
- **Exponential family distributions** (they're the max-entropy distributions for given sufficient statistics)
- **Regularization** (entropy regularization in RL encourages exploration)

### Reinforcement Learning: Entropy as Exploration

In policy gradient methods, adding an entropy bonus to the objective prevents premature convergence:

$$\mathcal{L} = \mathbb{E}[R] + \alpha H(\pi)$$

The $\alpha H(\pi)$ term rewards the policy for maintaining uncertainty — trying diverse actions rather than committing too early to a suboptimal strategy. SAC (Soft Actor-Critic) is built entirely around this idea.

## Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy

def entropy(probabilities, base='e'):
    """
    Calculate Shannon entropy of a probability distribution.

    Parameters:
    -----------
    probabilities : array-like
        Probability distribution (must sum to 1)
    base : str
        'e' for nats, '2' for bits

    Returns:
    --------
    float : Entropy value
    """
    p = np.asarray(probabilities, dtype=float)

    # Validate distribution
    if not np.isclose(p.sum(), 1.0):
        raise ValueError(f"Probabilities must sum to 1, got {p.sum()}")
    if np.any(p < 0):
        raise ValueError("Probabilities must be non-negative")

    # Filter out zeros (0 * log(0) = 0 by convention)
    p_nonzero = p[p > 0]

    if base == 'e':
        return -np.sum(p_nonzero * np.log(p_nonzero))
    elif base == '2':
        return -np.sum(p_nonzero * np.log2(p_nonzero))
    else:
        raise ValueError("Base must be 'e' or '2'")

def binary_entropy(p, base='2'):
    """Calculate entropy of a Bernoulli(p) distribution."""
    if p == 0 or p == 1:
        return 0.0
    return entropy([p, 1-p], base=base)

# Example 1: Entropy of various distributions
print("=== Entropy Examples ===\n")

distributions = {
    "Fair coin": [0.5, 0.5],
    "Biased coin (90/10)": [0.9, 0.1],
    "Fair 6-sided die": [1/6] * 6,
    "Loaded die": [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
    "Certain outcome": [1.0, 0.0, 0.0],
    "Uniform over 8": [1/8] * 8,
}

for name, dist in distributions.items():
    H_bits = entropy(dist, base='2')
    H_nats = entropy(dist, base='e')
    max_entropy = np.log2(len(dist))
    efficiency = H_bits / max_entropy if max_entropy > 0 else 0

    print(f"{name}:")
    print(f"  Distribution: {[f'{p:.2f}' for p in dist]}")
    print(f"  Entropy: {H_bits:.3f} bits = {H_nats:.3f} nats")
    print(f"  Max possible: {max_entropy:.3f} bits")
    print(f"  Efficiency: {efficiency:.1%}\n")

# Example 2: Binary entropy curve
print("=== Binary Entropy Function ===\n")

p_values = np.linspace(0.001, 0.999, 1000)
H_values = [binary_entropy(p) for p in p_values]

plt.figure(figsize=(10, 6))
plt.plot(p_values, H_values, 'b-', linewidth=2)
plt.xlabel('Probability p', fontsize=12)
plt.ylabel('Binary Entropy H(p) [bits]', fontsize=12)
plt.title('Binary Entropy Function: Uncertainty of a Coin Flip', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1.1)

# Mark key points
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Maximum entropy')
plt.axvline(x=0.5, color='g', linestyle='--', alpha=0.5, label='Fair coin')
plt.plot(0.5, 1, 'go', markersize=10)
plt.annotate('Maximum uncertainty\nat p=0.5', xy=(0.5, 1), xytext=(0.6, 0.85),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

plt.legend()
plt.tight_layout()
plt.savefig('binary_entropy.png', dpi=150)
plt.show()

# Example 3: Genre prediction — the running example
print("=== Genre Prediction: Entropy as Confidence ===\n")

genre_models = {
    "Model A (uniform)": [0.25, 0.25, 0.25, 0.25],
    "Model B (confident)": [0.9, 0.05, 0.03, 0.02],
    "Model C (somewhat confident)": [0.6, 0.2, 0.15, 0.05],
    "Model D (two-way toss-up)": [0.45, 0.45, 0.05, 0.05],
}

for name, probs in genre_models.items():
    H = entropy(probs, base='2')
    max_H = np.log2(4)
    confidence = 1 - (H / max_H)

    print(f"{name}:")
    print(f"  Softmax output: {probs}")
    print(f"  Entropy: {H:.3f} bits (max: {max_H:.3f})")
    print(f"  Confidence score: {confidence:.1%}\n")

# Example 4: Entropy reduction in decision trees
print("=== Entropy in Decision Trees ===\n")

# Before split
p_play = 9/14
H_before = binary_entropy(p_play)
print(f"Before split: P(play) = {p_play:.3f}, H = {H_before:.3f} bits")

# After split on "Outlook"
# Sunny: 2/5 play, Overcast: 4/4 play, Rain: 3/5 play
splits = [
    ("Sunny", 5/14, 2/5),
    ("Overcast", 4/14, 4/4),
    ("Rain", 5/14, 3/5),
]

H_after = 0
for outlook, weight, p_play_given in splits:
    H_conditional = binary_entropy(p_play_given)
    H_after += weight * H_conditional
    print(f"  {outlook}: weight={weight:.3f}, P(play|{outlook})={p_play_given:.2f}, H={H_conditional:.3f}")

print(f"\nAfter split: H(Play|Outlook) = {H_after:.3f} bits")
print(f"Information gain: {H_before - H_after:.3f} bits")

# Example 5: Verify with scipy
print("\n=== Verification with SciPy ===")
test_dist = [0.25, 0.25, 0.5]
our_entropy = entropy(test_dist, base='2')
scipy_ent = scipy_entropy(test_dist, base=2)
print(f"Our implementation: {our_entropy:.6f}")
print(f"SciPy entropy: {scipy_ent:.6f}")
print(f"Match: {np.isclose(our_entropy, scipy_ent)}")
```

**Output:**
```
=== Entropy Examples ===

Fair coin:
  Distribution: ['0.50', '0.50']
  Entropy: 1.000 bits = 0.693 nats
  Max possible: 1.000 bits
  Efficiency: 100.0%

Biased coin (90/10):
  Distribution: ['0.90', '0.10']
  Entropy: 0.469 bits = 0.325 nats
  Max possible: 1.000 bits
  Efficiency: 46.9%

Fair 6-sided die:
  Distribution: ['0.17', '0.17', '0.17', '0.17', '0.17', '0.17']
  Entropy: 2.585 bits = 1.792 nats
  Max possible: 2.585 bits
  Efficiency: 100.0%

=== Genre Prediction: Entropy as Confidence ===

Model A (uniform):
  Softmax output: [0.25, 0.25, 0.25, 0.25]
  Entropy: 2.000 bits (max: 2.000)
  Confidence score: 0.0%

Model B (confident):
  Softmax output: [0.9, 0.05, 0.03, 0.02]
  Entropy: 0.618 bits (max: 2.000)
  Confidence score: 69.1%

=== Entropy in Decision Trees ===

Before split: P(play) = 0.643, H = 0.940 bits
  Sunny: weight=0.357, P(play|Sunny)=0.40, H=0.971
  Overcast: weight=0.286, P(play|Overcast)=1.00, H=0.000
  Rain: weight=0.357, P(play|Rain)=0.60, H=0.971

After split: H(Play|Outlook) = 0.694 bits
Information gain: 0.247 bits
```

## Interpretation Guide

| Entropy (bits) | Interpretation |
|----------------|----------------|
| 0 | Complete certainty |
| 0.5 | Low uncertainty |
| 1 | Binary choice uncertainty |
| 2-3 | Moderate uncertainty |
| $\log_2(n)$ | Maximum uncertainty (uniform over n) |

### Comparing Entropies

When you're comparing entropy values across different scenarios, keep these rules in mind:

- **Same number of outcomes:** Higher entropy = more uncertainty. Simple.
- **Different number of outcomes:** Compare to the maximum possible ($\log_2 n$). Raw entropy values are misleading — 2.0 bits is maximum for 4 classes but moderate for 256 classes.
- **Normalized entropy** $H/H_{max}$ gives you a 0-to-1 scale that's comparable across different support sizes.

## Exercises

### Exercise 1: Maximum Entropy Proof

**Problem**: Prove that for a discrete distribution over $n$ outcomes, entropy is maximized when the distribution is uniform.

**Solution**:
Using Lagrange multipliers with constraint $\sum p_i = 1$:

$$\mathcal{L} = -\sum p_i \log p_i - \lambda(\sum p_i - 1)$$

Taking derivative and setting to zero:
$$\frac{\partial \mathcal{L}}{\partial p_i} = -\log p_i - 1 - \lambda = 0$$

This gives $p_i = e^{-1-\lambda}$ for all $i$, meaning all probabilities are equal.
With $\sum p_i = 1$, we get $p_i = 1/n$, so $H_{max} = \log n$.

### Exercise 2: Entropy of English Text

**Problem**: If English letters appeared uniformly, what would the entropy per letter be? Given that actual English has about 1.5 bits per letter, what does this tell you?

**Solution**:
```python
# Uniform distribution over 26 letters
H_uniform = np.log2(26)  # = 4.7 bits

# Actual English
H_english = 1.5  # bits (approximately)

# Redundancy
redundancy = 1 - (H_english / H_uniform)

print(f"Maximum entropy: {H_uniform:.2f} bits")
print(f"Actual entropy: {H_english:.2f} bits")
print(f"Redundancy: {redundancy:.1%}")
# Redundancy ~ 68% - English is highly predictable!
```

This is why compression works so well on English text — and why language models can predict the next token with surprising accuracy. The low entropy means there's a LOT of structure to exploit.

### Exercise 3: Information Gain Calculation

**Problem**: You have a dataset with class balance [0.7, 0.3]. After splitting on a feature, the two groups have class balances [0.9, 0.1] and [0.3, 0.7] with equal size. Calculate the information gain.

**Solution**:
```python
# Before split
H_before = binary_entropy(0.7)  # = 0.881 bits

# After split (equal-sized groups)
H_group1 = binary_entropy(0.9)  # = 0.469 bits
H_group2 = binary_entropy(0.3)  # = 0.881 bits
H_after = 0.5 * H_group1 + 0.5 * H_group2  # = 0.675 bits

info_gain = H_before - H_after  # = 0.206 bits
print(f"Information gain: {info_gain:.3f} bits")
```

### Exercise 4: Genre Prediction Entropy

**Problem**: Your genre classifier outputs $[0.4, 0.3, 0.2, 0.1]$ for a given song. Compute the entropy in bits, compare it to the maximum possible, and decide whether this prediction is "confident enough" to act on (threshold: normalized entropy below 0.5).

**Solution**:
```python
probs = [0.4, 0.3, 0.2, 0.1]
H = entropy(probs, base='2')       # = 1.846 bits
H_max = np.log2(4)                  # = 2.0 bits
normalized = H / H_max              # = 0.923

print(f"Entropy: {H:.3f} bits")
print(f"Normalized: {normalized:.3f}")
print(f"Confident enough? {'Yes' if normalized < 0.5 else 'No'}")
# Normalized entropy 0.923 > 0.5 — NOT confident enough.
# The model is almost as uncertain as random guessing.
```

## Summary

- **Shannon entropy** is the expected self-information: $H(X) = -\sum P(x) \log P(x)$
- **Interpretation**: Average uncertainty, average surprise, expected information content
- **Maximum entropy** is achieved by uniform distributions: $H_{max} = \log n$
- **Zero entropy** means complete certainty — only one outcome is possible
- **Binary entropy** peaks at $p=0.5$ with $H=1$ bit
- **Properties**: Non-negative, maximized by uniform, additive for independent variables, chain rule
- **SWE connections**: Compression lower bounds, decision tree splits, password strength, RNG quality
- **ML applications**: Decision trees (information gain), model confidence (softmax entropy), exploration in RL (entropy bonus), active learning (query high-entropy samples)
- **Units**: Bits (log base 2) or nats (natural log)

**Genre prediction takeaway**: a model that outputs uniform $[0.25, 0.25, 0.25, 0.25]$ has maximum entropy (2 bits) and has learned nothing. A confident model outputting $[0.9, 0.05, 0.03, 0.02]$ has low entropy (~0.6 bits) and is making a strong, actionable prediction.

## What Comes Next

Entropy measures uncertainty within one distribution. But how do you compare two distributions? Cross-entropy measures how well distribution $Q$ approximates distribution $P$ — and it turns out to be the loss function you've been minimizing every time you train a classifier.
