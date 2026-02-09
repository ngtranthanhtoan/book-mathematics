# Chapter 4: Exponentials and Logarithms

Softmax, cross-entropy loss, learning rate decay, sigmoid activation — they all use exp() and log(). These two functions are everywhere in ML because they convert between additive and multiplicative worlds. Master them, and half of ML's formulas suddenly make sense.

---

**Building On** — Polynomials give you curves. But some growth is faster than any polynomial — exponential growth. And its inverse, the logarithm, is the single most important function in information theory and ML.

---

## The ML Problem: Why Does Cross-Entropy Use log()?

Here is your running example for this chapter. Cross-entropy loss is the standard loss function for classification:

$$L = -\sum_i y_i \log(\hat{y}_i)$$

Why the log? Imagine you are training a model on 10,000 data points. Maximum likelihood says: find parameters that maximize the product of all predicted probabilities:

$$P(data|\theta) = \hat{y}_1 \times \hat{y}_2 \times \cdots \times \hat{y}_{10000}$$

Each $\hat{y}_i$ is a probability between 0 and 1. Multiply 10,000 of them together and you get a number so astronomically small that your float64 rounds it to zero. Training is dead.

But take the log of both sides:

$$\log P(data|\theta) = \log(\hat{y}_1) + \log(\hat{y}_2) + \cdots + \log(\hat{y}_{10000})$$

The product becomes a sum. No underflow. Gradients flow cleanly. That single property — **log turns products into sums** — is the reason log appears in almost every ML loss function.

Negate it (because we minimize loss, not maximize likelihood), and you get cross-entropy:

$$L = -\sum_i y_i \log(\hat{y}_i)$$

Keep this example in your head. Every rule you learn in this chapter connects back to it.

---

## You Already Know This

You have used exponentials and logarithms your entire career. You just did not call them that.

**Exponentials = compound growth.** Ever calculated compound interest? `balance = principal * (1 + rate)^years` — that is an exponential function. Exponential backoff in your retry logic? `wait_time = base_delay * 2^attempt` — same thing.

**Logarithms = "how many doublings?"** When you say "binary search is O(log n)," you are answering: "how many times do I halve n before I reach 1?" That is exactly what $\log_2(n)$ computes. When you say "merge sort is O(n log n)," the log is counting recursion depth.

**log(a * b) = log(a) + log(b)** — this is why we use log-probabilities in ML. Multiplying thousands of tiny probabilities causes underflow. Adding their logs does not. You saw this in the cross-entropy example above.

**Softmax** — your neural network outputs raw scores (logits). Softmax applies exp() to each one, then normalizes. The result: valid probabilities that sum to 1. The exp() ensures everything is positive and amplifies differences between scores.

```
SWE Concept               →  Math Equivalent
─────────────────────────────────────────────────────
Compound interest          →  Exponential growth: N(t) = N₀ · (1+r)^t
Exponential backoff        →  Powers of 2: delay = base · 2^attempt
O(log n) complexity        →  Logarithm: "how many halvings?"
Log-probabilities          →  log(a·b) = log(a) + log(b)
Softmax layer              →  exp(zᵢ) / Σ exp(zⱼ)
```

---

## What Are Exponentials and Logarithms?

They are inverses of each other, like multiplication and division.

The **exponential** $b^x$ asks: "what do I get when I multiply $b$ by itself $x$ times?"

The **logarithm** $\log_b(y)$ asks the reverse: "how many times must I multiply $b$ by itself to get $y$?"

$$2^3 = 8 \quad \iff \quad \log_2(8) = 3$$

Read it both ways: "2 to the power 3 is 8" and "log base 2 of 8 is 3."

```
  Exponential: base^exponent = result
                 2^3         = 8

  Logarithm:   log_base(result) = exponent
                log_2(8)         = 3

  They undo each other:
    log_2(2^x) = x       (log undoes exp)
    2^(log_2(x)) = x     (exp undoes log)
```

---

## The Exponential Function

An exponential function has the form:

$$f(x) = b^x$$

where $b > 0$ and $b \neq 1$ (the **base**).

**Key properties:**
- Domain: all real numbers ($x$ can be anything)
- Range: positive real numbers only ($b^x > 0$ always — this is why softmax outputs are always positive)
- $b^0 = 1$ for any valid base
- $b^1 = b$

### Exponential Growth vs. Decay

When the base is greater than 1, you get **growth**. When between 0 and 1, you get **decay**.

```
  y                          y = 2^x (growth)
  |                        .
  |                      .
  |                   .
  |                .
  |            .
  |        .
  |     .
  |  .  .  .  .  .  .       y = (0.5)^x (decay)
  | .                  .  .  .  .
  +─────────────────────────── x
        Both pass through (0, 1)
```

**Growth** ($b > 1$): Doubles, triples, ... with each unit of $x$. Think: viral spread, unchecked model parameter growth.

**Decay** ($0 < b < 1$): Halves, thirds, ... with each unit of $x$. Think: learning rate decay, radioactive decay, exponential moving average weight on old data.

---

## The Special Number e

$$e = \lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n \approx 2.71828...$$

You will see $e$ everywhere. Why is this particular number the default base in ML and calculus?

**Because the derivative of $e^x$ is itself: $\frac{d}{dx}e^x = e^x$.**

No other base has this property. This makes calculus with $e^x$ clean — no extra constants floating around. When you see `np.exp()` in ML code, it is computing $e^x$.

Other reasons $e$ matters:
- It arises naturally from continuous compounding (compounding interest infinitely often)
- The natural logarithm $\ln(x) = \log_e(x)$ has derivative $1/x$ — the simplest possible
- The normal distribution, softmax, sigmoid — all built on $e^x$

---

## Laws of Exponents

These rules let you simplify exponential expressions. You will use them constantly when manipulating ML formulas.

| Law | Formula | Example | Why You Care |
|-----|---------|---------|-------------|
| Product | $b^m \cdot b^n = b^{m+n}$ | $2^3 \cdot 2^4 = 2^7 = 128$ | Combining exponential terms |
| Quotient | $\frac{b^m}{b^n} = b^{m-n}$ | $\frac{3^5}{3^2} = 3^3 = 27$ | Simplifying ratios |
| Power | $(b^m)^n = b^{mn}$ | $(2^3)^2 = 2^6 = 64$ | Nested exponentials |
| Zero | $b^0 = 1$ | $5^0 = 1$ | Base case |
| Negative | $b^{-n} = \frac{1}{b^n}$ | $2^{-3} = \frac{1}{8}$ | Flipping growth to decay |
| Fractional | $b^{1/n} = \sqrt[n]{b}$ | $8^{1/3} = 2$ | Roots as exponents |
| Distribution | $(ab)^n = a^n b^n$ | $(2 \cdot 3)^2 = 4 \cdot 9 = 36$ | Distributing over products |

**Quick exercise** — Simplify $\frac{e^{3x} \cdot e^{2x}}{e^{x}}$ using these rules:

$$\frac{e^{3x} \cdot e^{2x}}{e^{x}} = \frac{e^{3x+2x}}{e^x} = \frac{e^{5x}}{e^x} = e^{5x-x} = e^{4x}$$

Product rule, then quotient rule. That is all there is to it.

---

## Logarithms

The **logarithm** base $b$ of $x$ is the exponent you need:

$$\log_b(x) = y \quad \iff \quad b^y = x$$

Think of it as asking: "$b$ to WHAT POWER gives me $x$?"

### Common Bases

| Notation | Name | Used In |
|----------|------|---------|
| $\log_{10}(x)$ | Common logarithm | Order-of-magnitude estimation |
| $\ln(x) = \log_e(x)$ | Natural logarithm | Calculus, ML (this is `np.log()`) |
| $\log_2(x)$ | Binary logarithm | CS complexity, information theory (bits) |

**Heads up on notation:** In pure math, $\log$ usually means $\ln$ (natural log). In CS, $\log$ often means $\log_2$. In ML papers, $\log$ almost always means $\ln$. When in doubt, check the context or be explicit.

### The Logarithm Curve

```
  y
  |
  3|            .  .  .  .  .  .  .   y = log₂(x)
  |         .
  2|      .
  |    .
  1|  .
  |.
  +──────────────────────────────── x
  |1  2     4        8           16
 -1| (log is negative for x < 1)
  |
```

Key visual properties:
- Passes through (1, 0) — because $\log_b(1) = 0$ for any base
- Passes through (b, 1) — because $\log_b(b) = 1$
- Grows slowly — logarithm is the "opposite" of exponential explosion
- Undefined for $x \leq 0$ — you cannot take $\log$ of zero or negative numbers

---

## Logarithmic Identities

These are the rules that make logarithms so powerful. The first one is the most important rule in all of ML mathematics.

| Identity | Formula | Example |
|----------|---------|---------|
| **Product** | $\log_b(xy) = \log_b(x) + \log_b(y)$ | $\log_2(8 \cdot 4) = 3 + 2 = 5$ |
| Quotient | $\log_b(x/y) = \log_b(x) - \log_b(y)$ | $\log_{10}(100/10) = 2 - 1 = 1$ |
| Power | $\log_b(x^n) = n \cdot \log_b(x)$ | $\ln(e^5) = 5 \cdot \ln(e) = 5$ |
| Identity | $\log_b(b) = 1$ | $\log_2(2) = 1$ |
| One | $\log_b(1) = 0$ | $\ln(1) = 0$ |
| Inverse | $b^{\log_b(x)} = x$ | $e^{\ln(5)} = 5$ |
| Inverse | $\log_b(b^x) = x$ | $\log_{10}(10^3) = 3$ |

### The Product Rule Is Everything

Go back to our running example. You had a product of 10,000 probabilities:

$$P = \hat{y}_1 \times \hat{y}_2 \times \cdots \times \hat{y}_{10000}$$

Apply the product rule:

$$\log(P) = \log(\hat{y}_1) + \log(\hat{y}_2) + \cdots + \log(\hat{y}_{10000})$$

Product becomes sum. That is the product rule doing all the heavy lifting. Every time you see `log` in an ML loss function, this rule is the reason it is there.

### Change of Base Formula

You can convert between any two logarithm bases:

$$\log_b(x) = \frac{\log_c(x)}{\log_c(b)} = \frac{\ln(x)}{\ln(b)}$$

**Example**: What is $\log_2(10)$?

$$\log_2(10) = \frac{\ln(10)}{\ln(2)} = \frac{2.303}{0.693} \approx 3.322$$

This tells you: you need about 3.32 doublings to reach 10. Or equivalently: binary search on 10 items takes about 3.32 comparisons (round up to 4).

---

## Exponential Growth and Decay

The general model:

$$N(t) = N_0 \cdot e^{kt}$$

- $N_0$: initial value
- $k > 0$: growth (population, viral spread, unbounded gradient norms)
- $k < 0$: decay (learning rate schedules, weight decay, radioactive decay)

**Half-life / doubling time:**

$$t_{1/2} = \frac{\ln(2)}{|k|}$$

This formula falls straight out of the log rules. Set $N(t) = 2N_0$, take log of both sides, solve for $t$.

---

## Softmax: Exponentials in Action

Here is where exponentials shine in ML. Your neural network outputs raw scores (logits) like $[2.0, 1.0, 0.1]$. These are not probabilities — they can be negative, they do not sum to 1. Softmax fixes that:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

```
Softmax Transformation
═══════════════════════════════════════════════════

  Logits (raw scores)         Probabilities
  ┌──────────────────┐        ┌──────────────────┐
  │ z₁ = 2.0        │──exp──▶│ e^2.0 = 7.389    │
  │ z₂ = 1.0        │──exp──▶│ e^1.0 = 2.718    │
  │ z₃ = 0.1        │──exp──▶│ e^0.1 = 1.105    │
  └──────────────────┘        └──────────────────┘
                                   │ divide by sum
                                   │ (7.389 + 2.718 + 1.105 = 11.212)
                                   ▼
                              ┌──────────────────┐
                              │ p₁ = 0.659       │
                              │ p₂ = 0.242       │
                              │ p₃ = 0.099       │
                              │ ─────────        │
                              │ sum = 1.000      │
                              └──────────────────┘
```

Why exp() specifically?
1. **Always positive**: $e^x > 0$ for any $x$, so all probabilities are positive
2. **Preserves ordering**: Larger logit = larger probability
3. **Amplifies differences**: exp() is convex, so gaps between large logits get magnified
4. **Clean gradients**: The derivative of softmax has a beautiful form thanks to $\frac{d}{dx}e^x = e^x$

---

## Common Mistakes

### The #1 Mistake: Confusing Sum and Product

> **log(a + b) != log(a) + log(b)**

Read that again. The log of a SUM is NOT the sum of logs. Only the log of a PRODUCT is the sum of logs:

$$\log(a \times b) = \log(a) + \log(b) \quad \checkmark$$
$$\log(a + b) \neq \log(a) + \log(b) \quad \times$$

There is no clean simplification for $\log(a + b)$. This trips up even experienced practitioners when deriving gradients.

### Other Pitfalls

**log(0) is undefined.** In code, `np.log(0)` gives `-inf`. Always clip: `np.log(np.clip(x, 1e-15, 1))`.

**Overflow with exp().** $e^{1000}$ overflows float64. In softmax, subtract the max first: `exp(x - max(x))`. This does not change the result (the shift cancels out in the ratio) but prevents overflow.

**Base confusion.** `np.log()` is $\ln$ (base $e$). `np.log2()` is base 2. `np.log10()` is base 10. ML papers that write "log" almost always mean $\ln$.

**Sigmoid saturation.** The sigmoid $\sigma(x) = \frac{1}{1 + e^{-x}}$ outputs values near 0 or 1 for large $|x|$. Gradients vanish there. This is why ReLU replaced sigmoid in hidden layers — but sigmoid is still used for output layers in binary classification.

---

## Code: Everything In Action

```python
import numpy as np
from typing import Callable

# ============================================================
# BASICS: exp() and log() are inverses
# ============================================================
print("=== Basics ===")
print(f"e = {np.e:.5f}")
print(f"e^2 = {np.exp(2):.5f}")
print(f"ln(e) = {np.log(np.e):.5f}")       # 1.0 — they undo each other
print(f"log10(100) = {np.log10(100):.1f}")  # 2.0 — "10 to what power = 100?"
print(f"log2(8) = {np.log2(8):.1f}")        # 3.0 — "how many doublings to 8?"

# ============================================================
# EXPONENT RULES in action
# ============================================================
print("\n=== Laws of Exponents ===")
b, m, n = 2, 3, 4

# Product rule: b^m * b^n = b^(m+n)
print(f"{b}^{m} * {b}^{n} = {b**m * b**n} = {b}^{m+n} = {b**(m+n)}")

# Quotient rule: b^m / b^n = b^(m-n)
print(f"{b}^{m} / {b}^{n} = {b**m / b**n} = {b}^{m-n} = {b**(m-n)}")

# Power rule: (b^m)^n = b^(m*n)
print(f"({b}^{m})^{n} = {(b**m)**n} = {b}^{m*n} = {b**(m*n)}")

# ============================================================
# LOG RULES: product rule is the star
# ============================================================
print("\n=== Logarithmic Identities ===")
x, y = 8, 4

# THE KEY RULE: log(x*y) = log(x) + log(y)
print(f"log2({x}*{y}) = log2({x*y}) = {np.log2(x*y):.4f}")
print(f"log2({x}) + log2({y}) = {np.log2(x)} + {np.log2(y)} = {np.log2(x) + np.log2(y)}")

# Power rule: log(x^n) = n * log(x)
print(f"log2({x}^3) = {np.log2(x**3)}")
print(f"3 * log2({x}) = {3 * np.log2(x)}")

# ============================================================
# CHANGE OF BASE
# ============================================================
def log_base(x: float, base: float) -> float:
    """Calculate logarithm with arbitrary base using change of base formula."""
    return np.log(x) / np.log(base)

print("\n=== Change of Base ===")
print(f"log_5(125) = {log_base(125, 5):.1f}")  # 3.0
print(f"log_3(81) = {log_base(81, 3):.1f}")    # 4.0

# ============================================================
# SOFTMAX: the bridge from logits to probabilities
# ============================================================
def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.

    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Subtract max for numerical stability (does not change the result
    because the shift cancels in the numerator/denominator ratio).
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"\n=== Softmax ===")
print(f"Logits:        {logits}")
print(f"Probabilities: {probs}")
print(f"Sum:           {np.sum(probs):.4f}")  # 1.0000

# ============================================================
# CROSS-ENTROPY LOSS: log turns products into sums
# ============================================================
def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray,
                       epsilon: float = 1e-15) -> float:
    """
    Compute cross-entropy loss: L = -sum(y_true * log(y_pred))

    This IS the running example from the chapter.
    epsilon prevents log(0) = -inf.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

# True label is class 0 (one-hot encoded)
y_true = np.array([1, 0, 0])

# Good prediction: model is confident and correct
y_pred_good = np.array([0.9, 0.05, 0.05])
loss_good = cross_entropy_loss(y_true, y_pred_good)

# Bad prediction: model is confident and WRONG
y_pred_bad = np.array([0.1, 0.5, 0.4])
loss_bad = cross_entropy_loss(y_true, y_pred_bad)

print(f"\n=== Cross-Entropy Loss ===")
print(f"Good prediction loss: {loss_good:.4f}")   # Small — log(0.9) is close to 0
print(f"Bad prediction loss:  {loss_bad:.4f}")     # Large — log(0.1) is very negative

# ============================================================
# WHY LOG-PROBABILITIES MATTER: the underflow problem
# ============================================================
print("\n=== Log-Probabilities Prevent Underflow ===")
# Imagine 10,000 data points, each with probability 0.99
n_samples = 10000
prob = 0.99

# Direct product: UNDERFLOWS TO ZERO
direct_product = prob ** n_samples
print(f"Direct product of {n_samples} probabilities: {direct_product}")  # 0.0 or tiny

# Log-sum: works perfectly
log_sum = n_samples * np.log(prob)
print(f"Sum of {n_samples} log-probs: {log_sum:.4f}")  # -100.5, perfectly fine

# ============================================================
# LOG-SUM-EXP TRICK: stable computation
# ============================================================
def log_sum_exp(x: np.ndarray) -> float:
    """
    Compute log(sum(exp(x))) without overflow.

    log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))

    Subtracting max prevents exp() from overflowing.
    """
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

large_values = np.array([1000, 1001, 1002])
print(f"\n=== Log-Sum-Exp Trick ===")
print(f"Large values: {large_values}")
print(f"log_sum_exp:  {log_sum_exp(large_values):.4f}")
# Direct np.log(np.sum(np.exp(large_values))) would give inf

# ============================================================
# SIGMOID: binary classification's output function
# ============================================================
def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation: sigma(x) = 1 / (1 + exp(-x))

    Maps any real number to (0, 1). Used as the output layer
    for binary classification.
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

print(f"\n=== Sigmoid Function ===")
test_values = np.array([-5, -1, 0, 1, 5])
print(f"x values:   {test_values}")
print(f"sigmoid(x): {sigmoid(test_values)}")
# Note: sigmoid(0) = 0.5 (decision boundary)
# sigmoid saturates near 0 and 1 for large |x|

# ============================================================
# EXPONENTIAL MOVING AVERAGE: smoothing in optimizers
# ============================================================
def exponential_moving_average(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}

    Used in Adam optimizer (momentum), batch normalization
    running statistics, and TensorBoard smoothing.
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]

    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]

    return ema

np.random.seed(42)
noisy_signal = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.3
smoothed = exponential_moving_average(noisy_signal, alpha=0.1)

print(f"\n=== Exponential Moving Average ===")
print(f"Original variance: {np.var(noisy_signal):.4f}")
print(f"Smoothed variance: {np.var(smoothed):.4f}")

# ============================================================
# LEARNING RATE DECAY: exponential schedule
# ============================================================
def exponential_decay(initial_lr: float, decay_rate: float,
                      epoch: int, decay_steps: int = 1) -> float:
    """
    lr = initial_lr * decay_rate^(epoch / decay_steps)

    The most common learning rate schedule. decay_rate < 1
    means the learning rate shrinks exponentially.
    """
    return initial_lr * (decay_rate ** (epoch / decay_steps))

print(f"\n=== Learning Rate Decay ===")
initial_lr = 0.1
for epoch in [0, 10, 20, 50, 100]:
    lr = exponential_decay(initial_lr, decay_rate=0.95, epoch=epoch)
    print(f"Epoch {epoch:3d}: lr = {lr:.6f}")

# ============================================================
# INFORMATION ENTROPY: logarithms measure surprise
# ============================================================
def entropy(probabilities: np.ndarray, epsilon: float = 1e-15) -> float:
    """
    Shannon entropy: H(p) = -sum(p * log2(p))

    Maximum when all outcomes equally likely (maximum uncertainty).
    Minimum (zero) when one outcome is certain.
    """
    p = np.clip(probabilities, epsilon, 1)
    return -np.sum(p * np.log2(p))

uniform_dist = np.array([0.25, 0.25, 0.25, 0.25])
peaked_dist = np.array([0.97, 0.01, 0.01, 0.01])

print(f"\n=== Information Entropy ===")
print(f"Uniform distribution: {entropy(uniform_dist):.4f} bits (maximum uncertainty)")
print(f"Peaked distribution:  {entropy(peaked_dist):.4f} bits (nearly certain)")
```

---

## Numerical Stability: The Practical Rules

You will write these patterns over and over. Burn them in.

```python
# ─── SOFTMAX ───────────────────────────────────────────
# BAD: exp(1000) overflows to inf
probs = np.exp(logits) / np.sum(np.exp(logits))

# GOOD: subtract max first (result is identical, no overflow)
shifted = logits - np.max(logits)
probs = np.exp(shifted) / np.sum(np.exp(shifted))

# ─── CROSS-ENTROPY ────────────────────────────────────
# BAD: log(0) = -inf
loss = -np.log(predictions)

# GOOD: clip to avoid log(0)
loss = -np.log(np.clip(predictions, 1e-15, 1))

# ─── BEST: combine softmax + cross-entropy ────────────
# Work directly with logits, never compute probabilities
def stable_cross_entropy(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Numerically stable cross-entropy directly from logits."""
    logits_stable = logits - np.max(logits)
    log_sum_exp = np.log(np.sum(np.exp(logits_stable)))
    log_softmax = logits_stable - log_sum_exp
    return -np.sum(y_true * log_softmax)
```

Why does subtracting max work for softmax? Because:

$$\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_j e^{z_j} \cdot e^{-c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

The $e^{-c}$ cancels out. You get the exact same answer, but no term overflows because the largest exponent is now 0.

---

## ML Applications Summary

| ML Concept | Uses | Why |
|-----------|------|-----|
| **Softmax** | $e^{z_i} / \sum e^{z_j}$ | Turns logits into probabilities |
| **Cross-entropy loss** | $-\sum y_i \log(\hat{y}_i)$ | Log turns likelihood product into sum |
| **Sigmoid** | $1 / (1 + e^{-x})$ | Maps reals to (0,1) for binary classification |
| **Log-likelihood** | $\sum_i \log P(x_i \| \theta)$ | Product of probabilities becomes sum |
| **Learning rate decay** | $lr_0 \cdot \gamma^{epoch}$ | Exponential shrinkage of step size |
| **Batch normalization** | $(1-\alpha)\mu_{old} + \alpha\mu_{new}$ | Exponential moving average of statistics |
| **Entropy** | $-\sum p_i \log p_i$ | Measures uncertainty in a distribution |
| **KL divergence** | $\sum p_i \log(p_i / q_i)$ | Measures difference between distributions |

---

## Exercises

### Exercise 1: Simplify Using Exponent Laws

Simplify: $\frac{e^{3x} \cdot e^{2x}}{e^{x}}$

**Solution:**

Using the laws of exponents:

$$\frac{e^{3x} \cdot e^{2x}}{e^{x}} = \frac{e^{3x+2x}}{e^x} = \frac{e^{5x}}{e^x} = e^{5x-x} = e^{4x}$$

### Exercise 2: Solve for x

Solve: $5^{2x-1} = 125$

**Solution:**

Recognize that $125 = 5^3$:

$$5^{2x-1} = 5^3$$

Since the bases are equal, the exponents must be equal:

$$2x - 1 = 3 \implies 2x = 4 \implies x = 2$$

### Exercise 3: Implement Stable Cross-Entropy From Logits

Write a numerically stable cross-entropy loss function that works directly from logits (not probabilities), combining the softmax and cross-entropy steps.

**Solution:**

```python
import numpy as np

def stable_cross_entropy(y_true: np.ndarray, logits: np.ndarray) -> float:
    """
    Numerically stable cross-entropy directly from logits.

    Combines softmax and cross-entropy to avoid:
    1. exp() overflow (by subtracting max)
    2. log(0) (by never computing probabilities explicitly)

    This is what PyTorch's CrossEntropyLoss does internally.
    """
    # Shift logits for numerical stability
    logits_stable = logits - np.max(logits)

    # log_softmax = logits - log(sum(exp(logits)))
    log_sum_exp = np.log(np.sum(np.exp(logits_stable)))
    log_softmax = logits_stable - log_sum_exp

    # Cross-entropy: -sum(y_true * log_softmax)
    return -np.sum(y_true * log_softmax)

# Test
y_true = np.array([1, 0, 0])
logits = np.array([10, 2, 1])  # High confidence on correct class
loss = stable_cross_entropy(y_true, logits)
print(f"Loss: {loss:.6f}")  # Should be small

# Even works with extreme logits
extreme_logits = np.array([1000, 1, 0])  # Would overflow without stability trick
loss_extreme = stable_cross_entropy(y_true, extreme_logits)
print(f"Extreme loss: {loss_extreme:.6f}")  # Still works!
```

### Exercise 4: Why Log-Probabilities?

You are training a language model. For a sentence of 50 words, the model assigns probability $P(w_i | w_{<i})$ to each word. The sentence probability is:

$$P(\text{sentence}) = \prod_{i=1}^{50} P(w_i | w_{<i})$$

If each word probability averages 0.1, what is the sentence probability computed directly? What happens in float64? What is the log-probability?

**Solution:**

Direct: $0.1^{50} = 10^{-50}$. Float64 can handle this (its minimum is about $10^{-308}$), but for longer sequences or smaller probabilities, you hit underflow fast. A 500-word document with average probability 0.01? That is $0.01^{500} = 10^{-1000}$ — way beyond float64.

Log-probability: $50 \times \log(0.1) = 50 \times (-2.302585) = -115.129$. A perfectly normal float64 number. No underflow, no precision loss, and you can compare sentence probabilities by comparing their log-probs.

---

## Summary

- **Exponential functions** $b^x$ model multiplicative growth and decay — and appear in softmax, sigmoid, and learning rate schedules

- **Laws of exponents** let you manipulate these expressions:
  - $b^m \cdot b^n = b^{m+n}$ (product rule)
  - $(b^m)^n = b^{mn}$ (power rule)
  - $b^{-n} = 1/b^n$ (negative exponent)

- **The natural base** $e \approx 2.718$ is special because $\frac{d}{dx}e^x = e^x$ — the derivative equals the function itself

- **Logarithms** are the inverse of exponentials: $\log_b(x) = y \iff b^y = x$

- **The product rule** $\log(xy) = \log(x) + \log(y)$ is the single most important identity — it is the reason cross-entropy loss, log-likelihood, and log-probabilities exist

- **Common mistake**: $\log(a + b) \neq \log(a) + \log(b)$ — only products become sums, never plain sums

- **Change of base**: $\log_b(x) = \frac{\ln(x)}{\ln(b)}$

- **Numerical stability is non-negotiable**: subtract max in softmax, clip before log, use log-sum-exp for large values

- **Running example recap**: Cross-entropy loss $-\sum y_i \log(\hat{y}_i)$ uses log to convert a product of likelihoods into a sum. Without log, you would multiply 10,000 probabilities together and get zero.

---

**What's Next** — You can express growth, decay, and relationships. But how do you express CONSTRAINTS? Inequalities define the boundaries of optimization — feasible regions, margin constraints, and regularization bounds.

Next: [Chapter 5: Inequalities](./05-inequalities.md) -->
