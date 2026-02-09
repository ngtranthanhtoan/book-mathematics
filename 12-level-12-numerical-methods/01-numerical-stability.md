# Numerical Stability

## Building On

Throughout this book, you've built a formidable mathematical toolkit — linear algebra for representing data, calculus for computing gradients, probability for reasoning under uncertainty, and optimization theory for finding solutions. All that math assumed one luxury: infinite precision. Real numbers with infinite decimal places. Perfect arithmetic.

Now we face an uncomfortable reality: **computers can't actually do math.**

Not perfectly, anyway. Every floating-point operation introduces a tiny error. One error is harmless. A billion of them, chained together during neural network training? That's how you end up staring at `loss: NaN` at 3 AM, wondering why your model that was training fine an hour ago just decided to produce garbage.

This chapter is about understanding *why* that happens, *how* to see it coming, and *what* to do about it. Think of it as defensive programming — but for arithmetic.

---

## The 3 AM NaN: A Horror Story

Let me paint a picture you'll probably recognize.

You've spent two weeks building a custom transformer model. The architecture is clean. The data pipeline is solid. You kick off training, watch the loss drop beautifully for the first 10,000 steps, go to bed feeling good, and wake up to this:

```
Step 10,000: loss = 2.341
Step 10,500: loss = 2.298
Step 11,000: loss = 2.187
Step 11,500: loss = nan
Step 12,000: loss = nan
Step 12,000: loss = nan
...forever nan...
```

Your first instinct? Check the data. Check the learning rate. Add gradient clipping. Restart from a checkpoint. But none of that helps, because the real bug isn't in your logic — it's in your *arithmetic*. Somewhere deep inside the computation, a number got too big, or too small, or two nearly-equal numbers got subtracted, and the result was floating-point nonsense that poisoned everything downstream.

Let's figure out why.

---

## Why Computers Can't Do Math (And How They Fake It)

### The Lie of `0.1 + 0.2`

Before we talk about ML, let's start with something you've definitely encountered:

```python
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

If you're a senior engineer, you've seen this before. You might have even explained it to a junior dev. But have you thought about what it means when this tiny imprecision is multiplied by *billions* of operations inside a neural network?

> **You Already Know This**: You've debugged floating-point comparison bugs. You've used `abs(a - b) < epsilon` instead of `a == b`. You've seen currency calculations go wrong with floats. You already know computers are bad at decimal arithmetic — this chapter explains *exactly how bad*, and what it means for ML specifically.

### How Floating-Point Numbers Actually Work

A computer stores every number in a fixed-width binary format. The dominant standard is IEEE 754. Here's what a 64-bit float (float64) looks like under the hood:

```
  IEEE 754 Double Precision (64 bits)
  ┌───────┬──────────────┬─────────────────────────────────────────────────────┐
  │ Sign  │   Exponent   │                    Mantissa                        │
  │ 1 bit │   11 bits    │                    52 bits                         │
  └───────┴──────────────┴─────────────────────────────────────────────────────┘
     ↓          ↓                            ↓
   + or -    Scale         Precision (significant digits)
             (2^e)         (~15-17 decimal digits)
```

The number is reconstructed as:

$$x = (-1)^{\text{sign}} \times (1 + \text{mantissa}) \times 2^{\text{exponent} - \text{bias}}$$

**Translation**: Think of it like scientific notation. The mantissa is the significant digits (`1.23456...`), the exponent is the power of 2 that scales it, and the sign is whether it's positive or negative. The key constraint is that the mantissa has *finite* bits — only 52 of them for float64, and only 23 for float32.

Here's what that looks like for float32 (the default in most ML training):

```
  IEEE 754 Single Precision (32 bits)
  ┌───────┬──────────┬───────────────────────────────┐
  │ Sign  │ Exponent │          Mantissa             │
  │ 1 bit │  8 bits  │          23 bits              │
  └───────┴──────────┴───────────────────────────────┘
     ↓         ↓                  ↓
   + or -    Scale      Precision (~7 decimal digits)
             (2^e)      ← That's it. 7 digits. That's all you get.
```

And float16, which mixed-precision training uses:

```
  IEEE 754 Half Precision (16 bits)
  ┌───────┬──────────┬─────────────┐
  │ Sign  │ Exponent │  Mantissa   │
  │ 1 bit │  5 bits  │  10 bits    │
  └───────┴──────────┴─────────────┘
     ↓         ↓            ↓
   + or -    Scale    ~3 decimal digits
             (2^e)    ← Yikes.
```

Three decimal digits. That means in float16, the numbers `1024` and `1025` are *the same number*. Let that sink in.

### The Number Line Is Not Uniform

Here's something critical that catches people off guard. Floating-point numbers are not uniformly spaced. They're packed densely near zero and spread far apart for large values:

```
  Float density on the number line:

  Near zero: numbers packed tight
  ||||||||||||||||||||||||||||||||           ... much further apart ...
  0     0.001   0.002  0.003

  Near 1.0: decent spacing
  |    |    |    |    |    |    |    |
  0.99 0.995 1.0 1.005 1.01 1.015 1.02

  Near 1,000,000: wide gaps
  |              |              |              |
  999,999    1,000,000     1,000,001     1,000,002
  (but 999,999.5 doesn't exist in float32!)

  Near 10^38 (float32 max range):
  |                                              |
  huge_number                      huge_number + 10^31
  (you've lost ALL decimal precision)
```

**Translation**: Floats give you *relative* precision, not *absolute* precision. You always get roughly the same number of significant digits, regardless of the magnitude. Near zero, that means you can distinguish incredibly tiny differences. Near $10^{38}$, you can't even distinguish numbers that differ by trillions.

This is why large logits in a softmax are dangerous — you're operating in a region where floating-point numbers are sparse and imprecise.

---

## Machine Epsilon: The Fundamental Speed Limit of Precision

Machine epsilon ($\epsilon_{\text{machine}}$) is the smallest number such that:

$$1 + \epsilon_{\text{machine}} \neq 1$$

in floating-point arithmetic. Any number smaller than this, when added to 1, gets rounded away to nothing.

| Type    | Bits | $\epsilon_{\text{machine}}$ | Decimal Digits |
|---------|------|-----------------------------|----------------|
| float16 | 16   | $2^{-10} \approx 9.77 \times 10^{-4}$ | ~3 |
| float32 | 32   | $2^{-23} \approx 1.19 \times 10^{-7}$ | ~7 |
| float64 | 64   | $2^{-52} \approx 2.22 \times 10^{-16}$ | ~16 |

When we store a real number $x$ as a floating-point number $\text{fl}(x)$, the relative error is bounded by:

$$\left|\frac{\text{fl}(x) - x}{x}\right| \leq \frac{\epsilon_{\text{machine}}}{2}$$

**Translation**: Every single floating-point operation introduces a relative error of up to half a machine epsilon. One operation? Negligible. A billion operations chained together? That's where things get interesting — and by "interesting," I mean "your model outputs NaN."

Let's see this in action:

```python
import numpy as np

# Machine epsilon for different types
print("Machine epsilon values:")
print(f"  float16: {np.finfo(np.float16).eps}")   # 0.000977
print(f"  float32: {np.finfo(np.float32).eps}")   # 1.1920929e-07
print(f"  float64: {np.finfo(np.float64).eps}")   # 2.220446049250313e-16

# The classic demonstration
print(f"\n0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3: {0.1 + 0.2 == 0.3}")
print(f"np.isclose(0.1 + 0.2, 0.3): {np.isclose(0.1 + 0.2, 0.3)}")

# Epsilon in action: adding something smaller than epsilon to 1.0
print(f"\n1.0 + 1e-16 == 1.0: {1.0 + 1e-16 == 1.0}")   # True!  Absorbed!
print(f"1.0 + 1e-15 == 1.0: {1.0 + 1e-15 == 1.0}")     # False. Just barely survived.
```

> **Common Mistake**: Comparing floats with `==`. Always use `np.isclose()` or `torch.allclose()` in your ML code. If you've ever seen a unit test pass locally and fail in CI, floating-point comparison might have been the culprit.

---

## The Three Horsemen of Numerical Apocalypse

Floating-point errors manifest in three primary ways. Let's investigate each one like we'd investigate a production incident.

### Horseman 1: Overflow (Numbers Too Big)

Overflow happens when a computation produces a number larger than the type can represent. The result: `inf`. And `inf` in a gradient computation means your model is done for.

```
  The overflow cliff:

  float32 range:
  ◄─────────────────────────────────────────────────────────────►
  -3.4×10^38                   0                    3.4×10^38
                                                        │
                                                     ┌──┴──┐
                                                     │CLIFF│
                                                     └─────┘
                                                        ↓
                                                      +inf
                                                    (game over)
```

For float64, the limits are:
- **Maximum value**: $\approx 1.8 \times 10^{308}$
- Going beyond: the number becomes `inf`

For float32:
- **Maximum value**: $\approx 3.4 \times 10^{38}$
- This sounds big, but $e^{89} \approx 4.5 \times 10^{38}$ — just one exponentiation and you're over the edge

For float16:
- **Maximum value**: $\approx 6.5 \times 10^{4}$ (that's just 65,504!)
- **This is why mixed-precision training is tricky**

```python
import numpy as np

# Overflow in action
print("=== Overflow ===")
large = np.float64(1e308)
print(f"1e308 = {large}")
print(f"1e308 * 10 = {large * 10}")           # inf

# The ML-relevant case: exp() overflow
print(f"\nexp(709) in float64: {np.exp(709)}")     # ~8.2e307, fine
print(f"exp(710) in float64: {np.exp(710)}")       # inf!

print(f"\nexp(88) in float32: {np.float32(np.exp(88))}")   # ~1.6e38, fine
print(f"exp(89) in float32: {np.float32(np.exp(89))}")     # inf!

# float16 is even more fragile
print(f"\nfloat16 max: {np.finfo(np.float16).max}")        # 65504.0
```

**Where this bites you in ML**: The softmax function computes $e^{x_i}$. If any logit $x_i > 89$ (in float32), you get `inf`. Then `inf / inf = nan`. Then `nan` propagates through your entire model like a virus, destroying every parameter it touches.

### Horseman 2: Underflow (Numbers Too Small)

Underflow is the opposite — a number too close to zero silently becomes exactly `0`. This is arguably *more* dangerous than overflow, because it's silent. No `inf`, no `nan`, just a quiet zero where there should have been a tiny positive number.

```
  The underflow trap:

  float32 range near zero:
          actual math          what the computer stores
          ────────────         ──────────────────────────
          1e-38                1e-38 ✓ (smallest normal)
          1e-39                1e-39 ✓ (denormalized, losing precision)
          1e-45                1e-45 ✓ (smallest denorm, only 1 bit of precision)
          1e-46                0     ✗ GONE. Silently replaced with zero.
```

For float64:
- **Minimum positive normal**: $\approx 2.2 \times 10^{-308}$
- Numbers smaller than this become `0`

```python
import numpy as np

print("=== Underflow ===")
small = np.float64(1e-308)
print(f"1e-308 = {small}")
print(f"1e-308 / 1e10 = {small / 1e10}")   # Approaches underflow

# The ML-relevant case: multiplying many small probabilities
probs = np.array([0.01] * 100, dtype=np.float32)
product = np.prod(probs)
print(f"\nProduct of 100 probabilities of 0.01: {product}")  # 0.0 (underflow!)
# The true answer is 1e-200, but float32 can't represent it

# The fix: work in log space
log_product = np.sum(np.log(probs))
print(f"Log of product (via sum of logs): {log_product}")   # -460.5 (correct!)
```

**Where this bites you in ML**: Vanishing gradients. When you backpropagate through 50+ layers, you're multiplying many small numbers together. The gradient for early layers can underflow to exactly zero, which means those layers stop learning entirely. This is why ResNets (skip connections), batch normalization, and careful initialization schemes exist — they're all defenses against underflow.

> **You Already Know This**: You've dealt with multiplicative error accumulation before. If your microservice chain has 10 hops, and each hop has a 1% chance of adding latency, the end-to-end latency distribution is very different from any single hop. Underflow in gradients is the same phenomenon — many small multiplications compounding into something pathological.

### Horseman 3: Catastrophic Cancellation (Subtracting Nearly Equal Numbers)

This is the sneakiest one. When you subtract two numbers that are nearly equal, the relative error in the result *explodes*.

Here's why. Suppose $a = 1.0000001$ and $b = 1.0000000$. Both are stored with about 16 digits of precision (float64). Their difference is:

$$a - b = 0.0000001$$

That's $10^{-7}$. But here's the problem: if both $a$ and $b$ have a relative error of $\epsilon$ (say, $10^{-16}$), the relative error in $a - b$ is approximately:

$$\text{relative error in } (a - b) \approx \frac{|a| + |b|}{|a - b|} \cdot \epsilon \approx \frac{2}{10^{-7}} \cdot 10^{-16} = 2 \times 10^{-9}$$

**Translation**: We started with 16 digits of precision and ended up with about 9. We lost 7 digits of precision in a single subtraction. If the numbers are even closer together, you can lose *all* your precision in one operation.

```
  Catastrophic cancellation visualized:

  a = 1.00000 01234 56789  (16 digits of precision)
  b = 1.00000 01234 00000  (16 digits of precision)
      ─────────────────────
  a-b= 0.00000 00000 56789  Only 5 digits of precision left!
                     ↑↑↑↑↑
                     These are the ONLY meaningful digits.
                     The leading zeros ate all the precision.
```

```python
import numpy as np

print("=== Catastrophic Cancellation ===")

# Demonstration: computing (1+x) - 1 for small x
# Should return x, but cancellation destroys precision
for x_exp in range(-1, -18, -2):
    x = 10.0 ** x_exp
    result = (1.0 + x) - 1.0
    error = abs(result - x) / abs(x) if x != 0 else 0
    print(f"x = 1e{x_exp:3d}: (1+x)-1 = {result:.6e}, "
          f"relative error = {error:.2e}")

# Near machine epsilon, the result is total nonsense
```

**Where this bites you in ML**: The numerical gradient check. When you're verifying your custom backward pass, you compute:

$$\frac{\partial f}{\partial x} \approx \frac{f(x + h) - f(x - h)}{2h}$$

If $h$ is too small, $f(x+h)$ and $f(x-h)$ are nearly equal, and catastrophic cancellation destroys your gradient estimate. If $h$ is too large, the approximation is inaccurate. There's a Goldilocks zone around $h \approx \sqrt{\epsilon_{\text{machine}}} \approx 10^{-8}$ for float64.

> **Common Mistake**: Setting the step size $h$ too small in numerical gradient checks (like $h = 10^{-15}$) and then concluding your analytical gradient is wrong because the numerical estimate doesn't match. The numerical estimate was garbage due to catastrophic cancellation — your analytical gradient was probably fine.

---

## Case Study: The Softmax Disaster

Let's trace through a real numerical failure end-to-end. This is the single most common numerical stability bug in ML.

### The Naive Implementation

The softmax function converts a vector of logits into probabilities:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

Looks simple enough. Let's implement it:

```python
import numpy as np

def unstable_softmax(x):
    """Naive softmax — looks correct, hides a bomb."""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# Works fine with small logits
small_logits = np.array([1.0, 2.0, 3.0])
print("Small logits:", unstable_softmax(small_logits))
# [0.09003057, 0.24472847, 0.66524096]  ← looks great!

# Now try logits that come from a real model deep in training
large_logits = np.array([1000.0, 1001.0, 1002.0])
print("Large logits:", unstable_softmax(large_logits))
# [nan, nan, nan]  ← Welcome to the NaN Zone.
```

### The Investigation

What happened? Let's trace it:

```
  Step 1: Compute exp(1000), exp(1001), exp(1002)

  exp(1000) = ... a number with 434 digits.
  float64 max ≈ 1.8 × 10^308.

  exp(1000) ≈ 1.97 × 10^434  →  OVERFLOW  →  inf

  Step 2: Compute inf / (inf + inf + inf)

  inf / inf = nan

  Step 3: nan propagates everywhere

  Loss = nan, gradients = nan, weights = nan, model = ruined.

  Timeline of destruction:
  ┌────────────┐    ┌────────┐    ┌──────────┐    ┌───────────┐
  │ exp(1000)  │───→│  inf   │───→│ inf/inf  │───→│    nan    │
  │            │    │        │    │          │    │           │
  │ (overflow) │    │        │    │  = nan   │    │ GAME OVER │
  └────────────┘    └────────┘    └──────────┘    └───────────┘
```

### The Fix: The Max-Subtraction Trick

Here's the key mathematical insight. For any constant $c$:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}$$

**Why?** Because $\frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{x_i} \cdot e^{-c}}{\sum_j e^{x_j} \cdot e^{-c}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$. The $e^{-c}$ cancels out.

If we choose $c = \max(x)$, then the largest exponent we'll ever compute is $e^0 = 1$. No overflow possible.

```python
import numpy as np

def stable_softmax(x):
    """Numerically stable softmax — the version all frameworks use."""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)  # Largest value becomes exp(0) = 1
    return exp_x / np.sum(exp_x)

# Same inputs, no explosion
large_logits = np.array([1000.0, 1001.0, 1002.0])
print("Stable softmax:", stable_softmax(large_logits))
# [0.09003057, 0.24472847, 0.66524096]  ← identical to the small logits case!

# Verify they match
small_logits = np.array([1.0, 2.0, 3.0])
print("Small logits: ", stable_softmax(small_logits))
# Same probabilities! The offset doesn't change the output.
```

> **You Already Know This**: This is the same principle as avoiding integer overflow in `(a + b) / 2` by writing `a + (b - a) / 2` instead. You shift the computation into a safer numerical range without changing the mathematical result. Same trick, different domain.

---

## The Log-Sum-Exp Trick: Softmax's Big Sibling

In ML, you often need $\log(\text{softmax}(x))$ — for example, in cross-entropy loss. Computing softmax first and then taking the log is doubly dangerous: overflow in `exp()`, then underflow in `log()` of near-zero probabilities.

The log-sum-exp trick handles this in one shot:

$$\log\left(\sum_i e^{x_i}\right) = x_{\max} + \log\left(\sum_i e^{x_i - x_{\max}}\right)$$

**Why?** Factor out $e^{x_{\max}}$ from the sum:

$$\log\left(\sum_i e^{x_i}\right) = \log\left(e^{x_{\max}} \sum_i e^{x_i - x_{\max}}\right) = x_{\max} + \log\left(\sum_i e^{x_i - x_{\max}}\right)$$

Now the exponents inside the sum are all $\leq 0$, so no overflow. And the log is taken of a sum that's $\geq 1$ (since one of the terms is $e^0 = 1$), so no underflow in the log.

```python
import numpy as np

def unstable_logsumexp(x):
    """Naive: overflow in exp, then log(inf) = inf"""
    return np.log(np.sum(np.exp(x)))

def stable_logsumexp(x):
    """Stable: shift before exp, compensate after log"""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))

# Test with large values
x = np.array([1000.0, 1000.5, 1001.0])
print(f"Unstable logsumexp: {unstable_logsumexp(x)}")  # inf
print(f"Stable logsumexp:   {stable_logsumexp(x)}")    # ~1001.41 (correct)
print(f"scipy logsumexp:    {np.logaddexp.reduce(x)}")  # same, built-in

# This is what PyTorch's F.cross_entropy uses internally
# It computes log_softmax in a single numerically stable pass
```

---

## Safe Operations: Defensive Arithmetic

Just like you write defensive code against null pointers and empty arrays, you need defensive code against numerical edge cases.

### Safe Log

$\log(0) = -\infty$. In your loss function, if any predicted probability is exactly 0, you get `-inf` in your loss, which becomes `nan` in your gradients.

```python
import numpy as np

def safe_log(x, eps=1e-10):
    """Prevent log(0) = -inf by clamping input."""
    return np.log(np.maximum(x, eps))

# Danger
print(f"log(0) = {np.log(0)}")           # -inf

# Safe
print(f"safe_log(0) = {safe_log(0)}")    # -23.03 (large negative, but finite)
```

### Safe Division

Division by zero produces `inf`. A variance of zero in batch normalization, a normalization constant of zero — these happen more often than you'd think.

```python
import numpy as np

def safe_divide(a, b, eps=1e-10):
    """Prevent division by zero."""
    return a / np.maximum(np.abs(b), eps) * np.sign(b + eps)

# Danger
print(f"1/0 = {1.0 / np.float64(0)}")                  # inf

# Safe
print(f"safe_divide(1, 0) = {safe_divide(1.0, 0.0)}")  # Large but finite
```

This is exactly what the $\epsilon$ in batch normalization does:

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

That $\epsilon$ (typically $10^{-5}$) prevents division by zero when the batch variance is zero. Every ML framework adds it — now you know why.

> **You Already Know This**: This is literally the same pattern as checking for null before dereferencing a pointer, or adding a default case to a switch statement. You're guarding against edge cases that "shouldn't happen" but inevitably do in production. Numerical defensive programming.

---

## The Accumulation Problem: Death by a Thousand Additions

Here's a subtlety that trips up even experienced engineers. Adding many small numbers to a large number can lose the small numbers entirely, because they fall below the relative precision of the large number.

```
  Why naive summation fails:

  total = 1,000,000.0          (needs ~7 digits to represent)
  + 0.001                      (this has 1 digit of significance)
  ─────────────────
  total = 1,000,000.001        (needs 10 digits!)
  But float32 only has ~7...
  So it gets rounded to:
  total = 1,000,000.0          ← the 0.001 just vanished!

  Do this a million times, and you've lost the sum of a
  million 0.001's = 1000.0, completely silently.
```

### Kahan Summation: The Compensated Algorithm

Kahan summation tracks the rounding error from each addition and compensates for it in the next step. It's like keeping a "change jar" for the pennies that get rounded away:

```python
import numpy as np

def naive_sum(arr):
    """Simple left-to-right summation."""
    total = 0.0
    for x in arr:
        total += x
    return total

def kahan_sum(arr):
    """Compensated summation — tracks lost precision."""
    total = 0.0
    compensation = 0.0       # The "change jar" for rounding errors
    for x in arr:
        y = x - compensation   # Add back what was lost last time
        t = total + y          # This addition may lose precision
        compensation = (t - total) - y  # Recover what was lost
        total = t
    return total

# One large value plus many small values
n = 10_000_000
arr = np.array([1.0] + [1e-10] * n, dtype=np.float64)
true_sum = 1.0 + n * 1e-10  # = 1.001

print(f"True sum:  {true_sum}")
print(f"Naive sum: {naive_sum(arr)}")     # Loses precision
print(f"Kahan sum: {kahan_sum(arr)}")     # Much better
print(f"NumPy sum: {np.sum(arr)}")        # NumPy uses pairwise summation
```

**Where this matters in ML**: When you're computing the mean of a large batch, summing gradients from many examples, or accumulating loss over thousands of steps for logging. NumPy and PyTorch use smarter summation algorithms internally, but if you ever write custom accumulation logic, use Kahan or at minimum pairwise summation.

> **Common Mistake**: Writing a training loop that accumulates `total_loss += loss.item()` over thousands of steps for logging, then dividing at the end. If the loss values are small and the running total is large, you'll gradually lose precision. Periodically reset and re-accumulate, or use compensated summation.

---

## Condition Numbers: The Error Amplification Factor

Now let's talk about a concept that connects beautifully to something you already understand as a systems engineer.

> **You Already Know This**: In distributed systems, a small delay in one service can cascade and amplify through a chain of dependent services. A 10ms delay at the database can become a 500ms delay at the API gateway. The "amplification factor" depends on the architecture of the dependency chain. Condition numbers are the mathematical version of this same idea — they measure how much a small error in the input gets amplified by a computation.

### What Is a Condition Number?

For a matrix $A$, the condition number is:

$$\kappa(A) = \|A\| \cdot \|A^{-1}\|$$

**Translation**: If you're solving $Ax = b$, and your input $b$ has a small relative error $\epsilon$, then the relative error in the solution $x$ can be as large as $\kappa(A) \cdot \epsilon$.

```
  Error amplification by condition number:

  Well-conditioned (κ ≈ 1-10):
  Input error: ε ──→ Output error: ~ε
  "What you put in is roughly what you get out"

  Moderately conditioned (κ ≈ 10³):
  Input error: ε ──→ Output error: ~1000ε
  "Getting shaky, but still workable"

  Ill-conditioned (κ ≈ 10¹⁰):
  Input error: ε ──→ Output error: ~10¹⁰ε
  "You started with 16 digits of precision.
   You now have 6. Maybe."

  Singular (κ = ∞):
  Input error: ε ──→ Output error: ¯\_(ツ)_/¯
  "No meaningful answer possible"
```

A rough rule: if $\kappa(A) \approx 10^k$, you lose about $k$ digits of precision when solving $Ax = b$.

```python
import numpy as np

print("=== Matrix Conditioning ===")

# Well-conditioned matrix
A_good = np.array([[4, 1], [1, 3]], dtype=np.float64)
cond_good = np.linalg.cond(A_good)
print(f"Well-conditioned matrix:")
print(f"  A = [[4, 1], [1, 3]]")
print(f"  Condition number: {cond_good:.2f}")    # ~2.6

# Ill-conditioned matrix (nearly singular)
epsilon = 1e-10
A_bad = np.array([[1, 1], [1, 1 + epsilon]], dtype=np.float64)
cond_bad = np.linalg.cond(A_bad)
print(f"\nIll-conditioned matrix:")
print(f"  A = [[1, 1], [1, 1+1e-10]]")
print(f"  Condition number: {cond_bad:.2e}")     # ~2e10 — yikes

# Solving the same system with both
b = np.array([1.0, 1.0])
x_good = np.linalg.solve(A_good, b)
x_bad = np.linalg.solve(A_bad, b)
print(f"\nSolving Ax = [1, 1]:")
print(f"  Well-conditioned solution: {x_good}")
print(f"  Ill-conditioned solution:  {x_bad}")

# Now perturb b slightly and see how the solution changes
b_perturbed = b + np.array([1e-10, 0])
x_good_perturbed = np.linalg.solve(A_good, b_perturbed)
x_bad_perturbed = np.linalg.solve(A_bad, b_perturbed)
print(f"\nSolving Ax = [1+1e-10, 1] (tiny perturbation):")
print(f"  Well-conditioned: solution changed by {np.linalg.norm(x_good_perturbed - x_good):.2e}")
print(f"  Ill-conditioned:  solution changed by {np.linalg.norm(x_bad_perturbed - x_bad):.2e}")
# The ill-conditioned solution jumps wildly from a tiny input change!
```

**Where this bites you in ML**:

- **Linear regression with correlated features**: If your feature matrix $X^TX$ is ill-conditioned (highly correlated columns), the closed-form solution $(X^TX)^{-1}X^Ty$ amplifies floating-point errors massively. This is one reason L2 regularization (Ridge regression) exists — adding $\lambda I$ to $X^TX$ directly improves the condition number.

- **Deep networks**: The effective condition number of the Hessian (second derivative matrix) of the loss landscape affects how well gradient descent works. Ill-conditioning is why you need learning rate schedules, Adam optimizer, and batch normalization.

- **Attention matrices in transformers**: The $QK^T$ product can produce matrices with poor conditioning, which is partly why the $\frac{1}{\sqrt{d_k}}$ scaling factor exists — it prevents softmax saturation from ill-conditioned attention scores.

---

## Mixed Precision Training: Dancing on the Edge

Modern ML training frequently uses mixed precision — float16 for speed, float32 for safety. Understanding numerical stability makes the design choices in mixed precision obvious.

```
  Mixed precision training flow:

  ┌──────────────────────────────────────────────────────────┐
  │                  FORWARD PASS (float16)                   │
  │                                                          │
  │  Weights       Activations     Loss                      │
  │  (float16) ──→ (float16) ──→ (float32!)                 │
  │                                ↑                         │
  │                  Loss computed in float32                 │
  │                  to avoid overflow                        │
  ├──────────────────────────────────────────────────────────┤
  │                 BACKWARD PASS (float16)                   │
  │                                                          │
  │  Gradients      Scaled by S     Unscaled                 │
  │  (float16) ←── (float16) ←── (float32)                  │
  │     ↓                                                    │
  │  Loss scaling (multiply loss by S=1024 to                │
  │  prevent gradient underflow in float16)                  │
  ├──────────────────────────────────────────────────────────┤
  │              WEIGHT UPDATE (float32!)                     │
  │                                                          │
  │  Master weights (float32) ← float32 update               │
  │  Copy to float16 for next forward pass                   │
  │                                                          │
  │  Why float32 here? Because weight updates are TINY       │
  │  (lr * grad ≈ 1e-4 * 1e-3 = 1e-7)                      │
  │  and float16 epsilon is ~1e-3. The update would vanish!  │
  └──────────────────────────────────────────────────────────┘
```

**Why loss scaling?** Gradients in deep networks are often very small (1e-5 to 1e-8). In float16, anything below ~6e-8 underflows to zero. By multiplying the loss by a large constant $S$ (say, 1024), all gradients get scaled up into the representable range. After the backward pass, you divide by $S$ before updating weights.

This is numerical stability engineering at its finest — you're manually managing the dynamic range of your computation to fit within the hardware's precision limits.

> **You Already Know This**: This is the same idea as audio normalization or image contrast stretching. You scale your data to use the full dynamic range of your representation format. In mixed precision, you're doing it for gradients to use the full dynamic range of float16.

---

## Putting It All Together: Stable Binary Cross-Entropy

Let's build something real that applies everything we've covered. Binary cross-entropy loss is used in every binary classifier and every sigmoid output. The naive formula is:

$$L = -[y \log(p) + (1-y) \log(1-p)]$$

where $y \in \{0, 1\}$ is the label and $p$ is the predicted probability.

### What Can Go Wrong?

Everything.

```
  Numerical landmines in binary cross-entropy:

  1. p = 0  →  log(0) = -inf       (if y=1, loss = inf)
  2. p = 1  →  log(1-1) = log(0)   (if y=0, loss = inf)
  3. p very close to 0 or 1 → catastrophic cancellation in 1-p
  4. exp() in sigmoid can overflow → p becomes exactly 0 or 1
```

### The Stable Implementation

```python
import numpy as np

def stable_binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Numerically stable binary cross-entropy.

    Strategy: clip predictions away from 0 and 1 to prevent
    log(0) and log(1-1).
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return np.mean(loss)

# Test with extreme predictions
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9999999999, 0.0000000001, 0.7, 0.3])
print(f"Stable BCE: {stable_binary_cross_entropy(y_true, y_pred):.6f}")

# What happens without clipping?
y_pred_dangerous = np.array([1.0, 0.0, 0.7, 0.3])
loss_dangerous = -y_true * np.log(y_pred_dangerous) - (1 - y_true) * np.log(1 - y_pred_dangerous)
print(f"Unclipped BCE: {loss_dangerous}")  # Contains -0.0 and inf
```

But the *truly* stable approach (what PyTorch does) is to work with logits directly, never converting to probabilities at all:

$$L = -[y \cdot z - \log(1 + e^z)]$$

where $z$ is the logit (pre-sigmoid value). This avoids both the sigmoid overflow *and* the log-of-probability issues.

```python
import numpy as np

def stable_bce_from_logits(y_true, logits):
    """
    The approach PyTorch's F.binary_cross_entropy_with_logits uses.
    Never computes probabilities at all — stays in logit space.
    """
    # max(0, z) - y*z + log(1 + exp(-abs(z)))
    # This formulation avoids overflow in exp()
    return np.mean(
        np.maximum(logits, 0) - y_true * logits +
        np.log(1 + np.exp(-np.abs(logits)))
    )

y_true = np.array([1, 0, 1, 0], dtype=np.float64)
logits = np.array([100.0, -100.0, 0.5, -0.5])  # Extreme logits!
print(f"Stable BCE from logits: {stable_bce_from_logits(y_true, logits):.6f}")
# Works perfectly even with logits of +/-100
```

---

## Real-World Stability Patterns in ML Frameworks

Modern frameworks handle most numerical stability for you — but only if you use the right APIs.

```python
import torch
import torch.nn.functional as F

# === Softmax & Cross-Entropy ===
logits = torch.tensor([1000.0, 1001.0, 1002.0])

# GOOD: framework handles stability
probs = F.softmax(logits, dim=0)
print(f"Stable softmax: {probs}")

# GOOD: combined log-softmax + NLL loss in one stable pass
targets = torch.tensor([2])
loss = F.cross_entropy(logits.unsqueeze(0), targets)
print(f"Cross-entropy loss: {loss.item():.4f}")

# BAD: computing softmax then log separately
# probs = F.softmax(logits, dim=0)
# log_probs = torch.log(probs)  # Underflow risk!
# GOOD: use log_softmax directly
log_probs = F.log_softmax(logits, dim=0)
print(f"Log-softmax: {log_probs}")
```

### Rules of Thumb

| Situation | DO | DON'T |
|---|---|---|
| Classification loss | `F.cross_entropy(logits, targets)` | `softmax` then `log` then `nll_loss` |
| Binary classification | `F.binary_cross_entropy_with_logits(logits, targets)` | `sigmoid` then `binary_cross_entropy` |
| Need log-probabilities | `F.log_softmax(logits)` | `torch.log(F.softmax(logits))` |
| Division in normalization | Add `eps` parameter | Divide raw values |
| Multiplying many probabilities | Sum logs: $\sum \log p_i$ | Multiply directly: $\prod p_i$ |
| Comparing floats | `torch.allclose(a, b)` | `a == b` |

---

## Exercises

### Exercise 1: Debug the NaN

**Problem**: This normalize function produces NaN. Diagnose the root cause and fix it.

```python
import numpy as np

def buggy_normalize(x):
    return x / np.sum(x)

x = np.array([1e-300, 2e-300, 3e-300])
result = buggy_normalize(x)  # Returns [nan, nan, nan]
```

**Hint**: What happens when you sum numbers near the underflow boundary?

**Solution**:

```python
import numpy as np

def stable_normalize(x, eps=1e-10):
    """
    Root cause: when all values are tiny (near underflow), their sum
    can underflow to 0, causing 0/0 = nan.

    Fix: scale up before normalizing. The normalization makes the
    scale factor cancel out anyway.
    """
    x_max = np.max(np.abs(x))
    if x_max > 0:
        x_scaled = x / x_max  # Now values are in [0, 1] range
        return x_scaled / np.sum(x_scaled)
    return np.ones_like(x) / len(x)  # All zeros → uniform

# Alternative: work entirely in log space
def log_normalize(log_x):
    """If values are given as logs, normalize without ever exponentiating naively."""
    log_sum = np.max(log_x) + np.log(np.sum(np.exp(log_x - np.max(log_x))))
    return np.exp(log_x - log_sum)

x = np.array([1e-300, 2e-300, 3e-300])
print(f"Stable normalize: {stable_normalize(x)}")
# [0.16666667, 0.33333333, 0.5]

print(f"Log normalize:    {log_normalize(np.log(x))}")
# [0.16666667, 0.33333333, 0.5]
```

### Exercise 2: Safe Linear System Solver

**Problem**: Write a function that checks the condition number before solving a linear system, warns about potential instability, and applies regularization when needed.

**Solution**:

```python
import numpy as np

def safe_solve(A, b, condition_threshold=1e10):
    """
    Solve Ax = b with condition number monitoring.

    Think of this like a circuit breaker in a microservice:
    if the error amplification factor is too high, we add
    regularization (like adding a timeout/fallback) instead
    of returning a garbage result.
    """
    cond = np.linalg.cond(A)

    if cond > condition_threshold:
        print(f"WARNING: condition number = {cond:.2e}")
        print(f"  You'll lose ~{int(np.log10(cond))} digits of precision.")
        print(f"  Applying Tikhonov regularization (adding λI to A).")

        # Regularize: A'x = b becomes (A + λI)x = b
        lam = 1e-6
        A_reg = A + np.eye(A.shape[0]) * lam
        cond_reg = np.linalg.cond(A_reg)
        print(f"  Regularized condition number: {cond_reg:.2e}")

        return np.linalg.solve(A_reg, b), False

    return np.linalg.solve(A, b), True

# Well-conditioned system
print("=== Well-conditioned ===")
A_good = np.array([[4.0, 1.0], [1.0, 3.0]])
x, stable = safe_solve(A_good, np.array([1.0, 1.0]))
print(f"Solution: {x}, stable: {stable}\n")

# Ill-conditioned system
print("=== Ill-conditioned ===")
A_bad = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]])
x, stable = safe_solve(A_bad, np.array([2.0, 2.0]))
print(f"Solution: {x}, stable: {stable}")
```

### Exercise 3: Numerical Gradient Checker

**Problem**: Implement a numerical gradient checker that uses an appropriate step size and handles potential catastrophic cancellation. Verify it against an analytical gradient.

**Solution**:

```python
import numpy as np

def numerical_gradient(f, x, h=None):
    """
    Central difference gradient with optimal step size.

    The optimal h balances two errors:
    - Too large: truncation error (bad approximation)
    - Too small: catastrophic cancellation
    Sweet spot: h ≈ epsilon^(1/3) for central differences
    """
    if h is None:
        h = np.cbrt(np.finfo(x.dtype).eps)  # Optimal for central diff

    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Test: f(x) = x^2, gradient should be 2x
def f(x):
    return np.sum(x ** 2)

def analytical_grad(x):
    return 2 * x

x = np.array([3.0, 4.0, 5.0])
num_grad = numerical_gradient(f, x)
ana_grad = analytical_grad(x)
print(f"Numerical gradient:  {num_grad}")
print(f"Analytical gradient: {ana_grad}")
print(f"Max difference: {np.max(np.abs(num_grad - ana_grad)):.2e}")
```

---

## Quick Reference: Numerical Stability Cheatsheet

```
  ┌─────────────────────────────────────────────────────────────────┐
  │              NUMERICAL STABILITY CHEATSHEET                      │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  MACHINE EPSILON                                                │
  │  float16: ~10⁻³  │  float32: ~10⁻⁷  │  float64: ~10⁻¹⁶       │
  │                                                                 │
  │  OVERFLOW LIMITS                                                │
  │  float16: 65,504  │  float32: 3.4×10³⁸  │  float64: 1.8×10³⁰⁸ │
  │                                                                 │
  │  THE BIG THREE TRICKS                                           │
  │  1. Softmax: subtract max before exp                            │
  │  2. LogSumExp: factor out max, log after                        │
  │  3. Always add ε when dividing or taking log                    │
  │                                                                 │
  │  GOLDEN RULES                                                   │
  │  • Use framework-provided loss functions                        │
  │  • Work in log space when multiplying probabilities             │
  │  • Never compare floats with ==                                 │
  │  • Check condition numbers before solving linear systems        │
  │  • Prefer logit-space APIs over probability-space APIs          │
  │  • h ≈ ε^(1/3) for numerical gradient checks                   │
  │                                                                 │
  │  CONDITION NUMBER RULE OF THUMB                                 │
  │  κ(A) ≈ 10ᵏ  →  you lose ~k digits of precision               │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

## Summary

- **Machine epsilon** defines the fundamental limit of floating-point precision — $\approx 10^{-16}$ for float64, $\approx 10^{-7}$ for float32, $\approx 10^{-3}$ for float16. Every operation introduces error up to this relative magnitude.

- **Overflow** happens when numbers exceed the type's maximum ($\approx 10^{308}$ for float64, $\approx 10^{38}$ for float32, just $65{,}504$ for float16). In ML, `exp()` in softmax is the usual culprit.

- **Underflow** silently replaces tiny numbers with zero ($< 10^{-308}$ for float64). This causes vanishing gradients and silent training stalls. The fix: work in log space.

- **Catastrophic cancellation** amplifies errors when subtracting nearly equal numbers. It can destroy precision in a single operation. Watch for it in numerical gradient checks and variance computations.

- **Stable softmax** subtracts the maximum before exponentiating: $\text{softmax}(x) = \text{softmax}(x - x_{\max})$. This is mathematically equivalent but numerically safe.

- **Log-sum-exp trick** prevents overflow in $\log\sum e^{x_i}$ by factoring out the max: $x_{\max} + \log\sum e^{x_i - x_{\max}}$.

- **Condition number** $\kappa(A)$ measures how much a matrix amplifies input errors. If $\kappa(A) \approx 10^k$, you lose about $k$ digits of precision. Regularization improves conditioning.

- **Always add epsilon** when dividing or taking logarithms of quantities that might be zero. This is the floating-point equivalent of null checks.

- **Mixed precision training** requires explicit management of numerical range through loss scaling and maintaining master weights in float32.

- **Modern frameworks** handle stability for you — if you use the right APIs. Prefer combined operations (`F.cross_entropy`, `F.log_softmax`, `F.binary_cross_entropy_with_logits`) over manual multi-step computations.

---

## What's Next

Now that you understand the ways computers fail at arithmetic — and how to defend against it — we're ready to tackle a related question: if we can't compute things exactly, how do we **approximate solutions reliably**?

In the next chapter, **Approximation Methods**, we'll explore iterative algorithms that converge to correct answers despite never reaching them exactly. You'll see how gradient descent, Newton's method, and other iterative schemes work — and why understanding numerical stability (this chapter) is essential for knowing when those approximations are trustworthy and when they're drifting into numerical quicksand.

The stability principles you've just learned — watching for overflow, avoiding cancellation, checking condition numbers — are exactly the tools you'll use to judge whether an iterative approximation is converging to truth or to nonsense.
