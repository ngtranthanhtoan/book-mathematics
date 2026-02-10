# Number Systems

Your model's loss just went to NaN. Or worse, to infinity. Or it's 0.30000000000000004 instead of 0.3. Welcome to the world of numbers as computers actually see them — because float32 ≠ real numbers, and that distinction will bite you.

---

**Building On** — Mathematical thinking gave you the reasoning toolkit. Now let's examine the most basic objects in math: numbers. But for ML engineers, the question isn't "what is a number?" — it's "how does my computer represent it, and where does that representation break?"

---

## The Problem: "Why Did My Loss Go NaN?"

You're training a transformer. Epoch 1 looks fine. Epoch 2, loss drops. Epoch 3 — `NaN`. You stare at the logs. Nothing obvious. You add gradient clipping. Still `NaN`. You switch from float16 to float32. It works. But *why*?

The answer lives in how computers represent numbers. And it's not the way mathematicians think about them. Let's build from the ground up.

---

## The Number Hierarchy

Mathematicians organize numbers into a neat containment hierarchy:

$$\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R}$$

Each set extends the previous one to handle something new.

> **You Already Know This** — This is a type hierarchy, and you've been working with one your entire career:
>
> `uint` → `int` → `float` → `complex`
>
> Each type "extends" the previous to represent more values, just like each number set contains the previous. A `uint8` can hold `42`, so can an `int32`, so can a `float64`. The reverse isn't true — `float64` can hold `3.14`, but `int32` can't (not without truncation). Subtyping in math, subtyping in code — same principle.

### Natural Numbers ($\mathbb{N}$)

$$\mathbb{N} = \{1, 2, 3, 4, 5, \ldots\}$$

Some definitions include 0. These are your counting numbers. They're closed under addition and multiplication:

- $a + b \in \mathbb{N}$ for all $a, b \in \mathbb{N}$
- $a \times b \in \mathbb{N}$ for all $a, b \in \mathbb{N}$

But subtraction can take you outside the set ($3 - 5 = -2 \notin \mathbb{N}$). That's why we need...

### Integers ($\mathbb{Z}$)

$$\mathbb{Z} = \{\ldots, -3, -2, -1, 0, 1, 2, 3, \ldots\}$$

Now subtraction always works. But division doesn't: $1 \div 3$ isn't an integer. So we extend again...

### Rational Numbers ($\mathbb{Q}$)

$$\mathbb{Q} = \left\{\frac{p}{q} : p, q \in \mathbb{Z}, q \neq 0\right\}$$

Rationals can always be expressed as terminating or repeating decimals:

- $\frac{1}{4} = 0.25$ (terminating)
- $\frac{1}{3} = 0.333\ldots$ (repeating)

Division is now closed (except by zero). But there are "holes" on the number line — numbers that *cannot* be written as fractions.

### Real Numbers ($\mathbb{R}$)

Real numbers fill in *all* the gaps. They include irrationals like:

- $\pi = 3.14159265\ldots$
- $e = 2.71828182\ldots$
- $\sqrt{2} = 1.41421356\ldots$

These have non-repeating, non-terminating decimal expansions. They can't be expressed as fractions. And here's the kicker: almost all real numbers are irrational. The rationals, despite being infinite, are a vanishingly thin dusting on the number line compared to the irrationals.

Real numbers are the foundation of calculus, and calculus is the foundation of gradient-based optimization. Every loss function, every gradient, every learning rate lives in $\mathbb{R}$.

**But your computer doesn't have $\mathbb{R}$. It has something much worse.**

---

## IEEE 754: How Your Computer Fakes Real Numbers

Your GPU doesn't have infinite precision. It has 32 bits (or 16, or 64). And it uses them according to the IEEE 754 standard — which you've seen in every language's floating-point spec, even if you've never read it carefully.

> **You Already Know This** — IEEE 754 is behind every `float` and `double` in C, Java, Python, Rust, Go, JavaScript — literally every language you've ever used. When Python gives you `0.1 + 0.2 = 0.30000000000000004`, that's IEEE 754 doing exactly what the spec says.

### The Bit Layout (float32)

Here's what a 32-bit float actually looks like in memory:

```
 31  30       23  22                    0
┌───┬──────────┬──────────────────────────┐
│ S │ Exponent │        Mantissa          │
│ 1 │  8 bits  │        23 bits           │
└───┴──────────┴──────────────────────────┘
 ↓       ↓                  ↓
 ±   2^(e-127)        1.xxxxx (binary)


Value = (-1)^S  x  2^(Exponent - 127)  x  (1 + Mantissa)

Example: the number 6.5
  S = 0          (positive)
  Exponent = 129 (binary: 10000001) → 2^(129-127) = 2^2 = 4
  Mantissa = .625 (binary: 10100000000000000000000) → 1.625
  Value = (+1) x 4 x 1.625 = 6.5

┌───┬──────────┬──────────────────────────┐
│ 0 │ 10000001 │ 10100000000000000000000  │
└───┴──────────┴──────────────────────────┘
```

And here's the comparison across precision levels you'll actually use in ML:

```
float16 (half):    1 sign + 5 exponent  + 10 mantissa  = 16 bits
float32 (single):  1 sign + 8 exponent  + 23 mantissa  = 32 bits
float64 (double):  1 sign + 11 exponent + 52 mantissa  = 64 bits
bfloat16 (brain):  1 sign + 8 exponent  + 7 mantissa   = 16 bits
```

Notice bfloat16 — Google's "Brain Float." Same exponent range as float32 (so it can represent the same *magnitude* of numbers), but with far less mantissa precision. This is intentional: in deep learning, range matters more than precision for many operations.

### Special Values

IEEE 754 reserves certain bit patterns for edge cases:

| Exponent | Mantissa | Meaning |
|----------|----------|---------|
| All zeros ($e = 0$) | All zeros ($m = 0$) | **Zero** ($\pm 0$) |
| All zeros ($e = 0$) | Non-zero ($m \neq 0$) | **Denormalized** (very small numbers near zero) |
| All ones ($e = 255$) | All zeros ($m = 0$) | **Infinity** ($\pm\infty$) |
| All ones ($e = 255$) | Non-zero ($m \neq 0$) | **NaN** (Not a Number) |

That NaN at the bottom of the table? That's what shows up in your loss. It comes from operations like:

- $0 / 0$
- $\infty - \infty$
- $\sqrt{-1}$ (in real arithmetic)
- Any operation involving an existing NaN (NaN is *contagious*)

### Precision: What "~7 Decimal Digits" Actually Means

```
float32:  ~7  decimal digits of precision
float64:  ~15-16 decimal digits of precision
float16:  ~3-4 decimal digits of precision
bfloat16: ~2-3 decimal digits of precision
```

This isn't about the *size* of the number. float32 can represent numbers up to ~$3.4 \times 10^{38}$. But it can only distinguish about 7 significant digits. That means:

```
float32(1000000.0 + 0.1) == float32(1000000.0)   # True! The 0.1 is lost.
float32(1.0000001)        ≈ 1.0000001             # Fine, within 7 digits.
float32(10000000.1)       ≈ 10000000.0            # The .1 is gone.
```

### The Number Line Is Not Uniform

This is the critical insight that most people miss. Floats are **not** evenly spaced. They're densely packed near zero and sparse far from it:

```
Near zero (dense):
    ←──|·····|·····|·····|·····|·····|──→
       0    1e-38  2e-38  3e-38  4e-38

Near 1.0 (moderate spacing):
    ←──|···|···|···|···|···|···|···|──→
      0.9  0.95  1.0  1.05  1.1

Near 1,000,000 (sparse):
    ←──|         |         |         |──→
    999,999    1,000,000   1,000,001

Near 1e30 (very sparse):
    ←──|                              |──→
    1.000000e30              1.000001e30
      (gap of ~1e23 between representable values!)
```

Between 1.0 and 2.0, there are $2^{23}$ = 8,388,608 representable float32 values. Between $2^{23}$ and $2^{24}$, there are *also* $2^{23}$ values — but spread over a range 8 million times wider. The density drops as magnitude increases. This is why adding a small number to a large number can simply *lose* the small number entirely.

---

## Machine Epsilon: The Precision Boundary

> **You Already Know This** — Machine epsilon is the smallest number that makes `1.0 + eps != 1.0`. It's why your tests use `np.allclose()` instead of `==`. Every time you've written a floating-point comparison with a tolerance, you were implicitly reasoning about machine epsilon.

Formally, machine epsilon ($\varepsilon$) is:

$$\varepsilon = \min\{x > 0 : \text{fl}(1.0 + x) \neq 1.0\}$$

For the standard types:

| Type | Machine Epsilon | Approximate |
|------|----------------|-------------|
| float16 | $2^{-10}$ | $\approx 9.77 \times 10^{-4}$ |
| bfloat16 | $2^{-7}$ | $\approx 7.81 \times 10^{-3}$ |
| float32 | $2^{-23}$ | $\approx 1.19 \times 10^{-7}$ |
| float64 | $2^{-52}$ | $\approx 2.22 \times 10^{-16}$ |

Machine epsilon tells you the *relative* precision of your format. If you're working with numbers around $10^6$, the absolute precision of float32 is roughly $10^6 \times 1.19 \times 10^{-7} \approx 0.119$. You can't distinguish values closer than ~0.1 apart at that scale.

**This directly explains the NaN problem.** If your model's logits are very large (say, 1000), and you're trying to compute softmax, the differences between `exp(1000)`, `exp(1001)`, and `exp(1002)` overflow to infinity. Infinity divided by infinity is NaN. Game over.

---

## Numerical Stability: Why Your Code Breaks (and How to Fix It)

> **You Already Know This** — This is why `np.allclose(a, b, rtol=1e-5, atol=1e-8)` exists. Every numerical library has tolerance-based comparison because exact equality is meaningless for floats.

### The Classic Gotchas

**Catastrophic Cancellation** — Subtracting two nearly equal numbers:

```python
a = 1.0000001
b = 1.0000000
# Mathematically: a - b = 1e-7
# In float32:    a - b = 1.1920929e-07  (only 1-2 correct digits!)
```

You had 7 digits of precision in each number, but the subtraction wiped out the leading digits, leaving you with the noisy trailing ones.

**Accumulation Drift** — Summing many small numbers:

```python
# Adding 0.1 one million times in float32
total = np.float32(0.0)
for _ in range(1_000_000):
    total += np.float32(0.1)
# Expected: 100,000.0
# Actual:   ~100,958.34  (nearly 1% error!)
```

Each addition introduces a tiny rounding error. Over a million iterations, those errors compound. This is exactly what happens during gradient accumulation across many microbatches.

### The Softmax Example (You Will See This Everywhere)

```python
import numpy as np

def unstable_softmax(x):
    """Naive softmax — breaks on large inputs."""
    return np.exp(x) / np.sum(np.exp(x))

def stable_softmax(x):
    """Numerically stable softmax — subtract max first."""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

logits = np.array([1000, 1001, 1002], dtype=np.float32)
print(f"Unstable: {unstable_softmax(logits)}")  # [nan, nan, nan]
print(f"Stable:   {stable_softmax(logits)}")    # [0.0900, 0.2447, 0.6652]
```

The math is *identical* — subtracting a constant from all inputs doesn't change the softmax output. But the numerics are completely different. This is the kind of thing that separates "my model trains" from "my model produces NaN on epoch 3."

---

## Running Example: Mixed-Precision Training

Here's the question that ties all of this together: **Why does mixed-precision training work?**

The setup: you use float16 (or bfloat16) for the forward pass and most of the backward pass, but float32 for gradient accumulation and the optimizer state. This nearly doubles throughput on modern GPUs. But why doesn't the reduced precision destroy your model?

### Why float16 Works for Forward/Backward

During a forward pass, what matters is the *relative* magnitude of activations, not their precise values. float16 has about 3-4 digits of precision, which is enough to distinguish "this neuron is strongly activated" from "this neuron is weakly activated." The signal survives.

Similarly, gradients in the backward pass don't need to be precise to the 7th decimal place. They just need to point in roughly the right direction. Stochastic gradient descent is already noisy (you're computing gradients on random mini-batches), so a few extra bits of noise from float16 rounding barely matter.

### Why float32 Is Needed for Accumulation

But here's where it breaks down. Suppose your learning rate is $10^{-4}$ and your weights are around $1.0$. The update is:

$$w \leftarrow w - \eta \cdot g = 1.0 - 10^{-4} \cdot g$$

In float16, the machine epsilon is $\sim 10^{-3}$. A weight update of $10^{-4}$ is *smaller than machine epsilon relative to the weight's magnitude*. The update would simply vanish — `float16(1.0 - 0.0001) == float16(1.0)`. Your model stops learning.

In float32, machine epsilon is $\sim 10^{-7}$. An update of $10^{-4}$ is well within the representable range. The weight actually changes.

```
float16 precision near 1.0:
    ←──|           |           |──→
     0.9990     1.0000     1.0010
     (gap = 0.001, update of 0.0001 is lost)

float32 precision near 1.0:
    ←──|·|·|·|·|·|·|·|·|·|·|·|──→
     0.9999999  1.0000000  1.0000001
     (gap = 0.0000001, update of 0.0001 is captured)
```

### The Full Mixed-Precision Recipe

```python
import numpy as np

# --- Simulating mixed-precision training ---

# Master weights in float32 (these are the "source of truth")
weights_fp32 = np.float32(1.0)

# Forward pass: cast to float16
weights_fp16 = np.float16(weights_fp32)
# ... compute loss in float16 ...
# ... compute gradients in float16 ...
gradient_fp16 = np.float16(0.0001)

# Gradient accumulation: cast gradient to float32, update master weights
gradient_fp32 = np.float32(gradient_fp16)
learning_rate = np.float32(1.0)
weights_fp32 -= learning_rate * gradient_fp32

print(f"Updated weight (float32 master): {weights_fp32}")
# 0.9999 — the update is preserved

# What would happen without master weights?
weights_fp16_only = np.float16(1.0)
weights_fp16_only -= np.float16(1.0) * np.float16(0.0001)
print(f"Updated weight (float16 only):   {weights_fp16_only}")
# 1.0 — the update was lost!
```

This is why NVIDIA's Automatic Mixed Precision (AMP) keeps a float32 "master copy" of weights. The forward and backward passes happen in float16 for speed, but the actual weight updates happen in float32 for precision. Best of both worlds.

### Loss Scaling: One More Trick

There's a subtlety: float16 can't represent numbers smaller than $\sim 6 \times 10^{-8}$ (smallest denormalized value). Many gradients are smaller than this and would underflow to zero. The fix is **loss scaling**: multiply the loss by a large factor (say, 1024) before the backward pass, which scales all gradients up into the representable range. Then divide by the same factor when updating weights.

```python
# Without loss scaling: gradient underflows in float16
gradient = np.float16(1e-8)  # Becomes 0.0 in float16

# With loss scaling:
loss_scale = np.float32(1024.0)
scaled_gradient = np.float16(1e-8 * 1024)  # 1.024e-5, representable!
# After update, divide by loss_scale to get correct magnitude
actual_update = np.float32(scaled_gradient) / loss_scale
```

---

## Code: Exploring Number Systems and Their Limits

```python
import numpy as np

# ============================================================
# Part 1: The Number Hierarchy as Type Hierarchy
# ============================================================
print("=" * 60)
print("PART 1: Number Hierarchy → Type Hierarchy")
print("=" * 60)

# Natural numbers → unsigned integers
natural = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
print(f"Natural numbers (uint32): {natural}")
print(f"  Memory per element: {natural.itemsize} bytes")

# Integers → signed integers
integers = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
print(f"Integers (int32): {integers}")

# Rational numbers → floats (approximate!)
rational = np.array([1/3, 1/4, 2/7], dtype=np.float64)
print(f"Rational (float64): {rational}")
print(f"  1/3 stored as: {rational[0]:.20f}")
print(f"  Exact 1/3:     0.33333333333333333333...")
print(f"  They differ at digit ~16 (float64 precision limit)")

# Real numbers → also floats (even more approximate!)
reals = np.array([np.pi, np.e, np.sqrt(2)], dtype=np.float64)
print(f"Reals (float64): pi={reals[0]:.15f}, e={reals[1]:.15f}")

# ============================================================
# Part 2: The 0.1 + 0.2 Problem (Yes, Again)
# ============================================================
print(f"\n{'=' * 60}")
print("PART 2: Why 0.1 + 0.2 != 0.3")
print("=" * 60)

result = 0.1 + 0.2
print(f"0.1 + 0.2 = {result}")
print(f"0.1 + 0.2 == 0.3? {result == 0.3}")  # False!
print(f"Difference: {result - 0.3:.20e}")
print(f"np.allclose(0.1 + 0.2, 0.3)? {np.allclose(result, 0.3)}")  # True

# ============================================================
# Part 3: Machine Epsilon
# ============================================================
print(f"\n{'=' * 60}")
print("PART 3: Machine Epsilon")
print("=" * 60)

for dtype in [np.float16, np.float32, np.float64]:
    info = np.finfo(dtype)
    print(f"{dtype.__name__}:")
    print(f"  Machine epsilon: {info.eps}")
    print(f"  Smallest normal: {info.tiny}")
    print(f"  Largest value:   {info.max}")
    print(f"  Decimal digits:  {info.nmant * np.log10(2):.1f}")

# ============================================================
# Part 4: Error Accumulation (The Gradient Problem)
# ============================================================
print(f"\n{'=' * 60}")
print("PART 4: Error Accumulation")
print("=" * 60)

for dtype, name in [(np.float16, "float16"), (np.float32, "float32"),
                     (np.float64, "float64")]:
    total = dtype(0.0)
    n = 10000
    for _ in range(n):
        total += dtype(0.1)
    expected = n * 0.1
    error = abs(float(total) - expected)
    print(f"Sum of 0.1 x {n} in {name}: {float(total):.4f} "
          f"(expected {expected:.1f}, error {error:.4f})")

# ============================================================
# Part 5: Special Values — Where NaN Comes From
# ============================================================
print(f"\n{'=' * 60}")
print("PART 5: Special Values (Sources of NaN)")
print("=" * 60)

print(f"inf:       {np.inf}")
print(f"-inf:      {-np.inf}")
print(f"nan:       {np.nan}")
print(f"inf - inf: {np.inf - np.inf}")     # NaN
print(f"0 * inf:   {0.0 * np.inf}")        # NaN
print(f"inf / inf: {np.inf / np.inf}")      # NaN
print(f"NaN == NaN? {np.nan == np.nan}")    # False! NaN != everything
print(f"Check NaN:  {np.isnan(np.nan)}")    # Use isnan() instead

# ============================================================
# Part 6: Softmax Stability
# ============================================================
print(f"\n{'=' * 60}")
print("PART 6: Softmax Stability")
print("=" * 60)

def unstable_softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def stable_softmax(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

logits_small = np.array([1.0, 2.0, 3.0], dtype=np.float32)
logits_large = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)

print(f"Small logits - Unstable: {unstable_softmax(logits_small)}")
print(f"Small logits - Stable:   {stable_softmax(logits_small)}")
print(f"Large logits - Unstable: {unstable_softmax(logits_large)}")
print(f"Large logits - Stable:   {stable_softmax(logits_large)}")

# ============================================================
# Part 7: Mixed-Precision Demo
# ============================================================
print(f"\n{'=' * 60}")
print("PART 7: Mixed-Precision Training Simulation")
print("=" * 60)

# Simulate: can float16 capture a typical weight update?
weight = 1.0
lr = 1e-4
gradient = 0.5
update = lr * gradient  # 0.00005

w16 = np.float16(weight)
w32 = np.float32(weight)

w16_updated = np.float16(float(w16) - update)
w32_updated = np.float32(float(w32) - update)

print(f"Weight: {weight}")
print(f"Update: lr * grad = {update}")
print(f"float16 after update: {float(w16_updated):.6f} "
      f"(changed: {float(w16) != float(w16_updated)})")
print(f"float32 after update: {float(w32_updated):.6f} "
      f"(changed: {float(w32) != float(w32_updated)})")
```

---

## Common Mistakes

Here are the traps that catch even experienced engineers:

**1. float32 has only ~7 digits of precision. Summing 1 million small numbers can lose most of them.**

This is the single most common numerical bug in ML code. If you're accumulating gradients, losses, or metrics across a large batch or many steps, use float64 for the accumulator — even if everything else is float32.

**2. Comparing floats with `==`**

Never do this. Use `np.allclose(a, b, rtol=1e-5, atol=1e-8)` or `torch.allclose()`. If you're writing unit tests for numerical code and using exact equality, your tests are wrong.

**3. Forgetting that NaN is contagious**

One NaN in your computation poisons everything downstream. NaN + anything = NaN. NaN * anything = NaN. And the worst part: NaN != NaN, so `if x != x` is actually a (terrible) NaN check. Use `np.isnan()` or `torch.isnan()`.

**4. Ignoring overflow in integer operations**

Python `int` has arbitrary precision. NumPy `int32` wraps around silently:

```python
np.int32(2_147_483_647) + np.int32(1)  # → -2147483648 (wrapped!)
```

**5. Not accounting for float16/bfloat16 limitations**

float16 max value is 65,504. If your logits exceed this, they become `inf`, and softmax produces NaN. bfloat16 has the range of float32 but only ~2-3 digits of precision — useful for gradients, dangerous for anything requiring accuracy.

---

## Exercises

### Exercise 1: Find Machine Epsilon Yourself

**Problem**: Write code to find the smallest positive float32 number `x` such that `1.0 + x != 1.0`. Compare it with `np.finfo(np.float32).eps`.

**Solution**:
```python
import numpy as np

x = np.float32(1.0)
while np.float32(1.0) + x != np.float32(1.0):
    last_x = x
    x = x / np.float32(2.0)

print(f"Found epsilon:   {last_x}")
print(f"np.finfo gives:  {np.finfo(np.float32).eps}")
# They should match: ~1.1920929e-07
```

**Why this matters**: Machine epsilon tells you the fundamental resolution of your number format. When you see `rtol=1e-5` in `np.allclose()`, you're choosing a tolerance that's about 100x larger than float32's epsilon — a reasonable margin for accumulated errors.

### Exercise 2: Safe Logarithm for Cross-Entropy

**Problem**: Cross-entropy loss computes $-\sum y_i \log(\hat{y}_i)$. If any $\hat{y}_i = 0$, you get $\log(0) = -\infty$. Implement a safe log function.

**Solution**:
```python
import numpy as np

def safe_log(x, epsilon=1e-10):
    """Compute log safely, clamping inputs away from zero."""
    return np.log(np.maximum(x, epsilon))

# Test with edge cases
predictions = np.array([0.0, 1e-50, 0.5, 1.0])
print(f"Predictions:  {predictions}")
print(f"Regular log:  {np.log(predictions + 1e-300)}")  # -inf for 0.0
print(f"Safe log:     {safe_log(predictions)}")          # Finite for all

# In practice, this is what PyTorch's F.cross_entropy does internally
# (it uses log-sum-exp for numerical stability)
```

### Exercise 3: Kahan Summation — Beating Naive Accumulation

**Problem**: Implement the Kahan summation algorithm to reduce floating-point error when summing many numbers. Compare with naive summation and NumPy's built-in `sum`.

**Solution**:
```python
import numpy as np

def naive_sum(arr):
    """Simple loop summation — accumulates error."""
    total = 0.0
    for x in arr:
        total += x
    return total

def kahan_sum(arr):
    """Kahan summation — compensates for lost precision."""
    total = 0.0
    compensation = 0.0  # Running compensation for lost low-order bits

    for x in arr:
        y = x - compensation       # Add the compensation to the next value
        t = total + y              # New (imprecise) total
        compensation = (t - total) - y  # Recover what was lost
        total = t

    return total

# Test: sum 10 million copies of 0.1
n = 10_000_000
arr = np.full(n, 0.1, dtype=np.float64)
exact = n * 0.1

naive = naive_sum(arr)
kahan = kahan_sum(arr)
numpy_result = np.sum(arr)

print(f"Expected:  {exact}")
print(f"Naive sum: {naive:.10f}  (error: {abs(naive - exact):.10f})")
print(f"Kahan sum: {kahan:.10f}  (error: {abs(kahan - exact):.10f})")
print(f"NumPy sum: {numpy_result:.10f}  (error: {abs(numpy_result - exact):.10f})")

# NumPy uses pairwise summation internally, which is also good.
# Kahan is better for streaming/online accumulation.
```

**Why this matters**: Gradient accumulation across many microbatches is exactly this problem. Frameworks like PyTorch handle this for you in most cases, but if you're writing custom training loops or accumulating metrics manually, you need to know this.

---

## Summary

Here's what you need to take away from this chapter:

- **Number hierarchy** ($\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R}$) maps to a **type hierarchy** (`uint` → `int` → `float` → `complex`). Each level extends what you can represent.
- **IEEE 754** is how every computer approximates real numbers. A float32 has 1 sign bit, 8 exponent bits, and 23 mantissa bits.
- **Precision is limited**: float32 gives ~7 decimal digits, float64 gives ~15-16. This isn't about range — it's about *resolution*.
- **Floats are not uniformly spaced**. They're dense near zero and sparse far from it. This is why large logits cause overflow and why small updates vanish.
- **Machine epsilon** ($\varepsilon \approx 1.19 \times 10^{-7}$ for float32) is the fundamental precision limit. Use tolerance-based comparison (`np.allclose`), not `==`.
- **Mixed-precision training** works because forward/backward passes tolerate low precision (float16), but weight updates require high precision (float32). Master weights, loss scaling, and careful accumulation make it practical.
- **NaN is not random**. It comes from overflow, 0/0, inf-inf, or propagation. If your loss is NaN, trace it back to one of these.

---

**What's Next** — Numbers alone aren't enough. You need to combine them: add, multiply, take absolute values. Arithmetic operations and their properties are the foundation of everything from loss functions to gradient updates.

*Next: [Chapter 2: Arithmetic Operations](./02-arithmetic-operations.md)*
