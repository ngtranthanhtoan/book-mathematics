# Arithmetic Operations

Addition, subtraction, multiplication, division — sounds like grade school? But here's what grade school didn't tell you: the ORDER of these operations affects numerical stability, the PROPERTIES of these operations enable vectorized GPU computation, and the EDGE CASES (division by zero, overflow) will crash your training loop.

> **Building On** — You know how computers store numbers. Now: what happens when you combine them? Arithmetic operations seem simple, but their properties (and failure modes) are the foundation of numerical computing.

---

## Running Example: Gradient Accumulation

Throughout this chapter, we'll use a single scenario to ground every concept:

> **Scenario**: You're training a large language model. Your GPU can't fit the full batch, so you split it into 1,000 micro-batches and **accumulate gradients** — summing 1,000 small gradient tensors into one. That's a million additions per parameter, per training step.

This is where arithmetic gets real:
- **Addition order** matters — naive summation loses precision after hundreds of terms.
- **Multiplication by the learning rate** can overflow or underflow.
- **Division** (averaging the gradients) can produce NaN if your denominator is wrong.
- **Absolute value** shows up when you clip gradients or compute L1 norms.

Keep this scenario in mind. We'll return to it in every section.

---

## Addition and Subtraction

### The Math

**Addition** combines quantities:

$$a + b = c$$

**Subtraction** is addition of the additive inverse:

$$a - b = a + (-b)$$

### Properties of Addition

| Property | Definition | Holds in Exact Math? | Holds in float32? |
|----------|-----------|---------------------|--------------------|
| Commutative | $a + b = b + a$ | Yes | Yes |
| Associative | $(a + b) + c = a + (b + c)$ | Yes | **NO** |
| Identity | $a + 0 = a$ | Yes | Yes |
| Inverse | $a + (-a) = 0$ | Yes | Usually (see subnormals) |

**Subtraction** is NOT commutative ($a - b \neq b - a$) and NOT associative.

### Vector/Matrix Addition (Element-wise)

$$\mathbf{A} + \mathbf{B} = \begin{bmatrix} a_{11} + b_{11} & a_{12} + b_{12} \\ a_{21} + b_{21} & a_{22} + b_{22} \end{bmatrix}$$

This is element-wise. Both matrices must have the same shape (or be broadcastable — more on that below).

> **You Already Know This: Commutativity**
>
> `a + b == b + a` feels obvious. But here's the trap: when you move to matrix multiplication later, this intuition **breaks**. $AB \neq BA$ in general. Enjoy commutativity while you have it — addition is one of the few operations where order doesn't matter (mathematically). Computationally, as you're about to see, even addition's order matters.

---

### Where Addition Breaks: Floating-Point Associativity

> **Common Mistake** — Floating-point addition is NOT associative. `(a + b) + c != a + (b + c)` in float32. This breaks reproducibility.

Here's the proof you can run yourself:

```python
import numpy as np

# Floating-point associativity failure
a = np.float32(1e8)
b = np.float32(1.0)
c = np.float32(-1e8)

left  = (a + b) + c   # (1e8 + 1.0) + (-1e8)
right = a + (b + c)   # 1e8 + (1.0 + (-1e8))

print(f"(a + b) + c = {left}")   # 0.0  — the 1.0 got absorbed into 1e8
print(f"a + (b + c) = {right}")  # 0.0  — same here? Actually...

# More dramatic example:
a = np.float32(1e-8)
b = np.float32(1.0)
c = np.float32(-1.0)

left  = (a + b) + c   # (1e-8 + 1.0) + (-1.0)
right = a + (b + c)   # 1e-8 + (1.0 + (-1.0))

print(f"\n(a + b) + c = {left}")   # 0.0       — WRONG, lost the 1e-8
print(f"a + (b + c) = {right}")    # 1e-08     — correct
print(f"They're equal? {left == right}")  # False
```

Why does this matter? Because **GPU parallelism changes the order of additions**. Run the same model on two different GPUs, get different results. This is why PyTorch has `torch.use_deterministic_algorithms()`.

### Back to Our Running Example: Kahan Summation

You're summing 1,000 micro-batch gradients. Each gradient is small (say, ~1e-5). The accumulator grows. Eventually, adding a tiny gradient to a large accumulator loses the gradient entirely — **catastrophic cancellation**.

**Kahan summation** fixes this by tracking the rounding error:

```python
import numpy as np

def naive_sum(values):
    """Standard summation — loses precision for many small values."""
    total = np.float32(0.0)
    for v in values:
        total += np.float32(v)
    return total

def kahan_sum(values):
    """Kahan summation — compensates for floating-point error."""
    total = np.float32(0.0)
    compensation = np.float32(0.0)  # Running error compensation

    for v in values:
        v = np.float32(v)
        y = v - compensation          # Compensated value
        t = total + y                  # New total (may lose low-order bits of y)
        compensation = (t - total) - y # Recover what was lost
        total = t

    return total

# Simulate gradient accumulation: 1000 micro-batches
np.random.seed(42)
gradients = np.random.randn(1000).astype(np.float32) * 1e-4

exact_sum    = np.float64(gradients.astype(np.float64).sum())
naive_result = naive_sum(gradients)
kahan_result = kahan_sum(gradients)

print("=== Gradient Accumulation: Naive vs. Kahan ===")
print(f"Exact sum (float64):  {exact_sum:.10f}")
print(f"Naive sum (float32):  {naive_result:.10f}")
print(f"Kahan sum (float32):  {kahan_result:.10f}")
print(f"Naive error:          {abs(float(naive_result) - exact_sum):.2e}")
print(f"Kahan error:          {abs(float(kahan_result) - exact_sum):.2e}")
```

**Takeaway**: If you're accumulating gradients across many micro-batches in float32, use compensated summation or accumulate in float64. PyTorch's `GradScaler` in mixed-precision training handles a related problem.

---

### Overflow and Underflow: The Number Line Has Edges

When you add or multiply numbers, you can fall off the representable range:

```
UNDERFLOW                                              OVERFLOW
   ◄──────────────────────────────────────────────────────►

   -inf ◄── -3.4e38 ────── -1e-45 ── 0 ── 1e-45 ──────── 3.4e38 ──► +inf
              │                │           │                │
              │  Representable │  Subnormal│  Representable │
              │  float32 range │  (losing  │  float32 range │
              │                │ precision)│                │
              ▼                ▼           ▼                ▼
         Large negative    Tiny         Tiny          Large positive
         numbers          negative     positive       numbers

   Adding two large positives → OVERFLOW → +inf
   Multiplying two tiny numbers → UNDERFLOW → 0.0 (silent precision loss!)
```

```python
import numpy as np

# Overflow
big = np.float32(3.4e38)
print(f"big + big = {big + big}")        # inf

# Underflow (gradual)
tiny = np.float32(1e-40)
print(f"tiny * 0.1 = {tiny * np.float32(0.1)}")  # 0.0 — silently gone

# This is why log-space computation exists:
# Instead of multiplying tiny probabilities, ADD their logs
probs = np.array([1e-30, 1e-30, 1e-30], dtype=np.float32)
print(f"Product of tiny probs: {np.prod(probs)}")              # 0.0 (underflow)
print(f"Sum of log probs:      {np.sum(np.log(probs)):.1f}")   # -207.2 (correct!)
```

---

## Multiplication and Division

### The Math

**Multiplication** scales or combines:

$$a \times b = c \quad \text{or} \quad a \cdot b = c \quad \text{or} \quad ab = c$$

**Division** is multiplication by the multiplicative inverse:

$$\frac{a}{b} = a \cdot b^{-1} = a \cdot \frac{1}{b}, \quad b \neq 0$$

### Properties of Multiplication

| Property | Definition | Holds in Exact Math? | Holds in float32? |
|----------|-----------|---------------------|--------------------|
| Commutative | $ab = ba$ | Yes | Yes (for scalars) |
| Associative | $(ab)c = a(bc)$ | Yes | Approximately |
| Identity | $a \cdot 1 = a$ | Yes | Yes |
| Zero property | $a \cdot 0 = 0$ | Yes | Yes (except NaN) |
| Distributive | $a(b + c) = ab + ac$ | Yes | Approximately |

> **You Already Know This: Commutativity Breaks for Matrices**
>
> Scalar multiplication is commutative: `3 * 5 == 5 * 3`. But matrix multiplication is NOT:
>
> $$AB \neq BA \text{ (in general)}$$
>
> If you've ever written `np.dot(W, x)` and gotten a shape error, then swapped to `np.dot(x, W)` and it worked — that's non-commutativity biting you. The order of operands in matrix multiplication is not interchangeable. This will become critical when we reach linear algebra.

> **You Already Know This: Distributive Property Enables Broadcasting**
>
> The distributive law $a(b + c) = ab + ac$ is the algebraic foundation of **broadcasting** and **vectorization**:
>
> ```python
> # Instead of this (two operations):
> result = a * b + a * c
>
> # The GPU does this (one fused operation):
> result = a * (b + c)
> ```
>
> NumPy's broadcasting rules are essentially the distributive property applied at scale across array dimensions. When you write `learning_rate * gradients`, that single scalar multiplies every element — distributive property at work.

### Division: The Operation That Bites Back

**Division by zero is undefined.** In computing, it produces `inf`, `-inf`, or `NaN`, and any of these will poison your entire tensor.

```python
import numpy as np

# The many faces of division-by-zero
print(f" 1.0 / 0.0 = {np.float64(1.0) / np.float64(0.0)}")    # inf
print(f"-1.0 / 0.0 = {np.float64(-1.0) / np.float64(0.0)}")   # -inf
print(f" 0.0 / 0.0 = {np.float64(0.0) / np.float64(0.0)}")    # nan

# NaN is contagious — it infects everything it touches
x = np.array([1.0, np.nan, 3.0])
print(f"sum with NaN: {np.sum(x)}")       # nan
print(f"mean with NaN: {np.mean(x)}")     # nan
print(f"NaN == NaN: {np.nan == np.nan}")   # False (!)
```

### Safe Division Pattern

This is defensive code you'll write constantly in ML:

```python
import numpy as np

def safe_divide(numerator, denominator, epsilon=1e-8):
    """Safely divide, adding epsilon to prevent division by zero.

    This is the standard pattern in ML codebases.
    You'll see it in: BatchNorm, LayerNorm, attention scores,
    any normalization, any ratio computation.
    """
    return numerator / (denominator + epsilon)

# Example: normalizing a vector (making it unit length)
v = np.array([3.0, 4.0, 0.0])
norm = np.linalg.norm(v)
print(f"v / ||v|| = {v / norm}")

# What if the vector is zero?
v_zero = np.array([0.0, 0.0, 0.0])
norm_zero = np.linalg.norm(v_zero)
# print(f"v / ||v|| = {v_zero / norm_zero}")  # NaN! NaN! NaN!
print(f"v / (||v|| + eps) = {safe_divide(v_zero, norm_zero)}")  # Safe: [0, 0, 0]
```

### Running Example: Learning Rate * Gradient

Back to gradient accumulation. After summing your 1,000 micro-batch gradients, you multiply by the learning rate:

$$\theta_{new} = \theta_{old} - \eta \cdot \nabla L$$

If $\nabla L$ is large (gradient explosion) and $\eta$ isn't tiny enough, the multiplication overflows. If $\nabla L$ is tiny (vanishing gradients) and $\eta$ is also small, the multiplication underflows to zero — your model stops learning with no error message.

---

## Modulo Operation

### The Math

The **modulo** operation returns the remainder of integer division:

$$a \mod n = r$$

where $a = qn + r$ and $0 \leq r < n$ (for positive $n$).

**Examples:**
- $17 \mod 5 = 2$ (because $17 = 3 \times 5 + 2$)
- $-17 \mod 5 = 3$ (in Python, result has same sign as divisor)

**Properties of Modulo:**
- $(a + b) \mod n = ((a \mod n) + (b \mod n)) \mod n$
- $(a \times b) \mod n = ((a \mod n) \times (b \mod n)) \mod n$

> **You Already Know This: Modulo Is Everywhere in Systems Programming**
>
> You've used modulo for years — hash tables (`hash(key) % num_buckets`), circular buffers (`buffer[idx % size]`), even checking odd/even (`n % 2`). In ML, the same pattern shows up in:
>
> - **Batch indexing with wrap-around**: cycling through a dataset that doesn't divide evenly by batch size
> - **Cyclic learning rate schedules**: `epoch % cycle_length` gives position within the cycle
> - **Positional encodings in transformers**: `sin(pos / 10000^(2i/d))` uses periodicity (modular structure)
> - **Data sharding**: `sample_id % num_workers` assigns data to workers

### Modulo Gotcha: Negative Numbers

Behavior differs between languages, and this is a classic bug source:

```python
# Python: result has the same sign as the DIVISOR
print(f"-7 % 3 = {-7 % 3}")    # 2  (positive, matching divisor sign)
print(f" 7 % -3 = {7 % -3}")   # -2 (negative, matching divisor sign)

# C/C++/Java: result has the same sign as the DIVIDEND
# -7 % 3 would give -1 in C

# Euclidean modulo (always non-negative) — safest for indexing:
def euclidean_mod(a, n):
    """Always returns a value in [0, |n|). Safe for array indexing."""
    return ((a % n) + n) % n

print(f"euclidean_mod(-7, 3) = {euclidean_mod(-7, 3)}")  # 2
print(f"euclidean_mod(-1, 5) = {euclidean_mod(-1, 5)}")  # 4
```

### Modulo in Practice: Cyclic Learning Rate

```python
import numpy as np

def cyclic_lr(epoch, base_lr=0.001, max_lr=0.01, step_size=10):
    """Cyclic learning rate using modulo for the cycle position."""
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, 1 - x)

# Batch indexing with wrap-around
dataset_size = 100
batch_size = 32

for i in range(5):
    start_idx = (i * batch_size) % dataset_size
    end_idx = ((i + 1) * batch_size) % dataset_size
    wraps = "  ← wraps around!" if end_idx <= start_idx else ""
    print(f"Batch {i}: indices [{start_idx}, {end_idx}){wraps}")
```

---

## Absolute Value

### The Math

The **absolute value** gives the magnitude (distance from zero):

$$|x| = \begin{cases} x & \text{if } x \geq 0 \\ -x & \text{if } x < 0 \end{cases}$$

**Properties:**
- $|x| \geq 0$ (always non-negative)
- $|x| = 0 \iff x = 0$
- $|xy| = |x||y|$ (multiplicative)
- $|x + y| \leq |x| + |y|$ (triangle inequality — fundamental!)

**For complex numbers:**
$$|a + bi| = \sqrt{a^2 + b^2}$$

> **You Already Know This: Absolute Value = Distance = Norms**
>
> Absolute value for scalars generalizes to **norms** for vectors:
>
> | Scalar | Vector Equivalent | Name | ML Use |
> |--------|------------------|------|--------|
> | $\|x\|$ | $\sum_i \|x_i\|$ | L1 norm | Lasso regression, MAE loss, sparsity |
> | $x^2$ | $\sum_i x_i^2$ | L2 norm (squared) | Ridge regression, MSE loss |
> | — | $\max_i \|x_i\|$ | L-infinity norm | Gradient clipping |
>
> When you see `torch.nn.L1Loss()`, that's absolute value. When you see "Manhattan distance," that's summed absolute values. The triangle inequality ($|x + y| \leq |x| + |y|$) is why the L1 norm is actually a valid norm — it satisfies the mathematical requirements.

### Absolute Value in Loss Functions

```python
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5,  0.0, 2.1, 7.8])

# MAE (L1 Loss) — uses absolute value
mae = np.mean(np.abs(y_true - y_pred))

# MSE (L2 Loss) — uses squaring instead
mse = np.mean((y_true - y_pred) ** 2)

# Huber Loss — combines both: L2 near zero, L1 far from zero
def huber_loss(y_true, y_pred, delta=1.0):
    """Quadratic for small errors, linear for large errors.
    Best of both worlds: smooth gradient near zero (L2),
    robust to outliers far from zero (L1).
    """
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = 0.5 * error ** 2
    linear = delta * abs_error - 0.5 * delta ** 2
    return np.where(abs_error <= delta, quadratic, linear)

print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
print(f"MAE (L1 Loss):  {mae:.4f}")
print(f"MSE (L2 Loss):  {mse:.4f}")
print(f"Huber Loss:     {np.mean(huber_loss(y_true, y_pred)):.4f}")
```

### The Gradient of Absolute Value: Not Smooth

One subtle issue: $|x|$ is NOT differentiable at $x = 0$. Its derivative is:

$$\frac{d|x|}{dx} = \begin{cases} +1 & \text{if } x > 0 \\ -1 & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$$

In practice, frameworks use a **subgradient** (typically 0) at $x = 0$. This is why Huber loss exists — it smooths out the kink.

```python
import numpy as np

def mae_forward(y_true, y_pred):
    """Forward pass: compute MAE."""
    return np.mean(np.abs(y_true - y_pred))

def mae_backward(y_true, y_pred):
    """Backward pass: gradient of MAE w.r.t. y_pred.
    Gradient of |x| is sign(x). Gradient of mean is 1/n.
    """
    n = len(y_true)
    return np.sign(y_pred - y_true) / n

y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.5, 1.8, 3.2, 3.5])

loss = mae_forward(y_true, y_pred)
grad = mae_backward(y_true, y_pred)

print(f"MAE: {loss:.4f}")
print(f"Gradient: {grad}")
print(f"Sum of |gradient|: {np.sum(np.abs(grad)):.4f}")  # Always 1.0
```

---

## Putting It All Together: Operations in ML Algorithms

### Where Each Operation Appears

| Operation | ML Application | Why It Matters |
|-----------|---------------|----------------|
| Addition | Bias terms, residual connections, gradient accumulation | Associativity failure affects reproducibility |
| Subtraction | Error computation, normalization (centering) | Catastrophic cancellation with similar values |
| Multiplication | Weights, attention scores, learning rate | Overflow/underflow risk in deep networks |
| Division | Normalization, softmax, averaging | Division by zero produces NaN |
| Modulo | Batch indexing, cyclic schedules, hashing | Negative number behavior varies by language |
| Absolute Value | L1 loss, gradient clipping, MAE | Non-differentiable at zero |

### Neural Network Forward Pass: All Operations at Work

```
Input x ──→ [Multiply by W] ──→ [Add bias b] ──→ [Activation] ──→ Output
            (multiplication)      (addition)        f(Wx + b)

Loss = |y_true - y_pred|      (subtraction + absolute value = MAE)
Loss = (y_true - y_pred)²     (subtraction + multiplication = MSE)
```

### Complete Example: All Operations in Real ML Code

```python
import numpy as np

# ============================================
# 1. Linear Layer: y = Wx + b
#    Uses: multiplication (matmul) + addition
# ============================================
def linear_forward(x, W, b):
    return np.dot(x, W) + b

# ============================================
# 2. Batch Normalization (simplified)
#    Uses: subtraction, division, multiplication
# ============================================
def batch_norm(x, epsilon=1e-5):
    mean = np.mean(x, axis=0)          # division (mean = sum/n)
    var = np.var(x, axis=0)            # subtraction + multiplication + division
    x_norm = (x - mean) / np.sqrt(var + epsilon)  # subtraction + division
    return x_norm

# ============================================
# 3. L1 Regularization
#    Uses: absolute value + addition (summation)
# ============================================
def l1_regularization(weights, lambda_param=0.01):
    return lambda_param * np.sum(np.abs(weights))

# ============================================
# 4. Numerically Stable Softmax
#    Uses: subtraction, multiplication (exp), division
# ============================================
def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)  # subtract max for stability
    exp_x = np.exp(x_shifted)                           # exponentiation
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)  # division (normalize)

# ============================================
# 5. Cyclic Learning Rate
#    Uses: modulo, division, absolute value, multiplication
# ============================================
def cyclic_lr(epoch, base_lr=0.001, max_lr=0.01, step_size=10):
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, 1 - x)

# ============================================
# Test everything
# ============================================
np.random.seed(42)
x = np.random.randn(4, 3)   # 4 samples, 3 features
W = np.random.randn(3, 2)   # 3 input, 2 output
b = np.random.randn(2)

print("=== All Operations in Action ===")
print(f"Linear output shape:  {linear_forward(x, W, b).shape}")
print(f"BatchNorm output shape: {batch_norm(x).shape}")
print(f"L1 regularization:    {l1_regularization(W):.4f}")

logits = np.array([[1000, 1001, 1002], [1, 2, 3]])
print(f"Softmax (large logits): {softmax(logits)}")
print(f"Softmax sums to 1?    {softmax(logits).sum(axis=1)}")

for epoch in range(0, 30, 5):
    print(f"Epoch {epoch}: lr = {cyclic_lr(epoch):.6f}")
```

---

## Vectorized Operations and Broadcasting

Every operation we've discussed works element-wise on arrays, and NumPy/PyTorch broadcast automatically:

```python
import numpy as np

# ============================================
# Element-wise Operations
# ============================================
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])

print("=== Element-wise Operations ===")
print(f"x + y = {x + y}")        # [6, 6, 6, 6, 6]
print(f"x - y = {x - y}")        # [-4, -2, 0, 2, 4]
print(f"x * y = {x * y}")        # [5, 8, 9, 8, 5] — element-wise, NOT dot product!
print(f"x / y = {x / y}")        # [0.2, 0.5, 1.0, 2.0, 5.0]
print(f"x % y = {x % y}")        # [1, 2, 0, 0, 0]
print(f"|x-y| = {np.abs(x - y)}")  # [4, 2, 0, 2, 4]

# ============================================
# Broadcasting: Scalar × Matrix
# The distributive property at scale
# ============================================
print("\n=== Broadcasting ===")

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
scalar = 10
vector = np.array([1, 0, -1])

print(f"Matrix + scalar ({scalar}):\n{matrix + scalar}")
print(f"\nMatrix * vector ({vector}):\n{matrix * vector}")

# This is real ML: centering data by subtracting the row mean
row_means = matrix.mean(axis=1, keepdims=True)
print(f"\nMatrix - row mean (centering):\n{matrix - row_means}")
```

---

## Computational Cost

All basic operations are O(1) for scalars and O(n) for n-element arrays. But the constants differ:

```python
import numpy as np
import time

n = 10_000_000
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32) + 0.1  # Avoid zeros for division

operations = [
    ("Addition",       lambda: a + b),
    ("Subtraction",    lambda: a - b),
    ("Multiplication", lambda: a * b),
    ("Division",       lambda: a / b),
    ("Modulo",         lambda: np.mod(a, np.abs(b))),
    ("Absolute value", lambda: np.abs(a)),
]

print("=== Operation Timing (10M float32 elements) ===")
for name, op in operations:
    start = time.time()
    for _ in range(10):
        _ = op()
    elapsed = (time.time() - start) / 10
    print(f"{name:18s} {elapsed*1000:.2f} ms")
```

On most hardware, addition/subtraction/multiplication are roughly the same speed. Division is slower (no dedicated fast circuit on all hardware). Modulo is slowest. This matters when you're choosing between mathematically equivalent formulations.

---

## Common Pitfalls Reference

### 1. Division by Zero — Always Add Epsilon

```python
# BAD — will produce inf or NaN
result = a / b

# GOOD — standard ML defensive pattern
result = a / (b + 1e-8)
```

### 2. Integer Division Surprise

```python
# Python 3 distinction:
5 / 2    # 2.5 (float division)
5 // 2   # 2   (floor division)

# Trap: NumPy integer arrays use integer division with /
a = np.array([5], dtype=np.int32)
b = np.array([2], dtype=np.int32)
# Use a.astype(np.float32) / b if you want 2.5
```

### 3. Floating-Point Associativity — Sum Order Matters

```python
# For large sums (like gradient accumulation), sort small-to-large:
sorted_values = sorted(values, key=abs)
total = sum(sorted_values)

# Or use Kahan summation (see above)
# Or accumulate in float64, then cast back to float32
```

### 4. Broadcasting Shape Mismatch

```python
a = np.array([[1, 2], [3, 4]])  # shape (2, 2)
b = np.array([1, 2, 3])          # shape (3,)
# a + b → ERROR: shapes (2,2) and (3,) don't broadcast
# Fix: ensure trailing dimensions match or are 1
```

### 5. NaN Propagation — One Bad Value Poisons Everything

```python
# Always check for NaN in training loops:
if np.any(np.isnan(loss)):
    print("NaN detected! Check for division by zero or overflow.")
```

---

## Exercises

### Exercise 1: Implement Kahan Summation for Gradient Accumulation

**Problem**: You have 10,000 gradient values in float32. Compare naive summation, sorted summation, and Kahan summation against a float64 reference.

**Solution**:
```python
import numpy as np

def naive_sum(values):
    total = np.float32(0.0)
    for v in values:
        total += np.float32(v)
    return total

def sorted_sum(values):
    sorted_vals = sorted(values, key=lambda x: abs(float(x)))
    return naive_sum(sorted_vals)

def kahan_sum(values):
    total = np.float32(0.0)
    compensation = np.float32(0.0)
    for v in values:
        v = np.float32(v)
        y = v - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total

# Simulate 10,000 micro-batch gradients
np.random.seed(42)
gradients = np.random.randn(10000).astype(np.float32) * 1e-4

exact = np.float64(gradients.astype(np.float64).sum())
print(f"Exact (float64): {exact:.12f}")
print(f"Naive (float32): {naive_sum(gradients):.12f}  error={abs(float(naive_sum(gradients))-exact):.2e}")
print(f"Sorted (float32):{sorted_sum(gradients):.12f}  error={abs(float(sorted_sum(gradients))-exact):.2e}")
print(f"Kahan (float32): {kahan_sum(gradients):.12f}  error={abs(float(kahan_sum(gradients))-exact):.2e}")
```

### Exercise 2: Numerically Stable Softmax

**Problem**: Implement softmax that handles extreme values (logits of 1000+).

**Solution**:
```python
import numpy as np

def softmax(x):
    """Numerically stable softmax.

    The trick: subtract max(x) before exponentiating.
    This prevents overflow (e^1000 = inf) while giving identical results,
    because softmax(x) = softmax(x - c) for any constant c.
    """
    x_shifted = x - np.max(x, axis=-1, keepdims=True)  # subtraction
    exp_x = np.exp(x_shifted)                           # exponentiation
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)  # division

# Test with extreme values
logits = np.array([[1000, 1001, 1002], [1, 2, 3]])
print(f"Input logits:\n{logits}")
print(f"Softmax output:\n{softmax(logits)}")
print(f"Row sums (should be 1.0): {softmax(logits).sum(axis=1)}")
```

### Exercise 3: Safe Batch Normalization

**Problem**: Implement batch normalization that handles edge cases (zero variance, single sample).

**Solution**:
```python
import numpy as np

def safe_batch_norm(x, epsilon=1e-5, gamma=None, beta=None):
    """Batch normalization with safe division.

    Handles:
    - Zero variance (constant features) via epsilon
    - Single sample (variance=0) via epsilon
    - Optional learnable scale (gamma) and shift (beta)
    """
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)

    # Safe division: epsilon prevents division by zero when var=0
    x_norm = (x - mean) / np.sqrt(var + epsilon)

    # Optional affine transform (learnable in real implementations)
    if gamma is not None:
        x_norm = gamma * x_norm
    if beta is not None:
        x_norm = x_norm + beta

    return x_norm

# Normal case
x = np.random.randn(32, 4)  # 32 samples, 4 features
print(f"BatchNorm output mean: {safe_batch_norm(x).mean(axis=0)}")  # ~0
print(f"BatchNorm output std:  {safe_batch_norm(x).std(axis=0)}")   # ~1

# Edge case: constant feature (zero variance)
x_const = np.ones((32, 4)) * 5.0
print(f"\nConstant input → output: {safe_batch_norm(x_const)[0]}")  # ~0, no NaN

# Edge case: single sample
x_single = np.array([[1.0, 2.0, 3.0]])
print(f"Single sample → output: {safe_batch_norm(x_single)}")  # No crash
```

---

## Summary

Here's what you should take away from this chapter:

- **Addition and Subtraction** — foundation of accumulation and error computation.
  - Commutative and associative in exact math; **NOT associative in float32**.
  - Used in: bias terms, residual connections, loss computation, gradient accumulation.
  - Danger: catastrophic cancellation, overflow. Fix: Kahan summation, float64 accumulators.

- **Multiplication and Division** — scaling and normalization.
  - Division by zero produces inf/NaN; **always protect with epsilon**.
  - Used in: weight application, attention scores, normalization layers.
  - Danger: overflow (large × large), underflow (tiny × tiny). Fix: log-space computation.

- **Modulo** — cyclic and remainder operations.
  - Behavior with negatives **varies by language** (Python vs. C).
  - Used in: batch indexing, cyclic learning rates, hash functions, data sharding.

- **Absolute Value** — magnitude without direction.
  - Generalizes to L1 norm for vectors; foundation of MAE loss and sparsity.
  - Triangle inequality is the key property.
  - Danger: not differentiable at zero. Fix: Huber loss, subgradients.

- **Performance**: All O(1) scalar, O(n) array. Division is slowest. Vectorize everything.

- **Numerical stability**: order of operations matters. Floating-point breaks mathematical guarantees you rely on. Defend against it.

---

> **What's Next** — Individual numbers and operations are the atoms. But ML needs ratios, proportions, and scaling — how to compare numbers and normalize data. That's next.

---

*Next: [Chapter 3: Ratios and Scales](./03-ratios-and-scales.md)*
