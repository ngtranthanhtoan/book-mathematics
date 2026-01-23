# Arithmetic Operations

## Intuition

Arithmetic operations are the atoms of computation. Every complex algorithm, every neural network forward pass, every gradient computation ultimately breaks down into these fundamental operations: addition, subtraction, multiplication, division, modulo, and absolute value.

Think of these operations as basic movements:
- **Addition** is accumulation - gathering things together
- **Subtraction** is finding the difference or removing
- **Multiplication** is scaling - making things larger or smaller proportionally
- **Division** is splitting into equal parts or finding ratios
- **Modulo** is finding what remains after division - like clock arithmetic
- **Absolute value** is measuring distance from zero, regardless of direction

### Why This Matters for ML

In machine learning, these operations happen billions of times per second. A single forward pass through a neural network involves:
- Matrix multiplications (millions of multiply-add operations)
- Subtraction for computing errors
- Division for normalization
- Modulo for batch indexing and cyclic learning rates
- Absolute value for certain loss functions (MAE)

Understanding the computational cost and numerical behavior of these operations helps you:
- Write efficient code
- Debug numerical issues
- Understand algorithm complexity
- Implement custom layers and loss functions

## Visual Explanation

### Operation Properties

```mermaid
graph LR
    subgraph Commutative[Commutative: a ⊕ b = b ⊕ a]
        A[Addition]
        M[Multiplication]
    end

    subgraph NonCommutative[Non-Commutative: a ⊕ b ≠ b ⊕ a]
        S[Subtraction]
        D[Division]
        MO[Modulo]
    end

    subgraph Associative[Associative: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)]
        A2[Addition]
        M2[Multiplication]
    end
```

### Neural Network Operations Flow

```
Input x ──→ [Multiply by W] ──→ [Add bias b] ──→ [Activation] ──→ Output
            (Wx)                 (Wx + b)         f(Wx + b)

Loss = |y_true - y_pred|  (uses subtraction and absolute value)
       or
Loss = (y_true - y_pred)²  (uses subtraction and multiplication)
```

## Mathematical Foundation

### Addition and Subtraction

**Addition** combines quantities:
$$a + b = c$$

**Properties of Addition:**
- Commutative: $a + b = b + a$
- Associative: $(a + b) + c = a + (b + c)$
- Identity element: $a + 0 = a$
- Inverse element: $a + (-a) = 0$

**Subtraction** is addition of the additive inverse:
$$a - b = a + (-b)$$

**Vector/Matrix Addition** (element-wise):
$$\mathbf{A} + \mathbf{B} = \begin{bmatrix} a_{11} + b_{11} & a_{12} + b_{12} \\ a_{21} + b_{21} & a_{22} + b_{22} \end{bmatrix}$$

### Multiplication and Division

**Multiplication** scales or combines:
$$a \times b = c \quad \text{or} \quad a \cdot b = c \quad \text{or} \quad ab = c$$

**Properties of Multiplication:**
- Commutative: $ab = ba$
- Associative: $(ab)c = a(bc)$
- Identity element: $a \cdot 1 = a$
- Zero property: $a \cdot 0 = 0$
- Distributive over addition: $a(b + c) = ab + ac$

**Division** is multiplication by the multiplicative inverse:
$$\frac{a}{b} = a \cdot b^{-1} = a \cdot \frac{1}{b}, \quad b \neq 0$$

**Important**: Division by zero is undefined. In computing, it may produce `inf`, `-inf`, or `NaN`.

### Modulo Operation

The **modulo** operation returns the remainder of integer division:
$$a \mod n = r$$

where $a = qn + r$ and $0 \leq r < n$ (for positive $n$).

**Examples:**
- $17 \mod 5 = 2$ (because $17 = 3 \times 5 + 2$)
- $-17 \mod 5 = 3$ (in Python, result has same sign as divisor)

**Properties:**
- $(a + b) \mod n = ((a \mod n) + (b \mod n)) \mod n$
- $(a \times b) \mod n = ((a \mod n) \times (b \mod n)) \mod n$

### Absolute Value

The **absolute value** gives the magnitude (distance from zero):

$$|x| = \begin{cases} x & \text{if } x \geq 0 \\ -x & \text{if } x < 0 \end{cases}$$

**Properties:**
- $|x| \geq 0$ (always non-negative)
- $|x| = 0 \iff x = 0$
- $|xy| = |x||y|$
- $|x + y| \leq |x| + |y|$ (triangle inequality)

**For complex numbers:**
$$|a + bi| = \sqrt{a^2 + b^2}$$

## Code Example

```python
import numpy as np

# ============================================
# Basic Operations with Scalars
# ============================================
a, b = 7, 3

print("=== Scalar Operations ===")
print(f"a = {a}, b = {b}")
print(f"Addition:       a + b = {a + b}")
print(f"Subtraction:    a - b = {a - b}")
print(f"Multiplication: a * b = {a * b}")
print(f"Division:       a / b = {a / b:.4f}")
print(f"Floor division: a // b = {a // b}")
print(f"Modulo:         a % b = {a % b}")
print(f"Absolute value: |−a| = {abs(-a)}")

# ============================================
# NumPy Array Operations (Vectorized)
# ============================================
print("\n=== NumPy Array Operations (Element-wise) ===")

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])

print(f"x = {x}")
print(f"y = {y}")
print(f"x + y = {x + y}")
print(f"x - y = {x - y}")
print(f"x * y = {x * y}")  # Element-wise, not dot product!
print(f"x / y = {x / y}")
print(f"x % y = {x % y}")
print(f"|x - y| = {np.abs(x - y)}")

# ============================================
# Broadcasting Operations
# ============================================
print("\n=== Broadcasting ===")

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
scalar = 10
vector = np.array([1, 0, -1])

print(f"Matrix:\n{matrix}")
print(f"\nMatrix + scalar ({scalar}):\n{matrix + scalar}")
print(f"\nMatrix * vector ({vector}):\n{matrix * vector}")
print(f"\nMatrix - row mean (normalization):\n{matrix - matrix.mean(axis=1, keepdims=True)}")

# ============================================
# Modulo Applications in ML
# ============================================
print("\n=== Modulo Applications ===")

# Batch indexing with wrap-around
dataset_size = 100
batch_size = 32
num_batches = 5

for i in range(num_batches):
    start_idx = (i * batch_size) % dataset_size
    end_idx = ((i + 1) * batch_size) % dataset_size
    print(f"Batch {i}: indices {start_idx} to {end_idx if end_idx > start_idx else dataset_size}")

# Cyclic learning rate
epochs = 10
cycle_length = 3
for epoch in range(epochs):
    cycle_position = epoch % cycle_length
    lr = 0.001 * (1 + cycle_position)  # Simple cyclic LR
    print(f"Epoch {epoch}: cycle_pos={cycle_position}, lr={lr:.4f}")

# ============================================
# Absolute Value in Loss Functions
# ============================================
print("\n=== Absolute Value in Loss Functions ===")

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.8])

# Mean Absolute Error (MAE / L1 Loss)
mae = np.mean(np.abs(y_true - y_pred))
print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
print(f"MAE (L1 Loss): {mae:.4f}")

# Mean Squared Error (MSE / L2 Loss) for comparison
mse = np.mean((y_true - y_pred) ** 2)
print(f"MSE (L2 Loss): {mse:.4f}")

# ============================================
# Division Edge Cases
# ============================================
print("\n=== Division Edge Cases ===")

# Safe division to avoid NaN/inf
def safe_divide(a, b, default=0.0):
    """Safely divide, returning default where b is zero."""
    result = np.where(b != 0, a / b, default)
    return result

numerator = np.array([1.0, 2.0, 3.0, 4.0])
denominator = np.array([2.0, 0.0, 1.0, 0.0])

print(f"Numerator:   {numerator}")
print(f"Denominator: {denominator}")
print(f"Unsafe division: {numerator / denominator}")  # Contains inf
print(f"Safe division:   {safe_divide(numerator, denominator)}")

# ============================================
# Computational Complexity
# ============================================
print("\n=== Operation Timing ===")
import time

n = 10000000
a = np.random.randn(n)
b = np.random.randn(n) + 0.1  # Avoid zeros for division

operations = [
    ("Addition", lambda: a + b),
    ("Subtraction", lambda: a - b),
    ("Multiplication", lambda: a * b),
    ("Division", lambda: a / b),
    ("Modulo", lambda: np.mod(a, np.abs(b))),
    ("Absolute", lambda: np.abs(a)),
]

for name, op in operations:
    start = time.time()
    for _ in range(10):
        result = op()
    elapsed = (time.time() - start) / 10
    print(f"{name:15} {elapsed*1000:.2f} ms")
```

## ML Relevance

### Where These Operations Appear

| Operation | ML Application |
|-----------|----------------|
| Addition | Bias terms, residual connections, ensemble averaging |
| Subtraction | Error computation, gradient calculation, normalization |
| Multiplication | Weight application, attention scores, learning rate scaling |
| Division | Normalization, softmax, mean computation |
| Modulo | Batch indexing, cyclic schedules, hash functions |
| Absolute Value | L1 loss/regularization, gradient clipping |

### Specific Algorithm Examples

```python
import numpy as np

# 1. Linear Layer: y = Wx + b
def linear_forward(x, W, b):
    """Uses multiplication (matmul) and addition."""
    return np.dot(x, W) + b

# 2. Batch Normalization (simplified)
def batch_norm(x, epsilon=1e-5):
    """Uses subtraction, multiplication, division."""
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_normalized = (x - mean) / np.sqrt(var + epsilon)  # Sub, div
    return x_normalized

# 3. L1 Regularization
def l1_regularization(weights, lambda_param=0.01):
    """Uses absolute value and summation."""
    return lambda_param * np.sum(np.abs(weights))

# 4. Huber Loss (combines L1 and L2)
def huber_loss(y_true, y_pred, delta=1.0):
    """Uses subtraction, absolute value, multiplication."""
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = 0.5 * error ** 2
    linear = delta * abs_error - 0.5 * delta ** 2
    return np.where(abs_error <= delta, quadratic, linear)

# 5. Learning Rate Scheduling with Modulo
def cyclic_lr(epoch, base_lr=0.001, max_lr=0.01, step_size=10):
    """Uses modulo for cyclic pattern."""
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, 1 - x)

# Test the functions
print("=== ML Operation Examples ===")
x = np.random.randn(4, 3)  # 4 samples, 3 features
W = np.random.randn(3, 2)  # 3 input, 2 output
b = np.random.randn(2)

print(f"Linear output shape: {linear_forward(x, W, b).shape}")
print(f"BatchNorm output shape: {batch_norm(x).shape}")
print(f"L1 regularization: {l1_regularization(W):.4f}")

y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.5, 2.1, 2.5])
print(f"Huber loss: {np.mean(huber_loss(y_true, y_pred)):.4f}")
```

## When to Use / Ignore

### Operation Selection Guide

| Goal | Preferred Operation | Reason |
|------|---------------------|--------|
| Accumulate values | Addition | Simple, efficient |
| Compute errors | Subtraction | Direct difference |
| Scale values | Multiplication | Proportional adjustment |
| Normalize | Division | Create ratios |
| Cyclic behavior | Modulo | Wrap-around indexing |
| Magnitude only | Absolute value | Direction-agnostic |
| Robust to outliers | Absolute value (L1) | Less sensitive than squared |

### Common Pitfalls

1. **Division by zero**: Always check denominators or add small epsilon
   ```python
   # Bad
   result = a / b
   # Good
   result = a / (b + 1e-8)
   ```

2. **Integer division surprise**: In Python 3, `/` gives float, `//` gives int
   ```python
   5 / 2   # 2.5 (float)
   5 // 2  # 2 (int)
   ```

3. **Modulo with negative numbers**: Behavior differs between languages
   ```python
   -7 % 3  # Python: 2 (same sign as divisor)
   # C/C++: -1 (same sign as dividend)
   ```

4. **Broadcasting mistakes**: Ensure shapes are compatible
   ```python
   # Shape mismatch can give unexpected results
   a = np.array([[1, 2], [3, 4]])  # (2, 2)
   b = np.array([1, 2, 3])          # (3,)
   # a + b  # Error: shapes don't broadcast
   ```

5. **Accumulation order matters**: For floating-point, sum small numbers first
   ```python
   # More accurate
   sorted_values = sorted(values, key=abs)
   total = sum(sorted_values)
   ```

## Exercises

### Exercise 1: Implement Softmax from Scratch
**Problem**: Implement a numerically stable softmax using only basic operations.

**Solution**:
```python
import numpy as np

def softmax(x):
    """Numerically stable softmax."""
    # Subtract max for numerical stability (subtraction)
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    # Exponentiate (this is a special operation, but built from multiplication)
    exp_x = np.exp(x_shifted)
    # Normalize (division)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Test
logits = np.array([[1000, 1001, 1002], [1, 2, 3]])
print(f"Input logits:\n{logits}")
print(f"Softmax output:\n{softmax(logits)}")
print(f"Sum of each row: {softmax(logits).sum(axis=1)}")  # Should be [1, 1]
```

### Exercise 2: Custom Modulo for Negative Numbers
**Problem**: Implement a modulo function that always returns positive results (Euclidean modulo).

**Solution**:
```python
import numpy as np

def euclidean_mod(a, n):
    """Euclidean modulo - always returns positive result."""
    return ((a % n) + n) % n

# Test with various inputs
test_cases = [(7, 3), (-7, 3), (7, -3), (-7, -3)]
for a, n in test_cases:
    python_mod = a % n
    euclidean = euclidean_mod(a, n)
    print(f"{a} mod {n}: Python={python_mod}, Euclidean={euclidean}")
```

### Exercise 3: Implement MAE and Its Gradient
**Problem**: Implement Mean Absolute Error and its gradient for backpropagation.

**Solution**:
```python
import numpy as np

def mae_forward(y_true, y_pred):
    """Forward pass: compute MAE."""
    return np.mean(np.abs(y_true - y_pred))

def mae_backward(y_true, y_pred):
    """Backward pass: gradient of MAE w.r.t. y_pred."""
    n = len(y_true)
    # Gradient of |x| is sign(x)
    # Gradient of mean is 1/n
    return np.sign(y_pred - y_true) / n

# Test
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.5, 1.8, 3.2, 3.5])

loss = mae_forward(y_true, y_pred)
grad = mae_backward(y_true, y_pred)

print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
print(f"MAE: {loss:.4f}")
print(f"Gradient: {grad}")
print(f"Sum of abs(gradient): {np.sum(np.abs(grad)):.4f}")  # Should be 1.0
```

## Summary

- **Addition and Subtraction**: Foundation of accumulation and error computation
  - Commutative and associative (for exact arithmetic)
  - Used in bias terms, residual connections, loss computation

- **Multiplication and Division**: Scaling and normalization
  - Division by zero is undefined; always protect against it
  - Used in weight application, attention, normalization layers

- **Modulo**: Cyclic and remainder operations
  - Behavior with negatives varies by language
  - Used for batch indexing, cyclic learning rates, hash functions

- **Absolute Value**: Magnitude without direction
  - Essential for L1 loss and regularization
  - Triangle inequality is fundamental property

- **Performance**: All basic operations are O(1) for scalars, O(n) for arrays
- **Numerical stability**: Order of operations matters for floating-point
- **Vectorization**: NumPy operations are much faster than Python loops

---

*Next: [Chapter 3: Ratios and Scales](./03-ratios-and-scales.md)*
