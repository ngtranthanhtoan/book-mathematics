# Chapter 2: Common Function Types

ReLU, sigmoid, tanh, softmax — if you've trained a neural network, you've used these functions. But WHY these functions? What makes sigmoid squeeze values to (0,1)? Why does ReLU work despite being so simple? Each activation function has mathematical properties that make it useful — and limitations that make it dangerous.

---

## Building On

You know what functions are. Now meet the specific functions that power ML. Each one was chosen for a reason — and understanding those reasons helps you pick the right tool.

We'll go through each function the same way: **what it does**, **where it's used in ML**, **mathematical properties**, **code**, and **when it fails**. Along the way, we'll run a showdown: training a simple network with sigmoid vs ReLU, so you can see how these choices play out in practice.

---

## "You Already Know This" — SWE Bridges

Before we dive in, notice how many of these functions map to things you already use every day:

| Math Concept | Code Equivalent |
|---|---|
| **Piecewise function** | `switch/case` statement — different logic for different input ranges |
| **ReLU** | Literally `max(0, x)` — one line of code |
| **Sigmoid** | The logistic function that maps any score to a probability |
| **Step function** | A boolean threshold: `x > 0 ? 1 : 0` |
| **Softmax** | Normalizing a vector of scores so they sum to 1 (like converting raw counts to percentages) |

You've written these patterns hundreds of times. The math just gives them precise names.

---

## Growth Rate Comparison

Before we look at individual functions, understand how fast they grow. This matters because growth rate determines whether gradients explode or vanish during training.

$$\text{For large } x: \quad \log(x) \ll x \ll x^2 \ll e^x$$

```
  y
  |                                                    . e^x
  |                                                  .
  |                                               .
  |                                           ..
  |                                       ...
  |                                   ...        ___ x^2
  |                              ...          __/
  |                          ...          ___/
  |                     ....          ___/
  |               .....           __/         __--- x
  |         ......           ___/         __--
  |   ......             __/          __--
  |...               __/          __--         ___--- ln(x)
  |              ___/         __--         __--
  |          __/          __--        ___--
  |      __/          _--        __--
  |   __/         __--      ___--
  |_/         __--     ___--
  |       __--    ___--
  |   __--  ___--
  |__--__--
  +-----------------------------------------------------> x
```

Logarithmic grows slowest. Linear is steady. Polynomial accelerates. Exponential explodes. Keep this hierarchy in mind — it's why `e^x` shows up in softmax (it amplifies differences) and `ln(x)` shows up in loss functions (it compresses them).

---

## 1. Linear Functions

### What It Does

$$f(x) = mx + b$$

A straight line. Constant rate of change. The simplest possible function.

```
        m > 0            m = 0             m < 0
   y                y                 y
   |        /       |  ________       |  \
   |       /        |                 |   \
   |      /         |                 |    \
   |     /          |                 |     \
   |    /           |                 |      \
   |   /            |                 |       \
   +---------> x    +---------> x    +---------> x
     slope = m        slope = 0        slope = m
```

### Where It's Used in ML

Everywhere. Every `nn.Linear` layer in PyTorch computes $f(x) = Wx + b$. Neural networks without activation functions are just stacked linear transformations — and a composition of linear functions is still linear. That's the entire reason we need activation functions: to break linearity.

### Mathematical Properties

- **Constant rate of change**: slope $m$ is the same everywhere
- **Domain**: $\mathbb{R}$, **Range**: $\mathbb{R}$
- **One-to-one** (if $m \neq 0$) — invertible
- **Derivative**: $f'(x) = m$ (constant)

### Code

```python
def linear_function(x, m=2, b=1):
    """f(x) = mx + b — the building block of every neural network layer."""
    return m * x + b
```

### When It Fails

A single linear function can only model straight-line relationships. Real data is almost never purely linear. Stack linear layers without activation functions and you still get a linear function — no matter how deep your network.

> **Common Mistake**: Forgetting that `nn.Linear` -> `nn.Linear` without an activation in between collapses to a single linear transformation. Your 50-layer network becomes a 1-layer network.

---

## 2. Polynomial Functions — Quadratic and Beyond

### What It Does

$$f(x) = a_n x^n + a_{n-1} x^{n-1} + \cdots + a_1 x + a_0$$

The simplest nonlinear functions. The quadratic $f(x) = ax^2 + bx + c$ is the one you'll see most often (it defines MSE loss, for instance).

```
      Quadratic (x^2)           Cubic (x^3)
   y                          y
   |  \           /           |             /
   |   \         /            |           /
   |    \       /             |         /
   |     \     /              |       /
   |      \   /               |     /
   |       \_/                +----*----------> x
   |                          |   /
   +-----------> x            | /
                              |/
     U-shaped (parabola)      S-shaped (inflection)
```

### Where It's Used in ML

- **Loss functions**: MSE is $\frac{1}{n}\sum(y - \hat{y})^2$ — a sum of quadratics
- **Polynomial regression**: fitting curves to data
- **Feature engineering**: adding $x^2$, $x^3$ terms to capture nonlinear patterns
- **Weierstrass approximation theorem**: any continuous function can be approximated by a polynomial (theoretical foundation)

### Mathematical Properties

- **Degree $n$** determines the number of turning points (at most $n - 1$)
- **Quadratic**: $f(x) = ax^2 + bx + c$ — one minimum or maximum
- **Cubic**: $f(x) = ax^3 + bx^2 + cx + d$ — one inflection point
- **Derivatives**: well-defined everywhere, degree drops by 1

### Code

```python
def polynomial_function(x, coefficients):
    """
    f(x) = a_0 + a_1*x + a_2*x^2 + ...
    coefficients: [a_0, a_1, ..., a_n] (lowest to highest degree)
    """
    result = np.zeros_like(x, dtype=float)
    for i, coef in enumerate(coefficients):
        result += coef * (x ** i)
    return result

# Quadratic: x^2 - 1
quadratic = polynomial_function(x, [-1, 0, 1])

# Cubic: x^3 - x
cubic = polynomial_function(x, [0, -1, 0, 1])
```

### When It Fails

Higher-degree polynomials oscillate wildly between data points — the classic overfitting problem called **Runge's phenomenon**. A degree-20 polynomial can fit your training data perfectly and produce insane predictions on new data.

> **Common Mistake**: Using high-degree polynomial features without regularization. The model memorizes training data instead of learning patterns.

---

## 3. Exponential Functions

### What It Does

$$f(x) = e^x$$

The function whose rate of change equals its value. It grows faster than any polynomial.

```
   y
   |                      .
   |                    .
   |                  .
   |                .
   |              .
   |            .
   |          .
   |        .
   |      ..
   |   ...
   |...
   +-----------------------------> x
       e^0 = 1    e^1 ≈ 2.72    e^2 ≈ 7.39
```

### Where It's Used in ML

- **Softmax**: $e^{x_i} / \sum e^{x_j}$ — converts logits to probabilities
- **Attention mechanisms**: scaled dot-product attention uses $e^x$ under the hood
- **Probability distributions**: Gaussian, Boltzmann, and many others are built on $e^x$
- **Learning rate schedules**: exponential decay

### Mathematical Properties

- **Domain**: $\mathbb{R}$, **Range**: $(0, \infty)$ — always positive
- **Self-derivative**: $\frac{d}{dx}e^x = e^x$ — the only function that is its own derivative
- **Additive exponents**: $e^{a+b} = e^a \cdot e^b$
- **Inverse**: $\ln(e^x) = x$

### Code

```python
import numpy as np

x = np.linspace(-2, 3, 100)
exp_x = np.exp(x)

# Key values
print(f"e^0 = {np.exp(0):.4f}")    # 1.0000
print(f"e^1 = {np.exp(1):.4f}")    # 2.7183
print(f"e^2 = {np.exp(2):.4f}")    # 7.3891
```

### When It Fails

$e^x$ grows so fast that it overflows for large $x$ (e.g., `np.exp(1000)` returns `inf`). This is why every real softmax implementation subtracts the max value first — pure numerical self-defense.

> **Common Mistake**: Computing `np.exp(logits)` directly without the max-subtraction trick. Your softmax returns `nan` for large logit values and your training silently breaks.

---

## 4. Logarithmic Functions

### What It Does

$$f(x) = \ln(x)$$

The inverse of $e^x$. It grows, but slower and slower — diminishing returns made mathematical.

```
   y
   |
   |                         ___________________
   |                   _____/
   |              ____/
   |          ___/
   |       __/
   |     _/
   |   _/
   |  /
   | /
   |/
   +---------------------------------------------> x
   |  (undefined        ln(1)=0     ln(e)=1
   |   for x <= 0)
```

### Where It's Used in ML

- **Cross-entropy loss**: $-\sum y_i \ln(\hat{y}_i)$ — the most common classification loss
- **Log probabilities**: working in log-space prevents underflow when multiplying tiny probabilities
- **Numerical stability**: $\ln(a \cdot b) = \ln(a) + \ln(b)$ turns products into sums (no underflow)
- **Information theory**: entropy, KL divergence, mutual information all use $\ln$

### Mathematical Properties

- **Domain**: $(0, \infty)$, **Range**: $\mathbb{R}$
- **Inverse of exponential**: $\ln(e^x) = x$ and $e^{\ln(x)} = x$
- **Products become sums**: $\ln(ab) = \ln(a) + \ln(b)$
- **Powers become products**: $\ln(a^n) = n \ln(a)$
- **Derivative**: $\frac{d}{dx}\ln(x) = \frac{1}{x}$

### Code

```python
import numpy as np

x = np.linspace(0.01, 5, 100)
log_x = np.log(x)

# Key values
print(f"ln(1) = {np.log(1):.4f}")      # 0.0000
print(f"ln(e) = {np.log(np.e):.4f}")   # 1.0000
print(f"ln(10) = {np.log(10):.4f}")    # 2.3026

# The product-to-sum rule in action (numerical stability)
a, b = 1e-100, 1e-100
print(f"a * b = {a * b}")                          # 0.0 (underflow!)
print(f"ln(a) + ln(b) = {np.log(a) + np.log(b)}")  # -460.5 (safe)
```

### When It Fails

$\ln(x)$ is undefined for $x \leq 0$. When your model predicts a probability of exactly 0 and you compute $\ln(0)$, you get $-\infty$. That's why frameworks add a tiny epsilon: `log(p + 1e-7)`.

> **Common Mistake**: Computing `np.log(model_output)` without clamping. A single zero prediction produces `-inf` loss and corrupts your entire training batch.

---

## 5. Sigmoid Function

### What It Does

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Takes any real number and squeezes it into $(0, 1)$. Think of it as a "soft switch" — smoothly transitions from off (0) to on (1).

```
   1.0 |                          ___________________
       |                        /
       |                      /
       |                    /
   0.5 |  - - - - - - - - * - - - - - - - - - - - -
       |                /
       |              /
       |            /
   0.0 |___________/
       +---------------------------------------------> x
      -10    -5         0         5         10

       sigma(-10) ≈ 0.00005       (nearly 0)
       sigma(0)   = 0.5           (midpoint)
       sigma(10)  ≈ 0.99995       (nearly 1)
```

### "You Already Know This"

You've written this pattern before:

```python
# This is sigmoid thinking:
def is_spam_probability(score):
    # Map any score to a 0-1 range
    return 1 / (1 + math.exp(-score))

# The ML version is identical
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

It's the logistic function. It takes a raw score (any real number) and returns something you can interpret as a probability.

### Where It's Used in ML

- **Binary classification output**: the final layer of a binary classifier
- **LSTM/GRU gates**: forget gate, input gate, output gate all use sigmoid to produce values in $(0, 1)$
- **Logistic regression**: the entire model is just $\sigma(w^T x + b)$
- **Attention gates**: controlling information flow

### Mathematical Properties

- **Domain**: $\mathbb{R}$, **Range**: $(0, 1)$
- **S-shaped curve**: smooth, monotonically increasing
- **Symmetry**: $\sigma(-x) = 1 - \sigma(x)$
- **Midpoint**: $\sigma(0) = 0.5$
- **Derivative**: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- **Maximum derivative**: $\sigma'(0) = 0.25$ — this is important, we'll come back to it

### Code

```python
import numpy as np

def sigmoid(x):
    """sigma(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """sigma'(x) = sigma(x) * (1 - sigma(x))"""
    s = sigmoid(x)
    return s * (1 - s)

# Key values
print(f"sigma(-10) = {sigmoid(-10):.6f}")  # 0.000045 (≈ 0)
print(f"sigma(0)   = {sigmoid(0):.6f}")    # 0.500000
print(f"sigma(10)  = {sigmoid(10):.6f}")   # 0.999955 (≈ 1)

# Symmetry property
print(f"sigma(-2)     = {sigmoid(-2):.4f}")
print(f"1 - sigma(2)  = {1 - sigmoid(2):.4f}")  # Same!

# Derivative — note how small it is away from 0
print(f"sigma'(0)  = {sigmoid_derivative(0):.4f}")   # 0.2500 (max)
print(f"sigma'(5)  = {sigmoid_derivative(5):.6f}")   # 0.006693 (tiny!)
print(f"sigma'(10) = {sigmoid_derivative(10):.8f}")  # 0.00000005 (vanished)
```

### When It Fails — The Vanishing Gradient Problem

Look at those derivative values. The maximum derivative of sigmoid is only **0.25** (at $x = 0$). For $|x| > 5$, the derivative is practically zero.

During backpropagation, gradients are multiplied through each layer. If every layer uses sigmoid, a 10-layer network multiplies gradients by at most $(0.25)^{10} \approx 0.000001$. The gradient vanishes. Early layers barely learn.

```
  Sigmoid derivative:

  0.25|          *
      |        *   *
      |      *       *
      |    *           *
      |  *               *
  0.0 |**                   **
      +---------------------------------------------> x
     -10    -5    0     5    10

  Max derivative is only 0.25 — and it drops to ~0 quickly.
  Multiply 10 layers: (0.25)^10 ≈ 0.000001. Gradients vanish.
```

> **Common Mistake**: Using sigmoid as hidden layer activation in deep networks. It was the default in the 1990s, but we switched to ReLU for exactly this reason. Sigmoid is still fine for **output layers** (binary classification) and **gates** (LSTMs), where you need the $(0, 1)$ range.

---

## 6. Tanh (Hyperbolic Tangent)

### What It Does

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Like sigmoid, but squeezed to $(-1, 1)$ instead of $(0, 1)$. It's actually just a rescaled sigmoid:

$$\tanh(x) = 2\sigma(2x) - 1$$

```
   1.0 |                          ___________________
       |                        /
       |                      /
       |                    /
   0.0 |  - - - - - - - - * - - - - - - - - - - - -
       |                /
       |              /
       |            /
  -1.0 |___________/
       +---------------------------------------------> x
      -10    -5         0         5         10

       tanh(-10) ≈ -1.0          (saturated negative)
       tanh(0)   =  0.0          (zero-centered!)
       tanh(10)  ≈  1.0          (saturated positive)
```

### Where It's Used in ML

- **RNN/LSTM hidden states**: outputs in $(-1, 1)$ are zero-centered, which helps optimization
- **Hidden layers** (historically): before ReLU took over, tanh was preferred over sigmoid for hidden layers because it's zero-centered
- **Normalization**: any time you need outputs centered around zero

### Mathematical Properties

- **Domain**: $\mathbb{R}$, **Range**: $(-1, 1)$
- **Zero-centered**: $\tanh(0) = 0$ — this is a real advantage over sigmoid
- **Odd function**: $\tanh(-x) = -\tanh(x)$
- **Derivative**: $\tanh'(x) = 1 - \tanh^2(x)$
- **Maximum derivative**: $\tanh'(0) = 1$ — four times larger than sigmoid's max!
- **Relationship to sigmoid**: $\tanh(x) = 2\sigma(2x) - 1$

### Code

```python
import numpy as np

def tanh(x):
    """tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))"""
    return np.tanh(x)  # NumPy has it built in

def tanh_derivative(x):
    """tanh'(x) = 1 - tanh^2(x)"""
    t = np.tanh(x)
    return 1 - t ** 2

# Key values
print(f"tanh(-5) = {np.tanh(-5):.6f}")   # -0.999909
print(f"tanh(0)  = {np.tanh(0):.6f}")    #  0.000000
print(f"tanh(5)  = {np.tanh(5):.6f}")    #  0.999909

# Derivative comparison with sigmoid
print(f"tanh'(0)    = {tanh_derivative(0):.4f}")        # 1.0 (max)
print(f"sigmoid'(0) = {sigmoid_derivative(0):.4f}")     # 0.25 (4x smaller)
```

### When It Fails

Same vanishing gradient problem as sigmoid, just slower. $\tanh'(x)$ maxes out at 1.0 (better than sigmoid's 0.25), but it still saturates for large $|x|$. In deep networks, gradients still shrink — just not as fast.

> **Common Mistake**: Assuming tanh "solves" the vanishing gradient problem. It reduces it (max derivative is 1.0 vs 0.25), but doesn't eliminate it. For deep feedforward networks, ReLU is still the better default.

---

## 7. Step Function (Heaviside)

### What It Does

$$H(x) = \begin{cases} 0 & \text{if } x < 0 \\ 1 & \text{if } x \geq 0 \end{cases}$$

A hard binary decision. On or off. True or false.

```
   1.0 |              ___________________
       |             |
       |             |
       |             |
   0.5 |             |
       |             |
       |             |
       |             |
   0.0 |_____________|
       +---------------------------------------------> x
                     0

       Not differentiable at x = 0. Derivative is 0 everywhere else.
       You can't compute gradients through this.
```

### "You Already Know This"

This is a ternary operator:

```python
# Step function in code:
output = 1 if x >= 0 else 0
# Or equivalently:
output = int(x >= 0)
```

### Where It's Used in ML

- **Perceptron** (historical): the original neural network activation from the 1950s
- **Thresholding**: converting probabilities to hard predictions ($p > 0.5 \rightarrow 1$)
- **Quantization**: discretizing continuous values
- **Straight-through estimator**: a trick where you use step in forward pass but pretend gradient is 1 in backward pass

### Mathematical Properties

- **Domain**: $\mathbb{R}$, **Range**: $\{0, 1\}$
- **Not differentiable** at $x = 0$
- **Derivative**: 0 everywhere except $x = 0$ (where it's undefined)
- **Not continuous** at $x = 0$

### Code

```python
import numpy as np

def step_function(x):
    """Heaviside step: H(x) = 0 if x < 0, else 1"""
    return np.where(x < 0, 0, 1)

print(f"H(-2) = {step_function(-2)}")  # 0
print(f"H(0)  = {step_function(0)}")   # 1
print(f"H(2)  = {step_function(2)}")   # 1
```

### When It Fails

You can't train with it. Gradient-based optimization needs derivatives, and the step function's derivative is zero everywhere (or undefined). This is why Rosenblatt's perceptron couldn't learn complex patterns — and why neural networks were largely abandoned until smooth activations (sigmoid, then ReLU) revived them.

> **Common Mistake**: Trying to use step functions in a differentiable pipeline. If you need hard decisions during training, look into the straight-through estimator or Gumbel-softmax.

---

## 8. ReLU (Rectified Linear Unit)

### What It Does

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} 0 & \text{if } x < 0 \\ x & \text{if } x \geq 0 \end{cases}$$

The function that changed deep learning. It's embarrassingly simple: pass positive values through, zero out negative values.

```
   y
   |                        /
   |                      /
   |                    /
   |                  /
   |                /
   |              /
   |            /
   |          /
   |        /
   |______*
   +---------------------------------------------> x
         0

   Left of 0: flat zero. Right of 0: identity line.
   That's it. That's the whole function.
```

### "You Already Know This"

ReLU is literally one line of code:

```python
# ReLU in any language:
output = max(0, x)

# In Python/NumPy:
def relu(x):
    return np.maximum(0, x)

# It's a piecewise function — a switch/case with two cases:
# case x < 0:  return 0
# case x >= 0: return x
```

### Where It's Used in ML

- **Default hidden layer activation**: used in almost every modern deep network
- **CNNs**: every convolutional layer typically uses ReLU
- **Transformers**: feed-forward sublayers use ReLU (or GELU)
- **Residual networks**: ReLU after each residual block

### Mathematical Properties

- **Domain**: $\mathbb{R}$, **Range**: $[0, \infty)$
- **Piecewise linear**: two linear pieces joined at $x = 0$
- **Derivative**: 0 for $x < 0$, 1 for $x > 0$ (undefined at 0, typically set to 0 or 1)
- **Sparse activations**: negative inputs produce exactly 0 — the network naturally learns sparse representations
- **No vanishing gradient for positive inputs**: derivative is exactly 1, gradients flow unchanged

### Code

```python
import numpy as np

def relu(x):
    """ReLU(x) = max(0, x) — the workhorse of deep learning."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative: 0 for x < 0, 1 for x > 0"""
    return np.where(x > 0, 1, 0)

# Key values
print(f"ReLU(-5) = {relu(-5)}")   # 0
print(f"ReLU(0)  = {relu(0)}")    # 0
print(f"ReLU(5)  = {relu(5)}")    # 5

# Derivative comparison
print(f"ReLU'(5)  = {relu_derivative(5)}")   # 1 (always, for positive)
print(f"ReLU'(-5) = {relu_derivative(-5)}")  # 0 (always, for negative)
```

### When It Fails — The Dying ReLU Problem

If a neuron's input becomes negative during training and stays negative, its gradient is always 0. It stops learning. It's "dead." In a large network, a significant fraction of neurons can die — especially with high learning rates.

```
  Dying ReLU:

  If weights push input to always < 0:
  ┌─────────────────────────┐
  │ ReLU(negative) = 0      │
  │ Gradient = 0             │
  │ Weight update = 0        │
  │ Still negative next step │
  │ → Neuron is permanently  │
  │   dead                   │
  └─────────────────────────┘
```

This spawned a family of variants:

```python
def leaky_relu(x, alpha=0.01):
    """Small slope for negative inputs — neurons can't fully die."""
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    """Smooth curve for negative inputs."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

```
       ReLU              Leaky ReLU           ELU
   y                  y                    y
   |          /       |          /         |           /
   |        /         |        /           |         /
   |      /           |      /             |       /
   |    /             |    /               |     /
   |  /               |  /                 |   /
   |_*                | *                  |  *
   |                  |/ (small slope)     |_/ (smooth curve)
   +--------> x      +--------> x         +--------> x
   Dead for x<0      Alive for x<0        Alive + smooth
```

> **Common Mistake**: Using a very high learning rate with ReLU. Large gradient updates can push weights so far negative that neurons die in the first few batches. Use learning rate warmup, or switch to Leaky ReLU / ELU.

---

## 9. Softmax Function

### What It Does

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

Takes a vector of raw scores (logits) and converts them into a probability distribution — all values positive, all values sum to 1. It's a "soft" version of argmax: instead of picking the winner, it assigns probabilities.

```
  Input logits:  [2.0,  1.0,  0.1, -1.0]
                   |      |     |     |
                  e^2    e^1   e^0.1 e^-1
                 [7.39, 2.72, 1.11, 0.37]
                   |      |     |     |
               divide by sum (11.59)
                   |      |     |     |
  Output probs: [0.64, 0.23, 0.10, 0.03]  ← sums to 1.0
```

### "You Already Know This"

```python
# You've done this normalization before:
counts = [64, 23, 10, 3]
total = sum(counts)
percentages = [c / total for c in counts]  # [0.64, 0.23, 0.10, 0.03]

# Softmax is the same idea, but starting from raw scores:
# 1. Exponentiate (makes everything positive, amplifies differences)
# 2. Normalize (divide by sum so they add to 1)
```

### Where It's Used in ML

- **Multi-class classification**: the last layer of any classifier with > 2 classes
- **Transformer attention**: attention weights are softmax of scaled dot products
- **Reinforcement learning**: converting Q-values to action probabilities
- **Language models**: converting logit vectors to token probabilities

### Mathematical Properties

- **Domain**: $\mathbb{R}^n$, **Range**: probability simplex (all positive, sum to 1)
- **Outputs sum to 1**: $\sum_i \text{softmax}(x_i) = 1$
- **Translation invariant**: $\text{softmax}(x) = \text{softmax}(x + c)$ for any constant $c$
- **Differentiable** everywhere
- **Amplifies differences**: the exponential makes large logits much larger relative to small ones

### Code

```python
import numpy as np

def softmax(x):
    """
    Numerically stable softmax.
    Subtract max to prevent overflow — safe because softmax(x) = softmax(x - c).
    """
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Basic usage
logits = np.array([2.0, 1.0, 0.1, -1.0])
probs = softmax(logits)
print(f"Logits:    {logits}")
print(f"Probs:     {probs}")
print(f"Sum:       {probs.sum():.6f}")        # 1.000000
print(f"Predicted: class {np.argmax(probs)}")  # class 0

# Temperature scaling — controls "sharpness"
def temperature_softmax(logits, temperature):
    return softmax(logits / temperature)

print("\nTemperature Scaling:")
for T in [0.5, 1.0, 2.0, 5.0]:
    p = temperature_softmax(logits, T)
    print(f"  T={T}: {p} (max={p.max():.3f})")

# T -> 0: approaches one-hot (argmax)
# T -> inf: approaches uniform distribution
```

### When It Fails

1. **Numerical overflow**: `np.exp(1000)` is `inf`. Always use the max-subtraction trick.
2. **Expensive for large vocabularies**: language models with 50K+ tokens compute softmax over 50K entries every step. This led to approximations like sampled softmax and hierarchical softmax.
3. **Overconfident**: softmax can assign near-1.0 probability to one class even when the model is uncertain. Temperature scaling and calibration techniques address this.

> **Common Mistake**: Computing `np.exp(logits) / np.sum(np.exp(logits))` directly. This overflows for large logits and underflows for very negative logits. Always subtract the max first. The math is identical ($\text{softmax}(x) = \text{softmax}(x - c)$), but the numerics are safe.

---

## Exercise: Sigmoid vs Softmax Equivalence

**Problem**: For a binary classification problem, show that sigmoid and 2-class softmax give the same result.

**Solution**:

For softmax with 2 classes and logits $[z_1, z_2]$:

$$P(\text{class 1}) = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{z_2 - z_1}}$$

Let $z = z_1 - z_2$:

$$P(\text{class 1}) = \frac{1}{1 + e^{-z}} = \sigma(z)$$

This is exactly the sigmoid function applied to the difference of logits. That's why we use sigmoid (1 output) for binary classification instead of softmax (2 outputs) — they're mathematically identical, but sigmoid is simpler.

---

## Running Example: Activation Function Showdown

Let's make this concrete. Here's a simple experiment that shows why we switched from sigmoid to ReLU. Train the same network architecture with different activation functions and watch how gradients behave.

```python
import numpy as np

# =========================================================
# ACTIVATION FUNCTION SHOWDOWN
# Sigmoid vs ReLU: Watch the gradients
# =========================================================

# --- Activation functions and their derivatives ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def tanh_fn(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

# --- Simulate gradient flow through a deep network ---
def simulate_gradient_flow(activation_deriv, n_layers=10, input_val=0.5):
    """
    Simulate how gradients shrink as they pass backward through layers.

    In backprop, the gradient at layer k is the product of all
    derivatives from layer n down to layer k.
    """
    np.random.seed(42)

    # Simulate forward pass: random pre-activation values
    pre_activations = np.random.randn(n_layers) * 0.5 + input_val

    # Compute derivative at each layer
    layer_derivatives = [activation_deriv(z) for z in pre_activations]

    # Gradient at output layer starts at 1.0
    # Each layer multiplies gradient by its local derivative
    cumulative_gradient = 1.0
    gradients = []
    for i in range(n_layers - 1, -1, -1):
        cumulative_gradient *= layer_derivatives[i]
        gradients.append(abs(cumulative_gradient))

    gradients.reverse()  # Now index 0 = first layer
    return gradients

# --- Run the showdown ---
print("=" * 65)
print("ACTIVATION FUNCTION SHOWDOWN: Gradient Flow")
print("How much gradient survives through 10 layers?")
print("=" * 65)

for name, deriv_fn in [("Sigmoid", sigmoid_deriv),
                         ("Tanh", tanh_deriv),
                         ("ReLU", relu_deriv)]:
    grads = simulate_gradient_flow(deriv_fn, n_layers=10)

    print(f"\n{name}:")
    print(f"  Layer 10 (output) gradient: 1.000000")
    print(f"  Layer  5 (middle) gradient: {grads[4]:.8f}")
    print(f"  Layer  1 (first)  gradient: {grads[0]:.8f}")

    if grads[0] < 1e-6:
        print(f"  --> VANISHED! First layer barely learns.")
    elif grads[0] > 0.01:
        print(f"  --> Healthy gradient. First layer can learn.")

print("\n" + "=" * 65)
print("VERDICT:")
print("  Sigmoid: gradients vanish exponentially.")
print("  Tanh:    better, but still shrinks.")
print("  ReLU:    gradients flow through unchanged (for positive inputs).")
print("=" * 65)
```

**What you'll see**: Sigmoid gradients decay to near-zero by layer 1. Tanh gradients decay slower but still shrink. ReLU gradients stay at 1.0 for positive inputs — they pass through unchanged. This is the fundamental reason modern deep learning uses ReLU (and its variants) as the default activation.

---

## Complete Comparison: Choosing Your Activation Function

Here's the cheat sheet. Print this out.

### Activation Functions At a Glance

| Function | Range | Derivative Max | Use Case | Killer Problem |
|----------|-------|----------------|----------|----------------|
| **Sigmoid** | $(0, 1)$ | 0.25 | Binary output, gates | Vanishing gradients |
| **Tanh** | $(-1, 1)$ | 1.0 | RNN hidden states | Still saturates |
| **ReLU** | $[0, \infty)$ | 1.0 | Hidden layers (default) | Dying neurons |
| **Leaky ReLU** | $(-\infty, \infty)$ | 1.0 | Hidden layers (safer) | Hyperparameter $\alpha$ |
| **Softmax** | $(0, 1)^n$, sum=1 | — | Multi-class output | Overflow, expensive |
| **Step** | $\{0, 1\}$ | 0 everywhere | Hard decisions | Not differentiable |

### Where Each Function Type Appears in ML

| Function Type | ML Application |
|---------------|----------------|
| **Linear** | Linear regression, neural network layers (before activation) |
| **Polynomial** | Polynomial regression, feature engineering, MSE loss |
| **Exponential** | Softmax, attention mechanisms, probability distributions |
| **Logarithmic** | Cross-entropy loss, log probabilities, numerical stability |
| **Sigmoid** | Binary classification output, LSTM/GRU gates |
| **Tanh** | RNN hidden states, zero-centered normalization |
| **Softmax** | Multi-class classification, transformer attention weights |
| **ReLU** | Hidden layer activation (default in modern deep learning) |

### Decision Guide

```
What layer are you building?
│
├── Output layer
│   ├── Binary classification? → Sigmoid
│   ├── Multi-class classification? → Softmax
│   ├── Regression (unbounded)? → Linear (no activation)
│   └── Regression (bounded)? → Sigmoid or Tanh
│
├── Hidden layer
│   ├── Default choice? → ReLU
│   ├── Dying ReLU problem? → Leaky ReLU or ELU
│   ├── Need zero-centered? → Tanh
│   └── Transformer FF layer? → ReLU or GELU
│
└── Recurrent gate
    ├── Forget/input/output gate? → Sigmoid
    └── Hidden state? → Tanh
```

---

## Exercises

### Exercise 1: Temperature Scaling

**Problem**: Implement temperature-scaled softmax. What happens as temperature approaches 0? As it approaches infinity?

**Solution**:

```python
def temperature_softmax(logits, temperature):
    return softmax(logits / temperature)

logits = np.array([2.0, 1.0, 0.1, -1.0])

# Low temperature (T -> 0): more confident, approaches argmax (one-hot)
print(temperature_softmax(logits, 0.1))  # Nearly [1, 0, 0, 0]

# High temperature (T -> inf): less confident, approaches uniform
print(temperature_softmax(logits, 100))  # Nearly [0.25, 0.25, 0.25, 0.25]
```

- **Low temperature**: amplifies differences. The highest logit dominates. In the limit, you get argmax.
- **High temperature**: flattens differences. All logits look similar. In the limit, you get uniform distribution.

This is used in knowledge distillation (training a small model to mimic a large one) and controlling randomness in language model generation.

### Exercise 2: Numerical Stability

**Problem**: Why does `softmax([1000, 1001, 1002])` cause problems? How do you fix it?

**Solution**:

Direct computation: $e^{1000}$ overflows to infinity.

Fix: subtract the maximum before exponentiating:

```python
def stable_softmax(x):
    x_shifted = x - np.max(x)  # Now [-2, -1, 0]
    exp_x = np.exp(x_shifted)   # Safe: [e^-2, e^-1, e^0]
    return exp_x / np.sum(exp_x)
```

This works because $\text{softmax}(x) = \text{softmax}(x - c)$ for any constant $c$. The translation invariance property isn't just elegant math — it's essential for making softmax work in practice.

### Exercise 3: Dying ReLU Detector

**Problem**: Write a function that counts "dead" neurons in a trained network.

**Solution**:

```python
def count_dead_neurons(model, data_loader, threshold=0.0):
    """
    A neuron is "dead" if it outputs 0 for every input in the dataset.
    This means its input is always negative → ReLU always returns 0
    → gradient is always 0 → it never learns again.
    """
    activation_counts = {}  # Track how often each neuron fires

    # Hook to capture activations after each ReLU
    def hook_fn(name):
        def hook(module, input, output):
            # Count non-zero activations
            if name not in activation_counts:
                activation_counts[name] = np.zeros(output.shape[1])
            activation_counts[name] += (output > threshold).sum(axis=0).detach().numpy()
        return hook

    # Register hooks on all ReLU layers
    # ... (framework-specific)

    # Run data through the network
    # ... (framework-specific)

    # Neurons that never fired are dead
    for name, counts in activation_counts.items():
        dead = (counts == 0).sum()
        total = len(counts)
        print(f"{name}: {dead}/{total} neurons dead ({100*dead/total:.1f}%)")
```

If more than 10-20% of neurons are dead, consider: lowering the learning rate, using Leaky ReLU, or improving weight initialization.

---

## Summary

- **Linear functions** ($mx + b$): the foundation of every neural network layer, but limited without activation
- **Polynomial functions**: capture nonlinear patterns; quadratics define MSE loss; high degrees overfit
- **Exponential functions** ($e^x$): rapid growth; central to softmax and probability distributions
- **Logarithmic functions** ($\ln x$): the inverse of $e^x$; crucial for loss functions and numerical stability
- **Sigmoid** ($\sigma(x)$): squashes to $(0, 1)$; perfect for binary outputs and gates; vanishing gradients in hidden layers
- **Tanh**: squashes to $(-1, 1)$; zero-centered (better than sigmoid); still saturates
- **ReLU** ($\max(0, x)$): stupidly simple, remarkably effective; default for hidden layers; watch for dying neurons
- **Softmax**: converts logits to probabilities; always use the numerically stable version
- **Step function**: hard binary decision; not differentiable; mostly historical

The pattern: each function was chosen for a specific mathematical property (range, derivative behavior, computational cost). Understanding **why** each one works (and fails) is what separates "I copied this from a tutorial" from "I chose this because it's right for my problem."

---

## What's Next

So far: functions with one input. But ML models take many inputs: $f(x_1, x_2, \ldots, x_n)$. Multivariable functions are where single-variable math meets the real world. A neural network layer doesn't compute $f(x)$ — it computes $f(\mathbf{x}) = W\mathbf{x} + \mathbf{b}$, where $\mathbf{x}$ is a vector of features. Understanding how functions behave with multiple inputs is essential for understanding gradients, optimization, and everything that makes deep learning work.

**Next**: [Chapter 3 - Multivariable Functions](03-multivariable-functions.md)
