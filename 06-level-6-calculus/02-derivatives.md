# Chapter 2: Derivatives — The Original `diff` Command

## Building On

Limits told us what value we're approaching. Now the key question for ML: how *fast* are we approaching it? The derivative measures the **instantaneous rate of change** — and it's the engine behind gradient descent.

---

## The Problem That Starts Everything

Your model's loss is 2.31. You nudge weight $w_1$ by 0.001 and the loss drops to 2.28. You nudge $w_2$ by 0.001 and the loss goes to 2.35. Which direction should you move? You just computed partial derivatives — the foundation of backpropagation.

That "nudge and measure" instinct? That's all a derivative is. You changed something, you observed the effect, and you computed the rate of change. Every gradient descent step in every ML framework on Earth does exactly this, billions of times per second.

Let's build the concept from scratch — code first, formulas second.

---

## Step 1: Compute Derivatives Numerically (Code First)

Before we touch any rules or formulas, let's just *do* it. You have a function, you want to know how fast it's changing at a point. Poke it and see.

```python
def numerical_derivative(f, x, h=1e-7):
    """
    The simplest possible derivative: nudge x, measure the change.
    This is literally what you did with the loss function above.
    """
    return (f(x + h) - f(x)) / h

# Let's try it on f(x) = x^3
f = lambda x: x**3

# How fast is x^3 changing at x = 2?
print(numerical_derivative(f, 2))  # ≈ 12.0
```

That's it. That's a derivative. You nudged `x` from 2 to 2.0000001, saw how much `f(x)` changed, and divided by the nudge size. The answer is approximately 12 — meaning at $x = 2$, the function $x^3$ is increasing at a rate of 12 units per unit of $x$.

But wait — we can do better. The "forward difference" above is slightly biased. If you nudge in *both* directions and average, you get a more accurate answer:

```python
def forward_difference(f, x, h=1e-7):
    """Nudge forward only: f'(x) ≈ (f(x+h) - f(x)) / h"""
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h=1e-7):
    """Nudge backward only: f'(x) ≈ (f(x) - f(x-h)) / h"""
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h=1e-7):
    """Nudge both ways: f'(x) ≈ (f(x+h) - f(x-h)) / (2h) — much more accurate"""
    return (f(x + h) - f(x - h)) / (2 * h)

f = lambda x: x**3
f_prime_exact = lambda x: 3 * x**2  # We'll derive this rule shortly

x = 2.0
print("Derivative of f(x) = x^3 at x = 2")
print(f"  Exact (spoiler):    {f_prime_exact(x):.10f}")
print(f"  Forward difference: {forward_difference(f, x):.10f}")
print(f"  Backward difference:{backward_difference(f, x):.10f}")
print(f"  Central difference: {central_difference(f, x):.10f}")
# Central difference wins — error is O(h^2) instead of O(h)
```

> **You Already Know This: Derivative as `git diff`**
>
> A derivative is a rate-of-change measurement between two infinitely close states.
> Think `git diff` — you're comparing the "before" snapshot `f(x)` with the
> "after" snapshot `f(x + h)`, then normalizing by how much you changed (`h`).
> As `h` shrinks toward zero, your diff gets infinitely precise — you're measuring
> the *exact* rate of change at a single point, not between two commits.

---

## Step 2: The Formal Definition

What we just coded has a mathematical name. The **derivative** of $f$ at point $x$ is:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Translation:** Take the numerical derivative we wrote above, and let the step size `h` shrink to zero. The limit is the exact answer — no approximation error, no floating point headaches.

### Notation Zoo

You'll see derivatives written several ways. They all mean the same thing:

- **Lagrange:** $f'(x)$ — compact, common in pure math
- **Leibniz:** $\frac{df}{dx}$ or $\frac{d}{dx}f(x)$ — reminds you what you're differentiating with respect to
- **Newton:** $\dot{f}$ — used for time derivatives in physics

In ML papers, you'll mostly see Leibniz notation ($\frac{\partial L}{\partial w}$) because it makes the chain rule obvious.

### What the Derivative *Means* — Three Views

The derivative $f'(a)$ simultaneously represents:

1. **Slope** of the tangent line to $f$ at point $a$
2. **Rate of change** of $f$ at point $a$
3. **Best linear approximation:** $f(a + h) \approx f(a) + f'(a) \cdot h$ for small $h$

That third one is huge in ML. It says: "If I change the input by a tiny amount $h$, the output changes by approximately $f'(a) \cdot h$." This is exactly how gradient descent works — it uses the derivative to *predict* how the loss will change when you adjust a weight.

---

## Step 3: Visualizing It — From Secant to Tangent

Here's what's happening geometrically when $h \to 0$:

```
    Secant lines converging to the tangent as h → 0
    ─────────────────────────────────────────────────

    f(x) ↑
         │                          ╱
         │                  ·    ╱     Secant (h = 1.0)
         │                ·   ╱         slope = [f(x+1) - f(x)] / 1
         │              ·  ╱╱
         │            ·  ╱╱     Secant (h = 0.5)
         │          ·  ╱╱         slope = [f(x+0.5) - f(x)] / 0.5
         │        ····╱╱
         │      ··  ╱╱       Secant (h = 0.1)
         │    ··  ●╱           slope = [f(x+0.1) - f(x)] / 0.1
         │   ·· ╱╱ P
         │  · ╱╱           ═══════════════════════════
         │ ·╱╱              As h → 0, all secant lines
         │╱╱                converge to the TANGENT at P
         ╱╱                 slope = f'(x)  ← the derivative
         ┼───────────────────────────────────→ x
```

The derivative transforms a **secant line** (connecting two points on the curve) into a **tangent line** (touching exactly one point) as you let the distance between points shrink to zero.

```
    Tangent line at a point
    ───────────────────────

    f(x) ↑
         │                              ╱╱
         │                           ╱╱
         │                 ·····   ╱╱  ← tangent line
         │            ····      ●╱       (slope = f'(a))
         │         ···       ╱╱· P = (a, f(a))
         │       ··       ╱╱  ·
         │      ·      ╱╱    ·
         │     ·    ╱╱       ·
         │    ·  ╱╱         ·
         │   ·╱╱           ·
         │  ╱╱  ·         ·
         │╱╱     ·       ·
         ╱────────·──────·─────────────→ x
                  a
```

---

## Step 4: Symbolic Rules — The Shortcuts

Numerical differentiation works, but it's slow (two function evaluations per derivative) and approximate. Centuries of mathematicians worked out closed-form rules so you don't have to nudge-and-measure every time. Think of these as **optimized implementations** of the general algorithm.

> **You Already Know This: Numerical vs. Symbolic Differentiation — Testing vs. Proof**
>
> Numerical differentiation is like **unit testing** — it checks the derivative at
> specific values of $x$ and gives you approximate answers. Symbolic rules are like
> **formal proofs** — they give you the exact derivative for *all* values of $x$ at
> once. You use tests to gain confidence; you use proofs to know for certain.

### Basic Derivative Rules

| Function | Derivative | Example | Why You Care |
|----------|------------|---------|--------------|
| Constant: $c$ | $0$ | $(5)' = 0$ | Bias terms don't change when you change weights |
| Power: $x^n$ | $nx^{n-1}$ | $(x^3)' = 3x^2$ | Polynomial regression, weight decay ($w^2$) |
| Exponential: $e^x$ | $e^x$ | $(e^x)' = e^x$ | Softmax, log-likelihoods everywhere |
| Natural log: $\ln x$ | $\frac{1}{x}$ | $(\ln x)' = \frac{1}{x}$ | Cross-entropy loss |
| Sine: $\sin x$ | $\cos x$ | $(\sin x)' = \cos x$ | Positional encodings (transformers) |
| Cosine: $\cos x$ | $-\sin x$ | $(\cos x)' = -\sin x$ | Positional encodings (transformers) |

**Translation:** The power rule says "bring the exponent down as a coefficient, then subtract one from the exponent." So $x^3$ becomes $3x^2$, and $x^{100}$ becomes $100x^{99}$. The exponential $e^x$ is special — it's its own derivative. That's not a coincidence; it's the *definition* of $e$.

Let's verify a couple of these numerically — trust but verify:

```python
import numpy as np

def central_diff(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

# Verify: d/dx[x^3] = 3x^2
print("Power rule: d/dx[x^3] at x=3")
print(f"  Symbolic: {3 * 3**2}")              # 27
print(f"  Numerical: {central_diff(lambda x: x**3, 3):.6f}")  # ≈ 27

# Verify: d/dx[e^x] = e^x
print("\nExponential: d/dx[e^x] at x=1")
print(f"  Symbolic: {np.exp(1):.6f}")          # e ≈ 2.718282
print(f"  Numerical: {central_diff(np.exp, 1):.6f}")  # ≈ 2.718282

# Verify: d/dx[ln(x)] = 1/x
print("\nLogarithm: d/dx[ln(x)] at x=2")
print(f"  Symbolic: {1/2:.6f}")                # 0.5
print(f"  Numerical: {central_diff(np.log, 2):.6f}")  # ≈ 0.5
```

### Combination Rules

Functions rarely come alone. You add them, multiply them, divide them. Here's how derivatives play with arithmetic:

**Sum Rule:** $(f + g)' = f' + g'$

**Translation:** The derivative of a sum is the sum of the derivatives. Derivatives are *linear* — they distribute over addition just like your favorite linear operators.

**Product Rule:** $(f \cdot g)' = f' \cdot g + f \cdot g'$

**Translation:** When two things are multiplied together and both are changing, the total change comes from two sources: the first thing changing while the second stays still, plus the first staying still while the second changes. Think of area = width $\times$ height. If both are growing, the rate of area change accounts for width growing (at fixed height) plus height growing (at fixed width).

**Quotient Rule:** $\left(\frac{f}{g}\right)' = \frac{f' \cdot g - f \cdot g'}{g^2}$

**Translation:** Honestly, you'll rarely need this directly. If you have $f/g$, you can rewrite it as $f \cdot g^{-1}$ and use the product rule + chain rule. But it's here when you want it.

---

## Step 5: The Chain Rule — This Is Backpropagation

Here's the punchline of the entire chapter. If you only remember one thing about derivatives, make it this:

> **Backpropagation IS the chain rule.** Every framework — PyTorch, TensorFlow, JAX — implements automatic differentiation by applying the chain rule through a computational graph. That's it. That's the secret.

### The Rule

If you have a composite function $y = f(g(x))$ — a function inside a function — the derivative is:

$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

In Leibniz notation, if $y = f(u)$ and $u = g(x)$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Translation:** To differentiate a composition, work from the outside in. Differentiate the outer function (evaluated at the inner function), then multiply by the derivative of the inner function. The derivatives *multiply* along the chain — they don't add.

> **You Already Know This: Chain Rule as Function Composition**
>
> In code, `f(g(x))` is just piping the output of `g` into `f`. The chain rule
> says: to find how the final output changes with respect to the input, multiply
> the "local derivatives" at each stage.
>
> Think of it as a pipeline where each stage has a "gain" (its derivative). The
> total gain through the pipeline is the product of all the individual gains.
> ```
> d/dx[f(g(x))] mirrors how backprop chains through layers:
>
>   compose:   x ──→ g(x) ──→ f(g(x)) ──→ output
>   backprop:  ∂L/∂x ←── ∂g/∂x · ←── ∂f/∂g · ←── ∂L/∂f
>
> Derivatives flow BACKWARD, multiplying at each stage.
> ```

### Chain Rule — The Pipeline Diagram

```
  Forward pass (compute output):
  ┌─────┐      ┌─────┐      ┌─────┐
  │  x  │─────→│ g() │─────→│ f() │─────→  output y
  └─────┘      └─────┘      └─────┘
                u=g(x)       y=f(u)

  Backward pass (compute derivatives):
  ┌─────┐      ┌─────┐      ┌─────┐
  │∂y/∂x│←─────│∂u/∂x│←─────│∂y/∂u│←─────  ∂L/∂y
  └─────┘      └─────┘      └─────┘
   = g'(x)       ×            = f'(u)

  Result: dy/dx  =  f'(g(x))  ·  g'(x)
          ▲            ▲            ▲
          │            │            │
       total      outer deriv   inner deriv
       deriv     (evaluated at
                  inner value)
```

### Chain Rule Example — By Hand

Differentiate $y = (3x + 1)^5$.

Set $u = 3x + 1$, so $y = u^5$.

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 5u^4 \cdot 3 = 15(3x+1)^4$$

Let's verify:

```python
import numpy as np

f = lambda x: (3*x + 1)**5
f_prime = lambda x: 15 * (3*x + 1)**4

x = 2.0
numerical = (f(x + 1e-7) - f(x - 1e-7)) / (2e-7)
print(f"Symbolic:  {f_prime(x):.4f}")   # 15 * 7^4 = 36015
print(f"Numerical: {numerical:.4f}")     # ≈ 36015
```

### Chain Rule in Action — Backpropagation

Here's a complete mini neural network where you can *see* the chain rule working:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# ─────────────────────────────────────────────
# Network: input → [weight] → [sigmoid] → output
# Loss:    L = (output - target)²
# ─────────────────────────────────────────────

x_input = 1.0
w = 0.5
target = 0.8

# === FORWARD PASS ===
z = w * x_input          # linear combination
a = sigmoid(z)           # activation
loss = (a - target)**2   # squared error loss

print(f"Input: {x_input}, Weight: {w}, Target: {target}")
print(f"Forward: z={z:.4f}, a=sigmoid(z)={a:.4f}, loss={loss:.4f}")

# === BACKWARD PASS (chain rule!) ===
# We need ∂Loss/∂w. The computation graph is:
#   w → z=w*x → a=σ(z) → L=(a-target)²
#
# By chain rule:
#   ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w

dL_da = 2 * (a - target)           # ∂Loss/∂a
da_dz = sigmoid_derivative(z)       # ∂a/∂z = σ'(z)
dz_dw = x_input                     # ∂z/∂w = x

dL_dw = dL_da * da_dz * dz_dw       # CHAIN RULE: multiply!

print(f"\nBackward pass (Chain Rule):")
print(f"  ∂L/∂a  = 2(a - target)          = {dL_da:.6f}")
print(f"  ∂a/∂z  = σ'(z) = σ(z)(1-σ(z))  = {da_dz:.6f}")
print(f"  ∂z/∂w  = x                      = {dz_dw:.6f}")
print(f"  ∂L/∂w  = ∂L/∂a · ∂a/∂z · ∂z/∂w = {dL_dw:.6f}")

# === VERIFY NUMERICALLY ===
h = 1e-5
loss_plus  = (sigmoid((w + h) * x_input) - target)**2
loss_minus = (sigmoid((w - h) * x_input) - target)**2
numerical_grad = (loss_plus - loss_minus) / (2 * h)

print(f"\nVerification:")
print(f"  Analytical gradient:  {dL_dw:.8f}")
print(f"  Numerical gradient:   {numerical_grad:.8f}")
print(f"  Difference:           {abs(numerical_grad - dL_dw):.2e}")

# === GRADIENT DESCENT STEP ===
learning_rate = 1.0
w_new = w - learning_rate * dL_dw
new_loss = (sigmoid(w_new * x_input) - target)**2
print(f"\nGradient descent step:")
print(f"  Old weight: {w:.4f}, Old loss: {loss:.4f}")
print(f"  New weight: {w_new:.4f}, New loss: {new_loss:.4f}")
print(f"  Loss decreased: {loss > new_loss}")
```

Notice the pattern in the backward pass: **multiply local derivatives along the chain**. This is exactly what PyTorch's `loss.backward()` does — it walks the computation graph in reverse, applying the chain rule at every node.

---

## Step 6: Partial Derivatives — Multiple Inputs

Real ML models don't have one parameter — they have millions. When a function has multiple inputs, you take **partial derivatives**: the derivative with respect to one variable while holding the others constant.

For $f(x, y)$:

$$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}$$

$$\frac{\partial f}{\partial y} = \lim_{h \to 0} \frac{f(x, y+h) - f(x, y)}{h}$$

**Translation:** A partial derivative asks: "If I wiggle *just this one input* and hold everything else fixed, how does the output change?" That's exactly what happened in our opening example — we nudged $w_1$ alone, then $w_2$ alone, and measured each effect independently.

The curly $\partial$ (as opposed to straight $d$) is a visual cue that says "there are other variables, but I'm ignoring them right now."

### Example

For $f(x, y) = x^2y + 3xy^2$:

$$\frac{\partial f}{\partial x} = 2xy + 3y^2 \qquad \text{(treat } y \text{ as a constant)}$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy \qquad \text{(treat } x \text{ as a constant)}$$

```python
import numpy as np

def f(x, y):
    """f(x,y) = x²y + 3xy²"""
    return x**2 * y + 3 * x * y**2

def partial_x_exact(x, y):
    """∂f/∂x = 2xy + 3y²"""
    return 2 * x * y + 3 * y**2

def partial_y_exact(x, y):
    """∂f/∂y = x² + 6xy"""
    return x**2 + 6 * x * y

def numerical_partial_x(f, x, y, h=1e-7):
    """Nudge x only, hold y constant"""
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

def numerical_partial_y(f, x, y, h=1e-7):
    """Nudge y only, hold x constant"""
    return (f(x, y + h) - f(x, y - h)) / (2 * h)

x, y = 2.0, 3.0
print(f"f(x,y) = x²y + 3xy² at point ({x}, {y})")
print(f"\n∂f/∂x = 2xy + 3y²:")
print(f"  Symbolic:  {partial_x_exact(x, y):.6f}")
print(f"  Numerical: {numerical_partial_x(f, x, y):.6f}")
print(f"\n∂f/∂y = x² + 6xy:")
print(f"  Symbolic:  {partial_y_exact(x, y):.6f}")
print(f"  Numerical: {numerical_partial_y(f, x, y):.6f}")
```

---

## Automatic Differentiation — The Best of Both Worlds

> **You Already Know This: Automatic Differentiation as a Compiler Trick**
>
> Numerical differentiation is approximate and slow ($O(n)$ function evaluations
> for $n$ parameters). Symbolic differentiation is exact but can produce enormous
> expressions ("expression swell"). **Automatic differentiation (autodiff)** is the
> compiler trick that gives you exact derivatives for free — same cost as the
> forward pass.
>
> PyTorch's `autograd`, TensorFlow's `GradientTape`, JAX's `grad` — they all
> record the computation graph during the forward pass, then walk it backward
> applying the chain rule at each node. No approximation. No expression swell.
> Just exact derivatives, computed efficiently.

```python
import torch

# PyTorch autograd: the chain rule, automated
w = torch.tensor(0.5, requires_grad=True)
x = torch.tensor(1.0)
target = torch.tensor(0.8)

# Forward pass — PyTorch records the computation graph
z = w * x
a = torch.sigmoid(z)
loss = (a - target)**2

# Backward pass — chain rule applied automatically
loss.backward()

print(f"PyTorch autograd gradient: {w.grad:.8f}")
# This matches our hand-computed gradient from the backprop example above!
```

---

## ML Relevance — Where Derivatives Run the Show

### Gradient Descent

The fundamental optimization algorithm uses derivatives to update parameters:

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta}$$

**Translation:** New parameter = old parameter minus (learning rate times derivative of loss with respect to that parameter). The derivative tells you the slope; you step downhill. Repeat until you reach a minimum.

### Backpropagation Through a Full Network

For a network with multiple layers:

$$\text{input} \xrightarrow{W_1} h_1 \xrightarrow{W_2} h_2 \xrightarrow{W_3} \text{output} \rightarrow \text{loss}$$

The chain rule computes derivatives all the way back:

$$\frac{\partial \text{loss}}{\partial W_1} = \frac{\partial \text{loss}}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}$$

This is just the chain rule applied repeatedly — nothing more. The "back" in "backpropagation" refers to the direction: you start at the loss and chain backward through each layer.

### Common Derivatives in ML (Your Cheat Sheet)

| Function | Derivative | Where It Shows Up |
|----------|------------|-------------------|
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | Sigmoid activation, logistic regression |
| $\tanh(x)$ | $1 - \tanh^2(x)$ | Tanh activation, LSTMs |
| $\text{ReLU}(x) = \max(0,x)$ | $\begin{cases}1 & x>0\\0 & x<0\end{cases}$ | Most hidden layers in modern networks |
| $\text{softmax}_i$ | $s_i(\delta_{ij} - s_j)$ | Output layer for classification |
| $x^2$ (L2 loss / weight decay) | $2x$ | MSE loss, regularization |
| $-\ln(x)$ (cross-entropy) | $-\frac{1}{x}$ | Cross-entropy loss |

---

## Common Mistakes

> **Mistake 1: "The chain rule adds derivatives"**
>
> No. The chain rule **multiplies** derivatives. If $y = f(g(x))$, then
> $\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$. The multiplication is what causes
> vanishing gradients in deep networks — if each layer's derivative is less
> than 1, the product shrinks exponentially.

> **Mistake 2: "$\frac{d}{dx}$ is a fraction"**
>
> Technically, $\frac{d}{dx}$ is an **operator** — it means "take the derivative
> with respect to $x$." It's not literally dividing $d$ by $dx$. *However*, in
> the chain rule, you *can* treat it like a fraction and "cancel" the $du$ terms:
> $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$. This works in single-variable
> calculus (a useful mnemonic), but be careful with partial derivatives where the
> analogy breaks down.

> **Mistake 3: "The derivative of a constant is 1"**
>
> The derivative of a constant is **0**. Constants don't change — that's the whole
> point. This matters in ML: bias terms have a gradient that depends on the chain
> rule, but the derivative of the bias itself with respect to the bias is 1, while
> the derivative of a constant *value* is 0. Don't confuse the two.

> **Mistake 4: Confusing the derivative at a point with the derivative function**
>
> $f'(2) = 12$ is a number — the slope at $x = 2$. $f'(x) = 3x^2$ is a
> function — it gives you the slope at *any* $x$. Numerical differentiation
> gives you the first; symbolic differentiation gives you the second.

---

## When to Use vs. When to Abstract Away

### When Derivatives Matter (Roll Up Your Sleeves)
- **Implementing custom layers or loss functions** — you need to define `backward()`
- **Debugging training instability** — vanishing/exploding gradients are derivative problems
- **Understanding optimizer behavior** — Adam, RMSprop, etc. all manipulate gradients
- **Reading ML papers** — derivations are written in the language of derivatives

### When to Let the Framework Handle It
- Using standard layers (autograd handles everything)
- Prototyping models quickly
- Most day-to-day ML development

### Gotchas to Watch For
1. **Numerical instability:** Very small or very large derivatives cause NaN/Inf
2. **Vanishing gradients:** Deep networks with sigmoid/tanh — derivatives compound toward zero
3. **Exploding gradients:** Derivatives compound to huge values — use gradient clipping
4. **Non-differentiable points:** ReLU at $x = 0$ — frameworks define it as 0 by convention

---

## Visualization — Function and Its Derivative

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_derivative():
    """Visualize a function, its derivative, and a tangent line."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.linspace(-2, 2, 200)

    # f(x) = x^3 - x
    f = lambda x: x**3 - x
    f_prime = lambda x: 3*x**2 - 1

    # Left plot: function with tangent line at x=1
    ax1 = axes[0]
    ax1.plot(x, f(x), 'b-', linewidth=2, label=r'$f(x) = x^3 - x$')

    x0 = 1
    tangent = f(x0) + f_prime(x0) * (x - x0)
    ax1.plot(x, tangent, 'r--', linewidth=2, label=f'Tangent at x={x0} (slope={f_prime(x0)})')
    ax1.plot(x0, f(x0), 'ko', markersize=8)

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Function and Tangent Line')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)

    # Right plot: function and its derivative
    ax2 = axes[1]
    ax2.plot(x, f(x), 'b-', linewidth=2, label=r'$f(x) = x^3 - x$')
    ax2.plot(x, f_prime(x), 'r-', linewidth=2, label=r"$f'(x) = 3x^2 - 1$")

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 4)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Function vs. Its Derivative')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('derivative_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved as 'derivative_visualization.png'")

plot_derivative()
```

---

## Exercises

### Exercise 1: Chain Rule — Compute a Derivative

**Problem:** Find $\frac{d}{dx}[\ln(x^2 + 1)]$

**Solution:** Using the chain rule with $u = x^2 + 1$:

$$\frac{d}{dx}[\ln(x^2 + 1)] = \frac{1}{x^2 + 1} \cdot 2x = \frac{2x}{x^2 + 1}$$

The outer function is $\ln(u)$ with derivative $\frac{1}{u}$. The inner function is $x^2 + 1$ with derivative $2x$. Multiply them together.

### Exercise 2: Partial Derivatives

**Problem:** For $f(x, y, z) = xy^2z^3$, find all partial derivatives.

**Solution:**
- $\frac{\partial f}{\partial x} = y^2z^3$ (treat $y$ and $z$ as constants, derivative of $x$ is 1)
- $\frac{\partial f}{\partial y} = 2xyz^3$ (treat $x$ and $z$ as constants, power rule on $y^2$)
- $\frac{\partial f}{\partial z} = 3xy^2z^2$ (treat $x$ and $y$ as constants, power rule on $z^3$)

### Exercise 3: Chain Rule in a Neural Network

**Problem:** A neural network computes $y = \text{ReLU}(wx + b)$. Given $\frac{\partial L}{\partial y} = -0.5$, $wx + b = 2$ (positive), and $x = 3$, find $\frac{\partial L}{\partial w}$.

**Solution:**

The computation graph is: $w \to wx+b \to \text{ReLU}(wx+b) \to L$

Apply the chain rule backward:

- Since $wx + b = 2 > 0$, the ReLU is in its linear region: $\frac{\partial \text{ReLU}}{\partial(wx+b)} = 1$
- $\frac{\partial(wx+b)}{\partial w} = x = 3$
- Chain rule: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial(wx+b)} \cdot \frac{\partial(wx+b)}{\partial w} = -0.5 \cdot 1 \cdot 3 = -1.5$

Since the gradient is negative, increasing $w$ will *decrease* the loss — so gradient descent will increase $w$.

### Exercise 4: Numerical Verification (Try It Yourself)

**Problem:** Write code to numerically verify that $\frac{d}{dx}[\sin(e^x)]$ at $x = 0$ equals $\cos(1) \approx 0.5403$.

**Hint:** The chain rule gives $\frac{d}{dx}[\sin(e^x)] = \cos(e^x) \cdot e^x$. At $x = 0$: $\cos(e^0) \cdot e^0 = \cos(1) \cdot 1 = \cos(1)$.

```python
import numpy as np

f = lambda x: np.sin(np.exp(x))
numerical = (f(0 + 1e-7) - f(0 - 1e-7)) / (2e-7)
exact = np.cos(1.0)

print(f"Exact:     {exact:.10f}")
print(f"Numerical: {numerical:.10f}")
# Both ≈ 0.5403023059
```

---

## Summary

- **Derivatives** measure instantaneous rates of change — the slope of a function at a point. In ML, they tell you which direction to adjust each parameter.
- **Numerical differentiation** (finite differences) is the intuitive "nudge and measure" approach. Central difference is most accurate. Use it to *verify* your math.
- **Symbolic rules** (power rule, exponential, etc.) are optimized shortcuts that give exact derivatives for all inputs at once.
- **The chain rule** is the single most important calculus concept for ML. It decomposes the derivative of a composition into a *product* of local derivatives — and this is exactly what backpropagation does.
- **Partial derivatives** extend derivatives to functions of multiple variables by varying one input at a time — essential for models with many parameters.
- **Automatic differentiation** (PyTorch autograd, etc.) applies the chain rule algorithmically through a computation graph, giving you exact derivatives efficiently.

---

## What's Next

Derivatives handle one variable at a time. But neural networks have millions of parameters. How do you take derivatives with respect to *all* of them simultaneously? That's **gradients** — the multivariable generalization that packages all partial derivatives into a single vector pointing in the direction of steepest ascent. Flip it, and you've got the direction of steepest *descent*. That's the next chapter.

---

*Next: Chapter 3 — Gradients, where we combine partial derivatives into vectors that point toward steepest change.*
