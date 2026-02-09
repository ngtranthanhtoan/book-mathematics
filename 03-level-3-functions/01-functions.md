# Chapter 1: Functions — The Core Abstraction

> **Building On** — Algebra taught you to manipulate expressions and solve equations. Now the central abstraction: functions. Everything from this point forward — calculus, linear algebra, probability — is about functions.

---

Every ML model is a function. Linear regression: f(x) = wx + b. Neural network: f(x) = sigma(W_3 sigma(W_2 sigma(W_1 x + b_1) + b_2) + b_3). The entire field of machine learning is: find the function that best maps inputs to outputs. So let's get precise about what a function actually is.

---

## You Already Know This

You've been writing functions for years:

```python
def f(x: float) -> float:
    return 2 * x + 1
```

Same input, same output. Every time. That's it — that's a function in the mathematical sense. No side effects, no randomness, no hidden state. Pure mapping from input to output.

Here's the thing: every concept in this chapter maps to something you already use daily.

| Math Concept | Your SWE Equivalent |
|---|---|
| Function | `def f(x): return ...` — you've written thousands |
| Domain / Range | Input type / output type — like type signatures |
| One-to-one (injective) | A hash function with no collisions |
| Composition | Function chaining: `f(g(x))` is `pipe(g, f)` or `f . g` |
| Inverse | Decode / decrypt: if `f` encodes, `f_inv` decodes |

The math just gives you a formal language for properties you've been reasoning about informally. Let's formalize it.

---

## What Is a Function?

A **function** $f: A \rightarrow B$ is a rule that assigns **exactly one** output in $B$ to **each** input in $A$.

Picture it as a box:

```
         ┌───────────┐
 input   │           │   output
 x ────▶ │     f     │ ────▶ f(x)
         │           │
         └───────────┘
```

You put something in, you get exactly one thing out. Always. The same input always produces the same output.

> **Common Mistake**: A function must give exactly ONE output for each input. A relation that gives multiple outputs is not a function. If you call `f(3)` and sometimes get `7` and sometimes get `12`, that's not a function — that's a bug.

### The Running Example

Throughout this chapter, we'll keep coming back to the building block of every neural network — a single layer:

$$\text{layer}(x) = \text{activation}(W \cdot x + b)$$

That's a function. It takes an input vector $x$, applies a linear transformation ($W \cdot x + b$), then passes the result through an activation function. And a full neural network? It's just layers composed together:

$$\text{model} = \text{layer}_3 \circ \text{layer}_2 \circ \text{layer}_1$$

Every concept we cover — domain, range, injectivity, composition, inverses — we'll see in action through this lens.

---

## Domain, Codomain, and Range

### The Formal Definitions

For a function $f: A \rightarrow B$:

- **Domain** ($A$): The set of all valid inputs — everything you're allowed to feed in.
- **Codomain** ($B$): The set where outputs are declared to live — the "output type."
- **Range** (or Image): The set of outputs that actually get produced: $\{f(x) : x \in A\}$.

The distinction between codomain and range matters. Think of it this way:

```python
def classify(image: Tensor) -> int:
    # Returns 0-9 for digit classification
    ...
```

The codomain is `int` — that's the declared return type. The range is `{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}` — the values that actually come out.

### ASCII Diagram

```
  Domain (valid inputs)              Codomain (declared output space)
  ┌──────────────────┐              ┌──────────────────────────────┐
  │                  │      f       │  ┌───────────────────────┐   │
  │   All valid x    │ ──────────▶  │  │  Range (actual output) │   │
  │                  │              │  │  { f(x) : x in A }     │   │
  │                  │              │  └───────────────────────┘   │
  └──────────────────┘              └──────────────────────────────┘
```

The range sits inside the codomain. Sometimes they're the same set. Sometimes the range is a proper subset.

### ML Activation Functions — Domain and Range

Here's where it gets concrete. These are the functions you'll meet in every neural network:

| Function | Definition | Domain | Range |
|---|---|---|---|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\mathbb{R}$ | $(0, 1)$ |
| ReLU | $\max(0, x)$ | $\mathbb{R}$ | $[0, \infty)$ |
| Tanh | $\tanh(x)$ | $\mathbb{R}$ | $(-1, 1)$ |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $\mathbb{R}^n$ | Probability simplex |
| Neural network classifier | $\text{model}(x)$ | $\mathbb{R}^d$ | $[0, 1]^k$ |

Notice something: the range of sigmoid is $(0, 1)$ — open interval, never actually reaching 0 or 1. That's why logistic regression outputs are probabilities strictly between 0 and 1.

**Back to our running example.** For a single layer $\text{layer}(x) = \text{ReLU}(W \cdot x + b)$ where $W$ is a $20 \times 10$ matrix:

- Domain: $\mathbb{R}^{10}$ (input vectors of dimension 10)
- Range: a subset of $[0, \infty)^{20}$ (ReLU kills negatives, output is 20-dimensional)

Why does this matter? Because if the range of layer 1 doesn't match the domain of layer 2, your network won't even run. You've seen this error message: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`.

---

## One-to-One (Injective) Functions

### The Definition

A function $f$ is **one-to-one** (injective) if different inputs always produce different outputs:

$$f(x_1) = f(x_2) \implies x_1 = x_2$$

Or equivalently: $x_1 \neq x_2 \implies f(x_1) \neq f(x_2)$.

Think: a perfect hash function. No collisions. Every input maps to a unique output.

```
  One-to-One (Injective):          Many-to-One (Not Injective):

  x₁ ──────▶ y₁                    x₁ ──┐
  x₂ ──────▶ y₂                    x₂ ──┼──────▶ y₁
  x₃ ──────▶ y₃                    x₃ ──┘

  No two arrows hit                 Multiple arrows hit
  the same output.                  the same output.
```

### Why It Matters for ML

**One-to-one = no information loss.** If your function is injective, you can (in principle) recover the input from the output. If it's many-to-one, information is destroyed.

- **ReLU is many-to-one**: `ReLU(-3) = ReLU(-7) = 0`. Once negative values hit ReLU, they're all crushed to zero. You can't tell what the original value was. Information is lost.
- **Leaky ReLU is one-to-one**: `LeakyReLU(x) = max(0.01x, x)`. Different inputs always give different outputs. Information is preserved.
- **Classification models are many-to-one**: Many different images of cats all map to the label "cat." That's the whole point — but it means you can't perfectly reconstruct the input from the output.

**Where injectivity is critical:**

| Application | Why Injectivity Matters |
|---|---|
| Invertible Neural Networks | Require one-to-one layers so you can run the network backwards |
| Normalizing Flows | Need bijective (one-to-one AND onto) transformations for density estimation |
| Autoencoders | The encoder is typically many-to-one (that's the compression) |

---

## Onto (Surjective) Functions

A function $f: A \rightarrow B$ is **onto** (surjective) if every element in the codomain $B$ is actually hit by some input:

$$\forall \, y \in B, \; \exists \, x \in A \text{ such that } f(x) = y$$

In other words: the range equals the codomain. No "unused" outputs.

```
  Surjective (Onto):               Not Surjective:

  x₁ ──┐                           x₁ ──────▶ y₁
  x₂ ──┼──────▶ y₁                 x₂ ──────▶ y₂
  x₃ ──────────▶ y₂                              y₃  (never hit!)

  Every y is reached.               y₃ is in the codomain
                                    but nothing maps to it.
```

In ML terms: if your classifier's codomain is {cat, dog, fish} but it never actually outputs "fish" for any input, it's not surjective. Practically this means your model has a dead class.

---

## Bijective Functions

A function that is **both one-to-one AND onto** is called **bijective**. This is the gold standard — a perfect pairing between domain and codomain.

$$f \text{ is bijective} \iff f \text{ is injective AND surjective}$$

```
  Bijective:

  x₁ ◀──────▶ y₁
  x₂ ◀──────▶ y₂
  x₃ ◀──────▶ y₃

  Perfect one-to-one correspondence.
  Every input has a unique output.
  Every output has a unique input.
```

**Why you care**: Only bijective functions have true, well-defined inverses. This is the mathematical foundation of normalizing flows — a family of generative models that use bijective transformations so they can compute exact likelihoods.

---

## Function Composition

### The Definition

Given functions $f: A \rightarrow B$ and $g: B \rightarrow C$, their **composition** $g \circ f$ is:

$$(g \circ f)(x) = g(f(x))$$

Read it right to left: first apply $f$, then apply $g$.

You already know this as function chaining:

```python
# Math notation:  (g ∘ f)(x) = g(f(x))
# Python:
result = g(f(x))

# Or with a pipe utility:
result = pipe(x, f, g)

# Or in functional style:
compose = lambda f, g: lambda x: g(f(x))
```

### ASCII Diagram — Composition as Chained Boxes

```
         ┌─────┐       ┌─────┐
 x ────▶ │  f  │ ────▶ │  g  │ ────▶ g(f(x))
         └─────┘       └─────┘

         └───────────────────────────┘
                    g ∘ f
```

Three functions composed:

```
         ┌─────┐       ┌─────┐       ┌─────┐
 x ────▶ │  f  │ ────▶ │  g  │ ────▶ │  h  │ ────▶ h(g(f(x)))
         └─────┘       └─────┘       └─────┘

         └───────────────────────────────────────┘
                       h ∘ g ∘ f
```

### Deep Learning IS Composition

This is the big idea. A neural network is literally a composition of layer functions.

**Running example — a 3-layer network:**

$$\text{model}(x) = \text{layer}_3(\text{layer}_2(\text{layer}_1(x)))$$

Where each layer is itself a composed function:

$$\text{layer}_i(x) = \text{activation}_i(W_i \cdot x + b_i)$$

Expanding it all out:

$$\text{model}(x) = \sigma(W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2) + b_3)$$

In code, this is exactly what you write:

```python
import numpy as np

def layer(x, W, b, activation):
    """One neural network layer: activation(W @ x + b)"""
    return activation(W @ x + b)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Composition: model = layer3 ∘ layer2 ∘ layer1
def model(x, params):
    h1 = layer(x,  params['W1'], params['b1'], relu)     # layer 1
    h2 = layer(h1, params['W2'], params['b2'], relu)     # layer 2
    out = layer(h2, params['W3'], params['b3'], sigmoid)  # layer 3
    return out
```

Notice the types (dimensions) must chain correctly — the range of each layer must match the domain of the next:

```
  layer₁: R^10 → R^20      (10 inputs, 20 hidden units)
  layer₂: R^20 → R^15      (20 inputs, 15 hidden units)
  layer₃: R^15 → R^3       (15 inputs, 3 outputs)

  model = layer₃ ∘ layer₂ ∘ layer₁ : R^10 → R^3
```

This is exactly like type-checking a chain of function calls. If the output type of one function doesn't match the input type of the next, the composition is undefined — or in your framework, a shape mismatch error.

### Key Properties of Composition

1. **Associativity**: $(h \circ g) \circ f = h \circ (g \circ f)$. The grouping doesn't matter, only the order. Just like `h(g(f(x)))` — parenthesization doesn't change the result.

2. **Not commutative**: $g \circ f \neq f \circ g$ in general. `relu(sigmoid(x))` is not the same as `sigmoid(relu(x))`. Layer order matters.

3. **Domain chaining**: For $g \circ f$ to be defined, the range of $f$ must be a subset of the domain of $g$.

---

## Inverse Functions

### The Definition

For a bijective function $f: A \rightarrow B$, the **inverse** $f^{-1}: B \rightarrow A$ satisfies:

$$f^{-1}(f(x)) = x \quad \text{for all } x \in A$$
$$f(f^{-1}(y)) = y \quad \text{for all } y \in B$$

It completely undoes what $f$ does. Encode then decode, encrypt then decrypt, compress then decompress.

```
         ┌─────┐                ┌───────┐
 x ────▶ │  f  │ ────▶ f(x) ──▶│ f⁻¹   │────▶ x
         └─────┘                └───────┘

   f⁻¹(f(x)) = x   — the round-trip gets you back to where you started
```

**Critical point**: Only bijective (one-to-one AND onto) functions have true inverses. If $f$ is many-to-one, there's no way to uniquely reverse it — multiple inputs mapped to the same output, so which one do you go back to?

### The Classic Example: Sigmoid and Logit

The sigmoid function $\sigma(x) = \frac{1}{1+e^{-x}}$ maps $\mathbb{R} \rightarrow (0, 1)$. It's bijective on this domain/range. So it has an inverse — the **logit** function:

$$\sigma^{-1}(p) = \text{logit}(p) = \ln\!\left(\frac{p}{1-p}\right)$$

**Derivation** (worth seeing once):

Starting with $y = \frac{1}{1+e^{-x}}$, solve for $x$:

1. $y(1 + e^{-x}) = 1$
2. $ye^{-x} = 1 - y$
3. $e^{-x} = \frac{1-y}{y}$
4. $-x = \ln\!\left(\frac{1-y}{y}\right)$
5. $x = \ln\!\left(\frac{y}{1-y}\right)$

So $\sigma^{-1}(y) = \ln\!\left(\frac{y}{1-y}\right)$. This is why logistic regression is called "logistic" — the log-odds (logit) is the inverse of the sigmoid.

### Inverses in ML

| Application | What's Happening |
|---|---|
| **Encoder-Decoder** | Encoder compresses (many-to-one, not truly invertible). Decoder *approximates* the inverse. |
| **Normalizing Flows** | Use bijective transformations so exact inverses exist. This allows computing exact likelihoods. |
| **VAEs / GANs** | Generator learns an approximate inverse of the data-to-latent mapping. |
| **Inverse Problems** | Given observed outputs, recover inputs. Only works cleanly if the forward model is invertible. |

---

## Code Example

```python
import numpy as np

# =============================================================================
# Domain and Range — See It Empirically
# =============================================================================

def demonstrate_domain_range():
    """Verify domain and range of common activation functions."""
    x = np.linspace(-5, 5, 1000)

    sigmoid = 1 / (1 + np.exp(-x))
    relu = np.maximum(0, x)
    tanh = np.tanh(x)

    print("Sigmoid — Domain: R, Range: (0, 1)")
    print(f"  Observed min: {sigmoid.min():.6f}, max: {sigmoid.max():.6f}")

    print("\nReLU — Domain: R, Range: [0, inf)")
    print(f"  Observed min: {relu.min():.6f}, max: {relu.max():.6f}")

    print("\nTanh — Domain: R, Range: (-1, 1)")
    print(f"  Observed min: {tanh.min():.6f}, max: {tanh.max():.6f}")


# =============================================================================
# Injectivity — One-to-One vs Many-to-One
# =============================================================================

def check_injectivity():
    """Show which functions are injective and which aren't."""

    def linear(x):
        return 2 * x + 1

    def square(x):
        return x ** 2

    def relu(x):
        return np.maximum(0, x)

    print("One-to-One Check:")
    print(f"  linear(2) = {linear(2)}, linear(3) = {linear(3)}")
    print(f"  -> Different inputs, different outputs: INJECTIVE")

    print(f"\n  square(-2) = {square(-2)}, square(2) = {square(2)}")
    print(f"  -> Different inputs, same output: NOT INJECTIVE")

    print(f"\n  relu(-1) = {relu(-1)}, relu(-2) = {relu(-2)}")
    print(f"  -> Different inputs, same output (0): NOT INJECTIVE")


# =============================================================================
# Composition — Neural Network as Composed Functions
# =============================================================================

def neural_network_as_composition():
    """
    A neural network IS function composition.
    model(x) = layer3(layer2(layer1(x)))
    """

    def layer(x, W, b, activation):
        return activation(W @ x + b)

    def relu(z):
        return np.maximum(0, z)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Random weights — the structure is what matters here
    np.random.seed(42)
    W1 = np.random.randn(4, 3) * 0.5    # 3 -> 4
    b1 = np.zeros(4)
    W2 = np.random.randn(2, 4) * 0.5    # 4 -> 2
    b2 = np.zeros(2)
    W3 = np.random.randn(1, 2) * 0.5    # 2 -> 1
    b3 = np.zeros(1)

    x = np.array([1.0, 2.0, 3.0])       # Input: R^3

    # Composition: layer3 ∘ layer2 ∘ layer1
    h1 = layer(x, W1, b1, relu)          # R^3 -> R^4
    h2 = layer(h1, W2, b2, relu)         # R^4 -> R^2
    output = layer(h2, W3, b3, sigmoid)  # R^2 -> R^1

    print("Neural Network as Function Composition:")
    print(f"  Input x (R^3):             {x}")
    print(f"  After layer1 (R^3 -> R^4): {h1}")
    print(f"  After layer2 (R^4 -> R^2): {h2}")
    print(f"  After layer3 (R^2 -> R^1): {output}")
    print(f"\n  model(x) = layer3(layer2(layer1(x)))")


# =============================================================================
# Inverse Functions — Encode/Decode Round-Trip
# =============================================================================

def inverse_function_demo():
    """If f is bijective, f_inv(f(x)) = x. Let's verify."""

    # exp and log are inverses
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print("Round-trip: log(exp(x)) should equal x")
    print(f"  x:            {x}")
    print(f"  exp(x):       {np.exp(x)}")
    print(f"  log(exp(x)):  {np.log(np.exp(x))}")
    print(f"  Match: {np.allclose(x, np.log(np.exp(x)))}")

    # sigmoid and logit are inverses
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def logit(p):
        return np.log(p / (1 - p))

    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    p = sigmoid(z)
    z_recovered = logit(p)

    print("\nRound-trip: logit(sigmoid(z)) should equal z")
    print(f"  z:                  {z}")
    print(f"  sigmoid(z):         {np.round(p, 4)}")
    print(f"  logit(sigmoid(z)):  {np.round(z_recovered, 4)}")
    print(f"  Match: {np.allclose(z, z_recovered)}")


# =============================================================================
# Run All Demos
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DOMAIN AND RANGE")
    print("=" * 60)
    demonstrate_domain_range()

    print("\n" + "=" * 60)
    print("INJECTIVITY — ONE-TO-ONE vs MANY-TO-ONE")
    print("=" * 60)
    check_injectivity()

    print("\n" + "=" * 60)
    print("COMPOSITION — NEURAL NETWORK")
    print("=" * 60)
    neural_network_as_composition()

    print("\n" + "=" * 60)
    print("INVERSE FUNCTIONS — ROUND-TRIP")
    print("=" * 60)
    inverse_function_demo()
```

---

## Quick Reference — Where These Concepts Appear in ML

| Concept | ML Application |
|---|---|
| **Domain** | Input space of your model (image dimensions, feature count) |
| **Range** | Output space (probabilities, regression values) |
| **Injective (one-to-one)** | Invertible networks, normalizing flows |
| **Many-to-one** | Classification, pooling, dimensionality reduction |
| **Surjective (onto)** | Full coverage of output space |
| **Bijective** | Normalizing flows, invertible ResNets |
| **Composition** | Deep learning layers, residual connections, feature pipelines |
| **Inverse** | Autoencoders, generative models, inverse problems |

---

## Exercises

### Exercise 1: Domain and Range Analysis

**Problem**: For the function $f(x) = \ln(x^2 + 1)$, determine the domain and range.

**Solution**:
- **Domain**: All real numbers $\mathbb{R}$, because $x^2 + 1 > 0$ for every $x$, so the logarithm is always defined.
- **Range**: $[0, \infty)$. The minimum occurs at $x = 0$, where $f(0) = \ln(1) = 0$. As $|x| \rightarrow \infty$, $f(x) \rightarrow \infty$. No upper bound.

### Exercise 2: Composition in Neural Networks

**Problem**: A network has layers with dimensions: Input(10) -> Hidden(20) -> Hidden(15) -> Output(3). Write this as function composition and identify domain/range of each layer.

**Solution**:

$$\text{Network} = f_3 \circ f_2 \circ f_1$$

- $f_1: \mathbb{R}^{10} \rightarrow \mathbb{R}^{20}$ — first hidden layer
- $f_2: \mathbb{R}^{20} \rightarrow \mathbb{R}^{15}$ — second hidden layer
- $f_3: \mathbb{R}^{15} \rightarrow \mathbb{R}^{3}$ — output layer
- Full network: $\mathbb{R}^{10} \rightarrow \mathbb{R}^{3}$

The range of each layer must match the domain of the next. This is exactly the dimension-checking you do (or your framework does) when defining architectures.

### Exercise 3: Derive the Inverse of Sigmoid

**Problem**: Given sigmoid $\sigma(x) = \frac{1}{1+e^{-x}}$, derive its inverse (the logit function).

**Solution**:

Let $y = \frac{1}{1+e^{-x}}$. Solve for $x$:

1. $y(1 + e^{-x}) = 1$
2. $y + ye^{-x} = 1$
3. $ye^{-x} = 1 - y$
4. $e^{-x} = \frac{1-y}{y}$
5. $-x = \ln\!\left(\frac{1-y}{y}\right)$
6. $x = \ln\!\left(\frac{y}{1-y}\right)$

Therefore: $\sigma^{-1}(y) = \text{logit}(y) = \ln\!\left(\frac{y}{1-y}\right)$

**Domain of logit**: $(0, 1)$ — which is exactly the range of sigmoid. Makes sense: the inverse's domain is the original function's range.

### Exercise 4: Is It Injective?

**Problem**: For each function, determine whether it is injective. Explain why or why not.

1. $f(x) = 3x - 7$
2. $g(x) = x^2$
3. $h(x) = e^x$
4. $\text{ReLU}(x) = \max(0, x)$

**Solution**:

1. **Injective.** If $3x_1 - 7 = 3x_2 - 7$, then $x_1 = x_2$. Linear functions with nonzero slope are always injective.
2. **Not injective.** $g(-2) = g(2) = 4$. Different inputs, same output.
3. **Injective.** The exponential function is strictly increasing, so different inputs always produce different outputs.
4. **Not injective.** $\text{ReLU}(-1) = \text{ReLU}(-100) = 0$. All negative inputs map to zero.

---

## Summary

- A **function** maps each input to exactly one output — `def f(x): return ...` with no side effects.
- **Domain** is the set of valid inputs (input type); **range** is the set of actual outputs (output type).
- **Injective** (one-to-one): no two inputs share an output. Preserves information.
- **Surjective** (onto): every element in the codomain is actually reached.
- **Bijective**: both injective and surjective. Only bijective functions have true inverses.
- **Composition** $(g \circ f)(x) = g(f(x))$: apply $f$ then $g$. Deep learning is composition of layers.
- **Inverse functions** reverse mappings: $f^{-1}(f(x)) = x$. Encode/decode, sigmoid/logit, encrypt/decrypt.
- A neural network layer is $\text{layer}(x) = \text{activation}(W \cdot x + b)$, and the full model is $\text{model} = \text{layer}_n \circ \cdots \circ \text{layer}_1$.

---

> **What's Next** — You know what functions are. Now: which specific functions show up in ML? Sigmoid, ReLU, softmax — the activation functions that make neural networks work.

**Next**: [Chapter 2 - Common Function Types](02-common-function-types.md)
