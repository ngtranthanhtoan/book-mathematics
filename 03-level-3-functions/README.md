# Level 3: Functions

Every ML model you've ever trained is a function. Not metaphorically — literally. A neural network that classifies images? That's a function. A transformer that generates text? That's a function. GPT, BERT, ResNet, XGBoost — all functions. They take inputs and produce outputs according to a mathematical rule. If you understand functions deeply, you understand what your models actually are.

## Functions in Math = Pure Functions in Code

You've been writing functions for years:

```python
def predict(features):
    return model(features)
```

The mathematical definition just formalizes what you already know. In code, a pure function always returns the same output for the same input — no side effects, no hidden state. In math, a function is exactly that: a mapping from inputs to outputs. `f(x) = x²` is pure. `f(x, y) = x + y` is pure. `f(x; θ) = θx + b` is pure (the semicolon just separates data from parameters).

Here's the insight: **every ML model is a parameterized function** `f(x; θ)`. The model takes data `x` and weights `θ`, and produces a prediction. Training means optimizing `θ`. Inference means calling `f` with fixed `θ`. That's it. That's all of machine learning.

## Neural Networks Are Function Composition

Consider a simple 3-layer neural network:

```
Input x (784 pixels from MNIST)
    ↓
Layer 1: f₁(x) = ReLU(W₁x + b₁)     → h₁
    ↓
Layer 2: f₂(h₁) = ReLU(W₂h₁ + b₂)   → h₂
    ↓
Layer 3: f₃(h₂) = softmax(W₃h₂ + b₃) → ŷ
```

The full model is `f(x; θ) = f₃ ∘ f₂ ∘ f₁(x)`. That's function composition — the same as middleware pipelines in web frameworks. Each layer transforms its input and passes the result to the next. The output of `f₁` must match the input of `f₂` (domain and range chaining). The parameters `θ = {W₁, b₁, W₂, b₂, W₃, b₃}` are what you optimize during training.

This level teaches you the formal language of functions so you can reason precisely about what models do, why architectures work, and where things break.

## What You'll Learn

### [Chapter 1: Functions](01-functions.md)
Domain, range, injectivity, composition, inverses — and exactly where each shows up in ML. Why ReLU is many-to-one (information loss). Why normalizing flows need bijective layers (invertibility). Why a neural network is literally `f₃ ∘ f₂ ∘ f₁` (composition). You already understand these ideas from code; this chapter gives you the mathematical vocabulary.

### [Chapter 2: Common Function Types](02-common-function-types.md)
ReLU, sigmoid, tanh, softmax — the functions that power ML. Why each exists, where it's used, when it fails. You'll see why we switched from sigmoid to ReLU (vanishing gradients), why softmax needs the max-subtraction trick (numerical stability), and how temperature scaling controls confidence. Includes gradient flow analysis so you understand which activation functions preserve signal during backprop.

### [Chapter 3: Multivariable Functions](03-multivariable-functions.md)
Real models take many inputs and produce many outputs. A MNIST classifier maps `R^784 → R^10`. A recommender maps `(user_features, movie_features) → rating`. This chapter covers multivariable notation, parameterized functions `f(x; θ)`, loss surfaces, contour plots, and why optimization is about navigating high-dimensional landscapes. Think of this as understanding functions that operate on vectors instead of scalars.

## Building On: Level 2 (Algebra)

Level 2 gave you the tools to manipulate expressions and solve equations. You learned about variables, polynomials, exponentials, and inequalities. Functions build directly on that foundation:

- **Variables** → arguments to functions
- **Expressions** → function definitions (`f(x) = 2x + 1`)
- **Equations** → finding inputs that produce specific outputs (`f(x) = 0`)
- **Exponentials and logs** → activation functions (sigmoid = `1/(1 + e^(-x))`)

If you're comfortable with algebraic manipulation — simplifying expressions, solving for variables, understanding exponential growth — you're ready for this level.

## What Comes Next: Level 4 (Linear Algebra)

After this level, you'll move to **Level 4: Linear Algebra** — vectors, matrices, and transformations. Linear algebra is what happens when your function arguments become vectors instead of scalars.

- **f(x)** where x is a number → Level 3 (this level)
- **f(x)** where x is a vector → Level 4 (linear algebra)

Every neural network layer is a linear transformation (`Wx + b`) followed by a nonlinear activation (`ReLU`, `sigmoid`). Linear algebra teaches you how to compute those transformations efficiently, what weight matrices actually do, and how to reason about high-dimensional spaces. Think of it as scaling up from `f: R → R` to `f: R^n → R^m`.

## Navigation

| Chapter | Topic | ML Connection |
|---------|-------|---------------|
| [01-functions.md](01-functions.md) | Domain, range, composition, inverses | Neural network layers, bijective flows |
| [02-common-function-types.md](02-common-function-types.md) | ReLU, sigmoid, tanh, softmax | Activation functions, gradient flow |
| [03-multivariable-functions.md](03-multivariable-functions.md) | R^n → R^m, loss surfaces, contours | Model architecture, optimization landscapes |

---

**Start here**: [Chapter 1 - Functions](01-functions.md)
