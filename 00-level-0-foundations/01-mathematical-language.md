# Mathematical Language

You can read Python fluently. You can parse a 500-line config file. But open an ML research paper and it looks like hieroglyphics: $\forall$, $\exists$, $\sum$, $\prod$, $\in$, $\subseteq$. Here's the thing -- mathematical notation is just another programming language. And you already know how to learn those.

Before we dive into sets, functions, or calculus, we need to learn the language. Just like you learned Python syntax before writing algorithms, we'll learn mathematical notation before doing mathematics.

---

## The Running Example: Your First ML Paper Sentence

Here's a line from a real ML paper:

$$\text{minimize} \quad \sum_{i} L(f(x_i; \theta), y_i) + \lambda \|\theta\|^2$$

Right now, that might look intimidating. By the end of this chapter, you'll read it as fluently as Python. In fact, you'll see that it's basically this:

```python
total_loss = sum(loss(model(x[i], theta), y[i]) for i in range(n)) + lam * norm(theta)**2
minimize(total_loss)
```

Same idea. Different syntax. Let's bridge the gap.

---

## The Rosetta Stone: Math Symbols as Code

The single most useful thing in this chapter is this table. Bookmark it, print it, tattoo it on your forearm -- whatever works.

### Operators You Already Know

```
+---------------------+-------------------+----------------------------+-----------------------------+
| Math Symbol         | Name              | Python Equivalent          | Where You'll See It in ML   |
+---------------------+-------------------+----------------------------+-----------------------------+
| +, -, x, /          | Arithmetic        | +, -, *, /                 | Everywhere                  |
| =                   | Equality          | == (comparison)            | Equations, constraints      |
| :=  or  =           | Assignment/def    | = (assignment)             | Update rules, definitions   |
| !=  or  =/=         | Not equal         | !=                         | Constraints                 |
| <, >, <=, >=        | Comparison        | <, >, <=, >=              | Inequalities, bounds        |
+---------------------+-------------------+----------------------------+-----------------------------+
```

### Operators That Look Scary but Aren't

```
+---------------------+-------------------+----------------------------+-----------------------------+
| Math Symbol         | Name              | Python Equivalent          | Where You'll See It in ML   |
+---------------------+-------------------+----------------------------+-----------------------------+
| Summation (big sigma)| Sum from i=1 to n | sum(x[i] for i in range(n))| Loss functions, expectations|
|   n                 |                   | or np.sum(x)               |                             |
|  SUM  x_i           |                   |                            |                             |
|  i=1                |                   |                            |                             |
+---------------------+-------------------+----------------------------+-----------------------------+
| Product (big pi)    | Product i=1 to n  | math.prod(x) or           | Probability (likelihoods)   |
|   n                 |                   | reduce(mul, x)             |                             |
|  PROD x_i           |                   | from functools import reduce|                             |
|  i=1                |                   | from operator import mul   |                             |
+---------------------+-------------------+----------------------------+-----------------------------+
| IN (element of)     | "is in"           | in                         | "x in set S"               |
|                     |                   | x in S                     |                             |
+---------------------+-------------------+----------------------------+-----------------------------+
| FOR ALL             | "for every"       | all(... for x in S)        | Universal statements        |
+---------------------+-------------------+----------------------------+-----------------------------+
| THERE EXISTS        | "there is at      | any(... for x in S)        | Existence proofs            |
|                     |  least one"       |                            |                             |
+---------------------+-------------------+----------------------------+-----------------------------+
| SUBSET OF           | "is contained in" | S.issubset(T)              | Set relationships           |
+---------------------+-------------------+----------------------------+-----------------------------+
| APPROX              | "roughly equal"   | np.isclose(a, b)           | Convergence, approximations |
+---------------------+-------------------+----------------------------+-----------------------------+
```

Here it is in proper notation for reference:

| Math Symbol | Name | Python Equivalent | Where You'll See It in ML |
|-------------|------|-------------------|--------------------------|
| $\sum_{i=1}^{n} x_i$ | Summation | `sum(x[i] for i in range(n))` | Loss functions, expectations |
| $\prod_{i=1}^{n} x_i$ | Product | `reduce(mul, x)` or `np.prod(x)` | Likelihoods, joint probabilities |
| $\in$ | Element of | `in` | "$x \in S$" means `x in S` |
| $\notin$ | Not element of | `not in` | "$x \notin S$" means `x not in S` |
| $\forall$ | For all | `all(...)` | Universal quantifiers |
| $\exists$ | There exists | `any(...)` | Existential quantifiers |
| $\subseteq$ | Subset of | `S.issubset(T)` | Set relationships |
| $\approx$ | Approximately equal | `np.isclose(a, b)` | Convergence checks |
| $\propto$ | Proportional to | (no direct equiv) | "Differs by a constant factor" |
| $\sim$ | Distributed as / similar | (context-dependent) | "$X \sim \mathcal{N}(0,1)$" -- X follows a distribution |

> **You Already Know This.** The $\sum$ symbol is a `for` loop that adds. The $\prod$ symbol is `functools.reduce` with multiplication. The $\in$ symbol is literally Python's `in` keyword. You've been using these concepts since your first week of programming. The math just uses different syntax.

---

## Greek Letters: Just Variable Naming Conventions

You know how Python has `snake_case`, JavaScript has `camelCase`, and Java has `PascalCase`? Greek letters are math's naming convention. There's nothing magical about them. They're just variable names that, by tradition, signal a specific *role*.

Here's your cheat sheet for ML papers:

### Greek Letters You'll See Every Day in ML

| Greek Letter | Name | Typical Role in ML | Think of it as... |
|-------------|------|-------------------|-------------------|
| $\alpha$ | alpha | Learning rate | `learning_rate` |
| $\beta$ | beta | Momentum, secondary rate | `momentum` or `beta_param` |
| $\gamma$ | gamma | Discount factor (RL), decay | `discount_factor` |
| $\delta$ | delta | Small change, error term | `delta` or `diff` |
| $\epsilon$ | epsilon | Tiny constant (e.g., 1e-8), noise | `eps` -- you've seen this in `Adam(eps=1e-8)` |
| $\theta$ | theta | Model parameters (THE big one) | `params` or `weights` |
| $\lambda$ | lambda | Regularization strength | `weight_decay` or `lambda_reg` |
| $\mu$ | mu | Mean | `mean` |
| $\sigma$ | sigma | Standard deviation, sigmoid | `std` or `sigma` |
| $\pi$ | pi | Policy (RL), or 3.14159... | `policy` or `math.pi` |
| $\phi$ | phi | Feature map, parameters | `features` or `phi_params` |
| $\eta$ | eta | Learning rate (alternate) | `lr` |
| $\rho$ | rho | Correlation, density | `rho` |
| $\tau$ | tau | Temperature, time constant | `temperature` |
| $\omega$ | omega | Frequency, weight | `omega` |

### Capital Greek Letters

| Greek Letter | Name | Typical Role in ML | Think of it as... |
|-------------|------|-------------------|-------------------|
| $\Sigma$ | Sigma (capital) | Covariance matrix, summation | `cov_matrix` |
| $\Omega$ | Omega (capital) | Sample space, constraint set | `domain` |
| $\Theta$ | Theta (capital) | Parameter space | `param_space` |
| $\Delta$ | Delta (capital) | Change, difference | `change` |
| $\nabla$ | Nabla | Gradient operator | `grad` -- as in `torch.autograd` |

> **You Already Know This.** Greek letters are just variable names with implied roles, exactly like how you'd never name a boolean `count` or an integer `is_valid`. When a paper writes $\theta$, it means "model parameters." When it writes $\alpha$, it means "learning rate." It's convention, not magic.

---

## Function Notation: You've Been Doing This Since "Hello World"

You've been calling functions since your first day of programming. Math function notation is almost identical.

### The Basics

| Math Notation | What It Means | Python Equivalent |
|--------------|---------------|-------------------|
| $f(x)$ | Apply function f to input x | `f(x)` |
| $f(x, y)$ | Function of two arguments | `f(x, y)` |
| $f: A \to B$ | f takes input from set A, returns something in set B | Type hint: `def f(x: A) -> B` |
| $f \circ g$ | Composition: apply g first, then f | `f(g(x))` |
| $f^{-1}(x)$ | Inverse function | (like decrypting what you encrypted) |
| $f(x; \theta)$ | f depends on x, parameterized by $\theta$ | `f(x, theta)` -- semicolon separates "input" from "config" |

That semicolon convention deserves a closer look. When you see $f(x; \theta)$ in a paper, the stuff before the semicolon is the *input data*, and the stuff after is the *model configuration*. It's like:

```python
# Math: f(x; theta)
# Code:
def model(x, theta):    # theta is the "config" -- learned parameters
    return theta @ x     # simple linear model
```

### Function Composition -- It's Just Chaining

When math writes $f \circ g$, it means "apply g first, then f." You do this all the time:

```python
# Math: (f . g)(x) = f(g(x))
# Code:
result = f(g(x))

# Or more practically, in a neural network:
# h = sigma(W2 * sigma(W1 * x + b1) + b2)
# That's just: activation(layer2(activation(layer1(x))))
```

---

## Equality, Identity, and Approximation: It's Subtle, and It Matters

Here's where math notation has more nuance than most programming languages.

### The Hierarchy

| Symbol | Meaning | Programming Analogy | Example |
|--------|---------|-------------------|---------|
| $\equiv$ | Identical by definition | `#define` or type alias | $f(x) \equiv x^2$ (this *defines* f) |
| $=$ | Equal in value | `==` (comparison) | $2 + 2 = 4$ (these happen to be equal) |
| $\approx$ | Approximately equal | `np.isclose(a, b)` | $\pi \approx 3.14159$ |
| $\sim$ | Distributed as / asymptotically similar | (context-dependent) | $X \sim \mathcal{N}(0, 1)$ or $f(n) \sim n^2$ |
| $\propto$ | Proportional to | "equal up to a constant" | $P(A|B) \propto P(B|A)P(A)$ |

> **Common Mistake.** Don't confuse $=$ (equality) with $\equiv$ (identity/definition). In math, $a = b$ means they happen to be equal right now; $a \equiv b$ means they're the same thing *by definition*, always and forever. It's the difference between `x == 5` (checking a value) and `MY_CONST = 5` (defining what something *is*). When a paper writes $L(\theta) \equiv \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$, it's *defining* the loss function, not claiming some coincidental equality.

### Approximation in ML

In computing, exact values are rare. Understanding approximation symbols saves you debugging time:

- $\approx$ : Numerically close -- think floating-point results
- $\sim$ : Behaves like, in the limit -- used for asymptotic analysis and distributions
- $\propto$ : Equal up to a constant factor -- used in Bayesian inference all the time ("we can ignore the normalizing constant")

Example -- gradient descent is inherently approximate:

$$\theta_{new} \approx \theta_{old} - \alpha \nabla L(\theta)$$

The $\approx$ is there because we're using a finite learning rate $\alpha$ instead of infinitesimal steps. We're taking discrete jumps, not following the exact continuous path.

```python
# The approximation in action:
# Exact: infinitesimal steps along the gradient (impossible in code)
# Approximate: finite step size (what we actually do)
theta_new = theta_old - alpha * gradient  # alpha is finite, so this is approximate
```

---

## Symbols Grouped by "When You'll See Them in ML Papers"

Rather than an alphabetical list, here's how symbols cluster in real ML work.

### Group 1: Loss Functions and Optimization (You'll See These on Page 1)

| Symbol | Meaning | Example |
|--------|---------|---------|
| $\sum$ | Summation | $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ (MSE loss) |
| $\arg\min$ | "The value that minimizes" | $\theta^* = \arg\min_\theta L(\theta)$ |
| $\nabla$ | Gradient | $\nabla_\theta L$ = gradient of loss w.r.t. parameters |
| $\partial$ | Partial derivative | $\frac{\partial L}{\partial w}$ = how L changes when w changes |
| $\alpha$ | Learning rate | $\theta \leftarrow \theta - \alpha \nabla L$ |
| $\|\cdot\|$ | Norm (magnitude) | $\|\theta\|^2$ = sum of squared parameter values |

### Group 2: Probability and Statistics (Every Other Paragraph)

| Symbol | Meaning | Example |
|--------|---------|---------|
| $P(A)$ | Probability of event A | $P(\text{spam}) = 0.3$ |
| $P(A|B)$ | Probability of A given B | $P(\text{spam}|\text{words})$ |
| $\mathbb{E}[X]$ | Expected value | $\mathbb{E}[X] = \sum x \cdot P(x)$ |
| $\sim$ | "Distributed as" | $X \sim \mathcal{N}(\mu, \sigma^2)$ |
| $\mu, \sigma$ | Mean, std deviation | Parameters of distributions |
| $\prod$ | Product | $P(x_1, \ldots, x_n) = \prod_{i} P(x_i)$ (independence) |

### Group 3: Sets and Logic (Model Definitions, Constraints)

| Symbol | Meaning | Example |
|--------|---------|---------|
| $\in$ | "Is an element of" | $x \in \mathbb{R}^n$ ("x is a real-valued vector") |
| $\forall$ | "For all" | $\forall x \in S$ ("for every x in set S") |
| $\exists$ | "There exists" | $\exists x$ such that $f(x) = 0$ |
| $\subseteq$ | "Is a subset of" | $S \subseteq T$ |
| $\cup, \cap$ | Union, intersection | Set operations |
| $\mathbb{R}$ | Real numbers | $\theta \in \mathbb{R}^d$ ("d-dimensional real vector") |

### Group 4: Linear Algebra (Neural Networks, Embeddings)

| Symbol | Meaning | Example |
|--------|---------|---------|
| Capital letters | Matrices | $W$, $X$, $A$ |
| Bold lowercase | Vectors | $\mathbf{x}$, $\mathbf{w}$ |
| $A^T$ | Transpose | Flip rows and columns |
| $A^{-1}$ | Inverse | $A A^{-1} = I$ |
| $\cdot$ or juxtaposition | Matrix multiply | $Wx + b$ (linear layer) |
| $\odot$ | Element-wise multiply | Hadamard product |

### Group 5: Calculus (Backpropagation, Optimization Theory)

| Symbol | Meaning | Example |
|--------|---------|---------|
| $\frac{d}{dx}$ | Derivative | Rate of change |
| $\frac{\partial}{\partial x}$ | Partial derivative | Derivative w.r.t. one variable |
| $\int$ | Integral | "Sum over continuous values" |
| $\lim$ | Limit | Behavior as a value approaches something |

---

## Order of Operations: Precedence Rules (You Already Think This Way)

You know operator precedence in Python. Math has the same concept, with a few extras.

**The standard rules** (you know these):

1. **Parentheses** (innermost first)
2. **Exponents** (including roots) -- and they associate right-to-left: $2^{3^2} = 2^{(3^2)} = 2^9 = 512$
3. **Multiplication and Division** (left to right)
4. **Addition and Subtraction** (left to right)

**The math-specific rules** (pay attention here):

5. **Function application** binds tighter than arithmetic: $\sin x + 1 = (\sin x) + 1$, not $\sin(x + 1)$
6. **Subscripts and superscripts** evaluate first: $x_i^2 = (x_i)^2$, not $x_{(i^2)}$
7. **Summation/Product** scope extends over the entire expression until a clear boundary

> **Common Mistake.** The expression $\sum_{i=1}^{n} x_i \cdot w_i$ means $\sum_{i=1}^{n} (x_i \cdot w_i)$, NOT $(\sum_{i=1}^{n} x_i) \cdot w_i$. The summation "absorbs" everything to its right until the context makes it clear the sum has ended. This trips up a lot of people. When in doubt, look for how the index $i$ is used -- everything involving $i$ is inside the sum.

```python
# CORRECT: sum of element-wise products
correct = np.sum(x * w)          # Sum_i (x_i * w_i)

# WRONG interpretation: product of two sums
wrong = np.sum(x) * np.sum(w)    # (Sum_i x_i) * (Sum_i w_i)

# These are NOT the same!
```

---

## Mathematical Conventions vs Programming Conventions

A few "gotchas" that will trip you up if you're coming purely from code.

### Implicit Multiplication

In math, $2x$ means $2 \times x$. There's no operator. In Python, `2x` is a syntax error -- you need `2 * x`. Simple, but easy to forget when translating formulas.

### Indexing: 1-Based vs 0-Based

Mathematics traditionally starts counting at 1. Python starts at 0. This means:

$$\sum_{i=1}^{n} x_i \quad \longleftrightarrow \quad \text{sum(x[i] for i in range(n))}$$

Notice that math says $i = 1$ to $n$, but Python's `range(n)` gives you $0$ to $n-1$. Both iterate over the same $n$ elements. Always check which convention a paper uses -- some ML papers do use 0-based indexing.

### Variable Naming Conventions

| Convention | Typical Usage |
|-----------|--------------|
| $i, j, k$ | Integer indices (like loop variables) |
| $n, m$ | Sizes, counts (like `len(array)`) |
| $x, y, z$ | Variables (inputs, outputs) |
| $a, b, c$ | Constants |
| $f, g, h$ | Functions |
| Capital letters ($W, X, A$) | Matrices or sets |
| Bold lowercase ($\mathbf{x}, \mathbf{w}$) | Vectors |
| Lowercase ($x, w$) | Scalars |

> **You Already Know This.** This is exactly like Python conventions: `i` for loop indices, `n` for counts, `x` for data, `UPPER_CASE` for constants. Math just has more specific conventions, especially the capital-letter-means-matrix rule.

---

## ML Notation in the Wild: Reading Real Formulas

Let's put it all together. Here are formulas you'll encounter in your first week of reading ML papers, fully decoded.

### Neural Network Forward Pass

$$\hat{y} = \sigma(W^{(L)} \cdot \sigma(W^{(L-1)} \cdots \sigma(W^{(1)} \cdot x)))$$

Reading it: "y-hat equals sigma of W-L times sigma of W-(L-1) times ... sigma of W-1 times x."

Translation: Apply layer 1 ($W^{(1)} \cdot x$), activate ($\sigma$), apply layer 2, activate, ..., apply the final layer, activate. It's just a chain of `linear -> activation -> linear -> activation`.

```python
# The formula above in PyTorch:
h = x
for layer in self.layers:
    h = activation(layer(h))
y_hat = h
```

### Loss Functions

**Mean Squared Error:**
$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

```python
L = np.mean((y - y_hat) ** 2)
```

**Cross-Entropy:**
$$L = -\sum_{i=1}^{n}y_i\log(\hat{y}_i)$$

```python
L = -np.sum(y * np.log(y_hat))
```

### Gradient Descent Update

$$\theta := \theta - \alpha \nabla_\theta L(\theta)$$

The $:=$ means "update" (assignment). Read it as: "Set theta to theta minus the learning rate times the gradient of the loss."

```python
theta = theta - alpha * grad_L  # That's it. That's gradient descent.
```

### Regularization

L1 (Lasso): $\lambda\sum_{i}|\theta_i|$
L2 (Ridge): $\lambda\sum_{i}\theta_i^2$

```python
l1_penalty = lambda_reg * np.sum(np.abs(theta))
l2_penalty = lambda_reg * np.sum(theta ** 2)
```

### Decoding the Running Example

Now let's return to our opening formula:

$$\text{minimize} \quad \sum_{i} L(f(x_i; \theta), y_i) + \lambda \|\theta\|^2$$

Piece by piece:

| Piece | Meaning | Python |
|-------|---------|--------|
| $\text{minimize}$ | Find the $\theta$ that makes this smallest | `optimizer.minimize(...)` |
| $\sum_{i}$ | Sum over all training examples | `sum(... for i in range(n))` |
| $L(\cdot, \cdot)$ | Loss function comparing two things | `loss_fn(prediction, target)` |
| $f(x_i; \theta)$ | Model prediction for input $x_i$ with params $\theta$ | `model(x[i], theta)` |
| $y_i$ | True label for example $i$ | `y[i]` |
| $\lambda$ | Regularization strength | `weight_decay` |
| $\|\theta\|^2$ | Squared norm of parameters | `np.sum(theta ** 2)` |

The full translation:

```python
def objective(theta):
    data_loss = sum(loss_fn(model(x[i], theta), y[i]) for i in range(n))
    reg_loss = weight_decay * np.sum(theta ** 2)
    return data_loss + reg_loss

# minimize this by adjusting theta
```

You can read it. It was always just code in a different font.

---

## Code: The Full Rosetta Stone in Action

```python
import numpy as np
from functools import reduce
from operator import mul

# ============================================================
# SUMMATION: the Sigma symbol
# Math:  sum_{i=1}^{n} x_i
# ============================================================
def summation_example():
    """
    Mathematical notation: Sigma_{i=1}^{n} x_i
    Python translation:   sum(x) or np.sum(x)
    """
    x = np.array([1, 2, 3, 4, 5])

    # The explicit for-loop version (what the sigma literally means):
    total = 0
    for i in range(len(x)):
        total += x[i]

    # The Pythonic version:
    total_py = sum(x)

    # The NumPy version (preferred in practice):
    total_np = np.sum(x)

    print(f"Sum of {x}: {total_np}")
    return total_np


# ============================================================
# PRODUCT: the Pi symbol
# Math:  prod_{i=1}^{n} x_i
# ============================================================
def product_example():
    """
    Mathematical notation: Pi_{i=1}^{n} x_i
    Python translation:   reduce(mul, x) or np.prod(x)
    """
    x = np.array([1, 2, 3, 4, 5])

    # The explicit version:
    product = 1
    for val in x:
        product *= val

    # The functools version (closest to the math concept):
    product_ft = reduce(mul, x)

    # The NumPy version:
    product_np = np.prod(x)

    print(f"Product of {x}: {product_np}")
    return product_np


# ============================================================
# QUANTIFIERS: for-all and there-exists
# Math:  forall x in S, P(x)    and    exists x in S, P(x)
# ============================================================
def quantifier_example():
    """
    forall  ->  all()
    exists  ->  any()
    """
    S = [2, 4, 6, 8, 10]

    # forall x in S: x > 0  (are all elements positive?)
    all_positive = all(x > 0 for x in S)
    print(f"All positive? {all_positive}")  # True

    # exists x in S: x > 7  (is there any element > 7?)
    any_gt_7 = any(x > 7 for x in S)
    print(f"Any > 7? {any_gt_7}")  # True

    # element-of:  5 in S?
    print(f"5 in S? {5 in S}")  # False
    print(f"6 in S? {6 in S}")  # True


# ============================================================
# EQUALITY vs APPROXIMATION
# Math:  = vs approx
# ============================================================
def approximation_example():
    """
    Demonstrating why math distinguishes = from approx,
    and why you need np.isclose() instead of ==
    """
    # Mathematically: 0.1 + 0.2 = 0.3
    # In floating point: 0.1 + 0.2 approx 0.3
    a = 0.1 + 0.2
    b = 0.3

    print(f"0.1 + 0.2 = {a}")
    print(f"0.3       = {b}")
    print(f"Are they equal (==)?           {a == b}")       # False!
    print(f"Are they approx equal?         {np.isclose(a, b)}")  # True

    # This is why ML uses tolerance-based comparisons everywhere
    tolerance = 1e-9
    print(f"Within tolerance ({tolerance})? {abs(a - b) < tolerance}")


# ============================================================
# ORDER OF OPERATIONS
# Sum of products != Product of sums
# ============================================================
def order_of_operations():
    """
    Demonstrating that Sum_i (x_i * w_i)  !=  (Sum_i x_i) * (Sum_i w_i)
    """
    x = np.array([1, 2, 3, 4, 5])
    w = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Sum_i x_i*w_i (weighted sum - the neuron's core operation)
    weighted_sum = np.sum(x * w)
    print(f"Weighted sum Sum(x_i * w_i): {weighted_sum}")

    # (Sum_i x_i) * (Sum_i w_i) -- a completely different thing
    product_of_sums = np.sum(x) * np.sum(w)
    print(f"Product of sums (Sum x_i)*(Sum w_i): {product_of_sums}")

    print(f"Are they equal? {weighted_sum == product_of_sums}")  # No!


# ============================================================
# GREEK LETTERS IN ML: translating a gradient descent step
# ============================================================
def ml_notation_example():
    """
    Common ML notation translated to code:
      theta_new = theta_old - alpha * nabla L(theta) + lambda * theta
    """
    # theta -- model parameters
    theta = np.array([0.5, -0.3, 0.8])

    # alpha -- learning rate
    alpha = 0.01

    # epsilon -- small constant for numerical stability
    epsilon = 1e-8

    # lambda -- regularization strength
    # (can't use 'lambda' as a variable name in Python -- it's a keyword!)
    lambda_reg = 0.001

    # Simulated gradient: nabla L(theta)
    gradient = np.array([0.1, -0.2, 0.05])

    # Gradient descent update: theta_new = theta_old - alpha * nabla L(theta)
    theta_new = theta - alpha * gradient

    print(f"Original theta:  {theta}")
    print(f"Updated theta:   {theta_new}")

    # L2 regularization term: lambda * Sum(theta_i^2)
    l2_penalty = lambda_reg * np.sum(theta ** 2)
    print(f"L2 penalty (lambda * Sum(theta_i^2)): {l2_penalty}")

    # Full update with L2 regularization:
    # theta_new = theta - alpha * (gradient + 2 * lambda * theta)
    theta_with_reg = theta - alpha * (gradient + 2 * lambda_reg * theta)
    print(f"Updated theta (with L2 reg):  {theta_with_reg}")


# ============================================================
# THE RUNNING EXAMPLE: decoding an ML paper formula
# minimize Sum_i L(f(x_i; theta), y_i) + lambda * ||theta||^2
# ============================================================
def running_example():
    """
    Full translation of: minimize Sum_i L(f(x_i; theta), y_i) + lambda ||theta||^2
    """
    np.random.seed(42)

    # "Training data" -- x_i and y_i
    n = 5
    x = np.random.randn(n)          # inputs
    y = 2.0 * x + 0.5               # true labels (linear relationship)

    # theta -- model parameters (we'll use a simple linear model: f(x; theta) = theta[0]*x + theta[1])
    theta = np.array([1.5, 0.3])    # initial guess

    # lambda -- regularization strength
    lambda_reg = 0.01

    # f(x_i; theta) -- model prediction
    def model(x_i, theta):
        return theta[0] * x_i + theta[1]

    # L(prediction, target) -- loss for one example (squared error)
    def loss_fn(pred, target):
        return (pred - target) ** 2

    # Sum_i L(f(x_i; theta), y_i)  -- data loss
    data_loss = sum(loss_fn(model(x[i], theta), y[i]) for i in range(n))

    # lambda * ||theta||^2  -- regularization
    reg_loss = lambda_reg * np.sum(theta ** 2)

    # Total objective
    total_loss = data_loss + reg_loss

    print(f"Data loss:    {data_loss:.4f}")
    print(f"Reg loss:     {reg_loss:.4f}")
    print(f"Total loss:   {total_loss:.4f}")
    print(f"(We'd minimize this by adjusting theta via gradient descent)")


if __name__ == "__main__":
    print("=" * 60)
    print("SUMMATION (Sigma)")
    print("=" * 60)
    summation_example()

    print(f"\n{'=' * 60}")
    print("PRODUCT (Pi)")
    print("=" * 60)
    product_example()

    print(f"\n{'=' * 60}")
    print("QUANTIFIERS (for-all / there-exists)")
    print("=" * 60)
    quantifier_example()

    print(f"\n{'=' * 60}")
    print("APPROXIMATION (= vs approx)")
    print("=" * 60)
    approximation_example()

    print(f"\n{'=' * 60}")
    print("ORDER OF OPERATIONS")
    print("=" * 60)
    order_of_operations()

    print(f"\n{'=' * 60}")
    print("GREEK LETTERS IN ML")
    print("=" * 60)
    ml_notation_example()

    print(f"\n{'=' * 60}")
    print("RUNNING EXAMPLE: Full ML formula decoded")
    print("=" * 60)
    running_example()
```

---

## Common Mistakes (a.k.a. "Bugs in Your Math Parser")

These are the "gotchas" that trip up software engineers reading math for the first time. Think of them as common bugs you'll want to watch for.

### 1. Indexing Off-by-One

Math uses 1-based indexing. Python uses 0-based. When translating $\sum_{i=1}^{n} x_i$, you need `range(1, n+1)` if indexing into a 1-based structure, or `range(n)` if indexing into a Python array. Always double-check.

### 2. Implicit Multiplication

$2x$ in math is `2 * x` in code. $xy$ in math is `x * y` in code. There's no `*` on the whiteboard. You have to add it.

### 3. Floating-Point Equality

Never use `==` for float comparison. If a math derivation says $a = b$, your code should use `np.isclose(a, b)` or a tolerance check. This isn't a math thing -- it's a computer thing -- but you'll encounter it every time you translate math to code.

### 4. Summation Scope

$\sum_{i=1}^{n} x_i w_i + b$ means $\left(\sum_{i=1}^{n} x_i w_i\right) + b$, not $\sum_{i=1}^{n} (x_i w_i + b)$. The $b$ doesn't depend on $i$, so it's outside the sum. Look at which variables use the index.

### 5. Greek Letter Lookalikes

| This letter... | Is NOT this letter... |
|----------------|----------------------|
| $\nu$ (nu) | $v$ (v) |
| $\rho$ (rho) | $p$ (p) |
| $\omega$ (omega) | $w$ (w) |
| $\eta$ (eta) | $n$ (n) |

When in doubt, check the context. If the paper defined $\eta$ as the learning rate three pages ago, that curvy thing is eta, not the letter n.

### 6. Confusing Definition ($\equiv$) with Equality ($=$)

As noted above: $a \equiv b$ means "by definition, a IS b." $a = b$ means "these happen to have the same value." When you see $\equiv$, you're looking at a *definition*. When you see $=$, you're looking at a *claim* or *result*.

---

## Mathematical Proofs are Unit Tests

One more bridge before we wrap up.

> **You Already Know This.** Mathematical proofs and unit tests serve the same purpose: they verify that a statement is true. A proof says "this property holds for ALL valid inputs." A unit test says "this property holds for THESE specific inputs." Proofs are more powerful (they cover every case), but the mindset is identical: state what should be true, then show that it is.

| Proof Concept | Testing Concept |
|--------------|-----------------|
| Theorem statement | Test description / docstring |
| Assumptions (given...) | Test setup / fixtures |
| Proof steps | Assertions |
| QED | Test passes |
| Proof by contradiction | Expecting an exception |
| Counterexample | Failing test case |

When you read "Proof: Assume for contradiction that..." -- that's just `with pytest.raises(...)`.

---

## Exercises

### Exercise 1: Notation Translation

Translate this mathematical expression into Python code:

$$z = \sum_{i=1}^{n} w_i \cdot x_i + b$$

This is the core operation of a single neuron. You've probably implemented it a hundred times.

**Solution:**

```python
import numpy as np

def linear_combination(w, x, b):
    """
    Computes z = Sum_i w_i * x_i + b  (a single neuron)
    This is literally np.dot(w, x) + b
    """
    return np.dot(w, x) + b

# Test
w = np.array([0.5, -0.3, 0.8])
x = np.array([1.0, 2.0, 3.0])
b = 0.1
z = linear_combination(w, x, b)
print(f"z = {z}")  # z = 2.0
```

### Exercise 2: Equality vs Approximation

Determine whether these should use `==` or `np.isclose()`:

a) Checking if an integer counter reached a limit
b) Checking if a loss value converged
c) Checking if two normalized vectors are equal

**Solution:**

```python
import numpy as np

# a) Integer counter -- use ==
# Integers are exact. No floating-point weirdness.
counter = 10
limit = 10
print(counter == limit)  # True

# b) Loss convergence -- use np.isclose() or a threshold
# Loss values are floats. They'll never be EXACTLY equal.
old_loss = 0.001234
new_loss = 0.001235
print(np.isclose(old_loss, new_loss, rtol=1e-3))  # True

# c) Normalized vectors -- use np.allclose()
# Normalization introduces floating-point errors.
v1 = np.array([1, 0, 0])
v2 = np.array([1, 0, 0]) / np.linalg.norm([1, 0, 0])
print(np.allclose(v1, v2))  # True
```

### Exercise 3: Order of Operations

What is the value of each expression?

a) $2^{3^2}$
b) $\sum_{i=1}^{3} i^2$
c) $\frac{1}{2}x^2$ where $x = 4$

**Solution:**

```python
import numpy as np

# a) 2^(3^2) = 2^9 = 512
# Exponents associate RIGHT to LEFT.
# It's 2^(3^2), not (2^3)^2.
result_a = 2 ** (3 ** 2)
print(f"2^(3^2) = {result_a}")  # 512

# b) 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
result_b = sum(i**2 for i in range(1, 4))
print(f"Sum i^2 for i=1 to 3: {result_b}")  # 14

# c) (1/2) * 4^2 = 0.5 * 16 = 8
x = 4
result_c = 0.5 * x**2
print(f"(1/2)x^2 where x=4: {result_c}")  # 8.0
```

### Exercise 4: Decode This ML Formula

Read this formula and write it in Python:

$$\hat{y} = \sigma\left(\sum_{j=1}^{m} w_j x_j + b\right) \quad \text{where} \quad \sigma(z) \equiv \frac{1}{1 + e^{-z}}$$

**Solution:**

```python
import numpy as np

def sigmoid(z):
    """sigma(z) = 1 / (1 + e^(-z))  -- note the 'triple equals' means this IS the definition"""
    return 1 / (1 + np.exp(-z))

def predict(w, x, b):
    """y_hat = sigma(Sum_j w_j * x_j + b)"""
    z = np.dot(w, x) + b          # the linear part: Sum_j w_j * x_j + b
    y_hat = sigmoid(z)              # the activation: sigma(z)
    return y_hat

# Test
w = np.array([0.5, -0.3, 0.8])
x = np.array([1.0, 2.0, 3.0])
b = 0.1
print(f"y_hat = {predict(w, x, b):.4f}")  # A probability between 0 and 1
```

---

## Summary

Here's what you now have in your toolkit:

- **Mathematical notation is just syntax** -- and you already understand the semantics from programming. The symbols are unfamiliar, but the concepts are not.
- **Greek letters are variable names** with implied roles ($\theta$ = params, $\alpha$ = learning rate, $\sigma$ = std dev). They follow conventions just like `snake_case` and `camelCase`.
- **The big operators translate directly**: $\sum$ = `sum()`, $\prod$ = `reduce(mul, ...)`, $\forall$ = `all()`, $\exists$ = `any()`, $\in$ = `in`.
- **Equality has layers**: $\equiv$ (definition) $\supset$ $=$ (equality) $\supset$ $\approx$ (approximation). Don't confuse them.
- **Order of operations** extends beyond PEMDAS: function application, subscripts, and summation scope all have rules. When in doubt, add parentheses.
- **Mathematical conventions differ from code**: 1-based indexing, implicit multiplication, and the semicolon convention ($f(x; \theta)$) for separating inputs from parameters.
- **Proofs are unit tests.** Both verify truth. You already have the mindset.
- **You can now read**: $\text{minimize} \sum_{i} L(f(x_i; \theta), y_i) + \lambda\|\theta\|^2$

---

**What's Next:** You can read the symbols. Now let's use them to talk about collections and logic -- sets and Boolean algebra.

*Next: [Sets and Logic](./02-sets-and-logic.md)*
