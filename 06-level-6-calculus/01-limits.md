# Chapter 1: Limits — The Mathematics of "Are We There Yet?"

## Building On

You have mastered linear algebra: vectors, matrices, transformations. You can rotate a point in space, project data onto lower dimensions, and decompose matrices into their fundamental components. But ML models do not just transform data -- they **learn** by iteratively reducing error. Your optimizer takes a step, checks the loss, takes another step, checks again. Every training loop is a journey toward a minimum.

To understand "approaching a minimum," you need **limits**. This is where calculus begins.

---

## The Problem That Starts Everything

Your neural network's loss is 0.523 at epoch 100 and 0.519 at epoch 101. Is it converging? How do you know it is not just bouncing around? What value is it heading toward, and will it ever get there?

Let us watch a training run:

```
Epoch   Loss
─────   ──────────
  1     2.4521
  5     1.1038
 10     0.7294
 50     0.5401
100     0.5230
200     0.5189
500     0.5180
1000    0.5180
2000    0.5180
```

The loss values form a sequence: $L_1, L_2, L_3, \ldots, L_n$. As $n$ grows, the values cluster tighter and tighter around 0.518. You *feel* the convergence. You know the model is "done." But how would you make that precise?

**That is a limit.**

$$\lim_{n \to \infty} L_n = 0.518$$

The sequence of loss values **approaches** 0.518 as the number of epochs grows without bound. The model may never hit 0.518000... exactly -- but it gets arbitrarily close. That distinction between "approaches" and "equals" is the entire game.

---

## Code First: Computing a Limit Numerically

Before we touch any formal math, let us do what engineers do -- run the code and observe the pattern.

```python
import numpy as np

def numerical_limit(f, a, tolerances=None):
    """
    Estimate lim_{x -> a} f(x) by evaluating f at points
    increasingly close to a from both sides.

    This is exactly what you do when you check convergence:
    shrink the step, see if the answer stabilizes.
    """
    if tolerances is None:
        tolerances = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    print(f"Estimating lim_{{x -> {a}}} f(x)")
    print(f"{'h':>12}  {'f(a - h)':>14}  {'f(a + h)':>14}  {'avg':>14}")
    print("─" * 60)

    for h in tolerances:
        left  = f(a - h)
        right = f(a + h)
        avg   = (left + right) / 2
        print(f"{h:>12.6f}  {left:>14.8f}  {right:>14.8f}  {avg:>14.8f}")

    return avg

# ── Example: lim_{x→0} sin(x)/x ──
# This limit is the backbone of the derivative of sin(x).
print("═" * 60)
print("Example: lim_{x→0} sin(x)/x")
print("═" * 60)
f = lambda x: np.sin(x) / x
result = numerical_limit(f, 0)
print(f"\nThe values converge to: {result:.8f}")
print(f"Actual answer:          1.00000000")
```

**Output:**

```
════════════════════════════════════════════════════════════
Example: lim_{x→0} sin(x)/x
════════════════════════════════════════════════════════════
Estimating lim_{x -> 0} f(x)
           h        f(a - h)        f(a + h)             avg
────────────────────────────────────────────────────────────
    0.100000    0.99833417    0.99833417    0.99833417
    0.010000    0.99998333    0.99998333    0.99998333
    0.001000    0.99999983    0.99999983    0.99999983
    0.000100    1.00000000    1.00000000    1.00000000
    0.000010    1.00000000    1.00000000    1.00000000

The values converge to: 1.00000000
Actual answer:          1.00000000
```

You just computed a limit. You picked a target point ($x = 0$), approached it from both sides, and watched the function values stabilize. The pattern -- shrink the distance, observe convergence -- is the entire intuition behind limits.

> **You Already Know This**
>
> This is the same pattern as a convergence check in any iterative algorithm:
> ```python
> while abs(current - previous) > tolerance:
>     previous = current
>     current = next_estimate()
> ```
> You keep iterating until successive values are "close enough." Limits formalize
> what "close enough" means when you let the tolerance shrink to zero.

---

## What Is a Limit? (Intuitive Version)

The limit $\lim_{x \to a} f(x) = L$ says:

> **As $x$ gets closer and closer to $a$ (but never equals $a$), the function values $f(x)$ get closer and closer to $L$.**

Three things to burn into memory:

1. **We never plug in $a$ itself.** The limit describes the *approach*, not the arrival.
2. **$f(a)$ might not exist.** The function can have a hole at $a$ and still have a limit there.
3. **$f(a)$ might exist but differ from the limit.** The function could be defined at $a$ but with a "wrong" value.

### ASCII Art: Function Approaching a Value

```
    f(x)
     ↑
   4 ┤
     │
   3 ┤· · · · · · · · · · · ·○· · · · · · ·  ← limit = 3
     │                      ╱  ╲
   2 ┤                    ╱      ╲
     │                  ╱          ╲
   1 ┤                ╱
     │              ╱
   0 ┤────────────╱──────┬──────────────→ x
                         2

    As x → 2, f(x) → 3
    The ○ means f(2) is UNDEFINED — there is a hole.
    But the limit is still 3.
```

> **Common Mistake**
>
> "A limit describes what a function APPROACHES, not what it equals AT that point."
> $f(2)$ might not exist, but $\lim_{x \to 2} f(x)$ can still be $3$.
> These are fundamentally different questions:
> - "What is $f(2)$?" asks about a single point.
> - "What is $\lim_{x \to 2} f(x)$?" asks about the *neighborhood* around that point.

---

## One-Sided Limits: Approaching From the Left and Right

Sometimes a function behaves differently depending on which direction you approach from. Think of it like a version rollout: the system behavior *before* the deploy (left) might differ from *after* the deploy (right).

$$\lim_{x \to a^-} f(x) = L^- \quad \text{(left-hand limit: approaching from below)}$$

$$\lim_{x \to a^+} f(x) = L^+ \quad \text{(right-hand limit: approaching from above)}$$

**The two-sided limit exists if and only if both one-sided limits exist and are equal:**

$$\lim_{x \to a} f(x) = L \iff L^- = L^+ = L$$

### ASCII Art: Left and Right Limits

**Case 1: They agree (limit exists)**

```
    f(x)
     ↑
   3 ┤· · · · · · ·○· · · · · · · · ·  ← both sides → 3
     │            ╱  ╲
   2 ┤          ╱      ╲
     │        ╱          ╲
   1 ┤      ╱
     │    ╱
   0 ┤──╱───────────┬──────────────→ x
                     a

     Left-hand limit  = 3  ✓
     Right-hand limit = 3  ✓
     Limit exists and equals 3
```

**Case 2: They disagree (limit does NOT exist)**

```
    f(x)
     ↑
   3 ┤              ●━━━━━━━━━━━━━━
     │             ╱
   2 ┤            |
     │            |   gap!
   1 ┤━━━━━━━━━━━●
     │
   0 ┤────────────┬────────────────→ x
                  a

     Left-hand limit  = 1
     Right-hand limit = 3
     Limit DOES NOT EXIST (the one-sided limits disagree)
```

> **You Already Know This**
>
> One-sided limits are like checking a feature flag's behavior:
> - What happens for users *just below* the rollout threshold?
> - What happens for users *just above* it?
>
> If the experience is different, you have a discontinuity -- a "jump" in behavior.

---

## Why Do We Need Rigor? The Epsilon-Delta Definition

The intuitive version -- "gets closer and closer" -- is fine for building intuition. But it is dangerously vague. How close is "close enough"? Who decides?

Consider this scenario: you are writing an SLA for a service. Your client says "I need the response time to be close to 50ms." That is useless. What they need to say is:

> "For **any** tolerance $\varepsilon$ I specify (say, within 2ms of 50ms), you must guarantee there exists a configuration $\delta$ (say, keeping query size under some bound) such that the response time stays within my tolerance."

That is the epsilon-delta definition. It makes "close" into a precise contract.

### Formal Definition

The limit $\lim_{x \to a} f(x) = L$ exists if and only if:

$$\forall \varepsilon > 0, \; \exists \delta > 0 \text{ such that } 0 < |x - a| < \delta \implies |f(x) - L| < \varepsilon$$

### Translation

Let us unpack this piece by piece, because every symbol matters:

| Symbol | Meaning | SWE Analogy |
|--------|---------|-------------|
| $\forall \varepsilon > 0$ | "For ANY tolerance you demand..." | "No matter how tight the SLA..." |
| $\exists \delta > 0$ | "...I can find a neighborhood..." | "...I can configure the system..." |
| $0 < \lvert x - a \rvert < \delta$ | "...for inputs within $\delta$ of $a$ (but not $a$ itself)..." | "...for requests within parameters..." |
| $\lvert f(x) - L \rvert < \varepsilon$ | "...the output stays within $\varepsilon$ of $L$." | "...response stays within SLA bounds." |

The $0 < |x - a|$ part (note: strictly greater than zero) is crucial -- we exclude the point $a$ itself. The limit does not care what happens *at* $a$, only *near* $a$.

> **You Already Know This**
>
> The epsilon-delta definition is an API contract:
> - **Client** (epsilon): "I demand accuracy within $\varepsilon$."
> - **Server** (delta): "Stay within $\delta$ of the input target, and I guarantee it."
>
> The limit exists if the server can meet *any* SLA the client proposes, no matter how strict.
> This is exactly "for any tolerance $\varepsilon$ you demand, I can find a neighborhood $\delta$
> that satisfies it."

### Epsilon-Delta in Code

```python
def verify_epsilon_delta(f, a, L, epsilon, search_range=1.0, num_points=10000):
    """
    Given a candidate limit L, verify the epsilon-delta condition:
    for the given epsilon, find a delta such that
    0 < |x - a| < delta  =>  |f(x) - L| < epsilon

    Returns (success, delta) if found.
    """
    # Search for a valid delta by trying progressively smaller values
    for delta in np.linspace(search_range, 1e-10, 1000):
        # Sample points in the punctured neighborhood (0 < |x-a| < delta)
        x_left  = np.linspace(a - delta, a - delta/num_points, num_points // 2)
        x_right = np.linspace(a + delta/num_points, a + delta, num_points // 2)
        x_test  = np.concatenate([x_left, x_right])

        # Check: are ALL function values within epsilon of L?
        if np.all(np.abs(f(x_test) - L) < epsilon):
            return True, delta

    return False, None

# Verify: lim_{x→0} sin(x)/x = 1
f = lambda x: np.sin(x) / x

for eps in [0.1, 0.01, 0.001]:
    success, delta = verify_epsilon_delta(f, a=0, L=1.0, epsilon=eps)
    print(f"ε = {eps}: found δ = {delta:.6f}" if success else f"ε = {eps}: FAILED")

# Output:
# ε = 0.1:   found δ = 0.774775
# ε = 0.01:  found δ = 0.244244
# ε = 0.001: found δ = 0.077077
```

Notice the pattern: as epsilon shrinks, delta shrinks too. The server tightens its constraints to meet the client's tighter SLA. That is the contract working.

---

## Running Example: Training Loss as a Limit

Let us formalize what we saw at the start. During training, you produce a sequence of loss values:

$$L_1, L_2, L_3, \ldots, L_n, \ldots$$

We say this sequence **converges** to $L^*$ if:

$$\lim_{n \to \infty} L_n = L^*$$

In epsilon-delta language for sequences:

$$\forall \varepsilon > 0, \; \exists N \in \mathbb{N} \text{ such that } n > N \implies |L_n - L^*| < \varepsilon$$

**Translation:** "For any tolerance you specify, there exists an epoch $N$ after which the loss *never strays* more than $\varepsilon$ from $L^*$."

This is literally your early stopping criterion:

```python
def train_with_convergence_check(model, data, lr=0.01, epsilon=1e-5, patience=10):
    """
    Train until the loss converges -- i.e., until the limit is reached
    within tolerance epsilon.

    This is the epsilon-delta definition in production code:
    - epsilon is your ε (how close is "converged")
    - patience is related to N (how many epochs must satisfy the condition)
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    prev_loss = float('inf')
    stable_count = 0

    for epoch in range(10000):
        loss = train_one_epoch(model, data, optimizer)

        # ── This is the limit check ──
        # |L_n - L_{n-1}| < epsilon  (successive differences shrinking)
        if abs(loss - prev_loss) < epsilon:
            stable_count += 1
            if stable_count >= patience:   # N epochs of stability
                print(f"Converged at epoch {epoch}, loss = {loss:.6f}")
                return loss
        else:
            stable_count = 0

        prev_loss = loss

    print("Did not converge within 10000 epochs")
    return loss
```

Every time you write a convergence check, you are implementing the definition of a limit.

---

## Limit Laws: The Algebra of Limits

Once you know individual limits exist, you can combine them. These laws are what let you simplify complex expressions instead of computing everything numerically.

Given $\lim_{x \to a} f(x) = L$ and $\lim_{x \to a} g(x) = M$:

| Law | Formula | Why You Care |
|-----|---------|-------------|
| **Sum** | $\lim_{x \to a} [f(x) + g(x)] = L + M$ | Loss = data_loss + regularization_loss. If both converge, the total converges. |
| **Product** | $\lim_{x \to a} [f(x) \cdot g(x)] = L \cdot M$ | Scaled loss converges to scaled limit. |
| **Quotient** | $\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{L}{M}$, if $M \neq 0$ | Normalized metrics are well-behaved when the denominator stays bounded away from 0. |
| **Power** | $\lim_{x \to a} [f(x)]^n = L^n$ | Polynomial expressions of convergent quantities converge. |
| **Composition** | $\lim_{x \to a} g(f(x)) = g(L)$, if $g$ continuous at $L$ | Applying a continuous activation to a converging pre-activation preserves convergence. |

**Translation:** Limits play nicely with arithmetic. If two things converge, their sum, product, and quotient (denominator nonzero) all converge to what you would expect. This means you can reason about complex loss functions by breaking them into parts.

---

## Important Limits You Will See Again

These three limits appear constantly in calculus and ML. Memorize them.

### 1. The sinc limit

$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

This is a $\frac{0}{0}$ form -- both numerator and denominator go to zero. But their *ratio* approaches 1. You will see this when we derive the derivative of $\sin x$.

### 2. The exponential rate limit

$$\lim_{x \to 0} \frac{e^x - 1}{x} = 1$$

Also a $\frac{0}{0}$ form. This tells you that $e^x \approx 1 + x$ for small $x$ -- a linearization you will use constantly.

### 3. The definition of $e$

$$\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e \approx 2.71828$$

Compound interest taken to its logical extreme. This connects to continuous growth, exponential decay, and the softmax function in ML.

```python
# Verify: (1 + 1/n)^n → e
print("n           (1 + 1/n)^n       error")
print("─" * 50)
for n in [10, 100, 1000, 10000, 100000, 1000000]:
    val = (1 + 1/n)**n
    print(f"{n:<12}{val:<18.10f}{abs(val - np.e):.2e}")

# Output:
# n           (1 + 1/n)^n       error
# ──────────────────────────────────────────────────
# 10          2.5937424601      1.18e-01
# 100         2.7048138294      1.34e-02
# 1000        2.7169239322      1.35e-03
# 10000       2.7181459268      1.35e-04
# 100000      2.7182686942      1.35e-05
# 1000000     2.7182804691      1.35e-06
```

The error shrinks by a factor of 10 each time $n$ grows by a factor of 10. The sequence converges to $e$ -- that is a limit.

---

## Indeterminate Forms and L'Hopital's Rule: Handling Edge Cases

Sometimes you try to evaluate a limit and get $\frac{0}{0}$ or $\frac{\infty}{\infty}$. These are called **indeterminate forms** -- the naive computation gives you garbage, and you need a smarter approach.

> **You Already Know This**
>
> This is exactly like handling edge cases in code. What does `0 / 0` return? A `NaN`.
> But the underlying quantity might be perfectly well-defined -- you just need a better
> way to compute it.
>
> L'Hopital's rule is the mathematical equivalent of "catch the `NaN` and handle it gracefully."

### L'Hopital's Rule

If $\lim_{x \to a} f(x) = 0$ and $\lim_{x \to a} g(x) = 0$ (or both $\to \pm\infty$), then:

$$\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$

provided the right-hand limit exists.

**Translation:** When you get $\frac{0}{0}$, take the derivative of the top and bottom separately, then try the limit again. You are replacing a hopeless computation with one that might work.

### Example

$$\lim_{x \to 0} \frac{\sin x}{x} = \frac{0}{0} \text{ (indeterminate)}$$

Apply L'Hopital's:

$$= \lim_{x \to 0} \frac{\cos x}{1} = \frac{1}{1} = 1$$

We will revisit L'Hopital's rule properly after we cover derivatives, but it is worth previewing here: the existence of this tool depends entirely on limits being well-defined.

---

## Continuity: Functions With No Surprises

A function $f$ is **continuous** at point $a$ if three conditions hold:

1. $f(a)$ is defined (the function has a value at $a$)
2. $\lim_{x \to a} f(x)$ exists (the approach is well-behaved)
3. $\lim_{x \to a} f(x) = f(a)$ (the value matches the approach)

**Translation:** No surprises. What you expect when approaching $a$ is exactly what you get when you arrive. There is no gap, no jump, no explosion.

> **You Already Know This**
>
> A continuous function is like a **pure function with no side effects**.
> - Same input, same output -- no hidden state that changes behavior.
> - No surprises at boundary values.
> - The function's behavior near a point *perfectly predicts* its behavior at that point.
>
> A discontinuous function is like a function with a hidden `if` statement that changes
> behavior at certain inputs -- a code smell if you did not expect it.

### Types of Continuity

| Category | Examples | Notes |
|----------|----------|-------|
| **Continuous everywhere** | Polynomials, $e^x$, $\sin x$, $\cos x$ | Safe to evaluate anywhere |
| **Continuous on domain** | $\ln x$ for $x > 0$, $\sqrt{x}$ for $x \geq 0$ | Continuous where they are defined |
| **Piecewise continuous** | Step functions, ReLU at $x = 0$ | Continuous except at specific points |

### Why Continuity Matters in ML

Most activation functions are chosen to be continuous (or at least piecewise continuous):

- **Sigmoid**, **tanh**: Continuous and smooth everywhere. Gradients always exist.
- **ReLU**: Continuous everywhere, but has a kink at $x = 0$ (not differentiable there). In practice, we just pick a subgradient and move on.
- **Step function**: Discontinuous. Gradient is zero almost everywhere, infinite at the jump. *Useless* for gradient descent.

The reason we care about continuity: **gradient descent requires the loss landscape to have no surprises.** If the loss function jumps discontinuously, the gradient gives you no useful information about which direction to step.

---

## Discontinuities: Where Things Break

When continuity fails, you have a **discontinuity**. There are three types, and understanding them helps you debug mathematical models just like understanding error types helps you debug code.

### ASCII Art: Three Types of Discontinuity

```
  REMOVABLE                  JUMP                     INFINITE
  (a hole you can fill)      (a step you can't fix)   (it blows up)

  f(x)                       f(x)                     f(x)
   ↑                          ↑                        ↑
   │                          │                        │      │
 3 ┤· · · ·○                3 ┤         ●━━━━━━━     3 ┤      │
   │      ╱                   │        ╱               │      │
 2 ┤    ╱                   2 ┤       │              2 ┤     ╱
   │  ╱                       │       │                │    ╱
 1 ┤╱                       1 ┤━━━━━━●              1 ┤  ╱
   │                          │                        │╱
   ┼──────┬──→ x              ┼───────┬──→ x           ┼──────┬──→ x
          a                           a                       a

  Limit exists (= 3)         Left limit = 1           Limit = ±∞
  f(a) undefined or ≠ 3      Right limit = 3          (does not exist
  FIX: define f(a) = 3       Cannot be fixed           as a finite number)
```

| Type | What Happens | Example | ML Analogy |
|------|-------------|---------|------------|
| **Removable** | Limit exists, but $f(a)$ is undefined or wrong | $\frac{x^2 - 1}{x - 1}$ at $x = 1$ | A `NaN` in your data that you can impute |
| **Jump** | Left and right limits exist but differ | Step functions, sign($x$) | A feature flag flipping behavior at a threshold |
| **Infinite** | Function blows up to $\pm\infty$ | $\frac{1}{x}$ at $x = 0$ | Division by zero in a loss function |

---

## Putting It All Together: A Continuity Checker

```python
import numpy as np

def check_continuity(f, a, epsilon=1e-6):
    """
    Check whether f is continuous at point a.

    Tests the three conditions:
    1. f(a) is defined
    2. lim_{x→a} f(x) exists (left and right limits agree)
    3. lim_{x→a} f(x) = f(a)

    Returns (is_continuous, diagnosis).
    """
    # Condition 1: Is f(a) defined?
    try:
        f_a = f(a)
        if np.isnan(f_a) or np.isinf(f_a):
            return False, f"f({a}) is undefined (NaN or Inf)"
    except Exception:
        return False, f"f({a}) is undefined (raised exception)"

    # Condition 2: Do left and right limits agree?
    left_limit  = f(a - epsilon)
    right_limit = f(a + epsilon)

    if not np.isclose(left_limit, right_limit, rtol=1e-3):
        return False, (
            f"Jump discontinuity: "
            f"left limit ≈ {left_limit:.6f}, right limit ≈ {right_limit:.6f}"
        )

    # Condition 3: Does the limit equal f(a)?
    limit_estimate = (left_limit + right_limit) / 2

    if not np.isclose(f_a, limit_estimate, rtol=1e-3):
        return False, (
            f"Removable discontinuity: "
            f"lim = {limit_estimate:.6f}, but f({a}) = {f_a:.6f}"
        )

    return True, f"Continuous at x = {a}: f({a}) = {f_a:.6f}"

# ── Test Cases ──

# 1. Continuous function
print(check_continuity(lambda x: x**2, 2))
# (True, 'Continuous at x = 2: f(2) = 4.000000')

# 2. Removable discontinuity: f(1) is defined but "wrong"
def f_removable(x):
    if x == 1:
        return 0           # "wrong" value; limit is 2
    return (x**2 - 1) / (x - 1)   # simplifies to x + 1

print(check_continuity(f_removable, 1))
# (False, 'Removable discontinuity: lim = 2.000000, but f(1) = 0.000000')

# 3. Jump discontinuity: sign function at 0
def sign(x):
    if x < 0: return -1.0
    if x > 0: return  1.0
    return 0.0

print(check_continuity(sign, 0))
# (False, 'Jump discontinuity: left limit ≈ -1.000000, right limit ≈ 1.000000')
```

---

## Limits at Infinity: What Happens in the Long Run

Some of the most important limits in ML involve $x \to \infty$ (or $n \to \infty$ for sequences).

$$\lim_{x \to \infty} f(x) = L$$

means $f(x)$ approaches $L$ as $x$ grows without bound.

### Examples That Matter for ML

**1. Sigmoid saturates:**

$$\lim_{x \to \infty} \frac{1}{1 + e^{-x}} = 1, \qquad \lim_{x \to -\infty} \frac{1}{1 + e^{-x}} = 0$$

This is why sigmoid neurons can "saturate" and stop learning -- the gradient approaches zero in both tails.

**2. Softmax temperature scaling:**

$$\lim_{T \to 0^+} \text{softmax}(z / T) = \text{one-hot}(\arg\max(z))$$

As temperature approaches zero, softmax approaches a hard argmax. As temperature approaches infinity, it approaches a uniform distribution.

**3. Training loss convergence (our running example):**

$$\lim_{n \to \infty} L_n = L^*$$

The entire premise of training is that this limit exists and is a useful minimum.

---

## Limits and the Derivative (Preview)

Here is why limits are not just theoretical niceties. The **derivative** -- the engine of all gradient-based optimization -- is defined as a limit:

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

This is the slope of the secant line as the two points merge into one. Without limits, you cannot make this precise. Without this being precise, you cannot trust gradient descent.

```python
def numerical_derivative(f, x, h=1e-7):
    """
    Compute f'(x) using the limit definition.

    We can't actually take h → 0 on a computer,
    so we pick a small h and hope for the best.
    (Automatic differentiation does this more cleverly.)
    """
    return (f(x + h) - f(x)) / h

# Derivative of x^2 at x = 3 should be 2*3 = 6
f = lambda x: x**2
print(f"f'(3) ≈ {numerical_derivative(f, 3):.8f}")   # 6.00000010
print(f"exact:   6.00000000")
```

The tiny error ($10^{-8}$) is because we used $h = 10^{-7}$ instead of the true limit $h \to 0$. Automatic differentiation avoids this by computing exact symbolic derivatives -- but the mathematical foundation is still this limit.

---

## When Limits Matter vs. When to Abstract Away

### When You Need to Think About Limits

- **Understanding derivative definitions**: The limit is *how* derivatives work.
- **Analyzing convergence**: Is your optimizer converging? That is a limit question.
- **Boundary behavior**: What happens to your loss function near parameter boundaries?
- **Proving algorithm correctness**: Convergence proofs are limit arguments.
- **Numerical stability**: Understanding when $\frac{0}{0}$ forms cause `NaN` in your code.

### When to Let the Framework Handle It

- Day-to-day training: PyTorch/JAX compute derivatives for you.
- Standard architectures: The math is already verified.
- Hyperparameter tuning: You are searching, not proving.

**The engineer's rule:** You do not need to compute limits by hand in production. But you need to *understand* them to debug convergence issues, choose appropriate learning rates, and reason about why your model is not learning.

---

## Common Pitfalls

| Pitfall | Why It Is Wrong | Correction |
|---------|----------------|------------|
| "The limit is just $f(a)$" | $f(a)$ might not exist, or it might differ from the limit | The limit describes the *approach*, not the value at the point |
| "Limits always exist" | Oscillating functions (like $\sin(1/x)$ near 0) may have no limit | Check both one-sided limits; they must agree |
| "If the limit is $\infty$, it exists" | Convention: a limit of $\pm\infty$ means the limit *does not exist* as a finite number | We say the function "diverges" or "grows without bound" |
| "Numerical estimation is always reliable" | Floating-point precision breaks down for very small $h$ | Use $h \approx 10^{-7}$ for float64; smaller is not better |

---

## Exercises

### Exercise 1: Numerical Limit Estimation

**Problem:** Estimate $\lim_{x \to 0} \frac{e^x - 1}{x}$ numerically.

**Solution:**

```python
import numpy as np

f = lambda x: (np.exp(x) - 1) / x

print("h              f(h)             |f(h) - 1|")
print("─" * 50)
for h in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
    val = f(h)
    print(f"{h:<15.5f}{val:<17.10f}{abs(val - 1):.2e}")

# h              f(h)             |f(h) - 1|
# ──────────────────────────────────────────────────
# 0.10000        1.0517091808     5.17e-02
# 0.01000        1.0050167084     5.00e-03
# 0.00100        1.0005001667     5.00e-04
# 0.00010        1.0000500017     5.00e-05
# 0.00001        1.0000050000     5.00e-06
#
# Converges to 1.0 ✓
```

### Exercise 2: Identifying Discontinuities

**Problem:** Classify the discontinuity of $f(x) = \frac{|x|}{x}$ at $x = 0$.

**Solution:** This is a **jump discontinuity**.

- From the left: $\lim_{x \to 0^-} \frac{|x|}{x} = \frac{-x}{x} = -1$
- From the right: $\lim_{x \to 0^+} \frac{|x|}{x} = \frac{x}{x} = 1$
- Left limit $\neq$ right limit, so the two-sided limit does not exist.

This is the sign function -- it jumps from $-1$ to $+1$ at the origin with no intermediate values.

### Exercise 3: Continuity Condition

**Problem:** For what value of $k$ is the following function continuous at $x = 2$?

$$f(x) = \begin{cases} x^2 & x < 2 \\ k & x = 2 \\ 4x - 4 & x > 2 \end{cases}$$

**Solution:**

For continuity at $x = 2$, we need $\lim_{x \to 2} f(x) = f(2) = k$.

- Left limit: $\lim_{x \to 2^-} x^2 = 4$
- Right limit: $\lim_{x \to 2^+} (4x - 4) = 4$
- Both one-sided limits equal 4, so $\lim_{x \to 2} f(x) = 4$
- Therefore $k = 4$.

### Exercise 4: Training Loss Convergence

**Problem:** You observe these training losses: $L_n = \frac{1}{n} + 0.5$. Does the training converge? If so, to what value? For $\varepsilon = 0.01$, find an epoch $N$ after which the loss stays within $\varepsilon$ of the limit.

**Solution:**

$$\lim_{n \to \infty} L_n = \lim_{n \to \infty} \left(\frac{1}{n} + 0.5\right) = 0 + 0.5 = 0.5$$

The training converges to a loss of $0.5$.

For the epsilon-delta condition: we need $|L_n - 0.5| < 0.01$, which means $\frac{1}{n} < 0.01$, so $n > 100$.

**After epoch 100, the loss is guaranteed to stay within 0.01 of the limit.** That is the epsilon-delta definition at work: you specified $\varepsilon = 0.01$, and we found $N = 100$.

```python
import numpy as np

L = lambda n: 1/n + 0.5
L_star = 0.5
epsilon = 0.01

for n in [10, 50, 100, 101, 500, 1000]:
    error = abs(L(n) - L_star)
    within = "✓" if error < epsilon else "✗"
    print(f"n = {n:>5}:  L_n = {L(n):.6f},  |L_n - L*| = {error:.6f}  {within}")

# n =    10:  L_n = 0.600000,  |L_n - L*| = 0.100000  ✗
# n =    50:  L_n = 0.520000,  |L_n - L*| = 0.020000  ✗
# n =   100:  L_n = 0.510000,  |L_n - L*| = 0.010000  ✗
# n =   101:  L_n = 0.509901,  |L_n - L*| = 0.009901  ✓
# n =   500:  L_n = 0.502000,  |L_n - L*| = 0.002000  ✓
# n =  1000:  L_n = 0.501000,  |L_n - L*| = 0.001000  ✓
```

---

## Summary

| Concept | What It Means | ML Connection |
|---------|--------------|---------------|
| **Limit** | What value $f(x)$ approaches as $x \to a$ | Training loss approaching a minimum: $\lim_{n \to \infty} L_n = L^*$ |
| **Epsilon-delta** | Rigorous contract: for any tolerance, a neighborhood exists | Convergence criteria with tolerance and patience |
| **One-sided limits** | Approaching from left vs. right | Behavior before vs. after a threshold |
| **Continuity** | Limit equals function value -- no surprises | Pure functions; smooth activation functions |
| **Discontinuity** | Removable, jump, or infinite | NaN values, feature flags, division by zero |
| **L'Hopital's rule** | Handling $\frac{0}{0}$ by differentiating top and bottom | Graceful edge-case handling |
| **Limits at infinity** | Long-run behavior | Sigmoid saturation, softmax temperature, convergence |

The sequence $L_1, L_2, \ldots, L_n \to L^*$ as $n \to \infty$ -- that is a limit. Every training run is a limit in action.

---

## What's Next

Limits answer "what are we approaching?" Derivatives answer the next question: **"how fast are we approaching it?"** That rate of change -- the slope of the loss curve at each epoch -- is exactly what gradient descent uses to decide which direction to step and how far. That is the foundation of gradient descent, and it is the subject of the next chapter.

---

*Next: Chapter 2 -- Derivatives: Measuring the Rate of Change*
