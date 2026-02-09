# Level 6: Calculus - The Math Behind Backpropagation

| [← Level 5: Analytic Geometry](/Users/toannguyen/Repo-Jeditech/book-mathematic/05-level-5-analytic-geometry/README.md) | [Level 7: Probability →](/Users/toannguyen/Repo-Jeditech/book-mathematic/07-level-7-probability/README.md) |
|---|---|

## Why You're Here

Every time you call `loss.backward()` in PyTorch, you're running calculus. Specifically, you're computing the gradient of your loss function with respect to every parameter in your network. That's the chain rule applied millions of times. Without calculus, there's no backprop. Without backprop, there's no practical way to train deep networks.

You've probably used gradient descent a hundred times. Maybe you've debugged vanishing gradients, tuned learning rates, or wondered why your loss surface has saddle points. This level explains the math behind all of that.

## What You'll Learn

### [Chapter 1: Limits](01-limits.md)
The foundation of calculus. Limits formalize what it means to "approach" a value without reaching it. You'll learn about epsilon-delta definitions, continuity, and why numerical stability depends on understanding limits. Think of limits like asymptotic analysis in algorithms—they describe behavior as you approach a boundary.

**Bridge for SWEs**: Limits are why `1e-8` shows up everywhere in ML code. They're the reason you add epsilon to denominators and why floating-point arithmetic breaks near zero.

### [Chapter 2: Derivatives](02-derivatives.md)
Rate of change. If you've ever run `git diff` to see what changed between commits, you understand derivatives—they measure change between states. You'll learn derivative rules, the chain rule, and how automatic differentiation works under the hood.

**Bridge for SWEs**: Derivatives = git diff for functions. The chain rule is literally backpropagation. Every `optimizer.step()` uses derivatives to update parameters.

### [Chapter 3: Gradients](03-gradients.md)
Derivatives in multiple dimensions. When your model has millions of parameters, you need the gradient vector (how the loss changes with respect to each parameter), the Jacobian (derivatives of vector-valued functions), and the Hessian (second derivatives, for understanding curvature).

**Bridge for SWEs**: Gradients tell you which direction is "downhill" in your loss landscape. They're the output of backprop and the input to every optimizer.

### [Chapter 4: Optimization](04-optimization.md)
Finding minima. You'll learn about local vs. global minima, saddle points, and critical points. This is where calculus meets ML training—understanding why gradient descent gets stuck, why initialization matters, and what second-order methods do differently.

**Bridge for SWEs**: Think binary search on a sorted array, but for continuous functions. Gradient descent uses derivatives to search for the minimum of your loss function.

### [Chapter 5: Integral Calculus](05-integral-calculus.md)
Accumulation and area under curves. Integrals reverse derivatives. They're essential for probability distributions (integrals must sum to 1), computing expectations, and understanding continuous random variables. You'll also learn numerical integration methods.

**Bridge for SWEs**: Integrals = accumulating values over a range, like summing log entries or computing area under a ROC curve. Monte Carlo integration is just sampling plus averaging.

## How It Connects

**Building On**: You need Level 4 (Linear Algebra) first. Gradients are vectors. Hessians are matrices. Backpropagation is the chain rule with matrix derivatives. Without vectors and matrices, you can't understand modern optimization.

**What Comes Next**:
- Level 7 (Probability) uses integrals everywhere—PDF normalization, expectations, cumulative distributions
- Level 9 (Optimization Theory) puts this calculus to work with SGD, Adam, momentum, and learning rate schedules
- Level 8 (Statistics) uses derivatives for maximum likelihood estimation and Fisher information

## The Calculus-to-ML Translation Table

| Calculus Concept | What It Actually Does in ML |
|------------------|----------------------------|
| Derivative | Tells you how much changing one weight affects your loss |
| Chain rule | Backpropagation—derivative flows backward through composed functions |
| Gradient vector | The direction of steepest increase in your loss (so you go the opposite way) |
| Jacobian matrix | Derivative of a vector function (like a layer's activations w.r.t. inputs) |
| Hessian matrix | Second derivatives—tells you about curvature (used in Newton's method) |
| Partial derivative | Derivative with respect to one parameter while holding others fixed |
| Critical point | Where gradient = 0 (could be minimum, maximum, or saddle point) |
| Integral | Accumulation—area under PDF, expected value, cumulative distribution |

## The Loss Landscape Mental Model

```
    Loss ↑
         |
         |    *              Local minimum (stuck here with GD)
         |   * *         *
         |  *   *   ?   * *  ← Saddle point (gradient = 0 but not minimum)
         | *     *     *   *
         |*       *   *     *
         |         * *       *___
         |          *         *  * ← Global minimum (we hope to find this)
         +----------------------------------→ Parameter space

    Gradient = slope at current position
    Optimizer = algorithm for walking downhill
```

When you train a neural network:
1. You start at a random point (initialization)
2. You compute the gradient (backprop)
3. You take a step in the negative gradient direction (optimizer update)
4. Repeat until gradient ≈ 0 or you run out of patience

Understanding calculus means understanding why this works, when it fails, and how to fix it.

## Prerequisites

You must be comfortable with:
- **Level 4: Linear Algebra** — vectors, matrices, dot products, matrix multiplication
- **Level 3: Functions** — composition, common function families
- **Level 2: Algebra** — manipulating expressions, solving equations

## Navigation

| Chapter | Topic | Key ML Connection |
|---------|-------|-------------------|
| [01-limits.md](01-limits.md) | Limits and continuity | Numerical stability, epsilon values |
| [02-derivatives.md](02-derivatives.md) | Derivatives and chain rule | Backpropagation, autograd |
| [03-gradients.md](03-gradients.md) | Multivariable derivatives | Gradient descent, parameter updates |
| [04-optimization.md](04-optimization.md) | Critical points and minima | Training convergence, saddle points |
| [05-integral-calculus.md](05-integral-calculus.md) | Integration and accumulation | Expected values, probability |

## Your Learning Approach

**Stop thinking abstractly**. Every time you see a derivative formula, mentally substitute `model.parameters()` and `loss_fn`. When you see an integral, think "summing over a continuous range."

**Implement as you learn**. NumPy's `np.gradient()` and `scipy.integrate.quad()` let you verify everything numerically. PyTorch's autograd shows you automatic differentiation in action.

**Connect to code you've written**. You've probably already used calculus without thinking about it. Now you'll understand what's actually happening when you call `.backward()`.

Let's dive in.

---

**Next**: [Chapter 1: Limits](01-limits.md) — The foundation that makes everything else work
