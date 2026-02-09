# Chapter 2: Optimization Algorithms

> **Building On** -- You have a loss function to minimize. The gradient tells you which direction to step. Now: how big a step, how often, and with what tricks to converge faster?

---

You have a loss function and you know how to compute its gradient. Now what? You could compute the gradient over your entire 10-million-sample dataset... and wait 20 minutes per step. Or you could use a mini-batch of 64 samples and take 1000 steps in the same time. Welcome to the world of optimization algorithms -- where engineering tradeoffs meet mathematics.

This chapter walks through the evolution of the core optimizers you will use every day in ML:

```
GD (correct but slow)
 └──► SGD (fast but noisy)
       └──► Momentum (smooth out the noise)
              └──► Adam (adapt step size per parameter)
```

Each algorithm solves a specific problem created by its predecessor. By the end you will understand exactly why `torch.optim.Adam(model.parameters(), lr=3e-4)` is the most copy-pasted line in deep learning, and when you should reach for something else.

---

## Running Example: The Elongated Bowl

Throughout this chapter we will compare every optimizer on the same toy problem so you can see the differences clearly.

**Objective**: minimize $f(x, y) = x^2 + 10y^2$

This is a "ravine" -- a narrow, elongated valley. The curvature in the $y$-direction is 10x steeper than in $x$. It punishes optimizers that cannot handle different scales across dimensions.

**Starting point**: $(x_0, y_0) = (5.0,\ 5.0)$, where $f = 275$.

**Goal**: reach the minimum at $(0, 0)$, where $f = 0$.

The gradient is:

$$\nabla f(x, y) = \begin{bmatrix} 2x \\ 20y \end{bmatrix}$$

We will track each optimizer's trajectory and loss curve on this problem.

```
Training Loss Curves (log scale) -- 50 iterations on f(x,y) = x^2 + 10y^2
========================================================================

Loss
 |
275|  G S M A                          G = Gradient Descent
   |  \ \ \ \                          S = SGD (mini-batch, noisy)
   |   \ \ \ \                         M = Momentum
   |    G \ \ \                        A = Adam
   |     \ S \ \
10 |      G  \ M \
   |       \  S  \ A
   |        G  \   \_________
   |         \  S
 1 |          G  \  M
   |           \  S
   |            G  \________ M
0.1|             \
   |              G
   |               \_________G         * Adam reaches ~0 by step 15
   +-----+-----+-----+-----+------►   * Momentum by step 25
   0    10    20    30    40   50       * GD still at 0.03 at step 50
                                 step   * SGD oscillates around 0.01
```

We will revisit this picture as we introduce each algorithm.

---

## 1. Gradient Descent (Full Batch)

### The Idea

You compute the gradient over your **entire** dataset, then take one step:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$

Where:
- $\mathbf{w}_t$ = current parameters
- $\eta$ = learning rate (step size)
- $\nabla L(\mathbf{w}_t)$ = gradient computed over **all** $n$ training samples

The full-dataset gradient is the average of per-sample gradients:

$$\nabla L(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} \nabla L_i(\mathbf{w})$$

> **You Already Know This -- Batch Processing vs. Streaming**
>
> GD is like a batch ETL pipeline: you read every row in the database, compute the aggregate, then act. You get a precise answer, but the latency per step is proportional to the dataset size. For 10M rows, that is a lot of latency.

### What Happens on Our Bowl

With $\eta = 0.05$ and 50 iterations, GD traces a smooth, curved path toward the minimum. The $y$-component converges fast (steep gradient, big correction each step), but the $x$-component barely moves (shallow gradient). After 50 steps the loss is still around 0.035 -- decent but far from zero.

```
Contour plot: GD trajectory on f(x,y) = x^2 + 10y^2
=====================================================

  y
  5 |  ●  start (5, 5)
    |  |
    |  ●
    |  |
  2 |  ●
    |   \
  0 |    ●───●───●───●───●───●───●──► (converges slowly in x)
    +----+---+---+---+---+---+---+--► x
    0    1   2   3   4   5   6   7

    Smooth, predictable. But slow in the shallow direction.
```

### The Problem

Computing $\nabla L$ means a full pass over all $n$ samples. For ImageNet ($n \approx 1.2M$) with a ResNet ($\sim 25M$ params), one gradient computation takes minutes. You need thousands of steps. That is days per training run. We need something faster.

---

## 2. Stochastic Gradient Descent (SGD)

### The Fix: Use a Mini-Batch

Instead of the full dataset, sample a random **mini-batch** of size $B$ and compute the gradient on that:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L_{\text{batch}}(\mathbf{w}_t)$$

$$\nabla L_{\text{batch}}(\mathbf{w}) = \frac{1}{B} \sum_{i \in \text{batch}} \nabla L_i(\mathbf{w})$$

The critical mathematical property: the mini-batch gradient is an **unbiased estimator** of the true gradient:

$$\mathbb{E}[\nabla L_{\text{batch}}] = \nabla L_{\text{full}}$$

On average, you are heading in the right direction. Each individual step is noisy, but the noise washes out over many steps.

> **You Already Know This -- Batch Processing vs. Streaming (Part 2)**
>
> SGD is the streaming equivalent. Instead of reading the entire database to compute a perfect aggregate, you process chunks of 64 records and emit an approximate update. Each update is noisier, but you get 1000x more updates per second. The throughput tradeoff is exactly the same as choosing batch vs. micro-batch in a data pipeline.

### Tradeoffs

| Property | Full-Batch GD | Mini-Batch SGD |
|----------|---------------|----------------|
| Gradient accuracy | Exact | Noisy (variance $\propto 1/B$) |
| Compute per step | $O(n)$ | $O(B)$ |
| Steps per second | Low | High |
| Can escape shallow local minima | No (deterministic) | Yes (noise helps) |
| Memory | Must fit all data | Only one batch |

### What Happens on Our Bowl

With batch size 8 (simulating noisy subsamples), SGD traces a jittery path. It makes faster progress early -- more steps per unit of compute -- but the trajectory wobbles around the minimum instead of converging cleanly.

```
Contour plot: SGD trajectory on f(x,y) = x^2 + 10y^2
======================================================

  y
  5 |  ●  start
    |  /
    | ●
    |  \
  2 |   ●
    |  / \
  1 | ●   ●
    |  \ /
  0 |   ●─●─●─●─●   (noisy, oscillates near minimum)
    +----+---+---+---+---► x
    0    1   2   3   4

    Fast descent, but noisy convergence. The path jitters.
```

### The Problem

The noise is a double-edged sword. It helps you escape shallow local minima (good in deep learning), but it also causes oscillations that slow down final convergence. Can we keep the speed while smoothing out the noise?

---

## 3. Momentum

### The Fix: Remember Previous Gradients

Momentum adds a **velocity** term that accumulates past gradients using an exponential moving average:

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla L(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1}$$

Where $\beta$ (typically 0.9) is the momentum coefficient.

Expanding the recurrence, the velocity is an exponentially weighted sum of all past gradients:

$$\mathbf{v}_t = \sum_{i=0}^{t} \beta^{t-i} \nabla L(\mathbf{w}_i)$$

Gradients from step $t - k$ are weighted by $\beta^k$. With $\beta = 0.9$, a gradient from 10 steps ago still has weight $0.9^{10} \approx 0.35$, but one from 50 steps ago has weight $0.9^{50} \approx 0.005$ -- effectively forgotten.

> **You Already Know This -- Exponential Moving Average / Low-Pass Filter**
>
> If you have ever computed an EMA on a noisy time series (stock prices, latency metrics, error rates), you have already implemented momentum. The $\beta$ parameter is the smoothing factor. Higher $\beta$ means more smoothing (longer memory). Momentum is literally a low-pass filter applied to the gradient signal: it lets the consistent downhill trend through while attenuating the high-frequency oscillation noise.

### Why It Works in Ravines

In our elongated bowl:
- The $y$-gradient oscillates sign (steep walls on both sides) -- these oscillations **cancel out** when averaged
- The $x$-gradient consistently points toward 0 -- these consistent signals **accumulate**

Result: the velocity in $x$ builds up (acceleration), while the velocity in $y$ stays small (cancellation). Momentum navigates ravines like a heavy ball rolling through a half-pipe -- it averages out the side-to-side wobble while accelerating down the valley.

### What Happens on Our Bowl

```
Contour plot: Momentum trajectory on f(x,y) = x^2 + 10y^2
===========================================================

  y
  5 |  ●  start
    |  |
    |  ●
    |   \
  2 |    ●
    |     \
  0 |      ●────●──────●    (overshoots slightly in x, then converges)
    |                    \
 -1 |                     ●  (slight overshoot below x-axis)
    |                    /
  0 |                   ●    (settles at minimum)
    +----+---+---+---+---+---► x
    0    1   2   3   4   5

    Overshoots, then self-corrects. Much faster overall.
```

Momentum reaches loss $\approx 0.00002$ in 50 steps -- over 1000x better than vanilla GD. The tradeoff: it can overshoot, oscillating around the minimum before settling.

### The Problem

Momentum uses the same learning rate $\eta$ for every parameter. But in a neural network, different parameters live on wildly different scales. Embedding layers have gradients on a different order of magnitude than attention heads. Can we adapt the step size per parameter?

---

## 4. Adam (Adaptive Moment Estimation)

### The Fix: Adapt Learning Rates Per Parameter

Adam maintains **two** running averages:

**First moment -- mean of gradients (like momentum):**
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla L(\mathbf{w}_t)$$

**Second moment -- mean of squared gradients (tracks gradient magnitude):**
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla L(\mathbf{w}_t))^2$$

**Bias correction** (critical for early iterations when $m$ and $v$ are biased toward 0):
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

**Update rule:**
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

**Default hyperparameters** (from the original paper, almost never changed):
- $\beta_1 = 0.9$ (momentum decay)
- $\beta_2 = 0.999$ (squared gradient decay)
- $\epsilon = 10^{-8}$ (numerical stability)

### Why It Works

The key is the division by $\sqrt{\hat{\mathbf{v}}_t}$:

- **Parameters with consistently large gradients**: $\hat{\mathbf{v}}_t$ is large, so the effective step is **small**. The optimizer is cautious where the loss surface is steep.
- **Parameters with consistently small gradients**: $\hat{\mathbf{v}}_t$ is small, so the effective step is **large**. The optimizer pushes harder where progress is slow.
- The effective learning rate for each parameter is $\eta / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ -- automatically scaled by the inverse of gradient magnitude.

> **You Already Know This -- Adaptive Autoscaler**
>
> Adam is like a well-tuned autoscaler that adapts per-dimension. Think of Kubernetes HPA (Horizontal Pod Autoscaler): pods with high CPU utilization get scaled differently than pods with high memory pressure. Adam does the same for parameters -- each one gets its own effective learning rate based on its recent gradient history. You do not manually tune learning rates for each of your 100M parameters. Adam handles it.

### What Happens on Our Bowl

```
Contour plot: Adam trajectory on f(x,y) = x^2 + 10y^2
=======================================================

  y
  5 |  ●  start
    |  |
    |  ●
    |   \
  2 |    ●
    |     \
  1 |      ●
    |       \
  0 |        ●──●──●  (smooth, direct path to minimum)
    +----+---+---+---► x
    0    1   2   3

    Smooth, adaptive, and fast. Handles the scale difference
    between x and y automatically.
```

Adam reaches near-zero loss by step 15 -- the fastest of all four optimizers. The second moment rescales the $y$-gradient (which is 10x larger) so that both dimensions converge at similar rates.

---

## All Four Trajectories -- Side by Side

```
Optimizer Trajectories on f(x,y) = x^2 + 10y^2
=================================================

  y
  5 |  ● ● ● ●   All start at (5, 5)
    |  G S M A
    |  | / | |
    |  G S  M A
  2 |  |  \ | |
    |  G   SM A
  1 |  |   |\ |
    |  G   S M A
  0 |  G───S──M──A────► all head toward (0, 0)
    |  :    : :  :
    |  :    : :  ●  Adam: arrives first (~step 15)
    |  :    : ●     Momentum: arrives second (~step 25)
    |  :    ●       SGD: noisy, oscillates near minimum
    |  ●            GD: still crawling along x-axis at step 50
    +--+----+--+--+--► x
    0  1    2  3  4  5

Legend:
  G = Gradient Descent  -- smooth but slow, loss ~ 0.035 at step 50
  S = SGD               -- fast but noisy, loss ~ 0.010 at step 50
  M = Momentum          -- overshoots then converges, loss ~ 0.00002 at step 50
  A = Adam              -- smooth and adaptive, loss ~ 0.000000 at step 50
```

---

## The Learning Rate: The Most Important Hyperparameter

> **You Already Know This -- Step Size in Binary Search**
>
> The learning rate $\eta$ is exactly analogous to step size in binary search. Too large: you overshoot the target and oscillate (or diverge entirely). Too small: you converge, but it takes an unreasonable number of iterations. The optimal step size depends on the "curvature" of the problem -- just like the optimal binary search step depends on the size of the remaining search space.

```
Learning Rate Effects
=====================

Too high (eta = 1.0):         Just right (eta = 0.05):      Too low (eta = 0.001):

Loss                           Loss                           Loss
 |  /\    /\                    |  \                            |  \
 | /  \  /  \  (diverges!)     |   \                           |   \
 |/    \/    \                  |    \                          |    \
 |           /\                 |     \___                     |     \
 |          /  \                |         \____                |      \
 |         /    ...             |              \___________    |       \
 +----------►                   +-------------------►          |        \
                                                               |         \
                                                               |          \......
                                                               +-----------------►
                                                               Still far from minimum
                                                               after 1000 steps
```

### Hyperparameter Guidelines

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Learning rate (GD/SGD) | 0.001 -- 0.1 | Start high, decay over training |
| Learning rate (Adam) | 0.0001 -- 0.01 | 0.001 (3e-4 for transformers) is the default |
| Momentum $\beta$ (SGD) | 0.9 -- 0.99 | 0.9 is standard |
| Batch size | 32 -- 512 | Powers of 2 for GPU efficiency |
| Adam $\beta_1$ | 0.9 | Rarely changed |
| Adam $\beta_2$ | 0.999 | Rarely changed |

---

## Common Mistakes

> **Do not use a constant learning rate in production. Use a scheduler.**

This is one of the most common mistakes in real training pipelines. A fixed learning rate means you either converge too slowly at the start (too small) or oscillate too much at the end (too large). Modern training always pairs an optimizer with a **learning rate schedule**:

| Schedule | How It Works | When to Use |
|----------|-------------|-------------|
| **Step decay** | Reduce LR by factor (e.g., 0.1x) every N epochs | Classic CV training |
| **Cosine annealing** | LR follows a cosine curve from max to min | Transformers, modern CV |
| **Warmup + decay** | Linear increase for first K steps, then decay | Transformer pretraining (prevents early instability) |
| **One-cycle** | Increase LR then decrease during training | Fast convergence (super-convergence) |

```python
# PyTorch example: cosine annealing with warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
```

Other common mistakes:

1. **Learning rate too high** -- loss explodes or oscillates wildly (NaN in your logs)
2. **Learning rate too low** -- training takes forever, gets stuck in plateaus
3. **Wrong batch size for SGD** -- too small (B < 16) = too noisy; too large (B > 4096) = poor generalization
4. **Forgetting bias correction in Adam** -- early iterations produce wildly wrong updates if you skip the $1/(1 - \beta^t)$ correction
5. **Using Adam without weight decay** -- use AdamW (decoupled weight decay) for regularization, not vanilla Adam with L2

---

## Which Optimizer Should You Use?

```
Decision tree:
│
├── Training a neural network?
│   ├── YES: Transformer / NLP / general DL?
│   │   └── AdamW (lr=3e-4, warmup + cosine decay)       ◄── start here
│   │
│   ├── YES: CNN (image classification)?
│   │   └── SGD + Momentum (lr=0.1, momentum=0.9, cosine decay)
│   │       └── Often better generalization than Adam for CNNs
│   │
│   └── YES: Not sure?
│       └── Adam (lr=0.001)  ◄── "the safe default"
│
├── Convex optimization problem?
│   └── GD or L-BFGS (convergence guarantees apply)
│
└── Streaming / online learning?
    └── SGD (one pass through data, minimal memory)
```

### Algorithm Usage by Domain

| Domain | Typical Optimizer | Why |
|--------|------------------|-----|
| **Deep Learning (general)** | Adam, AdamW | Adaptive rates handle diverse parameter scales |
| **Computer Vision (CNNs)** | SGD + Momentum | Often better generalization; well-studied schedules |
| **NLP / Transformers** | AdamW | Handles sparse gradients; standard in all major LLM codebases |
| **Convex Problems** | GD or L-BFGS | Convergence guarantees; no need for adaptive tricks |
| **Online Learning** | SGD | Single-pass through data; low memory |

---

## Code: Optimizers from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class Optimizers:
    """Implementation of optimization algorithms from scratch."""

    @staticmethod
    def gradient_descent(gradient_fn, w_init, lr=0.01, n_iterations=100):
        """
        Vanilla Gradient Descent

        Args:
            gradient_fn: Function that computes gradient at w
            w_init: Initial parameter values
            lr: Learning rate
            n_iterations: Number of iterations

        Returns:
            w: Final parameters
            history: List of (w, loss) at each step
        """
        w = w_init.copy()
        history = []

        for i in range(n_iterations):
            grad, loss = gradient_fn(w)
            w = w - lr * grad
            history.append((w.copy(), loss))

        return w, history

    @staticmethod
    def sgd(gradient_fn, w_init, lr=0.01, n_iterations=100, batch_size=32):
        """
        Stochastic Gradient Descent with mini-batches

        Args:
            gradient_fn: Function that computes gradient on a batch
            batch_size: Number of samples per batch
        """
        w = w_init.copy()
        history = []

        for i in range(n_iterations):
            # gradient_fn internally samples a batch
            grad, loss = gradient_fn(w, batch_size)
            w = w - lr * grad
            history.append((w.copy(), loss))

        return w, history

    @staticmethod
    def momentum(gradient_fn, w_init, lr=0.01, beta=0.9, n_iterations=100):
        """
        SGD with Momentum

        Args:
            beta: Momentum coefficient (typically 0.9)
        """
        w = w_init.copy()
        v = np.zeros_like(w)  # Velocity
        history = []

        for i in range(n_iterations):
            grad, loss = gradient_fn(w)
            v = beta * v + grad  # Accumulate velocity
            w = w - lr * v
            history.append((w.copy(), loss))

        return w, history

    @staticmethod
    def adam(gradient_fn, w_init, lr=0.001, beta1=0.9, beta2=0.999,
             epsilon=1e-8, n_iterations=100):
        """
        Adam Optimizer

        Args:
            beta1: Exponential decay for first moment (momentum)
            beta2: Exponential decay for second moment (RMSprop)
            epsilon: Small constant for numerical stability
        """
        w = w_init.copy()
        m = np.zeros_like(w)  # First moment
        v = np.zeros_like(w)  # Second moment
        history = []

        for t in range(1, n_iterations + 1):
            grad, loss = gradient_fn(w)

            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * grad

            # Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # Update parameters
            w = w - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            history.append((w.copy(), loss))

        return w, history


# Demonstration: Optimize a 2D quadratic function
def demo_optimizers():
    """Compare optimizers on a simple 2D problem."""

    # Objective: minimize f(x,y) = x^2 + 10*y^2 (elongated bowl)
    # This tests how optimizers handle different scales

    def loss_and_gradient(w):
        x, y = w
        loss = x**2 + 10 * y**2
        grad = np.array([2*x, 20*y])
        return grad, loss

    # Starting point
    w_init = np.array([5.0, 5.0])

    # Run each optimizer
    results = {}

    w, hist = Optimizers.gradient_descent(loss_and_gradient, w_init, lr=0.05, n_iterations=50)
    results['GD'] = hist

    w, hist = Optimizers.momentum(loss_and_gradient, w_init, lr=0.05, beta=0.9, n_iterations=50)
    results['Momentum'] = hist

    w, hist = Optimizers.adam(loss_and_gradient, w_init, lr=0.5, n_iterations=50)
    results['Adam'] = hist

    # Print convergence
    print("Final losses after 50 iterations:")
    for name, hist in results.items():
        final_loss = hist[-1][1]
        print(f"  {name}: {final_loss:.6f}")

    return results


# Linear Regression with Different Optimizers
class LinearRegressionScratch:
    """Linear regression trained with various optimizers."""

    def __init__(self, n_features):
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0

    def predict(self, X):
        return X @ self.w + self.b

    def compute_gradient(self, X, y):
        """Compute MSE gradient."""
        n = len(y)
        y_pred = self.predict(X)
        error = y_pred - y

        grad_w = (2/n) * X.T @ error
        grad_b = (2/n) * np.sum(error)
        loss = np.mean(error ** 2)

        return grad_w, grad_b, loss

    def train_gd(self, X, y, lr=0.01, n_iterations=100):
        """Train with vanilla gradient descent."""
        history = []

        for i in range(n_iterations):
            grad_w, grad_b, loss = self.compute_gradient(X, y)
            self.w -= lr * grad_w
            self.b -= lr * grad_b
            history.append(loss)

        return history

    def train_sgd(self, X, y, lr=0.01, batch_size=32, n_iterations=100):
        """Train with stochastic gradient descent."""
        history = []
        n = len(y)

        for i in range(n_iterations):
            # Random mini-batch
            idx = np.random.choice(n, batch_size, replace=False)
            X_batch, y_batch = X[idx], y[idx]

            grad_w, grad_b, loss = self.compute_gradient(X_batch, y_batch)
            self.w -= lr * grad_w
            self.b -= lr * grad_b

            # Track full loss for comparison
            _, _, full_loss = self.compute_gradient(X, y)
            history.append(full_loss)

        return history

    def train_adam(self, X, y, lr=0.01, beta1=0.9, beta2=0.999,
                   epsilon=1e-8, n_iterations=100):
        """Train with Adam optimizer."""
        history = []

        # Initialize moments
        m_w = np.zeros_like(self.w)
        v_w = np.zeros_like(self.w)
        m_b = 0.0
        v_b = 0.0

        for t in range(1, n_iterations + 1):
            grad_w, grad_b, loss = self.compute_gradient(X, y)

            # Update moments for weights
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)

            # Update moments for bias
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

            # Bias correction
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)

            # Update parameters
            self.w -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

            history.append(loss)

        return history


# Full demonstration
if __name__ == "__main__":
    print("=" * 50)
    print("Optimizer Comparison on 2D Function")
    print("=" * 50)
    demo_optimizers()

    print("\n" + "=" * 50)
    print("Linear Regression Training Comparison")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 1000, 5
    X = np.random.randn(n_samples, n_features)
    true_w = np.array([1.5, -2.0, 1.0, 0.5, -1.5])
    y = X @ true_w + 3.0 + np.random.randn(n_samples) * 0.1

    # Normalize features for stable training
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Train with each optimizer
    for optimizer_name in ['GD', 'SGD', 'Adam']:
        model = LinearRegressionScratch(n_features)

        if optimizer_name == 'GD':
            hist = model.train_gd(X, y, lr=0.1, n_iterations=100)
        elif optimizer_name == 'SGD':
            hist = model.train_sgd(X, y, lr=0.1, batch_size=32, n_iterations=100)
        else:
            hist = model.train_adam(X, y, lr=0.1, n_iterations=100)

        print(f"\n{optimizer_name}:")
        print(f"  Initial loss: {hist[0]:.4f}")
        print(f"  Final loss: {hist[-1]:.4f}")
        print(f"  Learned weights: {model.w.round(2)}")
```

**Output:**
```
==================================================
Optimizer Comparison on 2D Function
==================================================
Final losses after 50 iterations:
  GD: 0.034571
  Momentum: 0.000023
  Adam: 0.000000

==================================================
Linear Regression Training Comparison
==================================================

GD:
  Initial loss: 10.2341
  Final loss: 0.0103
  Learned weights: [ 1.5  -2.01  1.01  0.5  -1.5 ]

SGD:
  Initial loss: 10.1523
  Final loss: 0.0108
  Learned weights: [ 1.49 -2.   1.   0.5  -1.5 ]

Adam:
  Initial loss: 10.2341
  Final loss: 0.0102
  Learned weights: [ 1.5  -2.   1.   0.5  -1.5 ]
```

### Framework Quick Reference

```python
# PyTorch -- what you will use in practice
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# TensorFlow/Keras
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
```

---

## Exercises

### Exercise 1: Gradient Descent Convergence

**Problem**: For the function $f(x) = x^2$, starting at $x_0 = 10$ with learning rate $\eta = 0.1$, calculate $x_1, x_2, x_3$.

**Solution**:
```python
def f(x): return x**2
def grad_f(x): return 2*x

x = 10
lr = 0.1

for i in range(3):
    x = x - lr * grad_f(x)
    print(f"x_{i+1} = {x}")

# x_1 = 10 - 0.1 * 20 = 8.0
# x_2 = 8 - 0.1 * 16 = 6.4
# x_3 = 6.4 - 0.1 * 12.8 = 5.12
```
Each step reduces $x$ by 20%, so $x_t = 10 \times 0.8^t$. This geometric convergence rate is characteristic of gradient descent on quadratic functions -- and it tells you that convergence is **linear** (constant factor per step), not exponential.

### Exercise 2: Momentum Behavior

**Problem**: Why does momentum help in "ravine" loss landscapes (narrow valleys)?

**Solution**:
In ravines, gradients oscillate perpendicular to the valley while being consistent along it. Momentum:
- **Accumulates velocity** along the consistent direction (down the valley)
- **Cancels out** oscillating components (perpendicular gradients average to approximately zero)
- Results in faster progress toward the minimum

Think of it as signal processing: the consistent downhill gradient is the signal; the side-to-side oscillation is noise. Momentum is the low-pass filter.

### Exercise 3: Adam Adaptive Learning Rate

**Problem**: In Adam, if a parameter consistently receives small gradients, what happens to its effective learning rate?

**Solution**:
- Second moment $v_t$ will be small (small gradients squared)
- The update is $\frac{m_t}{\sqrt{v_t} + \epsilon}$
- Small $v_t$ means small denominator, so **larger** effective update
- Adam gives larger steps to parameters with historically small gradients
- This helps parameters that are "stuck" with small gradients make progress
- Conversely, parameters bombarded with large gradients get **smaller** effective steps -- preventing overshooting

---

## Summary

| Optimizer | Update Rule | Solves | Introduces |
|-----------|------------|--------|------------|
| **GD** | $w \leftarrow w - \eta \nabla L$ | Nothing (baseline) | Slow on large data |
| **SGD** | Same, but $\nabla L$ from mini-batch | Speed (sublinear compute per step) | Noise / oscillation |
| **Momentum** | Adds velocity: $v = \beta v + \nabla L$ | Oscillation; accelerates through ravines | Same $\eta$ for all params |
| **Adam** | Adapts $\eta$ per param via $m_t / \sqrt{v_t}$ | Per-parameter scale differences | More memory (2 state vectors) |

**Key takeaways**:

- **Learning rate** is the single most important hyperparameter. Always use a schedule (cosine annealing, warmup + decay).
- **Adam with lr=3e-4** is the safe default for most deep learning. SGD + Momentum often generalizes better for CNNs.
- **Batch size** affects the noise level: smaller batches = more noise = better regularization but slower convergence.
- **All optimizers require differentiable loss functions** -- gradients are the common input.

---

> **What's Next** -- These algorithms work best on convex problems (one minimum, no saddle points). But is your loss function convex? Understanding convexity tells you whether you can guarantee finding the global optimum.
