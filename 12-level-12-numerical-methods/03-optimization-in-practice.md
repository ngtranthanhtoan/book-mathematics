# Optimization in Practice

> **Building On**: In the previous chapter on approximation methods, we built the mathematical machinery for iterative convergence — fixed-point iteration, Newton's method, and convergence guarantees. We know *how* to take steps toward a solution. Now we face the engineering reality: how do you make optimization actually work on real hardware, with real data, at real scale? This is where theory meets production, and where most training runs go to die.

---

## The 3 AM PagerDuty Call Nobody Warned You About

It's Thursday night. Your team just kicked off a training run for a transformer model — 8 GPUs, estimated 72 hours, somewhere north of $15,000 in cloud compute. You check the loss curve before bed. Looks great: dropping smoothly, right on schedule.

At 3 AM, your monitoring fires. You pull up the dashboard and see this:

```
Loss Curve — Your $15,000 Training Run

  Loss
  10 |*
     | *
   8 |  *
     |   *
   6 |    **
     |      ***
   4 |         ****
     |             *****
   2 |                  ********
     |                          *****              ___---*
   1 |                               *****    ___--
     |                                    *--*   NaN!
   0 +----+----+----+----+----+----+----+----+----+----->
     0   10k  20k  30k  40k  50k  60k  70k  80k  90k
                         Steps

     ^                                    ^
     Looked great at bedtime              3 AM: loss
                                          explodes
```

The loss was decreasing beautifully for 70,000 steps. Then it spiked to NaN. $12,000 worth of compute, gone. The model weights are trash.

What happened? Let me save you the 2 hours of debugging: a single batch contained an unusual data point that produced a gradient 500x larger than normal. That gradient blew up the weight update, which produced even larger gradients on the next step, and within 50 iterations your parameters were at infinity.

This chapter is about the engineering tools that prevent exactly this scenario — and a dozen others like it. Learning rate schedules, gradient clipping, warmup strategies, mixed-precision training, and gradient accumulation. These aren't theoretical niceties. They're the difference between a successful training run and an expensive pile of NaN values.

---

## Part 1: Learning Rate — The One Hyperparameter That Rules Them All

### The Problem You Already Understand

If you've ever configured autoscaling for a web service, you already understand the core tension of learning rate scheduling.

> **You Already Know This**: Think about how you configure autoscaling policies. Scale too aggressively (large step size) and you get oscillation — your service keeps overshooting between 2 and 20 instances. Scale too conservatively (tiny step size) and you can't respond to traffic spikes fast enough. The sweet spot usually involves starting cautious during uncertain conditions, then ramping up as the system stabilizes, then backing off as you approach steady state. That's exactly what learning rate warmup + cosine annealing does for model training.

Here's what different learning rates actually look like on a loss landscape. Not a box diagram — the actual path optimization takes:

```
Learning Rate Effects on a 2D Loss Landscape
(Contour lines show equal-loss regions, * shows optimization path)

  Too Small (lr=0.0001)          Just Right (lr=0.01)           Too Large (lr=0.5)

  ╭─────────────────╮          ╭─────────────────╮          ╭─────────────────╮
  │  ╭───────────╮  │          │  ╭───────────╮  │          │  ╭───────────╮  │
  │  │ ╭───────╮ │  │          │  │ ╭───────╮ │  │          │  │ ╭───────╮ │  │
  │  │ │╭─────╮│ │  │          │  │ │╭─────╮│ │  │          │  │ │╭─────╮│ │  │
  │  │ ││ ╭─╮ ││ │  │          │  │ ││ ╭─╮ ││ │  │          │  │ ││ ╭─╮ ││ │  │
  │* │ ││ │.│ ││ │  │          │  │ ││ │.│ ││ │  │          │  │*││ │.│ ││*│  │
  │ *│ ││ ╰─╯ ││ │  │          │ *│ ││ ╰─╯ ││ │  │          │  │ ││ ╰─╯ ││ │  │
  │  *─│╰─────╯│ │  │          │  * │╰──*──╯│ │  │          │  │ │╰─────╯│*│  │
  │  │ *───────╯ │  │          │  │ ╰──*────╯ │  │          │ *│ ╰───────╯ │  │
  │  │  *────────╯  │          │  │   *───────╯  │          │  ╰──────*────╯  │
  │  ╰──*──────────╯           │  ╰──*──────────╯          │ *───────────╯*   │
  ╰─────*───────────           ╰────*───────────╯          ╰──*──────────*────╯

  Still far from               Converges                    Bouncing wildly.
  minimum after                efficiently to               Never converges.
  1000 steps.                  minimum.                     Might diverge!
```

### Why a Constant Learning Rate Almost Never Works

Let's see this concretely. Here's a simplified training loop that exposes the problem:

```python
import torch
import torch.nn as nn

# Simple model, simple data — still breaks with wrong LR
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Seems reasonable?
loss_fn = nn.MSELoss()

# Synthetic data
X = torch.randn(1000, 10)
y = X @ torch.randn(10, 1) + 0.1 * torch.randn(1000, 1)

losses = []
for epoch in range(100):
    pred = model(X)
    loss = loss_fn(pred, y)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot what happened
print(f"Epoch 0:  loss = {losses[0]:.4f}")
print(f"Epoch 10: loss = {losses[10]:.4f}")
print(f"Epoch 50: loss = {losses[50]:.4f}")
print(f"Epoch 99: loss = {losses[-1]:.4f}")
# You'll often see: loss drops fast, then oscillates, never fully converges
```

The loss drops quickly at first but then oscillates around the minimum. A learning rate of 0.1 was too big for the fine-grained adjustments needed near convergence, but it was great for the early phase when parameters were far from optimal.

**Translation**: You need different step sizes at different points in training. That's what learning rate schedules are.

### The Math Behind the Schedules

Each schedule is a function $\eta(t)$ that maps the current training step $t$ to a learning rate. Here they are, with what they look like:

```
Learning Rate Schedules Over Training Steps

  LR
 0.10 |--*--*--*--*--*--*--                    ..........
      |                    -------*--*--*--*  .          .
 0.08 |                                     .        Cyclic
      |   **                                .    *       .
 0.06 |  *  **              *               . * / \      .
      | *     **           * *              .* /   \     .
 0.04 |*        **        *   *             */     \  * .
      |           **     *     *           *       \ / \.
 0.02 |             **  *       **        /         *
      |               **         ***     /
 0.01 |                            *****/
      |
    0 +----+----+----+----+----+----+----+----+----+---->
      0   10   20   30   40   50   60   70   80   90 100
                           Steps

      ---- Constant    ---- Step Decay    ---- Cosine
      ---- Warmup+Cosine                  .... Cyclic
```

#### Step Decay

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}$$

where $\eta_0$ is the initial learning rate, $\gamma$ is the decay factor (typically 0.1), and $s$ is the number of steps per decay.

**Translation**: "Every $s$ steps, multiply the learning rate by $\gamma$." If you start at 0.1 with $\gamma = 0.1$ and $s = 30$, your LR is 0.1 for the first 30 epochs, then drops to 0.01, then 0.001. It's the staircase function of learning rates.

**When to use it**: ResNet training famously uses step decay at epochs 30, 60, and 90. It's simple, well-understood, and a solid baseline. If you don't know what schedule to use, start here.

#### Exponential Decay

$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

or equivalently:

$$\eta_t = \eta_0 \cdot \gamma^t$$

**Translation**: "Shrink the learning rate by a constant fraction every single step." Unlike step decay's staircase, this is a smooth ramp down. With $\gamma = 0.99$, you lose 1% of your learning rate at each step.

#### Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

**Translation**: The learning rate follows half a cosine wave from $\eta_{\max}$ down to $\eta_{\min}$ over $T$ steps. It starts high, decays slowly at first, then accelerates the decay, and gently lands at the minimum. Think of it as the "ease-in, ease-out" animation curve of learning rates.

This is the current default for most vision models and many LLM training runs. Why cosine specifically? Because it spends more time at moderate learning rates (where useful learning happens) and less time at very high or very low rates.

#### Linear Warmup

For the first $T_w$ steps:

$$\eta_t = \eta_{\text{target}} \cdot \frac{t}{T_w}$$

After warmup, transition to any other schedule.

**Translation**: "Start with a tiny learning rate and linearly ramp up to the target." This gives the optimizer time to estimate good gradient statistics (for Adam, this means the running mean and variance) before taking large steps.

> **Common Mistake**: Skipping warmup with Adam/AdamW. Adam relies on running estimates of gradient moments. In the first few hundred steps, these estimates are wildly inaccurate (they're initialized at zero). A large learning rate multiplied by noisy moment estimates = unstable early training. Warmup is not optional for transformers.

#### Cyclical Learning Rates

$$\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \text{triangular}(t)$$

**Translation**: The learning rate bounces between a minimum and maximum value in a repeating pattern. The periodic large learning rate helps the optimizer escape sharp local minima and explore the loss landscape more broadly.

#### Cosine Annealing with Warm Restarts (SGDR)

This combines the smoothness of cosine annealing with the exploration of cyclical schedules:

```
Cosine Annealing with Warm Restarts

  LR
 0.10 |*                  *                           *
      | *                 |*                          |*
 0.08 |  *                | *                         | *
      |   *               |  *                        |  *
 0.06 |    **             |   **                      |   **
      |      **           |     **                    |     *
 0.04 |        **         |       ***                 |
      |          ***      |          ****             |
 0.02 |             ***   |              *****        |
      |                ** |                   ******  |
 0.01 |                  *|                        ***|
      +----+----+----+----+----+----+----+----+----+----+-->
      0        10        20        30        40       50

      |<--- T_0=10 --->| |<------- T_0*2=20 ------->|
        1st cycle              2nd cycle (longer)
```

Each restart gives the optimizer a fresh chance to explore, and each subsequent cycle runs longer, allowing for finer convergence.

### The One-Cycle Policy: Super-Convergence

The one-cycle policy (from Leslie Smith's research) is one of the most practical scheduling tricks. It can achieve the same final loss in 5-10x fewer steps than a constant learning rate:

1. **Warmup phase** (first ~30% of training): Linearly increase LR from $\frac{\eta_{\max}}{25}$ to $\eta_{\max}$
2. **Annealing phase** (remaining ~70%): Cosine decrease from $\eta_{\max}$ to $\frac{\eta_{\max}}{1000}$

```
One-Cycle Policy

  LR
 0.10 |              *****
      |          ****     ****
 0.08 |        **             ***
      |      **                  ***
 0.06 |    **                       **
      |   *                           ***
 0.04 | **                               ***
      |*                                    ****
 0.02 |                                         ****
      |                                             ****
 0.00 +----+----+----+----+----+----+----+----+----+---->
      0   10   20   30   40   50   60   70   80   90 100

      |<- warmup 30% ->|<----- annealing 70% --------->|
```

> **You Already Know This**: The one-cycle policy is like a deployment strategy for feature flags. You roll out slowly (warmup) to detect issues early, ramp up to full traffic once you're confident, then gradually reduce to steady state. Same logic — manage risk by controlling the rate of change.

### The Learning Rate Range Test: Finding Your LR

Before choosing a schedule, you need to know what learning rate is even in the right ballpark. The LR range test automates this:

1. Start with a tiny LR (e.g., $10^{-7}$)
2. Train for a few hundred steps, exponentially increasing the LR each step
3. Plot loss vs. LR
4. Pick the LR where loss is decreasing fastest (steepest negative slope)

```
Learning Rate Range Test

  Loss
  10 |***
     |   **
   8 |     **
     |       *
   6 |        *
     |         **
   4 |           **
     |             ***
   2 |                ****             ___---**** NaN
     |                    *****  ___---
   1 |                         **
     |
     +----+----+----+----+----+----+----+----+----+----->
    1e-7  1e-6  1e-5  1e-4  1e-3  1e-2  1e-1   1   10
                     Learning Rate (log scale)

                               ^
                          Pick this region:
                     steepest descent, before
                        loss starts rising
```

**Translation**: The optimal learning rate is usually about 10x smaller than the rate where the loss starts increasing. If the test shows loss starts climbing around 0.1, try 0.01 as your peak learning rate.

Here's how to run this in PyTorch:

```python
import torch
import torch.nn as nn
import math

def lr_range_test(model, train_loader, loss_fn,
                  min_lr=1e-7, max_lr=10, num_steps=200):
    """
    Leslie Smith's learning rate range test.
    Returns lists of learning rates and corresponding losses.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=min_lr)
    lr_mult = (max_lr / min_lr) ** (1 / num_steps)

    lrs, losses = [], []
    best_loss = float('inf')
    lr = min_lr

    model.train()
    data_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Stop if loss explodes (4x best loss is a good heuristic)
        if loss.item() > 4 * best_loss and step > 10:
            break

        best_loss = min(best_loss, loss.item())
        lrs.append(lr)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        # Exponentially increase LR
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lrs, losses

# Usage:
# lrs, losses = lr_range_test(model, train_loader, nn.CrossEntropyLoss())
# Plot lrs (log scale) vs losses to find optimal LR
```

### PyTorch's Built-In Schedulers: The Production Toolkit

You rarely need to implement these from scratch. PyTorch provides battle-tested implementations:

```python
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
    LinearLR,
    SequentialLR,
)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Step decay: multiply by 0.1 every 30 epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing over 100 epochs
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Cosine with warm restarts: first cycle 10 epochs, then double
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# One-cycle policy (great for fast training)
scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=10000)

# Reduce on plateau (adaptive — watches your validation loss)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)

# Warmup + cosine (the transformer standard)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=1000)
cosine = CosineAnnealingLR(optimizer, T_max=9000, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[1000])

# Training loop
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer)

    val_loss = evaluate(model, val_loader)

    # For ReduceLROnPlateau, pass the metric:
    # scheduler.step(val_loss)

    # For all other schedulers:
    scheduler.step()

    print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.6f}")
```

> **Common Mistake**: Forgetting to call `scheduler.step()`. Your learning rate never changes, you wonder why training plateaus, and you spend an hour checking your data pipeline before finding the one-line bug. Ask me how I know.

> **Common Mistake**: Calling `scheduler.step()` before `optimizer.step()`. Some older PyTorch tutorials show this order. Since PyTorch 1.1+, the scheduler should be called AFTER the optimizer. Getting it backwards means your first step uses the wrong LR.

---

## Part 2: Gradient Clipping — The Circuit Breaker for Training

### The Failure That Motivates Everything

Remember that 3 AM PagerDuty scenario? Let's dig into exactly what happened.

In a deep network, gradients are computed via backpropagation — which is just the chain rule applied recursively through layers. For a network with $L$ layers, the gradient with respect to an early layer involves multiplying through all subsequent layers:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_L} \cdot \frac{\partial h_L}{\partial h_{L-1}} \cdot \frac{\partial h_{L-1}}{\partial h_{L-2}} \cdots \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}$$

Each $\frac{\partial h_{l+1}}{\partial h_l}$ involves the weight matrix at that layer. If those weight matrices have spectral norm (largest singular value) greater than 1, then this product of many terms **grows exponentially**.

```
Gradient Magnitude Through a 50-Layer Network

  Gradient
  Norm      Without Clipping          With Clipping
            (weight scale = 1.1)      (max_norm = 10)

  1e15 |                    *
       |                   *
  1e12 |                  *
       |                *
  1e9  |              *
       |            *
  1e6  |          *
       |        *
  1e3  |      *                         __________10.0
       |    *                     _____/
  1e0  | **               _____/
       |*           _____/
  1e-3 +----+----+----+----+----+----+----+----+----+----->
       1    5   10   15   20   25   30   35   40   45  50
                          Layer (depth)
```

With a weight scale of just 1.1 (barely above 1), gradients grow by $1.1^{50} \approx 117$ over 50 layers. At 1.5, that becomes $1.5^{50} \approx 6.4 \times 10^8$. At 2.0, it's $2^{50} \approx 10^{15}$. That's not a gradient — that's a bomb.

> **You Already Know This**: This is exactly a circuit breaker pattern. In microservice architectures, one failing downstream service can cascade failures through the entire call chain. The gradient explosion is a cascading failure through the computational graph. Gradient clipping is the circuit breaker: it detects when the "current" (gradient magnitude) exceeds a threshold and caps it before it fries the "circuit" (your model weights).

### Three Clipping Strategies

#### Strategy 1: Clip by Value

$$g_i^{\text{clipped}} = \max(\min(g_i, \theta), -\theta)$$

Each gradient component is clipped independently to $[-\theta, \theta]$.

```
Clip by Value: Each component clamped independently

  Before:  g = [100, -200, 150]     threshold = 50
  After:   g = [ 50,  -50,  50]

  Component view:
        -200    -50    0     50    100   150   200
   g[0]:  |      |     |     |===>==*     |     |   100 -> 50
   g[1]:  *=====>|     |     |      |     |     |  -200 -> -50
   g[2]:  |      |     |     |======|==>==*     |   150 -> 50

  Problem: original direction was [100, -200, 150]
           clipped direction is   [ 50,  -50,  50]
           These point in DIFFERENT directions!
```

**The problem**: Clip by value changes the gradient's direction. The original gradient pointed mostly in the $-y$ direction (component -200 dominates). After clipping, all components are the same magnitude, so the direction has changed significantly. In optimization terms, you're now walking in the wrong direction.

#### Strategy 2: Clip by Norm (The Standard)

$$g^{\text{clipped}} = g \cdot \min\left(1, \frac{\theta}{\|g\|}\right)$$

If $\|g\| > \theta$, scale the entire gradient vector down uniformly:

$$g^{\text{clipped}} = \frac{\theta}{\|g\|} \cdot g$$

```
Clip by Norm: Scale entire vector, preserve direction

  Before:  g = [100, -200, 150]     ||g|| = 269.3     max_norm = 50
  After:   g = [18.6, -37.1, 27.9]  ||g|| = 50.0

  Vector view (2D projection):

            |
       150 -|          * original g
            |         /
       100 -|        /
            |       /
        50 -|      /
            |   * / clipped g
            |   /  (same direction,
            |  /    shorter)
            | /
   ---------+------------>
            |

  Direction preserved! Only magnitude reduced.
```

**Translation**: Clip by norm asks "is this gradient too big?" and if so, shrinks the whole thing proportionally — like turning down the volume instead of muting individual speakers.

This is what you should use 99% of the time. It's the default in PyTorch's `clip_grad_norm_`.

#### Strategy 3: Adaptive Gradient Clipping (AGC)

From the NFNet paper (Brock et al., 2021):

$$g^{\text{clipped}} = g \cdot \min\left(1, \frac{\lambda \|W\|}{\|g\|}\right)$$

**Translation**: Instead of a fixed threshold, AGC clips based on the ratio of gradient norm to weight norm. The insight is that what constitutes a "large" gradient depends on how large the parameters are. A gradient of 10 is enormous for a weight of 0.001 but negligible for a weight of 1000.

This is particularly useful for training without batch normalization, which was the whole point of the NFNet architecture.

### Gradient Clipping in Practice

```python
import torch
import torch.nn as nn

# === The standard approach: clip by global norm ===
model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in train_loader:
    optimizer.zero_grad()
    output = model(batch.src, batch.tgt)
    loss = loss_fn(output, batch.labels)
    loss.backward()

    # Clip gradients AFTER backward, BEFORE optimizer step
    # Returns the total norm before clipping (useful for monitoring)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0  # Standard for transformers
    )

    # Log this! It's your early warning system
    if grad_norm > 10.0:
        print(f"Warning: grad_norm = {grad_norm:.1f} (clipped to 1.0)")

    optimizer.step()


# === Clip by value (rare, but available) ===
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)


# === Manual AGC implementation ===
def adaptive_gradient_clipping(model, clip_factor=0.01, eps=1e-3):
    """
    Adaptive Gradient Clipping from NFNet.
    Clips gradient based on weight-to-gradient norm ratio.
    """
    for param in model.parameters():
        if param.grad is None:
            continue

        param_norm = param.data.norm(2)
        grad_norm = param.grad.data.norm(2)

        max_norm = param_norm * clip_factor

        # Clamp param_norm to avoid division issues with very small params
        max_norm = torch.clamp(max_norm, min=eps)

        if grad_norm > max_norm:
            param.grad.data.mul_(max_norm / grad_norm)
```

> **Common Mistake**: Clipping gradients *after* `optimizer.step()`. At that point the gradients have already been applied — clipping does nothing. The correct order is always: `loss.backward()` -> `clip_grad_norm_()` -> `optimizer.step()`.

> **Common Mistake**: Clipping before gradient accumulation is complete (more on this later). If you're accumulating gradients across multiple mini-batches, clip once after all accumulation steps, not after each one. Clipping after each micro-batch changes the effective gradient direction.

### Monitoring Gradient Norms: Your Production Dashboard

In production training, you should log gradient norms every step. They're your seismograph:

```python
# Production-grade gradient monitoring
def training_step(model, batch, optimizer, scheduler, max_norm=1.0):
    optimizer.zero_grad()

    output = model(batch)
    loss = loss_fn(output, batch.labels)
    loss.backward()

    # Compute gradient norm BEFORE clipping
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    optimizer.step()
    scheduler.step()

    # Return metrics for logging
    return {
        'loss': loss.item(),
        'grad_norm': total_norm.item(),
        'lr': optimizer.param_groups[0]['lr'],
        'clipped': total_norm.item() > max_norm,
    }

# What to watch for in your logs:
#
# grad_norm pattern         | Diagnosis
# ========================= | ===================================
# Steady, moderate (1-10)   | Healthy training
# Gradually decreasing      | Converging, might need LR increase
# Sudden spike (100x)       | Bad batch or bug in data pipeline
# Consistently at max_norm  | Clipping too aggressively, or LR too high
# Increasing over time      | Approaching instability, reduce LR
# NaN or Inf                | Already exploded. Check loss scaling.
```

---

## Part 3: Batch Size, Gradient Accumulation, and the Memory Wall

### The Engineering Constraint You Face Daily

You have a model that needs a batch size of 2048 for stable training (the paper says so). You have a single GPU with 24GB VRAM. A single sample takes 800MB through the forward pass. You can fit... about 24 samples per batch. Not 2048.

> **You Already Know This**: This is the same problem as processing a 100GB file when you have 16GB of RAM. You don't try to load it all at once. You process it in chunks and accumulate the results. In ETL pipelines, you call this "chunked processing" or "micro-batching." In ML, we call it "gradient accumulation," but it's the exact same pattern.

### Gradient Accumulation: Chunked Processing for Gradients

The math is simple. For a loss averaged over a batch of size $B$:

$$L_{\text{batch}} = \frac{1}{B}\sum_{i=1}^{B} L(x_i, y_i)$$

The gradient of this loss is:

$$\nabla L_{\text{batch}} = \frac{1}{B}\sum_{i=1}^{B} \nabla L(x_i, y_i)$$

Since summation is associative, we can split the batch into $K$ micro-batches of size $b = B/K$ and accumulate:

$$\nabla L_{\text{batch}} = \frac{1}{K}\sum_{k=1}^{K}\left(\frac{1}{b}\sum_{i \in \text{micro-batch}_k} \nabla L(x_i, y_i)\right)$$

**Translation**: Compute gradients on small batches, add them up, and only update the weights after you've accumulated enough. Mathematically identical to using the large batch, but fits in memory.

```python
# Gradient accumulation in PyTorch

accumulation_steps = 4  # Effective batch = micro_batch_size * 4
optimizer.zero_grad()

for i, (inputs, targets) in enumerate(train_loader):
    # Forward + backward on micro-batch
    outputs = model(inputs)
    loss = loss_fn(outputs, targets) / accumulation_steps  # Scale loss!
    loss.backward()  # Gradients accumulate in .grad attributes

    if (i + 1) % accumulation_steps == 0:
        # Now we have gradients equivalent to the full batch
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()  # Reset for next accumulation cycle
```

> **Common Mistake**: Forgetting to divide the loss by `accumulation_steps`. Without this, your effective learning rate is multiplied by `accumulation_steps`, and you're now training with a learning rate 4x larger than intended. This is one of the most common bugs in distributed training code.

### Batch Size Effects on Training Dynamics

Batch size is not just a memory constraint — it fundamentally changes the optimization dynamics:

```
Effect of Batch Size on Gradient Noise

  Small Batch (32)                    Large Batch (2048)
  Noisy but explores                  Smooth but may overfit

  Loss                                Loss
   |  *                                |  *
   | * *                               |  **
   |*   * *                            |    **
   |     * * *                         |      **
   |      * * *  *                     |        ***
   |           * * *                   |           ****
   |            * * * *                |               *****
   |                * * * *            |                    ********
   |                  *   *  * *  *    |                            **********
   +----------------------------->     +------------------------------->
                Steps                               Steps

  Pro: natural regularization          Pro: faster per-step progress
  Pro: can escape local minima         Pro: better GPU utilization
  Con: more steps to converge          Con: needs LR scaling
  Con: noisier loss curves             Con: may converge to sharp minima
```

There's an important empirical rule for batch size and learning rate:

**Linear Scaling Rule**: When you multiply batch size by $k$, multiply the learning rate by $k$ too.

$$\eta_{\text{large}} = \eta_{\text{base}} \cdot \frac{B_{\text{large}}}{B_{\text{base}}}$$

This comes from the observation that a larger batch gives you a better estimate of the true gradient, so you can afford to take larger steps. But this rule breaks down for very large batch sizes, which is why warmup is critical when training with large batches.

```python
# Linear scaling rule with warmup for large batch training
base_lr = 0.1
base_batch_size = 256
actual_batch_size = 2048  # 8 GPUs x 256

# Scale learning rate
scaled_lr = base_lr * (actual_batch_size / base_batch_size)  # 0.8

# Use warmup to stabilize the larger learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90),
    ],
    milestones=[5],
)
```

---

## Part 4: Mixed Precision Training — When 16 Bits is Enough (Mostly)

### Why Throw Away Precision?

FP32 (32-bit floating point) gives you about 7 decimal digits of precision. FP16 (16-bit) gives you about 3-4. Why would you voluntarily halve your precision?

Because FP16 operations are **2-8x faster** on modern GPUs and use **half the memory**. That means you can either train 2x faster or fit a model 2x larger in the same VRAM. In the world of LLMs where GPU hours cost real money, this is not a nice-to-have — it's table stakes.

> **You Already Know This**: You already make precision tradeoffs in production software. You store timestamps as 64-bit integers even though you could use 128-bit for nanosecond precision. You use lossy JPEG for thumbnails and lossless PNG only where it matters. You compress logs to save storage, accepting that some are more expensive to read back. Mixed-precision training applies exactly this principle: use low precision where you can, high precision where you must.

### The Problem with Naive FP16

You can't just cast everything to FP16 and call it a day. Here's why:

```
FP32 vs FP16 Number Ranges

FP32:  ────|══════════════════════════════════════|────
       -3.4e38                                3.4e38
       smallest positive normal: 1.2e-38

FP16:  ──|════════════|──
       -65504      65504
       smallest positive normal: 6.1e-5

The problem zones:

1. Gradients often live here:     1e-8 to 1e-4
                                  ╰── below FP16 range!
                                      These become ZERO.

2. Loss values can be here:       1e-3 to 1e5
                                  ╰── fine for FP16

3. Weight updates (lr * grad):    1e-10 to 1e-6
                                  ╰── WAY below FP16 range!
                                      These become ZERO.
                                      Weights stop updating.
```

**Translation**: In FP16, very small gradients (which are common in the early and late layers of deep networks) underflow to zero. Your model appears to be training but some parameters are literally not updating. The model converges to a worse solution and you may never know why.

### The Fix: Loss Scaling

The solution is beautifully simple. Multiply the loss by a large number (say 1024) before backpropagation. All gradients get scaled up by the same factor (chain rule!). This pushes small gradients back into FP16's representable range. Then, after computing gradients but before updating weights, divide by the same scale factor.

$$L_{\text{scaled}} = L \cdot S$$

$$\nabla_{\text{computed}} = S \cdot \nabla L$$

$$\nabla_{\text{actual}} = \frac{\nabla_{\text{computed}}}{S} = \nabla L$$

The gradients you use for the update are mathematically identical — but they were computed in a range where FP16 didn't lose them to underflow.

**Dynamic loss scaling** goes further: it automatically adjusts the scale factor $S$. If no overflow occurs for several consecutive steps, it increases $S$ to use more of FP16's range. If overflow *does* occur (gradients become Inf), it skips that step and halves $S$.

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Mixed precision training — the production pattern
model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()  # Handles dynamic loss scaling

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()

        # autocast: forward pass in FP16 where safe, FP32 where needed
        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        # Backward pass: scaler scales the loss up, computes FP16 gradients,
        # then unscales before the optimizer step
        scaler.scale(loss).backward()

        # Unscale gradients before clipping (otherwise threshold is wrong)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step: applies the update with unscaled gradients
        # If overflow detected, skips this step and adjusts scale
        scaler.step(optimizer)
        scaler.update()  # Adjust scale factor for next iteration
```

> **Common Mistake**: Clipping gradients *before* calling `scaler.unscale_()`. If you clip the still-scaled gradients, your effective clipping threshold is off by a factor of $S$ (often 65536). Your clipping will either do nothing or clip way too aggressively, depending on the current scale.

### BFloat16: The Best of Both Worlds

BFloat16 (Brain Floating Point) uses 8 exponent bits (same as FP32) but only 7 mantissa bits:

```
Floating Point Format Comparison

Format    Sign  Exponent  Mantissa   Range          Precision
========  ====  ========  ========   =============  =========
FP32       1      8         23       +/- 3.4e38     ~7 digits
FP16       1      5         10       +/- 65504      ~3 digits
BF16       1      8          7       +/- 3.4e38     ~2 digits
                                     ^^^^^^^^^^^^
                                     Same range as FP32!

         FP32: [S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM]  32 bits
         FP16: [S|EEEEE|MMMMMMMMMM]                  16 bits
         BF16: [S|EEEEEEEE|MMMMMMM]                  16 bits
                         ^^^^^^^^
                    BF16 keeps the FP32 exponent range
```

**Translation**: BF16 has the range of FP32 (no more overflow/underflow headaches) but the memory savings of FP16 (half the memory, faster compute). The cost is lower precision, but in practice neural network training is robust to reduced precision in the mantissa. BF16 is now the default for training on modern hardware (A100, H100 GPUs and TPUs).

With BF16 on recent PyTorch and hardware, you often don't even need the GradScaler:

```python
# BF16 training — simpler because no loss scaling needed
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## Part 5: Distributed Training — When One GPU Isn't Enough

### Data Parallel Training

When your model fits on one GPU but your dataset is too large or you want to train faster, you distribute the data across multiple GPUs. Each GPU gets a different mini-batch, computes gradients independently, and then the gradients are averaged (all-reduce) before updating weights.

```
Data Parallel Training (4 GPUs)

   Full Batch (2048 samples)
   ┌─────────────────────────────────────────────┐
   │ micro-batch 0 │ micro-batch 1 │ micro-batch 2 │ micro-batch 3 │
   └───────┬───────┴───────┬───────┴───────┬───────┴───────┬───────┘
           │               │               │               │
           v               v               v               v
      ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
      │  GPU 0  │    │  GPU 1  │    │  GPU 2  │    │  GPU 3  │
      │ forward │    │ forward │    │ forward │    │ forward │
      │backward │    │backward │    │backward │    │backward │
      │  grad_0 │    │  grad_1 │    │  grad_2 │    │  grad_3 │
      └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
           │               │               │               │
           └───────────────┴───────┬───────┴───────────────┘
                                   │
                             All-Reduce:
                        avg_grad = (grad_0 + grad_1 +
                                    grad_2 + grad_3) / 4
                                   │
                    ┌──────────────┬┴──────────────┬──────────────┐
                    v              v               v              v
              ┌──────────┐ ┌──────────┐  ┌──────────┐  ┌──────────┐
              │  GPU 0   │ │  GPU 1   │  │  GPU 2   │  │  GPU 3   │
              │ W -= lr* │ │ W -= lr* │  │ W -= lr* │  │ W -= lr* │
              │ avg_grad │ │ avg_grad │  │ avg_grad │  │ avg_grad │
              └──────────┘ └──────────┘  └──────────┘  └──────────┘
```

Mathematically, the averaged gradient across $N$ GPUs is:

$$g_{\text{avg}} = \frac{1}{N}\sum_{n=1}^{N} g_n$$

This is mathematically equivalent to computing the gradient on the full batch, as long as each GPU uses the same model weights (which they do, since they all apply the same update).

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# DDP wraps your model and handles gradient synchronization
def setup_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_distributed(rank, world_size):
    setup_distributed(rank, world_size)

    model = MyModel().cuda(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # DistributedSampler ensures each GPU gets different data
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(dataset, sampler=sampler, batch_size=512)

    # Remember: effective batch = per_gpu_batch * num_gpus
    # With 4 GPUs and batch_size=512: effective_batch = 2048
    # Apply linear scaling rule to learning rate!

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Ensures different shuffling each epoch
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(rank), targets.cuda(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()  # DDP handles all-reduce automatically!

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
```

---

## Part 6: The Complete Production Training Recipe

Let's bring everything together. Here's what a production-grade training loop looks like — combining all the techniques from this chapter:

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import logging

logger = logging.getLogger(__name__)

class Trainer:
    """
    Production training loop combining:
    - Learning rate warmup + cosine annealing
    - Gradient clipping
    - Mixed precision (AMP)
    - Gradient accumulation
    - Gradient norm monitoring
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        # Optimization
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        # Schedule
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        min_lr: float = 1e-6,
        # Accumulation
        accumulation_steps: int = 1,
        # Mixed precision
        use_amp: bool = True,
    ):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Schedule: linear warmup -> cosine annealing
        warmup = LinearLR(
            self.optimizer, start_factor=0.01, total_iters=warmup_steps
        )
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=use_amp)

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_step(self, batch):
        """Single training step with all bells and whistles."""
        inputs, targets = batch
        inputs, targets = inputs.cuda(), targets.cuda()

        # Forward pass in mixed precision
        with autocast(enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            # Scale for accumulation
            loss = loss / self.accumulation_steps

        # Backward pass (scaler handles loss scaling for FP16)
        self.scaler.scale(loss).backward()

        return loss.item() * self.accumulation_steps  # Return unscaled loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0

        self.optimizer.zero_grad()

        for i, batch in enumerate(self.train_loader):
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1

            # Only update weights after accumulation_steps micro-batches
            if (i + 1) % self.accumulation_steps == 0:
                # Unscale before clipping
                self.scaler.unscale_(self.optimizer)

                # Clip and log gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # Optimizer step (scaler skips if overflow detected)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Log metrics
                if self.global_step % 100 == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Step {self.global_step}: "
                        f"loss={loss:.4f}, "
                        f"grad_norm={grad_norm:.2f}, "
                        f"lr={lr:.2e}, "
                        f"scale={self.scaler.get_scale():.0f}"
                    )

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            inputs, targets = batch
            inputs, targets = inputs.cuda(), targets.cuda()

            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def fit(self, num_epochs: int):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")
                logger.info(f"New best model saved (val_loss={val_loss:.4f})")
```

---

## Part 7: The Decision Matrix — What to Use When

### Learning Rate Schedule Guidelines

| Scenario | Recommended Schedule | Why |
|---|---|---|
| Quick experiment (<10 epochs) | Constant or one-cycle | Minimal tuning needed |
| Standard CNN training | Step decay or cosine | Well-understood, reliable |
| Transformer pretraining | Warmup + cosine decay | Essential for Adam stability |
| Fine-tuning a pretrained model | Lower LR + cosine or constant | Don't destroy pretrained features |
| Limited compute budget | One-cycle policy | Can achieve same loss in 5-10x fewer steps |
| Don't know the right LR | Run LR range test first | 5 minutes of testing saves hours of wasted runs |
| Very large batch training | Linear LR scaling + long warmup | Needed to match small-batch performance |

### Gradient Clipping Guidelines

| Scenario | Recommendation | Typical Value |
|---|---|---|
| Transformers | Always clip by norm | `max_norm=1.0` |
| RNNs / LSTMs | Always clip (essential!) | `max_norm=1.0` to `5.0` |
| Standard CNNs with BatchNorm | Usually not needed | -- |
| GANs | Clip both generator and discriminator | `max_norm=1.0` |
| Reinforcement learning | Clip (policy gradients are volatile) | `max_norm=0.5` to `1.0` |
| Training without BatchNorm | Use AGC | `clip_factor=0.01` |
| Unknown architecture | Start with clip_norm=1.0, monitor grad norms | Adjust based on logs |

### Mixed Precision Guidelines

| Scenario | Recommendation |
|---|---|
| Training on A100/H100 GPUs | BF16 (no loss scaling needed) |
| Training on V100 or older | FP16 with GradScaler |
| Inference only | FP16 or BF16, or even INT8 |
| Numerically sensitive operations (loss, softmax) | Keep in FP32 (autocast does this automatically) |
| Small models that fit easily in memory | FP32 is fine, don't add complexity |

### Batch Size and Accumulation Guidelines

| Scenario | Recommendation |
|---|---|
| Model fits in GPU memory at target batch size | No accumulation needed |
| Target batch >> GPU memory capacity | Gradient accumulation (chunks = target / GPU_batch) |
| Multi-GPU available | Data parallel + accumulation if needed |
| Very large batch (>8K) | Linear LR scaling + extended warmup (5-10 epochs) |

---

## Exercises

### Exercise 1: Implement ReduceLROnPlateau

**Problem**: Implement a scheduler that monitors a metric (validation loss) and reduces the learning rate when it stops improving. This is one of the most practical schedulers — it adapts to your training dynamics instead of following a fixed schedule.

> **You Already Know This**: This is exactly an adaptive backoff strategy, like exponential backoff in retry logic. When requests succeed, keep going. When they start failing (loss stops improving), back off (reduce LR). Wait for things to stabilize (patience), then try again.

```python
class ReduceOnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    Like exponential backoff for optimization.
    """
    def __init__(
        self,
        optimizer,
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-7,
        threshold: float = 1e-4,
    ):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.best_metric = float('inf')
        self.num_bad_epochs = 0

    def step(self, metric: float):
        """Call with current validation loss after each epoch."""
        # Check if this is a meaningful improvement
        if metric < self.best_metric - self.threshold:
            self.best_metric = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
                print(f"Reducing LR: {old_lr:.2e} -> {new_lr:.2e}")
            self.num_bad_epochs = 0

# Usage:
# scheduler = ReduceOnPlateau(optimizer, patience=5, factor=0.5)
# for epoch in range(100):
#     train(model, train_loader)
#     val_loss = evaluate(model, val_loader)
#     scheduler.step(val_loss)
```

### Exercise 2: Compare Clipping Strategies

**Problem**: Train a small model on the same data using (a) no clipping, (b) clip by value, and (c) clip by norm. Inject synthetic gradient explosions by adding noise, and measure both convergence speed and final loss.

```python
import torch
import torch.nn as nn

def compare_clipping_strategies():
    """Compare gradient clipping strategies on unstable training."""
    torch.manual_seed(42)

    # Create a model and dataset
    X = torch.randn(500, 20)
    y = (X[:, :5].sum(dim=1, keepdim=True) > 0).float()

    results = {}

    for strategy_name, clip_fn in [
        ("No clipping", lambda m: None),
        ("Clip by value (0.5)", lambda m: nn.utils.clip_grad_value_(m.parameters(), 0.5)),
        ("Clip by norm (1.0)", lambda m: nn.utils.clip_grad_norm_(m.parameters(), 1.0)),
    ]:
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(20, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        losses = []
        for epoch in range(200):
            pred = model(X)
            loss = nn.functional.binary_cross_entropy(pred, y)

            optimizer.zero_grad()
            loss.backward()

            # Inject gradient noise to simulate instability
            if epoch % 20 == 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * 10.0

            clip_fn(model)
            optimizer.step()

            losses.append(loss.item())
            if torch.isnan(loss):
                print(f"  {strategy_name}: diverged at epoch {epoch}")
                break

        results[strategy_name] = losses
        if not torch.isnan(loss):
            print(f"  {strategy_name}: final_loss={losses[-1]:.4f}")

    return results

compare_clipping_strategies()
```

### Exercise 3: Implement Cosine Annealing with Warm Restarts

**Problem**: Implement the SGDR schedule (cosine annealing with warm restarts) where each restart cycle is $T_{\text{mult}}$ times longer than the previous one.

The schedule within each cycle follows:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_i}\pi\right)\right)$$

where $T_{\text{cur}}$ is steps since last restart and $T_i$ is the current cycle length.

```python
class CosineAnnealingWarmRestarts:
    """
    SGDR: Cosine annealing with warm restarts.
    Each cycle is T_mult times longer than the previous.

    Example with T_0=10, T_mult=2:
      Cycle 1: steps 0-9   (length 10)
      Cycle 2: steps 10-29 (length 20)
      Cycle 3: steps 30-69 (length 40)
    """
    def __init__(
        self,
        optimizer,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.eta_max = optimizer.param_groups[0]['lr']
        self.eta_min = eta_min
        self.T_0 = T_0
        self.T_mult = T_mult
        self.T_cur = 0      # Steps since last restart
        self.T_i = T_0      # Current cycle length
        self.cycle = 0

    def get_lr(self) -> float:
        return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
               (1 + math.cos(math.pi * self.T_cur / self.T_i))

    def step(self):
        # Update learning rate
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Advance within current cycle
        self.T_cur += 1

        # Check for restart
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = int(self.T_i * self.T_mult)
            self.cycle += 1

# Verify the schedule
import math

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

lrs = []
for step in range(70):
    lrs.append(scheduler.get_lr())
    scheduler.step()

# Check restart points
for i in range(1, len(lrs)):
    if lrs[i] > lrs[i-1] + 0.01:
        print(f"  Restart at step {i}: LR jumped from {lrs[i-1]:.4f} to {lrs[i]:.4f}")
```

### Exercise 4: Build a Training Monitor

**Problem**: Create a simple monitor that tracks gradient norms, learning rates, and loss values, and raises warnings when training looks unhealthy.

```python
from collections import deque

class TrainingMonitor:
    """
    Monitors training health by tracking gradient norms, loss, and LR.
    Raises warnings when it detects:
    - Gradient explosions (norm spike)
    - Loss plateaus (no improvement for N steps)
    - NaN/Inf values
    """
    def __init__(self, window_size: int = 100, spike_threshold: float = 10.0):
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.grad_norms = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
        self.warnings = []

    def log_step(self, loss: float, grad_norm: float, lr: float, step: int):
        """Call after each training step."""
        # Check for NaN/Inf
        if math.isnan(loss) or math.isinf(loss):
            self.warnings.append(f"Step {step}: Loss is {loss}! Training has diverged.")
            return

        if math.isnan(grad_norm) or math.isinf(grad_norm):
            self.warnings.append(f"Step {step}: Grad norm is {grad_norm}!")
            return

        # Check for gradient spike
        if len(self.grad_norms) > 10:
            avg_norm = sum(self.grad_norms) / len(self.grad_norms)
            if grad_norm > avg_norm * self.spike_threshold:
                self.warnings.append(
                    f"Step {step}: Gradient spike! "
                    f"norm={grad_norm:.1f} vs avg={avg_norm:.1f}"
                )

        # Check for loss plateau
        if len(self.losses) >= self.window_size:
            recent_avg = sum(list(self.losses)[-20:]) / 20
            older_avg = sum(list(self.losses)[:20]) / 20
            if abs(recent_avg - older_avg) / (older_avg + 1e-8) < 0.001:
                self.warnings.append(
                    f"Step {step}: Loss plateau detected. "
                    f"Consider reducing LR or checking data."
                )

        self.grad_norms.append(grad_norm)
        self.losses.append(loss)

    def get_warnings(self) -> list:
        warnings = self.warnings.copy()
        self.warnings.clear()
        return warnings
```

---

## Summary

Here's the practical hierarchy of what matters most when training a model:

```
Production Training Checklist (in priority order)

  1. Learning Rate
     ├── Run LR range test to find the right ballpark
     ├── Use warmup (especially with Adam/AdamW)
     ├── Apply cosine annealing or one-cycle policy
     └── Scale LR with batch size (linear scaling rule)

  2. Gradient Clipping
     ├── Always use clip-by-norm (not clip-by-value)
     ├── Standard: max_norm=1.0 for transformers
     ├── Monitor gradient norms — they're your early warning
     └── Clip AFTER accumulation, BEFORE optimizer step

  3. Mixed Precision
     ├── Use BF16 on A100/H100 (simplest)
     ├── Use FP16 + GradScaler on older GPUs
     ├── Unscale gradients before clipping
     └── Keep loss computation in FP32 (autocast handles this)

  4. Gradient Accumulation
     ├── Divide loss by accumulation_steps
     ├── Only clip and step after all micro-batches
     └── Effective batch = micro_batch * accumulation_steps * num_GPUs

  5. Monitoring
     ├── Log loss, grad_norm, LR every N steps
     ├── Watch for grad_norm spikes (bad data or instability)
     ├── Watch for loss plateaus (reduce LR or increase batch diversity)
     └── Save checkpoints frequently (compute is expensive)
```

The formulas we covered:

- **Step decay**: $\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}$
- **Exponential decay**: $\eta_t = \eta_0 \cdot e^{-\lambda t}$
- **Cosine annealing**: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$
- **Linear warmup**: $\eta_t = \eta_{\text{target}} \cdot \frac{t}{T_w}$
- **Clip by norm**: $g^{\text{clipped}} = g \cdot \min\left(1, \frac{\theta}{\|g\|}\right)$
- **Clip by value**: $g_i^{\text{clipped}} = \max(\min(g_i, \theta), -\theta)$
- **AGC**: $g^{\text{clipped}} = g \cdot \min\left(1, \frac{\lambda \|W\|}{\|g\|}\right)$
- **Gradient explosion**: $\frac{\partial L}{\partial W_1} = \prod_{l=1}^{L} \frac{\partial h_{l+1}}{\partial h_l} \cdot \frac{\partial h_1}{\partial W_1}$ (product of many terms > 1 = explosion)
- **Linear LR scaling**: $\eta_{\text{large}} = \eta_{\text{base}} \cdot \frac{B_{\text{large}}}{B_{\text{base}}}$

None of these formulas are hard. The hard part is knowing which ones to combine, in what order, and how to debug them when things go wrong. Now you know.

---

> **What's Next**: You now have the complete practical toolkit for making optimization work in production — learning rate schedules, gradient clipping, mixed precision, accumulation, and distributed training. In **Level 14: Advanced Topics**, we look at where the field is heading: the mathematical foundations of diffusion models, geometric deep learning, neural ODEs, and the theoretical tools that appear in cutting-edge research papers. These are the concepts that separate "using ML" from "pushing ML forward."
