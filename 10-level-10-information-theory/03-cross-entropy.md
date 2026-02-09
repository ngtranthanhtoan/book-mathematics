# Chapter 3: Cross-Entropy

> **Transition In:** Entropy measures self-uncertainty. Cross-entropy measures how well your model's distribution matches reality.

Cross-entropy is THE loss function for classification. When PyTorch's `nn.CrossEntropyLoss` computes your training loss, it's computing H(p, q) = -sum p(x) log q(x). Understanding this formula means understanding what your classifier is actually optimizing.

---

## The Problem That Starts Everything

**What makes cross-entropy better than MSE for classification?**

Picture this. You have a binary classifier. The true label is 1. Your model predicts 0.99. Both MSE and cross-entropy give you a small loss -- great. Now your model predicts 0.001. MSE gives you roughly 1.0. Cross-entropy gives you roughly 6.9. That 7x difference matters. Cross-entropy screams at your model when it is confidently wrong, while MSE just politely suggests it reconsider.

But the real answer goes deeper than penalty magnitude. Let's build up from first principles.

---

## Running Example: The Genre Classifier

Throughout this chapter, you are building a music genre classifier with four classes: Rock, Pop, Jazz, Classical. Every concept will be grounded in this model's actual loss computations.

Your training sample is a Rock song. The true label is one-hot encoded:

```
true label p = [1, 0, 0, 0]     (Rock, Pop, Jazz, Classical)
```

Your model produces two different predictions on two different training steps:

```
Good prediction q1 = [0.7, 0.1, 0.1, 0.1]
Bad prediction  q2 = [0.1, 0.3, 0.3, 0.3]
```

What does cross-entropy say about each?

```
H(p, q1) = -log(0.7) ≈ 0.36 nats    (model is on the right track)
H(p, q2) = -log(0.1) ≈ 2.30 nats    (model is very confused)
```

That 0.36 vs 2.30 gap is the core of cross-entropy: it measures how many bits (or nats) your model wastes by being wrong. Hold onto these numbers -- we will revisit them at every stage.

---

## Mathematical Foundation

### Definition: Cross-Entropy

For discrete distributions P (true) and Q (model):

$$H(P, Q) = -\sum_{x} P(x) \log Q(x) = \mathbb{E}_{x \sim P}[-\log Q(x)]$$

In words: the expected surprise of the model Q when data actually comes from P.

### Cross-Entropy for Classification

For a single sample with true label $y$ and predicted probabilities $\hat{y}$:

**Binary classification:**

$$H(y, \hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

**Multi-class classification (one-hot encoded):**

$$H(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

Since $y$ is one-hot, this simplifies to:

$$H(\mathbf{y}, \hat{\mathbf{y}}) = -\log(\hat{y}_{true})$$

This is just the negative log probability of the true class!

Back to the genre classifier: with `p = [1, 0, 0, 0]`, all the zeros kill every term except the first. You are left with `-1 * log(q_Rock)`. That is it. The only thing cross-entropy cares about in a one-hot setting is how much probability mass your model put on the correct class.

### Batch Cross-Entropy Loss

For a dataset with $N$ samples:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} H(y^{(i)}, \hat{y}^{(i)}) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_c^{(i)} \log(\hat{y}_c^{(i)})$$

### Properties of Cross-Entropy

1. **Non-negativity**: $H(P, Q) \geq 0$

2. **Minimum at P = Q**: $H(P, Q) \geq H(P)$, with equality iff Q = P

3. **Not symmetric**: $H(P, Q) \neq H(Q, P)$ in general

4. **Decomposition**: $H(P, Q) = H(P) + D_{KL}(P \| Q)$

### Connection to Maximum Likelihood

Minimizing cross-entropy over a dataset is equivalent to maximum likelihood estimation:

$$\min_\theta H(P_{data}, Q_\theta) = \min_\theta \mathbb{E}_{x \sim P_{data}}[-\log Q_\theta(x)] = \max_\theta \mathbb{E}_{x \sim P_{data}}[\log Q_\theta(x)]$$

The last term is the log-likelihood. This is why cross-entropy loss is so fundamental -- it directly optimizes likelihood.

---

## SWE Bridge #1: Cross-Entropy as Log-Scoring

You might have seen the **Brier score** in evaluation contexts -- it is basically MSE applied to probabilities. Cross-entropy is the **logarithmic scoring rule**, and it is strictly proper, meaning the only way to minimize your expected loss is to predict the true probabilities.

Why does the ML world prefer log-scoring over Brier scoring? Two reasons:

1. **Extreme penalty for extreme confidence**: If you predict 0.001 for the true class, log-scoring gives you `-log(0.001) = 6.9`. Brier scoring gives you `(1 - 0.001)^2 = 0.998`. Log-scoring makes confident wrong predictions catastrophically expensive. Brier scoring barely notices.

2. **Decomposition into information-theoretic quantities**: Cross-entropy breaks down cleanly into entropy + KL divergence. Brier scoring has no such decomposition.

Think of it this way: log-scoring is to probability evaluation what logarithmic returns are to finance -- it treats errors multiplicatively, not additively.

### Genre classifier connection

Your model predicts `q = [0.7, 0.1, 0.1, 0.1]` for a Rock song.

```
Log score (cross-entropy):  -log(0.7) = 0.36
Brier score:                (1-0.7)^2 + (0-0.1)^2 + (0-0.1)^2 + (0-0.1)^2 = 0.12
```

Now for the bad prediction `q = [0.1, 0.3, 0.3, 0.3]`:

```
Log score (cross-entropy):  -log(0.1) = 2.30
Brier score:                (1-0.1)^2 + (0-0.3)^2 + (0-0.3)^2 + (0-0.3)^2 = 1.08
```

Cross-entropy went from 0.36 to 2.30 (6.4x increase). Brier went from 0.12 to 1.08 (9x increase). Both penalize the bad prediction more, but cross-entropy's logarithmic scale makes it especially punishing as predictions approach zero -- exactly where you need the strongest learning signal.

---

## SWE Bridge #2: H(p,q) vs H(p) -- The Gap Is Wasted Bits

Here is the key decomposition you need to internalize:

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

Cross-entropy = Entropy + KL Divergence

Let's unpack what each piece means for your classifier:

- **H(P)** is the entropy of the true distribution. For one-hot labels, H(P) = 0. There is zero inherent uncertainty -- the song IS Rock. This is the theoretical minimum cost of encoding the truth.

- **D_KL(P||Q)** is the KL divergence. It measures how many EXTRA bits you waste because your model Q differs from reality P. This is pure inefficiency -- every nat of KL divergence is a nat your model could eliminate by getting better.

- **H(P, Q)** is the total encoding cost. It is the minimum bits plus the wasted bits.

When your labels are one-hot (as they usually are in classification), H(P) = 0, so:

```
H(P, Q) = 0 + D_KL(P||Q) = D_KL(P||Q)
```

This means **minimizing cross-entropy IS minimizing KL divergence** in the standard classification setup. Every reduction in your training loss is a direct reduction in the information-theoretic distance between your model and reality.

### ASCII Diagram: H(P,Q) Decomposition

```
    H(P,Q): Total encoding cost
    ┌─────────────────────────────────────────┐
    │                                         │
    │   H(P): Inherent uncertainty    D_KL(P||Q): Model inefficiency
    │   (you cannot reduce this)      (you CAN reduce this)
    │                                         │
    │   ┌──────────────────┐┌─────────────────┐
    │   │                  ││  ///////////////│
    │   │   Entropy of P   ││  KL Divergence  │
    │   │   (fixed floor)  ││  (your target)  │
    │   │                  ││  ///////////////│
    │   └──────────────────┘└─────────────────┘
    │                                         │
    └─────────────────────────────────────────┘

    For one-hot labels: H(P) = 0, so the ENTIRE loss is D_KL.
    Your model's cross-entropy loss literally IS the divergence.

    Perfect model:  Q = P  -->  D_KL = 0  -->  H(P,Q) = H(P)
    Typical model:  Q ≠ P  -->  D_KL > 0  -->  H(P,Q) > H(P)
```

### Genre classifier connection

For one-hot `p = [1, 0, 0, 0]`:
- H(P) = -1 * log(1) = 0 (no uncertainty in the label)
- Good prediction: H(P, Q) = 0.36 = 0 + 0.36. All 0.36 nats are wasted bits.
- Bad prediction: H(P, Q) = 2.30 = 0 + 2.30. All 2.30 nats are wasted bits.

Training your model is literally squeezing out wasted bits until the loss approaches zero.

---

## The Cross-Entropy Loss Curve

Here is what the loss landscape looks like as a function of the predicted probability for the true class. This is the curve your optimizer walks down during training:

```
    Loss
    (-log q)
     ^
  5  |*
     | *
  4  |  *
     |   *
  3  |    *
     |     *
  2  |       *                       <-- Bad prediction: q=0.1, loss=2.30
     |         *
  1  |            **
     |               ***             <-- Good prediction: q=0.7, loss=0.36
     |                  *****
  0  +-------+----+--------*****--->  q (predicted prob for true class)
     0      0.2  0.4   0.6  0.8  1.0

  Key observations:
  - The curve is STEEP near 0: confident wrong predictions get massive gradients
  - The curve is FLAT near 1: confident correct predictions get tiny gradients
  - This is exactly the learning dynamic you want
```

Compare this to MSE's loss curve for classification, `(1 - q)^2`:

```
    Loss
     ^
  1  |*
     | *
     |  **
     |    **
     |      **                       <-- MSE is gentle everywhere
     |        ***
     |           ***
     |              ****
  0  +------------------*****--->  q
     0      0.2  0.4   0.6  0.8  1.0

  Problem: near q=0, MSE gradient is only -2(1-q) ≈ -2
  Cross-entropy gradient is -1/q ≈ -infinity
  MSE does not punish confident wrong predictions hard enough.
```

This visual difference is the answer to "why cross-entropy over MSE for classification." When your model says "I am 99% sure this is NOT Rock" about a Rock song, you want the gradient to be enormous. Cross-entropy delivers. MSE does not.

---

## Common Mistake Box

> **Cross-entropy with softmax has a beautifully simple gradient: y_hat - y. This is NOT a coincidence -- it is why this combination was chosen.**

When you combine softmax activation with cross-entropy loss, the gradient of the loss with respect to the logits simplifies to:

```
dL/dz_i = y_hat_i - y_i
```

That is it. The gradient is just "prediction minus truth." No logarithms, no divisions, no numerical instability. This is the same elegant simplification you get with sigmoid + binary cross-entropy, and with linear activation + MSE. These pairings are called **canonical link functions** in statistics, and they were chosen precisely because the math collapses to something clean.

This is why PyTorch's `nn.CrossEntropyLoss` fuses softmax and cross-entropy into a single operation. It is not just for numerical stability (though that helps). It is because the fused gradient computation is simpler and faster.

### Genre classifier connection

Your model outputs raw logits `z = [2.1, -0.5, -0.3, -0.1]`, which softmax converts to `q = [0.7, 0.05, 0.06, 0.08]` (approximately). The gradient for backprop is:

```
dL/dz = q - p = [0.7, 0.05, 0.06, 0.08] - [1, 0, 0, 0]
              = [-0.3, 0.05, 0.06, 0.08]
```

The model should increase the Rock logit (negative gradient) and decrease all others (positive gradients). Simple. Stable. Fast.

---

## SWE Bridge #3: Relationship to KL Divergence

You already saw the decomposition:

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

But let's make the engineering implications explicit:

**When you are training a classifier**, H(P) is fixed -- it depends only on your labels, not your model. So minimizing H(P, Q) is exactly the same as minimizing D_KL(P||Q). Your optimizer does not know or care about entropy. It is chasing the KL divergence to zero.

**When you are comparing two models**, cross-entropy gives you the total cost. KL divergence gives you the model-specific cost. If Model A has cross-entropy 1.5 and Model B has cross-entropy 0.8, and both use the same labels (same H(P)), then Model B has `1.5 - 0.8 = 0.7` fewer nats of KL divergence. It wastes 0.7 fewer bits per sample.

**When you are doing knowledge distillation**, the teacher's output is NOT one-hot, so H(P) is NOT zero. The decomposition matters: the student is trying to match the teacher's uncertainty, not eliminate it. Some cross-entropy is irreducible because the teacher itself is uncertain.

---

## SWE Bridge #4: Label Smoothing -- Adding Noise to Prevent Overconfidence

Standard one-hot labels are brutally absolute: "This is Rock with 100% certainty." Cross-entropy with one-hot labels pushes your model's logits toward infinity, trying to make `q_Rock = 1.0` exactly. This causes:

1. **Overconfidence**: Model outputs 0.999 when it should output 0.85
2. **Poor calibration**: Predicted probabilities do not match actual frequencies
3. **Reduced generalization**: Model memorizes instead of learning patterns

**Label smoothing** replaces one-hot `[1, 0, 0, 0]` with something like `[0.925, 0.025, 0.025, 0.025]`. You are saying: "This is probably Rock, but I acknowledge some uncertainty."

The formula with smoothing parameter $\epsilon$:

$$y_{smooth} = (1 - \epsilon) \cdot y_{one\_hot} + \frac{\epsilon}{C}$$

With $\epsilon = 0.1$ and C = 4 classes:

```
One-hot:   [1.0,   0.0,   0.0,   0.0  ]
Smoothed:  [0.925, 0.025, 0.025, 0.025]
```

### Genre classifier connection with label smoothing

With the smoothed label `p_smooth = [0.925, 0.025, 0.025, 0.025]` and model prediction `q = [0.7, 0.1, 0.1, 0.1]`:

```
H(p_smooth, q) = -(0.925 * log(0.7) + 0.025 * log(0.1) + 0.025 * log(0.1) + 0.025 * log(0.1))
               = -(0.925 * (-0.357) + 3 * 0.025 * (-2.303))
               = -(−0.330 + (−0.173))
               = 0.503 nats
```

Compare to one-hot: `H(p, q) = -log(0.7) = 0.357 nats`. The smoothed loss is higher because the model is also penalized for not distributing some probability to the other classes. This prevents the logits from going to infinity and keeps the model honest.

In PyTorch:

```python
# Standard cross-entropy
loss_fn = nn.CrossEntropyLoss()

# With label smoothing (epsilon = 0.1)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

One line of code. Measurable improvement in calibration. Use it.

---

## Why Cross-Entropy Beats MSE: The Full Picture

Now you have all the pieces. Let's assemble the definitive comparison:

| Aspect | Cross-Entropy | MSE |
|--------|--------------|-----|
| Gradient when confident & wrong (q near 0) | -1/q (huge, self-correcting) | -2(1-q) (bounded, sluggish) |
| Gradient when confident & right (q near 1) | -1/q (small, stops pushing) | -2(1-q) (small, stops pushing) |
| Probabilistic interpretation | Yes (negative log-likelihood) | No |
| Information-theoretic meaning | Expected encoding cost | No direct meaning |
| Canonical pairing with softmax | Yes (gradient = y_hat - y) | No |
| Convex for log-linear models | Yes | Yes |
| Handles class imbalance | Better (log penalty) | Worse (bounded penalty) |

The bottom line: cross-entropy provides the right learning dynamics for probability estimation. It punishes hard where it matters and backs off where it does not.

---

## Code Example

```python
import numpy as np
import matplotlib.pyplot as plt

def cross_entropy(p_true, q_pred, epsilon=1e-15):
    """
    Calculate cross-entropy between true distribution p and predicted q.

    Parameters:
    -----------
    p_true : array-like
        True probability distribution (or one-hot labels)
    q_pred : array-like
        Predicted probability distribution
    epsilon : float
        Small value to prevent log(0)

    Returns:
    --------
    float : Cross-entropy value
    """
    p_true = np.asarray(p_true, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)

    # Clip predictions to prevent log(0)
    q_pred = np.clip(q_pred, epsilon, 1 - epsilon)

    # Cross-entropy: -sum(p * log(q))
    return -np.sum(p_true * np.log(q_pred))

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Binary cross-entropy loss.

    Parameters:
    -----------
    y_true : float or array
        True labels (0 or 1)
    y_pred : float or array
        Predicted probabilities
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """
    Categorical cross-entropy loss for batches.

    Parameters:
    -----------
    y_true : array (N, C)
        One-hot encoded true labels
    y_pred : array (N, C)
        Predicted probabilities (softmax outputs)
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# ============================================================
# Genre Classifier Running Example
# ============================================================

print("=== Genre Classifier: Cross-Entropy in Action ===\n")

# True label: Rock (one-hot)
p_true = np.array([1, 0, 0, 0])
classes = ["Rock", "Pop", "Jazz", "Classical"]

# Good prediction
q_good = np.array([0.7, 0.1, 0.1, 0.1])
ce_good = cross_entropy(p_true, q_good)

# Bad prediction
q_bad = np.array([0.1, 0.3, 0.3, 0.3])
ce_bad = cross_entropy(p_true, q_bad)

print(f"True label: Rock {p_true}")
print(f"\nGood prediction: {q_good}")
print(f"  H(p, q) = -log(0.7) = {ce_good:.4f} nats")
print(f"\nBad prediction:  {q_bad}")
print(f"  H(p, q) = -log(0.1) = {ce_bad:.4f} nats")
print(f"\nThe bad prediction costs {ce_bad/ce_good:.1f}x more nats.\n")

# ============================================================
# Cross-Entropy Decomposition: H(P,Q) = H(P) + D_KL(P||Q)
# ============================================================

print("=== Cross-Entropy Decomposition ===\n")

def entropy(p):
    p_nonzero = p[p > 0]
    return -np.sum(p_nonzero * np.log(p_nonzero))

def kl_divergence(p, q, epsilon=1e-15):
    q = np.clip(q, epsilon, 1)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

# With soft labels (e.g., from knowledge distillation)
p_soft = np.array([0.7, 0.2, 0.05, 0.05])
q_model = np.array([0.5, 0.3, 0.1, 0.1])

H_p = entropy(p_soft)
D_kl = kl_divergence(p_soft, q_model)
H_pq = cross_entropy(p_soft, q_model)

print(f"Soft label P: {p_soft}")
print(f"Model Q:      {q_model}\n")
print(f"H(P)           = {H_p:.4f}  (inherent uncertainty)")
print(f"D_KL(P||Q)     = {D_kl:.4f}  (model inefficiency)")
print(f"H(P) + D_KL    = {H_p + D_kl:.4f}")
print(f"H(P, Q)        = {H_pq:.4f}")
print(f"Match?           {np.isclose(H_pq, H_p + D_kl)}\n")

# With one-hot labels (standard classification)
p_onehot = np.array([1, 0, 0, 0])
H_p_onehot = entropy(p_onehot)
D_kl_onehot = kl_divergence(p_onehot, q_good)
H_pq_onehot = cross_entropy(p_onehot, q_good)

print(f"One-hot P: {p_onehot}")
print(f"Model Q:   {q_good}\n")
print(f"H(P)       = {H_p_onehot:.4f}  (zero! no uncertainty in one-hot)")
print(f"D_KL(P||Q) = {D_kl_onehot:.4f}")
print(f"H(P, Q)    = {H_pq_onehot:.4f}")
print(f"Entire loss is KL divergence: {np.isclose(H_pq_onehot, D_kl_onehot)}\n")

# ============================================================
# Binary Cross-Entropy: Loss Landscape
# ============================================================

print("=== Binary Cross-Entropy Loss ===\n")

y_true = 1
probs = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]
print("True label: 1 (positive class)\n")
print("Predicted Prob | BCE Loss")
print("-" * 30)
for p in probs:
    loss = binary_cross_entropy(y_true, p)
    print(f"     {p:.2f}       | {loss:.4f}")

# ============================================================
# Visualize BCE loss landscape
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pred_range = np.linspace(0.001, 0.999, 1000)
loss_y1 = binary_cross_entropy(1, pred_range)
loss_y0 = binary_cross_entropy(0, pred_range)

axes[0].plot(pred_range, loss_y1, 'b-', linewidth=2, label='True label = 1')
axes[0].plot(pred_range, loss_y0, 'r-', linewidth=2, label='True label = 0')
axes[0].set_xlabel('Predicted Probability', fontsize=12)
axes[0].set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
axes[0].set_title('BCE Loss vs Predicted Probability', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 5)

# Gradient of BCE
gradient_y1 = -1 / pred_range
axes[1].plot(pred_range, gradient_y1, 'b-', linewidth=2)
axes[1].set_xlabel('Predicted Probability', fontsize=12)
axes[1].set_ylabel('Gradient of BCE (y=1)', fontsize=12)
axes[1].set_title('BCE Gradient: Larger for Worse Predictions', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(-20, 0)
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('bce_loss_landscape.png', dpi=150)
plt.show()

# ============================================================
# Batch Categorical Cross-Entropy
# ============================================================

print("\n=== Batch Genre Classifier Loss ===\n")

y_true_batch = np.array([
    [1, 0, 0, 0],  # Rock
    [0, 1, 0, 0],  # Pop
    [0, 0, 1, 0],  # Jazz
    [1, 0, 0, 0],  # Rock
    [0, 0, 0, 1],  # Classical
])

y_pred_batch = np.array([
    [0.7, 0.1, 0.1, 0.1],  # Correct, confident
    [0.1, 0.6, 0.2, 0.1],  # Correct, moderate
    [0.1, 0.2, 0.5, 0.2],  # Correct, uncertain
    [0.3, 0.3, 0.2, 0.2],  # Correct, barely
    [0.4, 0.2, 0.2, 0.2],  # WRONG: predicts Rock
])

batch_loss = categorical_cross_entropy_loss(y_true_batch, y_pred_batch)
individual_losses = -np.sum(y_true_batch * np.log(np.clip(y_pred_batch, 1e-15, 1)), axis=1)

print("Sample | True     | Predicted                | Loss")
print("-" * 62)
for i in range(5):
    true_class = classes[np.argmax(y_true_batch[i])]
    pred_probs = y_pred_batch[i]
    loss = individual_losses[i]
    correct = "ok" if np.argmax(y_true_batch[i]) == np.argmax(y_pred_batch[i]) else "WRONG"
    print(f"   {i}   | {true_class:9s}| {pred_probs} | {loss:.4f} {correct}")

print(f"\nMean batch loss: {batch_loss:.4f} nats")

# ============================================================
# Label Smoothing Demo
# ============================================================

print("\n=== Label Smoothing Effect ===\n")

epsilon = 0.1
C = 4
p_onehot = np.array([1, 0, 0, 0])
p_smooth = (1 - epsilon) * p_onehot + epsilon / C

print(f"Smoothing epsilon: {epsilon}")
print(f"One-hot label:  {p_onehot}")
print(f"Smoothed label: {p_smooth}\n")

q_pred = np.array([0.7, 0.1, 0.1, 0.1])
ce_hard = cross_entropy(p_onehot, q_pred)
ce_smooth = cross_entropy(p_smooth, q_pred)

print(f"Model prediction: {q_pred}")
print(f"CE with one-hot:  {ce_hard:.4f} nats")
print(f"CE with smoothed: {ce_smooth:.4f} nats")
print(f"Difference:       {ce_smooth - ce_hard:.4f} nats (smoothing adds penalty for ignoring other classes)")

# ============================================================
# Verification with Libraries
# ============================================================

print("\n=== Verification with Libraries ===\n")

try:
    from sklearn.metrics import log_loss

    y_true_labels = [np.argmax(y) for y in y_true_batch]
    sklearn_loss = log_loss(y_true_labels, y_pred_batch, labels=[0, 1, 2, 3])

    print(f"Our implementation: {batch_loss:.6f}")
    print(f"sklearn log_loss:   {sklearn_loss:.6f}")
    print(f"Match: {np.isclose(batch_loss, sklearn_loss)}")
except ImportError:
    print("sklearn not available for verification")
```

**Output:**
```
=== Genre Classifier: Cross-Entropy in Action ===

True label: Rock [1, 0, 0, 0]

Good prediction: [0.7, 0.1, 0.1, 0.1]
  H(p, q) = -log(0.7) = 0.3567 nats

Bad prediction:  [0.1, 0.3, 0.3, 0.3]
  H(p, q) = -log(0.1) = 2.3026 nats

The bad prediction costs 6.5x more nats.

=== Binary Cross-Entropy ===

True label: 1 (positive class)

Predicted Prob | BCE Loss
------------------------------
     0.99       | 0.0101
     0.90       | 0.1054
     0.70       | 0.3567
     0.50       | 0.6931
     0.30       | 1.2040
     0.10       | 2.3026
     0.01       | 4.6052
```

---

## Pitfalls That Will Bite You

### 1. Double Softmax

```python
# WRONG: Softmax then CrossEntropyLoss
probs = torch.softmax(logits, dim=1)
loss = nn.CrossEntropyLoss()(probs, labels)  # Double softmax!

# RIGHT: Logits directly to CrossEntropyLoss
loss = nn.CrossEntropyLoss()(logits, labels)  # Correct
```

PyTorch's `nn.CrossEntropyLoss` applies softmax internally. If you softmax your logits first, the loss function softmaxes them again. Your model will train, but the gradients will be wrong and convergence will be terrible. This is the single most common PyTorch classification bug.

### 2. Numerical Instability

Never compute `-p * log(q)` directly when q can be zero. Always clip:

```python
q_pred = np.clip(q_pred, 1e-15, 1 - 1e-15)
```

Or better yet, use PyTorch's fused `nn.CrossEntropyLoss` which works in log-space and never computes raw probabilities.

### 3. Label Encoding Mismatch

- `nn.CrossEntropyLoss` expects integer class indices, not one-hot vectors
- `nn.BCELoss` expects probabilities after sigmoid, not raw logits
- `nn.BCEWithLogitsLoss` expects raw logits (fuses sigmoid + BCE)

### 4. Base of Logarithm

Natural log (ln) gives you nats. Log base 2 gives you bits. Most ML frameworks use natural log. Be consistent and know which you are using.

---

## When to Use / When to Reach for Something Else

### Use Cross-Entropy When

- **Classification tasks**: It is the default for a reason
- **You need calibrated probabilities**: Cross-entropy directly optimizes log-likelihood
- **Neural networks**: The softmax + cross-entropy gradient is clean and stable
- **Multi-class problems**: One-hot or integer-label, it handles both

### Reach for Alternatives When

- **Regression**: Use MSE, MAE, or Huber loss
- **Ranking tasks**: Use pairwise losses, hinge loss, or NDCG-based losses
- **Extreme class imbalance**: Use focal loss -- it is cross-entropy multiplied by $(1-p_t)^\gamma$, which downweights easy examples
- **Noisy labels**: Use label smoothing (add `label_smoothing=0.1` to your loss function) or confidence penalty
- **Knowledge distillation**: Use cross-entropy with soft targets from the teacher model, where the decomposition H(P) + D_KL matters because H(P) is no longer zero

---

## Exercises

### Exercise 1: Computing Cross-Entropy

**Problem**: Your genre classifier outputs probabilities [0.1, 0.2, 0.3, 0.4] for [Rock, Pop, Jazz, Classical]. The true class is Jazz (index 2). What is the cross-entropy loss?

**Solution**:
```python
import numpy as np

y_true = [0, 0, 1, 0]  # One-hot for Jazz
y_pred = [0.1, 0.2, 0.3, 0.4]

# Cross-entropy simplifies to -log(prob of true class)
ce_loss = -np.log(0.3)
print(f"Cross-entropy: {ce_loss:.4f} nats")  # approx 1.204 nats
```

### Exercise 2: Cross-Entropy Decomposition

**Problem**: Given P = [0.5, 0.5] and Q = [0.9, 0.1], verify that H(P,Q) = H(P) + D_KL(P||Q).

**Solution**:
```python
import numpy as np

P = np.array([0.5, 0.5])
Q = np.array([0.9, 0.1])

# Entropy of P
H_P = -np.sum(P * np.log(P))  # = 0.693

# KL divergence
D_KL = np.sum(P * np.log(P / Q))  # = 0.5 * log(0.5/0.9) + 0.5 * log(0.5/0.1)

# Cross-entropy
H_PQ = -np.sum(P * np.log(Q))

print(f"H(P) = {H_P:.4f}")
print(f"D_KL(P||Q) = {D_KL:.4f}")
print(f"H(P,Q) = {H_PQ:.4f}")
print(f"H(P) + D_KL = {H_P + D_KL:.4f}")
print(f"Equal? {np.isclose(H_PQ, H_P + D_KL)}")
```

### Exercise 3: Gradient Analysis

**Problem**: For binary cross-entropy with true label y=1, derive the gradient with respect to the predicted probability p and explain why it is large when p is small.

**Solution**:
Binary cross-entropy: $L = -\log(p)$ when $y=1$

Gradient: $\frac{\partial L}{\partial p} = -\frac{1}{p}$

When $p$ is small (model assigns low probability to true class):
- $p = 0.1 \Rightarrow$ gradient $= -10$
- $p = 0.01 \Rightarrow$ gradient $= -100$

Large gradient means large update, pushing the model to increase probability for the true class. This is exactly what you want: learn more from mistakes. Compare to MSE where the gradient is $-2(1-p)$, which is at most $-2$ regardless of how wrong the model is.

### Exercise 4: Label Smoothing Impact

**Problem**: Your genre classifier predicts [0.95, 0.02, 0.02, 0.01] for a Rock song. Compute the cross-entropy loss with (a) one-hot labels and (b) label smoothing with epsilon = 0.1. Why is (b) higher?

**Solution**:
```python
import numpy as np

q = np.array([0.95, 0.02, 0.02, 0.01])

# (a) One-hot
p_hard = np.array([1, 0, 0, 0])
ce_hard = -np.sum(p_hard * np.log(q))
print(f"One-hot CE: {ce_hard:.4f}")  # -log(0.95) = 0.0513

# (b) Label smoothed
epsilon = 0.1
p_smooth = (1 - epsilon) * p_hard + epsilon / 4
# p_smooth = [0.925, 0.025, 0.025, 0.025]
ce_smooth = -np.sum(p_smooth * np.log(q))
print(f"Smoothed CE: {ce_smooth:.4f}")  # higher

# (b) is higher because the model is penalized for putting only 0.02
# on Pop, Jazz and 0.01 on Classical when the smoothed target
# expects 0.025 on each. The model must spread probability more evenly.
```

---

## Summary

- **Cross-entropy** measures how well model Q predicts data from P: $H(P,Q) = -\sum P(x) \log Q(x)$
- **For classification**: Simplifies to $-\log(\text{probability of true class})$
- **Log-scoring**: Like Brier score but with logarithmic penalties -- catastrophic for confident wrong predictions
- **Decomposition**: $H(P,Q) = H(P) + D_{KL}(P\|Q)$ -- the gap between H(P,Q) and H(P) is wasted bits
- **For one-hot labels**: H(P) = 0, so entire loss IS the KL divergence
- **Softmax + cross-entropy gradient**: $\hat{y} - y$ -- simple by design, not by accident
- **Label smoothing**: Prevents overconfidence by distributing small probability to wrong classes
- **Equivalent to negative log-likelihood**: Minimizing cross-entropy = maximizing likelihood
- **Gradient property**: Larger gradients for worse predictions (self-correcting)
- **Genre classifier**: Good prediction (0.7 on Rock) costs 0.36 nats. Bad prediction (0.1 on Rock) costs 2.30 nats. Training squeezes out those wasted nats.

> **Transition Out:** Cross-entropy tells you how wrong your model is. KL divergence tells you exactly how much of that wrongness comes from your model vs. inherent uncertainty.
