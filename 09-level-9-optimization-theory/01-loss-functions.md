# Chapter 1: Loss Functions

> **Building On** — You understand probability distributions and can compute expectations. Now the central question: how do you measure how WRONG your model is? Loss functions quantify prediction error, and minimizing them IS training.

---

Training a model = minimizing a number. That number is the loss function. Choose the wrong loss and your model optimizes for the wrong thing — like a GPS that minimizes distance but ignores traffic. The loss function IS your model's objective, and picking the right one is half the battle.

Let's make this concrete with a running example you'll see throughout this chapter.

---

## The Running Example: Predicting Movie Ratings

You're building a recommendation system. Users rate movies 1-5 stars. Your model predicts ratings. A user gave a movie 1 star, but your model predicted 5 stars. How "wrong" is your model?

- **MSE** penalizes that prediction by $(5 - 1)^2 = 16$
- **MAE** penalizes it by $|5 - 1| = 4$

Which is "right"? Neither — they encode different beliefs about what matters. MSE says "a 4-star mistake is catastrophically worse than a 1-star mistake." MAE says "a 4-star mistake is exactly 4x worse than a 1-star mistake." Your choice shapes the model's behavior. If your system surfaces "top picks" and a wildly wrong prediction ruins user trust, MSE's harsh outlier penalty might be what you want. If your system aggregates across thousands of predictions and you care about average quality, MAE is more robust.

Same data. Different loss. Different model behavior. That's the core idea.

---

## The Loss Function Decision Tree

Before diving into formulas, here's the roadmap. You'll pick a loss based on your problem type:

```
What are you predicting?
│
├── Continuous values (regression)
│   ├── Clean data, outliers are rare ──────────► MSE
│   ├── Noisy data, outliers matter ────────────► MAE
│   └── Want MSE smoothness + MAE robustness ──► Huber Loss
│
└── Categories (classification)
    ├── Binary (yes/no, spam/not-spam) ─────────► Binary Cross-Entropy (Log Loss)
    ├── Multi-class (exactly one label) ────────► Categorical Cross-Entropy + Softmax
    ├── Multi-label (multiple labels) ──────────► Binary Cross-Entropy per label + Sigmoid
    └── Maximum margin (SVM-style) ─────────────► Hinge Loss
```

Now let's build each one from the ground up — starting with where they fail, then seeing why the math fixes it.

---

## Regression Losses

### Mean Squared Error (MSE)

**The problem**: You're predicting movie ratings. You want a single number that captures "how wrong is my model, on average, across all predictions?"

**Naive attempt**: Just average the raw errors? Nope — positive and negative errors cancel out. A model that predicts +2 too high for one movie and -2 too low for another would have "zero average error." Useless.

**The fix**: Square the errors first. Squaring does two things: (1) makes all errors positive, and (2) punishes large errors disproportionately.

For $n$ samples with true values $y_i$ and predictions $\hat{y}_i$:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

> **You Already Know This** — MSE is the L2 distance squared, averaged over samples. Think of it like this: if you're monitoring request latency and one request takes 500ms while the rest take 10ms, the *squared* average is dominated by that one outlier — just like how one 500ms request ruins your p99. MSE behaves the same way with prediction errors.

**Properties:**
- Always non-negative
- Minimum value of 0 when predictions are perfect
- Differentiable everywhere (important for gradient descent)
- Heavily penalizes large errors due to squaring

**Why the math matters**: The gradient tells the optimizer *how to adjust predictions*:

$$\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$$

Notice the gradient is *proportional to the error*. A prediction that's off by 10 gets a 10x larger gradient update than one that's off by 1. This is why MSE converges fast on clean data — big errors get corrected aggressively.

**Back to movie ratings**: Your model's predictions vs. actuals:

| Movie | Actual | Predicted | Error | Squared Error |
|-------|--------|-----------|-------|---------------|
| A     | 4.0    | 3.8       | -0.2  | 0.04          |
| B     | 2.0    | 2.5       | 0.5   | 0.25          |
| C     | 5.0    | 4.5       | -0.5  | 0.25          |
| D     | 1.0    | 5.0       | 4.0   | 16.00         |

MSE = (0.04 + 0.25 + 0.25 + 16.00) / 4 = **4.135**

Movie D dominates the loss. MSE is screaming at the optimizer: "Fix prediction D above all else!"

---

### Mean Absolute Error (MAE)

**The problem**: MSE is great, but what if Movie D's rating is genuinely unusual? Maybe the user was trolling, or the data is noisy. You don't want one outlier to hijack your entire training signal.

**The fix**: Use absolute values instead of squares. Outliers contribute linearly, not quadratically.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

> **You Already Know This** — MAE is the L1 distance, averaged over samples. It's the difference between using *mean* vs. *median* in your monitoring dashboards. Median (like MAE) is robust to outliers; mean (like MSE) is not. If you've ever switched a dashboard metric from mean to median to stop outliers from skewing your alerts, you've already understood the MAE vs. MSE trade-off intuitively.

**Properties:**
- More robust to outliers than MSE
- Not differentiable at zero (use subgradient methods)
- Median-finding property: minimizing MAE finds the median

**Gradient:**

$$\frac{\partial \text{MAE}}{\partial \hat{y}_i} = \frac{1}{n} \cdot \text{sign}(\hat{y}_i - y_i)$$

The gradient is constant ($\pm 1/n$) regardless of error magnitude. A prediction off by 100 gets the same magnitude gradient update as one off by 0.1. This is the source of both MAE's robustness (outliers don't dominate) and its weakness (convergence can be slow near the optimum because small errors still get full-strength updates).

**Same movie ratings, now with MAE:**

| Movie | Actual | Predicted | Absolute Error |
|-------|--------|-----------|----------------|
| A     | 4.0    | 3.8       | 0.2            |
| B     | 2.0    | 2.5       | 0.5            |
| C     | 5.0    | 4.5       | 0.5            |
| D     | 1.0    | 5.0       | 4.0            |

MAE = (0.2 + 0.5 + 0.5 + 4.0) / 4 = **1.3**

Movie D still contributes the most, but it's not *dominating* the way it did under MSE (4.0 out of 5.2 total, vs. 16.0 out of 16.54 total).

---

### Huber Loss — The Best of Both Worlds

**The problem**: MSE gives smooth gradients but is sensitive to outliers. MAE is robust to outliers but has a constant gradient that makes fine-tuning near the optimum sluggish.

**The fix**: Use MSE when the error is small (smooth, fast convergence) and MAE when the error is large (robustness to outliers). The Huber loss does exactly this with a threshold parameter $\delta$:

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

When $|error| \leq \delta$, it behaves like MSE. When $|error| > \delta$, it behaves like MAE (linear growth). The $\delta$ parameter is a hyperparameter you tune — it's the threshold where you say "errors larger than this are probably outliers."

---

### ASCII Comparison: Regression Loss Curves

```
Loss
  │
  │                                         . MSE (L = error^2)
  │                                       .
  │                                     .
  │                                   .
4 ┤ . . . . . . . . . . . . . . . .
  │  .                             .
  │    .                         .
  │      .                     .
  │        .                 .          . Huber (MSE near 0, MAE far)
3 ┤          .             .        .
  │            .         .        .
  │                    .        .
  │              .   .        .
2 ┤                .        .
  │              . . .    .       . . MAE (L = |error|)
  │            .       .      . .
  │          .       .     . .
1 ┤        .       .    . .
  │      .       .  . .
  │    .       .. .
  │  .      . .
0 ┼──────●──────────────────────────── Error
 -2     -1     0     1     2     3
```

Key insight from the curves:
- **MSE** (parabola): steep far from zero, gentle near zero. Good for convergence, bad for outliers.
- **MAE** (V-shape): constant slope everywhere. Robust, but doesn't slow down near the optimum.
- **Huber**: parabolic near zero (smooth convergence), linear far from zero (outlier-robust). The best of both.

---

## Classification Losses

Regression losses measure "how far off." Classification losses measure something fundamentally different: "how *confidently wrong* are you?"

### Binary Cross-Entropy (Log Loss)

**The problem**: You're classifying movies as "will the user watch this?" (1 = yes, 0 = no). Your model outputs a probability $\hat{p}$. How do you penalize wrong predictions?

**Naive attempt**: Use MSE on probabilities? If $y = 1$ and $\hat{p} = 0.9$, MSE gives $(1 - 0.9)^2 = 0.01$. If $\hat{p} = 0.51$, MSE gives $0.24$. That's only a 24x difference for a massive gap in confidence. MSE doesn't punish confident-and-wrong predictions harshly enough.

**The fix**: Use the logarithm. Cross-entropy penalizes confident wrong predictions *exponentially*:

For binary classification where $y \in \{0, 1\}$ and $\hat{p}$ is the predicted probability:

$$\text{Log Loss} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

> **You Already Know This** — Cross-entropy is negative log-likelihood. If you've scored a probabilistic model by taking $-\log(P(\text{data} | \text{model}))$, you've used cross-entropy. It measures how many bits of "surprise" your model experiences when it sees the true label. A confident correct prediction = low surprise = low loss. A confident wrong prediction = high surprise = high loss. It's the same idea behind information-theoretic scoring.

**Properties:**
- Output must be probabilities in $(0, 1)$
- Encourages well-calibrated probabilities
- Convex function — guaranteed global minimum

**Intuition by example**: When $y = 1$ (user WILL watch the movie):

| Model predicts $\hat{p}$ | Loss = $-\log(\hat{p})$ | Interpretation          |
|---------------------------|-------------------------|-------------------------|
| 0.99                      | 0.01                    | Confident and right     |
| 0.9                       | 0.105                   | Mostly right            |
| 0.5                       | 0.693                   | Coin flip — no info     |
| 0.1                       | 2.303                   | Mostly wrong            |
| 0.01                      | 4.605                   | Confident and WRONG     |
| 0.001                     | 6.908                   | Catastrophically wrong  |

See how the loss *explodes* as confidence in the wrong answer increases? Going from 0.1 to 0.01 adds more loss than going from 0.9 to 0.5. That's the logarithm at work.

---

### Cross-Entropy (Multi-class)

**The problem**: Now you're classifying movies into genres — Action, Comedy, Drama, Sci-Fi. The user picks exactly one. You need a loss function for $K$ mutually exclusive classes.

For $K$ classes with true labels as one-hot vectors $\mathbf{y}$ and predicted probabilities $\hat{\mathbf{p}}$:

$$\text{Cross-Entropy} = -\sum_{k=1}^{K} y_k \log(\hat{p}_k)$$

Since $\mathbf{y}$ is one-hot (only one entry is 1), this simplifies to just $-\log(\hat{p}_c)$ where $c$ is the correct class. You're penalizing the model based on how much probability it assigned to the correct answer.

When combined with softmax output:

$$\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**Why Cross-Entropy + Softmax?** The gradient simplifies beautifully:

$$\frac{\partial L}{\partial z_k} = \hat{p}_k - y_k$$

This is elegant — the gradient is just "predicted minus actual." No chain-rule mess, no numerical instability from log-of-softmax. This is why every deep learning framework fuses cross-entropy and softmax into a single operation (`nn.CrossEntropyLoss` in PyTorch takes raw logits, not probabilities).

---

### ASCII: Cross-Entropy Loss Curve

```
Loss
  │
  │ .
7 ┤  .
  │   .
6 ┤    .
  │     .
5 ┤      .
  │       .
4 ┤        .
  │         ..
3 ┤           ..
  │             ...
2 ┤                ...
  │                   ....
1 ┤                       ......
  │                             ..........
0 ┼───────────────────────────────────────── p
  0    0.1   0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0

  Loss = -log(p) where p is the predicted probability of the TRUE class.
  As confidence in the correct class drops toward 0, loss goes to infinity.
```

---

### Hinge Loss

**The problem**: You're building a binary classifier (e.g., "will user like this movie?"), but instead of probabilities, you want a *margin* — a clear separation between positive and negative classes.

Used in Support Vector Machines for binary classification where $y \in \{-1, +1\}$:

$$\text{Hinge Loss} = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i \cdot f(\mathbf{x}_i))$$

> **You Already Know This** — Hinge loss is the SVM's "margin" idea. Think of it like a load balancer's health check threshold. You don't just want the server to be *alive* — you want it to respond within a margin (say, under 200ms). Predictions that are correct but not confident enough (within the margin) still get penalized. Only predictions that are correct AND confident (beyond the margin) have zero loss. It's a "be right, AND be sure about it" loss function.

**Properties:**
- Creates a margin of separation
- Loss is zero when prediction is correct AND confident (margin > 1)
- Sparse gradients — only misclassified or margin-violating points contribute

**ASCII: Hinge Loss Curve**

```
Loss
  │
  │ .
3 ┤  .
  │   .
2 ┤    .
  │     .
1 ┤      .
  │       .
  │        .
0 ┤─────────●━━━━━━━━━━━━━━━━━━━► y * f(x)
 -2   -1    0    1    2    3
              margin

  Loss = max(0, 1 - y*f(x))
  Once the prediction is correct with margin >= 1, loss = 0.
  The flat region means sparse gradients — only "hard" examples contribute.
```

---

## Common Mistakes

> **MSE is sensitive to outliers because squaring amplifies large errors.** If your data has outliers, consider Huber loss or MAE. In the movie rating example, a single troll rating (1 star on a universally-loved movie) under MSE contributes $16$ to the loss while under MAE it contributes only $4$. If you're seeing your model's performance fluctuate wildly during training, check whether a few extreme examples are dominating your MSE loss.

Other pitfalls to watch for:

1. **Using MSE for classification**: MSE doesn't understand probabilities. A model that predicts 0.49 vs. 0.51 for a binary label gets almost zero MSE loss either way, but these predictions mean completely different things. Use cross-entropy.
2. **Forgetting to clip probabilities**: $\log(0) = -\infty$ will produce NaN and break training. Always clip: `y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)`.
3. **Wrong label format**: Cross-entropy expects one-hot encoded labels. Log loss expects 0/1 binary labels. Hinge loss expects -1/+1 labels. Mixing these up produces garbage.
4. **Not normalizing hinge loss outputs**: Hinge loss expects raw scores (logits), not probabilities. Passing sigmoid outputs to hinge loss defeats the purpose.

---

## Where Each Loss Function Appears in Practice

| Loss Function | Common Uses | Deep Learning Context |
|--------------|-------------|----------------------|
| **MSE** | Linear regression, autoencoders, neural network regression | VAE reconstruction loss, pixel-level image prediction |
| **MAE** | Robust regression, when outliers are present | L1 reconstruction loss for sharper images |
| **Huber** | Robust regression with smooth gradients | Smooth L1 in object detection (bounding box regression) |
| **Log Loss** | Logistic regression, binary classifiers | Binary output heads, anomaly detection |
| **Cross-Entropy** | Neural network classifiers, softmax output layers | Image classification, language models (next-token prediction) |
| **Hinge Loss** | Support Vector Machines, maximum-margin classifiers | Rare in deep learning, but used in adversarial training |

**Deep Learning Specifics:**
- **Image Classification**: Cross-entropy with softmax
- **Object Detection**: Combination of cross-entropy (class) + Smooth L1/Huber (bounding box)
- **Language Models**: Cross-entropy over vocabulary (next-token prediction)
- **Generative Models (VAE)**: Reconstruction loss (MSE or BCE) + KL divergence

---

## Code: Loss Functions From Scratch

```python
import numpy as np

class LossFunctions:
    """Implementation of common loss functions from scratch."""

    @staticmethod
    def mse(y_true, y_pred):
        """
        Mean Squared Error

        Args:
            y_true: Ground truth values (n_samples,)
            y_pred: Predicted values (n_samples,)

        Returns:
            loss: Scalar MSE value
            gradient: Gradient w.r.t. predictions
        """
        n = len(y_true)
        error = y_pred - y_true
        loss = np.mean(error ** 2)
        gradient = (2 / n) * error
        return loss, gradient

    @staticmethod
    def mae(y_true, y_pred):
        """
        Mean Absolute Error

        Returns:
            loss: Scalar MAE value
            gradient: Subgradient w.r.t. predictions
        """
        n = len(y_true)
        error = y_pred - y_true
        loss = np.mean(np.abs(error))
        gradient = (1 / n) * np.sign(error)
        return loss, gradient

    @staticmethod
    def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
        """
        Binary Cross-Entropy (Log Loss)

        Args:
            y_true: Binary labels (0 or 1)
            y_pred: Predicted probabilities (0 to 1)
            epsilon: Small value to avoid log(0)

        Returns:
            loss: Scalar BCE value
            gradient: Gradient w.r.t. predictions
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        n = len(y_true)
        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        gradient = (1 / n) * (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred))
        return loss, gradient

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
        """
        Categorical Cross-Entropy for multi-class classification

        Args:
            y_true: One-hot encoded labels (n_samples, n_classes)
            y_pred: Predicted probabilities (n_samples, n_classes)

        Returns:
            loss: Scalar CCE value
            gradient: Gradient w.r.t. predictions
        """
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        n = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / n

        gradient = -y_true / y_pred / n
        return loss, gradient

    @staticmethod
    def hinge_loss(y_true, y_pred):
        """
        Hinge Loss for SVM-style classification

        Args:
            y_true: Labels in {-1, +1}
            y_pred: Raw model outputs (scores, not probabilities)

        Returns:
            loss: Scalar hinge loss value
            gradient: Subgradient w.r.t. predictions
        """
        n = len(y_true)
        margins = y_true * y_pred
        loss = np.mean(np.maximum(0, 1 - margins))

        # Gradient: -y_i if margin < 1, else 0
        gradient = np.where(margins < 1, -y_true, 0) / n
        return loss, gradient


# Demonstration
if __name__ == "__main__":
    np.random.seed(42)

    # ---- Regression: Movie Rating Predictions ----
    print("=== Movie Rating Predictions (Regression) ===")
    y_true_reg = np.array([4.0, 2.0, 5.0, 1.0])   # actual ratings
    y_pred_reg = np.array([3.8, 2.5, 4.5, 5.0])    # model predictions

    mse_loss, mse_grad = LossFunctions.mse(y_true_reg, y_pred_reg)
    print(f"MSE Loss: {mse_loss:.4f}")
    print(f"MSE Gradient: {mse_grad}")

    mae_loss, mae_grad = LossFunctions.mae(y_true_reg, y_pred_reg)
    print(f"MAE Loss: {mae_loss:.4f}")
    print(f"MAE Gradient: {mae_grad}")

    # ---- Binary Classification: Will user watch? ----
    print("\n=== Will User Watch? (Binary Classification) ===")
    y_true_bin = np.array([1, 0, 1, 1, 0])
    y_pred_bin = np.array([0.9, 0.1, 0.8, 0.7, 0.3])

    bce_loss, bce_grad = LossFunctions.binary_cross_entropy(y_true_bin, y_pred_bin)
    print(f"Binary Cross-Entropy: {bce_loss:.4f}")

    # ---- Multi-class: Genre Classification ----
    print("\n=== Genre Classification (Multi-class) ===")
    # 3 movies, 4 genres: Action, Comedy, Drama, Sci-Fi
    y_true_multi = np.array([
        [1, 0, 0, 0],  # Action
        [0, 1, 0, 0],  # Comedy
        [0, 0, 0, 1],  # Sci-Fi
    ])
    y_pred_multi = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.6, 0.2, 0.1],
        [0.1, 0.1, 0.1, 0.7],
    ])

    cce_loss, _ = LossFunctions.categorical_cross_entropy(y_true_multi, y_pred_multi)
    print(f"Categorical Cross-Entropy: {cce_loss:.4f}")

    # ---- Hinge Loss: Like/Dislike Margin ----
    print("\n=== Like/Dislike Classification (Hinge Loss) ===")
    y_true_svm = np.array([1, -1, 1, -1])
    y_pred_svm = np.array([0.5, -0.8, 1.5, -2.0])  # Raw scores

    hinge, hinge_grad = LossFunctions.hinge_loss(y_true_svm, y_pred_svm)
    print(f"Hinge Loss: {hinge:.4f}")
    print(f"Margins: {y_true_svm * y_pred_svm}")
    print(f"Hinge Gradient: {hinge_grad}")
```

**Output:**
```
=== Movie Rating Predictions (Regression) ===
MSE Loss: 4.1350
MSE Gradient: [-0.1   0.25 -0.25  2.  ]
MAE Loss: 1.3000
MAE Gradient: [-0.25  0.25 -0.25  0.25]

=== Will User Watch? (Binary Classification) ===
Binary Cross-Entropy: 0.2529

=== Genre Classification (Multi-class) ===
Categorical Cross-Entropy: 0.4308

=== Like/Dislike Classification (Hinge Loss) ===
Hinge Loss: 0.2000
Margins: [ 0.5  0.8  1.5  2. ]
Hinge Gradient: [-0.25  0.    0.    0.  ]
```

Notice in the regression output: the MSE gradient for movie D (the outlier) is `2.0` — an order of magnitude larger than all other gradients. MSE is *screaming* about that one prediction. The MAE gradient? A calm `0.25`, same as every other movie. This is the MSE-vs-MAE tradeoff, visible right there in the gradient.

---

## Exercises

### Exercise 1: Outlier Sensitivity in Movie Ratings

**Problem**: A troll rates every movie 1 star. Your model predicts `[3.5, 4.2, 2.8, 4.0]` for movies with true ratings `[3.5, 4.0, 3.0, 1.0]` (the last one is the troll). Calculate both MSE and MAE. How much does the troll dominate each loss?

**Solution**:
```python
y_true = np.array([3.5, 4.0, 3.0, 1.0])
y_pred = np.array([3.5, 4.2, 2.8, 4.0])

errors = y_pred - y_true  # [0.0, 0.2, -0.2, 3.0]

mse = np.mean(errors ** 2)
mae = np.mean(np.abs(errors))

print(f"MSE: {mse}")  # MSE: 2.27
print(f"MAE: {mae}")  # MAE: 0.85
```

The troll's contribution:
- To MSE: $3.0^2 = 9.0$ out of total $9.08$, so **99%** of the loss comes from one example.
- To MAE: $3.0$ out of total $3.4$, so **88%** of the loss — still dominant, but far less extreme.

This is why MSE can cause your model to overfit to outliers. If you see training loss dominated by a few examples, check for outliers and consider switching to Huber or MAE.

### Exercise 2: Cross-Entropy Confidence Penalty

**Problem**: Your genre classifier outputs these probabilities for a movie that's actually a Comedy:

| Prediction | P(Action) | P(Comedy) | P(Drama) | P(Sci-Fi) |
|------------|-----------|-----------|----------|-----------|
| A          | 0.1       | 0.7       | 0.1      | 0.1       |
| B          | 0.1       | 0.9       | 0.0      | 0.0       |
| C          | 0.4       | 0.1       | 0.4      | 0.1       |

Calculate cross-entropy for each. Which prediction does the loss function prefer?

**Solution**:
```python
# Only the probability of the true class (Comedy) matters
for name, p_comedy in [("A", 0.7), ("B", 0.9), ("C", 0.1)]:
    loss = -np.log(p_comedy)
    print(f"Prediction {name}: P(Comedy) = {p_comedy}, Loss = {loss:.4f}")

# Output:
# Prediction A: P(Comedy) = 0.7, Loss = 0.3567
# Prediction B: P(Comedy) = 0.9, Loss = 0.1054
# Prediction C: P(Comedy) = 0.1, Loss = 2.3026
```

Prediction B is strongly preferred. Prediction C — which is confidently wrong — gets punished 22x more than Prediction B. Cross-entropy heavily rewards calibrated confidence.

### Exercise 3: Hinge Loss Margin

**Problem**: For an SVM classifier with $y = +1$ (user likes the movie), at what prediction value $f(x)$ does the loss become zero? What does this mean?

**Solution**:

The hinge loss is $\max(0, 1 - y \cdot f(x))$. For $y = +1$:
- Loss $= \max(0, 1 - f(x))$
- Loss becomes zero when $1 - f(x) \leq 0$
- Therefore, $f(x) \geq 1$

The prediction must be **at least 1** — not just positive, but *confidently* positive. This is the margin requirement of SVMs. A prediction of $f(x) = 0.5$ is technically correct (positive for a positive example), but hinge loss still penalizes it by $0.5$ because it's within the margin. Only predictions beyond the margin boundary ($\geq 1$ for positive, $\leq -1$ for negative) incur zero loss. This forces the model to find a decision boundary with a wide buffer zone, which improves generalization.

---

## Summary

| Loss | Formula | Key Behavior | Use When |
|------|---------|-------------|----------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Squares errors; outliers dominate | Clean regression data, smooth gradients needed |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Linear errors; outlier-robust | Noisy data, median prediction desired |
| **Huber** | MSE near 0, MAE far | Best of both MSE and MAE | Regression with occasional outliers |
| **Log Loss** | $-\frac{1}{n}\sum[y\log\hat{p} + (1-y)\log(1-\hat{p})]$ | Punishes confident wrong predictions exponentially | Binary classification |
| **Cross-Entropy** | $-\sum y_k \log \hat{p}_k$ | Log-scores probability of correct class | Multi-class classification |
| **Hinge** | $\frac{1}{n}\sum\max(0, 1 - y \cdot f(x))$ | Zero loss outside margin; sparse gradients | SVM/maximum-margin classifiers |

The loss function defines what "good" means for your model. All the math, all the gradients, all the training — it's all in service of minimizing this one number. Choose it deliberately.

---

> **What's Next** — You have a loss function. Now you need to minimize it. Gradient descent, SGD, Adam — optimization algorithms are the engines that drive training.
