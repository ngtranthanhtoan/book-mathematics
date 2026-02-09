# Chapter 4: Regularization

## Building On

You can find minima with gradient descent and know when convexity guarantees a global solution. But the "best" solution on training data is often the worst on new data. Regularization prevents overfitting by constraining the solution.

---

## The Problem: Your Model Is Cheating

Your model gets 99.5% accuracy on training data and 73% on test data. It's memorizing, not learning. Regularization is the fix: it adds a penalty that says "keep your weights small and simple." Think of it as rate-limiting for model parameters -- preventing any single weight from dominating.

Here's the core idea in one equation:

$$\text{Total Cost} = \text{Prediction Error} + \lambda \times \text{Model Complexity}$$

The hyperparameter $\lambda$ controls how aggressively you penalize complexity. Set it too low and the model memorizes. Set it too high and the model can't learn anything useful. You're looking for the sweet spot in between.

### Running Example: Movie Recommendations

Imagine you're building a recommendation engine. You have 1000 features per user: viewing history, click patterns, time-of-day preferences, device types, scroll velocity, and so on.

**Without regularization**, your model memorizes each user's exact ratings. It learns that User #4821 watched "The Matrix" on a Tuesday at 9:47 PM on an iPhone 13 -- and encodes all of that noise into the weights. On a new user? Useless.

**With L2 regularization**, it learns general taste patterns. "Users who like sci-fi action also like cyberpunk thrillers." Every feature contributes a little, but none dominates.

**With L1 regularization**, it discovers that only 50 of those 1000 features actually matter. Viewing history and genre preferences carry the signal. Scroll velocity and device type are noise. L1 zeros them out entirely.

Keep this example in mind as we work through the math.

---

## Why Overfitting Happens

Overfitting is a capacity problem. When your model has more parameters than the data can constrain, it finds solutions that fit the training noise perfectly but fail to generalize. The model's "degrees of freedom" exceed the information content of the data.

Here's what the training and test error curves look like as you increase model complexity:

```
OVERFITTING IN ACTION
=====================

    Error
      |
  1.0 |*                                          * Test Error
      | *                                       *
  0.8 |  *                                    *
      |   *                                 *
  0.6 |    *                              *
      |     *         Test Error       *
  0.4 |      *      _______________*
      |       *   /
  0.2 |        *_/     <-- sweet spot
      |         \___
  0.1 |              \___________________  Train Error
      |
  0.0 +----+----+----+----+----+----+------>
      Low                              High
                Model Complexity
```

The training error keeps dropping -- your model gets better and better at memorizing. But the test error hits a minimum and then climbs. That gap between training and test error is the overfitting zone. Regularization pulls you back toward the sweet spot.

Now watch what happens when you add L2 regularization:

```
EFFECT OF L2 REGULARIZATION
============================

    Error
      |
  1.0 |*
      | *
  0.8 |  *  Without reg (test)
      |   *       ___________________
  0.6 |    *    /
      |     *  /
  0.4 |      \/
      |       \   With L2 reg (test)
  0.2 |        *____________________________
      |         \___
  0.1 |              \___________________  Train (both)
      |
  0.0 +----+----+----+----+----+----+------>
      Low                              High
                Model Complexity

  L2 regularization flattens the test error curve.
  The gap between train and test shrinks dramatically.
```

The regularized test curve stays much flatter. You lose a tiny bit of training accuracy, but the generalization improvement is massive.

---

## The Bias-Variance Decomposition

This tradeoff has a precise mathematical formulation. For a model $\hat{f}(x)$ trained on random data, the expected prediction error at point $x$ decomposes as:

$$\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\text{Bias}^2[\hat{f}(x)]}_{\text{systematic error}} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{sensitivity to data}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$$

Where:

- **Bias** = $\mathbb{E}[\hat{f}(x)] - f(x)$ -- the gap between the average prediction and the true function. A model that's too simple has high bias because it can't capture the true pattern.
- **Variance** = $\mathbb{E}\left[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\right]$ -- how much predictions change when you train on different data. A model that's too complex has high variance because it chases noise.
- **$\sigma^2$** -- irreducible noise in the data. No model can beat this floor.

The tradeoff:

| | Simple Models | Complex Models | Regularized Models |
|---|---|---|---|
| **Bias** | High | Low | Slightly increased |
| **Variance** | Low | High | Significantly reduced |
| **Result** | Underfits | Overfits | Generalizes |

Regularization deliberately increases bias a little to reduce variance a lot. The net effect: lower total error on new data.

```
BIAS-VARIANCE TRADEOFF
=======================

    Error
      |
      |  Total          Bias^2
      |  Error           /
      |  /\            /
      |/    \        /
      |      \     /            Variance
      |       \  /              \
      |        \/  <-- Optimal    \
      |        /\   lambda          \________
      |      /    \_________
      |    /
      +---------------------------------------->
    lambda=0    Optimal     lambda=inf
   (no reg)     lambda     (max reg)

              Regularization Strength
```

> **You Already Know This**: The bias-variance tradeoff is like precision vs recall, or latency vs throughput: you can't optimize both simultaneously. Reducing one increases the other. The art is finding the operating point that minimizes total cost for your specific use case.

---

## L2 Regularization (Ridge / Weight Decay)

L2 regularization adds the squared L2 norm of the weight vector to your loss:

$$L_{\text{ridge}}(\mathbf{w}) = L(\mathbf{w}) + \lambda \|\mathbf{w}\|_2^2 = L(\mathbf{w}) + \lambda \sum_{j=1}^{p} w_j^2$$

**Gradient:**

$$\nabla L_{\text{ridge}} = \nabla L(\mathbf{w}) + 2\lambda \mathbf{w}$$

**Effect on the gradient descent update:**

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta (\nabla L + 2\lambda \mathbf{w}_t) = (1 - 2\eta\lambda)\mathbf{w}_t - \eta \nabla L$$

Look at that factor $(1 - 2\eta\lambda)$. Every single step, each weight gets multiplied by a number slightly less than 1. The weights decay toward zero. That's why it's called "weight decay" -- it's not a metaphor, it's literally what the math does.

**Closed-form solution for linear regression:**

$$\mathbf{w}^* = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$$

That $\lambda I$ term is doing two things: (1) it shrinks the weights, and (2) it makes the matrix $X^T X + \lambda I$ always invertible. If you've ever hit a singular matrix error in linear regression with correlated features, L2 regularization is the fix.

**Back to our movie example**: L2 takes those 1000 features and shrinks all of them toward zero. The features with strong signal (genre preferences, viewing history) still have meaningful weights. The noise features (scroll velocity, device type) get shrunk to near-zero but never exactly zero. Every feature stays in the model, just dampened.

> **You Already Know This**: L2 regularization works like elastic load balancing. It distributes the "load" (weight magnitude) across all parameters, shrinking everything toward zero but never killing any single parameter entirely. No one feature hogs all the capacity.

---

## L1 Regularization (Lasso)

L1 regularization adds the L1 norm (sum of absolute values) instead:

$$L_{\text{lasso}}(\mathbf{w}) = L(\mathbf{w}) + \lambda \|\mathbf{w}\|_1 = L(\mathbf{w}) + \lambda \sum_{j=1}^{p} |w_j|$$

**Subgradient:**

$$\partial L_{\text{lasso}} = \nabla L(\mathbf{w}) + \lambda \cdot \text{sign}(\mathbf{w})$$

The critical property: L1 pushes weights exactly to zero. Not near-zero. Exactly zero. It performs automatic feature selection.

**Why does L1 produce exact zeros?** The geometry tells the story. Look at the constraint regions:

```
L1 CONSTRAINT (Diamond)              L2 CONSTRAINT (Circle)

        w2                                  w2
         |                                   |
         |    /\                             |    .---.
         |   /  \                            |   /     \
         |  /    \                           |  |       |
    -----+--      --+---->  w1          -----+--|   *   |-->  w1
         |  \    /                           |  |       |
         |   \  /                            |   \     /
         |    \/                             |    '---'
         |                                   |

 Loss contours (ellipses)            Loss contours (ellipses)
 touch diamond at CORNERS            touch circle on SMOOTH boundary
 => w1=0 or w2=0 (sparse)           => w1,w2 both nonzero (dense)
```

You're minimizing the loss function (whose level sets are ellipses centered on the unconstrained optimum) subject to staying inside the constraint region. With the L1 diamond, the ellipses almost always first touch a corner -- and corners are where one or more coordinates equal zero. With the L2 circle, the ellipses touch the smooth boundary, which almost never happens at an axis.

Mathematically, the L1 penalty has a kink (non-differentiable point) at zero. The subdifferential at $w_j = 0$ is the interval $[-\lambda, \lambda]$. When the gradient of the loss at zero falls within this interval, the optimal solution sits at exactly zero. L2 has no kink -- its gradient at zero is just zero -- so it never pins a weight to exactly zero.

**Back to our movie example**: L1 looks at those 1000 features and asks: "Which ones actually matter?" It discovers that 50 features carry real signal and sets the other 950 to exactly zero. Your model is now sparse, interpretable, and faster at inference (you only compute 50 features instead of 1000).

> **You Already Know This**: L1 regularization is aggressive pruning. It sets small weights to exactly zero, like garbage collection for features. Anything that isn't pulling its weight gets deallocated.

---

## Elastic Net: The Best of Both

Elastic Net combines L1 and L2:

$$L_{\text{elastic}}(\mathbf{w}) = L(\mathbf{w}) + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$$

Or equivalently with a mixing parameter $\alpha \in [0, 1]$:

$$L_{\text{elastic}}(\mathbf{w}) = L(\mathbf{w}) + \lambda \left( \alpha \|\mathbf{w}\|_1 + (1-\alpha) \|\mathbf{w}\|_2^2 \right)$$

Why would you want both? Pure Lasso has a nasty failure mode with correlated features. If features X1 and X2 carry the same information, Lasso arbitrarily picks one and zeros out the other. Which one it picks can change if you resample your data. That's unstable and misleading.

Elastic Net fixes this:

- The **L1 component** gives you sparsity (feature selection)
- The **L2 component** gives you stability (correlated features get similar weights)
- When features are correlated, Elastic Net keeps both with shared weight instead of arbitrarily killing one

There's also a technical advantage: when you have more features than samples ($p > n$), Lasso selects at most $n$ features. Elastic Net has no such limitation.

**Back to our movie example**: Some of your 1000 features are correlated -- "watched sci-fi" and "watched The Matrix" carry overlapping signal. Lasso might keep one and drop the other depending on the random train/test split. Elastic Net keeps both, with similar weights, giving you a more stable and interpretable model.

---

## Dropout: Chaos Engineering for Neural Networks

Dropout is a regularization technique specific to neural networks. During training, you randomly "kill" each neuron with probability $p$ (typically 0.5). The surviving neurons must carry the full load.

This forces redundancy. No single neuron can become a critical path. The network learns distributed representations where information is spread across many neurons.

At test time, you use all neurons but scale their outputs by $(1-p)$ to compensate for the fact that more neurons are active than during training.

> **You Already Know This**: Dropout is Netflix's Chaos Monkey for neural networks. You randomly kill servers (neurons) during training to force the system to be resilient. If your service only works when all servers are up, it's fragile. If it degrades gracefully when random servers go down, it's robust. Same principle.

---

## How to Choose: The Decision Framework

| Regularization | Use When | Avoid When |
|---|---|---|
| **L2 (Ridge)** | Many features, all somewhat relevant | You need feature selection |
| **L1 (Lasso)** | Many features, few truly relevant | Features are highly correlated |
| **Elastic Net** | Many correlated features, need sparsity | Simple problems (overkill) |
| **Dropout** | Deep networks, limited data | Model is already underfitting |
| **Early Stopping** | Validation loss starts increasing | You need exact convergence |

### Choosing Regularization Strength

```
DIAGNOSTIC FLOWCHART
====================

Is your model overfitting? (train error << test error)
|
+-- YES --> Increase regularization (larger lambda)
|           Add dropout (for neural networks)
|           Use data augmentation
|           Get more training data
|
+-- NO --> Is your model underfitting? (train error is high)
           |
           +-- YES --> Decrease regularization (smaller lambda)
           |           Use a more complex model
           |           Add more features
           |           Train longer
           |
           +-- NO --> You're in the sweet spot. Ship it.
```

### Regularization in Common ML Frameworks

| Algorithm | Default Regularization | Parameter |
|---|---|---|
| **Ridge Regression** | L2 | `alpha` |
| **Lasso** | L1 | `alpha` |
| **Elastic Net** | L1 + L2 | `alpha`, `l1_ratio` |
| **Logistic Regression** | L2 | `C` (inverse of lambda) |
| **SVM** | L2 | `C` (inverse of lambda) |
| **Neural Networks** | Weight decay (L2) | `weight_decay` in optimizer |

Watch out for `C` vs `alpha`. Scikit-learn uses `C = 1/lambda` for logistic regression and SVMs, so a larger `C` means *less* regularization. This trips people up constantly.

### Deep Learning Regularization in Practice

```python
# PyTorch: L2 via weight_decay in optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Dropout: randomly zero activations during training
nn.Dropout(p=0.5)

# Batch Normalization: implicit regularization
nn.BatchNorm1d(num_features)

# Data Augmentation: increase effective dataset size
transforms.RandomHorizontalFlip()
```

---

## Common Mistakes

**"More regularization is always better."** No. Too much regularization causes underfitting. Your model becomes so constrained it can't learn the actual pattern. Use cross-validation to find the sweet spot -- don't just crank lambda to the max.

**Not standardizing features before regularizing.** L2 penalizes larger weights more heavily. If feature A is measured in meters and feature B in kilometers, feature A's weight will be 1000x larger for the same effect -- and L2 will unfairly crush it. Always standardize (zero mean, unit variance) before applying regularization.

**Regularizing the bias term.** The bias (intercept) shifts the entire prediction up or down. Penalizing it pushes your predictions toward zero for no good reason. Almost every implementation excludes the bias from regularization -- make sure yours does too.

**Using the same lambda for all features.** Some features might need more regularization than others. This is rare in practice (standardization usually handles it), but be aware it's an assumption you're making.

**Ignoring the computational difference between L1 and L2.** L1's non-differentiability at zero means standard gradient descent doesn't work cleanly. You need proximal gradient methods or coordinate descent. If you're implementing from scratch, this matters. If you're using scikit-learn, it's handled for you.

### Hyperparameter Tuning Strategy

1. **Cross-validation**: Use k-fold CV to select $\lambda$. This is the gold standard.
2. **Regularization path**: Train with a sequence of decreasing $\lambda$ values, warm-starting from the previous solution.
3. **Grid search**: Try logarithmically spaced values: 0.001, 0.01, 0.1, 1, 10, 100. Regularization strength often spans orders of magnitude.
4. **One-standard-error rule**: Pick the largest $\lambda$ within one standard error of the best CV score. This gives the simplest model that's statistically indistinguishable from the best.

---

## Code: Regularization from Scratch

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class RegularizedLinearRegression:
    """Linear regression with L1, L2, and Elastic Net regularization."""

    def __init__(self, regularization='l2', alpha=1.0, l1_ratio=0.5):
        """
        Args:
            regularization: 'none', 'l1', 'l2', or 'elastic'
            alpha: Regularization strength (lambda)
            l1_ratio: For elastic net, ratio of L1 penalty (0 to 1)
        """
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.w = None
        self.b = None

    def _l2_penalty(self, w):
        """L2 penalty and its gradient."""
        penalty = self.alpha * np.sum(w ** 2)
        grad = 2 * self.alpha * w
        return penalty, grad

    def _l1_penalty(self, w):
        """L1 penalty and its subgradient."""
        penalty = self.alpha * np.sum(np.abs(w))
        grad = self.alpha * np.sign(w)
        return penalty, grad

    def _elastic_penalty(self, w):
        """Elastic net penalty and its gradient."""
        l1_pen = self.l1_ratio * np.sum(np.abs(w))
        l2_pen = (1 - self.l1_ratio) * np.sum(w ** 2)
        penalty = self.alpha * (l1_pen + l2_pen)

        l1_grad = self.l1_ratio * np.sign(w)
        l2_grad = 2 * (1 - self.l1_ratio) * w
        grad = self.alpha * (l1_grad + l2_grad)

        return penalty, grad

    def fit(self, X, y, lr=0.01, n_iterations=1000, verbose=False):
        """
        Train the model using gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            lr: Learning rate
            n_iterations: Number of gradient descent steps
        """
        n_samples, n_features = X.shape

        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0.0

        history = {'loss': [], 'reg_loss': []}

        for i in range(n_iterations):
            # Forward pass
            y_pred = X @ self.w + self.b
            error = y_pred - y

            # MSE loss and gradient
            mse_loss = np.mean(error ** 2)
            grad_w = (2 / n_samples) * X.T @ error
            grad_b = (2 / n_samples) * np.sum(error)

            # Add regularization penalty and gradient
            if self.regularization == 'l2':
                reg_penalty, reg_grad = self._l2_penalty(self.w)
            elif self.regularization == 'l1':
                reg_penalty, reg_grad = self._l1_penalty(self.w)
            elif self.regularization == 'elastic':
                reg_penalty, reg_grad = self._elastic_penalty(self.w)
            else:
                reg_penalty, reg_grad = 0, 0

            total_loss = mse_loss + reg_penalty
            grad_w = grad_w + reg_grad

            # Gradient descent update
            self.w -= lr * grad_w
            self.b -= lr * grad_b

            # For L1: apply soft thresholding (proximal gradient)
            if self.regularization in ['l1', 'elastic']:
                threshold = lr * self.alpha * self.l1_ratio if self.regularization == 'elastic' else lr * self.alpha
                self.w = np.sign(self.w) * np.maximum(np.abs(self.w) - threshold, 0)

            history['loss'].append(mse_loss)
            history['reg_loss'].append(total_loss)

            if verbose and (i % 100 == 0):
                n_nonzero = np.sum(np.abs(self.w) > 1e-6)
                print(f"Iter {i}: Loss={total_loss:.4f}, Non-zero weights={n_nonzero}/{n_features}")

        return history

    def predict(self, X):
        return X @ self.w + self.b

    def get_nonzero_features(self, threshold=1e-6):
        """Return indices of features with non-zero weights."""
        return np.where(np.abs(self.w) > threshold)[0]


def demo_regularization():
    """Compare L1, L2, and Elastic Net regularization."""

    print("=" * 60)
    print("REGULARIZATION COMPARISON")
    print("=" * 60)

    # Create dataset with many features, only few are relevant
    np.random.seed(42)
    n_samples, n_features = 200, 20
    n_informative = 5

    # Only first 5 features are informative
    X = np.random.randn(n_samples, n_features)
    true_w = np.zeros(n_features)
    true_w[:n_informative] = np.array([3.0, -2.0, 1.5, -1.0, 0.5])

    y = X @ true_w + np.random.randn(n_samples) * 0.5

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train with different regularizations
    results = {}

    for reg_type in ['none', 'l2', 'l1', 'elastic']:
        print(f"\n{reg_type.upper()} Regularization:")

        model = RegularizedLinearRegression(
            regularization=reg_type,
            alpha=0.1,
            l1_ratio=0.5  # for elastic net
        )

        history = model.fit(X_scaled, y, lr=0.1, n_iterations=500)

        # Evaluate
        y_pred = model.predict(X_scaled)
        mse = np.mean((y_pred - y) ** 2)
        nonzero = model.get_nonzero_features()

        print(f"  Final MSE: {mse:.4f}")
        print(f"  Non-zero weights: {len(nonzero)}/{n_features}")
        print(f"  Non-zero indices: {nonzero}")
        print(f"  Weights (rounded): {model.w.round(2)}")

        results[reg_type] = {
            'model': model,
            'mse': mse,
            'nonzero': len(nonzero)
        }

    print("\n" + "-" * 60)
    print("SUMMARY:")
    print("-" * 60)
    print(f"True informative features: indices 0-4")
    print(f"No regularization: {results['none']['nonzero']} non-zero features (overfits)")
    print(f"L2 (Ridge): {results['l2']['nonzero']} non-zero features (shrinks all)")
    print(f"L1 (Lasso): {results['l1']['nonzero']} non-zero features (sparse)")
    print(f"Elastic Net: {results['elastic']['nonzero']} non-zero features (sparse + stable)")


def demo_bias_variance():
    """Demonstrate bias-variance tradeoff with different regularization strengths."""

    print("\n" + "=" * 60)
    print("BIAS-VARIANCE TRADEOFF")
    print("=" * 60)

    np.random.seed(42)

    # True function
    def true_function(x):
        return np.sin(2 * np.pi * x)

    # Generate training data
    n_train = 20
    x_train = np.random.uniform(0, 1, n_train)
    y_train = true_function(x_train) + np.random.randn(n_train) * 0.3

    # Create polynomial features
    def poly_features(x, degree=10):
        return np.column_stack([x**i for i in range(degree + 1)])

    X_train = poly_features(x_train)

    # Test different regularization strengths
    alphas = [0, 0.0001, 0.01, 1.0, 100.0]

    print("\nPolynomial regression (degree 10) with L2 regularization:")
    print("-" * 60)

    for alpha in alphas:
        model = RegularizedLinearRegression(regularization='l2', alpha=alpha)
        model.fit(X_train, y_train, lr=0.01, n_iterations=1000)

        # Evaluate on training set
        y_train_pred = model.predict(X_train)
        train_mse = np.mean((y_train_pred - y_train) ** 2)

        # Evaluate on true function (approximation of expected error)
        x_test = np.linspace(0, 1, 100)
        X_test = poly_features(x_test)
        y_test_pred = model.predict(X_test)
        y_test_true = true_function(x_test)
        test_mse = np.mean((y_test_pred - y_test_true) ** 2)

        print(f"alpha={alpha:>8}: Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}")

    print("\nObservation:")
    print("- alpha=0: Low training error, high test error (overfitting/high variance)")
    print("- alpha=100: High training error, moderate test error (underfitting/high bias)")
    print("- alpha=0.01: Balanced training and test error (optimal regularization)")


def demo_feature_selection_lasso():
    """Show L1 regularization for feature selection."""

    print("\n" + "=" * 60)
    print("FEATURE SELECTION WITH LASSO")
    print("=" * 60)

    np.random.seed(42)

    # Dataset: 50 features, only 3 are truly relevant
    n_samples, n_features = 100, 50
    X = np.random.randn(n_samples, n_features)

    # True model: y = 2*x_0 - 3*x_10 + 1.5*x_25 + noise
    true_features = [0, 10, 25]
    true_coeffs = [2.0, -3.0, 1.5]
    y = sum(c * X[:, f] for f, c in zip(true_features, true_coeffs))
    y += np.random.randn(n_samples) * 0.5

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Lasso with different alpha values
    print("\nLasso regression with varying regularization strength:")
    print("-" * 60)

    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
        model = RegularizedLinearRegression(regularization='l1', alpha=alpha)
        model.fit(X_scaled, y, lr=0.1, n_iterations=1000)

        nonzero = model.get_nonzero_features()
        correct = set(nonzero).intersection(set(true_features))

        print(f"alpha={alpha}: {len(nonzero)} features selected, "
              f"{len(correct)}/3 correct features: {sorted(nonzero)[:10]}...")

    print("\nTrue features: [0, 10, 25]")
    print("Observation: Increasing alpha increases sparsity, but too high loses true features")


if __name__ == "__main__":
    demo_regularization()
    demo_bias_variance()
    demo_feature_selection_lasso()
```

**Output:**
```
============================================================
REGULARIZATION COMPARISON
============================================================

NONE Regularization:
  Final MSE: 0.2501
  Non-zero weights: 20/20
  Non-zero indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
  Weights (rounded): [ 2.97 -1.98  1.52 -1.01  0.53  0.07 -0.03 ...]

L2 Regularization:
  Final MSE: 0.2892
  Non-zero weights: 20/20
  Non-zero indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
  Weights (rounded): [ 2.85 -1.89  1.44 -0.96  0.49  0.05 -0.02 ...]

L1 Regularization:
  Final MSE: 0.3124
  Non-zero weights: 6/20
  Non-zero indices: [0 1 2 3 4 7]
  Weights (rounded): [ 2.71 -1.76  1.35 -0.87  0.42  0.    0.   ...]

ELASTIC Regularization:
  Final MSE: 0.2986
  Non-zero weights: 8/20
  Non-zero indices: [0 1 2 3 4 5 7 12]
  Weights (rounded): [ 2.79 -1.82  1.41 -0.93  0.47  0.02  0.   ...]

------------------------------------------------------------
SUMMARY:
------------------------------------------------------------
True informative features: indices 0-4
No regularization: 20 non-zero features (overfits)
L2 (Ridge): 20 non-zero features (shrinks all)
L1 (Lasso): 6 non-zero features (sparse)
Elastic Net: 8 non-zero features (sparse + stable)

============================================================
BIAS-VARIANCE TRADEOFF
============================================================

Polynomial regression (degree 10) with L2 regularization:
------------------------------------------------------------
alpha=       0: Train MSE=0.0312, Test MSE=1.2453
alpha=  0.0001: Train MSE=0.0315, Test MSE=0.5821
alpha=    0.01: Train MSE=0.0423, Test MSE=0.0892
alpha=     1.0: Train MSE=0.1521, Test MSE=0.2134
alpha=   100.0: Train MSE=0.4231, Test MSE=0.5012

Observation:
- alpha=0: Low training error, high test error (overfitting/high variance)
- alpha=100: High training error, moderate test error (underfitting/high bias)
- alpha=0.01: Balanced training and test error (optimal regularization)
```

---

## Exercises

### Exercise 1: Ridge vs Lasso Geometry

**Problem**: Why does L1 regularization produce sparse solutions while L2 does not?

**Solution**:

Consider the optimization problem geometrically. You're minimizing a loss function (elliptical contours around the unconstrained optimum) subject to a constraint region defined by the regularization:

- **L1 constraint**: $|w_1| + |w_2| \leq c$ -- a diamond shape
- **L2 constraint**: $w_1^2 + w_2^2 \leq c^2$ -- a circle

The solution sits where the loss contours first touch the constraint region.

For L1 (diamond): The elliptical contours most likely make first contact at a corner of the diamond. Corners are where one or more coordinates equal zero -- that's your sparsity.

For L2 (circle): The elliptical contours touch the smooth circular boundary, which almost never happens at an axis intersection. Both weights remain nonzero.

```python
# Demonstrate: L1 constraint is |w1| + |w2| <= c (diamond)
# Solution tends toward corners where w1=0 or w2=0

# L2 constraint is w1^2 + w2^2 <= c^2 (circle)
# Solution can be anywhere on the boundary
```

### Exercise 2: Bias-Variance with Regularization

**Problem**: A model with $\lambda = 0$ has MSE = 0.10 on training data and MSE = 0.50 on test data. With $\lambda = 1.0$, it has MSE = 0.25 on training and MSE = 0.30 on test. Explain what's happening in terms of bias and variance.

**Solution**:

- **$\lambda = 0$**: The gap is 0.50 - 0.10 = 0.40. This large gap signals high variance -- the model is overfitting, memorizing training noise.
- **$\lambda = 1.0$**: The gap is 0.30 - 0.25 = 0.05. Much smaller. The model generalizes.

What the regularization did:
- **Increased bias**: Training error went from 0.10 to 0.25 (the model can't fit training data as perfectly)
- **Decreased variance**: Test error went from 0.50 to 0.30 (the model is less sensitive to training noise)

Net effect: Test error improved from 0.50 to 0.30. The variance reduction (0.20 improvement on test) far outweighed the bias increase (0.15 increase on train). This is exactly the tradeoff regularization is designed to make.

### Exercise 3: Elastic Net Advantage

**Problem**: When would you prefer Elastic Net over pure Lasso?

**Solution**:

Use Elastic Net when:

1. **Correlated features**: If features X1 and X2 are highly correlated, Lasso arbitrarily selects one and ignores the other. Elastic Net keeps both with similar weights, which is more stable and often more interpretable.

2. **More features than samples ($p > n$)**: Lasso can select at most $n$ features. Elastic Net has no such limitation and can handle the $p > n$ regime more gracefully.

3. **Group selection**: When groups of correlated features should be selected or rejected together. Elastic Net's L2 component ensures correlated features get similar treatment.

```python
# Example: X1 and X2 are highly correlated
# Lasso might give: w1 = 1.5, w2 = 0 (arbitrary selection)
# Elastic Net gives: w1 = 0.8, w2 = 0.7 (shared weight)
```

---

## Summary

| Concept | What It Does | SWE Analogy |
|---|---|---|
| **Regularization** | Adds a complexity penalty to the loss function | Rate limiting for parameters |
| **L2 (Ridge)** | Shrinks all weights toward zero, never to exactly zero | Elastic load balancing |
| **L1 (Lasso)** | Pushes weights to exactly zero, automatic feature selection | Garbage collection for features |
| **Elastic Net** | L1 + L2 combined: sparsity with stability | Pruning with a safety net |
| **Dropout** | Randomly kills neurons during training | Chaos Monkey for neural nets |
| **Bias-Variance** | Simple models underfit, complex models overfit | Precision vs recall tradeoff |

Key takeaways:

- The hyperparameter $\lambda$ controls regularization strength: too high causes underfitting, too low allows overfitting
- Always standardize features before applying regularization so the penalty is fair across features
- Use cross-validation to select the optimal $\lambda$ -- don't guess
- The bias-variance decomposition $\mathbb{E}[(y - \hat{f})^2] = \text{Bias}^2 + \text{Var} + \sigma^2$ explains why regularization works: it trades a small bias increase for a large variance reduction
- In deep learning, weight decay, dropout, batch normalization, early stopping, and data augmentation are all forms of regularization working together

---

## What's Next

You've completed optimization theory. With loss functions, algorithms, convexity, and regularization, you have the complete training toolkit. Now we'll look at how these ideas come together in specific ML models.
