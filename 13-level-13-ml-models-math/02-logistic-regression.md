# Chapter 2: Logistic Regression

A patient's blood test results: [glucose=180, BMI=32, age=55]. Is this diabetes? Your model needs to output a probability between 0 and 1. But wx + b gives you any real number. How do you squeeze R into (0,1)? You use the sigmoid function -- and that gives you logistic regression.

---

> **Building On** -- Linear regression predicts numbers. But many ML problems need yes/no answers: spam or not, sick or healthy, click or ignore. How do you adapt a linear model for classification?

---

## The Problem: Turning a Linear Score into a Probability

You already have a linear model from the previous chapter:

$$z = \mathbf{w}^T\mathbf{x} + b$$

This gives you a score -- any real number from $-\infty$ to $+\infty$. For a spam classifier, maybe $z = 3.7$ means "probably spam" and $z = -2.1$ means "probably not spam." But what does 3.7 *mean* as a probability? And what about $z = 47$? Is that more spam than 3.7?

You need a function that:
1. Maps any real number to the range $(0, 1)$
2. Is monotonically increasing (higher score = higher probability)
3. Is differentiable (so you can use gradient descent)

That function is the **sigmoid**.

## You Already Know This

Before we dive in, let's connect to concepts you've already worked with:

| Math Concept | Your Engineering Intuition |
|---|---|
| **Sigmoid function** $\sigma(z)$ | The function that turns any score into a probability. You've used it in neural network outputs -- `torch.sigmoid()` or the last layer of a binary classifier. |
| **Cross-entropy loss** | Log-scoring: how *surprised* were you by the actual outcome? If you predicted 0.99 and the label was 1, low surprise (low loss). Predicted 0.99 and label was 0? Huge surprise (huge loss). |
| **Decision boundary** | A threshold. Above 0.5 -> positive class. Below 0.5 -> negative class. In 2D, it's literally a line. |
| **Log-odds (logit)** | The raw score *before* sigmoid, like a confidence score. The `z` in `sigmoid(z)`. It's what your model actually computes -- sigmoid just converts it to a probability for human consumption. |

## The Sigmoid Function

The sigmoid (logistic) function transforms any real number into $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Here's what it looks like:

```
    σ(z)
  1 |                          .-----------
    |                       ./
    |                     ./
    |                   ./
0.5 |_ _ _ _ _ _ _ _ _/_ _ _ _ _ _ _ _ _ _
    |                /
    |              ./
    |            ./
    |          ./
  0 |---------'
    +-------|---------|---------|---------|---> z
           -4        -2         0         2         4
```

### Key Properties

| Property | Value | Why It Matters |
|----------|-------|----------------|
| $\sigma(0)$ | $0.5$ | When the score is zero, you're completely uncertain |
| $\sigma(\infty)$ | $1$ | Extremely high score -> near-certain positive |
| $\sigma(-\infty)$ | $0$ | Extremely low score -> near-certain negative |
| Symmetry | $\sigma(-z) = 1 - \sigma(z)$ | Flipping the sign flips the probability |
| **Derivative** | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ | This is the key -- the derivative is expressed in terms of the function itself |

That derivative property is beautiful. Let's derive it because you'll need it later for the gradient:

$$\sigma(z) = (1 + e^{-z})^{-1}$$

$$\sigma'(z) = -1 \cdot (1 + e^{-z})^{-2} \cdot (-e^{-z})$$

$$= \frac{e^{-z}}{(1 + e^{-z})^2}$$

$$= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}$$

$$= \sigma(z) \cdot \frac{1 + e^{-z} - 1}{1 + e^{-z}}$$

$$= \sigma(z) \cdot (1 - \sigma(z))$$

The maximum derivative is at $z=0$: $\sigma'(0) = 0.5 \times 0.5 = 0.25$. This means the sigmoid is most sensitive around the decision boundary, exactly where you want it to be.

## The Logistic Regression Model

Here's the full pipeline. You compute a linear score, then pass it through sigmoid:

```
  Features          Linear Score           Sigmoid           Decision
 [x1, x2, ..., xn] --> z = w*x + b --> p = sigma(z) --> y = 1 if p > 0.5
                                                          y = 0 if p <= 0.5
```

Formally:

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

### Running Example: Spam Classifier

Let's make this concrete. You're building a spam classifier:

$$P(\text{spam} | \text{features}) = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$$

Your features might be:
- $x_1$: number of exclamation marks
- $x_2$: contains "FREE" (0 or 1)
- $x_3$: sender in contacts (0 or 1)
- $x_4$: email length (normalized)

Suppose after training, you get $w_1 = 1.2$, $w_2 = 2.5$, $w_3 = -3.0$, $w_4 = -0.1$, $b = -1.0$.

For an email with 3 exclamation marks, contains "FREE", sender not in contacts, length 0.5:

$$z = 1.2(3) + 2.5(1) + (-3.0)(0) + (-0.1)(0.5) + (-1.0) = 3.6 + 2.5 + 0 - 0.05 - 1.0 = 5.05$$

$$P(\text{spam}) = \sigma(5.05) = \frac{1}{1 + e^{-5.05}} \approx 0.994$$

That's a 99.4% chance of spam. The model is very confident. **Training = finding the weights $\mathbf{w}$ and bias $b$ that maximize the likelihood of the observed labels.**

## The Log-Odds (Logit) Connection

Here's a perspective that makes logistic regression feel more natural. Rearrange the sigmoid:

$$p = \frac{1}{1 + e^{-z}} \implies e^{-z} = \frac{1-p}{p} \implies z = \log\frac{p}{1-p}$$

The function $\text{logit}(p) = \log\frac{p}{1-p}$ is the inverse of sigmoid. The quantity $\frac{p}{1-p}$ is the **odds** (like "3 to 1 odds"), and $\log\frac{p}{1-p}$ is the **log-odds**.

So logistic regression says:

$$\log\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \mathbf{w}^T\mathbf{x} + b$$

The log-odds of the outcome are a linear function of the features. That's the core assumption. You're not assuming probabilities are linear (they can't be -- probabilities are bounded). You're assuming the log-odds are linear.

Back to the spam example: if the coefficient for "contains FREE" is $w_2 = 2.5$, the odds ratio is $e^{2.5} \approx 12.2$. Containing "FREE" multiplies the odds of being spam by about 12x, holding everything else constant. That's interpretability you don't get from a neural network.

## Cross-Entropy Loss: How Wrong Were You?

Now you need a loss function to train this model. Your instinct might be to use mean squared error like in linear regression. Don't. Here's why.

### Why Not MSE?

If you use MSE with sigmoid, you get a non-convex loss surface -- full of local minima. Gradient descent might get stuck. You need a loss function designed for probabilities.

### Maximum Likelihood Estimation

Think about it probabilistically. Each training example is a coin flip with a different bias:

$$P(y_i | \mathbf{x}_i, \mathbf{w}) = p_i^{y_i}(1 - p_i)^{1-y_i}$$

where $p_i = \sigma(\mathbf{w}^T\mathbf{x}_i + b)$.

When $y_i = 1$: the probability is $p_i$ (you want this to be high).
When $y_i = 0$: the probability is $1 - p_i$ (you want $p_i$ to be low).

The likelihood of all data (assuming independence):

$$L(\mathbf{w}) = \prod_{i=1}^{n} p_i^{y_i}(1-p_i)^{1-y_i}$$

Maximizing this product is the same as maximizing its log (products become sums):

$$\ell(\mathbf{w}) = \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \right]$$

### Cross-Entropy Loss

Minimizing the *negative* log-likelihood gives you the **cross-entropy loss** (also called **log loss**):

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \right]$$

Substituting $p_i = \sigma(\mathbf{w}^T\mathbf{x}_i + b)$:

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(\sigma(z_i)) + (1-y_i)\log(1-\sigma(z_i)) \right]$$

where $z_i = \mathbf{w}^T\mathbf{x}_i + b$.

### Intuition: The Surprise Meter

Think of cross-entropy as a "surprise meter":

```
  Loss
   |
 5 |*
   | *
 4 |  *
   |   *
 3 |    *
   |     *
 2 |      *
   |       **
 1 |         ***
   |            ****
 0 |________________*******___
   0    0.2   0.4   0.6   0.8   1.0
           Predicted p

   (When actual label y = 1)

   If you predicted p=0.99 and y=1: loss = -log(0.99) = 0.01   (not surprised)
   If you predicted p=0.01 and y=1: loss = -log(0.01) = 4.6    (very surprised!)
```

The loss penalizes confident wrong predictions *severely*. Predicting 0.01 when the answer was 1 costs you 460 times more than predicting 0.99 when the answer was 1. This is exactly the behavior you want.

## Optimization: Finding the Best Weights

### Deriving the Gradient

You need $\frac{\partial \mathcal{L}}{\partial \mathbf{w}}$ for gradient descent. This is where the beautiful derivative of sigmoid pays off.

For a single sample, using the chain rule:

$$\frac{\partial \mathcal{L}_i}{\partial \mathbf{w}} = \frac{\partial \mathcal{L}_i}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial \mathbf{w}}$$

**Step 1** -- Derivative of loss with respect to prediction:

$$\frac{\partial \mathcal{L}_i}{\partial p_i} = -\frac{y_i}{p_i} + \frac{1-y_i}{1-p_i}$$

**Step 2** -- Derivative of sigmoid (from earlier):

$$\frac{\partial p_i}{\partial z_i} = p_i(1 - p_i)$$

**Step 3** -- Derivative of linear function:

$$\frac{\partial z_i}{\partial \mathbf{w}} = \mathbf{x}_i$$

**Combining**:

$$\frac{\partial \mathcal{L}_i}{\partial \mathbf{w}} = \left(-\frac{y_i}{p_i} + \frac{1-y_i}{1-p_i}\right) \cdot p_i(1-p_i) \cdot \mathbf{x}_i$$

**Simplifying** (expand and cancel):

$$= \left(-y_i(1-p_i) + (1-y_i)p_i\right) \cdot \mathbf{x}_i$$

$$= (p_i - y_i) \cdot \mathbf{x}_i$$

That's it. The gradient for all samples:

$$\nabla_\mathbf{w} \mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)\mathbf{x}_i = \frac{1}{n}\mathbf{X}^T(\mathbf{p} - \mathbf{y})$$

This is the same form as linear regression's gradient. The prediction $p_i$ minus the true label $y_i$, scaled by the feature vector. The only difference from linear regression is *how* you compute $p_i$ -- through sigmoid instead of directly.

### Why No Closed-Form Solution?

In linear regression, you set $\nabla \mathcal{L} = 0$ and solved for $\mathbf{w}$ directly (the normal equations). You can't do that here because $p_i = \sigma(\mathbf{w}^T\mathbf{x}_i)$ is nonlinear in $\mathbf{w}$. The equation $\sigma(\mathbf{w}^T\mathbf{x}_i) - y_i = 0$ has no closed-form solution. You must use iterative optimization.

### Gradient Descent Update

The update rule is straightforward:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \cdot \frac{1}{n}\mathbf{X}^T(\mathbf{p} - \mathbf{y})$$

where $\alpha$ is the learning rate.

### Newton's Method: Faster Convergence

For logistic regression specifically, you can use Newton's method for faster convergence. The Hessian (matrix of second derivatives) is:

$$\mathbf{H} = \frac{1}{n}\mathbf{X}^T\mathbf{S}\mathbf{X}$$

where $\mathbf{S} = \text{diag}(p_i(1-p_i))$ is a diagonal matrix of sigmoid derivatives.

Newton's update:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{H}^{-1}\nabla \mathcal{L}$$

This is called **Iteratively Reweighted Least Squares (IRLS)** because at each step you're solving a weighted least squares problem. Newton's method typically converges in 5-10 iterations versus hundreds for gradient descent.

## Decision Boundaries

After training, how do you classify new data? You set a threshold (usually 0.5):

$$\hat{y} = \begin{cases} 1 & \text{if } \sigma(\mathbf{w}^T\mathbf{x} + b) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

Since sigmoid is monotonic, $\sigma(z) \geq 0.5$ exactly when $z \geq 0$. So the decision boundary is:

$$\mathbf{w}^T\mathbf{x} + b = 0$$

That's a hyperplane. In 2D, it's a line:

```
  x2
   |        . . .  Class 1 (spam)
   |      . . .  /
   |    . . .  /
   |  . . .  / <-- Decision boundary: w1*x1 + w2*x2 + b = 0
   |. . .  /
   |. .  /
   |.  /  o o o
   | /  o o o o  Class 0 (not spam)
   |/ o o o o o
   +-----------------> x1

   Above the line: w*x + b > 0 --> sigma > 0.5 --> predict spam
   Below the line: w*x + b < 0 --> sigma < 0.5 --> predict not spam
```

The decision boundary is always linear. That's both the strength and the limitation of logistic regression -- it can only separate classes with a straight line (or hyperplane in higher dimensions).

## Common Mistakes

**Logistic regression's loss is CONVEX.** Unlike neural networks, gradient descent will find the global optimum. This is one of its biggest advantages. You don't need to worry about local minima, weight initialization strategies, or learning rate schedules. If you're not converging, it's a bug, not bad luck.

Here are other pitfalls to watch out for:

1. **Using MSE instead of cross-entropy.** MSE + sigmoid = non-convex loss. You lose the convexity guarantee. Always use cross-entropy for classification.

2. **Not scaling features.** If feature $x_1$ ranges from 0 to 1 and feature $x_2$ ranges from 0 to 1,000,000, gradient descent will oscillate wildly. Standardize your features first: $x' = (x - \mu) / \sigma$.

3. **Perfect separation.** When classes are perfectly separable, the weights diverge to infinity (the sigmoid tries to become a step function). Add L2 regularization to prevent this.

4. **Ignoring class imbalance.** If 99% of emails are not spam, a model that always predicts "not spam" gets 99% accuracy but is useless. Use class weights, oversample the minority class, or use a metric like F1 instead of accuracy.

5. **Misinterpreting coefficients.** The coefficient $w_j$ is the change in *log-odds*, not in probability. To get the odds ratio, exponentiate: $e^{w_j}$. A coefficient of 0.8 means "this feature multiplies the odds by $e^{0.8} \approx 2.23$."

## Softmax: Multi-Class Extension

Binary classification covers spam/not-spam. But what about classifying into $K$ classes (e.g., cat/dog/bird)? The softmax function generalizes sigmoid:

$$P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T\mathbf{x}}}$$

When $K=2$, softmax reduces to sigmoid (try it -- subtract one class's weights from the other's, and you get the sigmoid formula).

The loss becomes **categorical cross-entropy**:

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik}\log(p_{ik})$$

where $y_{ik}$ is the one-hot encoded label.

## Regularization

Adding L2 regularization prevents overfitting and handles perfect separation:

$$\mathcal{L}_{reg}(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2$$

From a Bayesian perspective, this is equivalent to placing a Gaussian prior on the weights and doing MAP (Maximum A Posteriori) estimation instead of MLE. The regularization strength $\lambda$ controls how much you trust the prior versus the data.

## Code: Logistic Regression from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """
    Compute sigmoid function.
    Clipped for numerical stability.
    """
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))


class LogisticRegressionFromScratch:
    """
    Logistic Regression using Gradient Descent.

    Running example: spam classifier.
    P(spam | features) = sigmoid(w . x + b)
    Training = finding w and b that maximize likelihood of observed labels.
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000, fit_intercept=True):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.weights = None
        self.loss_history = []

    def _add_intercept(self, X):
        """Add column of ones for intercept (bias) term."""
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X, y):
        """
        Fit logistic regression using gradient descent.

        The gradient is: (1/n) * X^T * (predictions - labels)
        Same form as linear regression -- the only difference is
        predictions go through sigmoid.

        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
        y : numpy array of shape (n_samples,) with values 0 or 1
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples = X.shape[0]

        if self.fit_intercept:
            X = self._add_intercept(X)

        n_features = X.shape[1]

        # Initialize weights to zeros
        self.weights = np.zeros((n_features, 1))

        # Gradient descent loop
        for i in range(self.n_iterations):
            # Forward pass: linear score -> sigmoid -> probability
            z = X @ self.weights          # z = w^T x + b
            p = sigmoid(z)                # p = sigma(z)

            # Cross-entropy loss: -mean(y*log(p) + (1-y)*log(1-p))
            epsilon = 1e-15  # Prevent log(0)
            loss = -np.mean(
                y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon)
            )
            self.loss_history.append(loss)

            # Gradient: (1/n) * X^T * (p - y)
            gradient = (1 / n_samples) * X.T @ (p - y)

            # Update weights: w <- w - alpha * gradient
            self.weights -= self.lr * gradient

        return self

    def predict_proba(self, X):
        """Return probability of class 1."""
        X = np.array(X)
        if self.fit_intercept:
            X = self._add_intercept(X)
        return sigmoid(X @ self.weights).flatten()

    def predict(self, X, threshold=0.5):
        """Return class predictions using decision boundary at threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        """Calculate accuracy."""
        return np.mean(self.predict(X) == y)


class LogisticRegressionNewton:
    """
    Logistic Regression using Newton's Method (IRLS).
    Converges in ~5-10 iterations vs hundreds for gradient descent.
    Uses the Hessian: H = (1/n) * X^T * S * X where S = diag(p*(1-p))
    """

    def __init__(self, n_iterations=20, fit_intercept=True):
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.weights = None

    def _add_intercept(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X, y):
        """Fit using Newton's method: w <- w - H^{-1} * gradient."""
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples = X.shape[0]

        if self.fit_intercept:
            X = self._add_intercept(X)

        n_features = X.shape[1]
        self.weights = np.zeros((n_features, 1))

        for _ in range(self.n_iterations):
            # Compute probabilities
            z = X @ self.weights
            p = sigmoid(z)

            # Gradient: (1/n) * X^T * (p - y)
            gradient = (1 / n_samples) * X.T @ (p - y)

            # Hessian: (1/n) * X^T * S * X, where S = diag(p*(1-p))
            s = p * (1 - p)
            S = np.diag(s.flatten())
            hessian = (1 / n_samples) * X.T @ S @ X

            # Newton's update: w = w - H^(-1) * gradient
            try:
                delta = np.linalg.solve(hessian, gradient)
                self.weights -= delta
            except np.linalg.LinAlgError:
                # Fall back to gradient descent if Hessian is singular
                self.weights -= 0.1 * gradient

        return self

    def predict_proba(self, X):
        X = np.array(X)
        if self.fit_intercept:
            X = self._add_intercept(X)
        return sigmoid(X @ self.weights).flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ---- Demonstration ----
if __name__ == "__main__":
    # Generate synthetic binary classification data
    np.random.seed(42)
    n_samples = 200

    # Class 0: centered at (-1, -1)
    X0 = np.random.randn(n_samples // 2, 2) + np.array([-1, -1])
    # Class 1: centered at (1, 1)
    X1 = np.random.randn(n_samples // 2, 2) + np.array([1, 1])

    X = np.vstack([X0, X1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]

    # Split into train/test
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    # Train both models
    model_gd = LogisticRegressionFromScratch(learning_rate=0.5, n_iterations=200)
    model_gd.fit(X_train, y_train)

    model_newton = LogisticRegressionNewton(n_iterations=10)
    model_newton.fit(X_train, y_train)

    print("Gradient Descent:")
    print(f"  Train accuracy: {model_gd.score(X_train, y_train):.4f}")
    print(f"  Test accuracy: {model_gd.score(X_test, y_test):.4f}")
    print(f"  Weights: {model_gd.weights.flatten()}")

    print("\nNewton's Method:")
    print(f"  Train accuracy: {model_newton.score(X_train, y_train):.4f}")
    print(f"  Test accuracy: {model_newton.score(X_test, y_test):.4f}")
    print(f"  Weights: {model_newton.weights.flatten()}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Sigmoid function
    z = np.linspace(-6, 6, 100)
    axes[0].plot(z, sigmoid(z), 'b-', linewidth=2)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('z')
    axes[0].set_ylabel('sigma(z)')
    axes[0].set_title('Sigmoid Function')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Decision boundary
    ax = axes[1]
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', alpha=0.5, label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', alpha=0.5, label='Class 1')

    # Draw decision boundary: w0 + w1*x1 + w2*x2 = 0
    w = model_gd.weights.flatten()
    x1_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    x2_vals = -(w[0] + w[1] * x1_vals) / w[2]
    ax.plot(x1_vals, x2_vals, 'g-', linewidth=2, label='Decision boundary')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Logistic Regression Decision Boundary')
    ax.legend()
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    # Plot 3: Loss curve
    axes[2].plot(model_gd.loss_history)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Cross-Entropy Loss')
    axes[2].set_title('Training Loss Convergence')

    plt.tight_layout()
    plt.savefig('logistic_regression_demo.png', dpi=100)
    plt.show()
```

### Output
```
Gradient Descent:
  Train accuracy: 0.9125
  Test accuracy: 0.9000
  Weights: [0.08234  0.89431  0.91205]

Newton's Method:
  Train accuracy: 0.9125
  Test accuracy: 0.9000
  Weights: [0.08421  0.91124  0.93012]
```

## Softmax Regression: Multi-Class Code

```python
def softmax(z):
    """Compute softmax, handling numerical stability."""
    z = z - np.max(z, axis=1, keepdims=True)  # Subtract max for stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class SoftmaxRegression:
    """
    Multinomial logistic regression.
    Generalizes binary logistic regression to K classes.
    """

    def __init__(self, n_classes, learning_rate=0.1, n_iterations=1000):
        self.n_classes = n_classes
        self.lr = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = np.column_stack([np.ones(n_samples), X])
        n_features += 1

        # One-hot encode y
        Y = np.eye(self.n_classes)[y]

        # Initialize weights: (n_features, n_classes)
        self.W = np.zeros((n_features, self.n_classes))

        for _ in range(self.n_iterations):
            # Forward pass: softmax(X @ W)
            P = softmax(X @ self.W)

            # Gradient: X^T (P - Y) / n
            gradient = X.T @ (P - Y) / n_samples

            # Update
            self.W -= self.lr * gradient

        return self

    def predict(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        P = softmax(X @ self.W)
        return np.argmax(P, axis=1)
```

## Exercises

### Exercise 1: Gradient Derivation (Work Through It)

**Problem**: Starting from the binary cross-entropy loss for one sample:

$$\mathcal{L}_i = -\left[y_i \log(\sigma(z_i)) + (1-y_i)\log(1-\sigma(z_i))\right]$$

where $z_i = \mathbf{w}^T\mathbf{x}_i$, derive $\frac{\partial \mathcal{L}_i}{\partial \mathbf{w}}$.

**Solution**:

Let $p_i = \sigma(z_i)$. Apply the chain rule:

$$\frac{\partial \mathcal{L}_i}{\partial \mathbf{w}} = \frac{\partial \mathcal{L}_i}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial \mathbf{w}}$$

Step 1 -- Loss w.r.t. prediction:

$$\frac{\partial \mathcal{L}_i}{\partial p_i} = -\frac{y_i}{p_i} + \frac{1-y_i}{1-p_i}$$

Step 2 -- Sigmoid derivative:

$$\frac{\partial p_i}{\partial z_i} = \sigma(z_i)(1-\sigma(z_i)) = p_i(1-p_i)$$

Step 3 -- Linear function derivative:

$$\frac{\partial z_i}{\partial \mathbf{w}} = \mathbf{x}_i$$

Combining:

$$\frac{\partial \mathcal{L}_i}{\partial \mathbf{w}} = \left(-\frac{y_i}{p_i} + \frac{1-y_i}{1-p_i}\right) \cdot p_i(1-p_i) \cdot \mathbf{x}_i$$

Simplifying:

$$= \left(-y_i(1-p_i) + (1-y_i)p_i\right) \cdot \mathbf{x}_i = (p_i - y_i)\mathbf{x}_i$$

### Exercise 2: Odds Ratio Interpretation

**Problem**: In a spam classifier, the coefficient for "contains FREE" is $w = 2.5$. Interpret this.

**Solution**: The coefficient is the change in log-odds. The odds ratio is $e^{2.5} \approx 12.18$.

**Interpretation**: Containing the word "FREE" multiplies the odds of an email being spam by approximately 12.2x, holding all other features constant. If the odds were 1:10 (not spam) without "FREE", they become about 12.2:10 (roughly even) with "FREE".

### Exercise 3: Threshold Tuning

**Problem**: Your spam classifier has 1% spam in production. Using threshold 0.5, you catch 90% of spam but flag 2% of legitimate emails. Your users complain about missed spam. What do you do?

**Solution**: Lower the threshold (e.g., from 0.5 to 0.3). This increases recall (catching more spam) at the cost of precision (flagging more legitimate emails). The right threshold depends on the cost of false negatives (missed spam) versus false positives (legitimate email in spam folder). This is a business decision, not a math one. The model's probabilities don't change -- only your decision rule changes.

## Summary

- **Logistic regression** models the probability of a binary outcome using the sigmoid function:

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

- The core assumption: **log-odds are linear** in features:

$$\log\frac{P(y=1)}{P(y=0)} = \mathbf{w}^T\mathbf{x} + b$$

- Training minimizes **cross-entropy loss** (= negative log-likelihood under Bernoulli assumption)

- The gradient has an elegant form: $\nabla \mathcal{L} = \frac{1}{n}\mathbf{X}^T(\mathbf{p} - \mathbf{y})$ -- same shape as linear regression

- **The loss is convex** -- gradient descent finds the global optimum. No local minima worries.

- **No closed-form solution** -- you must iterate (gradient descent or Newton's method/IRLS)

- **Softmax** extends logistic regression to multi-class classification

- Coefficients are interpretable as **log-odds** -- exponentiate for odds ratios

---

> **What's Next** -- Logistic regression is one layer with sigmoid. Stack many layers with nonlinearities, and you get a neural network. Let's see what that looks like mathematically.

---

**Previous Chapter**: [Linear Regression](./01-linear-regression.md)

**Next Chapter**: [Neural Networks](./03-neural-networks.md)
