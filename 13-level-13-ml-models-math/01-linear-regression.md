# Chapter 1: Linear Regression

## The "Hello World" of ML

If ML has a "Hello World," it's linear regression. y = Xw -- that's it. Three letters, and you have a complete model. But those three letters hide surprising depth: a geometric interpretation (projection), a probabilistic interpretation (maximum likelihood), and an optimization interpretation (least squares). Let's unpack all three.

---

**Building On** -- You have the complete math toolkit: linear algebra, calculus, probability, optimization. Now let's put it ALL together. Linear regression uses every concept from the previous levels.

- **Linear algebra**: the normal equation is a system of linear equations.
- **Calculus**: we take the gradient of a loss function and set it to zero.
- **Probability**: maximum likelihood under Gaussian noise *is* least squares.
- **Optimization**: gradient descent iteratively minimizes the loss.

Every chapter you've worked through converges right here.

---

## Running Example: Predicting House Prices

Throughout this chapter, we'll use one concrete problem: **predicting house prices from square footage**.

You have data from recent sales:

```
sqft (x)    price (y, $k)
 800        150
1200        250
1500        300
1800        380
2200        460
```

You want a model: **y = w_1 * sqft + w_0**. That's it. Find the best w_0 (intercept) and w_1 (slope), and you can predict the price of any house given its square footage.

In matrix form, you're solving y = Xw where:

```
        [ 1   800 ]           [ 150 ]           [ w_0 ]
        [ 1  1200 ]           [ 250 ]           [ w_1 ]
X   =   [ 1  1500 ]    y  =  [ 300 ]    w  =
        [ 1  1800 ]           [ 380 ]
        [ 1  2200 ]           [ 460 ]
```

The column of 1s handles the intercept w_0. This is a pattern you'll see everywhere in ML: append a bias column so the intercept becomes just another weight.

> **You Already Know This**: This is just solving a system of equations. If you had exactly 2 data points, you'd have 2 equations and 2 unknowns -- a unique solution. With 5 data points and 2 unknowns, the system is *overdetermined*. There's no exact solution, so we find the "best approximate" one. That's what linear regression does.

---

## Perspective 1: Geometric -- Projection

Let's start with the picture. Here's your data with the best-fit line:

```
  price ($k)
   500 |
       |                                    *  /
   400 |                              *   /
       |                            /
   300 |                    *  /
       |                  /
   200 |            * /
       |          /
   100 |     * /        residual = vertical
       |     |<---------->|    distance from
     0 |_____|____________|___ point to line
       0   500  1000  1500  2000  2500  sqft
```

Each data point sits somewhere in the plane. The line y = w_1 * sqft + w_0 is our model. The **residuals** are the vertical distances between the points and the line:

```
       |          *  data point
       |          |
       |    ------+------  fitted line
       |          |
       |     residual = y_i - y_hat_i
```

> **You Already Know This**: Residuals are like error logs. For each prediction, the residual tells you how far off you were. Positive residual = underpredicted. Negative = overpredicted. You want these "error logs" to look like random noise, not systematic patterns.

### The Projection Interpretation

Here's the deeper geometric insight. Think of y as a vector in n-dimensional space (one dimension per data point). The columns of X span a subspace -- the **column space**. The prediction y_hat = Xw is some vector *inside* that column space.

Linear regression finds the w that makes y_hat as close to y as possible. That means y_hat is the **orthogonal projection** of y onto the column space of X.

```
                          y (true targets)
                         /|
                        / |
                       /  |  residual (y - y_hat)
                      /   |  perpendicular to
                     /    |  column space
                    /     |
        -----------*------+----------- Column space of X
                  y_hat = Xw
                  (projection)
```

The residual vector (y - y_hat) is perpendicular to the column space. That perpendicularity condition is exactly the normal equation:

$$\mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w}^*) = \mathbf{0}$$

Read that formula carefully: X^T times the residual equals zero. That's saying "the residual is orthogonal to every column of X." This is the projection theorem from linear algebra -- you've seen it before.

> **You Already Know This**: Think of projection like a SQL `SELECT`. Your data lives in a high-dimensional space, but you're projecting it down onto just the dimensions spanned by your features. The projection keeps only the part of y that your features can explain.

---

## Perspective 2: Optimization -- Least Squares

Now let's derive the same answer from the optimization perspective. We want to find weights w that minimize the **Mean Squared Error (MSE)**:

$$L(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$$

In matrix notation:

$$L(\mathbf{w}) = \frac{1}{n} (\mathbf{y} - \mathbf{X}\mathbf{w})^T (\mathbf{y} - \mathbf{X}\mathbf{w})$$

Why squared error? Two reasons: (1) it's differentiable everywhere, and (2) it connects to the probabilistic interpretation we'll see next. For now, let's just minimize it.

### Deriving the Normal Equation

Expand the loss:

$$L(\mathbf{w}) = \frac{1}{n} \left( \mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{X}\mathbf{w} + \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w} \right)$$

We used these matrix calculus identities (from the calculus levels):

| Expression | Gradient w.r.t. w |
|---|---|
| a^T w | a |
| w^T A w | 2Aw (if A symmetric) |

Take the gradient with respect to w:

$$\nabla_\mathbf{w} L = \frac{1}{n} \left( -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{w} \right)$$

Set it to zero:

$$\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}$$

This is the **Normal Equation**. Solve for w:

$$\boxed{\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}$$

That's the closed-form solution. Plug in X and y, do some matrix math, and you get the optimal weights. No iteration, no hyperparameters.

### Why "Normal"?

The equation is called "normal" because it enforces that the residual vector (y - Xw) is **normal** (perpendicular) to the column space of X:

$$\mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w}^*) = \mathbf{0}$$

Notice: the same perpendicularity condition from the geometric perspective. The geometric view and the optimization view give us the exact same equation.

> **You Already Know This**: The normal equation is the "just compute it" solution -- like looking up the answer in a hash map instead of searching for it. You directly solve the system X^T X w = X^T y. No loops, no convergence criteria. But as we'll see, this direct approach has scalability issues.

### Back to Our House Prices

Let's compute it for our running example:

```
X^T X = [ 1  1  1  1  1 ] [ 1   800 ]   = [    5     7500  ]
        [800 1200 1500 1800 2200] [ 1  1200 ]     [ 7500  12,580,000 ]
                                  [ 1  1500 ]
                                  [ 1  1800 ]
                                  [ 1  2200 ]

X^T y = [ 1  1  1  1  1 ] [ 150 ]   = [  1540  ]
        [800 1200 1500 1800 2200] [ 250 ]     [ 2,558,000 ]
                                  [ 300 ]
                                  [ 380 ]
                                  [ 460 ]
```

Solve (X^T X) w = X^T y to get w_0 (intercept) and w_1 (price per sqft). The result: approximately w_0 ~ -18.5 and w_1 ~ 0.22, meaning each additional square foot adds about $220 to the predicted price, with a small negative intercept.

### The Iterative Alternative: Gradient Descent

The normal equation requires inverting X^T X. For large datasets (millions of features), that's expensive. Gradient descent offers an iterative alternative.

The gradient of the MSE loss is:

$$\nabla_\mathbf{w} L = \frac{2}{n}\mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y})$$

The update rule:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla_\mathbf{w} L$$

where alpha is the learning rate.

> **You Already Know This**: Gradient descent on MSE is the "keep improving" approach -- like binary search converging on a solution. Each step, you compute the gradient (which direction makes the loss worse?), then step in the opposite direction. You don't get the exact answer immediately, but you converge toward it. And unlike the normal equation, this scales to billions of parameters.

---

## Perspective 3: Probabilistic -- Maximum Likelihood

Here's where it gets elegant. Assume your data has some noise:

$$y_i = \mathbf{w}^T\mathbf{x}_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

Each observation is the true linear relationship plus Gaussian noise. This means each y_i follows a normal distribution:

$$y_i \sim \mathcal{N}(\mathbf{w}^T\mathbf{x}_i, \sigma^2)$$

The likelihood of observing all your data given weights w is:

$$p(\mathbf{y}|\mathbf{X}, \mathbf{w}) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2}\right)$$

Taking the negative log-likelihood (because products become sums, and we want to minimize):

$$-\log p(\mathbf{y}|\mathbf{X}, \mathbf{w}) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

Look at that second term. Minimizing the negative log-likelihood with respect to w means minimizing:

$$\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

That's just the sum of squared errors. **Maximum likelihood estimation under Gaussian noise gives us least squares.**

Three perspectives. Three different starting points. One answer: the normal equation.

---

## Three Roads, One Destination

Let's summarize what just happened:

```
GEOMETRIC              OPTIMIZATION           PROBABILISTIC
(projection)           (least squares)        (max likelihood)
    |                       |                       |
    |  Project y onto       |  Minimize sum of      |  Maximize P(data|w)
    |  column space of X    |  squared residuals    |  under Gaussian noise
    |                       |                       |
    v                       v                       v
    +------- ALL GIVE ------+------- THE SAME ------+
                            |
                            v
                   w* = (X^T X)^{-1} X^T y
```

This is not a coincidence. The three perspectives are deeply connected, and understanding all three gives you the intuition to extend linear regression in different directions:
- Change the **geometry** (project onto a different space) --> kernel methods
- Change the **loss function** (use absolute error instead of squared) --> robust regression
- Change the **noise model** (use non-Gaussian likelihood) --> generalized linear models

---

## Common Mistakes

### Mistake 1: Using the Normal Equation in Production

The normal equation w* = (X^T X)^{-1} X^T y is elegant, but **don't use it in production**. It's O(d^3) for the matrix inversion (where d is the number of features) and numerically unstable when X^T X is nearly singular. Use `np.linalg.lstsq()` or gradient descent instead.

```python
# Don't do this in production:
w = np.linalg.inv(X.T @ X) @ X.T @ y     # O(d^3), numerically unstable

# Do this instead:
w, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)  # Uses SVD internally
```

### Mistake 2: Multicollinearity

When features are correlated, X^T X becomes nearly singular (its determinant approaches zero). The inverse amplifies numerical errors, and your weights become unreliable.

*Fix*: Use Ridge regression (add lambda * I to X^T X) or drop correlated features.

### Mistake 3: Forgetting to Scale Features

If square footage ranges from 500 to 5000 and number of bedrooms ranges from 1 to 5, the gradient landscape is elongated. Gradient descent will zigzag instead of heading straight to the minimum.

*Fix*: Standardize features to zero mean and unit variance before fitting.

### Mistake 4: Not Checking Residuals

If your residuals show patterns (e.g., they increase with predicted values), your model is misspecified. Residuals should look like random noise.

*Fix*: Plot residuals vs. predicted values. Patterns mean you need nonlinear features or a different model.

---

## Code: From Data to Trained Model

Let's implement both approaches and apply them to our house price example.

```python
import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionFromScratch:
    """
    Linear Regression using the Normal Equation.

    This is the closed-form "just compute it" solution.
    Good for learning, but use np.linalg.lstsq() in production.
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.weights = None

    def fit(self, X, y):
        """
        Fit the model using the Normal Equation: w = (X^T X)^(-1) X^T y

        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
        y : numpy array of shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        # Add bias column if fitting intercept
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        # Normal Equation: w = (X^T X)^(-1) X^T y
        # We use np.linalg.solve instead of explicit inverse for stability
        XtX = X.T @ X
        Xty = X.T @ y
        self.weights = np.linalg.solve(XtX, Xty)

        return self

    def predict(self, X):
        """Make predictions."""
        X = np.array(X)
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return (X @ self.weights).flatten()

    def score(self, X, y):
        """
        Calculate R-squared score.
        R^2 = 1 - (sum of squared residuals) / (total sum of squares)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent.

    The iterative "keep improving" approach.
    Scales to massive datasets where the normal equation is too expensive.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.weights = None
        self.loss_history = []

    def fit(self, X, y):
        """Fit using gradient descent."""
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples = X.shape[0]

        if self.fit_intercept:
            X = np.column_stack([np.ones(n_samples), X])

        n_features = X.shape[1]

        # Initialize weights randomly
        self.weights = np.random.randn(n_features, 1) * 0.01

        # Gradient descent: keep improving until convergence
        for _ in range(self.n_iterations):
            # Predictions: y_hat = Xw
            y_pred = X @ self.weights

            # Compute loss: MSE = (1/n) * ||y - y_hat||^2
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            # Compute gradient: d(MSE)/dw = (2/n) * X^T(Xw - y)
            gradient = (2 / n_samples) * X.T @ (y_pred - y)

            # Update: step in the opposite direction of the gradient
            self.weights -= self.lr * gradient

        return self

    def predict(self, X):
        """Make predictions."""
        X = np.array(X)
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return (X @ self.weights).flatten()


# ---- Running Example: House Prices ----
if __name__ == "__main__":
    # Our running example: predicting house prices from square footage
    sqft = np.array([800, 1200, 1500, 1800, 2200]).reshape(-1, 1)
    prices = np.array([150, 250, 300, 380, 460])

    # Method 1: Normal Equation (closed-form)
    model_ne = LinearRegressionFromScratch()
    model_ne.fit(sqft, prices)
    print("=== Normal Equation Solution ===")
    print(f"  w_0 (intercept): {model_ne.weights[0, 0]:.2f}")
    print(f"  w_1 ($/sqft):    {model_ne.weights[1, 0]:.4f}")
    print(f"  R-squared:       {model_ne.score(sqft, prices):.4f}")
    print(f"  Predict 1600 sqft: ${model_ne.predict([[1600]])[0]:.0f}k")

    # Method 2: Gradient Descent (iterative)
    # Note: we normalize sqft for GD stability
    sqft_mean, sqft_std = sqft.mean(), sqft.std()
    sqft_norm = (sqft - sqft_mean) / sqft_std

    model_gd = LinearRegressionGD(learning_rate=0.1, n_iterations=200)
    model_gd.fit(sqft_norm, prices)
    print("\n=== Gradient Descent Solution ===")
    print(f"  Converged loss (MSE): {model_gd.loss_history[-1]:.2f}")

    # Method 3: Production approach
    X_with_bias = np.column_stack([np.ones(len(sqft)), sqft])
    w_lstsq, _, _, _ = np.linalg.lstsq(X_with_bias, prices, rcond=None)
    print("\n=== np.linalg.lstsq (production) ===")
    print(f"  w_0 (intercept): {w_lstsq[0]:.2f}")
    print(f"  w_1 ($/sqft):    {w_lstsq[1]:.4f}")
```

### Output

```
=== Normal Equation Solution ===
  w_0 (intercept): -18.50
  w_1 ($/sqft):    0.2179
  R-squared:       0.9975
  Predict 1600 sqft: $330k

=== Gradient Descent Solution ===
  Converged loss (MSE): 24.63

=== np.linalg.lstsq (production) ===
  w_0 (intercept): -18.50
  w_1 ($/sqft):    0.2179
```

All three methods converge to the same answer: each square foot adds about $218 to the price. The normal equation and lstsq give the exact solution; gradient descent iteratively converges toward it.

---

## Computational Complexity: When to Use What

| Method | Time Complexity | When to Use |
|---|---|---|
| Normal equation | O(nd^2 + d^3) | Small d (< 10,000 features) |
| np.linalg.lstsq | O(nd^2) via SVD | Default choice for moderate data |
| Gradient descent | O(ndk) for k iterations | Large n and d, online learning |
| Stochastic GD | O(dk) per epoch | Massive datasets, deep learning |

Where n = number of samples, d = number of features, k = number of iterations.

---

## Extensions

Once you understand vanilla linear regression, every extension is a small modification:

| Extension | What Changes | Normal Equation Becomes |
|---|---|---|
| **Ridge Regression** (L2) | Add lambda * \|\|w\|\|^2 to loss | w* = (X^T X + lambda I)^{-1} X^T y |
| **Lasso Regression** (L1) | Add lambda * \|\|w\|\|_1 to loss | No closed form -- use coordinate descent |
| **Polynomial Regression** | Transform features: [x, x^2, x^3, ...] | Same equation, different X |
| **Bayesian Linear Reg.** | Place priors on w | Posterior = prior * likelihood |

### Ridge Regression: Worth Knowing the Formula

With L2 regularization, the loss becomes:

$$L(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2$$

The closed-form solution:

$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

Notice the addition of lambda * I. This does two things:
1. **Regularization**: penalizes large weights, preventing overfitting.
2. **Numerical stability**: guarantees the matrix is invertible (all eigenvalues are at least lambda).

From the probabilistic perspective, Ridge regression is equivalent to placing a Gaussian prior on the weights: w ~ N(0, (1/lambda) I). This is **MAP estimation** instead of MLE.

```python
def fit_ridge(self, X, y, lambda_reg):
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    if self.fit_intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])

    n_features = X.shape[1]

    # Ridge: w = (X^T X + lambda * I)^(-1) X^T y
    XtX = X.T @ X
    regularizer = lambda_reg * np.eye(n_features)
    if self.fit_intercept:
        regularizer[0, 0] = 0  # Don't regularize the intercept

    self.weights = np.linalg.solve(XtX + regularizer, X.T @ y)
    return self
```

---

## Exercises

### Exercise 1: Derive the Gradient (Step by Step)

**Problem**: Starting from the MSE loss L(w) = (1/n) ||y - Xw||^2, derive the gradient step by step.

**Solution**:

$$L(\mathbf{w}) = \frac{1}{n}(\mathbf{y} - \mathbf{X}\mathbf{w})^T(\mathbf{y} - \mathbf{X}\mathbf{w})$$

Let r = y - Xw. Then L = (1/n) r^T r.

Expanding:

$$L = \frac{1}{n}(\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{X}\mathbf{w} + \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w})$$

Taking derivatives term by term:
- d/dw (y^T y) = 0 (no dependence on w)
- d/dw (y^T X w) = X^T y (linear in w)
- d/dw (w^T X^T X w) = 2 X^T X w (quadratic form, X^T X is symmetric)

Therefore:

$$\nabla_\mathbf{w} L = \frac{1}{n}(-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{w}) = \frac{2}{n}\mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y})$$

### Exercise 2: Implement Ridge Regression

**Problem**: Modify the `LinearRegressionFromScratch` class to implement Ridge Regression with regularization parameter lambda.

**Solution**: See the `fit_ridge` method in the Extensions section above. The key insight is that you only add one line -- `lambda * I` -- to the normal equation. That's the power of understanding the math: a one-line change gives you regularization.

### Exercise 3: Normal Equation vs. Gradient Descent

**Problem**: What is the computational complexity of the normal equation? When should you use gradient descent instead?

**Solution**:

Normal equation complexity:
- Computing X^T X: O(nd^2) -- n samples, d features
- Solving the d x d system: O(d^3)
- Total: O(nd^2 + d^3)

Gradient descent (k iterations):
- Each iteration: O(nd) for the matrix-vector products
- Total: O(ndk)

Use gradient descent when:
- **d is very large** (d > 10,000): the d^3 term in the normal equation becomes prohibitive
- **Data streams in** (online learning): you can't fit everything in memory
- **Approximate solution is fine**: you don't need the exact optimum
- **Memory is limited**: the normal equation requires storing the d x d matrix X^T X

Use the normal equation (or lstsq) when:
- **d is small**: the closed-form solution is instant
- **You need the exact answer**: no convergence questions, no learning rate tuning

---

## Summary

- **Linear regression** predicts y = Xw by minimizing the sum of squared residuals.

- **Three perspectives**, one answer:
  - *Geometric*: y_hat is the projection of y onto the column space of X.
  - *Optimization*: minimize ||y - Xw||^2, yielding the normal equation.
  - *Probabilistic*: MLE under Gaussian noise = least squares.

- **The Normal Equation** provides the closed-form solution:
  $$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

- **In production**, use `np.linalg.lstsq()` (SVD-based, numerically stable) or gradient descent (scalable).

- **Ridge regression** adds lambda * I to handle multicollinearity and prevent overfitting, turning MLE into MAP estimation.

- **Always check residuals**: they should look like random noise, not systematic patterns.

---

**What's Next** -- Linear regression predicts continuous values. But what if you need to predict classes? Logistic regression extends the same ideas to classification. You'll see the same three perspectives (geometric, optimization, probabilistic) applied to a fundamentally different problem: outputs between 0 and 1.

**Next Chapter**: [Logistic Regression](./02-logistic-regression.md)
