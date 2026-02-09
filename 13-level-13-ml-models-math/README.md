# Level 13: Math Behind Core ML Models

## This Is The Payoff

This is the payoff. Every vector, matrix, derivative, and distribution you've learned converges here. Linear regression, logistic regression, neural networks, PCA — these aren't four separate algorithms. They're four expressions of the same mathematical ideas.

You've been using `model.fit()` for years. Now you're going to see what's actually happening inside that black box. Spoiler: it's maximum likelihood estimation driving gradient descent through matrix operations. You already know all these pieces. This level shows you how they snap together.

## What You Already Know (You Just Don't Know You Know It)

Remember when you learned dynamic programming? Backpropagation is the chain rule plus memoization. That's it.

Remember matrix multiplication and function composition? Neural networks are literally just `f(W3 * f(W2 * f(W1 * x)))`. You've been composing functions since day one of programming.

Remember data compression algorithms like gzip? PCA is that, but for high-dimensional data. It finds the minimal representation that preserves the maximum information.

Every ML algorithm is just optimization over matrices using calculus. And you've spent twelve levels learning exactly those tools.

## Building On

This level pulls from **everything**:

- **Level 4 (Linear Algebra)**: Matrix operations, projections, eigendecomposition, SVD
- **Level 6 (Calculus)**: Derivatives, partial derivatives, gradients, chain rule
- **Level 7 (Probability)**: Likelihood functions, Bernoulli and Gaussian distributions
- **Level 8 (Statistics)**: Maximum likelihood estimation, bias-variance tradeoff
- **Level 9 (Optimization)**: Gradient descent, SGD, Adam optimizer, convexity
- **Level 10 (Information Theory)**: Cross-entropy, KL divergence (the loss functions you use daily)

If any of those feel shaky, you can jump back. But honestly, you're ready.

## Navigation

| Chapter | Topic | Key Insight |
|---------|-------|-------------|
| [01](./01-linear-regression.md) | Linear Regression | Closed-form solution via matrix calculus |
| [02](./02-logistic-regression.md) | Logistic Regression | Probability + cross-entropy = classification |
| [03](./03-neural-networks.md) | Neural Networks | Composition + chain rule = backprop |
| [04](./04-dimensionality-reduction.md) | Dimensionality Reduction | PCA = SVD = smart compression |

## The Chapters

### Chapter 1: Linear Regression

**What you think it is**: Fitting a line through points.

**What it actually is**: Projecting your target vector onto the column space of your feature matrix. The normal equation `w = (X^T X)^(-1) X^T y` isn't magic — it's the derivative of squared error set to zero and solved algebraically.

**You'll learn**:
- The normal equation: closed-form solution via matrix calculus
- Geometric interpretation: why it's projection onto column space
- From-scratch implementation: just NumPy, no sklearn
- Regularized regression: Ridge and Lasso as constrained optimization

**The SWE bridge**: Every time you call `LinearRegression().fit()`, it's computing that matrix inverse (or QR decomposition for numerical stability). Now you know why it fails when features are collinear.

### Chapter 2: Logistic Regression

**What you think it is**: Classification with a sigmoid.

**What it actually is**: Maximum likelihood estimation under Bernoulli assumption, where the sigmoid function is the derivative of the log partition function. (Yes, really. We derive it.)

**You'll learn**:
- Sigmoid function: why `1/(1 + e^(-z))` specifically
- Log-odds interpretation: additive in log-space, multiplicative in probability-space
- Cross-entropy loss: derived from negative log-likelihood
- Gradient descent implementation: computing gradients and updating weights

**The SWE bridge**: `binary_crossentropy` in Keras is literally `-y*log(p) - (1-y)*log(1-p)`. You've used it a thousand times. Now you'll derive it from first principles in about ten lines of math.

### Chapter 3: Neural Networks

**What you think it is**: Deep learning magic.

**What it actually is**: Function composition (high school math) + matrix multiplication (Level 4) + chain rule (Level 6) + memoization (your CS degree). That's the entire algorithm.

**You'll learn**:
- Universal approximation: why depth beats width
- Forward pass: it's just matrix multiplication and activation functions
- Backpropagation from scratch: chain rule applied systematically
- Layer-by-layer gradients: computing `∂L/∂W` for every weight matrix

**The SWE bridge**: Remember computing derivatives recursively in calculus? Backprop is that, but you cache intermediate results to avoid recomputation. It's dynamic programming. You already know this pattern from LeetCode.

### Chapter 4: Dimensionality Reduction

**What you think it is**: PCA compresses data.

**What it actually is**: Finding the best low-rank approximation to your data matrix. It's simultaneously maximizing variance (signal) and minimizing reconstruction error (loss). And the solution is literally the singular value decomposition you learned in Level 4.

**You'll learn**:
- PCA: eigendecomposition of the covariance matrix
- SVD: decomposing any matrix into rotation-scaling-rotation
- The connection: they're the same algorithm in different notation
- Variance maximization = reconstruction minimization: two perspectives, one solution

**The SWE bridge**: Remember how JPEG compresses images? PCA is that for tabular data. You keep the top-k singular values and throw away the rest. It's lossy compression, but the loss is optimally distributed.

## The Mathematical Thread

The same patterns appear in every algorithm:

| Theme | Linear Regression | Logistic Regression | Neural Networks | PCA/SVD |
|-------|-------------------|---------------------|-----------------|---------|
| **What we optimize** | Mean squared error | Negative log-likelihood | Task-specific loss | Variance (or reconstruction) |
| **How we optimize** | Set gradient to zero (closed form) | Gradient descent | SGD/Adam via backprop | Eigendecomposition (closed form) |
| **Linear algebra** | Matrix inverse: `(X^T X)^(-1)` | Matrix-vector products | Matrix multiply per layer | SVD: `X = U Σ V^T` |
| **Calculus** | Derivative = 0 for optimum | Gradient tells us which way to step | Chain rule for every layer | Derivative defines principal components |
| **Probability** | Gaussian noise assumption | Bernoulli likelihood | (sometimes) Softmax for probabilities | Covariance matrix = data distribution |

Notice: they're all solving `argmin_w L(w)` where `L` is some loss function. The loss changes, but the pattern is identical.

## What You'll Actually Be Able To Do

By the end of this level, you will:

1. **Derive** the normal equation for linear regression from `∂/∂w (y - Xw)^T (y - Xw) = 0`
2. **Explain** why logistic regression uses cross-entropy (it's the negative log-likelihood)
3. **Compute** gradients through a 3-layer neural network by hand using the chain rule
4. **Prove** that PCA via covariance eigenvectors equals SVD of the data matrix
5. **Implement** all four algorithms from scratch using only NumPy (no sklearn, no PyTorch)

More importantly: you'll be able to read ML papers and understand the math. You'll debug your models by reasoning about the optimization landscape. You'll know why your training diverges or why your gradients vanish.

## Notation

| Symbol | Meaning |
|--------|---------|
| `X` | Design matrix (features), shape `(n, d)` — n samples, d features |
| `y` | Target vector, shape `(n, 1)` or `(n,)` |
| `w` or `θ` | Weight/parameter vector, shape `(d, 1)` |
| `ŷ` | Predictions: `ŷ = Xw` for linear regression |
| `∇_w L` | Gradient of loss L with respect to weights w |
| `σ(·)` | Sigmoid function: `σ(z) = 1/(1 + e^(-z))` |
| `⊙` | Element-wise (Hadamard) product |
| `⊗` | Outer product (sometimes, context-dependent) |

## How To Study This Level

1. **Read each chapter straight through** to get the big picture
2. **Go back with pencil and paper** and derive every equation yourself
3. **Run the code examples** — they're executable proofs
4. **Implement from scratch** — start with linear regression, then logistic, then a 2-layer network
5. **Compare to library implementations** — see how your code matches sklearn/PyTorch

Don't just read. **Derive.** Every formula here comes from first principles. If you can't derive it, you don't understand it yet.

## Philosophy: No Black Boxes

We don't give you formulas to memorize. We show you how to derive them.

Why? Because when your model doesn't converge, or your loss explodes, or your accuracy plateaus, you need to understand what's happening mathematically. You can't debug what you don't understand.

Plus: ML research moves fast. Today's state-of-the-art is next year's baseline. But calculus, linear algebra, and optimization don't change. When you understand the foundations, you can adapt to any new architecture or loss function.

## What Comes Next

**Level 14 (Advanced Topics)** covers research frontiers: attention mechanisms, transformers, diffusion models, reinforcement learning theory. But honestly? You're already equipped to read ML papers and implement algorithms from first principles.

You could stop here and go build things. The rest is specialization.

## Let's Go

Four algorithms. Four chapters. Every piece of math you've learned for twelve levels, finally assembled into systems that learn from data.

You've been calling `model.fit()` for years. Now you're going to understand what it's actually computing.

---

**Start here:** [Chapter 1: Linear Regression](./01-linear-regression.md)
