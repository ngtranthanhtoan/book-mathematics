# Appendix H: Glossary

> Definitions of key terms used throughout the book. The chapter in parentheses is where the term is first introduced or covered in depth.

---

## A

**Abstraction** (Level 0): The process of removing irrelevant details to focus on essential properties. In code: extracting an interface. In math: going from "5 apples" to "the number 5."

**Activation function** (Level 3): A nonlinear function applied element-wise after a linear transformation in a neural network. Common ones: ReLU, sigmoid, tanh.

**Argmax / Argmin** (Level 6): The input value that maximizes / minimizes a function. $\arg\max_x f(x)$ returns $x$, not $f(x)$.

---

## B

**Backpropagation** (Level 6, 13): The chain rule applied recursively through a computational graph to compute gradients of the loss with respect to every parameter.

**Basis** (Level 4): A set of linearly independent vectors that span a vector space. Every vector in the space can be written as a unique combination of basis vectors.

**Bayes' theorem** (Level 7): $P(A|B) = P(B|A)P(A)/P(B)$. Updates beliefs given new evidence.

**Bias (statistical)** (Level 8): The systematic error of an estimator. $\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$.

**Bias (neural network)** (Level 13): The constant term $b$ in $y = Wx + b$. Shifts the activation function.

**Bias-variance tradeoff** (Level 8): Decomposition of prediction error into bias (underfitting), variance (overfitting), and irreducible noise.

---

## C

**Central Limit Theorem** (Level 8): The sum of many independent random variables is approximately normally distributed, regardless of their individual distributions.

**Chain rule** (Level 6): $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$. The mathematical rule behind backpropagation.

**Condition number** (Level 12): Measures how sensitive a computation is to small changes in input. High condition number means numerically unstable.

**Convex function** (Level 9): A function where the line segment between any two points on the graph lies above or on the graph. Convex functions have no local minima that are not global.

**Covariance** (Level 7): Measures how two random variables vary together. $\text{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$.

**Cross-entropy** (Level 10): $H(p,q) = -\sum p(x)\log q(x)$. Measures how well distribution $q$ represents reality $p$. The standard classification loss function.

---

## D

**Derivative** (Level 6): The instantaneous rate of change of a function. The slope of the tangent line.

**Determinant** (Level 4): A scalar that measures how a matrix scales volume. Zero determinant means the matrix is singular (not invertible).

**Dimensionality reduction** (Level 13): Projecting high-dimensional data into fewer dimensions while preserving important structure. PCA is the classic method.

**Dot product** (Level 4): $\mathbf{x} \cdot \mathbf{y} = \sum x_i y_i$. Measures alignment between vectors. Also equals $\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$.

---

## E

**Eigenvalue / Eigenvector** (Level 4): For matrix $\mathbf{A}$, vector $\mathbf{v}$ is an eigenvector with eigenvalue $\lambda$ if $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$. The matrix only scales $\mathbf{v}$, does not change its direction.

**Entropy** (Level 10): $H(X) = -\sum p(x)\log p(x)$. Average surprise or uncertainty in a random variable.

**Expectation** (Level 7): $\mathbb{E}[X] = \sum x \cdot p(x)$. The average outcome, weighted by probability. Also called the mean.

---

## F

**Feature** (Level 2): An individual measurable property of a data point. A column in your data matrix.

**Function** (Level 3): A mapping from a domain to a codomain that assigns exactly one output to each input.

---

## G

**Gradient** (Level 6): The vector of partial derivatives $\nabla f = (\partial f/\partial x_1, \ldots, \partial f/\partial x_n)$. Points in the direction of steepest ascent.

**Gradient descent** (Level 9): Iterative optimization: $\theta \leftarrow \theta - \eta \nabla \mathcal{L}$. Move parameters in the direction that decreases loss.

---

## H

**Hessian** (Level 6): Matrix of second partial derivatives. Describes the curvature of a function. Used in Newton's method.

**Hypothesis testing** (Level 8): Framework for deciding whether observed data is consistent with a null hypothesis, using p-values and significance levels.

---

## I

**Independence** (Level 7): Events $A$ and $B$ are independent if $P(A \cap B) = P(A)P(B)$. Knowing $A$ tells you nothing about $B$.

**Invariant** (Level 0): A property that does not change under a transformation. Loop invariants, class invariants, and geometric invariants are all the same concept.

---

## J

**Jacobian** (Level 6): Matrix of all first-order partial derivatives for a vector-valued function. Generalizes the derivative to functions $\mathbb{R}^n \to \mathbb{R}^m$.

---

## K

**KL divergence** (Level 10): $D_{KL}(p\|q) = \sum p(x)\log(p(x)/q(x))$. Information lost when approximating $p$ with $q$. Always non-negative. Not symmetric.

---

## L

**Learning rate** (Level 9): The step size $\eta$ in gradient descent. Too large: divergence. Too small: slow convergence.

**Likelihood** (Level 8): $\mathcal{L}(\theta | \text{data}) = P(\text{data} | \theta)$. How probable the observed data is, given parameters $\theta$.

**Linear independence** (Level 4): Vectors are linearly independent if no vector in the set can be written as a linear combination of the others.

**Loss function** (Level 9): The function being minimized during training. Measures how wrong the model's predictions are.

---

## M

**Matrix** (Level 4): A rectangular array of numbers. Represents a linear transformation, a system of equations, or a dataset.

**Maximum likelihood estimation (MLE)** (Level 8): Finding the parameters $\theta$ that maximize $P(\text{data} | \theta)$. What `model.fit()` usually does.

---

## N

**Norm** (Level 4): A function that measures the "size" or "length" of a vector. L2 norm is Euclidean length. L1 norm is Manhattan distance.

**Normal distribution** (Level 7): The bell curve. $\mathcal{N}(\mu, \sigma^2)$. Central to statistics because of the Central Limit Theorem.

---

## O

**Optimization** (Level 9): Finding the input that minimizes (or maximizes) a function. Training a model is an optimization problem.

**Overfitting** (Level 8): When a model memorizes training data instead of learning the underlying pattern. Low training error, high test error.

---

## P

**PCA (Principal Component Analysis)** (Level 13): Projects data onto directions of maximum variance. Mathematically: eigendecomposition of the covariance matrix.

**Probability density function (PDF)** (Level 7): For continuous random variables, gives relative likelihood. Area under the curve equals probability.

**p-value** (Level 8): The probability of observing data at least as extreme as what was observed, assuming the null hypothesis is true.

---

## R

**Rank** (Level 4): The number of linearly independent rows (or columns) of a matrix. Determines whether $\mathbf{Ax} = \mathbf{b}$ has a unique solution.

**Regularization** (Level 9): Adding a penalty term to the loss to discourage complexity. L2 (Ridge): penalizes large weights. L1 (Lasso): promotes sparsity.

---

## S

**Sigmoid** (Level 3): $\sigma(x) = 1/(1+e^{-x})$. Maps any real number to $(0, 1)$. Used in logistic regression and as a gate in LSTMs.

**Singular value decomposition (SVD)** (Level 4): $\mathbf{A} = \mathbf{U\Sigma V}^\top$. The "Swiss army knife" of matrix decompositions. Works for any matrix.

**Softmax** (Level 3): $\text{softmax}(x_i) = e^{x_i} / \sum_j e^{x_j}$. Converts a vector of real numbers into a probability distribution.

**Span** (Level 4): The set of all linear combinations of a set of vectors. If vectors span $\mathbb{R}^n$, they can represent any point in that space.

---

## T

**Tensor** (Level 4): A multidimensional array. Scalar (0D), vector (1D), matrix (2D), and higher. The fundamental data structure in deep learning frameworks.

**Transpose** (Level 4): Swapping rows and columns of a matrix. $(\mathbf{A}^\top)_{ij} = \mathbf{A}_{ji}$.

---

## V

**Variance** (Level 7): $\text{Var}(X) = \mathbb{E}[(X - \mu)^2]$. Measures spread around the mean. Standard deviation is $\sqrt{\text{Var}(X)}$.

**Vector** (Level 4): An ordered list of numbers. Represents a point in space, a direction, or a data sample.

**Vector space** (Level 4): A set of vectors closed under addition and scalar multiplication. The "playing field" of linear algebra.

---

*Back to [Appendices](README.md)*
