# Appendix B: Notation Quick Reference

> A symbol-to-meaning dictionary. When you encounter an unfamiliar symbol in a paper or chapter, look it up here.

---

## Sets & Logic

| Symbol | Name | Meaning | Example |
|--------|------|---------|---------|
| $\{\ \}$ | set braces | A collection of elements | $\{1, 2, 3\}$ |
| $\in$ | element of | "$x$ belongs to set $S$" | $3 \in \{1, 2, 3\}$ |
| $\notin$ | not element of | "$x$ does not belong to $S$" | $4 \notin \{1, 2, 3\}$ |
| $\subseteq$ | subset | Every element of $A$ is in $B$ | $\{1\} \subseteq \{1, 2\}$ |
| $\cup$ | union | Elements in $A$ or $B$ (or both) | $A \cup B$ |
| $\cap$ | intersection | Elements in both $A$ and $B$ | $A \cap B$ |
| $\setminus$ | set difference | Elements in $A$ but not in $B$ | $A \setminus B$ |
| $\emptyset$ | empty set | The set with no elements | $\{x : x \neq x\} = \emptyset$ |
| $|S|$ | cardinality | Number of elements in $S$ | $|\{a, b, c\}| = 3$ |
| $\mathbb{R}$ | reals | The set of all real numbers | $x \in \mathbb{R}$ |
| $\mathbb{R}^n$ | n-dim real space | Vectors with $n$ real components | $\mathbf{x} \in \mathbb{R}^{784}$ |
| $\mathbb{Z}$ | integers | $\{\ldots, -2, -1, 0, 1, 2, \ldots\}$ | $n \in \mathbb{Z}$ |
| $\mathbb{N}$ | natural numbers | $\{0, 1, 2, 3, \ldots\}$ (or from 1) | $n \in \mathbb{N}$ |
| $\forall$ | for all | "For every" | $\forall x \in \mathbb{R}$ |
| $\exists$ | there exists | "There is at least one" | $\exists x : x^2 = 4$ |
| $\Rightarrow$ | implies | "If $A$ then $B$" | $x > 0 \Rightarrow x^2 > 0$ |
| $\Leftrightarrow$ | if and only if | "$A$ implies $B$ and $B$ implies $A$" | $x = 0 \Leftrightarrow |x| = 0$ |
| $\neg$ | not / negation | Logical negation | $\neg(A \wedge B)$ |
| $\wedge$ | and | Logical conjunction | $A \wedge B$ |
| $\vee$ | or | Logical disjunction | $A \vee B$ |

---

## Arithmetic & Algebra

| Symbol | Name | Meaning |
|--------|------|---------|
| $\sum_{i=1}^{n}$ | summation | Add up terms from $i=1$ to $n$ |
| $\prod_{i=1}^{n}$ | product | Multiply terms from $i=1$ to $n$ |
| $|x|$ | absolute value | Distance from zero |
| $\lfloor x \rfloor$ | floor | Largest integer $\leq x$ |
| $\lceil x \rceil$ | ceiling | Smallest integer $\geq x$ |
| $n!$ | factorial | $n \times (n-1) \times \cdots \times 1$ |
| $\binom{n}{k}$ | binomial coefficient | "n choose k" = $\frac{n!}{k!(n-k)!}$ |
| $\propto$ | proportional to | $a \propto b$ means $a = cb$ for some constant $c$ |
| $\approx$ | approximately equal | Values are close but not exact |
| $:=$ or $\triangleq$ | defined as | Left side is defined to equal right side |
| $\log$ | logarithm | Natural log ($\ln$) in most ML contexts |
| $\exp(x)$ | exponential | $e^x$ |

---

## Linear Algebra

| Symbol | Name | Meaning |
|--------|------|---------|
| $\mathbf{x}$, $\mathbf{v}$ | bold lowercase | Vector (column by default) |
| $\mathbf{A}$, $\mathbf{W}$ | bold uppercase | Matrix |
| $\mathbf{A}^\top$ | transpose | Flip rows and columns |
| $\mathbf{A}^{-1}$ | inverse | $\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$ |
| $\mathbf{I}$ | identity matrix | Diagonal of ones |
| $\mathbf{0}$ | zero vector/matrix | All entries zero |
| $a_{ij}$ | matrix entry | Element at row $i$, column $j$ |
| $\mathbf{x} \cdot \mathbf{y}$ or $\mathbf{x}^\top \mathbf{y}$ | dot product | $\sum_i x_i y_i$ |
| $\|\mathbf{x}\|$ or $\|\mathbf{x}\|_2$ | L2 norm | Euclidean length: $\sqrt{\sum x_i^2}$ |
| $\|\mathbf{x}\|_1$ | L1 norm | Manhattan distance: $\sum |x_i|$ |
| $\|\mathbf{x}\|_p$ | Lp norm | $(\sum |x_i|^p)^{1/p}$ |
| $\text{tr}(\mathbf{A})$ | trace | Sum of diagonal elements |
| $\det(\mathbf{A})$ or $|\mathbf{A}|$ | determinant | Scalar measuring "volume scaling" |
| $\text{rank}(\mathbf{A})$ | rank | Number of linearly independent columns |
| $\lambda$ | eigenvalue | $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ |
| $\mathbf{v}$ | eigenvector | Direction unchanged by $\mathbf{A}$ |
| $\odot$ | Hadamard product | Element-wise multiplication |
| $\otimes$ | Kronecker/outer product | Tensor product of two matrices/vectors |
| $\text{diag}(\mathbf{v})$ | diagonal matrix | Matrix with $\mathbf{v}$ on diagonal |

---

## Calculus

| Symbol | Name | Meaning |
|--------|------|---------|
| $\frac{df}{dx}$ or $f'(x)$ | derivative | Rate of change of $f$ with respect to $x$ |
| $\frac{\partial f}{\partial x}$ | partial derivative | Derivative holding other variables constant |
| $\nabla f$ | gradient | Vector of all partial derivatives |
| $\nabla^2 f$ or $\mathbf{H}$ | Hessian | Matrix of second partial derivatives |
| $\mathbf{J}$ | Jacobian | Matrix of partial derivatives for vector-valued functions |
| $\int_a^b f(x)\,dx$ | definite integral | Area under $f$ from $a$ to $b$ |
| $\frac{d}{dx}$ | derivative operator | "Take the derivative with respect to $x$" |
| $\Delta x$ | finite difference | A discrete change in $x$ |
| $dx$ | infinitesimal | An infinitely small change in $x$ |
| $\lim_{x \to a}$ | limit | Value $f(x)$ approaches as $x$ approaches $a$ |
| $O(\cdot)$ | Big-O | Asymptotic upper bound (also used in CS) |
| $\arg\min_x f(x)$ | argmin | The value of $x$ that minimizes $f$ |
| $\arg\max_x f(x)$ | argmax | The value of $x$ that maximizes $f$ |

---

## Probability & Statistics

| Symbol | Name | Meaning |
|--------|------|---------|
| $P(A)$ | probability | Probability of event $A$ |
| $P(A \mid B)$ | conditional probability | Probability of $A$ given $B$ occurred |
| $X, Y, Z$ | random variables | Variables with probabilistic outcomes |
| $p(x)$ or $f(x)$ | PDF / PMF | Probability density/mass function |
| $F(x)$ | CDF | Cumulative distribution function: $P(X \leq x)$ |
| $\mathbb{E}[X]$ or $\mu$ | expected value | Average outcome: $\sum x \cdot p(x)$ |
| $\text{Var}(X)$ or $\sigma^2$ | variance | Spread: $\mathbb{E}[(X - \mu)^2]$ |
| $\text{Cov}(X,Y)$ | covariance | How $X$ and $Y$ vary together |
| $\rho_{XY}$ | correlation | Normalized covariance $\in [-1, 1]$ |
| $X \sim \mathcal{N}(\mu, \sigma^2)$ | distributed as | $X$ follows a Normal distribution |
| $\mathcal{N}(\mu, \sigma^2)$ | Normal/Gaussian | Bell curve distribution |
| $X \perp Y$ | independent | $P(X, Y) = P(X)P(Y)$ |
| $\hat{\theta}$ | estimator | Estimated value of parameter $\theta$ |
| $\mathcal{L}(\theta)$ | likelihood | Probability of data given parameters |

---

## Optimization

| Symbol | Name | Meaning |
|--------|------|---------|
| $\min_x f(x)$ | minimize | Find $x$ that gives smallest $f(x)$ |
| $\text{s.t.}$ | subject to | Constraints follow |
| $\mathcal{L}$ | loss / Lagrangian | Function to optimize (context-dependent) |
| $\nabla_\theta \mathcal{L}$ | gradient of loss | Direction of steepest ascent in parameter space |
| $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ | gradient descent update | One step of training |
| $\|\theta\|_2^2$ | L2 penalty | Ridge regularization term |
| $\|\theta\|_1$ | L1 penalty | Lasso regularization term |
| $\lambda$ | regularization strength | Controls penalty magnitude |
| $x^*$ | optimal value | The solution to the optimization problem |

---

## Information Theory

| Symbol | Name | Meaning |
|--------|------|---------|
| $H(X)$ | entropy | Average surprise / uncertainty |
| $H(X \mid Y)$ | conditional entropy | Remaining uncertainty given $Y$ |
| $I(X;Y)$ | mutual information | Shared information between $X$ and $Y$ |
| $H(p, q)$ | cross-entropy | Expected surprise using model $q$ when reality is $p$ |
| $D_\text{KL}(p \| q)$ | KL divergence | Information lost using $q$ to approximate $p$ |

---

## Conventions Used in This Book

| Convention | Meaning |
|------------|---------|
| Bold lowercase ($\mathbf{x}$) | Vector |
| Bold uppercase ($\mathbf{A}$) | Matrix |
| Italic lowercase ($x, \alpha$) | Scalar |
| Calligraphic ($\mathcal{L}, \mathcal{N}$) | Loss functions, distributions, sets |
| Blackboard bold ($\mathbb{R}, \mathbb{E}$) | Number sets, expectation operator |
| Hat ($\hat{y}$) | Estimated / predicted value |
| Star ($x^*$) | Optimal value |
| Bar ($\bar{x}$) | Sample mean |
| Tilde ($\tilde{x}$) | Modified / approximate version |

---

*Back to [Appendices](README.md)*
