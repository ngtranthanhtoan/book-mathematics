# Appendix C: Key Theorems & Identities

> The results you will reach for most often, collected in one place. Each entry includes the chapter where it is covered in detail.

---

## Algebra

**Quadratic Formula** (Level 2)
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

**Binomial Theorem** (Level 2)
$$(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k$$

**Logarithm Properties** (Level 2)
$$\log(ab) = \log a + \log b, \quad \log(a/b) = \log a - \log b, \quad \log(a^n) = n \log a$$

**Change of Base**
$$\log_b a = \frac{\log_c a}{\log_c b}$$

---

## Linear Algebra

**Matrix Transpose Properties** (Level 4)
$$(\mathbf{AB})^\top = \mathbf{B}^\top\mathbf{A}^\top, \quad (\mathbf{A}^\top)^\top = \mathbf{A}$$

**Matrix Inverse Properties** (Level 4)
$$(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}, \quad (\mathbf{A}^\top)^{-1} = (\mathbf{A}^{-1})^\top$$

**Eigenvalue Equation** (Level 4)
$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

**Spectral Theorem** (Level 4): A real symmetric matrix $\mathbf{A}$ can be decomposed as $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$ where $\mathbf{Q}$ is orthogonal and $\mathbf{\Lambda}$ is diagonal.

**SVD** (Level 4)
$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$$

**Sherman-Morrison Formula** (Level 4)
$$(\mathbf{A} + \mathbf{uv}^\top)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{uv}^\top\mathbf{A}^{-1}}{1 + \mathbf{v}^\top\mathbf{A}^{-1}\mathbf{u}}$$

**Trace Properties** (Level 4)
$$\text{tr}(\mathbf{ABC}) = \text{tr}(\mathbf{CAB}) = \text{tr}(\mathbf{BCA})$$

---

## Calculus

**Derivative Rules** (Level 6)

| Rule | Formula |
|------|---------|
| Power | $(x^n)' = nx^{n-1}$ |
| Product | $(fg)' = f'g + fg'$ |
| Quotient | $(f/g)' = \frac{f'g - fg'}{g^2}$ |
| Chain | $(f \circ g)' = f'(g(x)) \cdot g'(x)$ |

**Key Derivatives** (Level 6)

| $f(x)$ | $f'(x)$ |
|---------|---------|
| $e^x$ | $e^x$ |
| $\ln x$ | $1/x$ |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1 - \sigma(x))$ |
| $\tanh(x)$ | $1 - \tanh^2(x)$ |
| $\text{ReLU}(x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$ |

**Fundamental Theorem of Calculus** (Level 6)
$$\frac{d}{dx}\int_a^x f(t)\,dt = f(x)$$

**Taylor Expansion** (Level 6, 12)
$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

---

## Probability

**Bayes' Theorem** (Level 7)
$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

**Law of Total Probability** (Level 7)
$$P(B) = \sum_i P(B \mid A_i) P(A_i)$$

**Linearity of Expectation** (Level 7)
$$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y] \quad \text{(always, even if dependent)}$$

**Variance** (Level 7)
$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Variance of Sum** (Level 7)
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

**Law of Large Numbers** (Level 8): Sample average converges to expected value as sample size grows.

**Central Limit Theorem** (Level 8): Sum of many independent random variables is approximately normally distributed, regardless of their individual distributions.

---

## Information Theory

**Entropy** (Level 10)
$$H(X) = -\sum_x p(x) \log p(x)$$

**Cross-Entropy** (Level 10)
$$H(p, q) = -\sum_x p(x) \log q(x)$$

**KL Divergence** (Level 10)
$$D_\text{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

**Gibbs' Inequality**: $D_\text{KL}(p \| q) \geq 0$ with equality iff $p = q$.

---

## Optimization

**Gradient Descent Update** (Level 9)
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

**Normal Equation** (Level 13)
$$\hat{\theta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

**Lagrangian** (Level 9)
$$\mathcal{L}(x, \lambda) = f(x) + \lambda g(x) \quad \text{for constraint } g(x) = 0$$

**Bias-Variance Decomposition** (Level 7, 8)
$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

---

## Trigonometric Identities (Reference)

$$\sin^2\theta + \cos^2\theta = 1$$

$$e^{i\theta} = \cos\theta + i\sin\theta \quad \text{(Euler's formula)}$$

---

*Back to [Appendices](README.md)*
