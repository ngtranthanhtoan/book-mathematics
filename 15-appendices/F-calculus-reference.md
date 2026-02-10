# Appendix F: Calculus Reference

> Common derivatives, integrals, series expansions, and rules in compact table form. Focused on the functions that appear most often in ML.

---

## Differentiation Rules

| Rule | Formula |
|------|---------|
| Constant | $\frac{d}{dx}c = 0$ |
| Power | $\frac{d}{dx}x^n = nx^{n-1}$ |
| Sum | $(f + g)' = f' + g'$ |
| Product | $(fg)' = f'g + fg'$ |
| Quotient | $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$ |
| Chain | $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$ |

---

## Common Derivatives

| $f(x)$ | $f'(x)$ | Notes |
|---------|---------|-------|
| $x^n$ | $nx^{n-1}$ | Power rule |
| $e^x$ | $e^x$ | Its own derivative |
| $a^x$ | $a^x \ln a$ | General exponential |
| $\ln x$ | $1/x$ | Natural log |
| $\log_a x$ | $\frac{1}{x \ln a}$ | General log |
| $\sin x$ | $\cos x$ | |
| $\cos x$ | $-\sin x$ | |
| $\frac{1}{1+e^{-x}}$ (sigmoid) | $\sigma(x)(1 - \sigma(x))$ | Key for logistic regression |
| $\tanh x$ | $1 - \tanh^2 x$ | Also $= \text{sech}^2 x$ |
| $\max(0, x)$ (ReLU) | $\begin{cases}1 & x > 0 \\ 0 & x < 0\end{cases}$ | Undefined at $x = 0$, use subgradient |
| $\ln(1 + e^x)$ (softplus) | $\sigma(x)$ | Smooth approximation of ReLU |
| $x \cdot \sigma(x)$ (SiLU/Swish) | $\sigma(x) + x\sigma(x)(1-\sigma(x))$ | Used in modern architectures |

---

## Common Integrals

| $\int f(x)\,dx$ | Result |
|---------|--------|
| $\int x^n\,dx$ | $\frac{x^{n+1}}{n+1} + C \quad (n \neq -1)$ |
| $\int \frac{1}{x}\,dx$ | $\ln|x| + C$ |
| $\int e^x\,dx$ | $e^x + C$ |
| $\int e^{ax}\,dx$ | $\frac{1}{a}e^{ax} + C$ |
| $\int \ln x\,dx$ | $x\ln x - x + C$ |
| $\int \sin x\,dx$ | $-\cos x + C$ |
| $\int \cos x\,dx$ | $\sin x + C$ |

---

## Gaussian Integral (Used Everywhere in Probability)

$$\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}$$

$$\int_{-\infty}^{\infty} e^{-\frac{(x-\mu)^2}{2\sigma^2}}\,dx = \sigma\sqrt{2\pi}$$

This is why the normalizing constant of the Gaussian is $\frac{1}{\sigma\sqrt{2\pi}}$.

---

## Taylor / Maclaurin Series

Centered at $a = 0$ (Maclaurin):

| Function | Series | Convergence |
|----------|--------|-------------|
| $e^x$ | $\sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots$ | All $x$ |
| $\ln(1+x)$ | $\sum_{n=1}^{\infty} \frac{(-1)^{n+1}x^n}{n} = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots$ | $-1 < x \leq 1$ |
| $\frac{1}{1-x}$ | $\sum_{n=0}^{\infty} x^n = 1 + x + x^2 + \cdots$ | $|x| < 1$ |
| $\sin x$ | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!} = x - \frac{x^3}{6} + \cdots$ | All $x$ |
| $\cos x$ | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!} = 1 - \frac{x^2}{2} + \cdots$ | All $x$ |

**ML relevance**: First-order Taylor expansion ($f(x) \approx f(a) + f'(a)(x-a)$) is the foundation of gradient descent. Second-order expansion gives Newton's method.

---

## Multivariable Calculus

**Gradient**
$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$

**Jacobian** (for $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$)
$$\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}$$

**Hessian** (for $f: \mathbb{R}^n \to \mathbb{R}$)
$$\mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

**Chain Rule (Multivariate)**
$$\frac{\partial f}{\partial x_i} = \sum_j \frac{\partial f}{\partial z_j} \frac{\partial z_j}{\partial x_i}$$

This is backpropagation.

---

*Back to [Appendices](README.md)*
