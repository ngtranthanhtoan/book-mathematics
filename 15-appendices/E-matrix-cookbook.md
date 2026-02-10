# Appendix E: Matrix Cookbook

> Quick-reference for matrix calculus, identities, and decomposition properties. Essential for deriving gradients and verifying backpropagation math.

---

## Matrix Calculus Rules

These are the results you need most when deriving gradients for ML models.

### Scalar-by-Vector Derivatives

| $f(\mathbf{x})$ | $\frac{\partial f}{\partial \mathbf{x}}$ | Notes |
|----------|------------|-------|
| $\mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}$ | Linear function |
| $\mathbf{x}^\top \mathbf{A} \mathbf{x}$ | $(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ | Quadratic form. If $\mathbf{A}$ symmetric: $2\mathbf{A}\mathbf{x}$ |
| $\mathbf{x}^\top \mathbf{x}$ | $2\mathbf{x}$ | Squared L2 norm |
| $\|\mathbf{x} - \mathbf{a}\|_2^2$ | $2(\mathbf{x} - \mathbf{a})$ | MSE gradient (per sample) |

### Scalar-by-Matrix Derivatives

| $f(\mathbf{X})$ | $\frac{\partial f}{\partial \mathbf{X}}$ |
|----------|------------|
| $\text{tr}(\mathbf{AX})$ | $\mathbf{A}^\top$ |
| $\text{tr}(\mathbf{X}^\top\mathbf{A})$ | $\mathbf{A}$ |
| $\text{tr}(\mathbf{X}^\top\mathbf{AX})$ | $(\mathbf{A} + \mathbf{A}^\top)\mathbf{X}$ |
| $\text{tr}(\mathbf{AXB})$ | $\mathbf{A}^\top\mathbf{B}^\top$ |
| $\log\det(\mathbf{X})$ | $\mathbf{X}^{-\top}$ |

### The Chain Rule for Matrices

If $f = g(\mathbf{h}(\mathbf{x}))$:

$$\frac{\partial f}{\partial \mathbf{x}} = \frac{\partial \mathbf{h}}{\partial \mathbf{x}}^\top \frac{\partial g}{\partial \mathbf{h}}$$

This is the mathematical backbone of backpropagation.

---

## Essential Matrix Identities

### Transpose

$$(\mathbf{A}^\top)^\top = \mathbf{A}$$
$$(\mathbf{AB})^\top = \mathbf{B}^\top\mathbf{A}^\top$$
$$(\mathbf{A} + \mathbf{B})^\top = \mathbf{A}^\top + \mathbf{B}^\top$$

### Inverse

$$(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$$
$$(\mathbf{A}^\top)^{-1} = (\mathbf{A}^{-1})^\top$$
$$(\alpha\mathbf{A})^{-1} = \frac{1}{\alpha}\mathbf{A}^{-1}$$

### Trace

$$\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})$$
$$\text{tr}(\alpha\mathbf{A}) = \alpha\,\text{tr}(\mathbf{A})$$
$$\text{tr}(\mathbf{A}^\top) = \text{tr}(\mathbf{A})$$
$$\text{tr}(\mathbf{ABC}) = \text{tr}(\mathbf{CAB}) = \text{tr}(\mathbf{BCA}) \quad \text{(cyclic property)}$$
$$\text{tr}(\mathbf{A}) = \sum_i \lambda_i \quad \text{(sum of eigenvalues)}$$

### Determinant

$$\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$$
$$\det(\mathbf{A}^{-1}) = 1/\det(\mathbf{A})$$
$$\det(\mathbf{A}^\top) = \det(\mathbf{A})$$
$$\det(\alpha\mathbf{A}) = \alpha^n\det(\mathbf{A}) \quad \text{for } n \times n \text{ matrix}$$
$$\det(\mathbf{A}) = \prod_i \lambda_i \quad \text{(product of eigenvalues)}$$

---

## Matrix Decompositions Summary

| Decomposition | Form | Requirements | ML Usage |
|---------------|------|-------------|----------|
| **Eigendecomposition** | $\mathbf{A} = \mathbf{Q\Lambda Q}^{-1}$ | Square matrix | PCA, spectral methods |
| **SVD** | $\mathbf{A} = \mathbf{U\Sigma V}^\top$ | Any matrix | PCA, low-rank approximation, pseudoinverse |
| **LU** | $\mathbf{A} = \mathbf{LU}$ | Square matrix | Solving linear systems |
| **QR** | $\mathbf{A} = \mathbf{QR}$ | Any matrix | Least squares, numerical stability |
| **Cholesky** | $\mathbf{A} = \mathbf{LL}^\top$ | Symmetric positive definite | Gaussian sampling, covariance matrices |

---

## Woodbury Identity

$$(\mathbf{A} + \mathbf{UCV})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{VA}^{-1}\mathbf{U})^{-1}\mathbf{VA}^{-1}$$

Useful when $\mathbf{A}$ is easy to invert but $\mathbf{A} + \mathbf{UCV}$ is not. Appears in Bayesian linear regression and Kalman filters.

---

## Positive Definite Matrices (Quick Tests)

A symmetric matrix $\mathbf{A}$ is positive definite iff any of:
- $\mathbf{x}^\top\mathbf{A}\mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$
- All eigenvalues are positive
- All leading principal minors are positive (Sylvester's criterion)
- There exists an invertible $\mathbf{L}$ such that $\mathbf{A} = \mathbf{L}\mathbf{L}^\top$ (Cholesky exists)

---

*Back to [Appendices](README.md)*
