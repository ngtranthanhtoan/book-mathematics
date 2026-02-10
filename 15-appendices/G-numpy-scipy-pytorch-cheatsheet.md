# Appendix G: NumPy / SciPy / PyTorch Cheat Sheet

> "I know the math, how do I write it in code?" Maps mathematical operations to library calls.

---

## Vectors & Basic Operations

| Math | NumPy | PyTorch |
|------|-------|---------|
| Create vector $\mathbf{x} = [1, 2, 3]$ | `np.array([1, 2, 3])` | `torch.tensor([1, 2, 3])` |
| Zero vector $\mathbf{0} \in \mathbb{R}^n$ | `np.zeros(n)` | `torch.zeros(n)` |
| Ones vector $\mathbf{1} \in \mathbb{R}^n$ | `np.ones(n)` | `torch.ones(n)` |
| $\mathbf{x} + \mathbf{y}$ | `x + y` | `x + y` |
| $c \cdot \mathbf{x}$ | `c * x` | `c * x` |
| $\mathbf{x} \cdot \mathbf{y}$ (dot product) | `np.dot(x, y)` or `x @ y` | `torch.dot(x, y)` |
| $\|\mathbf{x}\|_2$ (L2 norm) | `np.linalg.norm(x)` | `torch.norm(x)` |
| $\|\mathbf{x}\|_1$ (L1 norm) | `np.linalg.norm(x, 1)` | `torch.norm(x, 1)` |
| Element-wise product $\mathbf{x} \odot \mathbf{y}$ | `x * y` | `x * y` |

---

## Matrices

| Math | NumPy | PyTorch |
|------|-------|---------|
| Create matrix | `np.array([[1,2],[3,4]])` | `torch.tensor([[1,2],[3,4]])` |
| Identity $\mathbf{I}_n$ | `np.eye(n)` | `torch.eye(n)` |
| $\mathbf{AB}$ (matrix multiply) | `A @ B` | `A @ B` |
| $\mathbf{A}^\top$ | `A.T` | `A.T` or `A.t()` |
| $\mathbf{A}^{-1}$ | `np.linalg.inv(A)` | `torch.linalg.inv(A)` |
| $\det(\mathbf{A})$ | `np.linalg.det(A)` | `torch.linalg.det(A)` |
| $\text{tr}(\mathbf{A})$ | `np.trace(A)` | `torch.trace(A)` |
| $\text{rank}(\mathbf{A})$ | `np.linalg.matrix_rank(A)` | `torch.linalg.matrix_rank(A)` |
| Solve $\mathbf{Ax} = \mathbf{b}$ | `np.linalg.solve(A, b)` | `torch.linalg.solve(A, b)` |
| Pseudo-inverse $\mathbf{A}^+$ | `np.linalg.pinv(A)` | `torch.linalg.pinv(A)` |

---

## Decompositions

| Math | NumPy | PyTorch |
|------|-------|---------|
| Eigenvalues $\lambda$, eigenvectors $\mathbf{v}$ | `np.linalg.eig(A)` | `torch.linalg.eig(A)` |
| Symmetric eigen | `np.linalg.eigh(A)` | `torch.linalg.eigh(A)` |
| SVD: $\mathbf{U\Sigma V}^\top$ | `np.linalg.svd(A)` | `torch.linalg.svd(A)` |
| QR: $\mathbf{QR}$ | `np.linalg.qr(A)` | `torch.linalg.qr(A)` |
| Cholesky: $\mathbf{LL}^\top$ | `np.linalg.cholesky(A)` | `torch.linalg.cholesky(A)` |

---

## Calculus & Automatic Differentiation

| Math | PyTorch |
|------|---------|
| Track gradients | `x = torch.tensor(2.0, requires_grad=True)` |
| Compute $f(x)$ | `y = x**2 + 3*x` |
| Compute $\nabla f$ | `y.backward()` |
| Read $\frac{df}{dx}$ | `x.grad` |
| Stop gradient tracking | `x.detach()` or `with torch.no_grad():` |
| Jacobian | `torch.autograd.functional.jacobian(f, x)` |
| Hessian | `torch.autograd.functional.hessian(f, x)` |

---

## Probability & Statistics

| Math | NumPy / SciPy |
|------|--------------|
| Sample from $\mathcal{N}(\mu, \sigma^2)$ | `np.random.normal(mu, sigma, size)` |
| Sample from Uniform$[a, b]$ | `np.random.uniform(a, b, size)` |
| Gaussian PDF | `scipy.stats.norm.pdf(x, mu, sigma)` |
| Gaussian CDF | `scipy.stats.norm.cdf(x, mu, sigma)` |
| Mean $\mathbb{E}[X]$ | `np.mean(x)` |
| Variance $\text{Var}(X)$ | `np.var(x)` or `np.var(x, ddof=1)` for sample |
| Covariance matrix | `np.cov(X)` |
| Correlation matrix | `np.corrcoef(X)` |

| Math | PyTorch |
|------|---------|
| Sample from $\mathcal{N}(0, 1)$ | `torch.randn(size)` |
| Sample from Uniform$[0, 1)$ | `torch.rand(size)` |
| Softmax $\frac{e^{x_i}}{\sum e^{x_j}}$ | `torch.softmax(x, dim)` |
| Log-softmax (numerically stable) | `torch.log_softmax(x, dim)` |

---

## Common Activation Functions

| Math | PyTorch |
|------|---------|
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | `torch.sigmoid(x)` |
| $\tanh(x)$ | `torch.tanh(x)` |
| $\text{ReLU}(x) = \max(0, x)$ | `torch.relu(x)` |
| $\text{LeakyReLU}(x)$ | `torch.nn.functional.leaky_relu(x)` |
| $\text{GELU}(x)$ | `torch.nn.functional.gelu(x)` |
| $\text{SiLU}(x) = x\sigma(x)$ | `torch.nn.functional.silu(x)` |

---

## Loss Functions

| Math | PyTorch |
|------|---------|
| MSE: $\frac{1}{n}\sum(y - \hat{y})^2$ | `torch.nn.MSELoss()` |
| Cross-entropy: $-\sum y_i \log \hat{y}_i$ | `torch.nn.CrossEntropyLoss()` |
| Binary cross-entropy | `torch.nn.BCELoss()` |
| L1 loss: $\frac{1}{n}\sum|y - \hat{y}|$ | `torch.nn.L1Loss()` |
| Huber loss | `torch.nn.HuberLoss()` |
| KL divergence | `torch.nn.KLDivLoss()` |

---

## Useful Operations

| Math | NumPy | PyTorch |
|------|-------|---------|
| $e^x$ | `np.exp(x)` | `torch.exp(x)` |
| $\ln x$ | `np.log(x)` | `torch.log(x)` |
| $\log_2 x$ | `np.log2(x)` | `torch.log2(x)` |
| $|x|$ | `np.abs(x)` | `torch.abs(x)` |
| $\sum x_i$ | `np.sum(x)` | `torch.sum(x)` |
| $\arg\max$ | `np.argmax(x)` | `torch.argmax(x)` |
| $\arg\min$ | `np.argmin(x)` | `torch.argmin(x)` |
| Clip/clamp | `np.clip(x, lo, hi)` | `torch.clamp(x, lo, hi)` |
| Reshape | `x.reshape(shape)` | `x.view(shape)` or `x.reshape(shape)` |
| Broadcasting | Automatic | Automatic |

---

*Back to [Appendices](README.md)*
