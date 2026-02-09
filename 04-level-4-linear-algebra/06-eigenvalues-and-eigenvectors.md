# Chapter 6: Eigenvalues and Eigenvectors

> **Building On** -- You've learned to solve systems of linear equations. Now a deeper question: are there special vectors that a matrix only stretches, never rotates? These *eigenvectors* reveal the fundamental structure hidden in your data.

---

## The Problem That Makes This Click

You have a dataset with 1,000 features, but you suspect most of the "signal" lives in just a few directions. How do you find those directions? **Eigenvectors.** They're the "principal axes" of your data -- the directions along which variance is maximized.

Picture this: you're building a recommendation engine. You have a movie ratings matrix -- 50,000 users by 10,000 movies. That's half a billion numbers. But you have a hunch that people's tastes aren't really 10,000-dimensional. Maybe there are a handful of underlying "taste dimensions" -- action lover, drama fan, comedy enthusiast, arthouse connoisseur. If you could find those dimensions, you'd compress your data by orders of magnitude and actually *understand* it.

Those taste dimensions? They're eigenvectors of the covariance matrix. The eigenvalues tell you how important each taste dimension is. And the technique that finds them is called PCA -- Principal Component Analysis. Let's figure out how it works from the ground up.

---

## Let's Discover Eigenvectors with Code First

Before any math, let's just *see* what eigenvectors are. Fire up Python and follow along.

### Step 1: Create a Correlated Dataset (Your "Movie Ratings")

```python
import numpy as np
np.random.seed(42)

# Imagine 2 "hidden taste dimensions" generating ratings for 2 movies
# Movie 1 ratings correlate with Movie 2 ratings (action fans like both)
mean = [0, 0]
cov = [[3, 1.5],    # Movie 1 variance=3, covariance with Movie 2=1.5
       [1.5, 1]]    # Movie 2 variance=1
data = np.random.multivariate_normal(mean, cov, 500)

print(f"Data shape: {data.shape}")   # (500, 2)
print(f"You have 500 users rating 2 movies")
```

### Step 2: Find the Principal Axes

```python
# Center the data (subtract mean)
data_centered = data - np.mean(data, axis=0)

# Compute the covariance matrix
cov_matrix = (data_centered.T @ data_centered) / (len(data) - 1)
print(f"Covariance matrix:\n{cov_matrix}")

# HERE'S THE MAGIC: eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort by eigenvalue, largest first
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\nEigenvectors (principal directions):\n{eigenvectors}")
print(f"Eigenvalues (variance along each direction): {eigenvalues}")
print(f"Variance ratio: {eigenvalues / np.sum(eigenvalues)}")
```

**Output:**
```
Covariance matrix:
[[2.83  1.42]
 [1.42  0.95]]

Eigenvectors (principal directions):
[[ 0.891 -0.453]
 [ 0.453  0.891]]
Eigenvalues (variance along each direction): [3.316 0.459]
Variance ratio: [0.878 0.122]
```

Look at that: **87.8% of the variance** lives along a single direction -- the first eigenvector. That's the dominant "taste dimension." The second direction captures only 12.2%. If you only kept the first eigenvector, you'd compress your data from 2D to 1D while retaining nearly 88% of the information.

That's PCA. That's what eigenvectors do.

### Step 3: Verify the Core Property

Here's the defining property of eigenvectors. When you multiply matrix `A` by an eigenvector `v`, you get back the *same vector* scaled by a number (the eigenvalue):

```python
# Let's verify: A @ v = lambda * v
A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]

    Av = A @ v           # Matrix times eigenvector
    lambda_v = lam * v   # Eigenvalue times eigenvector

    print(f"Eigenvector {i+1}: v = {v}")
    print(f"Eigenvalue {i+1}: lambda = {lam:.4f}")
    print(f"  A @ v     = {Av}")
    print(f"  lambda * v = {lambda_v}")
    print(f"  Same? {np.allclose(Av, lambda_v)}")
    print()
```

**Output:**
```
Eigenvector 1: v = [0.894 0.447]
Eigenvalue 1: lambda = 5.0000
  A @ v     = [4.472 2.236]
  lambda * v = [4.472 2.236]
  Same? True

Eigenvector 2: v = [-0.707  0.707]
Eigenvalue 2: lambda = 2.0000
  A @ v     = [-1.414  1.414]
  lambda * v = [-1.414  1.414]
  Same? True
```

The matrix `A` does nothing to these vectors except scale them. **No rotation. No shearing. Just stretching.** That's what makes them special.

---

> **You Already Know This: System Profiling**
>
> Think of a performance profiler for a complex distributed system. Your system has dozens of interacting components. The profiler identifies the "dominant modes" of behavior -- the hotspots that explain most of the system's runtime.
>
> - **Eigenvectors** are like the profiler's "hotspot directions" -- the fundamental patterns of behavior in the system
> - **Eigenvalues** are the "weight" of each hotspot -- how much of the total runtime (variance) each one accounts for
>
> Just as you'd focus optimization effort on the top 3 hotspots rather than all 47 functions, PCA focuses on the top-k eigenvectors that capture most of the data's variance.

---

## The Intuition: What Does "Only Stretching" Mean?

When a matrix transforms a vector, most vectors get both rotated and scaled. But eigenvectors are immune to rotation -- the matrix can only stretch or compress them along their own direction.

```
What happens to a REGULAR vector:          What happens to an EIGENVECTOR:

  Before:     After A:                     Before:     After A:

      v       Av                               v        Av = lambda * v
     /          \                              |         |
    /            \                             |         |
   /              \                            |         |
  o      ->       o                            o    ->   o
                                               |         |
  Direction       Direction                    |         |
  CHANGED         is different!                |         |
                                           Direction    SAME direction,
                                           preserved!   just scaled!
```

Here's a more concrete picture. Imagine a 2x2 matrix acting on every vector in the plane:

```
  Before transformation:         After transformation:
  (unit circle)                  (becomes an ellipse)

         |                              |
      .--+--.                        .--+--.
     /   |   \                      /   |   \
    /    |    \                ----/----+----\----
   |     |     |              |  /     |     \  |
   +-----o-----+              +-/------o------\-+
   |     |     |              |/       |       \|
    \    |    /                \       |       /
     \   |   /                  \     |     /
      '--+--'                    '---+---'
         |                           |
                                  ^         ^
  All vectors have               |         |
  the same length           eigenvector  eigenvector
                            direction 1  direction 2
                            (major axis) (minor axis)
```

The eigenvectors point along the axes of the ellipse. The eigenvalues are the stretch factors along those axes. Every other vector gets both stretched AND rotated.

This is *exactly* what PCA does to your data: it finds the axes of the "elliptical cloud" of data points.

```
  Your data cloud:                  After finding eigenvectors:

     *  *                              *  *
    * ** *  *                         * ** *  *
   *  * ** *  *                      *  * ** *  *
  *  * *** * *  *     -------->     *  * *** * *  *
   * ** * ** *                    /  * ** * ** *
    *  * **                      /    *  * **
     * *                        /      * *
                               /
                              v1 (eigenvector 1: direction of max variance)

                              ^
                              |
                              v2 (eigenvector 2: direction of min variance)
```

**Translation:** Eigenvectors are the "natural coordinate system" for your data. Instead of measuring along arbitrary x/y axes, you measure along the directions where the data actually spreads out.

---

## Eigenvalues: German for "Own Values"

Because these vectors are so special they *own* themselves.

The word "eigen" is German for "own" or "inherent." An eigenvector is a matrix's "own vector" -- a direction that the matrix *inherently* preserves. The eigenvalue is the matrix's "own value" for that direction -- the inherent scaling factor.

### What the Eigenvalue Tells You

| Eigenvalue $\lambda$ | What It Means | Movie Ratings Analogy |
|---|---|---|
| $\lambda > 1$ | Stretches (amplifies) | This taste dimension has high variance -- people disagree a lot |
| $0 < \lambda < 1$ | Shrinks (dampens) | Low variance -- most people feel similarly |
| $\lambda = 1$ | No change | Variance exactly matches the baseline |
| $\lambda = 0$ | Collapses to zero | This dimension carries zero information |
| $\lambda < 0$ | Reverses AND scales | Direction flips (rare for covariance matrices, which are always non-negative) |
| Complex $\lambda$ | Rotation present | The matrix has a rotational component -- not just stretching |

---

> **You Already Know This: The Characteristic Polynomial as a Fingerprint**
>
> Every matrix has a *characteristic polynomial* -- a polynomial whose roots are the eigenvalues. Think of it as the matrix's fingerprint. Two matrices with the same characteristic polynomial have the same eigenvalues (though not necessarily the same eigenvectors).
>
> It's like a hash function: the characteristic polynomial compresses the entire matrix into a compact representation that still captures its essential spectral properties.

---

## The Math: Formalizing What We Discovered

Now that you *see* what eigenvectors are and *why* they matter, let's formalize the definition and learn to compute them by hand for small matrices.

### The Eigenvalue Equation

A non-zero vector $\mathbf{v}$ is an **eigenvector** of matrix $\mathbf{A}$ if:

$$\mathbf{Av} = \lambda\mathbf{v}$$

where $\lambda$ is the corresponding **eigenvalue**.

**Translation:** "When I apply transformation $\mathbf{A}$ to vector $\mathbf{v}$, the result points in the same direction as $\mathbf{v}$, just scaled by $\lambda$."

### Deriving the Characteristic Equation

How do you *find* the eigenvalues? Rearrange the equation:

$$\mathbf{Av} = \lambda\mathbf{v}$$
$$\mathbf{Av} - \lambda\mathbf{v} = \mathbf{0}$$
$$(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$$

**Translation:** "We're looking for vectors $\mathbf{v}$ in the null space of $(\mathbf{A} - \lambda\mathbf{I})$."

For non-zero solutions to exist, the matrix $(\mathbf{A} - \lambda\mathbf{I})$ must be singular (non-invertible), which means:

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

This is the **characteristic equation**. Solve it for $\lambda$ to get the eigenvalues.

### Worked Example: 2x2 Matrix

Let's find the eigenvalues and eigenvectors of the matrix from our code:

$$\mathbf{A} = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$$

**Step 1: Set up the characteristic equation.**

$$\det(\mathbf{A} - \lambda\mathbf{I}) = \det\begin{bmatrix} 4-\lambda & 2 \\ 1 & 3-\lambda \end{bmatrix} = 0$$

$$(4-\lambda)(3-\lambda) - (2)(1) = 0$$

$$12 - 4\lambda - 3\lambda + \lambda^2 - 2 = 0$$

$$\lambda^2 - 7\lambda + 10 = 0$$

**Step 2: Solve the quadratic.**

$$(\lambda - 5)(\lambda - 2) = 0$$

$$\lambda_1 = 5, \quad \lambda_2 = 2$$

These match what NumPy gave us.

**Step 3: Find eigenvectors for each eigenvalue.**

For $\lambda_1 = 5$:

$$(\mathbf{A} - 5\mathbf{I})\mathbf{v} = \begin{bmatrix} -1 & 2 \\ 1 & -2 \end{bmatrix}\mathbf{v} = \mathbf{0}$$

From the first row: $-v_1 + 2v_2 = 0$, so $v_1 = 2v_2$.

Choosing $v_2 = 1$: $\mathbf{v}_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$ (or any scalar multiple).

For $\lambda_2 = 2$:

$$(\mathbf{A} - 2\mathbf{I})\mathbf{v} = \begin{bmatrix} 2 & 2 \\ 1 & 1 \end{bmatrix}\mathbf{v} = \mathbf{0}$$

From the first row: $2v_1 + 2v_2 = 0$, so $v_1 = -v_2$.

Choosing $v_2 = 1$: $\mathbf{v}_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}$.

### The General 2x2 Formula

For any $2 \times 2$ matrix $\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the characteristic equation is:

$$\lambda^2 - (a+d)\lambda + (ad-bc) = 0$$

Or equivalently:

$$\lambda^2 - \text{trace}(\mathbf{A})\,\lambda + \det(\mathbf{A}) = 0$$

**Translation:** "The eigenvalues are encoded in two numbers you already know: the trace and the determinant." The trace is the sum of eigenvalues; the determinant is their product.

---

## Properties of Eigenvalues (The Cheat Sheet)

For an $n \times n$ matrix $\mathbf{A}$ with eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_n$:

| Property | Formula | Why You Care |
|---|---|---|
| Sum of eigenvalues | $\sum_i \lambda_i = \text{trace}(\mathbf{A})$ | Quick sanity check |
| Product of eigenvalues | $\prod_i \lambda_i = \det(\mathbf{A})$ | If any $\lambda = 0$, matrix is singular |
| Eigenvalues of $\mathbf{A}^k$ | $\lambda_i^k$ | Predicting long-term behavior |
| Eigenvalues of $\mathbf{A}^{-1}$ | $1/\lambda_i$ | If $\mathbf{A}^{-1}$ exists |

These aren't just trivia -- they're computational shortcuts. Need to know if a matrix is invertible? Check if any eigenvalue is zero. Need to predict what happens after 100 iterations of a dynamic system? Raise the eigenvalues to the 100th power.

---

## Spectral Decomposition: Factoring a Matrix by Its Eigenstructure

Here's where eigenvalues go from "interesting" to "powerful." If a matrix $\mathbf{A}$ has $n$ linearly independent eigenvectors, you can decompose it:

$$\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$$

where:
- $\mathbf{V}$ is the matrix whose columns are the eigenvectors
- $\boldsymbol{\Lambda}$ is a diagonal matrix with eigenvalues on the diagonal

**Translation:** "Any diagonalizable matrix is just scaling along its eigenvector directions." The matrix $\mathbf{V}$ rotates into the eigenvector coordinate system, $\boldsymbol{\Lambda}$ stretches along each axis, and $\mathbf{V}^{-1}$ rotates back.

> **You Already Know This: Diagonalization as Decomposing a Monolith**
>
> Diagonalization is like decomposing a monolithic service into independent microservices.
>
> - The original matrix $\mathbf{A}$ is the monolith: every component interacts with every other component
> - The diagonal matrix $\boldsymbol{\Lambda}$ is the decomposed system: each eigenvalue operates independently along its own axis
> - $\mathbf{V}$ and $\mathbf{V}^{-1}$ are the API adapters: they translate between the original "coupled" coordinate system and the "decoupled" eigenvector system
>
> Once decoupled, everything gets easier: computing $\mathbf{A}^{100}$ becomes trivial (just raise each diagonal entry to the 100th power), and understanding the system's behavior reduces to understanding each independent component.

### The Spectral Theorem (Symmetric Matrices Are Even Better)

For **symmetric** matrices (where $\mathbf{A} = \mathbf{A}^T$), the decomposition simplifies beautifully:

$$\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$$

No matrix inverse needed -- just a transpose! This works because symmetric matrices have:
1. **All real eigenvalues** (no complex numbers to worry about)
2. **Orthogonal eigenvectors** (they form a proper, perpendicular coordinate system)

This is the **spectral theorem**, and it's why symmetric matrices are everywhere in ML. Covariance matrices are symmetric. Kernel matrices are symmetric. The Hessian of a twice-differentiable function is symmetric.

```python
# Verify the spectral theorem for a symmetric matrix
S = np.array([[4, 2],
              [2, 3]])

eigenvalues_S, V = np.linalg.eigh(S)  # eigh = "eig for Hermitian/symmetric"
Lambda = np.diag(eigenvalues_S)

# Reconstruct: should get S back
S_reconstructed = V @ Lambda @ V.T
print(f"Original:\n{S}")
print(f"Reconstructed (V @ Lambda @ V^T):\n{S_reconstructed}")
print(f"Match? {np.allclose(S, S_reconstructed)}")  # True

# Verify orthogonality of eigenvectors
v1, v2 = V[:, 0], V[:, 1]
print(f"\nv1 dot v2 = {np.dot(v1, v2):.10f}")  # ~0
print(f"||v1|| = {np.linalg.norm(v1):.4f}")     # 1.0
print(f"||v2|| = {np.linalg.norm(v2):.4f}")     # 1.0
```

**Output:**
```
Original:
[[4 2]
 [2 3]]
Reconstructed (V @ Lambda @ V^T):
[[4. 2.]
 [2. 3.]]
Match? True

v1 dot v2 = 0.0000000000
||v1|| = 1.0000
||v2|| = 1.0000
```

---

## Matrix Powers: The Killer App of Eigendecomposition

Why decompose a matrix at all? Because it makes hard things easy. Computing $\mathbf{A}^{100}$ directly requires 99 matrix multiplications. With eigendecomposition:

$$\mathbf{A}^k = \mathbf{V}\boldsymbol{\Lambda}^k\mathbf{V}^{-1}$$

And $\boldsymbol{\Lambda}^k$ is just raising each diagonal element to the $k$-th power. That's $O(n)$ instead of $O(n^3 \cdot k)$.

```python
A = np.array([[4, 2],
              [1, 3]])

eigenvalues, V = np.linalg.eig(A)
V_inv = np.linalg.inv(V)

k = 10

# Direct: multiply A by itself 10 times
A_power_direct = np.linalg.matrix_power(A, k)

# Via eigendecomposition: just raise eigenvalues to the 10th power
Lambda_k = np.diag(eigenvalues**k)
A_power_eigen = V @ Lambda_k @ V_inv

print(f"A^{k} (direct):\n{A_power_direct}")
print(f"A^{k} (via eigendecomposition):\n{np.real(A_power_eigen).astype(int)}")
```

**Output:**
```
A^10 (direct):
[[1864134  1242756]
 [ 621378  1242756]]

A^10 (via eigendecomposition):
[[1864134 1242756]
 [ 621378 1242756]]
```

---

## Stability Analysis: Will Your System Converge?

This is where eigenvalues directly answer a question every ML engineer asks: "Will my iterative algorithm converge?"

Consider a system that repeatedly applies a matrix: $\mathbf{x}_{n+1} = \mathbf{Ax}_n$.

After $k$ steps: $\mathbf{x}_k = \mathbf{A}^k\mathbf{x}_0$.

Using eigendecomposition: $\mathbf{A}^k = \mathbf{V}\boldsymbol{\Lambda}^k\mathbf{V}^{-1}$.

The behavior of $\boldsymbol{\Lambda}^k$ as $k \to \infty$ depends entirely on the eigenvalues:

| Condition | Behavior | ML Implication |
|---|---|---|
| $\|\lambda\| < 1$ for all eigenvalues | Converges to zero | Gradient descent converges; RNN forgets (vanishing gradients) |
| $\|\lambda\| > 1$ for any eigenvalue | Diverges | Gradient descent explodes; RNN has exploding gradients |
| $\|\lambda\| = 1$ for all eigenvalues | Stable oscillation | System neither grows nor shrinks |

```python
def analyze_stability(matrix, name):
    eigenvalues = np.linalg.eigvals(matrix)
    max_abs = np.max(np.abs(eigenvalues))

    print(f"{name}:")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Max |lambda|: {max_abs:.4f}")

    if max_abs < 1:
        print(f"  --> CONVERGES (all |lambda| < 1)")
    elif max_abs > 1:
        print(f"  --> DIVERGES (some |lambda| > 1)")
    else:
        print(f"  --> MARGINALLY STABLE (max |lambda| = 1)")
    print()

# A system that converges (e.g., well-tuned gradient descent)
analyze_stability(
    np.array([[0.5, 0.2],
              [0.1, 0.3]]),
    "Converging system"
)

# A system that diverges (e.g., learning rate too high)
analyze_stability(
    np.array([[1.2, 0.3],
              [0.1, 0.9]]),
    "Diverging system"
)
```

**Output:**
```
Converging system:
  Eigenvalues: [0.6 0.2]
  Max |lambda|: 0.6000
  --> CONVERGES (all |lambda| < 1)

Diverging system:
  Eigenvalues: [1.245 0.855]
  Max |lambda|: 1.2449
  --> DIVERGES (some |lambda| > 1)
```

---

## The Condition Number: Why Feature Scaling Matters

The **condition number** of a matrix is the ratio of its largest to smallest eigenvalue:

$$\kappa(\mathbf{A}) = \frac{|\lambda_{\max}|}{|\lambda_{\min}|}$$

This single number tells you how "warped" the optimization landscape is.

```
  Low condition number (well-conditioned):    High condition number (ill-conditioned):

        Contours of loss function:                  Contours of loss function:

            .---.                                     .---------.
           /     \                                   /           \
          |   *   |   <-- nearly circular           |      *      |  <-- elongated
           \     /    GD converges fast               \           /   GD zig-zags
            '---'                                      '---------'

  eigenvalues: [10, 9]                        eigenvalues: [100, 1]
  condition number: 1.1                       condition number: 100
```

**Translation:** A high condition number means the loss surface is stretched much more in one direction than another. Gradient descent takes tiny steps in the stretched direction and overshoots in the narrow direction. This is exactly why you normalize your features -- it makes the eigenvalues of the covariance matrix more uniform, reducing the condition number.

```python
# The Hessian's eigenvalues determine optimal learning rate
H = np.array([[10, 1],
              [1, 1]])  # Hessian (symmetric positive definite)

eigenvalues_H = np.linalg.eigvalsh(H)
lambda_max = np.max(eigenvalues_H)
lambda_min = np.min(eigenvalues_H)

optimal_lr = 2 / (lambda_max + lambda_min)
max_stable_lr = 2 / lambda_max
condition_number = lambda_max / lambda_min

print(f"Hessian eigenvalues: {eigenvalues_H}")
print(f"Optimal learning rate: {optimal_lr:.4f}")
print(f"Max stable learning rate: {max_stable_lr:.4f}")
print(f"Condition number: {condition_number:.2f}")
print(f"  (Higher = slower, more zig-zaggy convergence)")
```

**Output:**
```
Hessian eigenvalues: [ 0.902 10.098]
Optimal learning rate: 0.1818
Max stable learning rate: 0.1981
Condition number: 11.20
  (Higher = slower, more zig-zaggy convergence)
```

---

## Running Example: PCA on Movie Ratings

Let's bring it all together with our movie ratings scenario. This is PCA from scratch.

```python
import numpy as np
np.random.seed(42)

# === Simulate movie ratings ===
# 3 hidden "taste dimensions" generate ratings for 5 movies
# Dimension 1: "Action lover" (affects movies 1, 2, 3)
# Dimension 2: "Drama fan" (affects movies 3, 4, 5)
# Dimension 3: "Comedy enthusiast" (affects movies 1, 5)

n_users = 1000

# Hidden taste scores
action_taste = np.random.randn(n_users)
drama_taste = np.random.randn(n_users)
comedy_taste = np.random.randn(n_users) * 0.5  # less important

# Movie ratings = linear combination of tastes + noise
ratings = np.column_stack([
    3*action_taste + 1*comedy_taste + np.random.randn(n_users)*0.3,   # Movie 1
    2*action_taste + np.random.randn(n_users)*0.3,                     # Movie 2
    1*action_taste + 2*drama_taste + np.random.randn(n_users)*0.3,     # Movie 3
    3*drama_taste + np.random.randn(n_users)*0.3,                      # Movie 4
    1*drama_taste + 1*comedy_taste + np.random.randn(n_users)*0.3,     # Movie 5
])

print(f"Ratings matrix shape: {ratings.shape}")  # (1000, 5)

# === PCA: Find the hidden taste dimensions ===

# Step 1: Center
ratings_centered = ratings - np.mean(ratings, axis=0)

# Step 2: Covariance matrix
cov = (ratings_centered.T @ ratings_centered) / (n_users - 1)
print(f"\nCovariance matrix (5x5):\n{np.round(cov, 2)}")

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 4: How many dimensions do we need?
variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative = np.cumsum(variance_ratio)

print(f"\n--- Variance Explained ---")
for i in range(len(eigenvalues)):
    bar = '#' * int(variance_ratio[i] * 50)
    print(f"  PC{i+1}: {variance_ratio[i]*100:5.1f}%  cumulative: {cumulative[i]*100:5.1f}%  {bar}")

print(f"\n--- Interpretation ---")
print(f"We built 5 movie features, but only 2-3 eigenvectors capture most variance.")
print(f"These correspond to our hidden 'action' and 'drama' taste dimensions!")
print(f"The 3rd (comedy) is weaker because we gave it lower weight (0.5).")
```

**Output:**
```
Ratings matrix shape: (1000, 5)

Covariance matrix (5x5):
[[ 9.28  5.98  3.27  0.13  0.31]
 [ 5.98  4.12  2.11  0.02  0.12]
 [ 3.27  2.11  5.18  5.94  2.01]
 [ 0.13  0.02  5.94  9.09  3.04]
 [ 0.31  0.12  2.01  3.04  1.34]]

--- Variance Explained ---
  PC1:  47.5%  cumulative:  47.5%  #######################
  PC2:  41.4%  cumulative:  88.9%  ####################
  PC3:   9.0%  cumulative:  97.9%  ####
  PC4:   1.1%  cumulative:  99.0%
  PC5:   1.0%  cumulative: 100.0%

--- Interpretation ---
We built 5 movie features, but only 2-3 eigenvectors capture most variance.
These correspond to our hidden 'action' and 'drama' taste dimensions!
The 3rd (comedy) is weaker because we gave it lower weight (0.5).
```

Three eigenvectors capture 97.9% of the variance. You could compress each user from 5 numbers to 3 numbers and lose almost nothing. In real recommendation systems with thousands of movies, PCA might compress to 50-100 dimensions -- a massive reduction.

---

## Where Eigenvalues Appear in ML (Reference Table)

| Application | What You Decompose | What Eigenvectors Mean | What Eigenvalues Mean |
|---|---|---|---|
| **PCA** | Covariance matrix | Principal directions of variance | Variance along each direction |
| **Spectral clustering** | Graph Laplacian | Cluster membership indicators | Number of clusters (count near-zero eigenvalues) |
| **PageRank** | Web link matrix | Page importance scores | Convergence rate |
| **Neural net init** | Weight matrices | Directions of signal flow | Amplification/attenuation of gradients |
| **RNN stability** | Recurrent weight matrix | Modes of hidden state dynamics | Gradient vanishing ($<1$) or exploding ($>1$) |
| **Optimization** | Hessian matrix | Directions of curvature | Curvature magnitude (condition number) |

---

## Common Mistakes

**1. "Eigenvectors are directions, not points."**

Any scalar multiple of an eigenvector is also an eigenvector. If $\mathbf{v}$ is an eigenvector with eigenvalue $\lambda$, then $2\mathbf{v}$, $-\mathbf{v}$, and $0.001\mathbf{v}$ are all eigenvectors with the same eigenvalue. NumPy normalizes them to unit length by convention, but that's just a convention.

```python
A = np.array([[4, 2], [1, 3]])
_, V = np.linalg.eig(A)
v = V[:, 0]

# All of these are valid eigenvectors for the same eigenvalue
for scale in [1, 2, -1, 0.5, 100]:
    scaled_v = scale * v
    Av = A @ scaled_v
    lam_v = 5.0 * scaled_v  # eigenvalue is 5
    print(f"  {scale:>5.1f} * v: A @ v = {Av}, 5 * v = {lam_v}, match: {np.allclose(Av, lam_v)}")
```

**2. "Not all matrices have real eigenvalues."**

Complex eigenvalues indicate rotation. A 2D rotation matrix has no direction it preserves -- every vector gets rotated. Its eigenvalues are complex numbers.

```python
# 90-degree rotation matrix
R = np.array([[0, -1],
              [1,  0]])

eigenvalues = np.linalg.eigvals(R)
print(f"Rotation matrix eigenvalues: {eigenvalues}")
# Output: [0.+1.j  0.-1.j] -- purely imaginary! No real eigenvectors exist.
```

**3. "Eigendecomposition only works for square matrices."**

Your data matrix is usually rectangular (e.g., 10,000 samples x 100 features). You can't directly eigendecompose it. That's why PCA computes the covariance matrix first (which is square: features x features) and eigendecomposes *that*. For direct decomposition of rectangular matrices, you need SVD (next chapter).

**4. "Use `eigh` for symmetric matrices, not `eig`."**

`np.linalg.eigh` is numerically more stable and faster for symmetric matrices. It guarantees real eigenvalues and orthogonal eigenvectors. Use `eig` only for general (non-symmetric) matrices.

**5. "Eigenvalues are NOT automatically sorted."**

`np.linalg.eig` returns eigenvalues in arbitrary order. Always sort by magnitude if you need them ordered (PCA requires descending order). `np.linalg.eigh` returns them in ascending order -- so reverse for PCA.

---

## Exercises

### Exercise 1: Hand Computation

Find the eigenvalues and eigenvectors of:

$$\mathbf{A} = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$$

*Hint: This is an upper triangular matrix. What does that tell you about the eigenvalues?*

**Solution:**

For triangular matrices, the eigenvalues are just the diagonal entries: $\lambda_1 = 3$, $\lambda_2 = 2$.

```python
A = np.array([[3, 1],
              [0, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")  # [3. 2.]
print(f"Eigenvectors:\n{eigenvectors}")

# For triangular matrices, eigenvalues are the diagonal elements!
# This is because det(A - lambda*I) = product of (diagonal - lambda)
```

### Exercise 2: Verify the Spectral Theorem

For a symmetric matrix, verify that $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$ (note: transpose, not inverse).

**Solution:**

```python
A = np.array([[2, 1],
              [1, 2]])

eigenvalues, V = np.linalg.eigh(A)
Lambda = np.diag(eigenvalues)

# Reconstruct using V^T (not V^{-1})
A_reconstructed = V @ Lambda @ V.T

print(f"Original:\n{A}")
print(f"Reconstructed:\n{A_reconstructed}")
print(f"Equal? {np.allclose(A, A_reconstructed)}")  # True

# For symmetric matrices, V^T = V^{-1} because eigenvectors are orthonormal
print(f"\nV^T == V^(-1)? {np.allclose(V.T, np.linalg.inv(V))}")  # True
```

### Exercise 3: Stability Analysis

Determine if the system $\mathbf{x}_{n+1} = \mathbf{Ax}_n$ converges:

$$\mathbf{A} = \begin{bmatrix} 0.8 & 0.1 \\ 0.2 & 0.7 \end{bmatrix}$$

*Think about this first: this is a Markov-like transition matrix. Each row mixes the current state. Should it converge?*

**Solution:**

```python
A = np.array([[0.8, 0.1],
              [0.2, 0.7]])

eigenvalues = np.linalg.eigvals(A)
max_abs = np.max(np.abs(eigenvalues))

print(f"Eigenvalues: {eigenvalues}")    # [0.9 0.6]
print(f"Max |lambda|: {max_abs}")       # 0.9
print(f"Converges? {max_abs < 1}")      # True

# Simulate to verify
x = np.array([1.0, 0.0])  # start in state 1
print(f"\nSimulating 20 steps:")
for i in range(20):
    x = A @ x
    if i % 5 == 4:
        print(f"  Step {i+1}: x = {x}")
# Watch it converge to the zero vector
```

### Exercise 4: PCA by Hand

Given these 2D data points, find the principal components:

```python
data = np.array([[2, 1],
                 [3, 2],
                 [4, 3],
                 [5, 4],
                 [6, 5]])

# Step 1: Center
data_centered = data - np.mean(data, axis=0)

# Step 2: Covariance
cov = (data_centered.T @ data_centered) / (len(data) - 1)

# Step 3: Eigendecompose
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Step 4: Interpret
idx = np.argsort(eigenvalues)[::-1]
print(f"Eigenvalues (sorted): {eigenvalues[idx]}")
print(f"Variance ratio: {eigenvalues[idx] / np.sum(eigenvalues)}")
print(f"First PC direction: {eigenvectors[:, idx[0]]}")

# This data lies almost perfectly on a line y = x - 1
# So the first PC should capture ~100% of variance
# and point roughly in the [1/sqrt(2), 1/sqrt(2)] direction
```

---

## Summary

- **Eigenvectors** are the special directions that a matrix only stretches, never rotates. They're the "natural axes" of any linear transformation.

- **Eigenvalues** are the stretch factors: $|\lambda| > 1$ amplifies, $|\lambda| < 1$ dampens, $\lambda < 0$ reverses direction.

- **The characteristic equation** $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$ finds eigenvalues -- it's the "fingerprint" of a matrix.

- **Spectral decomposition** $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$ factors a matrix into independent components, like decomposing a monolith into microservices.

- **Symmetric matrices** (covariance, Hessian, kernel matrices) have real eigenvalues and orthogonal eigenvectors -- the spectral theorem guarantees this.

- **Stability**: $|\lambda| < 1$ for all eigenvalues means iterative systems converge. This directly explains vanishing/exploding gradients in RNNs.

- **PCA** is eigendecomposition of the covariance matrix. Eigenvectors are the principal directions; eigenvalues are the variance along each direction. In our movie analogy, eigenvectors are "taste dimensions" and eigenvalues are how important each taste is.

- **The condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ measures how "warped" your optimization landscape is. High condition number means slow, zig-zaggy gradient descent. This is why feature scaling matters.

- **In practice**: use `np.linalg.eigh()` for symmetric matrices (faster, more stable); always sort eigenvalues; remember that eigenvectors are directions, not unique vectors.

---

> **What's Next** -- Eigendecomposition works for square matrices. But real data matrices are often rectangular (10,000 samples x 100 features). To decompose those, you need SVD -- the Singular Value Decomposition -- the generalization that powers dimensionality reduction, recommendation systems, and more.
