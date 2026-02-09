# Chapter 3: Matrices

> **Building On** — Vectors let you represent single data points. But ML operates on entire datasets at once — thousands of data points, each with hundreds of features. You need a way to organize and transform all that data simultaneously. Enter the matrix.

---

## The Problem That Creates the Need

A neural network layer takes 784 inputs (pixels from a 28x28 grayscale image) and produces 256 outputs (learned features). That is 784 x 256 = 200,704 connections. Each connection has a weight. How do you represent all those connections at once?

You could use 200,704 separate variables. You could use a flat list with some indexing scheme. But you have been down that road before — it is the "store everything in a flat file and parse by position" approach. Fragile, unreadable, impossible to reason about.

What you actually need is a 2D structure where rows correspond to outputs and columns correspond to inputs (or vice versa). You need a **matrix**.

And here is the punchline: a single neural network layer — the fundamental building block of deep learning — is literally this:

```
output = W @ input + bias
```

That `@` is matrix multiplication. That is it. Every forward pass through every layer of every neural network is a matrix multiply. Matrix multiplication: the operation that funds NVIDIA.

---

## Your Movie Recommendation System — A Running Example

Imagine you are building a recommendation engine. You have:

- **10,000 users**
- **1,000 movies**
- Each user has rated some movies on a 1-5 scale (0 means "not rated")

Your entire dataset is a single matrix — 10,000 rows, 1,000 columns, 10 million entries:

```
                 Movie_1  Movie_2  Movie_3  ...  Movie_1000
    User_1    [   5        0        3       ...     0      ]
    User_2    [   0        4        0       ...     2      ]
    User_3    [   3        5        4       ...     0      ]
      ...           ...      ...      ...   ...    ...
    User_10000[   0        0        2       ...     5      ]
```

Every operation in this chapter — addition, multiplication, transpose, inverse — has a direct meaning in this recommendation system. We will keep coming back to it.

---

## What Is a Matrix? (Code-First)

> **You Already Know This**
>
> A matrix is a 2D array — but with algebraic rules attached. Think of it like a class with operator overloading: the underlying data is just a grid of numbers, but `+`, `@`, and `.T` have specific mathematical meanings that make the whole thing more powerful than a bare `list[list[float]]`.

Let's start in NumPy and discover the structure:

```python
import numpy as np

# Your ratings data — 4 users, 3 movies (a tiny slice of the full system)
ratings = np.array([
    [5, 0, 3],
    [0, 4, 0],
    [3, 5, 4],
    [0, 0, 2]
])

print(f"Shape: {ratings.shape}")      # (4, 3) — 4 users, 3 movies
print(f"Rows (users): {ratings.shape[0]}")
print(f"Cols (movies): {ratings.shape[1]}")
print(f"Total entries: {ratings.size}")  # 12
print(f"User 2's ratings: {ratings[1]}")     # [0, 4, 0]
print(f"Movie 3's ratings: {ratings[:, 2]}") # [3, 0, 4, 2]
```

```
Shape: (4, 3)
Rows (users): 4
Cols (movies): 3
Total entries: 12
User 2's ratings: [0 4 0]
Movie 3's ratings: [3 0 4 2]
```

Notice the indexing: `ratings[i, j]` gives you user `i`'s rating for movie `j`. Rows and columns have meaning.

### The Formal Notation

Now that you see the structure in code, here is how mathematicians write it. A matrix $\mathbf{A}$ with $m$ rows and $n$ columns (an $m \times n$ matrix):

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

**Translation:** $a_{ij}$ is the entry in row $i$, column $j$. That is exactly `A[i-1, j-1]` in NumPy (adjusting for zero-based indexing). The notation $\mathbf{A} \in \mathbb{R}^{m \times n}$ means "A is a matrix of real numbers with $m$ rows and $n$ columns." Think of it as the type signature: `A: Matrix[float, m, n]`.

Here is the ASCII structure:

```
           n columns
          |<------->|
          c1  c2  c3
     r1 [ 5   0   3 ]  ---+
 m   r2 [ 0   4   0 ]     |  m rows
rows r3 [ 3   5   4 ]     |
     r4 [ 0   0   2 ]  ---+

     This is a 4x3 matrix (4 rows, 3 columns)

     Access pattern:  A[row, col]  =  A[i, j]
```

**Special cases you will encounter constantly:**
- **Row vector**: a $1 \times n$ matrix — one user's ratings across all movies
- **Column vector**: an $m \times 1$ matrix — all users' rating for one movie
- **Square matrix**: $n \times n$ — shows up in covariance, adjacency, and attention matrices

---

## Matrix Addition — Combining Information

Suppose two critics provide separate ratings for the same movies. You want the combined score. That is element-wise addition:

```python
import numpy as np

critic_A = np.array([[5, 3, 4],
                     [2, 1, 5]])

critic_B = np.array([[3, 4, 2],
                     [5, 5, 1]])

combined = critic_A + critic_B
print(f"Combined ratings:\n{combined}")
```

```
Combined ratings:
[[ 8  7  6]
 [ 7  6  6]]
```

Every entry gets added to its corresponding entry. Simple.

### The Math

$$(\mathbf{A} + \mathbf{B})_{ij} = a_{ij} + b_{ij}$$

**Translation:** "Add each entry to the entry in the same position." Both matrices must have the exact same dimensions — you cannot add a 4x3 to a 3x4. That is a shape mismatch, and NumPy will raise a `ValueError`.

**Properties** (these should feel obvious if you think of element-wise operations):
- Commutative: $\mathbf{A} + \mathbf{B} = \mathbf{B} + \mathbf{A}$ — order does not matter
- Associative: $(\mathbf{A} + \mathbf{B}) + \mathbf{C} = \mathbf{A} + (\mathbf{B} + \mathbf{C})$

No surprises here. Addition is the easy part. Multiplication is where it gets interesting.

---

## Matrix Multiplication — The Core Operation of ML

This is the big one. If you understand one operation in this chapter, make it this one.

> **You Already Know This**
>
> Matrix multiplication is batch processing. When you multiply a matrix by a vector, you are applying the same linear transformation to one data point. When you multiply a matrix by another matrix, you are applying that transformation to many data points at once — like a `map()` over your dataset, except it runs on a GPU at terrifying speed.

### Code First — Watch What Happens

```python
import numpy as np

# A neural network layer: 4 inputs -> 2 outputs
W = np.array([[0.1, 0.2, 0.3, 0.4],   # weights for output neuron 1
              [0.5, 0.6, 0.7, 0.8]])   # weights for output neuron 2

# A single input vector (1 data point, 4 features)
x = np.array([[1],
              [2],
              [3],
              [4]])

# Matrix-vector multiplication: apply the layer to one input
output = W @ x
print(f"Single input result:\n{output}")

# Now a BATCH of 3 inputs (each is a column)
X_batch = np.array([[1, 0, 2],
                    [2, 1, 0],
                    [3, 1, 1],
                    [4, 0, 3]])

# Matrix-matrix multiplication: apply the layer to ALL inputs at once
outputs = W @ X_batch
print(f"\nBatch result (2 outputs x 3 samples):\n{outputs}")
```

```
Single input result:
[[3. ]
 [7. ]]

Batch result (2 outputs x 3 samples):
[[2.8 0.5 1.5]
 [6.4 1.3 3.9]]
```

One `@` operator. Three data points processed simultaneously. Scale that up to millions of data points and hundreds of layers, and you have modern deep learning.

### How It Actually Works — The Visual

Matrix multiplication computes each output entry as a **dot product** between a row from the left matrix and a column from the right matrix:

```
    Matrix A          Matrix B              Result C
    (2 x 3)           (3 x 2)              (2 x 2)

    [a  b  c]       [x  u]
    [d  e  f]   @   [y  v]    =    [c00  c01]
                    [z  w]         [c10  c11]

    How c00 is computed (row 0 of A . col 0 of B):

    [a  b  c] . [x]  =  a*x + b*y + c*z
                [y]
                [z]

    How c01 is computed (row 0 of A . col 1 of B):

    [a  b  c] . [u]  =  a*u + b*v + c*w
                [v]
                [w]

    Every entry c[i,j] = dot(row_i(A), col_j(B))
```

Let's trace through a concrete example:

```
    A              B             C = A @ B
    [1  2  3]    [ 7   8]      [ 58   64]
    [4  5  6]  @ [ 9  10]  =   [139  154]
                 [11  12]

    c[0,0] = 1*7  + 2*9  + 3*11 = 7  + 18 + 33 = 58
    c[0,1] = 1*8  + 2*10 + 3*12 = 8  + 20 + 36 = 64
    c[1,0] = 4*7  + 5*9  + 6*11 = 28 + 45 + 66 = 139
    c[1,1] = 4*8  + 5*10 + 6*12 = 32 + 50 + 72 = 154
```

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

C = A @ B
print(f"A @ B =\n{C}")

# Manual verification of one element
print(f"\nVerification c[0,0] = 1*7 + 2*9 + 3*11 = {1*7 + 2*9 + 3*11}")
```

```
A @ B =
[[ 58  64]
 [139 154]]

Verification c[0,0] = 1*7 + 2*9 + 3*11 = 58
```

### The Dimension Rule

This is the rule you will use a hundred times a day when debugging shape errors:

```
    (m x n) @ (n x p)  =  (m x p)
      |   |    |   |        |   |
      |   +----+   |        |   |
      |  must match |        |   |
      +-------------+--------+   |
      outer dim               outer dim
```

$$\text{For } \mathbf{A} \in \mathbb{R}^{m \times n} \text{ and } \mathbf{B} \in \mathbb{R}^{n \times p}: \quad (\mathbf{AB})_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

**Translation:** To multiply A (m x n) by B (n x p), the inner dimensions must match (both n). The result is m x p. If the inner dimensions do not match, the operation is undefined — you will get a `ValueError` in NumPy.

### The Properties That Trip People Up

> **Common Mistake: Matrix multiplication is NOT commutative**
>
> $\mathbf{AB} \neq \mathbf{BA}$ in general. In fact, if A is (2x3) and B is (3x2), then AB is (2x2) but BA is (3x3) — they are not even the same shape! Even when both are square and both products exist, the values will differ. Never reorder matrix multiplications assuming you can.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"AB:\n{A @ B}")
print(f"\nBA:\n{B @ A}")
print(f"\nAB == BA? {np.array_equal(A @ B, B @ A)}")  # False!
```

```
AB:
[[19 22]
 [43 50]]

BA:
[[23 34]
 [31 46]]

AB == BA? False
```

**Other properties** (these DO hold):
- Associative: $(\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})$ — you can group however you like, which matters for computational efficiency
- Distributive: $\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{AB} + \mathbf{AC}$

### Element-wise vs. Matrix Multiplication — A Critical Distinction

This is one of the most common bugs in ML code:

```python
import numpy as np

E = np.array([[1, 2], [3, 4]])
F = np.array([[5, 6], [7, 8]])

print(f"Element-wise (E * F) — Hadamard product:\n{E * F}")
print(f"\nMatrix multiply (E @ F):\n{E @ F}")
```

```
Element-wise (E * F) — Hadamard product:
[[ 5 12]
 [21 32]]

Matrix multiply (E @ F):
[[19 22]
 [43 50]]
```

> **Common Mistake: Confusing `*` and `@` in NumPy**
>
> `*` is element-wise multiplication (Hadamard product): each entry is multiplied by the entry in the same position. `@` is true matrix multiplication (dot products of rows and columns). Using the wrong one will not raise an error if the shapes happen to match — it will silently give you wrong results.

---

## Transpose — Flipping Rows and Columns

> **You Already Know This**
>
> Transpose is like pivoting a database table. If your matrix has users as rows and movies as columns, the transpose has movies as rows and users as columns. Same data, different orientation.

### See It First

```
    Original A (2x3)          Transpose A^T (3x2)

    [1  2  3]                 [1  4]
    [4  5  6]                 [2  5]
                              [3  6]

    Row 0 of A  --> Column 0 of A^T
    Row 1 of A  --> Column 1 of A^T

    Or equivalently:
    A[i, j]  =  A^T[j, i]
```

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"A (2x3):\n{A}")
print(f"\nA.T (3x2):\n{A.T}")
```

```
A (2x3):
[[1 2 3]
 [4 5 6]]

A.T (3x2):
[[1 4]
 [2 5]
 [3 6]]
```

### Why Transpose Matters in ML

In your recommendation system, the ratings matrix is users x movies. The transpose is movies x users. Why would you want that?

- **$\mathbf{X}^T \mathbf{X}$** gives you the **feature covariance** matrix — how correlated are the movies' ratings? This is the foundation of PCA (dimensionality reduction).
- **$\mathbf{X} \mathbf{X}^T$** gives you the **sample similarity** matrix — how similar are users to each other? This is the basis of collaborative filtering.
- **Gradient computation** in backpropagation: $\nabla_{\mathbf{W}} L$ almost always involves a transpose of the input matrix.

### The Math

$$(\mathbf{A}^T)_{ij} = a_{ji}$$

If $\mathbf{A}$ is $m \times n$, then $\mathbf{A}^T$ is $n \times m$.

**Properties:**
- $(\mathbf{A}^T)^T = \mathbf{A}$ — double transpose gets you back where you started
- $(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$ — transpose distributes over addition
- $(\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T$ — **order reverses!** (This one bites people. Think of it like reversing the order of function composition.)
- $(c\mathbf{A})^T = c\mathbf{A}^T$ — scalars pass through

**Symmetric matrix:** When $\mathbf{A} = \mathbf{A}^T$, the matrix is symmetric. Covariance matrices are always symmetric — the correlation between feature $i$ and feature $j$ is the same as between $j$ and $i$.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Verify (AB)^T = B^T @ A^T
AB_T = (A @ B).T
BT_AT = B.T @ A.T

print(f"(AB)^T:\n{AB_T}")
print(f"\nB^T @ A^T:\n{BT_AT}")
print(f"\nEqual? {np.array_equal(AB_T, BT_AT)}")
```

```
(AB)^T:
[[19 43]
 [22 50]]

B^T @ A^T:
[[19 43]
 [22 50]]

Equal? True
```

---

## Identity Matrix — The No-Op

> **You Already Know This**
>
> The identity matrix is the `noop` or pass-through middleware of linear algebra. Multiply any matrix by the identity and you get the same matrix back. It is the `return x` of matrix functions, the `/dev/null` of transformations (except it keeps everything). It is `x * 1` but for matrices.

```
    I_3 (3x3 identity):

    [1  0  0]
    [0  1  0]
    [0  0  1]

    Ones on the diagonal, zeros everywhere else.
```

$$\mathbf{I}_n = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

**The defining property:** For any compatible matrix $\mathbf{A}$:

$$\mathbf{AI} = \mathbf{IA} = \mathbf{A}$$

**Translation:** Multiplying by the identity matrix does nothing. It is the multiplicative identity — what 1 is to regular multiplication, $\mathbf{I}$ is to matrix multiplication.

```python
import numpy as np

I3 = np.eye(3)
G = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"I_3:\n{I3}")
print(f"\nG @ I_3 (unchanged):\n{G @ I3}")
print(f"\nSame as G? {np.array_equal(G, G @ I3)}")
```

```
I_3:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]

G @ I_3 (unchanged):
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]

Same as G? True
```

Why does this matter? Because the identity matrix is what the inverse is defined in terms of. It is the target: "find me the matrix that, when multiplied by A, gives back the identity."

---

## Inverse Matrix — The Undo Button

> **You Already Know This**
>
> The inverse matrix is the "undo" operation — `Ctrl+Z` for linear transformations. If matrix $\mathbf{A}$ represents a transformation (rotating, scaling, shearing), then $\mathbf{A}^{-1}$ is the transformation that reverses it. Like a function and its inverse: if `f(x) = y`, then `f_inverse(y) = x`.

### The Definition

The **inverse** $\mathbf{A}^{-1}$ of a square matrix $\mathbf{A}$ satisfies:

$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

**Translation:** Apply the transformation, then apply the inverse, and you are back to where you started. Identity in, identity out.

### Code Exploration

```python
import numpy as np

H = np.array([[4, 7],
              [2, 6]])
H_inv = np.linalg.inv(H)

print(f"H:\n{H}")
print(f"\nH inverse:\n{H_inv}")
print(f"\nH @ H_inv (should be identity):\n{np.round(H @ H_inv, 10)}")
print(f"\nIs H @ H_inv ~= I? {np.allclose(H @ H_inv, np.eye(2))}")
```

```
H:
[[4 7]
 [2 6]]

H inverse:
[[ 0.6 -0.7]
 [-0.2  0.4]]

H @ H_inv (should be identity):
[[1. 0.]
 [0. 1.]]

Is H @ H_inv ~= I? True
```

### When Does an Inverse Exist?

> **Common Mistake: Not all matrices have inverses**
>
> Singular matrices are like functions with no inverse — `f(x) = 0` maps everything to zero, and you cannot undo that because you have lost information. A matrix that squashes a 2D plane into a line has destroyed a dimension of information. No inverse can recover it.

A matrix $\mathbf{A}$ has an inverse if and only if:
- Its **determinant** is non-zero: $\det(\mathbf{A}) \neq 0$
- Its rows (and columns) are **linearly independent** — no row is a scaled copy or combination of other rows
- It has **full rank**: $\text{rank}(\mathbf{A}) = n$ for an $n \times n$ matrix

These are all saying the same thing in different ways: the matrix does not collapse any dimension.

```python
import numpy as np

# This matrix IS invertible
good = np.array([[4, 7],
                 [2, 6]])
print(f"det(good) = {np.linalg.det(good):.1f}")   # 10.0 — nonzero, good

# This matrix is NOT invertible (row 2 = 2 * row 1)
singular = np.array([[1, 2],
                     [2, 4]])
print(f"det(singular) = {np.linalg.det(singular):.1f}")  # 0.0 — singular!

# Trying to invert it will give numerical garbage or an error
# np.linalg.inv(singular)  # Raises LinAlgError: Singular matrix
```

```
det(good) = 10.0
det(singular) = 0.0
```

### The 2x2 Inverse Formula

For a $2 \times 2$ matrix, the inverse has a clean closed form:

$$\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \quad \mathbf{A}^{-1} = \frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

The denominator $ad - bc$ is the **determinant**. When it is zero, you are dividing by zero — no inverse exists. For anything larger than 2x2, let NumPy handle it.

### Inverse Properties

- Only **square** matrices can have inverses
- $(\mathbf{A}^{-1})^{-1} = \mathbf{A}$ — undo the undo and you are back to the original
- $(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$ — **order reverses!** (Same as transpose — think of removing layers in reverse order, like a stack)
- $(\mathbf{A}^T)^{-1} = (\mathbf{A}^{-1})^T$ — transpose and inverse commute

### A Practical Warning

> **Common Mistake: Do not compute matrix inverses explicitly in production code**
>
> If you need to solve $\mathbf{Ax} = \mathbf{b}$, do NOT compute $\mathbf{A}^{-1}$ and then multiply. Use `np.linalg.solve(A, b)` instead. It is faster, more numerically stable, and uses less memory. Computing the inverse is like computing `1/x * y` when you could just compute `y/x` — it works, but it is wasteful and introduces floating point error.

---

## Putting It All Together — ML Applications

### A Neural Network Layer Is a Matrix Multiplication

This is not an analogy. A single neural network layer literally performs:

$$\mathbf{y} = \sigma(\mathbf{Wx} + \mathbf{b})$$

where:
- $\mathbf{x}$: input vector (e.g., 784 pixel values)
- $\mathbf{W}$: weight matrix (e.g., 256 x 784 — the 200,704 connections from the opening)
- $\mathbf{b}$: bias vector (e.g., 256 values)
- $\sigma$: activation function (applied element-wise)
- $\mathbf{y}$: output vector (e.g., 256 feature values)

When you process a **batch** of inputs (which you always do in practice for GPU efficiency):

$$\mathbf{Y} = \sigma(\mathbf{XW}^T + \mathbf{b})$$

where $\mathbf{X}$ has shape (batch_size x input_features) and $\mathbf{Y}$ has shape (batch_size x output_features).

```python
import numpy as np

# Simulating a neural network layer
# Input: batch of 3 samples, 4 features each
batch_input = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2]
])

# Weight matrix: 4 input features -> 2 output features
weights = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8]
])

# Bias vector
bias = np.array([0.1, 0.2])

# Forward pass: output = input @ weights + bias
output = batch_input @ weights + bias

print(f"Input shape:   {batch_input.shape}")   # (3, 4) — 3 samples, 4 features
print(f"Weights shape: {weights.shape}")       # (4, 2) — 4 inputs, 2 outputs
print(f"Output shape:  {output.shape}")        # (3, 2) — 3 samples, 2 outputs
print(f"\nOutput:\n{output}")
```

```
Input shape:   (3, 4)
Weights shape: (4, 2)
Output shape:  (3, 2)

Output:
[[0.6 0.8]
 [1.4 1.8]
 [2.2 2.8]]
```

> **Common Mistake: Matrix dimensions must match for multiplication: (m x n)(n x p) = (m x p)**
>
> The single most common error in ML code is shape mismatches. When you see `ValueError: matmul: Input operand 1 has a mismatch in its core dimension`, it means the inner dimensions do not agree. Develop the habit of annotating shapes in comments:
> ```python
> # X: (batch, features) @ W: (features, outputs) -> (batch, outputs)
> ```

### Where Matrices Appear in ML

| Matrix | Shape | What It Represents |
|--------|-------|--------------------|
| Data matrix $\mathbf{X}$ | $n \times d$ | $n$ samples, $d$ features |
| Weight matrix $\mathbf{W}$ | $d_{\text{out}} \times d_{\text{in}}$ | Neural network layer connections |
| Covariance matrix $\mathbf{X}^T\mathbf{X}$ | $d \times d$ | Feature correlations (used in PCA) |
| Confusion matrix | $k \times k$ | Classification accuracy per class |
| Adjacency matrix | $n \times n$ | Graph structure (graph neural networks) |
| Attention matrix | $\text{seq} \times \text{seq}$ | Token relationships (Transformers) |

Back to our movie recommendation system: the ratings matrix is $\mathbf{X} \in \mathbb{R}^{10000 \times 1000}$. Matrix factorization decomposes it as $\mathbf{X} \approx \mathbf{U}\mathbf{V}^T$ where $\mathbf{U}$ is (10000 x 50) and $\mathbf{V}$ is (1000 x 50). Those 50 columns are "latent features" — maybe genre preferences — and the whole recommendation engine is just matrix multiplication.

### Dataset as a Matrix — A Prediction Example

```python
import numpy as np

# Each row is a house: [sqft, bedrooms, bathrooms]
X = np.array([
    [1500, 3, 2],
    [2000, 4, 3],
    [1200, 2, 1],
    [1800, 3, 2]
])

# Learned weights: price contribution of each feature
w = np.array([[100],      # $ per sqft
              [50000],    # $ per bedroom
              [30000]])   # $ per bathroom

# Predictions for ALL houses at once — one matrix multiply
predictions = X @ w
print(f"House features (4 houses x 3 features):\n{X}")
print(f"\nWeights: {w.flatten()}")
print(f"\nPredicted prices: {predictions.flatten()}")
# [360000, 490000, 250000, 390000]
```

```
House features (4 houses x 3 features):
[[1500    3    2]
 [2000    4    3]
 [1200    2    1]
 [1800    3    2]]

Weights: [   100  50000  30000]

Predicted prices: [360000 490000 250000 390000]
```

Four predictions, one operation. Now imagine 10 million houses. Same code, same single `@` — NumPy (and the GPU behind it) handles the parallelism.

---

## How a Matrix Transforms a Vector

To build intuition for what matrix multiplication *does geometrically*, consider a 2D transformation:

```
    Before: vector v = [1, 0]          After: A @ v

    Input space:                       Output space:
         y                                  y
         |                                  |   /
         |                                  |  / A @ [0,1] = [b,d]
         +-----> x                          | /
         |  v = [1,0]                       +-------> x
                                               A @ [1,0] = [a,c]

    Matrix A = [a  b]   transforms the basis vectors:
               [c  d]
      - [1, 0] maps to [a, c]  (first column of A)
      - [0, 1] maps to [b, d]  (second column of A)

    Every other vector is a linear combination of these.
```

```python
import numpy as np

# Rotation matrix (90 degrees counterclockwise)
theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

v = np.array([1, 0])
print(f"Original vector: {v}")
print(f"After rotation:  {np.round(R @ v, 10)}")  # [0, 1] — rotated 90 degrees
```

```
Original vector: [1 0]
After rotation:  [0. 1.]
```

This is the deeper meaning of matrices: they are *transformations*. Every matrix encodes a function from vectors to vectors. Multiplication *is* application.

---

## Exercises

### Exercise 1: Non-Commutativity of Matrix Multiplication

Given:
$$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

Compute $\mathbf{AB}$ and $\mathbf{BA}$. Are they equal?

**Solution:**
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

AB = A @ B
BA = B @ A

print(f"AB:\n{AB}")
# [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
# [[19, 22], [43, 50]]

print(f"\nBA:\n{BA}")
# [[5*1+6*3, 5*2+6*4], [7*1+8*3, 7*2+8*4]]
# [[23, 34], [31, 46]]

print(f"\nAB == BA? {np.array_equal(AB, BA)}")  # False
```

### Exercise 2: Transpose Properties

Verify that $(\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T$ for the matrices above.

**Solution:**
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

AB_T = (A @ B).T
BT_AT = B.T @ A.T

print(f"(AB)^T:\n{AB_T}")
print(f"\nB^T @ A^T:\n{BT_AT}")
print(f"\nAre they equal? {np.array_equal(AB_T, BT_AT)}")  # True
```

### Exercise 3: Inverse Verification

Find the inverse of $\mathbf{M} = \begin{bmatrix} 2 & 1 \\ 5 & 3 \end{bmatrix}$ and verify $\mathbf{MM}^{-1} = \mathbf{I}$.

**Solution:**
```python
import numpy as np

M = np.array([[2, 1], [5, 3]])
M_inv = np.linalg.inv(M)

# Analytical: det = 2*3 - 1*5 = 1
# M_inv = 1/1 * [[3, -1], [-5, 2]] = [[3, -1], [-5, 2]]
print(f"M:\n{M}")
print(f"\nM_inv:\n{M_inv}")       # [[3, -1], [-5, 2]]
print(f"\nM @ M_inv:\n{M @ M_inv}")  # Identity matrix
print(f"\nClose to identity? {np.allclose(M @ M_inv, np.eye(2))}")
```

### Exercise 4: Shape Debugging (Practical)

You have a batch of 32 images, each flattened to 784 pixels. Your first layer has 256 neurons. What shapes do the weight matrix and bias vector need? Write the forward pass.

**Solution:**
```python
import numpy as np

# Input: 32 images, each 784 pixels
X = np.random.randn(32, 784)

# Weights: must be (784, 256) so that (32, 784) @ (784, 256) = (32, 256)
W = np.random.randn(784, 256) * 0.01
b = np.zeros(256)

# Forward pass
output = X @ W + b  # (32, 256) — 32 images, each with 256 features
print(f"Input:  {X.shape}")       # (32, 784)
print(f"Weight: {W.shape}")       # (784, 256)
print(f"Bias:   {b.shape}")       # (256,)
print(f"Output: {output.shape}")  # (32, 256)
```

---

## Common Pitfalls — Quick Reference

1. **Dimension mismatch**: Always annotate shapes in comments. `(m, n) @ (n, p) -> (m, p)`. If the inner `n` values do not match, it will not work.

2. **Confusing `*` and `@`**: `*` is element-wise, `@` is matrix multiplication. They give different results even on same-shaped matrices.

3. **Assuming commutativity**: $\mathbf{AB} \neq \mathbf{BA}$ in general. Never reorder matrix products without justification.

4. **Computing inverses explicitly**: Use `np.linalg.solve(A, b)` instead of `np.linalg.inv(A) @ b`. It is faster and more numerically stable.

5. **Forgetting about broadcasting**: NumPy silently broadcasts lower-dimensional arrays. `bias` with shape `(256,)` gets added to each row of a `(32, 256)` matrix. This is usually what you want, but can hide bugs when shapes are accidentally compatible.

6. **Inverting singular matrices**: Check `np.linalg.det(A)` or use `np.linalg.cond(A)` to check condition number before inverting. Better yet, use `np.linalg.solve()`.

---

## Summary

- A **matrix** is a 2D array of numbers with $m$ rows and $n$ columns — your dataset, your weights, your transformation
- **Matrix addition** is element-wise; both matrices must have the same shape
- **Matrix multiplication** computes dot products between rows and columns; inner dimensions must match: $(m \times n)(n \times p) = (m \times p)$
- Matrix multiplication is **NOT commutative**: $\mathbf{AB} \neq \mathbf{BA}$
- The **transpose** $\mathbf{A}^T$ swaps rows and columns — crucial for covariance, gradients, and reshaping data
- The **identity matrix** $\mathbf{I}$ is the no-op: $\mathbf{AI} = \mathbf{IA} = \mathbf{A}$
- The **inverse** $\mathbf{A}^{-1}$ undoes a transformation: $\mathbf{AA}^{-1} = \mathbf{I}$ — but not all matrices have one
- In NumPy, use `@` for matrix multiplication and `*` for element-wise
- A neural network layer is literally a matrix multiplication: `output = input @ weights + bias`
- Your 10,000 x 1,000 movie ratings matrix is the foundation of your recommendation engine

---

> **What's Next** — You have learned matrix arithmetic. But what does it *mean* to multiply a matrix by a vector? It is a transformation — rotation, scaling, projection. That geometric interpretation is what makes matrices powerful for ML. In the next chapter, we explore how matrices act as functions that reshape space itself.
