# Chapter 4: Matrices as Transformations

## Building On

You've learned matrix arithmetic — how to add, multiply, and invert matrices. But what does matrix multiplication actually *do* geometrically? It transforms space itself.

When you resize an image, rotate a 3D model, or run data through a neural network layer — you're applying a matrix transformation. Every one of those operations is just multiplying by a matrix.

This chapter is where matrix algebra stops being abstract bookkeeping and starts being something you can *see*. And once you see it, you'll never look at a neural network the same way again.

---

## The Punchline Up Front

Let's start with code and work backward to the math. Here's a rotation in NumPy:

```python
import numpy as np

# Rotate a point 45 degrees counterclockwise
theta = np.pi / 4
point = np.array([1, 0])

# "Manual" rotation using trig
x_new = point[0] * np.cos(theta) - point[1] * np.sin(theta)
y_new = point[0] * np.sin(theta) + point[1] * np.cos(theta)
print(f"Manual rotation: [{x_new:.4f}, {y_new:.4f}]")
# Manual rotation: [0.7071, 0.7071]

# Now the same thing as matrix multiplication
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
print(f"Matrix rotation: {R @ point}")
# Matrix rotation: [0.70710678 0.70710678]
```

Same answer. The matrix *is* the rotation. That `R @ point` call is doing exactly the trig you'd write by hand — it's just packaged into a 2x2 grid of numbers.

And here's the thing that makes this matter for your career: **every layer in a neural network is a linear transformation followed by a nonlinearity. The matrix IS the learned transformation.** When you train a model, gradient descent is searching for the right transformation matrices. Everything in this chapter is the geometry of what those weight matrices do.

---

## What Is a Linear Transformation?

A function $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$ is a **linear transformation** if it obeys two rules:

1. **Additivity**: $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **Homogeneity (scaling)**: $T(c\mathbf{u}) = cT(\mathbf{u})$

That's it. If stretching or adding vectors before or after the transformation gives the same result, it's linear.

**Translation**: "Linear" means the transformation can't *curve* anything, and it can't *shift* the origin. Straight lines stay straight, the origin stays put, and parallel lines stay parallel. If you've ever wondered why neural networks need bias terms — it's because a pure matrix multiplication can't translate your data, only rotate/scale/project it.

> **You Already Know This: Middleware Pipelines**
>
> A linear transformation is a function with a specific shape: it takes a vector of size $n$ and returns a vector of size $m$. Sound familiar? That's a middleware function with a typed input and typed output. The "linearity" constraint is like saying your middleware can't have side effects that break composition — the same way pure functions compose cleanly, linear transformations compose cleanly (via matrix multiplication).

The deep result: **every** linear transformation can be represented as a matrix multiplication:

$$T(\mathbf{x}) = \mathbf{A}\mathbf{x}$$

No exceptions. If it's linear, there's a matrix for it. If you have the matrix, you have the transformation. They're the same thing.

---

## The Secret: Watch the Basis Vectors

Here's the single most important insight in this chapter:

**A matrix transformation is determined entirely by what it does to the basis vectors — that's why you only need to store $n^2$ numbers to define a transformation on all of $\mathbb{R}^n$.**

In 2D, the standard basis vectors are:

$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

When you multiply a matrix $\mathbf{A}$ by $\mathbf{e}_1$, you get the **first column** of $\mathbf{A}$. When you multiply by $\mathbf{e}_2$, you get the **second column**. So the columns of a matrix literally tell you where the basis vectors land after the transformation.

Let's verify:

```python
import numpy as np

# Some arbitrary 2x2 matrix
A = np.array([[2, -1],
              [1,  3]])

e1 = np.array([1, 0])
e2 = np.array([0, 1])

print(f"A @ e1 = {A @ e1}")  # [2, 1] — that's column 1 of A
print(f"A @ e2 = {A @ e2}")  # [-1, 3] — that's column 2 of A
```

**Output:**
```
A @ e1 = [2 1]
A @ e2 = [-1  3]
```

Every other vector is just a combination of $\mathbf{e}_1$ and $\mathbf{e}_2$, so linearity forces the transformation of every point to be determined by where these two arrows land. You can read a matrix like a recipe: "send the x-axis arrow here, send the y-axis arrow there."

Here's what that looks like. The unit square (formed by the basis vectors) gets reshaped:

```
  BEFORE (Identity)              AFTER (A = [[2,-1],[1,3]])

       e2                              A·e2 = (-1, 3)
        ^                                  /
        | (0,1)                           /
        |                                /
        |______> e1              _______/
       O        (1,0)          O ---------> A·e1 = (2, 1)

  Unit square:                  Parallelogram:
  (0,0)-(1,0)-(1,1)-(0,1)      (0,0)-(2,1)-(1,4)-(-1,3)
```

The unit square became a parallelogram. That's what matrices DO — they warp the grid. The determinant (which you've already learned) measures how much the area changes.

> **Common Mistake**
>
> "But I have 4 numbers in my 2x2 matrix — shouldn't that define 4 things?" It does: two numbers per basis vector, two basis vectors. The matrix is just the destination coordinates of each basis vector stacked as columns. That's the whole story.

---

## The Catalog of Standard Transformations

Let's build up the standard transformations one by one. For each, we'll see the code, the picture, and the matrix.

### Scaling

Scaling stretches or compresses along the coordinate axes. Independently.

$$\mathbf{S} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

- $s_x > 1$: stretch horizontally
- $0 < s_x < 1$: compress horizontally
- $s_x < 0$: reflect *and* scale horizontally
- Same logic for $s_y$ vertically

```python
import numpy as np

S = np.array([[2,   0],
              [0, 0.5]])

# Where do the basis vectors go?
print(f"e1 -> {S @ [1, 0]}")  # [2, 0]   — stretched 2x along x
print(f"e2 -> {S @ [0, 1]}")  # [0, 0.5] — compressed to half along y
```

**Output:**
```
e1 -> [2.  0. ]
e2 -> [0.  0.5]
```

```
  BEFORE                 AFTER  S = [[2, 0], [0, 0.5]]

    ^                       ^
    | (0,1)                 | (0, 0.5)
    |_____|                 |_____________
    O     (1,0)             O             (2, 0)

  1x1 square        -->   2 x 0.5 rectangle
```

> **You Already Know This: CSS Transforms**
>
> `transform: scale(2, 0.5)` in CSS does exactly this matrix multiplication to every pixel coordinate. The browser is doing linear algebra on your behalf, thousands of times per frame.

---

### Rotation

Rotation by angle $\theta$ counterclockwise:

$$\mathbf{R} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

Where does this formula come from? Think about where $\mathbf{e}_1 = (1, 0)$ goes when you rotate it by $\theta$: it lands at $(\cos\theta, \sin\theta)$. That's the first column. And $\mathbf{e}_2 = (0, 1)$ goes to $(-\sin\theta, \cos\theta)$. That's the second column. Done — the matrix writes itself from the geometry.

```python
import numpy as np

theta = np.pi / 4  # 45 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print(f"e1 -> {R @ [1, 0]}")  # [0.707, 0.707]
print(f"e2 -> {R @ [0, 1]}")  # [-0.707, 0.707]
print(f"(1,1) -> {R @ [1, 1]}")  # [0, 1.414]
```

**Output:**
```
e1 -> [0.70710678 0.70710678]
e2 -> [-0.70710678  0.70710678]
(1,1) -> [0.         1.41421356]
```

```
  BEFORE                 AFTER  R (45 degrees)

        e2                     R·e2 /  / R·e1
        ^                         /  /
        |                       / O
        |______> e1            /
        O
                           Both basis vectors rotated 45 degrees CCW
                           The square becomes a rotated square (same area!)
```

Notice: rotation preserves lengths and areas. The determinant of a rotation matrix is always 1. Rotations don't stretch anything — they just spin it.

---

### Reflection

Reflection across the x-axis:

$$\mathbf{F}_x = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$

```python
import numpy as np

F = np.array([[1,  0],
              [0, -1]])

print(f"e1 -> {F @ [1, 0]}")  # [1, 0]  — unchanged
print(f"e2 -> {F @ [0, 1]}")  # [0, -1] — flipped
print(f"(3, 4) -> {F @ [3, 4]}")  # [3, -4]
```

**Output:**
```
e1 -> [1 0]
e2 -> [ 0 -1]
(3, 4) -> [ 3 -4]
```

```
  BEFORE                   AFTER  F (reflect across x-axis)

        (0,1)                    O--------> (1, 0)
        ^                        |
        |                        |
        |______> (1,0)           v
        O                        (0, -1)

                           e1 stays, e2 flips. Mirror image.
```

The determinant is $-1$: reflection flips orientation (your right hand becomes a left hand). This is how you detect whether a transformation preserves orientation — check the sign of the determinant.

---

### Shear

Shearing slides things sideways. Horizontal shear with factor $k$:

$$\mathbf{H} = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$$

```python
import numpy as np

k = 1.0
H = np.array([[1, k],
              [0, 1]])

print(f"e1 -> {H @ [1, 0]}")  # [1, 0] — unchanged
print(f"e2 -> {H @ [0, 1]}")  # [1, 1] — slid to the right by k=1
```

**Output:**
```
e1 -> [1. 0.]
e2 -> [1. 1.]
```

```
  BEFORE                   AFTER  H (shear k=1)

        (0,1)                       (1,1)
        ^                          /
        |                         /
        |______> (1,0)           /______> (1,0)
        O                       O

  Square                   Parallelogram (leaning right)
  Area unchanged!          det(H) = 1
```

> **You Already Know This: CSS Skew**
>
> `transform: skewX(45deg)` in CSS is literally this shear matrix. Image processing libraries call it "skew" or "shear" — same operation, same matrix.

---

## Composition: Transformations Stack

Applying transformation $\mathbf{A}$ and then transformation $\mathbf{B}$ is the same as applying the single matrix $\mathbf{BA}$:

$$\mathbf{y} = \mathbf{B}(\mathbf{A}\mathbf{x}) = (\mathbf{BA})\mathbf{x}$$

Note the order: $\mathbf{B}$ is on the *left* because it's applied *second*. This is exactly like function composition: $g(f(x)) = (g \circ f)(x)$.

> **You Already Know This: Function Composition / Middleware Chaining**
>
> If you have middleware `f` then `g`, the composed pipeline is `g(f(request))`. Matrix composition works identically — and the same gotcha applies: **order matters**. `g(f(x))` is not the same as `f(g(x))`, and $\mathbf{BA} \neq \mathbf{AB}$ in general.

Let's prove it:

```python
import numpy as np

# Scale 2x horizontally, 0.5x vertically
S = np.array([[2,   0],
              [0, 0.5]])

# Rotate 45 degrees
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# Scale THEN rotate
SR = R @ S
# Rotate THEN scale
RS = S @ R

point = np.array([1, 0])

print(f"Scale then rotate: {SR @ point}")
print(f"Rotate then scale: {RS @ point}")
print(f"Same result? {np.allclose(SR @ point, RS @ point)}")
```

**Output:**
```
Scale then rotate: [1.41421356 1.41421356]
Rotate then scale: [1.41421356 0.35355339]
Same result? False
```

The point `[1, 0]` ends up in completely different places depending on the order. This is not a math curiosity — it matters every time you chain operations in a graphics pipeline, a data preprocessing step, or a neural network.

```
  Start: (1, 0)

  Path 1: Scale then Rotate          Path 2: Rotate then Scale
  (1,0) --scale--> (2, 0)            (1,0) --rotate--> (0.707, 0.707)
  (2,0) --rotate--> (1.41, 1.41)     (0.707, 0.707) --scale--> (1.41, 0.35)

  Different endpoints!
```

---

## Projection: Collapsing Dimensions

Projection is what happens when you drop information — mapping from a higher-dimensional space onto a lower-dimensional subspace. This is the geometric soul of dimensionality reduction.

**Projection onto a line** through the origin with direction $\mathbf{u}$:

$$\mathbf{P} = \frac{\mathbf{u}\mathbf{u}^T}{\mathbf{u}^T\mathbf{u}}$$

If $\hat{\mathbf{u}}$ is already a unit vector:

$$\mathbf{P} = \hat{\mathbf{u}}\hat{\mathbf{u}}^T$$

Two key properties of projection matrices:

- **Symmetric**: $\mathbf{P} = \mathbf{P}^T$
- **Idempotent**: $\mathbf{P}^2 = \mathbf{P}$ — projecting twice gives the same result as projecting once. Once you're on the line, you're on the line.

```python
import numpy as np

# Project onto the line y = x (direction [1, 1])
u = np.array([1, 1])
u_hat = u / np.linalg.norm(u)

P = np.outer(u_hat, u_hat)
print(f"Projection matrix onto y=x:\n{P}")

point = np.array([3, 1])
projected = P @ point
print(f"\n(3, 1) projected onto y=x: {projected}")

# Idempotent: project again — nothing changes
print(f"Project again: {P @ projected}")
print(f"Same? {np.allclose(projected, P @ projected)}")
```

**Output:**
```
Projection matrix onto y=x:
[[0.5 0.5]
 [0.5 0.5]]

(3, 1) projected onto y=x: [2. 2.]
Project again: [2. 2.]
Same? True
```

```
         ^
         |        * (3, 1)
         |       /
         |     /   <-- projection drops onto the line
         |   * (2, 2)
         | /
  -------O----------->
         |  line: y = x

  The point (3, 1) is projected to (2, 2) on the line y = x.
  The projection "drops" the component perpendicular to the line.
```

**Translation**: Projection kills the component of a vector that's orthogonal to the target subspace and keeps only the component along it. In ML, this is the core operation behind PCA: project high-dimensional data onto the directions of maximum variance.

---

## Change of Basis: Same Vector, Different Coordinates

A **basis** is a set of linearly independent vectors that span a space. The standard basis in 2D is $\{\mathbf{e}_1, \mathbf{e}_2\}$, but it's not the only one — and sometimes a different basis reveals structure you can't see in the standard one.

If $\mathbf{P}$ is a matrix whose columns are the new basis vectors, then:

- $\mathbf{x}_{new} = \mathbf{P}^{-1}\mathbf{x}_{standard}$ — convert from standard coordinates to new coordinates
- $\mathbf{x}_{standard} = \mathbf{P}\mathbf{x}_{new}$ — convert back

The vector itself hasn't moved. You're just describing the same arrow using different reference directions.

```python
import numpy as np

# New basis: 45-degree rotated axes
b1 = np.array([1, 1]) / np.sqrt(2)    # normalized
b2 = np.array([-1, 1]) / np.sqrt(2)   # perpendicular to b1

# Basis change matrix (new basis vectors as columns)
B = np.column_stack([b1, b2])
print(f"Basis matrix B:\n{B}")

# The point (1, 0) in standard coordinates
x_std = np.array([1, 0])

# Express in new basis
x_new = np.linalg.inv(B) @ x_std
print(f"\n(1, 0) in standard basis")
print(f"Same point in new basis: {x_new}")

# Convert back to verify
x_back = B @ x_new
print(f"Converted back: {x_back}")
```

**Output:**
```
Basis matrix B:
[[ 0.70710678 -0.70710678]
 [ 0.70710678  0.70710678]]

(1, 0) in standard basis
Same point in new basis: [0.70710678 0.70710678]
Converted back: [1. 0.]
```

```
  Standard basis:           New basis (45 degrees rotated):

        e2 ^                       b2 /  \ b1
           |                         /    \
           |                        /      \
           |___> e1                O
           O
                                The same point (1, 0) is described
                                as (0.707, 0.707) in the new basis.
```

**Why this matters in ML**: PCA is literally a change of basis. You find the eigenvectors of the covariance matrix, use them as your new basis, and suddenly your data's variance is aligned with the coordinate axes. The same data, described in the "right" coordinates, becomes much simpler.

---

## Full Transformation Visualization

Let's put it all together with a single example showing how different matrices transform the same unit square:

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_transformation(ax, matrix, title):
    """Visualize how a matrix transforms the unit square and basis vectors."""
    # Unit square vertices (closed polygon)
    square = np.array([[0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 0]])

    transformed = matrix @ square

    # Plot original
    ax.fill(square[0], square[1], alpha=0.2, color='blue', label='Original')
    ax.plot(square[0], square[1], 'b-', linewidth=1)

    # Plot transformed
    ax.fill(transformed[0], transformed[1], alpha=0.2, color='red', label='Transformed')
    ax.plot(transformed[0], transformed[1], 'r-', linewidth=2)

    # Basis vectors: original
    ax.annotate('', xy=(1, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax.annotate('', xy=(0, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Basis vectors: transformed
    col1 = matrix @ [1, 0]
    col2 = matrix @ [0, 1]
    ax.annotate('', xy=col1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=col2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_xlim(-2.5, 3)
    ax.set_ylim(-2.5, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.set_title(title)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

theta = np.pi / 4
transformations = [
    (np.array([[2, 0], [0, 0.5]]),        "Scaling (2x, 0.5y)"),
    (np.array([[np.cos(theta), -np.sin(theta)],
               [np.sin(theta),  np.cos(theta)]]), "Rotation (45 deg)"),
    (np.array([[1, 0], [0, -1]]),          "Reflection (x-axis)"),
    (np.array([[1, 1], [0, 1]]),           "Shear (k=1)"),
]

for ax, (M, title) in zip(axes.flat, transformations):
    plot_transformation(ax, M, title)

plt.tight_layout()
plt.savefig("transformations.png", dpi=150)
plt.show()
```

---

## Neural Networks: Composed Transformations

This is where everything connects. A neural network layer does exactly what we've been discussing:

$$f_i(\mathbf{x}) = \sigma(\mathbf{W}_i\mathbf{x} + \mathbf{b}_i)$$

Three operations chained together:

1. **Linear transformation**: $\mathbf{W}_i\mathbf{x}$ — rotate, scale, project (the matrix is the learned transformation)
2. **Translation**: $+ \mathbf{b}_i$ — shift the origin (bias term)
3. **Non-linearity**: $\sigma(\cdot)$ — bend space (ReLU, sigmoid, etc.)

A deep network is just a composition of these:

$$\mathbf{y} = f_L(f_{L-1}(\cdots f_2(f_1(\mathbf{x}))\cdots))$$

> **You Already Know This: Middleware Chaining (Again)**
>
> A deep neural network is literally a pipeline of transformations, each one taking the output of the previous as input. The weight matrices are the "configuration" of each middleware function. Training is the process of tuning those configurations so the pipeline maps inputs to the correct outputs.

### Why Depth Matters (and Why Non-Linearities Are Non-Negotiable)

Here's a critical insight. Without non-linearities between layers:

$$\mathbf{W}_2(\mathbf{W}_1\mathbf{x}) = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x} = \mathbf{W}_{combined}\mathbf{x}$$

Multiple linear layers without activations **collapse into a single matrix**. You get nothing from stacking them. This is a direct consequence of composition of linear transformations being linear.

The non-linearity ($\sigma$) breaks this collapse. It bends the space between layers so the composition can represent transformations that no single matrix could.

```python
import numpy as np

# === A Neural Network Layer Is a Transformation ===
# Input: batch of 3 points in 2D
X = np.array([[1, 0],
              [0, 1],
              [1, 1]])

# Weight matrix: learned transformation (2D -> 3D)
W = np.array([[1, 0],
              [0, 1],
              [1, 1]])

# Bias vector
b = np.array([0, -0.5, 0])

# Linear transformation: Z = XW^T + b
Z = X @ W.T + b
print(f"Input X (3 samples, 2 features):\n{X}")
print(f"\nWeight matrix W (transforms 2D -> 3D):\n{W}")
print(f"\nAfter linear transformation Z = XW^T + b:\n{Z}")

# ReLU: the non-linearity that prevents layer collapse
Z_relu = np.maximum(0, Z)
print(f"\nAfter ReLU activation (bends the space):\n{Z_relu}")

# === Demonstrating layer collapse without non-linearity ===
W1 = np.array([[1, 2], [3, 4]])
W2 = np.array([[5, 6], [7, 8]])

x = np.array([1, 1])
two_layers = W2 @ (W1 @ x)
one_layer = (W2 @ W1) @ x

print(f"\nTwo layers without activation: {two_layers}")
print(f"Single combined matrix:        {one_layer}")
print(f"Identical? {np.allclose(two_layers, one_layer)}")  # True!
```

**Output:**
```
Input X (3 samples, 2 features):
[[1 0]
 [0 1]
 [1 1]]

Weight matrix W (transforms 2D -> 3D):
[[1 0]
 [0 1]
 [1 1]]

After linear transformation Z = XW^T + b:
[[ 1.  -0.5  1. ]
 [ 0.   0.5  1. ]
 [ 1.   0.5  2. ]]

After ReLU activation (bends the space):
[[1.  0.  1. ]
 [0.  0.5 1. ]
 [1.  0.5 2. ]]

Two layers without activation: [23 31]
Single combined matrix:        [23 31]
Identical? True
```

Notice how ReLU zeroed out the $-0.5$ in the first row. That's the non-linearity *bending* the transformation — creating a piecewise-linear boundary that a single matrix can never produce.

---

## Where Transformations Show Up in ML

| Transformation | ML Application | What It Does Geometrically |
|----------------|----------------|---------------------------|
| Scaling | Feature normalization, batch norm | Rescales axes so features are comparable |
| Rotation | PCA (finding principal components) | Rotates to align with variance directions |
| Projection | Dimensionality reduction, attention | Collapses onto a lower-dimensional subspace |
| Composition | Deep networks (stacking layers) | Builds complex warps from simple ones |
| Change of basis | PCA, whitening, diagonalization | Describes data in more natural coordinates |
| Shear | Data augmentation, image distortion | Tilts the coordinate grid |

---

## Common Mistakes

1. **Forgetting that order matters in composition**: $\mathbf{AB} \neq \mathbf{BA}$ in general. When you chain two transformations, the *rightmost* matrix is applied *first*. This trips up everyone at least once.

2. **Confusing row and column vector conventions**: Some textbooks (and some ML frameworks) use row vectors with $\mathbf{x}\mathbf{W}$ instead of column vectors with $\mathbf{W}\mathbf{x}$. PyTorch's `nn.Linear` stores weights transposed relative to the mathematical convention. Be aware which convention you're in.

3. **Thinking you need the inverse**: Computing $\mathbf{A}^{-1}$ explicitly is almost always the wrong move numerically. Use `np.linalg.solve(A, b)` instead of `np.linalg.inv(A) @ b`. Same answer, much more stable.

4. **Forgetting that linear transformations fix the origin**: If your data needs to be shifted (translated), you need an affine transformation — that's $\mathbf{W}\mathbf{x} + \mathbf{b}$, not just $\mathbf{W}\mathbf{x}$. This is exactly why neural network layers have bias terms.

5. **Not visualizing in 2D first**: Every transformation concept in 2D generalizes to $n$ dimensions. If you can't picture what a transformation does to a 2D square, you definitely won't understand what it does to a 768-dimensional embedding. Start small.

---

## Exercises

### Exercise 1: Read the Matrix

Look at this matrix and, *without computing anything*, describe what transformation it represents. Then verify with code.

$$\mathbf{A} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

*Hint: Where does $\mathbf{e}_1$ go? Where does $\mathbf{e}_2$ go?*

**Solution:**

$\mathbf{e}_1 = (1, 0) \to (0, 1)$ — that's a 90-degree rotation.
$\mathbf{e}_2 = (0, 1) \to (-1, 0)$ — consistent with 90-degree counterclockwise rotation.

```python
import numpy as np

A = np.array([[0, -1],
              [1,  0]])

# Verify: this should be a 90-degree rotation
theta = np.pi / 2
R_90 = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta),  np.cos(theta)]])

print(f"Our matrix A:\n{A}")
print(f"Rotation matrix for 90 degrees:\n{np.round(R_90)}")
print(f"Match? {np.allclose(A, R_90)}")

v = np.array([1, 0])
print(f"\n[1, 0] rotated 90 degrees CCW: {A @ v}")  # [0, 1]
```

---

### Exercise 2: Projection Matrix

Create a projection matrix onto the x-axis. Project the vector $[3, 4]$ and verify that projecting twice gives the same result.

**Solution:**

```python
import numpy as np

# Direction of x-axis: [1, 0] (already a unit vector)
u = np.array([1, 0])

# Projection matrix: P = u * u^T
P = np.outer(u, u)
print(f"Projection matrix onto x-axis:\n{P}")

v = np.array([3, 4])
projected = P @ v
print(f"\n[3, 4] projected onto x-axis: {projected}")  # [3, 0]

# Verify idempotent: P(Pv) = Pv
print(f"Projected again: {P @ projected}")  # [3, 0]
print(f"Idempotent? {np.allclose(projected, P @ projected)}")  # True
```

---

### Exercise 3: Composition Order

Apply scaling by 2 in both directions, then rotate 45 degrees. Compare with rotating first, then scaling. Apply both to $[1, 0]$ and show they give different results.

**Solution:**

```python
import numpy as np

# Uniform scaling
S = np.array([[2, 0],
              [0, 2]])

# 45-degree rotation
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# Composition 1: Scale first, then rotate -> T1 = R @ S
T1 = R @ S
# Composition 2: Rotate first, then scale -> T2 = S @ R
T2 = S @ R

v = np.array([1, 0])

print(f"Scale then rotate: {T1 @ v}")
print(f"Rotate then scale: {T2 @ v}")
print(f"Same? {np.allclose(T1 @ v, T2 @ v)}")

# Interesting: for UNIFORM scaling, these actually ARE the same!
# Uniform scaling commutes with rotation.
# Try non-uniform scaling to see the difference:
S_nonuniform = np.array([[2, 0],
                          [0, 0.5]])

T3 = R @ S_nonuniform  # scale then rotate
T4 = S_nonuniform @ R  # rotate then scale

print(f"\nWith non-uniform scaling:")
print(f"Scale then rotate: {T3 @ v}")
print(f"Rotate then scale: {T4 @ v}")
print(f"Same? {np.allclose(T3 @ v, T4 @ v)}")
# NOW they differ — non-uniform scaling doesn't commute with rotation
```

---

### Exercise 4: Neural Network Layer by Hand

You have a weight matrix $\mathbf{W} = \begin{bmatrix} 1 & -1 \\ 2 & 0.5 \end{bmatrix}$, bias $\mathbf{b} = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$, and input $\mathbf{x} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}$. Compute the output of one layer with ReLU activation: $\text{ReLU}(\mathbf{W}\mathbf{x} + \mathbf{b})$.

**Solution:**

```python
import numpy as np

W = np.array([[1, -1],
              [2, 0.5]])
b = np.array([0, -1])
x = np.array([3, 2])

# Step 1: Linear transformation
z = W @ x + b
print(f"W @ x = {W @ x}")       # [1, 7]
print(f"W @ x + b = {z}")       # [1, 6]

# Step 2: ReLU
output = np.maximum(0, z)
print(f"ReLU(z) = {output}")    # [1, 6] — both positive, so no change

# Try with an input that produces a negative pre-activation
x2 = np.array([1, 3])
z2 = W @ x2 + b
output2 = np.maximum(0, z2)
print(f"\nWith x = {x2}:")
print(f"W @ x + b = {z2}")      # [-2, 2.5]
print(f"ReLU(z) = {output2}")   # [0, 2.5] — ReLU kills the negative
```

---

## Summary

- A matrix **is** a linear transformation. Multiplying a matrix by a vector transforms that vector — stretching, rotating, reflecting, projecting, or some combination.
- **The columns of a matrix tell you where the basis vectors land.** That's the single most important thing to remember. If you know where $\mathbf{e}_1$ and $\mathbf{e}_2$ go, you know the entire transformation.
- Standard transformations — **scaling, rotation, reflection, shear, projection** — each have clean matrix formulas that follow directly from asking "where do the basis vectors go?"
- **Composition** of transformations is matrix multiplication. The rightmost matrix is applied first. Order matters.
- **Change of basis** lets you describe the same vectors in different coordinate systems. PCA is a change of basis.
- **Projection** collapses dimensions. Projection matrices are symmetric and idempotent ($\mathbf{P}^2 = \mathbf{P}$).
- **Every neural network layer is a linear transformation ($\mathbf{W}\mathbf{x}$) plus a bias ($+ \mathbf{b}$) plus a non-linearity ($\sigma$)**. Without the non-linearity, stacked layers collapse into a single matrix — you get no benefit from depth.
- Visualize in 2D. The intuition transfers directly to 768 dimensions — the geometry is the same, just harder to draw.

---

## What's Next

Transformations are powerful, but real ML problems give you equations to solve: "Given these inputs and outputs, find the transformation." That's **systems of linear equations** — and it's where linear algebra becomes the workhorse behind regression, least squares, and model fitting.
