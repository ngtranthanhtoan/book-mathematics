# Chapter 1: Coordinate Systems

## Building On

In Level 4, we broke matrices apart. LU, QR, SVD -- each decomposition revealed hidden structure by factoring a matrix into simpler components. But here is the question we never stopped to ask: **what space do those components live in?**

When SVD decomposes a user-movie ratings matrix into $U \Sigma V^T$, the columns of $U$ and $V$ are vectors in some coordinate system. Change that coordinate system -- rotate it, scale it, skew it -- and the same data tells a completely different story. PCA literally *is* a coordinate system change: it rotates your axes to align with the directions of maximum variance.

So before we can truly understand what decompositions *mean*, we need to understand coordinate systems themselves -- how we describe where points live, how different descriptions relate to each other, and why the choice of coordinates is one of the most consequential decisions you make in any ML pipeline.

---

## The Problem That Makes This Click

You have trained an image classifier. It works beautifully on your test set -- 97% accuracy. You deploy it, and within a week, accuracy drops to 73%.

After some debugging, you discover the issue. Your training images were 224x224 RGB pixels -- that is 150,528 raw features per image. Your model learned to classify in this raw pixel coordinate system. But in production, images arrive with slightly different lighting, different cameras, different white balance.

A colleague suggests: "Run PCA first. Project the images into a lower-dimensional coordinate system that captures the important variation and ignores the noise."

You try it. Accuracy recovers to 94%.

**What just happened?** You changed the coordinate system. Instead of representing each image as a point in 150,528-dimensional raw-pixel space, you represented it as a point in, say, 500-dimensional PCA space. Same images, same data, completely different representation -- and dramatically different model performance.

This is the core insight of this chapter: **coordinate systems are not just a mathematical formality. They are an engineering choice that directly affects whether your ML pipeline works or fails.**

---

## Code First: See It Before You Formalize It

Let's start where you are comfortable -- in a terminal. Before any definitions or formulas, let's watch coordinate systems in action.

### Experiment 1: Same data, different coordinates, different results

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Generate a classic ML toy dataset: two interleaving half-moons
np.random.seed(42)
X, y = make_moons(n_samples=200, noise=0.2)

# Add 50 pure-noise features (simulating irrelevant dimensions)
noise_features = np.random.randn(200, 50) * 0.5
X_noisy = np.hstack([X, noise_features])

print(f"Original feature space:  {X.shape[1]} dimensions")
print(f"Noisy feature space:     {X_noisy.shape[1]} dimensions")

# KNN in the original 2D coordinate system
knn_original = KNeighborsClassifier(n_neighbors=5)
knn_original.fit(X[:160], y[:160])
acc_original = knn_original.score(X[160:], y[160:])

# KNN in the noisy 52D coordinate system
knn_noisy = KNeighborsClassifier(n_neighbors=5)
knn_noisy.fit(X_noisy[:160], y[:160])
acc_noisy = knn_noisy.score(X_noisy[160:], y[160:])

# KNN after PCA projects back to 2D (coordinate system change!)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X_noisy))
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_pca[:160], y[:160])
acc_pca = knn_pca.score(X_pca[160:], y[160:])

print(f"\nKNN accuracy in original 2D space:   {acc_original:.0%}")
print(f"KNN accuracy in noisy 52D space:     {acc_noisy:.0%}")
print(f"KNN accuracy after PCA back to 2D:   {acc_pca:.0%}")
```

Run this. You will see the noisy coordinate system kills KNN performance, and PCA (a coordinate system change) largely recovers it. The data did not change -- the *coordinate system* did.

> **You Already Know This**: If you have ever worked with CSS, you have dealt with coordinate systems daily. An element positioned with `top: 50px; left: 100px` looks completely different depending on whether its parent has `position: relative` or not. Same numbers, different coordinate system, different result. The ML version of this is: same feature values, different coordinate representation (raw pixels vs. PCA components vs. embeddings), different model behavior.

---

## What Is a Coordinate System, Really?

Now that you have seen the *why*, let's nail down the *what*.

A **coordinate system** is an agreement about three things:

1. **An origin** -- the "zero point" everything is measured from
2. **A set of axes** (basis directions) -- the directions you measure along
3. **A scale** -- the units along each axis

That is it. With these three things, any point in space can be described as a list of numbers -- its **coordinates**.

### The Cartesian coordinate system

The one you know from school. In 2D:

```
          y
          ^
          |
     4  --+------------------*  P(3, 4)
          |                . |
     3  --+-----------  .    |
          |          .       |
     2  --+-------.          |     P is at x=3, y=4
          |     .            |     which means: "go 3 right, 4 up"
     1  --+---.              |
          |.                 |
   -------O--+--+--+--+--+--+-------> x
          0  1  2  3  4  5
          |
    -1  --+
```

The point $P = (3, 4)$ means: starting from the origin $O$, walk 3 units along the x-axis and 4 units along the y-axis.

In 3D, we add a z-axis perpendicular to both:

```
            z
            ^
            |
            |   /  P(2, 3, 4)
        4 --+-/- - - - - *
            |/          /|
        3 --+          / |
           /|         /  |
        2 -/--       /   |
         / |        /    |
     ---O--+--+--+--+--+-+---> y
       /   0  1  2  3  4
      /    |
     v     |
     x
```

And in $n$ dimensions? Mathematically, nothing changes. We just can't draw it anymore.

### Formal definition

A **Cartesian coordinate system** in $\mathbb{R}^n$ consists of:

1. An origin $O = (0, 0, \ldots, 0)$
2. $n$ mutually perpendicular (orthogonal) basis vectors $\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n$
3. A unit of measurement along each axis

Any point $P$ in this space is uniquely identified by an ordered $n$-tuple:

$$P = (x_1, x_2, \ldots, x_n) \in \mathbb{R}^n$$

where $x_i$ is the signed distance from the origin along the $i$-th axis.

**Translation**: A point's coordinates are just its "recipe" -- they tell you how much of each basis direction to combine to get from the origin to that point.

The **standard basis vectors** in $\mathbb{R}^3$ are:

$$\mathbf{e}_1 = (1, 0, 0), \quad \mathbf{e}_2 = (0, 1, 0), \quad \mathbf{e}_3 = (0, 0, 1)$$

And any point can be written as a weighted combination (linear combination) of these basis vectors:

$$\vec{OP} = x_1\mathbf{e}_1 + x_2\mathbf{e}_2 + \cdots + x_n\mathbf{e}_n = \sum_{i=1}^{n} x_i\mathbf{e}_i$$

**Translation**: This formula says exactly what you would expect -- to reach point $P$, walk $x_1$ units in the $\mathbf{e}_1$ direction, then $x_2$ units in the $\mathbf{e}_2$ direction, and so on. The coordinates *are* the weights.

> **You Already Know This**: Think of a database index. A composite index on `(city, zipcode, street)` is a coordinate system for addresses -- three axes, each adding a dimension of specificity. An address is a "point" in that index space. Changing the index (adding columns, reordering them) changes how efficiently you can "find" nearby points -- exactly like changing coordinate systems changes how well KNN finds neighbors.

---

## From Points to Feature Spaces

Here is where coordinate systems meet machine learning.

When you load a tabular dataset with $n$ columns, each row is a point in $n$-dimensional space. The columns *are* the axes. The values *are* the coordinates:

```
                        Feature Space: R^n
  ================================================================

  Your DataFrame:                    Geometric view:
  +---------+--------+------+        Each row is a point:
  | height  | weight | age  |
  +---------+--------+------+          height
  | 170     | 70     | 25   |  ---->    ^
  | 160     | 55     | 30   |           |     * (170,70,25)
  | 180     | 85     | 22   |           |  * (160,55,30)
  | 165     | 60     | 35   |           | * (180,85,22)
  +---------+--------+------+           |    * (165,60,35)
                                        +---------> weight
  4 samples, 3 features               /
  = 4 points in R^3                  v
                                    age
```

| Mathematical Concept | ML Interpretation | SWE Analogy |
|---------------------|-------------------|-------------|
| Dimension $n$ | Number of features | Number of columns in a table |
| Coordinate $x_i$ | Value of feature $i$ | Cell value at column $i$ |
| Point $\mathbf{x}$ | One data sample | One row in the table |
| Set of points | Dataset | The entire table |
| Origin | Zero vector (or mean-centered reference) | Default/null record |
| Basis vector $\mathbf{e}_i$ | "Pure" unit of one feature | A column with value 1 and all others 0 |

### Experiment 2: Your dataset is literally a coordinate system

```python
import numpy as np

# A small "dataset" -- 4 people with 3 features each
data = np.array([
    [170, 70, 25],   # Person 0: height=170cm, weight=70kg, age=25
    [160, 55, 30],   # Person 1
    [180, 85, 22],   # Person 2
    [165, 60, 35],   # Person 3
])

print(f"Dataset shape: {data.shape}")
print(f"  {data.shape[0]} samples (points)")
print(f"  {data.shape[1]} features (dimensions)")

# Each sample IS its coordinates in feature space
person_0 = data[0]
print(f"\nPerson 0 coordinates: {person_0}")
print(f"  height axis: {person_0[0]}")
print(f"  weight axis: {person_0[1]}")
print(f"  age axis:    {person_0[2]}")

# The centroid (mean point) -- the "center of mass" of the data cloud
centroid = np.mean(data, axis=0)
print(f"\nCentroid: {centroid}")
print(f"  average height: {centroid[0]:.1f}")
print(f"  average weight: {centroid[1]:.1f}")
print(f"  average age:    {centroid[2]:.1f}")
```

**Translation**: `np.mean(data, axis=0)` computes the centroid -- the point at the geometric center of your data. This is the same "mean point" that K-Means iteratively computes for each cluster. The formula for the centroid of $m$ points is:

$$\bar{\mathbf{x}} = \frac{1}{m}\sum_{j=1}^{m} \mathbf{x}_j = \left(\frac{1}{m}\sum_{j=1}^{m} x_{j1},\ \frac{1}{m}\sum_{j=1}^{m} x_{j2},\ \ldots,\ \frac{1}{m}\sum_{j=1}^{m} x_{jn}\right)$$

That just says: average each coordinate separately. Nothing mysterious.

---

## Coordinate Operations You Will Use Constantly

### Midpoint

The midpoint $M$ between two points $P$ and $Q$ is the average of their coordinates:

$$M = \frac{P + Q}{2} = \left(\frac{x_P + x_Q}{2},\ \frac{y_P + y_Q}{2},\ \ldots \right)$$

More generally, the point that is fraction $t$ of the way from $P$ to $Q$:

$$R = P + t(Q - P) = (1 - t)P + tQ$$

When $t = 0$, you are at $P$. When $t = 1$, you are at $Q$. When $t = 0.5$, you are at the midpoint.

**Translation**: This is linear interpolation -- the `lerp` function from game development and graphics programming. If you have ever animated a smooth transition between two positions, you used this exact formula.

> **You Already Know This**: The parametric interpolation $R = (1-t)P + tQ$ is the same `lerp` you have written a hundred times. In CSS animations, in easing functions, in smooth scrolling. Now you know the math name for it: **affine combination**.

### Mean centering

In ML, we often *shift the origin* to be at the centroid of the data. This is called **mean centering**:

$$\tilde{\mathbf{x}}_j = \mathbf{x}_j - \bar{\mathbf{x}}$$

After centering, the centroid sits at the origin $(0, 0, \ldots, 0)$.

Why bother? Because many algorithms (PCA, SVD-based methods, many neural network initializations) work better -- or only work correctly -- when data is centered. It removes the "offset" and lets the algorithm focus on the *variation* in the data rather than its absolute position.

```python
import numpy as np

data = np.array([
    [170, 70, 25],
    [160, 55, 30],
    [180, 85, 22],
    [165, 60, 35],
])

# Mean centering
centroid = np.mean(data, axis=0)
centered = data - centroid

print("Original centroid:", np.mean(data, axis=0))
print("Centered centroid:", np.mean(centered, axis=0))  # [0, 0, 0]

print("\nOriginal data:\n", data)
print("\nCentered data:\n", centered)
# Now each coordinate tells you "how far from average" instead of absolute value
```

> **Common Mistake**: Mean centering is *not* the same as normalization. Centering shifts the origin but does not change the scale. If height ranges from 160-180 (range 20) and age ranges from 22-35 (range 13), centering does not fix the scale mismatch. You need standardization for that (subtract mean *and* divide by standard deviation).

---

## The Scale Problem: Why Normalization Is a Coordinate System Fix

Here is a bug that bites every ML engineer at least once.

You build a KNN classifier on a dataset with features `[height_cm, weight_kg, age_years]`. Heights range from 150-190 (range 40), weights from 45-95 (range 50), ages from 18-65 (range 47). Seems similar enough, right?

But now imagine you convert height to *millimeters*. Heights now range from 1500-1900 (range 400). **Nothing about the data changed.** You just used different units -- a different *scale* on one axis. But KNN's distance calculations are now completely dominated by the height dimension:

```
  Without normalization:           With normalization:

  height(mm)                       height(std)
  ^                                ^
  | * * * *                        |    *   *
  | * * * *                        |  *   *
  | * * * *                        |    *  *
  | * * * *                        |  *  *
  | * * * *                        |    *  *
  +-----------> weight(kg)         +-----------> weight(std)

  Points form a narrow              Points form a roughly
  vertical stripe -- height         circular cloud -- all
  dominates all distances           dimensions contribute equally
```

### Experiment 3: Scale distortion in action

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)

# Two clusters, clearly separated in ALL dimensions
cluster_0 = np.random.randn(50, 3) + np.array([0, 0, 0])
cluster_1 = np.random.randn(50, 3) + np.array([2, 2, 2])
X = np.vstack([cluster_0, cluster_1])
y = np.array([0]*50 + [1]*50)

# KNN with balanced scales -- works great
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X[:80], y[:80])
print(f"Balanced scales accuracy: {knn.score(X[80:], y[80:]):.0%}")

# Now "convert height to millimeters" -- multiply first feature by 100
X_distorted = X.copy()
X_distorted[:, 0] *= 100  # just a unit change!

knn_distorted = KNeighborsClassifier(n_neighbors=5)
knn_distorted.fit(X_distorted[:80], y[:80])
print(f"After scaling one feature x100: {knn_distorted.score(X_distorted[80:], y[80:]):.0%}")

# Fix it with standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_fixed = scaler.fit_transform(X_distorted)

knn_fixed = KNeighborsClassifier(n_neighbors=5)
knn_fixed.fit(X_fixed[:80], y[:80])
print(f"After standardization: {knn_fixed.score(X_fixed[80:], y[80:]):.0%}")
```

**Standardization** (z-score normalization) transforms each feature to have mean 0 and standard deviation 1:

$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$

where $\mu_i$ is the mean and $\sigma_i$ is the standard deviation of feature $i$.

**Translation**: Standardization is a coordinate system change. You are moving the origin to the centroid (subtracting the mean) and then rescaling each axis so that one "unit" means "one standard deviation." After standardization, all features live on the same scale. It is the geometric equivalent of converting all measurements to the same units before comparing them.

**Min-max normalization** is another option -- it rescales each feature to the range $[0, 1]$:

$$x_i^{\prime} = \frac{x_i - \min_i}{\max_i - \min_i}$$

| Method | When to Use |
|--------|-------------|
| Standardization (z-score) | Features are roughly Gaussian; you want to preserve outlier information |
| Min-max normalization | You need bounded values (e.g., for neural networks expecting [0,1] input) |
| No normalization | Tree-based models (they split on thresholds, not distances) |

> **Common Mistake**: Never fit the scaler on test data. Fit on training data, then *transform* both training and test data using those same parameters. Otherwise you are leaking information from the test set into your preprocessing -- a subtle but devastating form of data leakage.

---

## Coordinate System Changes: The Big Idea

Here is the concept that ties this entire chapter to real ML practice.

A **coordinate system change** (also called a **change of basis**) re-expresses the same points using different axes. The data does not move. Only our *description* of it changes.

Think of it this way. You have a point $P$ in a room. One person describes $P$'s location relative to the front door. Another describes it relative to the window. They give different numbers, but they are talking about the same physical point.

In matrix form, a change of basis looks like this:

$$\mathbf{x}' = A\mathbf{x}$$

where $\mathbf{x}$ is the point in the old coordinates, $A$ is the change-of-basis matrix, and $\mathbf{x}'$ is the same point in new coordinates.

**Translation**: $A$ is a translation table between two coordinate languages. Multiply your old coordinates by $A$, and you get the new coordinates. That is all a change of basis is.

> **You Already Know This**: If you have ever done graphics programming, the model-view-projection pipeline is exactly this -- a chain of coordinate system changes. Model coordinates -> world coordinates -> camera coordinates -> screen coordinates. Each step is a matrix multiplication. ML does the same thing: raw features -> standardized features -> PCA features -> model input.

### The ML coordinate changes you will use

Here is a visual map of the most common coordinate transformations in ML:

```
   Raw Feature Space          Standardized Space         PCA Space
   (your DataFrame)           (mean=0, std=1)            (aligned to variance)

      height                     z_height                   PC1
        ^                          ^                         ^
        |   * *                    |    *  *                  |  * *
        |  *  *                    |  *   *                   | * *
        | *  *                     | *    *                   |* *
        +-------> weight           +---------> z_weight       +-------> PC2

   Axes = original features    Same axes, rescaled         Axes = directions of
                                                           maximum variance

   Step 1: StandardScaler()    Step 2: PCA()
       x' = (x - mu) / sigma      x'' = W^T * x'
```

Each of these is a coordinate system change:

1. **Standardization**: shift origin to centroid, rescale each axis
2. **PCA**: rotate axes to align with directions of maximum variance
3. **Embedding layers** (Word2Vec, autoencoders): learn a completely new coordinate system where "nearby" means "semantically similar"

### Experiment 4: PCA as a coordinate system rotation

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Generate correlated 2D data (points along a diagonal)
# This simulates features that are NOT independent
theta = np.pi / 4  # 45 degrees
n = 100
t = np.random.randn(n) * 2
noise = np.random.randn(n) * 0.3

X = np.column_stack([
    t * np.cos(theta) + noise * np.sin(theta),
    t * np.sin(theta) - noise * np.cos(theta)
])

print("Original coordinates (first 5 points):")
for i in range(5):
    print(f"  ({X[i,0]:+.2f}, {X[i,1]:+.2f})")

# PCA finds new axes aligned with the data's spread
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

print(f"\nPCA directions (new basis vectors):")
for i, comp in enumerate(pca.components_):
    print(f"  PC{i+1}: ({comp[0]:+.3f}, {comp[1]:+.3f})")

print(f"\nVariance explained by each new axis:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.1%}")

print(f"\nPCA coordinates (first 5 points):")
for i in range(5):
    print(f"  ({X_pca[i,0]:+.2f}, {X_pca[i,1]:+.2f})")

print(f"\nNotice: in PCA coordinates, PC1 captures most variance.")
print(f"We could drop PC2 and keep {pca.explained_variance_ratio_[0]:.1%} of the information.")
```

Here is what happened visually:

```
  Original coordinate system:         After PCA rotation:

      y                                  PC2
      ^                                   ^
      |     * * *                         |
      |   * * * *                     * * | * *
      | * * * *                     * * * | * * *
      |* * * *                    * * * * | * * * *
      +-----------> x               ------+---------> PC1
                                          |
  Data lies along a diagonal.         Data is now aligned with
  Both x and y needed to              the axes. PC1 alone captures
  describe the spread.                most of the information.
```

**This is dimensionality reduction.** PCA found a new coordinate system where the first axis (PC1) captures most of the variation. You can throw away PC2 with minimal information loss.

---

## High Dimensions: Where Intuition Breaks

Everything so far generalizes cleanly from 2D to $n$ dimensions. The formulas are identical -- just longer sums. But your geometric intuition from 2D/3D *does not* generalize to high dimensions. Here are the traps.

### The curse of dimensionality

In high-dimensional spaces, distances become meaningless. Here is a demonstration that will surprise you:

```python
import numpy as np

np.random.seed(42)

for n_dims in [2, 10, 100, 1000, 10000]:
    # Generate random points in the unit hypercube [0,1]^n
    points = np.random.rand(100, n_dims)

    # Calculate all pairwise distances
    from scipy.spatial.distance import pdist
    distances = pdist(points)

    ratio = distances.max() / distances.min()
    print(f"Dims={n_dims:>5}: "
          f"mean dist = {distances.mean():.2f}, "
          f"max/min ratio = {ratio:.2f}")
```

As dimensions increase, the ratio of the farthest distance to the nearest distance approaches 1. **All points become roughly equidistant.** This is devastating for distance-based algorithms like KNN and K-Means, which rely on meaningful distance differences.

> **Common Mistake**: Do not assume that what works in 2D or 3D will work in 100D. In high dimensions, most of the volume of a hypersphere is concentrated near its surface. The "center" of a high-dimensional distribution is almost empty. Your intuition about "most points are near the center" is flat wrong in high dimensions.

### Why this matters practically

| Dimension | Behavior | Consequence for ML |
|-----------|----------|-------------------|
| 2-3 | Distances meaningful, visualization possible | KNN, K-Means work well |
| 10-50 | Distances still useful with enough data | Standard ML algorithms work |
| 100-1000 | Curse kicks in; distances converge | Need dimensionality reduction or different metrics |
| 10,000+ | Raw distances nearly useless | Must use embeddings, manifold learning, or domain-specific metrics |

---

## Non-Cartesian Coordinate Systems (Brief Tour)

Cartesian coordinates are not the only game in town. Two alternatives show up in ML and related fields.

### Polar / Spherical coordinates

Instead of "go right $x$, go up $y$," polar coordinates say "go distance $r$ at angle $\theta$":

```
          y
          ^                            Polar:  P = (r, theta)
          |                            Cartesian: P = (x, y)
     4  --+--------*  P
          |       /|                   r = sqrt(x^2 + y^2) = 5
     3  --+      / |                   theta = atan2(y, x) = 53.1 deg
          |     /  |
     2  --+  r/   |                   Conversion:
          |  /    |                     x = r * cos(theta) = 3
     1  --+ / th  |                     y = r * sin(theta) = 4
          |/ _____|
   -------O-----------> x
          0  1  2  3
```

**Where this shows up in ML**: Anywhere you care about magnitude and direction separately. For example, in word embeddings, the *direction* of a word vector captures its semantic meaning while the *magnitude* captures something like word frequency. Cosine similarity explicitly uses this angular view.

The conversion formulas between Cartesian and polar:

$$x = r\cos\theta, \quad y = r\sin\theta$$
$$r = \sqrt{x^2 + y^2}, \quad \theta = \text{atan2}(y, x)$$

### Homogeneous coordinates

In computer graphics and robotics (and some ML geometric pipelines), we use **homogeneous coordinates** -- adding an extra dimension to make translations representable as matrix multiplications:

$$\text{2D point } (x, y) \rightarrow \text{homogeneous } (x, y, 1)$$

This lets you combine rotation, scaling, AND translation into a single matrix multiply:

$$\begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta & t_x \\ \sin\theta & \cos\theta & t_y \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$

> **You Already Know This**: If you have touched CSS `transform: matrix(a, b, c, d, tx, ty)` or worked with OpenGL/WebGL model matrices, you have used homogeneous coordinates. That mysterious 4x4 matrix in every 3D graphics tutorial? Homogeneous coordinates.

---

## Putting It All Together: The ML Coordinate Pipeline

Let's build a complete example that ties every concept in this chapter together. We will take raw data, apply coordinate transformations, and see how each step affects model performance.

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Load a real dataset: Wine recognition (13 features, 3 classes)
wine = load_wine()
X, y = wine.data, wine.target
print(f"Wine dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Feature names: {wine.feature_names}")

# Look at the scale problem
print(f"\nFeature ranges (showing the scale mismatch):")
for i, name in enumerate(wine.feature_names):
    print(f"  {name:>30s}: [{X[:,i].min():.1f}, {X[:,i].max():.1f}]"
          f"  (range = {X[:,i].max() - X[:,i].min():.1f})")

# Coordinate system 1: Raw features (no transformation)
raw_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=5), X, y, cv=5)

# Coordinate system 2: Standardized features
std_scores = cross_val_score(
    Pipeline([('scaler', StandardScaler()),
              ('knn', KNeighborsClassifier(n_neighbors=5))]),
    X, y, cv=5)

# Coordinate system 3: Min-max normalized
minmax_scores = cross_val_score(
    Pipeline([('scaler', MinMaxScaler()),
              ('knn', KNeighborsClassifier(n_neighbors=5))]),
    X, y, cv=5)

# Coordinate system 4: PCA (after standardization)
pca_scores = cross_val_score(
    Pipeline([('scaler', StandardScaler()),
              ('pca', PCA(n_components=5)),
              ('knn', KNeighborsClassifier(n_neighbors=5))]),
    X, y, cv=5)

print(f"\nKNN accuracy under different coordinate systems:")
print(f"  Raw features (13D):          {raw_scores.mean():.1%} +/- {raw_scores.std():.1%}")
print(f"  Standardized (13D):          {std_scores.mean():.1%} +/- {std_scores.std():.1%}")
print(f"  Min-max normalized (13D):    {minmax_scores.mean():.1%} +/- {minmax_scores.std():.1%}")
print(f"  Standardized + PCA (5D):     {pca_scores.mean():.1%} +/- {pca_scores.std():.1%}")

print(f"\nSame data, same algorithm, different coordinate systems.")
print(f"The coordinate system IS the feature engineering.")
```

Run this and notice: the *same algorithm* on the *same data* gives very different results depending on the coordinate system. That is the entire lesson of this chapter in one experiment.

---

## Exercises

### Exercise 1: Coordinate Operations

**Problem**: Given points $A = (1, 2, 3)$ and $B = (4, 6, 8)$, find:
- The midpoint between $A$ and $B$
- The point that is 1/3 of the way from $A$ to $B$
- The point that is 3/4 of the way from $A$ to $B$

```python
import numpy as np

A = np.array([1, 2, 3])
B = np.array([4, 6, 8])

# Midpoint: t = 0.5
midpoint = A + 0.5 * (B - A)  # or equivalently (A + B) / 2
print(f"Midpoint: {midpoint}")  # [2.5, 4.0, 5.5]

# 1/3 of the way from A to B: t = 1/3
t = 1/3
point_one_third = A + t * (B - A)
print(f"1/3 point: {point_one_third}")  # [2.0, 3.33, 4.67]

# 3/4 of the way from A to B: t = 3/4
t = 3/4
point_three_quarter = A + t * (B - A)
print(f"3/4 point: {point_three_quarter}")  # [3.25, 5.0, 6.75]

# Verify: all points lie on the line from A to B
# They should satisfy P = A + t*(B-A) for some t in [0, 1]
```

### Exercise 2: Normalization Showdown

**Problem**: Build a dataset where standardization helps KNN dramatically but min-max normalization does not (hint: think about outliers).

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.random.seed(42)

# Two clear clusters with a couple of extreme outliers
n = 50
cluster_0 = np.random.randn(n, 3) + np.array([0, 0, 0])
cluster_1 = np.random.randn(n, 3) + np.array([3, 3, 3])

# Add extreme outliers to cluster_0
cluster_0[0] = [100, 0, 0]   # outlier in feature 0
cluster_0[1] = [0, -80, 0]   # outlier in feature 1

X = np.vstack([cluster_0, cluster_1])
y = np.array([0]*n + [1]*n)

# Min-max: outliers compress everything else into a tiny range
X_minmax = MinMaxScaler().fit_transform(X)
knn_mm = KNeighborsClassifier(5).fit(X_minmax[:80], y[:80])
print(f"Min-max accuracy:       {knn_mm.score(X_minmax[80:], y[80:]):.0%}")

# Standardization: more robust to outliers
X_std = StandardScaler().fit_transform(X)
knn_std = KNeighborsClassifier(5).fit(X_std[:80], y[:80])
print(f"Standardization accuracy: {knn_std.score(X_std[80:], y[80:]):.0%}")

# Why? Min-max maps everything to [0,1] based on min and max,
# so outliers squash the "real" data into a tiny range.
# Standardization uses mean and std, which are less affected by outliers.
```

### Exercise 3: Centroid Calculation and the K-Means Connection

**Problem**: Given cluster points, calculate the centroid and verify it minimizes total squared distance to all points. Then manually simulate one step of K-Means.

```python
import numpy as np

# Cluster points in 2D
points = np.array([
    [1, 2],
    [2, 3],
    [3, 2],
    [2, 1],
])

# Calculate centroid (= mean point)
centroid = np.mean(points, axis=0)
print(f"Centroid: {centroid}")  # [2.0, 2.0]

# Total squared distance to centroid
def total_sq_dist(center, pts):
    return np.sum((pts - center) ** 2)

print(f"Total squared dist to centroid: {total_sq_dist(centroid, points):.2f}")

# Try any other point -- it will always be worse
for test in [np.array([2.5, 2.5]), np.array([1.0, 1.0]), np.array([3.0, 3.0])]:
    print(f"Total squared dist to {test}: {total_sq_dist(test, points):.2f}")

# Mini K-Means: one iteration
# Suppose we have two clusters and initial centroids
all_points = np.array([[1,1],[1.5,2],[2,1.5],  # cluster A (roughly)
                        [5,5],[6,4],[5.5,6]])   # cluster B (roughly)

# Initial (bad) centroids
c1, c2 = np.array([0.0, 0.0]), np.array([3.0, 3.0])
print(f"\nInitial centroids: {c1}, {c2}")

# Assign each point to nearest centroid
dists_to_c1 = np.sum((all_points - c1)**2, axis=1)
dists_to_c2 = np.sum((all_points - c2)**2, axis=1)
assignments = (dists_to_c2 < dists_to_c1).astype(int)
print(f"Assignments: {assignments}")  # 0 = cluster 1, 1 = cluster 2

# Update centroids
c1_new = all_points[assignments == 0].mean(axis=0)
c2_new = all_points[assignments == 1].mean(axis=0)
print(f"Updated centroids: {c1_new}, {c2_new}")
# Centroids moved toward the true cluster centers -- that's K-Means!
```

### Exercise 4: Polar Coordinate Conversion

**Problem**: Convert between Cartesian and polar coordinates, and use polar coordinates to generate points on a circle (useful for visualizing embeddings).

```python
import numpy as np

# Cartesian to Polar
x, y = 3.0, 4.0
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)
print(f"Cartesian ({x}, {y}) -> Polar (r={r:.2f}, theta={np.degrees(theta):.1f} deg)")

# Polar to Cartesian
r, theta = 5.0, np.radians(53.13)
x = r * np.cos(theta)
y = r * np.sin(theta)
print(f"Polar (r={r}, theta=53.13 deg) -> Cartesian ({x:.2f}, {y:.2f})")

# Generate points on a circle (useful for embedding visualization)
n_points = 8
angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
radius = 3.0
circle_x = radius * np.cos(angles)
circle_y = radius * np.sin(angles)

print(f"\n{n_points} points on a circle of radius {radius}:")
for i in range(n_points):
    print(f"  angle={np.degrees(angles[i]):6.1f} deg -> ({circle_x[i]:+.2f}, {circle_y[i]:+.2f})")
```

---

## Summary

Here is what you should take away from this chapter:

- **A coordinate system is an engineering choice**, not a mathematical given. It consists of an origin, a set of axes (basis vectors), and a scale. Different choices lead to different model behaviors on the same data.

- **Your ML dataset IS a coordinate system.** Each feature is an axis, each sample is a point. The dataset shape `(m, n)` means $m$ points in $\mathbb{R}^n$.

- **Normalization is a coordinate system fix.** Standardization ($z = (x - \mu) / \sigma$) and min-max scaling ($x' = (x - \min) / (\max - \min)$) are coordinate transformations that ensure all features contribute fairly to distance calculations.

- **PCA is a coordinate system rotation.** It finds new axes aligned with the directions of maximum variance, enabling dimensionality reduction by dropping low-variance axes.

- **Change of basis** ($\mathbf{x}' = A\mathbf{x}$) is the general mechanism. Standardization, PCA, embedding layers, and the graphics model-view-projection pipeline are all instances of the same mathematical operation.

- **High dimensions break intuition.** The curse of dimensionality makes distances converge, volumes concentrate near surfaces, and the "center" becomes empty. Do not trust your 2D/3D intuition in 100D.

- **The position vector formula** $\vec{OP} = \sum_{i=1}^n x_i \mathbf{e}_i$ says that coordinates are weights -- they tell you how much of each basis direction to combine to reach a point.

---

## What's Next

Now that you understand *where* points live (coordinate systems) and *how* to re-describe them (coordinate transformations), the natural next question is: **how far apart are two points?**

That depends on how you define "distance" -- and it turns out there are many valid definitions. Euclidean distance, Manhattan distance, cosine similarity, and Mahalanobis distance all answer the question differently, and the choice dramatically affects your ML algorithm's behavior.

In the next chapter, [Distance Metrics](./02-distance-metrics.md), we will build on the coordinate systems foundation and explore how to measure the space between points -- the mathematical backbone of KNN, clustering, anomaly detection, and recommendation systems.
