# Coordinate Systems

## Intuition

Imagine you're meeting a friend in a city. You could give directions like "walk three blocks east and two blocks north from the central station." This simple instruction contains the essence of a coordinate system: a reference point (the station) and a way to describe positions relative to it (blocks east and north).

A coordinate system is simply an agreed-upon method for describing the location of points in space using numbers. In two dimensions, we need two numbers; in three dimensions, three numbers; and in the high-dimensional spaces of machine learning, we might need hundreds or thousands of numbers.

**Real-world analogy**: Think of GPS coordinates. Every location on Earth can be described by two numbers: latitude and longitude. This is a coordinate system for the surface of a sphere. Similarly, every data point in your ML dataset can be described by its feature values—its "coordinates" in feature space.

**Why this matters for ML**: In machine learning, your data *is* geometry. A dataset with 100 features means each sample is a point in 100-dimensional space. Understanding coordinate systems is understanding how your data exists and relates in this abstract space.

## Visual Explanation

### The Cartesian Coordinate System

The most familiar coordinate system is the Cartesian system, named after René Descartes. In 2D, we have two perpendicular axes (x and y) meeting at an origin (0, 0).

```
        y
        ▲
        │
    4   │       • P(3, 4)
        │      /│
    3   │     / │
        │    /  │
    2   │   /   │
        │  /    │
    1   │ /     │
        │/      │
────────┼───────┼───────► x
   -1   0   1   2   3   4
        │
   -1   │
```

The point $P(3, 4)$ is located 3 units along the x-axis and 4 units along the y-axis.

### Extension to 3D

In three dimensions, we add a z-axis perpendicular to both x and y:

```
            z
            ▲
            │
            │    • P(2, 3, 4)
            │   /│
            │  / │
            │ /  │
            │/   │
────────────┼────┼────► y
           /│    │
          / │    │
         /  │    │
        ▼   │    │
        x   └────┘
```

A point in 3D is represented as $(x, y, z)$.

### N-Dimensional Feature Space

In machine learning, we generalize to $n$ dimensions. A point in $n$-dimensional space is represented as:

$$\mathbf{x} = (x_1, x_2, x_3, \ldots, x_n)$$

While we cannot visualize more than 3 dimensions directly, the mathematics works identically. Each feature in your dataset becomes a dimension:

```
┌─────────────────────────────────────────────────────┐
│                    Feature Space                     │
├─────────────────────────────────────────────────────┤
│  Sample 1:  (age, income, score, ..., feature_n)    │
│             ───────────────────────────────────     │
│              ↓    ↓      ↓              ↓           │
│             x₁   x₂     x₃            xₙ            │
│                                                      │
│  Each sample = one point in n-dimensional space     │
└─────────────────────────────────────────────────────┘
```

## Mathematical Foundation

### Definition: Cartesian Coordinates

A **Cartesian coordinate system** in $n$-dimensional space $\mathbb{R}^n$ consists of:

1. An **origin** $O$ at position $(0, 0, \ldots, 0)$
2. $n$ mutually **perpendicular axes** (orthogonal basis vectors)
3. A **unit of measurement** along each axis

A point $P$ in this space is uniquely identified by an ordered $n$-tuple:

$$P = (x_1, x_2, \ldots, x_n) \in \mathbb{R}^n$$

where $x_i$ represents the signed distance from the origin along the $i$-th axis.

### Key Formulas

**Position vector**: A point can also be represented as a vector from the origin:

$$\vec{OP} = x_1\mathbf{e}_1 + x_2\mathbf{e}_2 + \cdots + x_n\mathbf{e}_n = \sum_{i=1}^{n} x_i\mathbf{e}_i$$

where $\mathbf{e}_i$ are the standard basis vectors (unit vectors along each axis).

**Standard basis vectors in $\mathbb{R}^3$**:

$$\mathbf{e}_1 = (1, 0, 0), \quad \mathbf{e}_2 = (0, 1, 0), \quad \mathbf{e}_3 = (0, 0, 1)$$

**Midpoint formula**: The midpoint $M$ between points $P$ and $Q$ is:

$$M = \left(\frac{x_P + x_Q}{2}, \frac{y_P + y_Q}{2}, \ldots\right)$$

### Feature Space Interpretation

In machine learning, we interpret the coordinate system as follows:

| Mathematical Concept | ML Interpretation |
|---------------------|-------------------|
| Dimension $n$ | Number of features |
| Coordinate $x_i$ | Value of feature $i$ |
| Point $\mathbf{x}$ | One data sample |
| Set of points | Dataset |
| Origin | Zero vector (or mean-centered reference) |

## Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# Cartesian Coordinates: From 2D to N-Dimensions
# =============================================================================

# 2D Point
point_2d = np.array([3, 4])
print(f"2D Point: {point_2d}")
print(f"  x-coordinate: {point_2d[0]}")
print(f"  y-coordinate: {point_2d[1]}")

# 3D Point
point_3d = np.array([2, 3, 4])
print(f"\n3D Point: {point_3d}")

# N-Dimensional Point (feature vector)
# Example: A sample with 5 features
feature_vector = np.array([0.5, 1.2, -0.3, 2.1, 0.8])
print(f"\n5D Feature Vector: {feature_vector}")
print(f"  Number of dimensions: {feature_vector.shape[0]}")

# =============================================================================
# Working with Multiple Points (Dataset)
# =============================================================================

# A dataset is a collection of points in feature space
# Each row is a sample, each column is a feature
dataset = np.array([
    [1.0, 2.0, 3.0],   # Sample 1
    [4.0, 5.0, 6.0],   # Sample 2
    [7.0, 8.0, 9.0],   # Sample 3
    [2.0, 4.0, 6.0],   # Sample 4
])

print(f"\nDataset shape: {dataset.shape}")
print(f"  Number of samples: {dataset.shape[0]}")
print(f"  Number of features (dimensions): {dataset.shape[1]}")

# Access individual coordinates
sample_idx = 1
feature_idx = 2
print(f"\nSample {sample_idx}, Feature {feature_idx}: {dataset[sample_idx, feature_idx]}")

# =============================================================================
# Coordinate Operations
# =============================================================================

# Midpoint between two points
p1 = np.array([1, 2, 3])
p2 = np.array([5, 6, 7])
midpoint = (p1 + p2) / 2
print(f"\nMidpoint between {p1} and {p2}: {midpoint}")

# Centroid (mean point) of multiple points
centroid = np.mean(dataset, axis=0)
print(f"Centroid of dataset: {centroid}")

# =============================================================================
# Visualization: 2D and 3D Coordinate Systems
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 2D Plot
ax1 = axes[0]
points_2d = np.array([[1, 2], [3, 4], [2, 1], [4, 3]])
ax1.scatter(points_2d[:, 0], points_2d[:, 1], s=100, c='blue', label='Points')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('2D Cartesian Coordinate System')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_aspect('equal')

# Annotate points
for i, (x, y) in enumerate(points_2d):
    ax1.annotate(f'({x}, {y})', (x, y), xytext=(5, 5), textcoords='offset points')

# 3D Plot
ax2 = fig.add_subplot(122, projection='3d')
points_3d = np.array([[1, 2, 3], [3, 4, 2], [2, 1, 4], [4, 3, 1]])
ax2.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=100, c='red')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('3D Cartesian Coordinate System')

# Remove the 2D subplot we added initially for 3D
axes[1].remove()

plt.tight_layout()
plt.savefig('coordinate_systems.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Feature Space Example: Iris-like Data
# =============================================================================

# Simulating a feature space scenario
np.random.seed(42)

# Three clusters in 4D space (like Iris dataset with 4 features)
n_samples = 30
cluster1 = np.random.randn(n_samples, 4) + np.array([0, 0, 0, 0])
cluster2 = np.random.randn(n_samples, 4) + np.array([3, 3, 3, 3])
cluster3 = np.random.randn(n_samples, 4) + np.array([6, 0, 6, 0])

feature_space_data = np.vstack([cluster1, cluster2, cluster3])
print(f"\nFeature Space Data:")
print(f"  Shape: {feature_space_data.shape}")
print(f"  Each sample is a point in 4D space")
print(f"  First sample coordinates: {feature_space_data[0]}")
```

## ML Relevance

### Where Coordinate Systems Appear in ML

1. **Data Representation**: Every dataset is fundamentally a collection of points in a coordinate system. A tabular dataset with $n$ features places each sample in $\mathbb{R}^n$.

2. **Embedding Spaces**: Word embeddings (Word2Vec, GloVe) and image embeddings place semantic concepts as points in high-dimensional space. Similar concepts cluster together.

3. **Latent Spaces**: Autoencoders and VAEs learn compressed coordinate systems where the coordinates capture meaningful data variations.

4. **Parameter Spaces**: Neural network weights exist in a high-dimensional parameter space. Training is a journey through this space.

### Specific Algorithms

| Algorithm | How It Uses Coordinates |
|-----------|------------------------|
| **KNN** | Finds nearest neighbors based on coordinates |
| **K-Means** | Finds cluster centroids in coordinate space |
| **PCA** | Rotates coordinate system to align with variance |
| **t-SNE/UMAP** | Creates new coordinate system for visualization |
| **SVM** | Finds separating hyperplanes in coordinate space |

## When to Use / Ignore

### Best Practices

**Do**:
- Normalize features to similar scales (standardization or min-max scaling)
- Consider the meaning of each dimension when interpreting results
- Use dimensionality reduction when you have too many features
- Visualize in 2D/3D projections to build intuition

**Don't**:
- Mix features with vastly different scales without normalization
- Assume Euclidean intuitions always apply in high dimensions (curse of dimensionality)
- Forget that coordinate systems are a choice—different representations can reveal different patterns

### Common Pitfalls

1. **Curse of Dimensionality**: In high dimensions, points become increasingly equidistant. Distances lose meaning when $n$ is very large.

2. **Scale Sensitivity**: If one feature ranges from 0-1000 and another from 0-1, the first will dominate distance calculations.

3. **Meaningless Dimensions**: Not all features contribute equally. Feature selection and dimensionality reduction help focus on what matters.

## Exercises

### Exercise 1: Coordinate Manipulation
**Problem**: Given points $A = (1, 2, 3)$ and $B = (4, 6, 8)$, find:
- The midpoint between A and B
- The point that is 1/3 of the way from A to B

**Solution**:
```python
import numpy as np

A = np.array([1, 2, 3])
B = np.array([4, 6, 8])

# Midpoint
midpoint = (A + B) / 2
print(f"Midpoint: {midpoint}")  # [2.5, 4.0, 5.5]

# Point 1/3 of the way from A to B
# Formula: A + t*(B - A) where t = 1/3
t = 1/3
point_one_third = A + t * (B - A)
print(f"1/3 point: {point_one_third}")  # [2.0, 3.33, 4.67]
```

### Exercise 2: Feature Space Analysis
**Problem**: You have a dataset where each sample has features [height_cm, weight_kg, age_years]. Explain why normalization is important and implement it.

**Solution**:
```python
import numpy as np

# Sample data: [height_cm, weight_kg, age_years]
data = np.array([
    [170, 70, 25],
    [160, 55, 30],
    [180, 85, 22],
    [165, 60, 35],
])

# Without normalization, height (range ~20) dominates
# compared to age (range ~13) or weight (range ~30)

# Standardization (z-score normalization)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
normalized = (data - mean) / std

print("Original data:\n", data)
print("\nNormalized data:\n", normalized)
print("\nNormalized mean:", np.mean(normalized, axis=0))  # ~[0, 0, 0]
print("Normalized std:", np.std(normalized, axis=0))     # ~[1, 1, 1]
```

### Exercise 3: Centroid Calculation
**Problem**: Given cluster points, calculate the centroid (mean point) and verify it minimizes total squared distance to all points.

**Solution**:
```python
import numpy as np

# Cluster points in 2D
points = np.array([
    [1, 2],
    [2, 3],
    [3, 2],
    [2, 1],
])

# Calculate centroid
centroid = np.mean(points, axis=0)
print(f"Centroid: {centroid}")  # [2.0, 2.0]

# Calculate total squared distance to centroid
def total_squared_distance(center, points):
    return np.sum((points - center) ** 2)

# The centroid minimizes this
print(f"Total squared dist to centroid: {total_squared_distance(centroid, points)}")

# Compare with a different point
other_point = np.array([2.5, 2.5])
print(f"Total squared dist to {other_point}: {total_squared_distance(other_point, points)}")
# The centroid gives a smaller value
```

## Summary

- **Coordinate systems** provide a numerical framework for representing points in space using ordered tuples of numbers.

- **Cartesian coordinates** use perpendicular axes meeting at an origin; each point is described by its distance along each axis.

- **N-dimensional space** ($\mathbb{R}^n$) generalizes 2D and 3D concepts to any number of dimensions, which is essential for ML where datasets have many features.

- **Feature space** is the ML interpretation: each feature is a dimension, each sample is a point, and the dataset is a point cloud in this space.

- **Normalization** is crucial when working with features of different scales to ensure fair contribution from each dimension.

- Understanding coordinates is foundational for **KNN, clustering, PCA, embeddings**, and virtually all geometric ML algorithms.

- Be aware of the **curse of dimensionality**: high-dimensional spaces behave counterintuitively, and distance-based methods may struggle.
