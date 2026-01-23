# Distance Metrics

## Intuition

How do you measure "closeness" between two things? The answer depends on context. If you're walking through a city with a grid layout, you might count blocks (you can't walk diagonally through buildings). If you're a bird flying between two points, you'd measure the straight-line distance. If you're comparing documents, you might care about the angle between their word frequency vectors rather than their magnitudes.

In machine learning, **distance metrics** are the mathematical tools we use to quantify similarity and dissimilarity between data points. The choice of metric profoundly affects algorithm behavior—the same dataset can yield completely different clusters or classifications depending on how we measure "distance."

**Real-world analogy**: Consider three ways to measure the distance between your home and office:
1. **As the crow flies** (Euclidean): The straight-line distance through the air
2. **By city blocks** (Manhattan): Walking only along streets, no diagonal shortcuts
3. **By travel time**: A completely different notion that depends on traffic, not geometry

Each captures something different about "how far apart" two locations are.

**Why this matters for ML**: Distance metrics are the foundation of:
- K-Nearest Neighbors (finding similar samples)
- Clustering algorithms (grouping similar points)
- Anomaly detection (finding points far from normal)
- Recommendation systems (finding similar users or items)

## Visual Explanation

### Euclidean vs Manhattan Distance

Consider two points in 2D: $A = (0, 0)$ and $B = (3, 4)$.

```
        y
        ▲
    4   │           ● B(3,4)
        │         ╱ │
    3   │       ╱   │
        │     ╱     │  Manhattan: |3-0| + |4-0| = 7
    2   │   ╱       │  (go right 3, then up 4)
        │ ╱         │
    1   │╱          │
        │           │
────────●───────────┴───► x
        A(0,0)  1  2  3  4

        Euclidean: √(3² + 4²) = 5
        (straight line)
```

The Euclidean distance (5) is always less than or equal to the Manhattan distance (7) because the straight line is the shortest path.

### Cosine Similarity: Direction vs Magnitude

Cosine similarity measures the angle between vectors, ignoring their lengths:

```
        y
        ▲
        │     B
        │    /
        │   /  θ
        │  /___
        │ /    A
        │/
────────●─────────────► x

Cosine similarity = cos(θ)
- θ = 0°  → cos(θ) = 1  (identical direction)
- θ = 90° → cos(θ) = 0  (perpendicular)
- θ = 180°→ cos(θ) = -1 (opposite direction)
```

### The Minkowski Family

The Minkowski distance generalizes Euclidean and Manhattan:

```
┌────────────────────────────────────────────────────┐
│              Minkowski Distance Family              │
├────────────────────────────────────────────────────┤
│                                                    │
│   p = 1:  Manhattan (L1)    ──  Diamond shape     │
│   p = 2:  Euclidean (L2)    ──  Circle shape      │
│   p → ∞: Chebyshev (L∞)     ──  Square shape      │
│                                                    │
│      Unit "circles" for different p values:        │
│                                                    │
│        p=1         p=2         p=∞                │
│         ◇           ○           □                 │
│                                                    │
└────────────────────────────────────────────────────┘
```

## Mathematical Foundation

### Euclidean Distance (L2 Norm)

The **Euclidean distance** is the straight-line distance between two points, derived from the Pythagorean theorem.

**Definition**: For points $\mathbf{p} = (p_1, p_2, \ldots, p_n)$ and $\mathbf{q} = (q_1, q_2, \ldots, q_n)$ in $\mathbb{R}^n$:

$$d_{\text{Euclidean}}(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2} = \|\mathbf{p} - \mathbf{q}\|_2$$

**Properties**:
- Always non-negative: $d(\mathbf{p}, \mathbf{q}) \geq 0$
- Identity: $d(\mathbf{p}, \mathbf{q}) = 0 \iff \mathbf{p} = \mathbf{q}$
- Symmetry: $d(\mathbf{p}, \mathbf{q}) = d(\mathbf{q}, \mathbf{p})$
- Triangle inequality: $d(\mathbf{p}, \mathbf{r}) \leq d(\mathbf{p}, \mathbf{q}) + d(\mathbf{q}, \mathbf{r})$

### Manhattan Distance (L1 Norm)

The **Manhattan distance** (or taxicab distance) measures distance along axes only.

**Definition**:

$$d_{\text{Manhattan}}(\mathbf{p}, \mathbf{q}) = \sum_{i=1}^{n} |p_i - q_i| = \|\mathbf{p} - \mathbf{q}\|_1$$

**When to use**: When features represent fundamentally different quantities that shouldn't be combined quadratically, or when you want robustness to outliers in individual dimensions.

### Minkowski Distance (Lp Norm)

The **Minkowski distance** is a generalization that includes both Euclidean and Manhattan as special cases.

**Definition**: For $p \geq 1$:

$$d_{\text{Minkowski}}(\mathbf{p}, \mathbf{q}) = \left( \sum_{i=1}^{n} |p_i - q_i|^p \right)^{1/p} = \|\mathbf{p} - \mathbf{q}\|_p$$

**Special cases**:
- $p = 1$: Manhattan distance
- $p = 2$: Euclidean distance
- $p \to \infty$: Chebyshev distance (maximum absolute difference): $d_\infty = \max_i |p_i - q_i|$

### Cosine Distance and Similarity

**Cosine similarity** measures the cosine of the angle between two vectors:

$$\text{cosine\_similarity}(\mathbf{p}, \mathbf{q}) = \frac{\mathbf{p} \cdot \mathbf{q}}{\|\mathbf{p}\| \|\mathbf{q}\|} = \frac{\sum_{i=1}^{n} p_i q_i}{\sqrt{\sum_{i=1}^{n} p_i^2} \cdot \sqrt{\sum_{i=1}^{n} q_i^2}}$$

**Cosine distance** converts similarity to a distance:

$$d_{\text{cosine}}(\mathbf{p}, \mathbf{q}) = 1 - \text{cosine\_similarity}(\mathbf{p}, \mathbf{q})$$

**Range**: Cosine similarity is in $[-1, 1]$; cosine distance is in $[0, 2]$ (or $[0, 1]$ for non-negative vectors).

**Key insight**: Cosine similarity is invariant to vector magnitude. Vectors $[1, 2, 3]$ and $[2, 4, 6]$ have cosine similarity of 1 (identical direction) despite different lengths.

## Code Example

```python
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

# =============================================================================
# Distance Metrics: Implementation and Comparison
# =============================================================================

# Define two points
p = np.array([1, 2, 3, 4, 5])
q = np.array([2, 3, 5, 7, 8])

print("Points:")
print(f"  p = {p}")
print(f"  q = {q}")
print()

# =============================================================================
# 1. Euclidean Distance (L2)
# =============================================================================

# Manual calculation
euclidean_manual = np.sqrt(np.sum((p - q) ** 2))

# Using scipy
euclidean_scipy = distance.euclidean(p, q)

# Using numpy
euclidean_numpy = np.linalg.norm(p - q)

print("Euclidean Distance (L2):")
print(f"  Manual: {euclidean_manual:.4f}")
print(f"  SciPy:  {euclidean_scipy:.4f}")
print(f"  NumPy:  {euclidean_numpy:.4f}")
print()

# =============================================================================
# 2. Manhattan Distance (L1)
# =============================================================================

# Manual calculation
manhattan_manual = np.sum(np.abs(p - q))

# Using scipy
manhattan_scipy = distance.cityblock(p, q)

# Using numpy with ord=1
manhattan_numpy = np.linalg.norm(p - q, ord=1)

print("Manhattan Distance (L1):")
print(f"  Manual: {manhattan_manual:.4f}")
print(f"  SciPy:  {manhattan_scipy:.4f}")
print(f"  NumPy:  {manhattan_numpy:.4f}")
print()

# =============================================================================
# 3. Minkowski Distance (Lp)
# =============================================================================

def minkowski_distance(p, q, p_norm):
    """Calculate Minkowski distance with parameter p_norm."""
    return np.sum(np.abs(p - q) ** p_norm) ** (1 / p_norm)

print("Minkowski Distance for various p:")
for p_val in [1, 2, 3, 4, 10]:
    mink_manual = minkowski_distance(p, q, p_val)
    mink_scipy = distance.minkowski(p, q, p_val)
    print(f"  p={p_val}: {mink_scipy:.4f}")

# Chebyshev (p -> infinity)
chebyshev = distance.chebyshev(p, q)
print(f"  p=inf (Chebyshev): {chebyshev:.4f}")
print()

# =============================================================================
# 4. Cosine Distance and Similarity
# =============================================================================

# Manual calculation
dot_product = np.dot(p, q)
norm_p = np.linalg.norm(p)
norm_q = np.linalg.norm(q)
cosine_sim_manual = dot_product / (norm_p * norm_q)
cosine_dist_manual = 1 - cosine_sim_manual

# Using scipy
cosine_dist_scipy = distance.cosine(p, q)
cosine_sim_scipy = 1 - cosine_dist_scipy

print("Cosine Similarity and Distance:")
print(f"  Similarity (manual): {cosine_sim_manual:.4f}")
print(f"  Similarity (scipy):  {cosine_sim_scipy:.4f}")
print(f"  Distance (manual):   {cosine_dist_manual:.4f}")
print(f"  Distance (scipy):    {cosine_dist_scipy:.4f}")
print()

# Demonstrate magnitude invariance
p_scaled = p * 10  # Scale p by 10
cosine_dist_scaled = distance.cosine(p_scaled, q)
print(f"  Distance with p*10:  {cosine_dist_scaled:.4f} (same as original!)")
print()

# =============================================================================
# Visualization: Comparing Distance Metrics in 2D
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Point for comparison
origin = np.array([0, 0])

# Generate points at equal "distance" from origin for each metric
theta = np.linspace(0, 2 * np.pi, 100)

# L1 (Manhattan) unit circle - diamond shape
ax1 = axes[0]
t = np.linspace(0, 1, 25)
l1_points = []
# Four edges of the diamond
for start, end in [([1, 0], [0, 1]), ([0, 1], [-1, 0]),
                   ([-1, 0], [0, -1]), ([0, -1], [1, 0])]:
    for i in t:
        l1_points.append([(1-i)*start[0] + i*end[0],
                          (1-i)*start[1] + i*end[1]])
l1_points = np.array(l1_points)
ax1.plot(l1_points[:, 0], l1_points[:, 1], 'b-', linewidth=2)
ax1.fill(l1_points[:, 0], l1_points[:, 1], alpha=0.3)
ax1.set_title('Manhattan (L1) Unit Circle\n|x| + |y| = 1')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# L2 (Euclidean) unit circle - actual circle
ax2 = axes[1]
l2_x = np.cos(theta)
l2_y = np.sin(theta)
ax2.plot(l2_x, l2_y, 'g-', linewidth=2)
ax2.fill(l2_x, l2_y, alpha=0.3, color='green')
ax2.set_title('Euclidean (L2) Unit Circle\n$\\sqrt{x^2 + y^2} = 1$')
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

# L-infinity (Chebyshev) unit circle - square
ax3 = axes[2]
linf_points = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
ax3.plot(linf_points[:, 0], linf_points[:, 1], 'r-', linewidth=2)
ax3.fill(linf_points[:, 0], linf_points[:, 1], alpha=0.3, color='red')
ax3.set_title('Chebyshev (L$\\infty$) Unit Circle\nmax(|x|, |y|) = 1')
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('distance_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Practical Example: Document Similarity with Cosine
# =============================================================================

print("=" * 60)
print("Document Similarity Example (using word counts)")
print("=" * 60)

# Simple word count vectors for 3 documents
# Words: [the, cat, sat, on, mat, dog, ran, fast]
doc1 = np.array([2, 1, 1, 1, 1, 0, 0, 0])  # "The cat sat on the mat"
doc2 = np.array([1, 1, 0, 0, 0, 1, 1, 1])  # "The dog ran fast"
doc3 = np.array([4, 2, 2, 2, 2, 0, 0, 0])  # "The cat sat on the mat" x2

print("\nDocument vectors (word counts):")
print(f"Doc1 (cat story):     {doc1}")
print(f"Doc2 (dog story):     {doc2}")
print(f"Doc3 (cat story x2):  {doc3}")

print("\nEuclidean distances:")
print(f"  Doc1 vs Doc2: {distance.euclidean(doc1, doc2):.4f}")
print(f"  Doc1 vs Doc3: {distance.euclidean(doc1, doc3):.4f}")
print("  (Doc1 and Doc3 are far apart due to magnitude!)")

print("\nCosine similarities:")
print(f"  Doc1 vs Doc2: {1 - distance.cosine(doc1, doc2):.4f}")
print(f"  Doc1 vs Doc3: {1 - distance.cosine(doc1, doc3):.4f}")
print("  (Doc1 and Doc3 have similarity 1.0 - same content, different length)")
```

## ML Relevance

### Where Distance Metrics Appear

| Algorithm | Distance Metric Usage |
|-----------|----------------------|
| **K-Nearest Neighbors** | Finds k closest points; choice of metric affects neighbors |
| **K-Means Clustering** | Assigns points to nearest centroid (typically Euclidean) |
| **DBSCAN** | Defines neighborhoods using epsilon-distance |
| **Hierarchical Clustering** | Builds dendrograms based on inter-cluster distances |
| **SVM (RBF kernel)** | Kernel based on Euclidean distance |
| **Anomaly Detection** | Outliers are points far from normal data |
| **Recommendation Systems** | Find similar users/items using cosine similarity |
| **NLP Embeddings** | Word/document similarity via cosine distance |

### Choosing the Right Metric

```
┌─────────────────────────────────────────────────────────────┐
│                 Distance Metric Selection Guide             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Use EUCLIDEAN when:                                        │
│    • Features are continuous and comparable                 │
│    • Spatial/geometric interpretation makes sense           │
│    • Data is dense (not sparse)                            │
│                                                             │
│  Use MANHATTAN when:                                        │
│    • Features are on different scales/units                 │
│    • You want robustness to outliers                        │
│    • Grid-like movement is natural (e.g., city blocks)     │
│                                                             │
│  Use COSINE when:                                           │
│    • Direction matters more than magnitude                  │
│    • Data is sparse (text, recommendations)                │
│    • Comparing documents or embeddings                      │
│    • Features are counts or frequencies                     │
│                                                             │
│  Use MINKOWSKI (p > 2) when:                               │
│    • You want to penalize large differences more           │
│    • Outlier dimensions should dominate                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## When to Use / Ignore

### Euclidean Distance

**Use when**:
- Features are normalized/standardized
- Physical or spatial interpretation is meaningful
- Data lives in dense, continuous space

**Avoid when**:
- Features have very different scales (without normalization)
- Data is high-dimensional (curse of dimensionality)
- Data is sparse (many zeros)

### Manhattan Distance

**Use when**:
- Features represent different physical quantities
- You want less sensitivity to large differences in single dimensions
- Working with discrete or grid-based data

**Avoid when**:
- You need rotation invariance
- Diagonal movements are natural in your domain

### Cosine Distance

**Use when**:
- Comparing text documents (TF-IDF, embeddings)
- Magnitude should be ignored (e.g., document length)
- Data is sparse and high-dimensional
- Working with embeddings (word2vec, BERT)

**Avoid when**:
- Magnitude carries important information
- Data can have negative values with meaningful interpretation
- Zero vectors are possible (undefined cosine similarity)

### Common Pitfalls

1. **Forgetting to normalize**: Euclidean distance is dominated by large-scale features. Always standardize first.

2. **Using Euclidean for text**: Sparse, high-dimensional text data works poorly with Euclidean; use cosine instead.

3. **Ignoring the curse of dimensionality**: In very high dimensions, all points become roughly equidistant. Consider dimensionality reduction.

4. **Zero vectors with cosine**: Cosine similarity is undefined for zero vectors. Handle this edge case explicitly.

## Exercises

### Exercise 1: Computing Distances

**Problem**: Given points $A = (1, 2)$ and $B = (4, 6)$, calculate:
- Euclidean distance
- Manhattan distance
- Chebyshev distance

**Solution**:
```python
import numpy as np
from scipy.spatial import distance

A = np.array([1, 2])
B = np.array([4, 6])

# Euclidean: sqrt((4-1)^2 + (6-2)^2) = sqrt(9 + 16) = sqrt(25) = 5
euclidean = distance.euclidean(A, B)
print(f"Euclidean: {euclidean}")  # 5.0

# Manhattan: |4-1| + |6-2| = 3 + 4 = 7
manhattan = distance.cityblock(A, B)
print(f"Manhattan: {manhattan}")  # 7

# Chebyshev: max(|4-1|, |6-2|) = max(3, 4) = 4
chebyshev = distance.chebyshev(A, B)
print(f"Chebyshev: {chebyshev}")  # 4
```

### Exercise 2: Cosine Similarity Interpretation

**Problem**: Three document vectors are given:
- $D_1 = [3, 0, 0, 2, 0, 0]$
- $D_2 = [0, 2, 1, 0, 1, 0]$
- $D_3 = [6, 0, 0, 4, 0, 0]$

Which pairs are most similar? Explain why.

**Solution**:
```python
import numpy as np
from scipy.spatial import distance

D1 = np.array([3, 0, 0, 2, 0, 0])
D2 = np.array([0, 2, 1, 0, 1, 0])
D3 = np.array([6, 0, 0, 4, 0, 0])

# Calculate cosine similarities
sim_12 = 1 - distance.cosine(D1, D2)
sim_13 = 1 - distance.cosine(D1, D3)
sim_23 = 1 - distance.cosine(D2, D3)

print(f"Similarity D1-D2: {sim_12:.4f}")  # 0.0 (orthogonal - no shared terms)
print(f"Similarity D1-D3: {sim_13:.4f}")  # 1.0 (identical direction, D3 = 2*D1)
print(f"Similarity D2-D3: {sim_23:.4f}")  # 0.0 (orthogonal)

# D1 and D3 are most similar (cosine = 1.0)
# They point in exactly the same direction; D3 is just "longer"
# This makes sense: D3 is like D1 but with doubled word counts
```

### Exercise 3: Metric Selection

**Problem**: For each scenario, choose the most appropriate distance metric and explain why.

a) Finding similar movies based on genre tags (action: 1, comedy: 0, drama: 1, ...)
b) Grouping customers by [age, income, purchase_count]
c) Finding similar news articles based on word frequencies

**Solution**:

a) **Cosine similarity** - Genre tags are sparse binary vectors. Cosine handles sparsity well and ignores the number of genres (magnitude), focusing on which genres overlap.

b) **Euclidean with standardization** - These are continuous features on different scales. Standardize first (z-score), then use Euclidean. Manhattan is also acceptable if you want robustness to outliers in individual features.

c) **Cosine similarity** - Word frequency vectors are sparse and high-dimensional. Document length shouldn't affect similarity (a long article about sports should be similar to a short one about sports). Cosine ignores magnitude and works well with sparse data.

```python
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import numpy as np

# Scenario (b) example
customers = np.array([
    [25, 50000, 10],    # Young, medium income, few purchases
    [45, 120000, 50],   # Middle-aged, high income, many purchases
    [30, 55000, 15],    # Young-ish, medium income, some purchases
])

# Standardize first!
scaler = StandardScaler()
customers_scaled = scaler.fit_transform(customers)

# Now Euclidean distance is meaningful
dist_01 = distance.euclidean(customers_scaled[0], customers_scaled[1])
dist_02 = distance.euclidean(customers_scaled[0], customers_scaled[2])
print(f"Customer 0 vs 1: {dist_01:.4f}")
print(f"Customer 0 vs 2: {dist_02:.4f}")
# Customer 0 and 2 are more similar
```

## Summary

- **Euclidean distance** ($L_2$) is the straight-line distance; use for normalized continuous data where geometric intuition applies.

- **Manhattan distance** ($L_1$) sums absolute differences along each axis; more robust to outliers and appropriate when features are on different scales.

- **Minkowski distance** ($L_p$) generalizes both: $p=1$ gives Manhattan, $p=2$ gives Euclidean, and $p \to \infty$ gives Chebyshev (maximum difference).

- **Cosine similarity** measures the angle between vectors, ignoring magnitude; ideal for sparse, high-dimensional data like text.

- **Always normalize** features before using Euclidean distance to prevent large-scale features from dominating.

- **Match the metric to your problem**: text data typically uses cosine, spatial data uses Euclidean, and robust applications may prefer Manhattan.

- The **curse of dimensionality** makes all distances converge in high dimensions; consider dimensionality reduction for very high-dimensional data.

- Distance metrics are fundamental to **KNN, clustering, anomaly detection, and similarity search** throughout machine learning.
