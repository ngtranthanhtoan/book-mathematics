# Distance Metrics — How Your ML Model Decides "Close Enough"

## Building On

In the previous chapter on coordinate systems, you learned how to pin data points to positions in space — Cartesian grids, polar coordinates, and the high-dimensional feature spaces where your ML data actually lives. You can now *locate* any data point.

But location alone is useless without one critical question: **how far apart are two points?**

Think about it — k-Nearest Neighbors needs to know which points are "nearest." Clustering needs to know which points "belong together." Your vector database needs to decide which embeddings are "similar." Every single one of those operations boils down to measuring distance. And as you are about to see, the word "distance" is far more slippery than you think.

---

## The Problem That Starts Everything

You have just deployed an embedding-based search service. Users type a query, you encode it into a 768-dimensional vector using a transformer model, then find the most similar documents in your vector store. Standard stuff — you have done this before.

Then the bug reports start rolling in.

User searches for "Python web frameworks" and the top result is a 40-page whitepaper about Django. Relevant? Sure. But the *second* result is a two-sentence blurb: "Flask is a micro web framework." Meanwhile, a detailed, highly relevant tutorial on FastAPI is buried at position 15.

You dig into the code and find this:

```python
import numpy as np

def find_similar(query_embedding, document_embeddings):
    """Find the most similar documents to a query."""
    distances = []
    for i, doc_emb in enumerate(document_embeddings):
        dist = np.linalg.norm(query_embedding - doc_emb)  # Euclidean distance
        distances.append((i, dist))
    return sorted(distances, key=lambda x: x[1])
```

There it is. You are using **Euclidean distance** on embedding vectors. And that is the bug.

Why? Because Euclidean distance cares about *magnitude* — how "long" the vector is — not just its *direction*. The 40-page whitepaper produces a longer embedding vector (more content, more signal) than the two-sentence blurb. So even when the blurb and the tutorial point in the same direction as your query, the whitepaper's raw size makes it appear "closer" in Euclidean space.

The fix is exactly one line:

```python
from scipy.spatial import distance

def find_similar(query_embedding, document_embeddings):
    """Find the most similar documents to a query."""
    distances = []
    for i, doc_emb in enumerate(document_embeddings):
        dist = distance.cosine(query_embedding, doc_emb)  # Cosine distance!
        distances.append((i, dist))
    return sorted(distances, key=lambda x: x[1])
```

Same data, same embeddings, completely different results — because you changed how you define "close."

This is the core lesson of this chapter: **the choice of distance metric is not a footnote. It is a design decision that can make or break your ML system.** Let's explore why, and build your intuition for when to use which metric.

---

## Code-First Discovery: What Does "Distance" Even Mean?

Before we touch any formulas, let's run some experiments. Fire up a Python session and follow along.

```python
import numpy as np
from scipy.spatial import distance

# Two 2D points — simple enough
A = np.array([0, 0])
B = np.array([3, 4])

# The "obvious" distance
euclidean = np.linalg.norm(A - B)
print(f"Euclidean distance: {euclidean}")  # 5.0

# But what about this?
manhattan = np.sum(np.abs(A - B))
print(f"Manhattan distance: {manhattan}")  # 7

# Or this?
chebyshev = np.max(np.abs(A - B))
print(f"Chebyshev distance: {chebyshev}")  # 4

# Same two points. Three different "distances." All are valid.
```

Same two points. Three different answers. And every one of them is a perfectly legitimate "distance." This should make you uncomfortable — and that discomfort is exactly the right instinct. Let's explore why they differ.

### Visualizing the Three Distances

```
    y
    5 |
    4 |                           * B(3,4)
      |                        ./ |
    3 |                     ./    |
      |                  ./       |      Euclidean (L2) = 5.0
    2 |               ./          |      "As the crow flies"
      |            ./             |
    1 |         ./                |
      |      ./                   |
    0 *------+----+----+----+----+----  x
      A(0,0) 1    2    3    4    5

    Manhattan path (L1) = 7:
    Go right 3, then up 4  (or any staircase path)

    4 |                     +-----* B
      |                     |
    3 |                     |
      |                     |
    2 |                     |
      |                     |
    1 |                     |
      |                     |
    0 *-----+-----+-----+--+----  x
      A     1     2     3

    Chebyshev (L-inf) = 4:
    The single biggest step in any dimension = max(3, 4) = 4
```

Three different metrics, three different answers, because each one has a different notion of what "moving through space" means:

- **Euclidean**: You can fly in a straight line. Shortest possible path.
- **Manhattan**: You are on a grid. No diagonals allowed. Like walking city blocks.
- **Chebyshev**: You only care about the worst-case dimension. Like a chess king who moves one step in all directions at once — the number of moves is the maximum coordinate difference.

> **You Already Know This**
>
> You have used distance metrics your entire career, even if you never called them that:
>
> - **Levenshtein distance** (edit distance) between strings? That is Manhattan distance on the space of character operations.
> - **Hamming distance** between binary strings or hash values? That is L1 distance on binary vectors.
> - **Similarity search in a vector database** (Pinecone, Weaviate, pgvector)? You chose between Euclidean, cosine, and dot product when you created the index. That choice is exactly what this chapter is about.
> - **Redis geospatial queries** with `GEODIST`? Haversine distance — a specialized metric for points on a sphere.
>
> You have been picking distance metrics for years. Now you will understand *why* certain choices work better than others.

---

## The Minkowski Family: One Formula to Rule Them All

Here is the pattern you just discovered through code. Look at Euclidean, Manhattan, and Chebyshev side by side:

| Metric | Formula (2D) | What it does |
|--------|-------------|-------------|
| Manhattan | $\|x_1 - x_2\| + \|y_1 - y_2\|$ | Sum of absolute differences |
| Euclidean | $\sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$ | Square root of sum of squared differences |
| Chebyshev | $\max(\|x_1 - x_2\|, \|y_1 - y_2\|)$ | Maximum absolute difference |

Do you see the pattern? They are all doing the same thing with different "power levels." The **Minkowski distance** unifies them with a single parameter $p$.

For two points $\mathbf{a} = (a_1, a_2, \ldots, a_n)$ and $\mathbf{b} = (b_1, b_2, \ldots, b_n)$ in $\mathbb{R}^n$:

$$d_p(\mathbf{a}, \mathbf{b}) = \left( \sum_{i=1}^{n} |a_i - b_i|^p \right)^{1/p}$$

**Translation**: Take the absolute difference in each dimension, raise it to the power $p$, sum them all up, then take the $p$-th root. That is it. The parameter $p$ controls how much you penalize big differences in any single dimension.

The special cases:

- $p = 1$: **Manhattan distance** (L1 norm) — $d_1(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} |a_i - b_i|$
- $p = 2$: **Euclidean distance** (L2 norm) — $d_2(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2} = \|\mathbf{a} - \mathbf{b}\|_2$
- $p \to \infty$: **Chebyshev distance** (L-infinity norm) — $d_\infty(\mathbf{a}, \mathbf{b}) = \max_i |a_i - b_i|$

Why does $p \to \infty$ give the max? Because as $p$ grows, the largest term in the sum dominates everything else. If you raise 3 and 4 to the power 1000, the ratio $4^{1000} / 3^{1000}$ is astronomically large — the smaller terms become irrelevant.

Let's verify with code:

```python
import numpy as np
from scipy.spatial import distance

A = np.array([0, 0])
B = np.array([3, 4])

# Watch what happens as p increases
for p in [1, 2, 3, 5, 10, 50, 100, 1000]:
    d = distance.minkowski(A, B, p)
    print(f"  p={p:>4d}: {d:.6f}")

# p=   1: 7.000000    (Manhattan: 3 + 4 = 7)
# p=   2: 5.000000    (Euclidean: sqrt(9 + 16) = 5)
# p=   3: 4.497941
# p=   5: 4.146312
# p=  10: 4.021714
# p=  50: 4.000001
# p= 100: 4.000000
# p=1000: 4.000000    (Chebyshev: max(3, 4) = 4)
```

As $p$ increases, the distance converges to 4 — the Chebyshev distance. The math checks out.

### What the Unit "Circles" Look Like

One of the most revealing ways to understand these metrics is to ask: **what does "distance = 1 from the origin" look like?** In Euclidean space, that is a circle. But for other metrics, the shape is very different.

```
    L1 (Manhattan)           L2 (Euclidean)           L-inf (Chebyshev)
    |x| + |y| = 1           x^2 + y^2 = 1            max(|x|,|y|) = 1

         *                       ***                  *-----------*
        / \                    **   **                |           |
       /   \                  *       *               |           |
      /     \                *         *              |           |
     *       *               *         *              |     .     |
      \     /                *         *              |           |
       \   /                  *       *               |           |
        \ /                    **   **                |           |
         *                       ***                  *-----------*

    Diamond shape            Circle shape             Square shape
    p = 1                    p = 2                    p = infinity
```

**Translation**: Each shape shows all the points that are "exactly 1 unit away" from the center, according to that metric. The diamond, circle, and square are all "unit circles" — just with different definitions of distance.

This is not just a pretty picture. It has real consequences:

- **L1 (diamond)**: Favors axis-aligned differences. Points along the axes are "farther" than diagonal points at the same Euclidean distance. This makes L1 good at promoting **sparsity** — a property heavily exploited in L1 regularization (Lasso).
- **L2 (circle)**: Treats all directions equally. Rotation-invariant. The "default" metric for continuous data.
- **L-inf (square)**: Only the worst dimension matters. If one feature is way off, the whole distance is large regardless of the others.

---

## Euclidean Distance: The Default You Should Question

The Euclidean distance is the one you learned in high school, extended to $n$ dimensions:

$$d_2(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2} = \|\mathbf{a} - \mathbf{b}\|_2$$

**Translation**: Square the difference in each dimension (so negatives don't cancel out), add them up, and take the square root to get back to the original units. This is the Pythagorean theorem in $n$ dimensions.

It satisfies all four properties of a mathematical **metric** (these are the rules that make "distance" behave like distance):

1. **Non-negativity**: $d(\mathbf{a}, \mathbf{b}) \geq 0$ — distances are never negative
2. **Identity of indiscernibles**: $d(\mathbf{a}, \mathbf{b}) = 0 \iff \mathbf{a} = \mathbf{b}$ — zero distance means same point
3. **Symmetry**: $d(\mathbf{a}, \mathbf{b}) = d(\mathbf{b}, \mathbf{a})$ — distance from A to B equals distance from B to A
4. **Triangle inequality**: $d(\mathbf{a}, \mathbf{c}) \leq d(\mathbf{a}, \mathbf{b}) + d(\mathbf{b}, \mathbf{c})$ — the direct path is never longer than going through a third point

These properties matter because many ML algorithms *depend* on them for correctness. If your "distance" violates the triangle inequality, nearest-neighbor search structures like KD-trees and ball trees will give you wrong answers.

> **You Already Know This**
>
> The metric properties map directly to software engineering:
>
> - **Non-negativity**: A function `distance(a, b)` that returns negative values makes no sense — like a `strlen()` returning -3.
> - **Symmetry**: If the distance from server A to server B is 50ms, you expect B-to-A to also be 50ms (at least in theory).
> - **Triangle inequality**: Network routing relies on this. If A-to-C takes 100ms, but A-to-B is 30ms and B-to-C is 40ms, you route through B because 30 + 40 < 100. Without the triangle inequality, routing algorithms break.
> - **Identity**: Two objects at zero distance are the same object — like referential equality.

### The Euclidean Trap: Scale Sensitivity

Here is where most engineers get burned. Consider a k-NN classifier for predicting whether a customer will churn:

```python
import numpy as np
from scipy.spatial import distance

# Customer features: [age (years), annual_income ($), monthly_usage (hours)]
customer_A = np.array([25, 50000, 20])
customer_B = np.array([30, 51000, 18])
customer_C = np.array([26, 90000, 19])

# Euclidean distances from A
dist_AB = distance.euclidean(customer_A, customer_B)
dist_AC = distance.euclidean(customer_A, customer_C)

print(f"Distance A->B: {dist_AB:.2f}")   # ~1000.01
print(f"Distance A->C: {dist_AC:.2f}")   # ~40000.00

# Customer B (similar age, income, and usage) is "closer" than C.
# But wait — the distance is ENTIRELY dominated by income.
# Age difference of 5 years contributes ~25 to the squared sum.
# Income difference of 1000 contributes ~1,000,000 to the squared sum.
# Income dominates by a factor of 40,000x!
```

Income completely dominates the distance calculation. Age and usage might as well not exist. This is the **scale sensitivity** problem — Euclidean distance treats a $1 difference in income the same as a 1-year difference in age.

The fix: **always normalize your features** before using Euclidean distance.

```python
from sklearn.preprocessing import StandardScaler

customers = np.array([
    [25, 50000, 20],
    [30, 51000, 18],
    [26, 90000, 19],
])

scaler = StandardScaler()
customers_scaled = scaler.fit_transform(customers)

# Now each feature has mean=0 and std=1
dist_AB = distance.euclidean(customers_scaled[0], customers_scaled[1])
dist_AC = distance.euclidean(customers_scaled[0], customers_scaled[2])

print(f"Distance A->B (normalized): {dist_AB:.4f}")  # Now balanced
print(f"Distance A->C (normalized): {dist_AC:.4f}")  # All features contribute equally
```

> **Common Mistakes**
>
> **"I'll just use Euclidean distance, it's the default."**
>
> Euclidean distance is the default in scikit-learn's `KNeighborsClassifier`, in many clustering algorithms, and in most textbooks. But "default" does not mean "correct." Before using Euclidean distance, always ask:
>
> 1. Are my features on comparable scales? If not, normalize first.
> 2. Is my data sparse (many zeros)? If so, cosine is probably better.
> 3. Am I in a very high-dimensional space? If so, beware the curse of dimensionality (more on this below).
> 4. Does magnitude matter, or only direction? If only direction, use cosine.

---

## Manhattan Distance: The Robust Alternative

$$d_1(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} |a_i - b_i| = \|\mathbf{a} - \mathbf{b}\|_1$$

**Translation**: Add up the absolute differences along each dimension. No squaring, no square root. Just "how far would I walk if I could only move along the axes?"

Why would you ever use this over Euclidean? Two reasons:

**Reason 1: Robustness to outliers.** Euclidean distance *squares* differences, which amplifies large values. A single outlier dimension can dominate the whole distance. Manhattan just sums absolute differences, so outliers have proportionally less impact.

```python
import numpy as np
from scipy.spatial import distance

# Normal point and a point with one outlier dimension
normal = np.array([1, 2, 3, 4, 5])
outlier = np.array([1, 2, 3, 4, 100])  # Dimension 5 is an outlier

reference = np.array([0, 0, 0, 0, 0])

euclidean_normal = distance.euclidean(reference, normal)
euclidean_outlier = distance.euclidean(reference, outlier)

manhattan_normal = distance.cityblock(reference, normal)
manhattan_outlier = distance.cityblock(reference, outlier)

print(f"Euclidean — normal: {euclidean_normal:.2f}, outlier: {euclidean_outlier:.2f}")
print(f"  Ratio: {euclidean_outlier / euclidean_normal:.2f}x")
# Ratio: ~13.5x  (outlier is 13.5x farther!)

print(f"Manhattan — normal: {manhattan_normal:.2f}, outlier: {manhattan_outlier:.2f}")
print(f"  Ratio: {manhattan_outlier / manhattan_normal:.2f}x")
# Ratio: ~7.1x  (outlier has less impact)
```

**Reason 2: Sparsity promotion.** This one is subtle but incredibly important in ML. When used as a regularization penalty (L1 regularization / Lasso), the Manhattan distance's geometry — that diamond shape — tends to push solutions toward the axes. In practice, this means L1 regularization drives some parameters exactly to zero, effectively performing feature selection. The L2 penalty (Euclidean) shrinks all parameters toward zero but rarely makes them exactly zero.

> **You Already Know This**
>
> If you have ever done a database migration where you update fields one at a time — first name, then email, then address — that is Manhattan distance thinking. You are moving along one axis at a time, not teleporting diagonally to the new state.
>
> And if you have used Lasso regression or seen the `penalty='l1'` parameter in logistic regression, you have been using Manhattan distance as a regularizer.

---

## Cosine Distance: When Direction Is All That Matters

Now we come to the metric that saved our embedding search from the opening story. Cosine similarity does not care how "big" your vectors are — it only cares about the angle between them.

$$\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \cdot \sqrt{\sum_{i=1}^{n} b_i^2}}$$

**Translation**: Compute the dot product of the two vectors (multiply corresponding elements and sum), then divide by the product of their lengths. This normalizes away the magnitude, leaving only the angular relationship.

The cosine distance converts this similarity into a proper distance:

$$d_{\text{cosine}}(\mathbf{a}, \mathbf{b}) = 1 - \text{cosine\_similarity}(\mathbf{a}, \mathbf{b})$$

**Range**: Cosine similarity lives in $[-1, 1]$. Cosine distance lives in $[0, 2]$. For non-negative vectors (like word counts or TF-IDF), cosine similarity is in $[0, 1]$ and distance in $[0, 1]$.

```
    Cosine similarity measures the ANGLE, not the distance:

         y
         ^         B = (1, 4)
         |        /
     4   |       /
         |      /      A = (2, 3)
     3   |     /      /
         |    /      /
     2   |   /   <--/--- angle theta between A and B
         |  /    /
     1   | /  /
         |/ /    cos(theta) ~ 0.9839
    -----*----------> x         High similarity!
         |
    C = (3, -1)
         |     cos(angle A,C) ~ -0.09
         |     Low (negative) similarity

    Key insight: scaling does NOT change the angle.
    A  = (2, 3)     and   2*A = (4, 6)
    cos(A, B) = cos(2*A, B) = 0.9839
```

### Why Cosine Dominates NLP and Embeddings

Let's revisit the document similarity problem with concrete code:

```python
import numpy as np
from scipy.spatial import distance

# Word count vectors: [the, cat, sat, on, mat, dog, ran, fast]
doc1 = np.array([2, 1, 1, 1, 1, 0, 0, 0])  # "The cat sat on the mat"
doc2 = np.array([1, 1, 0, 0, 0, 1, 1, 1])  # "The cat dog ran fast"
doc3 = np.array([4, 2, 2, 2, 2, 0, 0, 0])  # "The cat sat on the mat" (doubled)

print("Euclidean distances:")
print(f"  doc1 vs doc2: {distance.euclidean(doc1, doc2):.4f}")
print(f"  doc1 vs doc3: {distance.euclidean(doc1, doc3):.4f}")
print(f"  doc2 vs doc3: {distance.euclidean(doc2, doc3):.4f}")
# doc1 vs doc2: 2.6458
# doc1 vs doc3: 2.6458  <-- SAME distance as doc1 vs doc2!
# Euclidean thinks doc3 (same content, just longer) is as different as doc2 (different topic)

print("\nCosine similarities:")
print(f"  doc1 vs doc2: {1 - distance.cosine(doc1, doc2):.4f}")
print(f"  doc1 vs doc3: {1 - distance.cosine(doc1, doc3):.4f}")
print(f"  doc2 vs doc3: {1 - distance.cosine(doc2, doc3):.4f}")
# doc1 vs doc2: 0.4082  <-- Different topics
# doc1 vs doc3: 1.0000  <-- Same content! Cosine sees right through the magnitude
# doc2 vs doc3: 0.4082  <-- Different topics
```

This is exactly why every vector database defaults to cosine similarity for text embeddings. When you call `pgvector`'s `<=>` operator or Pinecone's `metric='cosine'`, this is the math underneath.

**The magnitude invariance property** is the key insight: for any scalar $\alpha > 0$:

$$\text{cosine\_similarity}(\alpha \mathbf{a}, \mathbf{b}) = \text{cosine\_similarity}(\mathbf{a}, \mathbf{b})$$

This is because both the numerator and denominator scale linearly with $\alpha$, and the $\alpha$ cancels out.

> **Common Mistakes**
>
> **"Cosine distance is a proper metric."**
>
> Technically, cosine distance ($1 - \cos\theta$) does *not* satisfy the triangle inequality in general, so it is not a true metric in the mathematical sense. However, **angular distance** ($\arccos(\text{cosine\_similarity}) / \pi$) *is* a true metric. In practice, this rarely matters — most ML algorithms work fine with cosine distance. But if you are building a data structure that requires the triangle inequality (like a metric tree), be aware of this subtlety.
>
> **"Cosine similarity of a zero vector."**
>
> If either vector is all zeros, the denominator is zero and cosine similarity is undefined ($0/0$). This can happen in practice when a document has no matching terms or when an embedding fails. Always add a guard:
>
> ```python
> def safe_cosine_similarity(a, b):
>     norm_a = np.linalg.norm(a)
>     norm_b = np.linalg.norm(b)
>     if norm_a == 0 or norm_b == 0:
>         return 0.0  # or np.nan, depending on your needs
>     return np.dot(a, b) / (norm_a * norm_b)
> ```

---

## The Full Distance Toolkit: When to Use What

Let's put everything together. Here is a comprehensive comparison with a realistic ML scenario: **building a k-NN classifier**.

```python
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load a classic dataset
iris = load_iris()
X, y = iris.data, iris.target

# Always normalize for fair comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test different distance metrics in k-NN
metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine']

print("k-NN accuracy with different distance metrics (Iris dataset):")
print("-" * 50)

for metric in metrics:
    if metric == 'cosine':
        # scikit-learn doesn't support 'cosine' directly in KNN,
        # but we can use it via the 'metric' parameter with brute force
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric, algorithm='brute')
    else:
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)

    scores = cross_val_score(knn, X_scaled, y, cv=5)
    print(f"  {metric:>10s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# You'll typically see:
#   euclidean: ~0.9667
#   manhattan: ~0.9600
#   chebyshev: ~0.9467
#   cosine:    ~0.9600
# All close! But on other datasets, the differences can be dramatic.
```

### The Decision Framework

Rather than a static table, think of metric selection as a decision tree based on your data characteristics:

```
                        What kind of data do you have?
                                    |
                 +------------------+------------------+
                 |                  |                  |
           Dense, continuous    Sparse, high-dim    Text / Embeddings
           (tabular data)       (many zeros)        (NLP, images)
                 |                  |                  |
          Are features on     +----+----+             |
          the same scale?     |         |          Use COSINE
                 |          Counts/   Binary           |
           +-----+-----+   freqs                 (direction matters,
           |           |     |         |          magnitude doesn't)
          Yes         No   COSINE    HAMMING
           |           |
      EUCLIDEAN   Normalize,      Want robustness
           |      then EUCLIDEAN   to outlier dims?
           |           |                |
           |           |          +-----+-----+
           |           |         Yes          No
           |           |          |            |
           |           |      MANHATTAN   EUCLIDEAN
           |           |
           +-----+-----+
                 |
         Need rotation     Very high
         invariance?       dimensions (>100)?
              |                  |
         EUCLIDEAN        Consider dim
                          reduction first,
                          then COSINE
```

### ML Algorithm Cheat Sheet

Here is where each metric shows up in the wild:

| Algorithm | Typical Metric | Why |
|-----------|---------------|-----|
| **k-Nearest Neighbors** | Euclidean (default), but tune this! | Finds k closest points; metric choice changes who the "neighbors" are |
| **K-Means Clustering** | Euclidean (baked in) | Centroids are means; only makes sense with L2 |
| **DBSCAN** | Euclidean (default), any metric works | Defines epsilon-neighborhoods; metric determines cluster shape |
| **Hierarchical Clustering** | Any metric | Builds dendrograms; metric choice affects the tree structure |
| **SVM (RBF kernel)** | Euclidean | RBF kernel is $\exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$ — Euclidean by definition |
| **Vector search (Pinecone, Weaviate)** | Cosine, dot product | Embeddings encode direction, not magnitude |
| **Recommendation systems** | Cosine | User/item vectors compared by angle |
| **Anomaly detection (Isolation Forest)** | Implicit (tree-based) | Not explicitly distance-based, but related |
| **NLP (TF-IDF, word2vec, BERT)** | Cosine | Sparse/dense embeddings; magnitude is noise |

---

## The Curse of Dimensionality: When Distance Stops Working

There is a nasty surprise waiting for you in high dimensions. Let's discover it through code:

```python
import numpy as np

def distance_concentration(n_dims, n_points=1000):
    """Show how distances concentrate as dimensions increase."""
    # Generate random points in [0, 1]^n
    points = np.random.rand(n_points, n_dims)

    # Compute all pairwise Euclidean distances
    from scipy.spatial.distance import pdist
    dists = pdist(points, metric='euclidean')

    min_d = dists.min()
    max_d = dists.max()
    ratio = (max_d - min_d) / min_d  # Relative contrast

    return min_d, max_d, ratio

print("Curse of Dimensionality: Distance Concentration")
print("-" * 55)
print(f"{'Dims':>6s} | {'Min Dist':>10s} | {'Max Dist':>10s} | {'Contrast':>10s}")
print("-" * 55)

for dims in [2, 10, 50, 100, 500, 1000]:
    min_d, max_d, ratio = distance_concentration(dims)
    print(f"{dims:>6d} | {min_d:>10.4f} | {max_d:>10.4f} | {ratio:>10.4f}")

# You'll see something like:
# Dims  |   Min Dist |   Max Dist |   Contrast
#     2 |     0.0042 |     1.3528 |   320.4762
#    10 |     0.5987 |     2.5614 |     3.2778
#    50 |     2.3841 |     4.4573 |     0.8694
#   100 |     3.7136 |     5.8914 |     0.5864
#   500 |     8.8901 |    12.0134 |     0.3513
#  1000 |    12.7401 |    16.5882 |     0.3020
```

**The punchline**: As dimensions increase, the ratio between the farthest and nearest points shrinks. In 1000 dimensions, the "nearest" neighbor is almost as far as the "farthest" neighbor. When every point is roughly equidistant from every other point, the concept of "nearest neighbor" becomes meaningless.

```
    Distance distribution visualization:

    Low dimensions (2D):           High dimensions (1000D):

    |                              |
    |  *                           |
    |  **                          |       ****
    |  ***                         |      ******
    |  *****                       |     ********
    |   *******                    |    **********
    |     **********               |   ************
    |         ***********          |   ************
    |              **********      |    **********
    |                   *****      |     ********
    |                       **     |      ******
    +------------------------->    +------------------------->
      0     distance     max         0     distance     max

    Wide spread: clear                Concentrated: everything
    nearest/farthest distinction      is roughly the same distance
```

**What to do about it**:

1. **Dimensionality reduction** (PCA, t-SNE, UMAP) before computing distances
2. **Use cosine distance** — it often holds up better in high dimensions than Euclidean
3. **Use approximate nearest neighbor** algorithms (HNSW, Annoy, FAISS) that are designed for this regime
4. **Feature selection** — fewer, more meaningful features beat hundreds of noisy ones

> **You Already Know This**
>
> The curse of dimensionality is like database index degradation. An index on 2 columns works great. An index on 50 columns? The B-tree is so deep and the selectivity so poor that it is slower than a full table scan. Same principle: too many dimensions make your search structure useless.

---

## A Complete Worked Example: Building a Similarity Search Engine

Let's tie everything together with a realistic example. You are building a semantic search system for a documentation site. Users type a query, and you need to find the most relevant docs.

```python
import numpy as np
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer

# Our "document corpus"
documents = [
    "Python web framework for building REST APIs quickly",
    "Django is a high-level Python web framework",
    "Flask is a lightweight WSGI web application framework",
    "Machine learning with scikit-learn and Python",
    "Deep learning neural networks with PyTorch",
    "Natural language processing and text classification",
    "Database optimization and SQL query tuning",
    "Kubernetes container orchestration and deployment",
    "React JavaScript frontend framework components",
    "Building microservices with Go and gRPC",
]

# Step 1: Convert text to vectors using TF-IDF
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents).toarray()

print(f"Document vectors shape: {doc_vectors.shape}")
# (10, N) — 10 documents, each is a sparse vector in N-dimensional space

# Step 2: Search function with different metrics
def search(query, metric='cosine', top_k=3):
    query_vec = vectorizer.transform([query]).toarray()[0]

    results = []
    for i, doc_vec in enumerate(doc_vectors):
        if metric == 'cosine':
            dist = distance.cosine(query_vec, doc_vec)
        elif metric == 'euclidean':
            dist = distance.euclidean(query_vec, doc_vec)
        elif metric == 'manhattan':
            dist = distance.cityblock(query_vec, doc_vec)
        results.append((i, dist, documents[i]))

    results.sort(key=lambda x: x[1])
    return results[:top_k]

# Step 3: Compare metrics on the same query
query = "Python web development framework"

for metric in ['cosine', 'euclidean', 'manhattan']:
    print(f"\nTop 3 results with {metric} distance:")
    for rank, (idx, dist, doc) in enumerate(search(query, metric), 1):
        print(f"  {rank}. [{dist:.4f}] {doc}")

# Cosine will likely rank the Python web frameworks highest.
# Euclidean might be thrown off by document length differences.
# Manhattan will give yet another ranking.
# The "best" metric depends on what you mean by "similar."
```

This example demonstrates the full pipeline from the opening problem: text data, vectorization, and how metric choice changes your search results. In production, you would use pre-trained embeddings (sentence-transformers, OpenAI embeddings) and a vector database, but the principles are identical.

---

## Exercises

### Exercise 1: Computing and Comparing Distances

**Problem**: Given points $A = (1, 2)$ and $B = (4, 6)$, compute:
- Euclidean distance
- Manhattan distance
- Chebyshev distance
- Verify that Euclidean $\leq$ Manhattan (always true — can you prove why from the formulas?)

**Solution**:

```python
import numpy as np
from scipy.spatial import distance

A = np.array([1, 2])
B = np.array([4, 6])

# Euclidean: sqrt((4-1)^2 + (6-2)^2) = sqrt(9 + 16) = sqrt(25) = 5
euclidean = distance.euclidean(A, B)
print(f"Euclidean:  {euclidean}")  # 5.0

# Manhattan: |4-1| + |6-2| = 3 + 4 = 7
manhattan = distance.cityblock(A, B)
print(f"Manhattan:  {manhattan}")  # 7

# Chebyshev: max(|4-1|, |6-2|) = max(3, 4) = 4
chebyshev = distance.chebyshev(A, B)
print(f"Chebyshev:  {chebyshev}")  # 4

# Verify: Euclidean <= Manhattan
# This follows from the Cauchy-Schwarz inequality.
# Squaring both sides: sum(d_i^2) <= (sum(|d_i|))^2
# The right side expands to sum(d_i^2) + 2*sum_{i<j}(|d_i|*|d_j|)
# The cross terms are non-negative, so the inequality always holds.
print(f"\nEuclidean <= Manhattan? {euclidean <= manhattan}")  # True

# Also verify: Chebyshev <= Euclidean <= Manhattan
print(f"Chebyshev <= Euclidean <= Manhattan? "
      f"{chebyshev <= euclidean <= manhattan}")  # True
```

**Why the ordering?** For any two points in $\mathbb{R}^n$, the following always holds:

$$d_\infty(\mathbf{a}, \mathbf{b}) \leq d_2(\mathbf{a}, \mathbf{b}) \leq d_1(\mathbf{a}, \mathbf{b}) \leq n \cdot d_\infty(\mathbf{a}, \mathbf{b})$$

**Translation**: Chebyshev is always the smallest (it only takes the max), Euclidean is in the middle, and Manhattan is the largest (it sums everything). But Manhattan is never more than $n$ times the Chebyshev distance, where $n$ is the number of dimensions.

### Exercise 2: Cosine Similarity and the Magnitude Trap

**Problem**: Three document vectors are given:
- $D_1 = [3, 0, 0, 2, 0, 0]$
- $D_2 = [0, 2, 1, 0, 1, 0]$
- $D_3 = [6, 0, 0, 4, 0, 0]$

Compute all pairwise similarities using both Euclidean and cosine. Which metric correctly identifies the "same content" pair?

**Solution**:

```python
import numpy as np
from scipy.spatial import distance

D1 = np.array([3, 0, 0, 2, 0, 0])
D2 = np.array([0, 2, 1, 0, 1, 0])
D3 = np.array([6, 0, 0, 4, 0, 0])

# Euclidean distances
print("Euclidean distances:")
print(f"  D1 vs D2: {distance.euclidean(D1, D2):.4f}")  # 4.2426
print(f"  D1 vs D3: {distance.euclidean(D1, D3):.4f}")  # 3.6056
print(f"  D2 vs D3: {distance.euclidean(D2, D3):.4f}")  # 7.6811

# Cosine similarities
print("\nCosine similarities:")
sim_12 = 1 - distance.cosine(D1, D2)
sim_13 = 1 - distance.cosine(D1, D3)
sim_23 = 1 - distance.cosine(D2, D3)
print(f"  D1 vs D2: {sim_12:.4f}")  # 0.0000 (orthogonal — no shared terms)
print(f"  D1 vs D3: {sim_13:.4f}")  # 1.0000 (identical direction: D3 = 2*D1)
print(f"  D2 vs D3: {sim_23:.4f}")  # 0.0000 (orthogonal)

# D1 and D3 are the "same content" — D3 is just D1 with doubled counts.
# Cosine correctly identifies them as identical (similarity = 1.0).
# Euclidean says they're 3.6 apart — misleading!
```

The takeaway: $D_3 = 2 \cdot D_1$. They point in exactly the same direction; $D_3$ is just "longer." Cosine correctly identifies them as identical content. Euclidean treats the magnitude difference as real distance. For document comparison, cosine is the right call.

### Exercise 3: Metric Selection in Practice

**Problem**: For each scenario, choose the most appropriate distance metric and explain why.

a) Finding similar movies based on genre tags (`action: 1, comedy: 0, drama: 1, ...`)
b) Grouping customers by `[age, income, purchase_count]`
c) Finding similar news articles based on word frequencies
d) Detecting anomalous network requests based on `[packet_size, latency_ms, error_rate]`

**Solution**:

a) **Cosine similarity** (or Jaccard distance for binary vectors). Genre tags are sparse binary vectors. Cosine handles sparsity well and ignores how many genres are tagged (magnitude), focusing on which genres overlap. A movie tagged with 3 genres and a movie tagged with 6 genres can still be very similar in direction.

b) **Euclidean with standardization**. These are continuous features on wildly different scales (age in years, income in dollars, purchases in counts). Standardize first with `StandardScaler`, then use Euclidean. Manhattan is also acceptable if you want more robustness to outlier customers (e.g., one with extremely high income).

c) **Cosine similarity**. Word frequency vectors are sparse and high-dimensional. A long article about sports and a short article about sports should be similar — cosine ignores document length (magnitude) and focuses on word distribution (direction).

d) **Euclidean with normalization** (or Mahalanobis distance if you want to account for feature correlations). For anomaly detection, you typically want all dimensions to contribute, and the *magnitude* of deviation matters — a request with 10x normal packet size is suspicious regardless of direction. Normalize the features first so that latency in milliseconds does not dominate.

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

# WITHOUT normalization — income dominates
print("Without normalization:")
for i in range(3):
    for j in range(i+1, 3):
        d = distance.euclidean(customers[i], customers[j])
        print(f"  Customer {i} vs {j}: {d:.2f}")

# WITH normalization — balanced contribution
scaler = StandardScaler()
customers_scaled = scaler.fit_transform(customers)

print("\nWith normalization:")
for i in range(3):
    for j in range(i+1, 3):
        d = distance.euclidean(customers_scaled[i], customers_scaled[j])
        print(f"  Customer {i} vs {j}: {d:.4f}")
# Customer 0 and 2 are now correctly identified as most similar
```

---

## Summary

Let's recap what you have learned, tied back to the real problems you will face:

- **Euclidean distance** ($L_2$) is the straight-line distance — the Pythagorean theorem generalized to $n$ dimensions. It is the default in most ML libraries, but that does not mean it is always right. It is **scale-sensitive** (normalize first!) and struggles with sparse or high-dimensional data.

- **Manhattan distance** ($L_1$) sums absolute differences along each axis. It is more **robust to outliers** than Euclidean and promotes **sparsity** when used as a regularization penalty. Think of it as "city block" distance.

- **Minkowski distance** ($L_p$) is the family that unifies them all: $p=1$ gives Manhattan, $p=2$ gives Euclidean, and $p \to \infty$ gives **Chebyshev** (the maximum difference in any single dimension). The parameter $p$ controls how much you penalize big differences in individual dimensions.

- **Cosine similarity** measures the angle between vectors, ignoring magnitude entirely. It is the go-to metric for **text, embeddings, and any domain where "direction" matters more than "size."** Nearly every vector database uses it as the default.

- **Always normalize** your features before using Euclidean distance. A feature measured in dollars will crush a feature measured in years if you do not.

- **The curse of dimensionality** causes all distances to converge in very high-dimensional spaces, making nearest-neighbor concepts less meaningful. Combat this with dimensionality reduction, cosine distance, or approximate nearest-neighbor algorithms.

- **Match the metric to your data**: there is no universally "best" distance metric. The right choice depends on your data characteristics (dense vs. sparse, scale, dimensionality) and what "similarity" means in your domain.

---

## What's Next

Now that you can measure how far apart points are, a natural question arises: what happens as one point *approaches* another? If you shrink the distance between two points toward zero, what happens to the function values at those points? That question — "what value does this approach as we get infinitely close?" — is the concept of a **limit**, and it is the gateway to calculus.

In the next chapter on **Limits**, you will see how distance metrics define the "neighborhoods" that make limits precise. The epsilon in an epsilon-delta proof? That is literally a distance. Everything you have learned here about measuring closeness becomes the foundation for continuity, derivatives, and ultimately gradient descent — the algorithm that trains every neural network you will ever build.
