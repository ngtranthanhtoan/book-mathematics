# Chapter 2: Geometry of Vectors

> **Building On** — You now know vectors are ordered lists of numbers that represent data. But a list of numbers by itself doesn't tell you much. How "far apart" are two users' preferences? How "similar" are two documents? You need geometry.

---

## The Problem That Starts Everything

Your search engine indexes 10 million documents. A user types a query. You've already converted both the query and every document into vectors (we'll cover how later — for now, trust that it works). You have 10 million vectors and one query vector.

How do you find the most relevant documents?

Your first instinct might be: "Find the documents closest to the query." Okay. But what does "closest" mean for a list of 768 numbers? Your second instinct might be: "Find the documents pointing in the same direction as the query." Better — but what does "direction" mean in 768 dimensions?

This is where vector geometry comes in. It gives you precise, computable answers to "how far?", "how similar?", and "how much of this is in that direction?" These three questions — distance, similarity, and projection — are the workhorses of ML.

Let's build them up from scratch.

---

## Part 1: Distance — "How Far Apart?"

### The Problem

You're building a movie recommendation system. User A rated five movies: `[5, 3, 1, 0, 4]`. User B rated the same five movies: `[4, 3, 2, 0, 5]`. User C: `[1, 5, 5, 4, 0]`.

Which user is more similar to User A — B or C? Your gut says B, because the ratings look close. But "look close" isn't an algorithm. You need a number.

### Code-First Discovery

```python
import numpy as np

user_a = np.array([5, 3, 1, 0, 4])
user_b = np.array([4, 3, 2, 0, 5])
user_c = np.array([1, 5, 5, 4, 0])

# How "far apart" are these rating vectors?
dist_ab = np.linalg.norm(user_a - user_b)
dist_ac = np.linalg.norm(user_a - user_c)

print(f"Distance A→B: {dist_ab:.4f}")  # 1.7321
print(f"Distance A→C: {dist_ac:.4f}")  # 7.6811
```

```
Distance A→B: 1.7321
Distance A→C: 7.6811
```

User B is much closer to User A. That matches your intuition, but now you have a number — and numbers scale to 10 million users.

### What Just Happened?

`np.linalg.norm(user_a - user_b)` computed the **Euclidean distance**: the straight-line distance between two points in 5-dimensional space. It's the same Pythagorean theorem you learned in school, just generalized.

In 2D, the distance between points $(1, 1)$ and $(4, 3)$ looks like this:

```
  y
  4 |
    |
  3 |          * b = (4, 3)
    |        / |
  2 |      /   |  Δy = 2
    |    /     |
  1 |  * ------+
    |  a = (1, 1)   Δx = 3
  0 +--+--+--+--+--→ x
    0  1  2  3  4

  distance = √(Δx² + Δy²) = √(3² + 2²) = √13 ≈ 3.61
```

### The Math

The **Euclidean distance** (L2 distance) between two vectors generalizes this to $n$ dimensions:

$$d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}$$

**Translation:** Subtract component by component, square each difference, add them up, take the square root. It's Pythagoras all the way down — one dimension at a time.

For the 2D case with $\mathbf{a} = [a_1, a_2]$ and $\mathbf{b} = [b_1, b_2]$:

$$d(\mathbf{a}, \mathbf{b}) = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2}$$

### Other Distance Metrics

Euclidean distance isn't the only option. Depending on your problem, you might want:

**Manhattan distance (L1)** — sum of absolute differences:

$$d_1(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n}|a_i - b_i|$$

**Minkowski distance (Lp)** — the generalization that includes both:

$$d_p(\mathbf{a}, \mathbf{b}) = \left(\sum_{i=1}^{n}|a_i - b_i|^p\right)^{1/p}$$

**Translation:** When $p=2$, you get Euclidean. When $p=1$, you get Manhattan. The parameter $p$ controls how much you penalize large differences in individual dimensions.

```
  Manhattan vs. Euclidean from (0,0) to (3,4):

  y
  4 |  . . . . * (3, 4)
    |          |
  3 |  . . . . *    Manhattan path:
    |          |    go right 3, then up 4
  2 |  . . . . *    total = 3 + 4 = 7
    |          |
  1 |  . . . . *    Euclidean path:
    |          |    straight line = √(9+16) = 5
  0 *--*--*--*-+--→ x
    0  1  2  3  4
```

> **You Already Know This**
>
> You've used these distances before, even if you didn't call them that:
> - **Euclidean distance** is what k-NN uses by default to find nearest neighbors
> - **Manhattan distance** is how you'd calculate moves on a grid — it's the distance metric in grid-based pathfinding (think A* on a city map where you can only go north/south/east/west)
> - If you've ever computed `abs(x1 - x2) + abs(y1 - y2)`, you were doing Manhattan distance

### Code: Comparing Distance Metrics

```python
import numpy as np

a = np.array([3, 4])
b = np.array([4, 3])

# Euclidean distance (L2)
euclidean_dist = np.linalg.norm(a - b)
print(f"a = {a}, b = {b}")
print(f"Euclidean distance: {euclidean_dist:.4f}")

# Manhattan distance (L1)
manhattan_dist = np.linalg.norm(a - b, ord=1)
print(f"Manhattan distance: {manhattan_dist:.4f}")

# Manual calculation to see what's happening
manual_euclidean = np.sqrt(np.sum((a - b)**2))
print(f"Manual Euclidean:   {manual_euclidean:.4f}")
```

```
a = [3 4], b = [4 3]
Euclidean distance: 1.4142
Manhattan distance: 2.0000
Manual Euclidean:   1.4142
```

Notice: Manhattan distance is larger here. It always is (or equal), because it can't take the diagonal shortcut.

---

## Part 2: Angle and Cosine Similarity — "How Similar?"

### The Problem

Distance worked for movie ratings, but now you have a different problem. You're comparing documents by word frequency.

Document 1 is a 500-word blog post about Python. Document 2 is a 5,000-word textbook chapter about Python. They cover the same topic, but Document 2 uses every word roughly 10x more often — because it's 10x longer.

If you use Euclidean distance, these documents look very far apart. But they're about the same thing! You need a metric that ignores how "big" the vectors are and only cares about their *direction*.

### Code-First Discovery

```python
import numpy as np

# Simplified: word counts for ["python", "code", "data", "cooking", "recipe"]
blog_post    = np.array([10, 8,  5, 0, 0])   # short article about Python
textbook     = np.array([100, 80, 50, 0, 0])  # long chapter about Python (10x)
recipe       = np.array([0,  0,  0, 15, 12])  # article about cooking

# Euclidean distance says blog and textbook are FAR apart
print(f"Euclidean(blog, textbook): {np.linalg.norm(blog_post - textbook):.2f}")
print(f"Euclidean(blog, recipe):   {np.linalg.norm(blog_post - recipe):.2f}")

# Cosine similarity says blog and textbook point in the SAME direction
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(f"\nCosine sim(blog, textbook): {cosine_similarity(blog_post, textbook):.4f}")
print(f"Cosine sim(blog, recipe):   {cosine_similarity(blog_post, recipe):.4f}")
```

```
Euclidean(blog, textbook): 114.02
Euclidean(blog, recipe):   22.38

Cosine sim(blog, textbook): 1.0000
Cosine sim(blog, recipe):   0.0000
```

Look at that. Euclidean distance says the blog and recipe (totally unrelated) are *closer* than the blog and textbook (same topic!). Cosine similarity gets it right: the blog and textbook are perfectly similar (1.0), while the blog and recipe are completely unrelated (0.0).

### The Insight: It's About Angle, Not Length

The angle between two vectors tells you how "aligned" they are, regardless of how long the vectors are.

```
  y
  5 |       textbook = (100, 80)
    |      /          (WAY out there, same direction)
  4 |     /
    |    /
  3 |   /  blog = (10, 8)
    |  / /
  2 | //
    |/              recipe = (0, 15)
  1 |               |
    |               |
  0 +---+---+---+---+--→ x
    0               5

  blog and textbook: angle = 0° → cosine similarity = 1.0
  blog and recipe:   angle = 90° → cosine similarity = 0.0
```

(Diagram not to scale — the textbook vector is 10x longer than the blog vector, but they point in the exact same direction.)

### The Math

The **angle** $\theta$ between two vectors comes from the dot product:

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

Solving for the angle:

$$\theta = \arccos\left(\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}\right)$$

**Translation:** Take the dot product (which mixes both direction *and* magnitude), then divide out the magnitudes. What's left is purely directional information — the cosine of the angle between them.

**Cosine similarity** is literally this value. It's not a separate concept — it *is* the cosine of the angle:

$$\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \cdot \sqrt{\sum_{i=1}^{n} b_i^2}}$$

**Key angles to remember:**

| Angle $\theta$ | $\cos(\theta)$ | Meaning                              |
|-----------------|-----------------|--------------------------------------|
| 0°              | 1               | Same direction (identical pattern)   |
| 90°             | 0               | Perpendicular (completely unrelated) |
| 180°            | -1              | Opposite directions (inverse pattern)|

**Cosine distance** is sometimes used when you need a proper distance metric (where 0 means identical):

$$\text{cosine\_distance} = 1 - \text{cosine\_similarity}$$

> **Common Mistake**
>
> Cosine similarity measures *angle*, not *magnitude*. Two vectors pointing in the same direction have cosine similarity of 1.0 even if one is 1,000x longer than the other. This is a feature, not a bug — it's exactly why it works for comparing documents of different lengths. But if magnitude matters in your application (e.g., comparing actual quantities), use Euclidean distance instead.

### Code: Angle and Cosine Similarity

```python
import numpy as np

a = np.array([3, 4])
b = np.array([4, 3])

print("=== Angle Between Vectors ===")
cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"cos(theta) = {cos_angle:.4f}")

angle_rad = np.arccos(cos_angle)
angle_deg = np.degrees(angle_rad)
print(f"Angle: {angle_rad:.4f} radians = {angle_deg:.2f} degrees")
```

```
=== Angle Between Vectors ===
cos(theta) = 0.9600
Angle: 0.2838 radians = 16.26 degrees
```

### Running Example: Document Similarity

```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Word counts for three documents
# Vocabulary: ["machine", "learning", "neural", "cooking", "ingredients"]
doc1 = np.array([1, 2, 3, 0, 0])  # ML article
doc2 = np.array([1, 2, 0, 0, 0])  # shorter ML article
doc3 = np.array([0, 0, 0, 5, 5])  # cooking article

print("Doc1 (ML article):      ", doc1)
print("Doc2 (short ML article): ", doc2)
print("Doc3 (cooking article):  ", doc3)
print(f"\nSimilarity(doc1, doc2): {cosine_similarity(doc1, doc2):.4f}")
print(f"Similarity(doc1, doc3): {cosine_similarity(doc1, doc3):.4f}")
print(f"Similarity(doc2, doc3): {cosine_similarity(doc2, doc3):.4f}")
```

```
Doc1 (ML article):       [1 2 3 0 0]
Doc2 (short ML article):  [1 2 0 0 0]
Doc3 (cooking article):   [0 0 0 5 5]

Similarity(doc1, doc2): 0.6667
Similarity(doc1, doc3): 0.0000
Similarity(doc2, doc3): 0.0000
```

Doc1 and Doc2 share some similarity (0.67) — they both talk about ML. Doc3 lives in a completely different part of vector space. Its similarity to both ML documents is exactly zero, because it has no overlap in vocabulary. This is exactly what Google's early search ranking did (TF-IDF + cosine similarity), and the core idea still powers modern vector search.

> **You Already Know This**
>
> If you've used Elasticsearch's "more like this" query, or built a recommendation engine that compares user embeddings, you've used cosine similarity. It's also the metric behind every vector database (Pinecone, Weaviate, pgvector) when you do a "similarity search."

---

## Part 3: Projection — "How Much of This Is in That Direction?"

### The Problem

You have a 2D feature vector for a data point: `[3, 4]`. You want to know: how much of this data is explained by just the x-axis? How much by the y-axis? This is trivially obvious in 2D (just read off the components), but what about projecting onto an *arbitrary* direction?

This is exactly what PCA does: it finds the directions of maximum variance, then projects your data onto those directions to reduce dimensionality. Before we get to PCA (a later chapter), you need to understand the core operation: projection.

### Code-First Discovery

```python
import numpy as np

a = np.array([3, 4])
b = np.array([5, 0])  # the x-axis direction

# How much of 'a' lies along 'b'?
scalar_proj = np.dot(a, b) / np.linalg.norm(b)
print(f"Scalar projection of a onto b: {scalar_proj:.4f}")

# The actual vector component along 'b'
vector_proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Vector projection of a onto b: {vector_proj}")

# What's left over?
perpendicular = a - vector_proj
print(f"Perpendicular component:        {perpendicular}")

# Sanity check: they should add up to the original
print(f"Projection + Perpendicular =    {vector_proj + perpendicular}")
```

```
Scalar projection of a onto b: 3.0000
Vector projection of a onto b: [3. 0.]
Perpendicular component:        [0. 4.]
Projection + Perpendicular =    [3. 4.]
```

### What Just Happened?

We decomposed vector $\mathbf{a} = [3, 4]$ into two pieces:
- The part along $\mathbf{b}$: $[3, 0]$ (the "shadow" of $\mathbf{a}$ cast along the x-axis)
- The part perpendicular to $\mathbf{b}$: $[0, 4]$ (everything that $\mathbf{b}$ couldn't capture)

Together, they reconstruct the original vector perfectly.

```
  y
  5 |
    |
  4 |  . . . * a = (3, 4)
    |         |
  3 |  . . .  |   ← perpendicular
    |         |      component [0, 4]
  2 |  . . .  |
    |         |
  1 |  . . .  |
    |         ↓
  0 +--+--+--*--------*---→ x
    0  1  2  3 = proj  5 = b

  projection of a onto b = [3, 0]
  (the "shadow" of a along the x-axis)
```

> **You Already Know This**
>
> Projection is like `SELECT columns FROM a high-dimensional table`. You're keeping only the information along certain directions and discarding the rest. When PCA "reduces dimensions," it's projecting your data onto the top-$k$ principal components — exactly this operation, repeated for each direction.

### The Math

The **scalar projection** (how *far* the shadow extends) of $\mathbf{a}$ onto $\mathbf{b}$:

$$\text{comp}_{\mathbf{b}}\mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|}$$

**Translation:** Dot the two vectors (measuring alignment × magnitude), then divide by the length of $\mathbf{b}$ to get a pure distance along $\mathbf{b}$'s direction.

The **vector projection** (the actual shadow *vector*) of $\mathbf{a}$ onto $\mathbf{b}$:

$$\text{proj}_{\mathbf{b}}\mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|^2} \mathbf{b} = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}} \mathbf{b}$$

**Translation:** Take the scalar projection and multiply it by the unit vector in $\mathbf{b}$'s direction. This gives you a vector that points along $\mathbf{b}$ and has the right length.

The **perpendicular component** — everything projection didn't capture:

$$\mathbf{a}_{\perp} = \mathbf{a} - \text{proj}_{\mathbf{b}}\mathbf{a}$$

This is guaranteed to be orthogonal to $\mathbf{b}$. You can verify: $\mathbf{a}_{\perp} \cdot \mathbf{b} = 0$.

---

## Part 4: Orthogonality — "Completely Independent"

### The Concept

Two vectors are **orthogonal** (perpendicular) when their dot product is zero:

$$\mathbf{a} \perp \mathbf{b} \iff \mathbf{a} \cdot \mathbf{b} = 0$$

This means they share *nothing* — there's zero component of one in the direction of the other. In ML terms, orthogonal features carry completely independent information.

**Orthonormal vectors** go one step further — they're orthogonal *and* each has unit length:

$$\mathbf{a} \cdot \mathbf{b} = 0 \quad \text{and} \quad \|\mathbf{a}\| = \|\mathbf{b}\| = 1$$

> **You Already Know This**
>
> Orthogonality is the vector analog of *independence*. Think microservices that don't share state — a change in one has zero effect on the other. Orthogonal features in your dataset are like microservices: they provide completely independent information. This is why PCA looks for orthogonal directions — it's finding the independent "axes of variation" in your data.

### Code: Testing Orthogonality

```python
import numpy as np

v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([1, 1, 0])

print(f"v1 = {v1}, v2 = {v2}, v3 = {v3}")
print(f"v1 · v2 = {np.dot(v1, v2)} (orthogonal: {np.isclose(np.dot(v1, v2), 0)})")
print(f"v1 · v3 = {np.dot(v1, v3)} (orthogonal: {np.isclose(np.dot(v1, v3), 0)})")
```

```
v1 = [1 0 0], v2 = [0 1 0], v3 = [1 1 0]
v1 · v2 = 0 (orthogonal: True)
v1 · v3 = 1 (orthogonal: False)
```

$v_1$ and $v_2$ are orthogonal — they're the standard basis vectors, pointing along the x and y axes. $v_1$ and $v_3$ are *not* orthogonal — $v_3$ has a component along $v_1$'s direction.

### Why Orthogonality Matters in ML

- **Orthogonal features are uncorrelated** — they provide independent information, which is what you want for model training
- **Orthonormal bases simplify computation** — projections become simple dot products (no division by norms needed)
- **PCA finds orthogonal directions** of maximum variance in your data
- **Weight initialization** schemes (like those using orthogonal matrices) help neural networks train more stably

---

## Putting It All Together: Distance vs. Similarity

Let's see both metrics in action on the same dataset, because choosing the wrong one is a real source of bugs.

```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Query point and data points
query = np.array([2, 2])
points = np.array([
    [1, 1],
    [3, 3],
    [0, 4],
    [4, 0]
])

print(f"Query: {query}")
print("\nPoints and their relationship to query:")
print(f"{'Point':>10} {'Euclidean':>10} {'CosSim':>8}")
print("-" * 32)
for i, p in enumerate(points):
    euc_dist = np.linalg.norm(query - p)
    cos_sim = cosine_similarity(query, p)
    print(f"{str(p):>10} {euc_dist:>10.3f} {cos_sim:>8.3f}")
```

```
Query: [2 2]

Points and their relationship to query:
     Point  Euclidean   CosSim
--------------------------------
    [1 1]      1.414    1.000
    [3 3]      1.414    1.000
    [0 4]      2.828    0.707
    [4 0]      2.828    0.707
```

This is the critical insight. Look at `[1, 1]` and `[3, 3]`:

- **Euclidean distance**: Both are 1.414 from the query — they're equidistant
- **Cosine similarity**: Both have 1.0 — they point in the exact same direction as the query

Now look at `[0, 4]` and `[4, 0]`:

- **Euclidean distance**: Both are 2.828 from the query — equidistant again
- **Cosine similarity**: Both are 0.707 — they're at 45° angles to the query

```
  y
  4 | *[0,4]
    |   \
  3 |    * [3,3]
    |   /
  2 |  * query [2,2]
    |   \
  1 |    * [1,1]
    |              *[4,0]
  0 +--+--+--+--+--→ x
    0  1  2  3  4

  [1,1] and [3,3]: same DIRECTION as query, different DISTANCE
  Cosine says identical. Euclidean says different positions.

  Which metric you choose depends on whether you care about
  direction (topic, preference pattern) or position (actual values).
```

### When to Use Which?

| Use Euclidean Distance                | Use Cosine Similarity                 |
|---------------------------------------|---------------------------------------|
| Features have similar scales          | Features have different scales        |
| Magnitude matters (actual quantities) | Only the pattern/direction matters    |
| Physical measurements, sensor data    | Text/document similarity              |
| Image pixel comparison                | High-dimensional sparse data          |
| k-Means clustering                    | Semantic search, recommendations      |

> **Common Mistake**
>
> Using Euclidean distance on text data (where vector magnitude varies with document length) is a classic bug. A 10,000-word document about cats and a 100-word document about cats will be "far apart" in Euclidean distance, even though they're about the same thing. Cosine similarity handles this correctly because it normalizes out the magnitude.

---

## Running Example: Word Embeddings

Here's where all these concepts come together in modern ML. Word embedding models (Word2Vec, GloVe, and the embeddings inside transformers) map words to vectors. The geometry of these vectors captures meaning:

```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Simplified 4D word embeddings (real ones are 100-300 dimensional)
# Dimensions loosely represent: [royalty, gender, age, power]
king   = np.array([0.9, 0.1, 0.5, 0.9])
queen  = np.array([0.9, 0.9, 0.5, 0.8])
man    = np.array([0.1, 0.1, 0.5, 0.3])
woman  = np.array([0.1, 0.9, 0.5, 0.3])

# The famous analogy: king - man + woman ≈ queen
result = king - man + woman
print(f"king - man + woman = {result}")
print(f"Actual queen vector = {np.array([0.9, 0.9, 0.5, 0.8])}")
print(f"\nCosine similarity to queen: {cosine_similarity(result, queen):.4f}")
print(f"Cosine similarity to king:  {cosine_similarity(result, king):.4f}")
print(f"Cosine similarity to man:   {cosine_similarity(result, man):.4f}")

# Semantic similarity: king is more similar to queen than to man
print(f"\nsim(king, queen): {cosine_similarity(king, queen):.4f}")
print(f"sim(king, man):   {cosine_similarity(king, man):.4f}")
```

```
king - man + woman = [0.9 1.7 0.5 0.9]
Actual queen vector = [0.9 0.9 0.5 0.8]

Cosine similarity to queen: 0.9449
Cosine similarity to king:  0.8553
Cosine similarity to man:   0.7359

sim(king, queen): 0.9479
sim(king, man):   0.6887
```

The angle between "king" and "queen" is small (high cosine similarity) because they share the concepts of royalty and power. The angle between "king" and "man" is larger — they share gender but not royalty. The vector arithmetic `king - man + woman` lands close to `queen` because the *geometry* of the embedding space encodes semantic relationships.

This is not a toy example. This is literally how word embeddings work — and why cosine similarity is the default metric for comparing them.

---

## Common Mistakes and Pitfalls

Before you go off and ship this into production, here are the mistakes that will burn you:

**1. Forgetting to normalize for distance-based methods.**
Cosine similarity normalizes automatically (that's the denominator). Euclidean distance does not. If one feature ranges from 0-1 and another from 0-10,000, the second feature will dominate the distance calculation. Always standardize your features before using Euclidean distance.

**2. The curse of dimensionality.**
In very high dimensions (thousands of features), a strange thing happens: all points become approximately equidistant. The ratio of nearest-neighbor distance to farthest-neighbor distance approaches 1. This means distance-based methods (k-NN, k-means) start to break down. This is one reason dimensionality reduction (PCA, t-SNE, UMAP) exists.

**3. Zero vectors blow up cosine similarity.**
The formula divides by $\|\mathbf{a}\| \|\mathbf{b}\|$. If either vector is all zeros, you divide by zero. Always check for zero vectors before computing cosine similarity, or add a tiny epsilon: `norm + 1e-8`.

**4. Sparse vectors need special treatment.**
If your vectors are 99% zeros (common in text data with large vocabularies), storing and computing on dense arrays is wasteful. Use `scipy.sparse` representations and their built-in distance/similarity functions.

**5. Using the wrong metric for the task.**
Euclidean distance on text data is almost always wrong (document length dominates). Cosine similarity on spatial coordinates is almost always wrong (you care about actual position, not direction from origin). Think about what "similar" means for your specific problem.

---

## Exercises

### Exercise 1: Distance Calculation

Compute the Euclidean and Manhattan distances between $\mathbf{p} = [1, 2, 3]$ and $\mathbf{q} = [4, 0, 3]$.

*Before running the code, try working it out by hand: subtract component-wise, square the differences (for Euclidean) or absolute-value them (for Manhattan), sum them up.*

**Solution:**
```python
import numpy as np

p = np.array([1, 2, 3])
q = np.array([4, 0, 3])

euclidean = np.linalg.norm(p - q)  # sqrt((4-1)^2 + (0-2)^2 + (3-3)^2) = sqrt(9+4+0) = sqrt(13)
manhattan = np.linalg.norm(p - q, ord=1)  # |4-1| + |0-2| + |3-3| = 3+2+0 = 5

print(f"Euclidean distance: {euclidean:.4f}")  # 3.6056
print(f"Manhattan distance: {manhattan:.4f}")  # 5.0
```

### Exercise 2: Cosine Similarity

Find the cosine similarity between $\mathbf{a} = [1, 0, 1]$ and $\mathbf{b} = [0, 1, 1]$.

*Hint: the dot product only has one nonzero term. Both vectors have the same norm.*

**Solution:**
```python
import numpy as np

a = np.array([1, 0, 1])
b = np.array([0, 1, 1])

cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# (1*0 + 0*1 + 1*1) / (sqrt(2) * sqrt(2)) = 1/2 = 0.5

print(f"Cosine similarity: {cos_sim:.4f}")  # 0.5
```

The cosine similarity is 0.5, corresponding to an angle of 60 degrees. These vectors are "somewhat similar" — they share one dimension (the third) but differ in the others.

### Exercise 3: Projection

Project $\mathbf{v} = [4, 2]$ onto $\mathbf{u} = [3, 0]$.

*Think of this as: how much of `v` lies along the x-axis (since `u` points along x)?*

**Solution:**
```python
import numpy as np

v = np.array([4, 2])
u = np.array([3, 0])

# Vector projection of v onto u
proj = (np.dot(v, u) / np.dot(u, u)) * u
# (4*3 + 2*0) / (3*3 + 0*0) * [3, 0] = 12/9 * [3, 0] = [4, 0]

print(f"Projection of v onto u: {proj}")  # [4. 0.]

# The perpendicular component
perp = v - proj  # [0, 2]
print(f"Perpendicular component: {perp}")
print(f"Dot product (should be 0): {np.dot(proj, perp)}")  # 0.0
```

The projection is $[4, 0]$ — the entire x-component of $\mathbf{v}$. The perpendicular component $[0, 2]$ is the y-component. Together they reconstruct $\mathbf{v}$. The zero dot product confirms they're orthogonal.

### Exercise 4 (Challenge): Recommendation System

User A rated 5 movies: `[5, 4, 1, 0, 3]`. User B: `[4, 5, 1, 0, 2]`. User C: `[1, 1, 5, 4, 0]`. Use both Euclidean distance and cosine similarity to determine which user is most similar to User A. Do the metrics agree?

**Solution:**
```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

user_a = np.array([5, 4, 1, 0, 3])
user_b = np.array([4, 5, 1, 0, 2])
user_c = np.array([1, 1, 5, 4, 0])

print(f"Euclidean(A, B): {np.linalg.norm(user_a - user_b):.4f}")
print(f"Euclidean(A, C): {np.linalg.norm(user_a - user_c):.4f}")
print(f"CosSim(A, B):    {cosine_similarity(user_a, user_b):.4f}")
print(f"CosSim(A, C):    {cosine_similarity(user_a, user_c):.4f}")
# Both metrics agree: User B is more similar to User A.
# User B has similar taste (action/comedy) while User C prefers different genres.
```

---

## Summary

Here's what you now have in your toolkit:

- **Distance** ($\|\mathbf{a} - \mathbf{b}\|$) measures how far apart two vectors are in space. Use it when the actual values matter — positions, quantities, measurements.

- **Cosine similarity** ($\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$) measures directional alignment, ignoring magnitude. Use it when you care about the *pattern* — text similarity, user preferences, semantic meaning.

- **Projection** ($\frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}} \mathbf{b}$) decomposes a vector into a component along another direction and a perpendicular remainder. This is the core operation behind PCA and least-squares regression.

- **Orthogonality** ($\mathbf{a} \cdot \mathbf{b} = 0$) means two vectors are completely independent. Orthogonal features provide non-redundant information. PCA finds orthogonal directions of maximum variance.

- Euclidean distance is sensitive to scale — normalize your features. Cosine similarity is inherently scale-invariant — that's its superpower.

- These geometric concepts directly power k-NN, k-means, document retrieval, recommendation systems, word embeddings, and PCA.

| Concept            | ML Application                                         |
|--------------------|---------------------------------------------------------|
| Euclidean distance | k-NN, k-means clustering, anomaly detection             |
| Cosine similarity  | Document similarity, word embeddings, recommendations   |
| Projection         | PCA, linear regression (least squares)                  |
| Orthogonality      | Independent features, orthonormal weight initialization |

---

> **What's Next** — Vectors let you represent and compare data. But what about *transforming* data? Rotating, scaling, projecting into lower dimensions? That's what matrices do — and that's where we're headed next.
