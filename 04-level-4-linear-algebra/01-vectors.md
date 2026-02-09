# Chapter 1: Vectors

> **Building On** — In the previous chapters on functions, you learned that every ML model maps inputs to outputs. But what ARE those inputs? They're vectors.

---

## The 150,528 Problem

Your image classifier takes in a 224x224 RGB image. That's 224 x 224 x 3 = 150,528 numbers. How do you work with 150,528 numbers at once?

You could try passing them as 150,528 separate arguments to a function. Good luck writing that function signature. You could shove them into a dictionary, but then you lose the ordering and any notion of "closeness" between two images. You could use a plain list — and you'd be *almost* right.

The answer is: you treat them as a single mathematical object — a **vector**.

A vector is an ordered list of numbers, yes. But calling a vector "just a list" is like calling a Git repository "just a folder." Technically true, completely missing the point. Vectors come with a set of operations — addition, scaling, measuring distance, computing similarity — that make them the lingua franca of machine learning.

Every piece of data your ML model touches is a vector:
- That 224x224 RGB image? A vector in $\mathbb{R}^{150528}$.
- A user's ratings across 1000 movies? A vector in $\mathbb{R}^{1000}$.
- The word "king" in Word2Vec? A vector in $\mathbb{R}^{300}$.
- The weights of a neural network layer? Also vectors.

Let's build the intuition from code you already write every day.

---

## You Already Know Arrays. Now Give Them Superpowers.

> **You Already Know This**
>
> If you've ever used a `float[]` in Java, a `list[float]` in Python, or a `number[]` in TypeScript — you already know what a vector *looks like*. The difference is what you're allowed to *do* with it. A vector isn't just storage; it's storage with algebraic rules. Think of it like the jump from a raw `char[]` to a `String` class — same underlying data, but now you have `.length()`, `.contains()`, `.equals()`, and a contract about how those operations behave.

Let's start where you're comfortable — in Python.

```python
import numpy as np

# You already know how to make a list of numbers.
# In ML, we call this a vector.
user_alice = np.array([5, 3, 0, 1, 4])   # Alice's ratings for 5 movies
user_bob   = np.array([4, 0, 0, 1, 5])   # Bob's ratings for 5 movies

print(f"Alice: {user_alice}")
print(f"Bob:   {user_bob}")
print(f"Dimensions: {user_alice.shape[0]}")  # 5 — one rating per movie
```

```
Alice: [5 3 0 1 4]
Bob:   [4 0 0 1 5]
Dimensions: 5
```

Those two arrays are vectors in $\mathbb{R}^5$. Each position corresponds to a movie: position 0 is *The Matrix*, position 1 is *Inception*, and so on. The ordering matters — swap two entries and you've described a completely different set of preferences.

Now here's the question that launches an entire field: **how similar are Alice and Bob?**

You could eyeball it. They both liked movie 0 and movie 4, both rated movie 3 low. But "eyeballing it" doesn't scale to 1000 movies and 10 million users. You need a *number* that measures similarity. That number comes from vector operations.

Let's discover them.

---

## Scalars vs. Vectors: Setting the Vocabulary

Before we go further, let's lock down the terminology — because sloppy vocabulary causes real bugs later.

A **scalar** is a single number. Just one:

$$x = 5$$

A **vector** is an ordered collection of $n$ scalars:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

**Translation:** $\mathbf{v}$ is a column of $n$ real numbers. The $\in \mathbb{R}^n$ part is a type signature — it says "this vector lives in n-dimensional real space." You can read it as: `v: Vector<float, n>`.

Convention: vectors get **bold lowercase** letters ($\mathbf{v}$), scalars get plain italic ($x$). In code, you won't see the bold, so context matters — just like knowing whether `node` means a DOM node or a tree node.

```
Scalar: a single number                Vector: an ordered list of numbers

  x = 5                                  v = [ 3 ]
                                             [ 4 ]
                                             [ 1 ]

  One dimension.                         Three dimensions. (v is in R^3)
  A point on a number line.              A point — or arrow — in 3D space.
```

In 2D, you can picture a vector as an arrow from the origin to a point:

```
        y
        ^
    5   |
    4   |           * (3, 4)       <-- the "tip" of vector v
        |          /
    3   |         /
        |        /
    2   |       /
        |      /
    1   |     /
        |    /
        |   /
        |  /
        | /  theta
    0   +-----|------|------|----> x
        0     1      2      3
```

The vector $\mathbf{v} = [3, 4]$ is the arrow from $(0, 0)$ to $(3, 4)$. Its direction tells you *where* it points; its length tells you *how far*. Both matter.

---

## Operation 1: Vector Addition — Combining Information

### Code First

Suppose Alice rates 5 more movies next week. You want her combined profile across both weeks:

```python
import numpy as np

week1 = np.array([5, 3, 0, 1, 4])
week2 = np.array([0, 1, 4, 2, 0])

combined = week1 + week2
print(f"Week 1:    {week1}")
print(f"Week 2:    {week2}")
print(f"Combined:  {combined}")
```

```
Week 1:    [5 3 0 1 4]
Week 2:    [0 1 4 2 0]
Combined:  [5 4 4 3 4]
```

Simple element-wise addition. You've done this a thousand times. But there's a pattern here worth naming.

### The Math Formalizes It

For two vectors $\mathbf{a}$ and $\mathbf{b}$ of the same dimension $n$:

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}$$

**Translation:** zip the two vectors together and add corresponding pairs. In Python terms: `[a + b for a, b in zip(vec_a, vec_b)]`.

This addition has properties you can rely on — and they matter for proofs, optimizations, and understanding why gradient descent converges:

- **Commutative:** $\mathbf{a} + \mathbf{b} = \mathbf{b} + \mathbf{a}$ — order doesn't matter
- **Associative:** $(\mathbf{a} + \mathbf{b}) + \mathbf{c} = \mathbf{a} + (\mathbf{b} + \mathbf{c})$ — grouping doesn't matter
- **Identity:** $\mathbf{a} + \mathbf{0} = \mathbf{a}$ — the zero vector is the identity element

> **You Already Know This**
>
> These are the same properties that make integer addition safe to reorder and parallelize. When you see a `reduce(+, vectors)` in your ML pipeline, these properties are why the framework can split that sum across GPUs and get the same answer regardless of reduction order.

### The Geometry: Tip-to-Tail

Geometrically, adding two vectors means placing the second vector's tail at the first vector's tip:

```
        y
        ^
    7   |
    6   |                     * (4, 6) = a + b
        |                   / |
    5   |                 /   |
        |         b     /    |
    4   |       (1,2) /      |
        | * - - - ->*        | b = (1, 2)
    3   | |       (3, 4)     |
        | |      /           |
    2   | |    /             |
        | |  /               |
    1   | |/  a = (3, 4)     |
        | /                  |
    0   +-----|------|------|------|-> x
        0     1      2      3     4

    a       = [3, 4]
    b       = [1, 2]
    a + b   = [4, 6]     (place b's tail at a's tip)
```

This is how displacement works in physics, how gradient updates work in optimization, and how residual connections work in ResNets (the output is `x + f(x)` — vector addition of the input and the learned residual).

---

## Operation 2: Scalar Multiplication — Scaling Up and Down

### Code First

Your movie recommendation system learns that certain features should be weighted more heavily. You multiply the entire feature vector by a scaling factor:

```python
import numpy as np

movie_features = np.array([0.8, 0.2, 0.9])  # [action, romance, sci-fi]
emphasis = 2.0

boosted = emphasis * movie_features
print(f"Original:  {movie_features}")
print(f"Boosted:   {boosted}")

dampened = 0.5 * movie_features
print(f"Dampened:  {dampened}")

flipped = -1 * movie_features
print(f"Flipped:   {flipped}")
```

```
Original:  [0.8 0.2 0.9]
Boosted:   [1.6 0.4 1.8]
Dampened:  [0.4 0.1 0.45]
Flipped:   [-0.8 -0.2 -0.9]
```

### The Math

$$c \cdot \mathbf{v} = c \cdot \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \\ \vdots \\ c \cdot v_n \end{bmatrix}$$

**Translation:** multiply every element by the scalar $c$. That's it.

The behavior depends on the scalar:

- $c > 1$: stretches the vector (longer, same direction)
- $0 < c < 1$: shrinks the vector (shorter, same direction)
- $c = 0$: collapses to the zero vector
- $c < 0$: **reverses direction** and scales

```
    Scalar multiplication visualized (vector v = [2, 1]):

        y
        ^
    3   |
        |
    2   |         * 2v = [4, 2]
        |        /
    1   |   * v = [2, 1]
        |  /
    0   +--------|--------|-------> x
        0        2        4
   -1   |  \
        |   * -v = [-2, -1]
   -2   |
```

This shows up constantly in ML: the **learning rate** in gradient descent is scalar multiplication. When you do `weights = weights - learning_rate * gradient`, you're scaling the gradient vector and subtracting it. Too large a scalar? You overshoot. Too small? You crawl.

---

## Operation 3: The Dot Product — The Most Important Operation You'll Learn

### The Problem

Back to our movie recommendation system. You have Alice's ratings and Bob's ratings. You want a single number that tells you: "How similar are these two users?"

```python
import numpy as np

alice = np.array([5, 3, 0, 1, 4])
bob   = np.array([4, 0, 0, 1, 5])

# Attempt 1: subtract and see?
diff = alice - bob
print(f"Difference: {diff}")     # [1, 3, 0, 0, -1]
print(f"Sum of diff: {sum(diff)}")  # 3... but is that "similar" or "different"?
```

```
Difference: [ 1  3  0  0 -1]
Sum of diff: 3
```

That doesn't work well — positives and negatives cancel out. You need something smarter.

### The Insight

What if you multiply corresponding ratings and add them up? If both users rate a movie high, the product is large and positive. If one rates high and the other low, the product is small. If both rate low (or zero), the product contributes nothing.

```python
import numpy as np

alice = np.array([5, 3, 0, 1, 4])
bob   = np.array([4, 0, 0, 1, 5])

# Element-wise multiply, then sum
products = alice * bob
print(f"Element-wise: {products}")  # [20, 0, 0, 1, 20]
print(f"Sum:          {sum(products)}")  # 41

# NumPy has a name for this: the dot product
dot = np.dot(alice, bob)  # or: alice @ bob
print(f"Dot product:  {dot}")  # 41
```

```
Element-wise: [20  0  0  1 20]
Sum:          41
Dot product:  41
```

That number — 41 — is the **dot product** of Alice and Bob's rating vectors. A higher dot product means more similar preferences (both liking the same movies drives the product up).

### The Math

The dot product of two vectors $\mathbf{a}$ and $\mathbf{b}$ in $\mathbb{R}^n$ is:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

**Translation:** multiply matching elements, sum everything. Input: two vectors. Output: one scalar.

There's a stunning geometric interpretation:

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \, \|\mathbf{b}\| \cos(\theta)$$

where $\|\mathbf{a}\|$ is the length (norm) of $\mathbf{a}$, and $\theta$ is the angle between the two vectors.

**Translation:** the dot product is the product of the two lengths, scaled by how much they point in the same direction. The $\cos(\theta)$ term is the key:

- $\cos(0°) = 1$ — vectors point the same way — dot product is **maximally positive**
- $\cos(90°) = 0$ — vectors are perpendicular — dot product is **zero**
- $\cos(180°) = -1$ — vectors point opposite ways — dot product is **maximally negative**

This is why:
- $\mathbf{a} \cdot \mathbf{b} > 0$: vectors point in roughly the same direction (similar users)
- $\mathbf{a} \cdot \mathbf{b} = 0$: vectors are orthogonal (no relationship)
- $\mathbf{a} \cdot \mathbf{b} < 0$: vectors point in roughly opposite directions (opposite tastes)

> **You Already Know This**
>
> If you've ever used cosine similarity in a search engine, recommendation system, or text similarity pipeline — that's just a normalized dot product: $\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}$. The dot product does the heavy lifting; the normalization just removes the effect of vector length so you're comparing pure direction.

```
    Dot product as angle measurement:

    Case 1: Similar          Case 2: Orthogonal      Case 3: Opposite
    (small angle)            (90 degrees)             (180 degrees)

         b  /                     b                        b
           / small                |                        ^
          /  angle                | 90°                    |
    ----a-------->           ----a-------->           <----a---- (pointing
                                                            opposite ways)

    a . b > 0                a . b = 0                a . b < 0
    "Similar tastes"         "Unrelated"              "Opposite tastes"
```

---

## Operation 4: Norms — Measuring the "Size" of a Vector

### The Problem

You're comparing two users' movie ratings, and one user has rated everything on a scale of 1-5 while the other uses 1-10. Their dot product will be inflated — not because they're more similar, but because one has larger numbers. You need to measure (and correct for) the "magnitude" of a vector.

> **You Already Know This**
>
> If you've ever used `Math.abs()`, `len()`, or computed a distance metric in an algorithm — you've used the concept of a norm. A norm is just a function that takes a vector and returns a single non-negative number representing its "size." Different norms measure "size" in different ways, just like different distance functions in your code (Manhattan distance in a grid pathfinding algorithm vs. Euclidean distance in a physics engine).

### Code First: Discovering Norms

```python
import numpy as np

v = np.array([3, -4, 0])

# How "big" is this vector?

# Way 1: Sum up the absolute values of each component
l1 = np.sum(np.abs(v))   # |3| + |-4| + |0| = 7
print(f"L1 norm (sum of absolute values): {l1}")

# Way 2: Pythagorean theorem — straight-line distance from origin
l2 = np.sqrt(np.sum(v**2))  # sqrt(9 + 16 + 0) = sqrt(25) = 5
print(f"L2 norm (Euclidean length):       {l2}")

# NumPy provides np.linalg.norm for this
print(f"L1 via numpy: {np.linalg.norm(v, ord=1)}")  # 7.0
print(f"L2 via numpy: {np.linalg.norm(v, ord=2)}")  # 5.0
print(f"L2 (default): {np.linalg.norm(v)}")          # 5.0 — L2 is the default
```

```
L1 norm (sum of absolute values): 7
L2 norm (Euclidean length):       5.0
L1 via numpy: 7.0
L2 via numpy: 5.0
L2 (default): 5.0
```

### The Math

**L1 Norm (Manhattan distance):**

$$\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i| = |v_1| + |v_2| + \cdots + |v_n|$$

**Translation:** add up the absolute values. Called "Manhattan distance" because it's like navigating a grid of city blocks — you can only move along axes, never diagonally.

**L2 Norm (Euclidean distance):**

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

**Translation:** the straight-line distance from the origin. This is the Pythagorean theorem generalized to $n$ dimensions.

**General Lp Norm:**

$$\|\mathbf{v}\|_p = \left(\sum_{i=1}^{n} |v_i|^p\right)^{1/p}$$

**Translation:** the L1 and L2 norms are special cases of a family. You rarely need $p$ values other than 1 and 2, but knowing the general form helps you read papers.

```
    L1 vs L2 norm for v = [3, 4]:

    L1 path (Manhattan):            L2 path (Euclidean):

         y                               y
     4   +----*  (3,4)              4   |    * (3,4)
         |    |                         |   /
     3   |    |                     3   |  /
         |    |                         | /   length = 5
     2   |    |  total = 3 + 4 = 7  2   |/    (sqrt of 9+16)
         |    |                         |
     1   |    |                     1   |
         |    |                         |
     0   +----+----->  x            0   +---------->  x
         0    3                         0    3

    Walk the grid edges: 7          Fly straight there: 5
```

### Norms in ML: Regularization

Here's where norms stop being abstract and start affecting your model's performance directly. In regularization, you penalize large weight vectors to prevent overfitting:

**L1 Regularization (Lasso):** adds $\lambda \|\mathbf{w}\|_1$ to the loss.

$$\text{Loss} = \text{MSE} + \lambda \|\mathbf{w}\|_1$$

**Translation:** "Penalize the total absolute size of the weights." This drives many weights to exactly zero, producing sparse models — effectively automatic feature selection.

**L2 Regularization (Ridge):** adds $\lambda \|\mathbf{w}\|_2^2$ to the loss.

$$\text{Loss} = \text{MSE} + \lambda \|\mathbf{w}\|_2^2$$

**Translation:** "Penalize large weights, but don't force them to zero — just keep them small." This prevents any single feature from dominating.

> **Common Mistake**
>
> Notice that L2 regularization uses $\|\mathbf{w}\|_2^2$ (the norm *squared*), not $\|\mathbf{w}\|_2$. This isn't a typo — squaring removes the square root, making the gradient cleaner to compute. If you see it written both ways in different sources, check whether they mean the norm or the squared norm. Getting this wrong silently changes your regularization strength.

---

## Unit Vectors: Pure Direction, No Magnitude

### The Problem

In the movie recommendation system, Alice has rated 500 movies and Bob has rated 50. Alice's rating vector is naturally "longer" (larger norm) because she has more non-zero entries. If you compare them with a raw dot product, Alice will seem more similar to *everyone* just because her vector is bigger. You need to strip out magnitude and compare pure direction.

### The Solution: Normalization

```python
import numpy as np

alice = np.array([5, 3, 0, 1, 4])
bob   = np.array([4, 0, 0, 1, 5])

# Normalize to unit vectors (length = 1)
alice_unit = alice / np.linalg.norm(alice)
bob_unit   = bob / np.linalg.norm(bob)

print(f"Alice unit vector: {alice_unit}")
print(f"Bob unit vector:   {bob_unit}")
print(f"Length of alice_unit: {np.linalg.norm(alice_unit):.6f}")
print(f"Length of bob_unit:   {np.linalg.norm(bob_unit):.6f}")

# Now the dot product IS the cosine similarity
cosine_sim = np.dot(alice_unit, bob_unit)
print(f"\nCosine similarity: {cosine_sim:.4f}")
```

```
Alice unit vector: [0.70014004 0.42008402 0.         0.14002801 0.56011203]
Bob unit vector:   [0.61721340 0.         0.         0.15430335 0.77151675]
Length of alice_unit: 1.000000
Length of bob_unit:   1.000000

Cosine similarity: 0.8649
```

### The Math

A **unit vector** is any vector with norm equal to 1. You create one by dividing a vector by its own norm:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

**Translation:** take the vector, divide every element by the vector's length. The result points in the same direction but has length exactly 1. The hat notation ($\hat{\mathbf{v}}$) is the convention for "this is a unit vector."

This is used constantly in ML:
- **Cosine similarity** = dot product of unit vectors
- **Batch normalization** normalizes activations (a related but more complex idea)
- **Weight normalization** decouples direction and magnitude of weight vectors

---

## Putting It All Together: The Movie Recommender

Let's wire up a minimal recommendation engine using everything we've covered. Each user's ratings across movies form a vector. Each movie can also be represented as a vector of features. We compute similarity via cosine similarity (normalized dot product) and recommend movies that similar users liked.

```python
import numpy as np

# === Movie Recommendation with Vectors ===
# 5 users rating 8 movies (0 = not rated)
# Movies: Matrix, Inception, Notebook, Titanic, Interstellar, Avengers, Pride&Prejudice, Alien
ratings = np.array([
    [5, 5, 1, 1, 5, 4, 1, 5],  # User 0: sci-fi fan
    [1, 1, 5, 5, 1, 1, 5, 1],  # User 1: romance fan
    [5, 4, 1, 2, 5, 5, 1, 4],  # User 2: sci-fi/action fan
    [2, 2, 4, 5, 1, 2, 5, 1],  # User 3: romance fan
    [4, 5, 2, 1, 4, 3, 0, 0],  # User 4: new user (sparse ratings)
])

print("=== User Rating Vectors ===")
for i, user in enumerate(ratings):
    norm = np.linalg.norm(user)
    print(f"User {i}: {user}  (L2 norm: {norm:.2f})")

# Compute cosine similarity between User 4 and all others
print("\n=== Cosine Similarity with User 4 ===")
user4 = ratings[4]
user4_norm = np.linalg.norm(user4)

for i in range(4):
    user_i = ratings[i]
    # Cosine similarity = dot product / (norm_a * norm_b)
    cos_sim = np.dot(user4, user_i) / (user4_norm * np.linalg.norm(user_i))
    print(f"User 4 vs User {i}: {cos_sim:.4f}")

# Find the most similar user
similarities = []
for i in range(4):
    cos_sim = np.dot(user4, ratings[i]) / (user4_norm * np.linalg.norm(ratings[i]))
    similarities.append(cos_sim)

most_similar = np.argmax(similarities)
print(f"\nMost similar to User 4: User {most_similar} (similarity: {similarities[most_similar]:.4f})")

# Recommend: movies the similar user rated highly that User 4 hasn't rated
similar_user_ratings = ratings[most_similar]
unrated_by_user4 = user4 == 0  # boolean mask
recommendations = [(j, similar_user_ratings[j])
                   for j in range(len(unrated_by_user4))
                   if unrated_by_user4[j] and similar_user_ratings[j] >= 4]

movie_names = ["Matrix", "Inception", "Notebook", "Titanic",
               "Interstellar", "Avengers", "Pride&Prejudice", "Alien"]
print(f"\nRecommendations for User 4:")
for movie_idx, rating in recommendations:
    print(f"  {movie_names[movie_idx]} (rated {rating} by similar user)")
```

```
=== User Rating Vectors ===
User 0: [5 5 1 1 5 4 1 5]  (L2 norm: 10.10)
User 1: [1 1 5 5 1 1 5 1]  (L2 norm: 8.72)
User 2: [5 4 1 2 5 5 1 4]  (L2 norm: 10.10)
User 3: [2 2 4 5 1 2 5 1]  (L2 norm: 8.12)
User 4: [4 5 2 1 4 3 0 0]  (L2 norm: 8.06)

=== Cosine Similarity with User 4 ===
User 4 vs User 0: 0.9242
User 4 vs User 1: 0.3938
User 4 vs User 2: 0.9313
User 4 vs User 3: 0.4897

Most similar to User 4: User 2 (similarity: 0.9313)

Recommendations for User 4:
  Avengers (rated 5 by similar user)
  Alien (rated 4 by similar user)
```

This entire recommendation system is built from four concepts: vectors (the rating data), the dot product (measuring alignment), norms (measuring magnitude), and unit vectors (normalizing for fair comparison). That's the power of vectors in ML — a small toolkit that builds complex systems.

---

## Where Vectors Appear Across ML

Now that you've seen vectors in action, here's the bigger picture of where they show up:

| Application | Vector Representation | Typical Dimensions |
|---|---|---|
| Tabular data | Each row is a feature vector | 10 - 1000 |
| Images | Flattened pixel values (28x28 = 784D, 224x224x3 = 150,528D) | 784 - 150,528 |
| NLP (word embeddings) | Word2Vec, GloVe | 50 - 300 |
| NLP (sentence/document) | BERT, GPT embeddings | 768 - 4096 |
| Recommender systems | User/item embeddings | 32 - 512 |
| Time series | Sequence of values as a vector | varies |

### Key Algorithms That Run on Vectors

1. **k-Nearest Neighbors (kNN):** Finds the k closest vectors using distance (that's a norm). Your movie recommender above is basically kNN with k=1.
2. **Support Vector Machines (SVMs):** Finds a hyperplane that separates vectors in high-dimensional space. The "support vectors" are the data points closest to the decision boundary.
3. **Neural Networks:** Every layer transforms input vectors into output vectors via learned weights. The forward pass is a chain of vector operations.
4. **Word2Vec:** Learns vector representations of words such that vector arithmetic captures semantic relationships: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$.

That Word2Vec result isn't magic — it's vector addition and subtraction in a learned embedding space. The vectors capture *meaning* as *direction*.

---

## Common Mistakes

> **Common Mistake: "Vectors are just arrays"**
>
> Vectors are arrays with mathematical structure. You can't meaningfully add two arbitrary arrays — what does it mean to add `["alice", 42, true]` to `["bob", 17, false]`? Vectors require that addition, scalar multiplication, and the dot product all behave according to specific axioms. When someone says "vector," they're making a promise about what operations are valid.

> **Common Mistake: Dimension mismatch**
>
> You can't add a vector in $\mathbb{R}^3$ to a vector in $\mathbb{R}^5$. NumPy will either broadcast (silently giving you a wrong answer in some cases) or throw an error. In ML pipelines, dimension mismatches are one of the most common bugs. Always check `.shape` before operating on vectors.

> **Common Mistake: Forgetting to normalize**
>
> Raw dot products are affected by vector magnitude. If you're measuring *similarity* (direction), normalize first. If you're measuring *strength* (magnitude matters), don't. Knowing which one you want is the difference between a cosine similarity search and a raw inner product search — and they give different rankings.

> **Common Mistake: Scale differences across features**
>
> If your feature vector has `[salary_in_dollars, age_in_years, height_in_meters]`, the salary component (tens of thousands) will completely dominate any norm or dot product calculation. Feature scaling (standardization, min-max normalization) isn't optional — it's a prerequisite for most vector-based algorithms.

> **Common Mistake: Confusing row and column vectors**
>
> Mathematically, a column vector and a row vector are different objects (one is $n \times 1$, the other is $1 \times n$). NumPy 1D arrays are technically neither — they're shape `(n,)`. This usually doesn't matter until you hit matrix multiplication, where `(n,)` vs `(n, 1)` vs `(1, n)` can produce silently different results. When in doubt, be explicit: `v.reshape(-1, 1)` for column, `v.reshape(1, -1)` for row.

---

## Exercises

### Exercise 1: Basic Operations

Given $\mathbf{a} = [1, 2, 3]$ and $\mathbf{b} = [4, 5, 6]$, compute by hand first, then verify with code:
- $\mathbf{a} + \mathbf{b}$
- $3\mathbf{a}$
- $\mathbf{a} \cdot \mathbf{b}$

**Solution:**
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"a + b = {a + b}")           # [5, 7, 9]
print(f"3a = {3 * a}")              # [3, 6, 9]
print(f"a . b = {np.dot(a, b)}")    # 1*4 + 2*5 + 3*6 = 32
```

### Exercise 2: Norms

For the vector $\mathbf{v} = [3, -4, 0]$, compute the L1 and L2 norms by hand, then verify.

**Solution:**
```python
import numpy as np

v = np.array([3, -4, 0])

l1 = np.linalg.norm(v, ord=1)  # |3| + |-4| + |0| = 7
l2 = np.linalg.norm(v, ord=2)  # sqrt(9 + 16 + 0) = sqrt(25) = 5

print(f"L1 norm: {l1}")  # 7.0
print(f"L2 norm: {l2}")  # 5.0
```

### Exercise 3: Unit Vectors

Normalize the vector $\mathbf{u} = [1, 1, 1, 1]$ and verify it has unit length.

**Solution:**
```python
import numpy as np

u = np.array([1, 1, 1, 1])

# L2 norm of u is sqrt(1 + 1 + 1 + 1) = sqrt(4) = 2
unit_u = u / np.linalg.norm(u)

print(f"Unit vector: {unit_u}")                          # [0.5, 0.5, 0.5, 0.5]
print(f"Norm of unit vector: {np.linalg.norm(unit_u)}")  # 1.0
```

### Exercise 4: Movie Similarity (Putting It All Together)

Three users have rated 6 movies. Compute the cosine similarity between each pair and determine which two users are most similar.

```python
import numpy as np

# Ratings for 6 movies
user_a = np.array([5, 4, 1, 0, 5, 3])
user_b = np.array([1, 0, 5, 4, 1, 2])
user_c = np.array([4, 5, 2, 1, 4, 4])
```

**Solution:**
```python
import numpy as np

user_a = np.array([5, 4, 1, 0, 5, 3])
user_b = np.array([1, 0, 5, 4, 1, 2])
user_c = np.array([4, 5, 2, 1, 4, 4])

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

sim_ab = cosine_similarity(user_a, user_b)
sim_ac = cosine_similarity(user_a, user_c)
sim_bc = cosine_similarity(user_b, user_c)

print(f"Similarity A-B: {sim_ab:.4f}")
print(f"Similarity A-C: {sim_ac:.4f}")
print(f"Similarity B-C: {sim_bc:.4f}")

# A and C are most similar — they're both sci-fi/action fans
```

---

## Summary

- **Vectors** are ordered lists of numbers with mathematical structure — not just arrays, but arrays with a contract about valid operations
- **Vector addition** is element-wise; both vectors must have the same dimension (think: combining information from two sources)
- **Scalar multiplication** scales every element uniformly (think: learning rate, feature weighting)
- The **dot product** multiplies corresponding elements and sums them, producing a scalar that measures *alignment* between two vectors
- The **geometric interpretation** of the dot product ($\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$) connects algebra to geometry — positive means similar direction, zero means orthogonal, negative means opposite
- **L1 norm** sums absolute values (Manhattan distance); **L2 norm** is the Euclidean length (straight-line distance)
- **Unit vectors** have norm 1 and preserve only direction — normalizing lets you compare pure direction via cosine similarity
- In ML, every data point is a vector, and these operations power similarity search, neural network layers, regularization, and embeddings
- Always check dimensions, always consider whether to normalize, and always be aware of feature scale differences

---

> **What's Next** — Now that you can represent data as vectors, how do you measure similarity, distance, and angles between them systematically? How do you project one vector onto another, and why does that matter for dimensionality reduction? That's the geometry of vectors.
