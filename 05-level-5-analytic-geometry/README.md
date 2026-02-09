# Level 5: Analytic Geometry

Your embedding search returns garbage results. The vectors are fine—BERT encoded them, the dimensions look reasonable, the cosine similarities are in range. But users searching for "machine learning" get results about cooking recipes, and nobody can figure out why. You check the distance metric. Someone switched it from cosine to Euclidean last sprint. The vectors are normalized, so L2 distance is meaningless. One line change, entire feature broken. That's geometry in production.

This level is about the spaces your data lives in and how you measure distance within them. Not abstract coordinate planes from high school—the 512-dimensional spaces where your embeddings cluster, the feature spaces where KNN makes decisions, the manifolds where your model searches for optima. You already know vectors and matrices from Level 4. Now you learn how to navigate the geometry they create.

## Building On: From Vectors to Spaces

Level 4 gave you vectors (points with direction) and matrices (transformations). But vectors exist *somewhere*—coordinate systems define where. And if two vectors are points in space, you need to measure how far apart they are. That's geometry: the study of position and distance.

You've been using coordinate systems and distance metrics without thinking about them. Every time you run `sklearn.neighbors.KNeighborsClassifier`, you're choosing a distance metric. Every time you normalize features before training, you're changing the geometry of your space. Every time you apply PCA, you're rotating your coordinate system to align with variance.

This level makes those choices explicit. You'll learn which coordinate systems expose structure in your data, which distance metrics match your problem semantics, and how high-dimensional spaces behave differently than your 2D intuitions.

## What You'll Learn

| Chapter | Topics | ML Applications |
|---------|--------|-----------------|
| **1. Coordinate Systems** | Cartesian, polar, n-dimensional spaces, coordinate transforms, PCA as rotation, curse of dimensionality | Feature engineering, dimensionality reduction, embedding spaces |
| **2. Distance Metrics** | Euclidean (L2), Manhattan (L1), Minkowski (Lp), cosine similarity, when to use which, metric decision tree | KNN, clustering, similarity search, recommendation systems |

### Chapter 1: Coordinate Systems

You have 50 features describing user behavior. Are those the right coordinates, or should you rotate into principal components? Are your features independent axes, or do they encode redundant information? This chapter covers how to represent points in space, how to transform between coordinate systems, and why high dimensions break your intuition.

You'll see Cartesian coordinates generalize to n dimensions naturally. You'll see polar coordinates show up when angles matter more than positions. You'll see PCA reframe as "finding a better coordinate system." And you'll confront the curse of dimensionality: in 1000 dimensions, all points are roughly equidistant, so distance-based methods stop working.

### Chapter 2: Distance Metrics

Euclidean distance treats all dimensions equally. Manhattan distance allows independent axis movement. Cosine similarity ignores magnitude and measures angle. Each metric encodes different assumptions about what "similar" means.

If you're comparing document embeddings, cosine similarity works because vector magnitude doesn't indicate semantic similarity. If you're routing delivery trucks, Manhattan distance works because you move on a grid. If your features have different units (age in years, salary in dollars), Euclidean distance is meaningless until you normalize.

This chapter gives you a decision tree: "Use L1 when... Use L2 when... Use cosine when..." You'll understand why recommender systems use cosine, why k-means uses Euclidean, and how to choose the right metric for your data.

## What Comes Next

This geometry becomes concrete in three directions:

**Level 6: Calculus** builds on coordinate systems to define gradients (directions of steepest increase) and optimization landscapes (surfaces you navigate during training). Distance metrics become loss functions you minimize.

**Level 12: Numerical Methods** handles the practical issues: How do you compute distances efficiently in high dimensions? How do you avoid numerical instability? How do you approximate nearest neighbors without checking every point?

**Level 13: ML Models** brings it together: KNN classifies by distance, k-means clusters by distance, neural networks learn embeddings where cosine similarity matches semantic similarity. Geometry becomes the operational principle.

## Prerequisites

You need Level 4: Linear Algebra. Vectors, matrices, dot products, matrix multiplication. This level adds the geometry layer on top of that algebra.

## Learning Path

Start with coordinate systems → understand the spaces your data lives in
Then move to distance metrics → learn how to measure within those spaces

```
Coordinate Systems          Distance Metrics           Applications
------------------          ----------------           ------------
Cartesian, polar      -->   L1, L2, cosine      -->    KNN, clustering,
n-dimensional spaces        When to use which          similarity search,
Transforms, PCA             Metric decision tree       retrieval systems
Curse of dimensionality
```

## Getting Started

Begin with [01-coordinate-systems.md](./01-coordinate-systems.md). You'll start with 2D Cartesian coordinates, generalize to n dimensions, see how PCA rotates your coordinate system, and confront high-dimensional geometry. Then move to [02-distance-metrics.md](./02-distance-metrics.md) to learn how different metrics capture different notions of similarity and when to use each one.

By the end of this level, you'll know why your embedding search broke (wrong metric), how to choose distance functions that match your problem semantics, and how to reason about the geometry of high-dimensional spaces where most ML happens.
