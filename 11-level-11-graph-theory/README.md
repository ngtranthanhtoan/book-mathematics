# Level 11: Graph Theory

You've been working with graphs your whole career — dependency trees, network topologies, routing algorithms, Git DAGs. Graph theory formalizes what you already know and extends it to Graph Neural Networks (GNNs), knowledge graphs, and molecular property prediction.

The difference? You've been using graphs as **data structures**. Now you'll understand them as **mathematical objects** with properties you can compute, algorithms you can prove, and transformations you can learn.

## What You Already Know

Let's connect your existing knowledge:

**Graphs as Data Structures**
```python
# You've written this hundreds of times
class Node:
    def __init__(self, value):
        self.value = value
        self.neighbors = []
```

**Adjacency Matrix = Sparse Matrix Storage**
That graph traversal? It's matrix multiplication. The adjacency matrix `A[i][j] = 1` if node `i` connects to node `j`. GNNs aggregate neighbor features via `A @ H` — matrix multiply you already know, now applied to neural network layers.

**BFS/DFS = Tree Traversal You Know**
You've implemented breadth-first search for level-order traversal and depth-first search for dependency resolution. Same algorithms, different context. Now you'll see how they power graph sampling, node embeddings (DeepWalk), and GNN neighborhood aggregation.

**PageRank = Distributed Reputation Scoring**
It's like consensus algorithms in distributed systems — nodes vote for each other's importance, weighted by their own importance. You've seen this pattern in upvote systems, citation counts, and package dependency ranking.

## Why This Matters for ML

Real-world data is relational:

- **Fraud Detection**: Model transactions as a graph. Fraud patterns emerge from network topology (rings of accounts, rapid money movement).
- **Molecular Property Prediction**: Atoms are nodes, bonds are edges. GNNs predict drug toxicity, solubility, and binding affinity without manual feature engineering.
- **Recommendation Systems**: Users and items form a bipartite graph. Pinterest's PinSage uses GNNs to recommend billions of pins.
- **Knowledge Graphs**: Entities and relationships. Google's search uses graph reasoning to answer "who directed Inception's cinematographer's first film?"
- **Social Networks**: Community detection, influence propagation, bot detection — all graph problems.

Traditional ML assumes data points are independent. Graph ML exploits structure. That's the paradigm shift.

## Building On: Linear Algebra (Level 4)

Everything here builds on Level 4:

- **Adjacency matrices are matrices**: Sparse storage, eigendecomposition, spectral properties
- **GNN message passing is matrix multiplication**: `H_new = σ(A @ H @ W)` — aggregate neighbors, apply weights, activate
- **Laplacian matrix** `L = D - A`: Its eigenvectors reveal graph structure (spectral clustering)
- **Random walk transition matrix**: Stochastic matrix whose steady state gives node importance (PageRank)

You're not learning new matrix operations. You're applying them to graph-structured data.

## What You'll Learn

| Chapter | Topic | ML Application |
|---------|-------|----------------|
| [01-graph-basics.md](01-graph-basics.md) | Nodes, edges, directed/undirected graphs, adjacency matrices, bipartite graphs | GNN foundations, knowledge graph representation, user-item matrices |
| [02-graph-properties.md](02-graph-properties.md) | Degree distributions, paths, cycles, connectivity, Laplacian matrix | Spectral clustering, graph partitioning, community detection |
| [03-graph-algorithms.md](03-graph-algorithms.md) | BFS/DFS traversal, PageRank, random walks (DeepWalk/node2vec), topological sort | Node embeddings, importance ranking, graph sampling for mini-batch training |

## Chapter Breakdown

### Chapter 1: Graph Basics
You know graphs as data structures with nodes and edges. Here you'll formalize them mathematically. The key insight: **adjacency matrices turn graph operations into linear algebra**. A GNN layer is just `A @ H @ W` — multiply by adjacency matrix, apply learned weights. You'll understand directed vs. undirected graphs, weighted edges, bipartite graphs (user-item networks), and why sparse matrix formats matter for million-node graphs.

**SWE Bridge**: Adjacency lists are CSR (Compressed Sparse Row) format you've seen in scipy.sparse. You've already chosen the right representation for cache locality — now you'll see why it's mathematically principled.

### Chapter 2: Graph Properties
Degree distributions tell you if a network is scale-free (social networks) or uniform (grid graphs). Paths and connectivity determine if information can flow between nodes. The **Laplacian matrix** `L = D - A` encodes graph structure in its spectrum — its eigenvectors reveal communities via spectral clustering.

**SWE Bridge**: Cycle detection is dependency resolution in your build system. Strongly connected components are failure domains in distributed systems. Same algorithms, now with formal properties you can prove.

### Chapter 3: Graph Algorithms
BFS and DFS you've coded hundreds of times. Now you'll see how they power **random walks** for node embeddings (DeepWalk, node2vec) and graph sampling for mini-batch GNN training. **PageRank** is the eigenvector of a stochastic matrix — distributed reputation scoring at scale. Topological sort orders nodes by dependencies (your build system does this).

**SWE Bridge**: PageRank is like Elo rating or HackerNews ranking — nodes vote for each other's importance, weighted by their own credibility. The math formalizes what you've intuited.

## Prerequisites

- **Level 4 (Linear Algebra)**: Matrix multiplication, eigenvalues/eigenvectors, sparse matrices
- **Basic Python**: NumPy array operations, scipy.sparse for large graphs

You don't need probability theory for the core concepts. PageRank uses a stochastic matrix, but the intuition is "voting weighted by importance" — no probability required to start.

## Tools You'll Use

```python
import numpy as np                   # Adjacency matrices
import scipy.sparse as sp            # Sparse graphs (millions of nodes)
import networkx as nx                # Graph algorithms (prototype)
import torch_geometric as pyg        # Production GNNs
```

For production ML:
- **PyTorch Geometric**: GNN layers (GCN, GAT, GraphSAGE)
- **DGL (Deep Graph Library)**: Message passing framework
- **NetworkX**: Rapid prototyping and algorithm testing

## What Comes Next

- **Level 12 (Numerical Methods)**: Implement GNNs efficiently, sparse matrix operations, graph sampling strategies
- **Level 13 (ML Models Math)**: See graph concepts in action — molecular property prediction, social network embeddings, recommendation systems

## How to Approach This Level

1. **Map to what you know**: Every graph concept has a SWE analog. Find it.
2. **Think in matrices**: Graph operations are linear algebra in disguise.
3. **Code it**: Implement BFS, DFS, PageRank from scratch before using libraries.
4. **Visualize**: Draw small graphs (5-10 nodes) to build intuition.
5. **Scale up**: Then think about million-node graphs and sparse representations.

## Estimated Time

- **Chapter 1 (Graph Basics)**: 2-3 hours
- **Chapter 2 (Graph Properties)**: 3-4 hours
- **Chapter 3 (Graph Algorithms)**: 4-5 hours

**Total**: 10-12 hours. You'll move faster than someone learning graphs from scratch — you've built this intuition debugging dependency graphs and reasoning about system topologies.

## The Mental Shift

You've been thinking: "Graph = {nodes, edges, traversal methods}"

Now think: "Graph = adjacency matrix = tensor you can multiply, decompose, and learn over"

That shift unlocks GNNs, spectral clustering, and graph-based deep learning. Let's formalize what you already know and extend it to ML.

---

**Start with [Chapter 1: Graph Basics](01-graph-basics.md)** to see how the data structure you know becomes a mathematical object you can analyze and learn from.
