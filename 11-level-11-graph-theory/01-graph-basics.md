# Chapter 1: Graph Basics

> You have been representing relationships as graphs your entire career. Now it is time to formalize that intuition so you can feed those structures into neural networks.

---

## Building On

Throughout this book, we have built up vectors, matrices, and linear transformations. We used information theory to measure how distributions diverge. But all of that assumed our data points are independent — each row in a dataset, each sample in a batch.

What happens when the *relationships between data points* are the data?

Your microservices call each other. Your git commits form a history. Your users follow each other. Your molecules are atoms bonded together. None of these fit neatly into a matrix of independent rows. They fit into graphs. And graphs, it turns out, can be represented as matrices — which means everything you learned about linear algebra and matrix operations applies directly.

This chapter formalizes the structures you already use daily, then shows you exactly how they power Graph Neural Networks, knowledge graphs, and molecular property prediction.

---

## The Problem: Why Your Recommender Keeps Failing

You are building a recommendation engine for an e-commerce platform. You have user features (age, location, browsing history) and item features (category, price, description embedding). You train a standard model — maybe a neural network that takes the concatenation of user and item features and predicts a rating.

It works okay. But it misses something obvious to you as a human: *social influence*. When Alice buys something, her friends Bob and Carol are more likely to buy it too. Your model treats every user as an isolated data point, so it cannot capture this.

You think: "I need to encode the social graph somehow." But how do you turn a web of friendships into something a neural network can process?

```python
# Your first attempt: just add a "number of friends" feature
user_features = {
    "alice": {"age": 30, "num_friends": 3},
    "bob":   {"age": 28, "num_friends": 2},
    "carol": {"age": 32, "num_friends": 4},
    "david": {"age": 25, "num_friends": 1},
}
# But this throws away WHO the friends are.
# Alice having 3 friends who all bought Product X
# is very different from 3 friends who bought nothing.
```

The "number of friends" feature is a lossy compression of the graph. You need the actual structure — who is connected to whom — so information can flow along edges. That is what graph-based ML does: it lets your model reason about the *topology* of relationships.

To do that, you need a precise mathematical language for graphs. Let us build it.

---

## You Already Know This

Before we write a single formula, let me show you something.

> **You Already Know This**: You have been working with graphs your entire engineering career. You just called them different names.

Here is a quick inventory of graphs you have already mastered:

| What You Call It | What It Really Is |
|---|---|
| Dependency graph (`package.json`, `requirements.txt`) | Directed Acyclic Graph (DAG) |
| CI/CD pipeline (build -> test -> deploy) | DAG with weighted edges (time) |
| Git commit history | DAG (each commit points to its parent) |
| Microservice architecture diagram | Directed graph (service A calls service B) |
| Database schema (foreign keys) | Directed graph (table A references table B) |
| Network topology (load balancers, servers) | Undirected or directed graph |
| Class inheritance hierarchy | Rooted tree (a special graph) |
| Social network (LinkedIn connections) | Undirected graph |
| Twitter follow graph | Directed graph |
| DNS resolution chain | Directed graph |

That `npm install` you ran this morning? It resolved a dependency DAG. That Kubernetes service mesh routing traffic? It is traversing a directed weighted graph. That `git log --graph` you love? Literally a graph.

The math we are about to formalize is the same structure you already manipulate in code every day. The difference: once you formalize it, you can feed it into matrix operations, and from there into neural networks.

---

## Formalizing What You Know

### The Core Abstraction: Nodes and Edges

Let us start with the simplest possible definition and build up.

**Definition (Graph)**: A graph $G$ is an ordered pair $G = (V, E)$ where:
- $V$ is a finite set of **vertices** (also called nodes)
- $E$ is a set of **edges**, where each edge connects two vertices

**Translation**: $V$ is your set of "things" (users, servers, atoms, commits). $E$ is your set of "connections" between those things (friendships, API calls, chemical bonds, parent-child relationships).

That is it. Two sets. One for things, one for connections. Everything else is details about what kind of connections you allow.

### Undirected vs. Directed: The Two Flavors

**Definition (Undirected Graph)**: An edge $e \in E$ is an unordered pair $\{u, v\}$ where $u, v \in V$. The edge $\{u, v\}$ is the same as $\{v, u\}$.

**Definition (Directed Graph / Digraph)**: An edge $e \in E$ is an ordered pair $(u, v)$ where $u, v \in V$. The edge $(u, v)$ goes *from* $u$ *to* $v$, and is different from $(v, u)$.

**Translation**: Undirected = mutual relationship (LinkedIn connection, chemical bond, ethernet cable). Directed = one-way relationship (Twitter follow, git parent pointer, HTTP request, `import` statement).

Here is the difference, visualized:

```
  UNDIRECTED (LinkedIn)                 DIRECTED (Twitter)
  Friendship is mutual                 Following is one-way

  Alice ------- Bob                    Alice -------> Bob
    |  \                                              |
    |   \                                             v
    |    \                              Carol ------> David
  David -- Carol                          |
                                          +---------> Bob

  {Alice, Bob} = {Bob, Alice}          (Alice, Bob) =/= (Bob, Alice)
  Same edge                            Different edges (or one missing)
```

> **SWE Insight**: If you have worked with Protocol Buffers or Thrift, think of undirected edges as a bidirectional RPC and directed edges as a unidirectional RPC. Or think of HTTP: a request from client to server is a directed edge. A WebSocket connection is closer to an undirected edge.

### Weighted Graphs: When Edges Have Values

**Definition (Weighted Graph)**: A weighted graph is a triple $G = (V, E, w)$ where $w: E \to \mathbb{R}$ assigns a real-valued weight to each edge.

**Translation**: Not all connections are equal. Some friendships are stronger. Some API calls have higher latency. Some chemical bonds have different energies. The weight captures "how much" or "how strong" the connection is.

```
  WEIGHTED GRAPH (Network Latency in ms)

  Server A ----[2ms]---- Server B
     |                      |
   [5ms]                  [1ms]
     |                      |
  Server C ----[3ms]---- Server D

  Edge weights represent connection costs.
  Shortest path A->D: A-B-D (3ms), not A-C-D (8ms).
```

In ML, edge weights show up everywhere:
- **Similarity graphs**: weight = cosine similarity between feature vectors
- **Distance graphs**: weight = Euclidean distance between data points
- **Attention graphs**: weight = learned attention score (Transformers!)
- **Knowledge graphs**: weight = confidence score for a fact

---

## Running Example: Molecular Graphs for Drug Discovery

Let us establish a running example that we will build on throughout this chapter and the next two chapters on graph properties and algorithms.

In drug discovery, molecules are naturally represented as graphs:
- **Nodes** = atoms (carbon, nitrogen, oxygen, etc.)
- **Edges** = chemical bonds (single, double, triple)
- **Node features** = atomic number, charge, hybridization
- **Edge features** = bond type, bond length

Here is caffeine as a graph:

```
  Caffeine molecule (C8H10N4O2) as a graph:

        O                              Node types:
        ||                               C = Carbon
    N---C---N                             N = Nitrogen
   / \     / \                            O = Oxygen
  C   C---C   C---CH3                     H = Hydrogen (implicit)
  ||  |   ||  |
  N   N   C   N                          Edge types:
   \ / \ / \ /                            --- single bond
    C    C   CH3                          === double bond (shown as ||)
    |    |
    O    CH3

  V = {C1, C2, C3, ..., N1, N2, ..., O1, O2}
  E = {(C1,N1), (C1,C2), (N1,C3), ...}
  Node features: atomic_number, charge, ...
  Edge features: bond_type (single=1, double=2, triple=3)
```

Why does this matter? Graph Neural Networks (GNNs) can predict molecular properties — toxicity, solubility, binding affinity — directly from this graph structure. No hand-engineered molecular fingerprints needed. The GNN learns which structural patterns (subgraphs) correlate with the target property.

But before we can feed a molecule into a GNN, we need to represent it as a matrix. That brings us to the key representation.

---

## The Adjacency Matrix: Where Graphs Meet Linear Algebra

This is where it gets exciting. You already know matrices from linear algebra. It turns out that a graph can be perfectly encoded as a matrix, and that encoding is what makes graph-based ML possible.

### Building the Adjacency Matrix from Scratch

For a graph with $n$ nodes, the **adjacency matrix** $A$ is an $n \times n$ matrix where:

$$A_{ij} = \begin{cases} 1 & \text{if there is an edge from node } i \text{ to node } j \\ 0 & \text{otherwise} \end{cases}$$

**Translation**: Row $i$, column $j$ of the matrix answers one question: "Is there an edge from node $i$ to node $j$?" Yes = 1, No = 0.

Let us build one by hand with our social network example:

```
  The graph:                    Node index mapping:
                                  Alice = 0
  Alice ------- Bob               Bob   = 1
    \          /                   Carol = 2
     \        /                    David = 3
      \      /
       Carol
         |
       David


  Step-by-step adjacency matrix construction:

  Ask: "Is there an edge from row to column?"

              Alice  Bob  Carol  David
              (0)    (1)  (2)    (3)
  Alice (0) [  0      1    1      0  ]    Alice->Alice? No. Alice->Bob? Yes. ...
  Bob   (1) [  1      0    1      0  ]    Bob->Alice? Yes. Bob->Bob? No. ...
  Carol (2) [  1      1    0      1  ]    Carol knows everyone except herself
  David (3) [  0      0    1      0  ]    David only knows Carol
```

$$A = \begin{bmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 0 \\ 1 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

Notice something? The matrix is **symmetric**: $A_{ij} = A_{ji}$. That is because this is an undirected graph — if Alice is friends with Bob, then Bob is friends with Alice. The matrix mirrors across its diagonal.

> **You Already Know This**: This is like a distance matrix. If you have ever built a pairwise distance table (e.g., for caching nearest-neighbor lookups), you know it is symmetric because distance(A, B) = distance(B, A). An undirected adjacency matrix has the same property.

### Directed Adjacency Matrix: Asymmetry Tells a Story

For a directed graph, the matrix is generally **not** symmetric:

$$A_{ij} = \begin{cases} 1 & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

```
  Directed graph (Twitter follows):

  Alice -------> Bob -------> Carol
                  ^             |
                  |             v
                  +--- Carol   David

  (Alice follows Bob, Bob follows Carol,
   Carol follows Bob back, Carol follows David)

              Alice  Bob  Carol  David
              (0)    (1)  (2)    (3)
  Alice (0) [  0      1    0      0  ]    Alice follows Bob only
  Bob   (1) [  0      0    1      0  ]    Bob follows Carol only
  Carol (2) [  0      1    0      1  ]    Carol follows Bob and David
  David (3) [  0      0    0      0  ]    David follows nobody
```

$$A = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

$A_{01} = 1$ (Alice follows Bob) but $A_{10} = 0$ (Bob does not follow Alice). The asymmetry encodes directionality.

**Translation**: For directed graphs, rows are "from" and columns are "to." Row $i$ tells you who node $i$ points to. Column $j$ tells you who points to node $j$.

### Weighted Adjacency Matrix

For weighted graphs, replace the 1s with actual weights:

$$A_{ij} = \begin{cases} w(i, j) & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

```
  Weighted graph (API latency between services, in ms):

  Auth ----[12ms]---- Gateway ----[3ms]---- API
    |                                        |
  [8ms]                                    [5ms]
    |                                        |
   DB  ---------------[2ms]--------------- Cache

              Auth  Gateway  API  DB  Cache
              (0)   (1)      (2)  (3) (4)
  Auth   (0) [  0     12       0    8    0 ]
  Gateway(1) [ 12      0       3    0    0 ]
  API    (2) [  0      3       0    0    5 ]
  DB     (3) [  8      0       0    0    2 ]
  Cache  (4) [  0      0       5    2    0 ]
```

> **Common Mistake**: A zero in a weighted adjacency matrix is ambiguous — does it mean "no edge" or "edge with weight 0"? In practice, many implementations use `float('inf')` or `NaN` for missing edges and reserve 0 for actual zero-weight edges. When using sparse representations (scipy.sparse, PyTorch sparse tensors), absent entries are implicitly zero, so this distinction is handled automatically. Watch out for this when converting between dense and sparse formats.

---

## Three Representations: Pick the Right Data Structure

Just as you choose between a HashMap, a TreeMap, and a sorted array depending on your access patterns, graphs have multiple representations. Each has different performance characteristics.

### 1. Adjacency Matrix

You have already seen this. Here is the formal summary:

For a graph with $n$ nodes:

**Undirected**:
$$A_{ij} = A_{ji} = \begin{cases} 1 & \text{if } \{i, j\} \in E \\ 0 & \text{otherwise} \end{cases}$$

**Directed**:
$$A_{ij} = \begin{cases} 1 & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

**Weighted**:
$$A_{ij} = \begin{cases} w(i, j) & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

```
  Performance characteristics:

  Operation               Time        Space
  ─────────────────────────────────────────────
  Store graph             -           O(n^2)
  Check if edge exists    O(1)        -
  Find all neighbors      O(n)        -
  Add an edge             O(1)        -
  Iterate all edges       O(n^2)      -
```

**Best for**: Dense graphs, matrix operations (this is what GNNs use), spectral methods, any algorithm that needs the full adjacency structure at once.

**Worst for**: Sparse graphs with millions of nodes. A social network with 1 billion users and average 100 friends each would need a $10^9 \times 10^9$ matrix — roughly $10^{18}$ entries. That is an exabyte of memory for a matrix that is 99.99999% zeros.

### 2. Adjacency List

Each node maintains a list of its neighbors:

$$\text{adj}[v] = \{u \in V : (v, u) \in E\}$$

**Translation**: It is a dictionary (hash map) where keys are nodes and values are lists of neighbors. Exactly like you would implement it.

```
  Adjacency list for the social network:

  Alice -> [Bob, Carol]
  Bob   -> [Alice, Carol]
  Carol -> [Alice, Bob, David]
  David -> [Carol]

  In Python: {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}
```

```
  Performance characteristics:

  Operation               Time              Space
  ─────────────────────────────────────────────────────
  Store graph             -                 O(n + m)
  Check if edge exists    O(deg(v))         -
  Find all neighbors      O(deg(v))         -
  Add an edge             O(1)              -
  Iterate all edges       O(n + m)          -

  where m = |E| (number of edges), deg(v) = degree of node v
```

**Best for**: Sparse graphs (most real-world graphs), graph traversals (BFS, DFS), streaming graph algorithms. This is what NetworkX uses internally.

**Worst for**: "Does edge (u, v) exist?" queries on high-degree nodes. For that, use an adjacency set (dict of sets) instead of dict of lists — $O(1)$ average lookup.

### 3. Edge List

Simply a list of all edges:

$$E = [(u_1, v_1), (u_2, v_2), \ldots, (u_m, v_m)]$$

```
  Edge list for the social network:

  [(Alice, Bob), (Alice, Carol), (Bob, Carol), (Carol, David)]

  In Python: [(0, 1), (0, 2), (1, 2), (2, 3)]

  For weighted graphs, add a third element:
  [(0, 1, 1.0), (0, 2, 0.8), (1, 2, 0.5), (2, 3, 0.3)]
```

```
  Performance characteristics:

  Operation               Time        Space
  ─────────────────────────────────────────────
  Store graph             -           O(m)
  Check if edge exists    O(m)        -
  Find all neighbors      O(m)        -
  Add an edge             O(1)        -
  Iterate all edges       O(m)        -
```

**Best for**: Storage and transmission (CSV files, database tables), algorithms that iterate over all edges (Kruskal's MST), and as input format for graph libraries. This is the format used by most graph datasets (e.g., the OGB benchmark suite).

> **You Already Know This**: An edge list is essentially a two-column table — it is how you would store a graph in a relational database. One table with columns `(source_id, target_id, weight)`. That is exactly what it is. Every time you have had a `follows` table or a `friendships` table, you were storing an edge list.

### How to Choose

```
  Decision guide:

  Is your graph dense (>30% of possible edges exist)?
      YES -> Adjacency Matrix
      NO  -> Continue

  Do you need matrix operations (GNN, spectral methods)?
      YES -> Adjacency Matrix (sparse format if large)
      NO  -> Continue

  Do you need fast neighbor lookups?
      YES -> Adjacency List (or Adjacency Set for edge queries)
      NO  -> Edge List (simplest storage)
```

> **Common Mistake**: Do not assume you must pick one representation. In practice, you often store as an edge list (compact, serializable), convert to adjacency list for traversals, and convert to a sparse adjacency matrix for GNN training. PyTorch Geometric does exactly this — it stores `edge_index` (a 2-row edge list tensor) and converts to sparse matrix form for message passing.

---

## Code: Building Graph Representations from Scratch

Let us implement all three representations and see them work together. This is production-style Python — not toy code.

```python
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set

class Graph:
    """
    A graph with simultaneous adjacency matrix, adjacency list, and edge
    list representations. In production you would pick one; here we maintain
    all three to compare them side-by-side.
    """

    def __init__(self, num_nodes: int, directed: bool = False):
        self.num_nodes = num_nodes
        self.directed = directed

        # Representation 1: Adjacency matrix (dense)
        self.adj_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

        # Representation 2: Adjacency list (dict of lists)
        self.adj_list: Dict[int, List[int]] = defaultdict(list)

        # Representation 3: Edge list
        self.edges: List[Tuple[int, int, float]] = []

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        """Add an edge from node u to node v with optional weight."""
        # Update all three representations simultaneously
        self.adj_matrix[u][v] = weight
        self.adj_list[u].append(v)
        self.edges.append((u, v, weight))

        if not self.directed:
            self.adj_matrix[v][u] = weight
            self.adj_list[v].append(u)

    def has_edge(self, u: int, v: int) -> bool:
        """O(1) edge lookup using adjacency matrix."""
        return self.adj_matrix[u][v] != 0

    def get_neighbors(self, v: int) -> List[int]:
        """O(deg(v)) neighbor lookup using adjacency list."""
        return self.adj_list[v]

    def degree(self, v: int) -> int:
        """Return the degree of node v."""
        return len(self.adj_list[v])

    def is_symmetric(self) -> bool:
        """Check if the adjacency matrix is symmetric (undirected graph test)."""
        return np.allclose(self.adj_matrix, self.adj_matrix.T)

    def __repr__(self) -> str:
        kind = "Directed" if self.directed else "Undirected"
        return (f"{kind} Graph: {self.num_nodes} nodes, "
                f"{len(self.edges)} edges")


# ── Example 1: Social Network (Undirected) ──────────────────────────

def demo_social_network():
    """
    Alice ------- Bob
      \          /
       \        /
        \      /
         Carol
           |
         David
    """
    g = Graph(num_nodes=4, directed=False)
    names = {0: "Alice", 1: "Bob", 2: "Carol", 3: "David"}
    ALICE, BOB, CAROL, DAVID = 0, 1, 2, 3

    g.add_edge(ALICE, BOB)
    g.add_edge(ALICE, CAROL)
    g.add_edge(BOB, CAROL)
    g.add_edge(CAROL, DAVID)

    print(g)
    print(f"\nAdjacency Matrix:\n{g.adj_matrix.astype(int)}")
    print(f"\nSymmetric? {g.is_symmetric()}")
    print(f"\nNeighbors (adjacency list):")
    for node in range(4):
        friends = [names[n] for n in g.get_neighbors(node)]
        print(f"  {names[node]}: {friends}")
    print(f"\nEdge list: {g.edges}")
    return g


# ── Example 2: Twitter Follows (Directed) ───────────────────────────

def demo_twitter_follows():
    """
    Alice --> Bob --> Carol --> David
                ^       |
                +-------+
    """
    g = Graph(num_nodes=4, directed=True)
    ALICE, BOB, CAROL, DAVID = 0, 1, 2, 3

    g.add_edge(ALICE, BOB)     # Alice follows Bob
    g.add_edge(BOB, CAROL)     # Bob follows Carol
    g.add_edge(CAROL, BOB)     # Carol follows Bob back
    g.add_edge(CAROL, DAVID)   # Carol follows David

    print(g)
    print(f"\nAdjacency Matrix:\n{g.adj_matrix.astype(int)}")
    print(f"\nSymmetric? {g.is_symmetric()}")
    print(f"\nAlice follows Bob? {g.has_edge(ALICE, BOB)}")   # True
    print(f"Bob follows Alice? {g.has_edge(BOB, ALICE)}")     # False
    return g


# ── Example 3: Molecular Graph (Weighted) ───────────────────────────

def demo_molecular_graph():
    """
    A simplified molecular graph for water (H2O):
    H -- O -- H  (single bonds, weight=1)

    And ethylene (C2H4):
    H       H
     \     /
      C = C     (C=C double bond, weight=2; C-H single bonds, weight=1)
     /     \\
    H       H
    """
    # Water: 3 atoms (H=0, O=1, H=2)
    water = Graph(num_nodes=3, directed=False)
    water.add_edge(0, 1, weight=1.0)  # H-O single bond
    water.add_edge(1, 2, weight=1.0)  # O-H single bond

    print("Water (H2O):")
    print(f"  Adjacency matrix:\n{water.adj_matrix}")
    print()

    # Ethylene: 6 atoms (C=0, C=1, H=2, H=3, H=4, H=5)
    ethylene = Graph(num_nodes=6, directed=False)
    ethylene.add_edge(0, 1, weight=2.0)  # C=C double bond
    ethylene.add_edge(0, 2, weight=1.0)  # C-H single bond
    ethylene.add_edge(0, 3, weight=1.0)  # C-H single bond
    ethylene.add_edge(1, 4, weight=1.0)  # C-H single bond
    ethylene.add_edge(1, 5, weight=1.0)  # C-H single bond

    print("Ethylene (C2H4):")
    print(f"  Adjacency matrix:\n{ethylene.adj_matrix}")
    print(f"\n  Bond between C0-C1 (double bond): weight={ethylene.adj_matrix[0][1]}")
    print(f"  Bond between C0-H2 (single bond): weight={ethylene.adj_matrix[0][2]}")
    return ethylene


if __name__ == "__main__":
    print("=" * 55)
    print("SOCIAL NETWORK (Undirected Graph)")
    print("=" * 55)
    demo_social_network()

    print("\n" + "=" * 55)
    print("TWITTER FOLLOWS (Directed Graph)")
    print("=" * 55)
    demo_twitter_follows()

    print("\n" + "=" * 55)
    print("MOLECULAR GRAPH (Weighted Graph)")
    print("=" * 55)
    demo_molecular_graph()
```

**Output:**
```
=======================================================
SOCIAL NETWORK (Undirected Graph)
=======================================================
Undirected Graph: 4 nodes, 4 edges

Adjacency Matrix:
[[0 1 1 0]
 [1 0 1 0]
 [1 1 0 1]
 [0 0 1 0]]

Symmetric? True

Neighbors (adjacency list):
  Alice: ['Bob', 'Carol']
  Bob: ['Alice', 'Carol']
  Carol: ['Alice', 'Bob', 'David']
  David: ['Carol']

Edge list: [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (2, 3, 1.0)]

=======================================================
TWITTER FOLLOWS (Directed Graph)
=======================================================
Directed Graph: 4 nodes, 4 edges

Adjacency Matrix:
[[0 1 0 0]
 [0 0 1 0]
 [0 1 0 1]
 [0 0 0 0]]

Symmetric? False

Alice follows Bob? True
Bob follows Alice? False
```

---

## Converting Between Representations

In practice, you will constantly convert between representations. Graph datasets arrive as edge lists (CSV files). You convert to adjacency lists for traversals. You convert to sparse adjacency matrices for GNN training. Let us code all the conversions.

```python
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

# ── Edge List -> Adjacency Matrix ────────────────────────────────────

def edge_list_to_adj_matrix(
    edges: List[Tuple[int, int]],
    num_nodes: int,
    directed: bool = True
) -> np.ndarray:
    """Convert an edge list to a dense adjacency matrix."""
    A = np.zeros((num_nodes, num_nodes), dtype=int)
    for u, v in edges:
        A[u][v] = 1
        if not directed:
            A[v][u] = 1
    return A

# ── Edge List -> Adjacency List ──────────────────────────────────────

def edge_list_to_adj_list(
    edges: List[Tuple[int, int]],
    directed: bool = True
) -> Dict[int, List[int]]:
    """Convert an edge list to an adjacency list."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        if not directed:
            adj[v].append(u)
    return dict(adj)

# ── Adjacency Matrix -> Edge List ────────────────────────────────────

def adj_matrix_to_edge_list(
    A: np.ndarray,
    directed: bool = True
) -> List[Tuple[int, int]]:
    """Convert an adjacency matrix to an edge list."""
    edges = []
    n = A.shape[0]
    for i in range(n):
        start_j = 0 if directed else i + 1  # avoid duplicates for undirected
        for j in range(start_j, n):
            if A[i][j] != 0:
                edges.append((i, j))
    return edges

# ── Adjacency List -> Adjacency Matrix ───────────────────────────────

def adj_list_to_matrix(
    adj_list: Dict[int, List[int]],
    num_nodes: int
) -> np.ndarray:
    """Convert an adjacency list to an adjacency matrix."""
    A = np.zeros((num_nodes, num_nodes), dtype=int)
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            A[node][neighbor] = 1
    return A

# ── Demo ─────────────────────────────────────────────────────────────

edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 0)]

print("Edge list:", edges)
print()

A = edge_list_to_adj_matrix(edges, num_nodes=5, directed=True)
print("Adjacency matrix (directed):")
print(A)
print()

adj = edge_list_to_adj_list(edges, directed=True)
print("Adjacency list:", dict(adj))
print()

# Round-trip: matrix -> edge list -> matrix
edges_back = adj_matrix_to_edge_list(A, directed=True)
print("Edge list recovered from matrix:", edges_back)
print()

# Verify round-trip
A_back = edge_list_to_adj_matrix(edges_back, num_nodes=5, directed=True)
print(f"Round-trip preserves graph? {np.array_equal(A, A_back)}")
```

**Output:**
```
Edge list: [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 0)]

Adjacency matrix (directed):
[[0 1 1 0 0]
 [0 0 1 0 0]
 [0 0 0 1 0]
 [0 0 0 0 1]
 [1 0 0 0 0]]

Adjacency list: {0: [1, 2], 1: [2], 2: [3], 3: [4], 4: [0]}

Edge list recovered from matrix: [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 0)]

Round-trip preserves graph? True
```

---

## Why This Matters for ML: The Adjacency Matrix Is the Engine of GNNs

Now for the payoff. All the formalism above was not just for mathematical elegance — it is the literal input to Graph Neural Networks.

### The GNN Message Passing Equation

The core operation in most GNNs is **message passing** — each node collects information from its neighbors and updates its own representation. Written as a matrix equation:

$$H^{(l+1)} = \sigma\left(\tilde{A} \, H^{(l)} \, W^{(l)}\right)$$

Where:
- $\tilde{A}$ is a normalized version of the adjacency matrix (we will derive the normalization in the next chapter)
- $H^{(l)}$ is the $n \times d$ node feature matrix at layer $l$ ($n$ nodes, each with $d$ features)
- $W^{(l)}$ is a $d \times d'$ learnable weight matrix at layer $l$
- $\sigma$ is a nonlinear activation function (ReLU, etc.)

**Translation**: $\tilde{A} \, H^{(l)}$ multiplies the adjacency matrix by the feature matrix. What does that do? For each node, it sums up the feature vectors of all its neighbors. That is the "message passing" — each node receives messages (features) from its neighbors. Then $W^{(l)}$ applies a learned linear transformation, and $\sigma$ adds nonlinearity.

Let us see this concretely:

```python
import numpy as np

# Our social network adjacency matrix
A = np.array([
    [0, 1, 1, 0],  # Alice: connected to Bob, Carol
    [1, 0, 1, 0],  # Bob: connected to Alice, Carol
    [1, 1, 0, 1],  # Carol: connected to Alice, Bob, David
    [0, 0, 1, 0],  # David: connected to Carol
], dtype=float)

# Node features: each person has 3 features
# [income_level, tech_interest, age_group]
H = np.array([
    [0.8, 0.9, 0.3],  # Alice: high income, high tech, young
    [0.6, 0.7, 0.4],  # Bob: medium income, medium tech, young-ish
    [0.9, 0.5, 0.7],  # Carol: high income, medium tech, older
    [0.3, 0.8, 0.2],  # David: low income, high tech, young
], dtype=float)

# One step of message passing (simplified, no weights or activation):
# Each node's new features = sum of its neighbors' features
H_new = A @ H   # matrix multiplication!

print("Original features H:")
print(H)
print("\nAdjacency matrix A:")
print(A.astype(int))
print("\nAfter message passing (A @ H):")
print(H_new)

# Let's verify manually for Alice (row 0):
# Alice's neighbors are Bob (row 1) and Carol (row 2)
alice_manual = H[1] + H[2]  # sum of Bob's and Carol's features
print(f"\nAlice's new features (manual): {alice_manual}")
print(f"Alice's new features (A @ H):  {H_new[0]}")
print(f"Match? {np.allclose(alice_manual, H_new[0])}")
```

**Output:**
```
Original features H:
[[0.8 0.9 0.3]
 [0.6 0.7 0.4]
 [0.9 0.5 0.7]
 [0.3 0.8 0.2]]

Adjacency matrix A:
[[0 1 1 0]
 [1 0 1 0]
 [1 1 0 1]
 [0 0 1 0]]

After message passing (A @ H):
[[1.5 1.2 1.1]
 [1.7 1.4 1. ]
 [1.7 2.4 0.9]
 [0.9 0.5 0.7]]

Alice's new features (manual): [1.5 1.2 1.1]
Alice's new features (A @ H):  [1.5 1.2 1.1]
Match? True
```

There it is. The adjacency matrix times the feature matrix *is* message passing. Each node's new representation is the sum of its neighbors' representations. That is why the adjacency matrix is the engine of GNNs — it defines who talks to whom.

> **SWE Insight**: Think of message passing as a MapReduce operation. The "map" phase sends each node's features to its neighbors (defined by edges). The "reduce" phase aggregates (sums) the received messages. The adjacency matrix encodes the routing table for this MapReduce. GNN layers stack like middleware — each layer aggregates from a wider neighborhood, just as multiple hops in a network reach farther.

### Knowledge Graphs: Directed Edges as Facts

Knowledge graphs store facts as directed edges, forming triples:

```
  (Subject)  --[Predicate]--> (Object)

  Examples:
  (Einstein)  --[born_in]-->    (Germany)
  (Einstein)  --[field]-->      (Physics)
  (Germany)   --[continent]-->  (Europe)
  (Physics)   --[subfield_of]->(Science)

  As an edge list of triples:
  [("Einstein", "born_in", "Germany"),
   ("Einstein", "field", "Physics"),
   ("Germany", "continent", "Europe"),
   ("Physics", "subfield_of", "Science")]
```

These power Google's Knowledge Graph, Amazon's product ontology, and medical knowledge bases. The ML task: given an incomplete knowledge graph, predict missing edges (link prediction). "Einstein was born in Germany and was a physicist. What continent was he from?" A GNN over the knowledge graph can answer this by propagating information through the graph.

### Recommendation Systems: Bipartite Graphs

User-item interactions form a **bipartite graph** — a graph with two types of nodes where edges only connect nodes of different types:

```
  Users          Items
  ─────          ─────
  Alice ───────> Movie A
  Alice ───────> Movie C
  Bob   ───────> Movie A
  Bob   ───────> Movie B
  Carol ───────> Movie B
  Carol ───────> Movie C

  Adjacency matrix (users x items):

              Movie A  Movie B  Movie C
  Alice    [    1        0        1    ]
  Bob      [    1        1        0    ]
  Carol    [    0        1        1    ]

  This is the user-item interaction matrix.
  Collaborative filtering = finding patterns in this matrix.
  Matrix factorization = decomposing it into user and item embeddings.
```

> **You Already Know This**: If you have used a recommendation system (Netflix, Spotify, Amazon), you have interacted with a bipartite graph. The "customers who bought this also bought" feature is literally a graph traversal: User -> Item -> Other Users who bought Item -> Other Items those users bought.

### Molecular Property Prediction (Our Running Example)

Coming back to our drug discovery running example: given a molecular graph, predict whether the molecule is toxic.

```python
# Simplified molecular graph for aspirin (acetylsalicylic acid)
# We encode atoms as node features and bonds as edges

# Node features: one-hot encoding of atom type
# [C, O, H] (simplified)
atom_features = np.array([
    [1, 0, 0],  # C (carbon)
    [1, 0, 0],  # C
    [1, 0, 0],  # C
    [0, 1, 0],  # O (oxygen)
    [0, 1, 0],  # O
    [1, 0, 0],  # C
    [0, 0, 1],  # H (hydrogen, simplified)
], dtype=float)

# Adjacency matrix (simplified benzene ring + functional groups)
A_mol = np.array([
    [0, 1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
], dtype=float)

# One step of message passing
H_mol = A_mol @ atom_features
print("After one message-passing step on molecular graph:")
print("Each atom now 'knows about' its neighbors' atom types:")
print(H_mol)
print()
print("Node 0 (Carbon) neighbors' types:", H_mol[0])
print("  -> 1 Carbon neighbor, 1 Oxygen neighbor, 0 Hydrogen neighbors")
```

After multiple message-passing steps, each atom's representation encodes information from its local chemical neighborhood — exactly the information a chemist uses to predict molecular properties. This is the foundation of models like SchNet, DimeNet, and the GNN architectures behind AlphaFold.

---

## When to Use Graphs (and When Not To)

### Use Graphs When:

- **Data has natural relationships**: Social networks, molecules, citation networks, road networks, communication networks
- **Relationships carry signal**: Who you are connected to predicts your behavior better than your features alone
- **Data is non-Euclidean**: It does not fit neatly into a grid (images) or a sequence (text) — it has irregular connectivity
- **You need to propagate information**: Influence spreading, infection modeling, recommendation propagation
- **Structure varies across samples**: Each molecule has a different graph topology — you cannot use a fixed-size input layer

### Do Not Use Graphs When:

- **Data is truly independent**: Standard tabular data where rows have no meaningful relationships
- **Relationships are fully captured by features**: If "number of friends" is sufficient, you do not need the full graph
- **Your graph is too dense**: If everything is connected to everything, the graph does not add structure — you are better off with standard attention mechanisms
- **Simpler methods work**: A well-tuned XGBoost on tabular features often beats a GNN. Do not add graph complexity unless you have evidence it helps.

> **Common Mistake**: Forcing graph structure where none exists. If you have to artificially construct edges (e.g., "connect all users who share a zip code"), your graph is likely not adding real information — it is adding noise. The edges should represent genuine, meaningful relationships.

---

## Exercises

### Exercise 1: Edge List to Adjacency Matrix

**Problem**: Given this directed graph as an edge list, construct the adjacency matrix by hand, then verify with code.

```
edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 0)]
```

Hint: draw the graph first.

```
  0 --> 1
  |     |
  v     v
  2 --> 3 --> 4
  ^                |
  |________________|
```

**Solution**:

```python
import numpy as np

def edge_list_to_adj_matrix(edges, num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=int)
    for u, v in edges:
        A[u][v] = 1
    return A

edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 0)]
A = edge_list_to_adj_matrix(edges, 5)
print(A)
# Output:
# [[0 1 1 0 0]
#  [0 0 1 0 0]
#  [0 0 0 1 0]
#  [0 0 0 0 1]
#  [1 0 0 0 0]]
#
# Verify: row 0 has 1s at columns 1 and 2 (edges 0->1 and 0->2). Correct.
# Row 4 has a 1 at column 0 (edge 4->0). Correct.
# The graph has a cycle: 0->1->2->3->4->0.
```

### Exercise 2: Symmetry Check (Is It Undirected?)

**Problem**: Write a function that checks whether a graph is undirected by verifying that its adjacency matrix is symmetric. Test it on both an undirected and a directed graph.

**Solution**:

```python
import numpy as np

def is_undirected(adj_matrix: np.ndarray) -> bool:
    """Check if a graph is undirected by verifying A = A^T."""
    return np.allclose(adj_matrix, adj_matrix.T)

# Test with undirected graph (friendship)
undirected = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])
print(f"Undirected graph symmetric? {is_undirected(undirected)}")  # True

# Test with directed graph (follows)
directed = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])
print(f"Directed graph symmetric? {is_undirected(directed)}")  # False

# Edge case: weighted undirected graph
weighted_undirected = np.array([
    [0.0, 2.5, 0.0],
    [2.5, 0.0, 1.3],
    [0.0, 1.3, 0.0]
])
print(f"Weighted undirected symmetric? {is_undirected(weighted_undirected)}")  # True
```

**Why this matters**: In GNN libraries, some operations assume undirected graphs (symmetric normalization of the adjacency matrix). If you accidentally pass a directed graph, the results will be wrong but the code will not crash — a silent bug. Always verify.

### Exercise 3: Representation Conversion Round-Trip

**Problem**: Start with an adjacency list, convert it to an adjacency matrix, then convert back to an adjacency list. Verify the round-trip preserves the graph.

**Solution**:

```python
import numpy as np
from collections import defaultdict

def adj_list_to_matrix(adj_list, num_nodes):
    """Convert adjacency list to adjacency matrix."""
    A = np.zeros((num_nodes, num_nodes), dtype=int)
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            A[node][neighbor] = 1
    return A

def matrix_to_adj_list(A):
    """Convert adjacency matrix to adjacency list."""
    adj = defaultdict(list)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if A[i][j] != 0:
                adj[i].append(j)
    return dict(adj)

# Original adjacency list
original = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1, 3],
    3: [2]
}

# Convert to matrix
A = adj_list_to_matrix(original, 4)
print("Adjacency matrix:")
print(A)

# Convert back to adjacency list
recovered = matrix_to_adj_list(A)
print(f"\nOriginal adj list:  {original}")
print(f"Recovered adj list: {recovered}")

# Verify: same neighbors for each node
for node in original:
    assert sorted(original[node]) == sorted(recovered[node]), \
        f"Mismatch at node {node}"
print("\nRound-trip verified: all neighbors match.")
```

### Exercise 4: Message Passing by Hand (Challenge)

**Problem**: Given this graph and node features, compute one step of message passing ($H' = A \cdot H$) by hand, then verify with NumPy.

```
Graph (undirected):           Node features:
  0 --- 1                     Node 0: [1, 0]
  |     |                     Node 1: [0, 1]
  2 --- 3                     Node 2: [1, 1]
                              Node 3: [0, 0]
Edges: {0,1}, {0,2}, {1,3}, {2,3}
```

**Solution**:

```python
import numpy as np

A = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
], dtype=float)

H = np.array([
    [1, 0],  # Node 0
    [0, 1],  # Node 1
    [1, 1],  # Node 2
    [0, 0],  # Node 3
], dtype=float)

# By hand for Node 0:
# Neighbors: Node 1 and Node 2
# New features = H[1] + H[2] = [0,1] + [1,1] = [1, 2]

# By hand for Node 3:
# Neighbors: Node 1 and Node 2
# New features = H[1] + H[2] = [0,1] + [1,1] = [1, 2]

H_new = A @ H
print("Message passing result (A @ H):")
print(H_new)
# [[1. 2.]    <- Node 0: sum of neighbors 1 and 2
#  [1. 0.]    <- Node 1: sum of neighbors 0 and 3
#  [1. 0.]    <- Node 2: sum of neighbors 0 and 3
#  [1. 2.]]   <- Node 3: sum of neighbors 1 and 2

# Notice: Nodes 0 and 3 have the same neighbors, so they get the same
# new features. Same for Nodes 1 and 2. This is a property of the graph
# structure — structurally equivalent nodes get equivalent representations.
```

**Insight**: After message passing, structurally equivalent nodes (same neighbors) get identical representations. This is both a feature and a limitation of basic GNNs — it is why more expressive architectures (like GIN, or higher-order GNNs) were developed.

---

## Summary

Here is what we covered and why each piece matters:

| Concept | What It Is | Why It Matters for ML |
|---|---|---|
| Graph $G = (V, E)$ | Nodes + edges | The fundamental structure for relational data |
| Undirected graph | Edges are symmetric: $\{u,v\} = \{v,u\}$ | Social networks, molecular bonds, similarity graphs |
| Directed graph | Edges have direction: $(u,v) \neq (v,u)$ | Knowledge graphs, citation networks, causal models |
| Weighted graph | Edges carry values: $w: E \to \mathbb{R}$ | Attention scores, distances, similarity strengths |
| Adjacency matrix | $n \times n$ matrix encoding connectivity | The input to GNNs; enables message passing via $A \cdot H$ |
| Adjacency list | Dict mapping nodes to neighbor lists | Efficient traversals; what NetworkX uses internally |
| Edge list | List of (source, target) pairs | Storage format; what PyTorch Geometric uses |
| Message passing | $H^{(l+1)} = \sigma(\tilde{A} H^{(l)} W^{(l)})$ | The core GNN operation — neighbors share information |

**The big idea**: A graph is a structure that encodes relationships. The adjacency matrix converts that structure into a matrix. And once you have a matrix, all of linear algebra — the tools you learned in earlier chapters — applies. That is the bridge from graph theory to graph-based ML.

---

## What's Next

**Chapter 2: Graph Properties** — Now that you can represent graphs as matrices, what mathematical properties make those graphs useful for ML? We will cover degree (how connected a node is), paths (how information flows), cycles (where feedback loops form), and connectivity (whether your graph has isolated components). These properties become node features, normalization factors, and structural priors that make GNNs work in practice.

Specifically, you will learn why the degree matrix $D$ appears in the GNN equation (it is $\tilde{A}$ — the normalized adjacency matrix), why shortest paths matter for positional encodings, and how connected components determine whether your GNN can even reach all nodes.
