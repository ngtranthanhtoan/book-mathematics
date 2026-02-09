# Chapter 3: Graph Algorithms

> **Building On**: In Chapters 1 and 2, we learned what graphs are and how to measure their properties -- degrees, paths, connectivity, cycles. We know how to *describe* a graph. But knowing that your social network has 2 billion nodes and an average degree of 338 does not help you recommend friends, detect fraud, or rank search results. For that, you need algorithms that *traverse*, *rank*, and *learn from* graph structure. That is what this chapter is about.

---

## The Problem That Launched a Trillion-Dollar Company

Here is a real problem that shaped the internet. It is 1998, and you have crawled 26 million web pages. A user types "Stanford University" into your search engine. Hundreds of thousands of pages mention "Stanford." Which ten do you show first?

The naive approach: rank by keyword frequency. Pages that mention "Stanford" the most go first. You try this. The top result is a spam page that repeats "Stanford Stanford Stanford" a thousand times. Your second result is a random student's homepage because it mentions "Stanford" in every paragraph.

Your competitor -- some guys named Larry and Sergey -- tries a completely different approach. They ignore *what pages say* and instead look at *how pages link to each other*. Their insight: a page is important if important pages link to it. This circular definition turns out to have an elegant mathematical solution. They call it PageRank. Google is born.

That algorithm, and the graph traversal algorithms that support it, are what you will learn in this chapter. And here is what might surprise you: you have already implemented most of them. BFS, DFS, topological sort, Dijkstra -- these are not new to you. What IS new is seeing why these same algorithms are the computational backbone of modern graph-based ML: Graph Neural Networks, node embeddings (node2vec, DeepWalk), spectral clustering, and recommendation systems.

Let us start with what you know and build toward what you don't.

---

## BFS and DFS: You Already Implement These

> **You Already Know This**: If you have ever implemented a web crawler, a file system walker, level-order traversal of a binary tree, or even a simple flood-fill in a game -- you have written BFS or DFS. The queue-vs-stack distinction you learned in your algorithms course is *exactly* the same here. The only new angle: in ML, *how* you traverse a graph determines what your model can learn.

### The Algorithms You Know

Let me draw the graph we will work with throughout this section:

```
        ┌───┐
        │ A │
        └─┬─┘
         ╱ ╲
        ╱   ╲
   ┌───┐     ┌───┐
   │ B │     │ C │
   └─┬─┘     └─┬─┘
    ╱ ╲         ╲
   ╱   ╲         ╲
┌───┐ ┌───┐   ┌───┐
│ D │ │ E │   │ F │
└───┘ └───┘   └───┘
```

**BFS explores level by level** (you use a queue):

```
Visit order:  A → B → C → D → E → F

Level 0:       [A]              ← dequeue A, enqueue B, C
Level 1:       [B] [C]          ← dequeue B, enqueue D, E; dequeue C, enqueue F
Level 2:       [D] [E] [F]     ← dequeue D, E, F (no new neighbors)

Queue trace:
  Start:       [A]
  After A:     [B, C]
  After B:     [C, D, E]
  After C:     [D, E, F]
  After D:     [E, F]
  After E:     [F]
  After F:     []  ← done
```

**DFS explores depth-first** (you use a stack, or recursion):

```
Visit order:  A → B → D → E → C → F

Stack trace (iterative):
  Start:       [A]
  Pop A:       [C, B]         ← push children right-to-left
  Pop B:       [C, E, D]
  Pop D:       [C, E]         ← D has no children
  Pop E:       [C]            ← E has no children
  Pop C:       [F]
  Pop F:       []             ← done

Or think of it as a recursive call stack:
  DFS(A)
    DFS(B)
      DFS(D) ← dead end, backtrack
      DFS(E) ← dead end, backtrack
    DFS(C)
      DFS(F) ← dead end, backtrack
```

Both are $O(V + E)$ time and you have written them dozens of times. Here is a clean implementation:

```python
import numpy as np
from collections import deque, defaultdict

def bfs(adj_list, start):
    """BFS traversal -- returns visit order and distances from start."""
    visited = {start}
    queue = deque([start])
    order = []
    distances = {start: 0}

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    return order, distances

def dfs(adj_list, start):
    """DFS traversal -- returns visit order."""
    visited = set()
    order = []
    stack = [start]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            # Push in reverse so we visit in natural order
            for neighbor in reversed(adj_list[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order

# Demo with the graph from the ASCII diagram above
adj = defaultdict(list, {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [], 'E': [], 'F': []
})

print("BFS:", bfs(adj, 'A'))
# BFS: (['A', 'B', 'C', 'D', 'E', 'F'], {'A': 0, 'B': 1, 'C': 1, 'D': 2, 'E': 2, 'F': 2})

print("DFS:", dfs(adj, 'A'))
# DFS: ['A', 'B', 'D', 'E', 'C', 'F']
```

No surprises so far. Now here is where it gets interesting for ML.

### The ML Twist: Why BFS and DFS Matter for Graph Learning

Consider this question: you have a social network with 2 billion nodes. You want to learn a vector representation (embedding) for each user. You cannot feed the entire graph into a neural network. How do you sample the "neighborhood" around a node?

**BFS-style sampling** captures your *local community*. If you BFS out 2 hops from a user, you get their friends and friends-of-friends. This tells the model about the user's immediate social circle. This is exactly what **GraphSAGE** does -- it samples a fixed number of neighbors at each hop via BFS to build mini-batches for training.

**DFS-style sampling** captures your *structural role*. A random walk that goes deep (DFS-like) might wander far from the starting node, visiting nodes that are structurally similar (e.g., other "bridge" nodes between communities) even if they are not nearby. This is the insight behind **node2vec** -- it uses parameterized random walks that interpolate between BFS-like and DFS-like exploration.

```
BFS-like (local, homophily):       DFS-like (global, structural):

    ╭─── You ───╮                     You
    │     │     │                      │
  Friend Friend Friend               Friend
    │     │     │                      │
  FoF   FoF   FoF                   FoF
                                      │
  "Who is in your circle?"          Stranger in another community
                                      │
                                    Their friend
                                    "Who plays the same role as you?"
```

Here is a simplified node2vec-style random walk:

```python
import random

def biased_random_walk(adj_list, start, walk_length, p=1.0, q=1.0):
    """
    node2vec-style random walk.

    p controls likelihood of returning to previous node (low p = BFS-like).
    q controls likelihood of exploring outward (low q = DFS-like).

    p=1, q=1: unbiased random walk (DeepWalk)
    p=1, q=0.5: biased toward DFS (explore outward)
    p=0.5, q=1: biased toward BFS (stay local)
    """
    walk = [start]
    for _ in range(walk_length - 1):
        current = walk[-1]
        neighbors = adj_list[current]
        if not neighbors:
            break

        if len(walk) == 1:
            walk.append(random.choice(neighbors))
            continue

        prev = walk[-2]
        # Compute unnormalized transition probabilities
        weights = []
        for neighbor in neighbors:
            if neighbor == prev:
                weights.append(1.0 / p)  # Return to previous
            elif neighbor in adj_list[prev]:
                weights.append(1.0)      # Neighbor of previous (BFS-like)
            else:
                weights.append(1.0 / q)  # Move further away (DFS-like)

        # Normalize and sample
        total = sum(weights)
        probs = [w / total for w in weights]
        walk.append(random.choices(neighbors, weights=probs, k=1)[0])

    return walk

# Example: social network
social = defaultdict(list, {
    0: [1, 2, 3],    # Person 0 knows 1, 2, 3
    1: [0, 2, 4],    # Person 1 knows 0, 2, 4
    2: [0, 1, 5],    # Person 2 knows 0, 1, 5
    3: [0, 6],       # Person 3 knows 0, 6
    4: [1, 7],       # Person 4 knows 1, 7
    5: [2, 7],       # Person 5 knows 2, 7
    6: [3, 8],       # Person 6 knows 3, 8
    7: [4, 5, 8],    # Person 7 (bridge node)
    8: [6, 7]        # Person 8
})

random.seed(42)
print("BFS-like walk (p=0.5, q=2):", biased_random_walk(social, 0, 8, p=0.5, q=2))
print("DFS-like walk (p=2, q=0.5):", biased_random_walk(social, 0, 8, p=2, q=0.5))
# BFS-like tends to stay near node 0's neighborhood
# DFS-like tends to wander further into the graph
```

**Translation**: BFS and DFS are not just interview questions anymore. They are the sampling strategies that determine what information your graph ML model can learn. BFS gives you homophily (similar neighbors have similar labels). DFS gives you structural equivalence (nodes with similar roles get similar embeddings).

> **Common Mistake**: Thinking that BFS/DFS in ML means exactly the classical algorithms. In practice, you almost always use *randomized* versions -- random walks, neighbor sampling with replacement, importance sampling. The classical algorithms give you the conceptual framework; the randomized variants make things scalable to billions of nodes.

---

## Topological Sort: Your Build System, Formalized

> **You Already Know This**: If you have ever used `make`, `gradle`, `webpack`, `pip install`, `terraform apply`, or `docker-compose` -- you have relied on topological sort. These tools resolve dependency order: "build library A before service B, because B depends on A." Topological sort is the algorithm that computes this ordering. In ML, it shows up in computational graph execution (TensorFlow, PyTorch autograd) and DAG-based model architectures.

A topological sort of a Directed Acyclic Graph (DAG) produces a linear ordering of vertices such that for every directed edge $(u, v)$, vertex $u$ comes before $v$.

```
  Dependency graph (build system):        One valid topological order:

  ┌──────────┐    ┌──────────┐            1. libcrypto
  │libcrypto │───→│ libauth  │            2. libauth
  └────┬─────┘    └────┬─────┘            3. libhttp
       │               │                  4. api-server
       ▼               ▼                  5. frontend
  ┌──────────┐    ┌──────────┐            6. tests
  │ libhttp  │───→│api-server│
  └──────────┘    └────┬─────┘
                       │
                  ┌────▼─────┐
                  │ frontend │
                  └────┬─────┘
                       │
                  ┌────▼─────┐
                  │  tests   │
                  └──────────┘
```

The algorithm (Kahn's algorithm) is essentially BFS on in-degree-zero nodes:

1. Find all nodes with in-degree 0 (no dependencies). These can be built first.
2. "Build" them (add to output), then decrement the in-degree of everything they point to.
3. Repeat until all nodes are processed. If you cannot process all nodes, there is a cycle (circular dependency).

```python
from collections import deque, defaultdict

def topological_sort(adj_list, all_nodes):
    """
    Kahn's algorithm for topological sort.
    Returns sorted order, or None if a cycle exists.

    This is exactly what 'make' does to figure out build order.
    """
    # Compute in-degrees
    in_degree = defaultdict(int)
    for node in all_nodes:
        in_degree[node]  # ensure every node appears
    for node in all_nodes:
        for neighbor in adj_list[node]:
            in_degree[neighbor] += 1

    # Start with nodes that have no dependencies
    queue = deque([n for n in all_nodes if in_degree[n] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in adj_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(all_nodes):
        return None  # Cycle detected -- like a circular import!

    return result

# Build system example
build_deps = defaultdict(list, {
    'libcrypto': ['libauth', 'libhttp'],
    'libauth': ['api-server'],
    'libhttp': ['api-server'],
    'api-server': ['frontend'],
    'frontend': ['tests'],
    'tests': []
})
all_packages = ['libcrypto', 'libauth', 'libhttp', 'api-server', 'frontend', 'tests']

print("Build order:", topological_sort(build_deps, all_packages))
# Build order: ['libcrypto', 'libauth', 'libhttp', 'api-server', 'frontend', 'tests']
```

**The ML Connection**: When PyTorch executes `loss.backward()`, it is doing a topological sort of the computational graph in reverse. Each operation node computes its gradient only after all downstream gradients are ready. That is backpropagation -- topological sort on an autograd DAG.

**Complexity**: $O(V + E)$ -- same as BFS, because it essentially IS BFS with in-degree tracking.

---

## Shortest Paths: From Load Balancers to Knowledge Graphs

> **You Already Know This**: If you have worked with network routing (OSPF, BGP), CDN edge selection, or even Google Maps -- you know shortest path algorithms. Dijkstra's algorithm in your load balancer picks the server with the lowest latency. The ML twist: shortest path distances become *features* for link prediction and graph kernels.

### BFS for Unweighted Shortest Paths

In an unweighted graph, BFS naturally finds shortest paths (in terms of hop count). You already saw this above. The key property:

> When BFS visits node $v$ via node $u$, the distance $\text{dist}[v] = \text{dist}[u] + 1$ is guaranteed to be the shortest path from the source to $v$.

Why? Because BFS explores in order of distance. By the time it reaches $v$, it has already explored all nodes at closer distances. This is a greedy invariant -- no node at distance $k$ is discovered before all nodes at distance $k-1$ are processed.

### Dijkstra's Algorithm for Weighted Shortest Paths

When edges have weights (e.g., latency between servers, similarity scores between entities), you need Dijkstra. The intuition is the same as BFS, but instead of a FIFO queue, you use a priority queue (min-heap) ordered by cumulative distance.

```
  Weighted graph:                        Dijkstra from A:

  ┌───┐ ──2── ┌───┐ ──1── ┌───┐        Step 1: A=0, B=inf, C=inf, D=inf, E=inf
  │ A │       │ B │       │ D │        Step 2: Visit A → B=2, C=5
  └───┘       └───┘       └───┘        Step 3: Visit B → D=3, C=min(5,4)=4
     \          │            │          Step 4: Visit D → E=4
    5 \         3            1          Step 5: Visit C → (no improvement)
       \        │            │          Step 6: Visit E → done
      ┌───┐   ┌───┐
      │ C │───│ E │                    Shortest: A→B→D→E (cost=4)
      └───┘ 2 └───┘                    Not:      A→C→E   (cost=7)
                                        Not:      A→B→E   (cost=5)
```

```python
import heapq
from collections import defaultdict

def dijkstra(adj_list_weighted, start):
    """
    Dijkstra's shortest path -- the same algorithm your load balancer uses.

    adj_list_weighted: {node: [(neighbor, weight), ...]}
    Returns: distances dict, predecessors dict
    """
    distances = {start: 0}
    predecessors = {start: None}
    pq = [(0, start)]  # (distance, node)
    visited = set()

    while pq:
        dist, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        for neighbor, weight in adj_list_weighted[node]:
            new_dist = dist + weight
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    return distances, predecessors

def reconstruct_path(predecessors, target):
    """Trace back from target to source."""
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = predecessors[current]
    return path[::-1]

# Example: server latency graph
network = defaultdict(list, {
    'A': [('B', 2), ('C', 5)],
    'B': [('A', 2), ('D', 1), ('C', 2), ('E', 3)],
    'C': [('A', 5), ('B', 2), ('E', 2)],
    'D': [('B', 1), ('E', 1)],
    'E': [('B', 3), ('C', 2), ('D', 1)]
})

distances, preds = dijkstra(network, 'A')
print("Shortest distances from A:", distances)
# {'A': 0, 'B': 2, 'D': 3, 'C': 4, 'E': 4}

print("Shortest path A → E:", reconstruct_path(preds, 'E'))
# ['A', 'B', 'D', 'E']
```

**Complexity**: $O((V + E) \log V)$ with a binary heap.

### ML Application: Shortest Path Features for Link Prediction

In knowledge graph completion (predicting missing links), shortest path distances are powerful features. If two entities are close in the graph, they are more likely to be related:

```python
def shortest_path_features(adj_list, node_pairs):
    """
    Compute shortest path distance as a feature for link prediction.
    Used in knowledge graph completion and social network analysis.
    """
    features = []
    for u, v in node_pairs:
        _, distances = bfs(adj_list, u)
        sp_dist = distances.get(v, float('inf'))
        features.append(sp_dist)
    return features
```

> **Common Mistake**: Using Dijkstra when BFS suffices. If your graph is unweighted (like a social network where edges are binary "follows/doesn't follow"), BFS gives you shortest paths in $O(V + E)$. Dijkstra's heap overhead makes it $O((V + E) \log V)$ for no benefit. Know your graph.

---

## Minimum Spanning Trees: Network Optimization You Have Done Before

> **You Already Know This**: If you have ever designed a network topology -- connecting data centers with minimum total cable cost, or building a backbone network between offices -- you have solved the minimum spanning tree problem. Kruskal's and Prim's algorithms select the cheapest set of edges that keeps the entire graph connected.

A **spanning tree** of a connected graph is a subgraph that includes all vertices, is connected, and has no cycles. A **minimum spanning tree (MST)** is the spanning tree with the smallest total edge weight.

```
  Original graph:                      MST (total weight = 7):

  ┌───┐ ──2── ┌───┐                  ┌───┐ ──2── ┌───┐
  │ A │       │ B │                  │ A │       │ B │
  └───┘       └───┘                  └───┘       └───┘
   │ ╲         │ ╲                              │
   4   5       3   6                             3
   │     ╲     │     ╲                           │
  ┌───┐   ┌───┐     ┌───┐                ┌───┐ ┌───┐
  │ C │──1│ D │     │ E │                │ C │─1│ D │──1──┌───┐
  └───┘   └───┘     └───┘                └───┘  └───┘     │ E │
            │                                              └───┘
            1
            │
          ┌───┐
          │ E │
          └───┘

  Edges removed: A-C(4), A-D(5), B-E(6)       MST weight: 2+3+1+1 = 7
  Edges kept:    A-B(2), B-D(3), C-D(1), D-E(1)
```

**Kruskal's Algorithm** (greedy, edge-centric):
1. Sort all edges by weight.
2. For each edge in sorted order: add it to the MST if it does not create a cycle (use Union-Find to check).

```python
class UnionFind:
    """The same Union-Find you use for connected components."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already connected -- adding edge would create cycle
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def kruskal_mst(num_nodes, edges):
    """
    Kruskal's MST -- sort edges, greedily add if no cycle.
    edges: list of (weight, u, v)
    Returns: list of MST edges and total weight
    """
    edges_sorted = sorted(edges)  # Sort by weight
    uf = UnionFind(num_nodes)
    mst = []
    total_weight = 0

    for weight, u, v in edges_sorted:
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight
            if len(mst) == num_nodes - 1:
                break  # MST has exactly V-1 edges

    return mst, total_weight

# Data center connectivity example
# Nodes: 0=NYC, 1=London, 2=Tokyo, 3=Sydney, 4=SaoPaulo
edges = [
    (10, 0, 1),  # NYC-London
    (25, 0, 2),  # NYC-Tokyo
    (30, 0, 3),  # NYC-Sydney
    (15, 0, 4),  # NYC-SaoPaulo
    (20, 1, 2),  # London-Tokyo
    (35, 1, 3),  # London-Sydney
    (22, 2, 3),  # Tokyo-Sydney
    (28, 3, 4),  # Sydney-SaoPaulo
]

mst, cost = kruskal_mst(5, edges)
print(f"MST edges: {mst}")
print(f"Total cable cost: {cost}")
# Connects all 5 data centers with minimum total cost
```

**Complexity**: $O(E \log E)$ for sorting edges. Union-Find operations are nearly $O(1)$ amortized (inverse Ackermann function).

**ML Applications**:
- **Clustering**: MST-based clustering removes the longest edges in the MST to split the graph into clusters. This is a non-parametric alternative to k-means that respects graph structure.
- **Feature engineering**: MST edge weights summarize the "skeleton" of a point cloud, useful in geometric deep learning.
- **Approximation algorithms**: MST provides a 2-approximation for the Traveling Salesman Problem, relevant in routing optimization.

---

## PageRank: The Algorithm That Built Google

Now we get to the big one. Let me walk you through it the way I wish someone had explained it to me.

### The Problem (Revisited)

You have a graph of web pages. Each page links to other pages. You want to compute a single number for each page: its "importance." More links pointing to a page should mean higher importance, but not all links are equal -- a link from the New York Times should count more than a link from a random blog.

This creates a circular dependency: to know how important page A is, you need to know how important the pages linking to A are. But to know THEIR importance, you need to know the importance of pages linking to THEM. And so on.

> **You Already Know This**: This is the same problem as computing "influence" in a microservice dependency graph. If service A calls service B, and service B calls service C, how "critical" is each service? The answer depends on how critical the services that depend on it are. You have probably seen this framed as "blast radius analysis" in incident response.

### The Random Surfer Model

PageRank resolves the circular dependency with a beautiful probabilistic model. Imagine a person randomly browsing the web:

1. They start on a random page.
2. At each step, with probability $d = 0.85$, they click a random outgoing link on the current page.
3. With probability $1 - d = 0.15$, they get bored, close the tab, and jump to a completely random page.

The **PageRank** of a page is the fraction of time the random surfer spends on it, in the long run. Pages that many important pages link to will be visited more often.

```
  A random surfer's journey:

  ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
  │Blog │────→│ NYT │────→│Wiki │────→│ Gov │
  │ (B) │     │ (N) │     │ (W) │     │ (G) │
  └──┬──┘     └──┬──┘     └──┬──┘     └─────┘
     │           │           │           ▲
     │           ▼           │           │
     │        ┌─────┐       │           │
     └───────→│Forum│───────┘           │
              │ (F) │──────────────────-┘
              └─────┘

  After many steps, the surfer visits:
    NYT   ≈ 28%    ← many pages link here, high rank
    Wiki  ≈ 24%    ← linked by NYT and Forum
    Gov   ≈ 18%    ← linked by Wiki and Forum
    Forum ≈ 18%    ← linked by Blog and NYT
    Blog  ≈ 12%    ← only random jumps bring you here
```

### The Math

The PageRank of node $v$ is:

$$PR(v) = \frac{1 - d}{N} + d \sum_{u \in B(v)} \frac{PR(u)}{L(u)}$$

**Translation**: The rank of page $v$ equals a baseline of $\frac{1-d}{N}$ (the probability the surfer randomly jumps here) plus $d$ times the sum of contributions from all pages $u$ that link to $v$. Each contributing page $u$ gives a share of $\frac{PR(u)}{L(u)}$ -- its own rank divided evenly among all its outgoing links.

Where:
- $d$ = damping factor (typically 0.85) -- probability of following a link
- $N$ = total number of nodes
- $B(v)$ = set of nodes that link TO $v$ (backlinks)
- $L(u)$ = number of outgoing links FROM $u$ (out-degree)

### Matrix Formulation

We can express the entire system as a matrix equation. Define the transition matrix $M$ where:

$$M_{ij} = \begin{cases} \frac{1}{L(j)} & \text{if page } j \text{ links to page } i \\ 0 & \text{otherwise} \end{cases}$$

**Translation**: Column $j$ of $M$ represents how page $j$ distributes its rank among the pages it links to. If page $j$ has 4 outgoing links, each gets $\frac{1}{4}$ of $j$'s rank.

The PageRank vector $\mathbf{r}$ satisfies:

$$\mathbf{r} = \frac{1 - d}{N} \mathbf{1} + d \cdot M \mathbf{r}$$

This is a fixed-point equation. The PageRank vector is the eigenvector of the modified transition matrix corresponding to eigenvalue 1.

> **You Already Know This (kind of)**: This is the same as solving for the steady state of a Markov chain. If you have worked with queuing theory, load balancer steady-state analysis, or even Markov chain Monte Carlo (MCMC) -- you have seen this pattern: iterate until the distribution stops changing.

### Power Iteration: How to Actually Compute It

You do not solve the matrix equation directly. Instead, you iterate:

1. Start with uniform ranks: $\mathbf{r}^{(0)} = \frac{1}{N}\mathbf{1}$
2. Repeat: $\mathbf{r}^{(t+1)} = \frac{1-d}{N}\mathbf{1} + d \cdot M \mathbf{r}^{(t)}$
3. Stop when $\|\mathbf{r}^{(t+1)} - \mathbf{r}^{(t)}\|_1 < \epsilon$

```
  Power iteration convergence:

  Iteration 0:  [0.20, 0.20, 0.20, 0.20, 0.20]   ← uniform start
  Iteration 1:  [0.12, 0.28, 0.22, 0.20, 0.18]   ← NYT pulls ahead
  Iteration 2:  [0.11, 0.27, 0.24, 0.19, 0.19]   ← settling
  Iteration 3:  [0.12, 0.28, 0.24, 0.18, 0.18]   ← nearly converged
  ...
  Iteration 28: [0.12, 0.28, 0.24, 0.18, 0.18]   ← converged!
                  Blog   NYT  Wiki   Gov  Forum
```

This converges because the damping factor guarantees the matrix is irreducible and aperiodic (a "regular" Markov chain). The Perron-Frobenius theorem guarantees a unique stationary distribution exists.

```python
import numpy as np

def pagerank(adj_matrix, damping=0.85, max_iter=100, tol=1e-6):
    """
    PageRank via power iteration.

    Think of this as: "If a random person clicks links forever,
    what fraction of time do they spend on each page?"
    """
    n = len(adj_matrix)
    A = np.array(adj_matrix, dtype=float)

    # Build column-stochastic transition matrix
    out_degree = A.sum(axis=1)
    M = np.zeros((n, n))
    for j in range(n):
        if out_degree[j] == 0:
            # Dangling node: surfer jumps to random page
            M[:, j] = 1.0 / n
        else:
            M[:, j] = A[j, :] / out_degree[j]

    # Power iteration
    r = np.ones(n) / n  # Start uniform

    for i in range(max_iter):
        r_new = (1 - damping) / n + damping * M @ r

        if np.sum(np.abs(r_new - r)) < tol:
            print(f"  Converged after {i + 1} iterations")
            return r_new
        r = r_new

    print(f"  Warning: did not converge after {max_iter} iterations")
    return r

# The web graph from our ASCII diagram
# Nodes: 0=Blog, 1=NYT, 2=Wiki, 3=Gov, 4=Forum
web = np.array([
    [0, 0, 0, 0, 0],  # Blog links to: NYT, Forum
    [1, 0, 0, 0, 0],  # NYT links to: Wiki, Forum
    [0, 1, 0, 0, 0],  # Wiki links to: Gov
    [0, 0, 1, 0, 1],  # Gov: linked by Wiki, Forum
    [1, 1, 1, 0, 0],  # Forum links to: Wiki, Gov
])
# Correcting: adjacency matrix where A[i][j]=1 means i links to j
web = np.array([
    [0, 1, 0, 0, 1],  # Blog → NYT, Forum
    [0, 0, 1, 0, 1],  # NYT → Wiki, Forum
    [0, 0, 0, 1, 0],  # Wiki → Gov
    [0, 0, 0, 0, 0],  # Gov → (dangling)
    [0, 0, 1, 1, 0],  # Forum → Wiki, Gov
])

ranks = pagerank(web)
labels = ['Blog', 'NYT', 'Wiki', 'Gov', 'Forum']
print("\nPageRank scores:")
for label, rank in sorted(zip(labels, ranks), key=lambda x: -x[1]):
    print(f"  {label:>8s}: {rank:.4f}  {'█' * int(rank * 100)}")
```

### Personalized PageRank for Recommendations

Here is where PageRank becomes directly useful in modern ML. Standard PageRank uses a uniform random jump distribution -- the bored surfer jumps to ANY page with equal probability. **Personalized PageRank (PPR)** changes this: the surfer always jumps back to a specific node (or set of nodes).

$$PPR(v; s) = \frac{1 - d}{N} \cdot \mathbf{e}_s + d \sum_{u \in B(v)} \frac{PPR(u; s)}{L(u)}$$

Where $\mathbf{e}_s$ is a one-hot vector for the "seed" node $s$. This computes personalized importance *relative to node $s$*.

**Translation**: Instead of "where does a random surfer end up?", PPR answers "where does a random surfer *who keeps coming back to node $s$* end up?" Nodes near $s$ in the graph get higher personalized rank.

This is used directly in:
- **Pinterest (PinSage)**: PPR from a user node ranks which pins to recommend
- **Twitter**: PPR from your account ranks which tweets to show in your feed
- **Knowledge graphs**: PPR from an entity finds related entities

```python
def personalized_pagerank(adj_matrix, seed_node, damping=0.85, max_iter=100, tol=1e-6):
    """
    Personalized PageRank: importance relative to a specific node.
    Used by Pinterest, Twitter, and many recommendation systems.
    """
    n = len(adj_matrix)
    A = np.array(adj_matrix, dtype=float)

    out_degree = A.sum(axis=1)
    M = np.zeros((n, n))
    for j in range(n):
        if out_degree[j] == 0:
            M[:, j] = 1.0 / n
        else:
            M[:, j] = A[j, :] / out_degree[j]

    # Personalization vector: always jump back to seed
    e = np.zeros(n)
    e[seed_node] = 1.0

    r = np.ones(n) / n
    for i in range(max_iter):
        r_new = (1 - damping) * e + damping * M @ r
        if np.sum(np.abs(r_new - r)) < tol:
            return r_new
        r = r_new

    return r

# "Recommend pages for a Blog reader"
ppr = personalized_pagerank(web, seed_node=0)  # Personalized to Blog
print("\nPersonalized PageRank (seed=Blog):")
for label, rank in sorted(zip(labels, ppr), key=lambda x: -x[1]):
    print(f"  {label:>8s}: {rank:.4f}")
# Blog-adjacent pages (NYT, Forum) rank higher than in standard PageRank
```

> **Common Mistake**: Confusing PageRank with simple in-degree counting. Yes, more incoming links generally means higher PageRank. But the *quality* of those links matters enormously. One link from a high-PageRank page can outweigh hundreds of links from low-PageRank pages. This recursive quality weighting is the entire point.

---

## Random Walks: The Bridge to Modern Graph Embeddings

Random walks are the conceptual glue between classical graph algorithms and modern graph ML. A random walk is exactly what it sounds like: start at a node, repeatedly move to a random neighbor.

### Why Random Walks?

Consider the problem of learning a vector representation (embedding) for each node in a graph, such that nodes with similar "neighborhoods" have similar vectors. You cannot feed an entire graph into Word2Vec. But you CAN generate sequences of nodes via random walks, and then feed THOSE into Word2Vec.

This is the insight behind **DeepWalk** (2014) and **node2vec** (2016):

1. Generate many random walks from each node.
2. Treat each walk as a "sentence" (sequence of "words").
3. Feed the walks into Word2Vec's Skip-gram model.
4. The resulting word vectors ARE your node embeddings.

```
  Random walks as "sentences":

  Graph:                    Random walks from node A:
  ┌───┐───┌───┐            Walk 1: A → B → D → E → B → A
  │ A │   │ B │──┌───┐     Walk 2: A → C → E → D → B → A
  └───┘   └───┘  │ D │     Walk 3: A → B → E → C → A → B
    │       │     └───┘
    │     ┌───┐     │      Treat like sentences for Word2Vec:
    └─────│ C │─────┘        "A B D E B A"
          └───┘──┌───┐       "A C E D B A"
                 │ E │       "A B E C A B"
                 └───┘
                             Nodes that appear in similar contexts
                             get similar embeddings.
```

```python
import random
import numpy as np

def generate_random_walks(adj_list, num_walks=10, walk_length=8):
    """
    Generate random walks -- the first step of DeepWalk.
    Each walk becomes a 'sentence' for Word2Vec.
    """
    walks = []
    nodes = list(adj_list.keys())

    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            for _ in range(walk_length - 1):
                current = walk[-1]
                neighbors = adj_list[current]
                if not neighbors:
                    break
                walk.append(random.choice(neighbors))
            walks.append(walk)

    return walks

# Example
graph = defaultdict(list, {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 4],
    3: [1, 4],
    4: [2, 3]
})

random.seed(42)
walks = generate_random_walks(graph, num_walks=3, walk_length=6)
print("Random walks (DeepWalk style):")
for i, walk in enumerate(walks):
    print(f"  Walk {i}: {walk}")

# In practice, you'd feed these into gensim's Word2Vec:
# from gensim.models import Word2Vec
# walks_str = [[str(n) for n in walk] for walk in walks]
# model = Word2Vec(walks_str, vector_size=64, window=5, min_count=0, sg=1)
# embedding = model.wv[str(node_id)]
```

### The Math: Stationary Distribution of Random Walks

A random walk on an undirected graph has a well-defined stationary distribution. For a connected, non-bipartite graph, the probability of being at node $v$ in the long run is:

$$\pi(v) = \frac{d(v)}{2|E|}$$

where $d(v)$ is the degree of node $v$ and $|E|$ is the number of edges.

**Translation**: In the long run, a random walker spends more time at high-degree nodes. This is proportional to degree, not PageRank. (PageRank adds the "random jump" mechanism which changes things.)

This connects to the **transition probability matrix** $P$:

$$P_{ij} = \frac{A_{ij}}{d(i)}$$

where $A$ is the adjacency matrix and $d(i) = \sum_j A_{ij}$ is the degree. The stationary distribution $\pi$ satisfies $\pi^T P = \pi^T$ -- it is the left eigenvector of $P$ with eigenvalue 1.

> **Common Mistake**: Assuming random walks on any graph converge to a stationary distribution. This requires the graph to be connected and non-bipartite (aperiodic). On a bipartite graph, the random walk oscillates between the two partitions and never converges. The "teleportation" trick in PageRank fixes this.

---

## Graph Neural Networks: Message Passing as Generalized Aggregation

This is where everything comes together. Graph Neural Networks (GNNs) are the deep learning approach to graph data, and they are built on a surprisingly simple idea: **each node updates its representation by aggregating information from its neighbors**. This is called **message passing**.

### The Intuition

Think of it this way. You are at a party. You know some things about yourself (your features). After one round of conversations with your friends, you now know something about your friends too. After TWO rounds (your friends talked to their friends, then talked to you), you know something about friends-of-friends. After $k$ rounds, you have information about everyone within $k$ hops.

```
  Message passing: 3 rounds on a social network

  Round 0 (initial):        Round 1:                 Round 2:
  Each node knows           Each node knows          Each node knows
  only itself               itself + neighbors       2-hop neighborhood

  ┌─A─┐  ┌─B─┐            ┌─A─┐  ┌─B─┐            ┌─A─┐  ┌─B─┐
  │[a]│──│[b]│            │[a,│──│[b,│            │[a,b│──│[a,b│
  └───┘  └─┬─┘            │b,c│  │a,c│            │c,d]│  │c,d]│
           │               └───┘  └─┬─┘            └───┘  └─┬─┘
         ┌─┴─┐                    ┌─┴─┐                   ┌─┴─┐
  ┌─C─┐  │ D │            ┌─C─┐  │ D │            ┌─C─┐  │ D │
  │[c]│──│[d]│            │[c,│──│[d,│            │[a,b│──│[a,b│
  └───┘  └───┘            │a,b│  │b,c│            │c,d]│  │c,d]│
                           │d] │  └───┘            └───┘  └───┘
                           └───┘

  After 2 layers, every node in this small graph has seen everyone.
  In a larger graph, k layers = k-hop "receptive field".
```

### The Math: Message Passing Framework

The general message passing framework at layer $k$:

$$h_v^{(k+1)} = \text{UPDATE}\left(h_v^{(k)},\ \text{AGGREGATE}\left(\left\{h_u^{(k)} : u \in \mathcal{N}(v)\right\}\right)\right)$$

**Translation**: To compute node $v$'s new representation at layer $k+1$:
1. **Gather**: Collect the current representations $h_u^{(k)}$ from all neighbors $u$ of $v$.
2. **Aggregate**: Combine them using some permutation-invariant function (sum, mean, max).
3. **Update**: Combine the aggregated neighbor info with $v$'s own current representation.

The key mathematical requirement: AGGREGATE must be **permutation-invariant** (the order of neighbors should not matter, since graphs have no natural ordering). Sum, mean, and max all satisfy this.

### Graph Convolutional Networks (GCN)

The most widely-used GNN variant is the **Graph Convolutional Network (GCN)** by Kipf and Welling (2017). Its layer update rule is:

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

This looks intimidating. Let me break it down piece by piece.

**$\tilde{A} = A + I_N$** -- the adjacency matrix with self-loops added. Self-loops ensure each node includes its own features in the aggregation (not just neighbor features).

**$\tilde{D}$** -- the degree matrix of $\tilde{A}$. This is a diagonal matrix where $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$.

**$\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$** -- symmetric normalization. This scales the aggregation so that high-degree nodes do not dominate.

**Translation of the normalization**: Without normalization, if node $v$ has 1000 neighbors and node $u$ has 3, the aggregated signal for $v$ would be ~333x larger just because of degree. The $\tilde{D}^{-1/2}$ factors fix this by normalizing by $\frac{1}{\sqrt{\text{deg}(v) \cdot \text{deg}(u)}}$ for each edge $(v, u)$.

**$H^{(l)}$** -- the matrix of node features at layer $l$. Each row is a node, each column is a feature. Shape: $(N, F_l)$.

**$W^{(l)}$** -- learnable weight matrix for layer $l$. Shape: $(F_l, F_{l+1})$. This is the equivalent of the weight matrix in a standard neural network layer.

**$\sigma$** -- nonlinear activation (typically ReLU).

Putting it all together: "Average your neighbors' features (with normalization), apply a linear transformation, then a nonlinearity." That is one GCN layer. Stack $k$ of them, and each node's representation captures information from its $k$-hop neighborhood.

```python
import numpy as np

def gcn_layer(A, H, W):
    """
    One layer of Graph Convolutional Network.

    A: adjacency matrix (n x n)
    H: node feature matrix (n x f_in)
    W: weight matrix (f_in x f_out)

    Returns: new node features (n x f_out)
    """
    n = A.shape[0]

    # Step 1: Add self-loops
    A_tilde = A + np.eye(n)

    # Step 2: Compute degree matrix of A_tilde
    D_tilde = np.diag(A_tilde.sum(axis=1))

    # Step 3: Symmetric normalization: D^{-1/2} * A * D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    # Step 4: Aggregate neighbor features, transform, activate
    out = A_norm @ H @ W  # (n x n) @ (n x f_in) @ (f_in x f_out)
    out = np.maximum(0, out)  # ReLU activation

    return out

def multi_layer_gcn(A, features, weights_list):
    """
    Stack multiple GCN layers.
    k layers = k-hop neighborhood aggregation.
    """
    H = features
    for i, W in enumerate(weights_list):
        H = gcn_layer(A, H, W)
        print(f"  Layer {i+1}: shape {H.shape}")
    return H

# Example: small social network with features
# 4 people: features are [age, income, education_years]
A = np.array([
    [0, 1, 1, 0],  # Alice knows Bob, Carol
    [1, 0, 1, 1],  # Bob knows Alice, Carol, Dave
    [1, 1, 0, 1],  # Carol knows Alice, Bob, Dave
    [0, 1, 1, 0],  # Dave knows Bob, Carol
])

features = np.array([
    [25, 50, 16],   # Alice: 25yo, 50k income, 16yr education
    [35, 80, 18],   # Bob
    [30, 65, 17],   # Carol
    [28, 45, 14],   # Dave
], dtype=float)

# Normalize features (important for neural networks!)
features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

np.random.seed(42)
# 2-layer GCN: 3 features → 4 hidden → 2 output (e.g., for 2-class classification)
W1 = np.random.randn(3, 4) * 0.5
W2 = np.random.randn(4, 2) * 0.5

print("GCN forward pass:")
output = multi_layer_gcn(A, features, [W1, W2])
print(f"\nFinal node representations (for classification):")
names = ['Alice', 'Bob', 'Carol', 'Dave']
for name, rep in zip(names, output):
    print(f"  {name}: {rep}")
print(f"\nAfter 2 layers, each person's representation encodes")
print(f"their own features AND their 2-hop neighborhood's features.")
```

### GCN vs. Simple Neighbor Averaging: Why Learnable Weights Matter

You might wonder: why not just average neighbor features without the weight matrix? Let me show you:

```python
def naive_aggregation(A, features):
    """Just average neighbor features. No learning."""
    A_tilde = A + np.eye(len(A))
    D_inv = np.diag(1.0 / A_tilde.sum(axis=1))
    return D_inv @ A_tilde @ features

# After averaging, similar nodes become IDENTICAL
avg_features = naive_aggregation(A, features)
print("Naive averaging (no weights):")
for name, feat in zip(names, avg_features):
    print(f"  {name}: {np.round(feat, 3)}")
print("\nAlice and Dave have the same neighbors → same averaged features!")
print("Learnable weights W let the model CHOOSE which features matter.")
```

**Translation**: Without learnable weights, neighbor averaging is a fixed smoothing operation. It makes connected nodes more similar (which is sometimes what you want for homophily), but it cannot learn to emphasize relevant features or suppress noisy ones. The weight matrices $W^{(l)}$ are what make GCNs actual *neural networks* rather than just feature smoothers.

> **Common Mistake**: Stacking too many GCN layers. Each layer averages neighbor features, which smooths signals. After 5-6 layers, all node representations converge to the same value -- this is called **over-smoothing**. In practice, 2-3 layers works best for most tasks. This is unlike standard deep learning where "deeper is usually better."

---

## Spectral Clustering via the Graph Laplacian

This section connects graph algorithms to a topic from linear algebra: eigenvalue decomposition. If you have ever wondered "what do eigenvectors have to do with graphs?", here is the answer.

### The Problem

You want to cluster nodes in a graph into groups. Nodes within a group should be densely connected; edges between groups should be sparse. This is **graph clustering** (or community detection).

```
  A graph with two natural clusters:

  Cluster 1          Cluster 2
  ┌─────────┐       ┌─────────┐
  │ 0───1   │       │   4───5 │
  │ │ ╲ │   │       │   │ ╲ │ │
  │ │  ╲│   │       │   │  ╲│ │
  │ 3───2   │╌╌╌╌╌╌╌│   7───6 │
  └─────────┘ weak  └─────────┘
              edges
  (2─4 and 3─7 connect the clusters with few, weak edges)
```

### The Graph Laplacian

The **graph Laplacian** $L$ is defined as:

$$L = D - A$$

where $D$ is the diagonal degree matrix and $A$ is the adjacency matrix.

For our example:

$$L_{ij} = \begin{cases} \text{deg}(i) & \text{if } i = j \\ -1 & \text{if } i \text{ and } j \text{ are adjacent} \\ 0 & \text{otherwise} \end{cases}$$

**Translation**: The Laplacian encodes "how different is this node from its neighbors?" For each node, the diagonal entry is its degree, and each edge contributes a $-1$ to the off-diagonal entries.

The key property: for any vector $\mathbf{f}$ (assigning a value to each node):

$$\mathbf{f}^T L \mathbf{f} = \sum_{(i,j) \in E} (f_i - f_j)^2$$

**Translation**: $\mathbf{f}^T L \mathbf{f}$ measures how much $f$ varies across edges. If connected nodes have similar values, this is small. If connected nodes have very different values, this is large.

### Why Eigenvectors Give You Clusters

The eigenvectors of $L$ are the vectors that vary *least* across edges (for the smallest eigenvalues). The smallest eigenvalue of $L$ is always 0, with eigenvector $\mathbf{1}$ (the constant vector -- it varies zero across all edges, trivially).

The **second smallest eigenvector** (the "Fiedler vector") is the non-trivial vector that varies least across edges. It assigns similar values to tightly connected nodes and different values to weakly connected nodes. Thresholding this vector at 0 gives you a 2-way partition.

For $k$ clusters, use the $k$ smallest eigenvectors and run k-means on them.

```python
import numpy as np
from collections import defaultdict

def spectral_clustering(A, k=2):
    """
    Spectral clustering using the graph Laplacian.

    1. Compute Laplacian L = D - A
    2. Find k smallest eigenvectors of L
    3. Use them as features for k-means clustering
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # Degree matrix
    D = np.diag(A.sum(axis=1))

    # Laplacian
    L = D - A

    # Eigendecomposition (sorted by eigenvalue)
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Take k smallest eigenvectors (skip the trivial one if desired)
    # Using indices 1:k+1 skips the constant eigenvector
    features = eigenvectors[:, 1:k+1]

    # Simple k-means (2 clusters: threshold Fiedler vector at 0)
    if k == 2:
        fiedler = eigenvectors[:, 1]
        labels = (fiedler > 0).astype(int)
        return labels, eigenvalues, fiedler

    # For k > 2, use proper k-means on eigenvector features
    # (simplified: just return the features for sklearn.cluster.KMeans)
    return features, eigenvalues, None

# Graph with two clear communities
A = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],  # 0
    [1, 0, 1, 1, 0, 0, 0, 0],  # 1
    [1, 1, 0, 1, 1, 0, 0, 0],  # 2 (bridge)
    [1, 1, 1, 0, 0, 0, 0, 1],  # 3 (bridge)
    [0, 0, 1, 0, 0, 1, 1, 1],  # 4
    [0, 0, 0, 0, 1, 0, 1, 1],  # 5
    [0, 0, 0, 0, 1, 1, 0, 1],  # 6
    [0, 0, 0, 1, 1, 1, 1, 0],  # 7
])

labels, eigenvalues, fiedler = spectral_clustering(A, k=2)

print("Eigenvalues of Laplacian:", np.round(eigenvalues, 3))
print(f"\nFiedler vector (2nd eigenvector): {np.round(fiedler, 3)}")
print(f"Cluster labels: {labels}")
print(f"  Cluster 0 (nodes): {np.where(labels == 0)[0]}")
print(f"  Cluster 1 (nodes): {np.where(labels == 1)[0]}")
# Should roughly split into {0,1,2,3} and {4,5,6,7}

print(f"\nAlgebraic connectivity (2nd eigenvalue): {eigenvalues[1]:.3f}")
print(f"(Measures how well-connected the graph is. 0 = disconnected.)")
```

**Translation**: Spectral clustering works because the eigenvectors of the Laplacian encode the graph's connectivity structure. The Fiedler vector is the "smoothest" non-constant function on the graph -- it changes as little as possible across edges. That means it assigns similar values to well-connected nodes and different values to weakly-connected nodes. Cutting at zero naturally separates clusters.

> **You Already Know This**: If you have used PCA on tabular data, you already understand eigenvector decomposition for dimensionality reduction. Spectral clustering is "PCA for graphs" -- instead of finding the directions of maximum variance, you find the directions of minimum variation across edges.

---

## Putting It All Together: A Complete Pipeline

Let me tie all these algorithms together in a realistic scenario. You are building a fraud detection system for a financial transaction graph.

```python
import numpy as np
from collections import defaultdict, deque

class FraudDetectionPipeline:
    """
    Combining graph algorithms for fraud detection.

    Nodes: bank accounts
    Edges: money transfers
    Goal: identify suspicious accounts
    """

    def __init__(self, adj_matrix, account_features):
        self.A = np.array(adj_matrix, dtype=float)
        self.features = np.array(account_features, dtype=float)
        self.n = self.A.shape[0]
        self.adj_list = defaultdict(list)
        for i in range(self.n):
            for j in range(self.n):
                if self.A[i][j] > 0:
                    self.adj_list[i].append(j)

    def compute_pagerank_features(self, damping=0.85):
        """Step 1: PageRank as a feature (how 'important' is this account?)."""
        out_degree = self.A.sum(axis=1)
        M = np.zeros((self.n, self.n))
        for j in range(self.n):
            if out_degree[j] == 0:
                M[:, j] = 1.0 / self.n
            else:
                M[:, j] = self.A[j, :] / out_degree[j]

        r = np.ones(self.n) / self.n
        for _ in range(100):
            r_new = (1 - damping) / self.n + damping * M @ r
            if np.sum(np.abs(r_new - r)) < 1e-8:
                break
            r = r_new
        return r

    def compute_bfs_features(self):
        """Step 2: BFS-based features (how far from known fraud?)."""
        # Assume node 0 is a known fraudulent account
        known_fraud = 0
        visited = {known_fraud}
        queue = deque([known_fraud])
        distances = {known_fraud: 0}

        while queue:
            node = queue.popleft()
            for neighbor in self.adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)

        return np.array([distances.get(i, self.n) for i in range(self.n)])

    def compute_gcn_features(self, weights_list):
        """Step 3: GCN for learned representations."""
        A_tilde = self.A + np.eye(self.n)
        D_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1)))
        A_norm = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt

        H = self.features
        for W in weights_list:
            H = np.maximum(0, A_norm @ H @ W)
        return H

    def run_pipeline(self):
        """Combine all features for fraud scoring."""
        print("=== Fraud Detection Pipeline ===\n")

        # PageRank features
        pr = self.compute_pagerank_features()
        print(f"1. PageRank scores: {np.round(pr, 3)}")

        # BFS distance from known fraud
        dist = self.compute_bfs_features()
        print(f"2. Distance from known fraud: {dist}")

        # GCN features
        np.random.seed(42)
        W1 = np.random.randn(self.features.shape[1], 4) * 0.3
        W2 = np.random.randn(4, 2) * 0.3
        gcn_out = self.compute_gcn_features([W1, W2])
        print(f"3. GCN embeddings shape: {gcn_out.shape}")

        # Combine into feature matrix
        combined = np.column_stack([
            pr,                # 1 feature: PageRank
            1.0 / (dist + 1), # 1 feature: proximity to fraud (inverse distance)
            gcn_out            # 2 features: GCN embeddings
        ])
        print(f"\nCombined feature matrix shape: {combined.shape}")
        print(f"Features per account: [PageRank, FraudProximity, GCN_1, GCN_2]")

        # Simple fraud score: weighted combination
        # (In practice, you'd train a classifier on labeled data)
        fraud_score = 0.3 * pr + 0.5 * (1.0 / (dist + 1)) + 0.2 * np.abs(gcn_out).sum(axis=1)
        print(f"\nFraud scores: {np.round(fraud_score, 3)}")
        print(f"Most suspicious account: {np.argmax(fraud_score)}")

        return combined, fraud_score

# Simulate: 6 accounts, node 0 is known fraud
adj = np.array([
    [0, 1, 1, 0, 0, 0],  # Fraud account → 1, 2
    [1, 0, 1, 1, 0, 0],  # Account 1 (suspicious)
    [1, 1, 0, 0, 1, 0],  # Account 2 (suspicious)
    [0, 1, 0, 0, 1, 1],  # Account 3
    [0, 0, 1, 1, 0, 1],  # Account 4
    [0, 0, 0, 1, 1, 0],  # Account 5 (clean)
])

# Account features: [transaction_volume, avg_amount, account_age_days]
features = np.array([
    [100, 5000, 30],    # Fraud: high volume, large amounts, new account
    [80, 3000, 45],     # Suspicious
    [60, 2500, 60],     # Suspicious
    [20, 500, 365],     # Normal
    [15, 300, 730],     # Normal
    [10, 200, 1000],    # Normal
], dtype=float)

pipeline = FraudDetectionPipeline(adj, features)
combined_features, scores = pipeline.run_pipeline()
```

---

## Exercises

### Exercise 1: BFS Shortest Path in a Grid

**Problem**: You have a 2D grid representing a game map. Find the shortest path from the top-left to bottom-right, moving only through open cells.

```
Grid:       . = open, # = wall

  . . . # .
  # . # . .
  . . . . #
  . # # . .
  . . . . .
```

**Solution**:

```python
from collections import deque

def grid_bfs(grid):
    """
    BFS on a grid -- same algorithm, different data structure.
    This is exactly what pathfinding in games uses.
    """
    rows, cols = len(grid), len(grid[0])
    start = (0, 0)
    end = (rows - 1, cols - 1)

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()

        if (r, c) == end:
            return path

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                and grid[nr][nc] == '.' and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))

    return None  # No path exists

grid = [
    ['.', '.', '.', '#', '.'],
    ['#', '.', '#', '.', '.'],
    ['.', '.', '.', '.', '#'],
    ['.', '#', '#', '.', '.'],
    ['.', '.', '.', '.', '.'],
]

path = grid_bfs(grid)
print(f"Shortest path length: {len(path) - 1} steps")
print(f"Path: {path}")

# Visualize
for r in range(len(grid)):
    row = ""
    for c in range(len(grid[0])):
        if (r, c) in path:
            row += " * "
        elif grid[r][c] == '#':
            row += " # "
        else:
            row += " . "
    print(row)
```

### Exercise 2: PageRank by Hand

**Problem**: Calculate one iteration of PageRank (with $d = 0.85$) for this graph:

```
A ──→ B ──→ C
      ▲     │
      └─────┘
```

Initial ranks: $PR(A) = PR(B) = PR(C) = 1/3$.

**Solution**:

Using $PR(v) = \frac{1-d}{N} + d \sum_{u \in B(v)} \frac{PR(u)}{L(u)}$:

- $PR(A) = \frac{0.15}{3} + 0.85 \times 0 = 0.05$. Nothing links to A.
- $PR(B) = \frac{0.15}{3} + 0.85 \times \left(\frac{PR(A)}{1} + \frac{PR(C)}{1}\right) = 0.05 + 0.85 \times \frac{2}{3} = 0.6167$.
- $PR(C) = \frac{0.15}{3} + 0.85 \times \frac{PR(B)}{1} = 0.05 + 0.85 \times \frac{1}{3} = 0.3333$.

```python
d = 0.85
N = 3
pr = {'A': 1/3, 'B': 1/3, 'C': 1/3}

# One iteration
new_pr = {
    'A': (1-d)/N + d * 0,                          # No incoming links
    'B': (1-d)/N + d * (pr['A']/1 + pr['C']/1),    # A→B and C→B
    'C': (1-d)/N + d * (pr['B']/1),                 # B→C
}

print("After iteration 1:")
for node, rank in new_pr.items():
    print(f"  PR({node}) = {rank:.4f}")
# PR(A) = 0.0500 -- A has no incoming links, only gets the "random jump" baseline
# PR(B) = 0.6167 -- B gets rank from both A and C
# PR(C) = 0.3333 -- C gets rank from B
```

Note that the ranks do not sum to exactly 1.0 after one iteration (they sum to 1.0 only at convergence). Keep iterating and they will converge.

### Exercise 3: GNN Message Passing by Hand

**Problem**: Given this graph and features, compute one round of mean-aggregation with self-loops (no weights, no activation). Verify that the GCN normalization produces different results than simple mean aggregation.

```
Graph: A ── B ── C      Features: A=[1,0], B=[0,1], C=[1,1]
```

**Solution**:

```python
import numpy as np

A = np.array([
    [0, 1, 0],  # A -- B
    [1, 0, 1],  # B -- A, C
    [0, 1, 0],  # C -- B
])

features = np.array([
    [1, 0],  # A
    [0, 1],  # B
    [1, 1],  # C
], dtype=float)

# Method 1: Simple mean aggregation (with self-loops)
A_self = A + np.eye(3)
D_inv = np.diag(1.0 / A_self.sum(axis=1))
simple_mean = D_inv @ A_self @ features

print("Simple mean aggregation:")
for name, feat in zip(['A', 'B', 'C'], simple_mean):
    print(f"  {name}: {feat}")
# A: mean of [A, B] = mean of [[1,0], [0,1]] = [0.5, 0.5]
# B: mean of [A, B, C] = mean of [[1,0], [0,1], [1,1]] = [0.667, 0.667]
# C: mean of [B, C] = mean of [[0,1], [1,1]] = [0.5, 1.0]

# Method 2: GCN symmetric normalization
D_inv_sqrt = np.diag(1.0 / np.sqrt(A_self.sum(axis=1)))
gcn_norm = D_inv_sqrt @ A_self @ D_inv_sqrt @ features

print("\nGCN symmetric normalization:")
for name, feat in zip(['A', 'B', 'C'], gcn_norm):
    print(f"  {name}: {np.round(feat, 4)}")
# Different! GCN normalization accounts for BOTH sender and receiver degree.
# Node B (degree 3) contributes less per-edge than A or C (degree 2).

print("\nDifference matters: GCN normalization prevents high-degree nodes")
print("from dominating the aggregation in heterogeneous-degree graphs.")
```

### Exercise 4: Random Walk Co-occurrence

**Problem**: Generate 5 random walks of length 4 from node 0. Count how often each pair of nodes co-occurs within a window of size 2. Which pairs would get similar embeddings in DeepWalk?

```python
import random
from collections import defaultdict, Counter

graph = defaultdict(list, {
    0: [1, 3],
    1: [0, 2],
    2: [1, 3],
    3: [0, 2],
})

random.seed(123)
walks = []
for _ in range(5):
    walk = [0]
    for _ in range(3):
        walk.append(random.choice(graph[walk[-1]]))
    walks.append(walk)
    print(f"  Walk: {walk}")

# Count co-occurrences within window of 2
cooccurrence = Counter()
window = 2
for walk in walks:
    for i, node in enumerate(walk):
        for j in range(max(0, i - window), min(len(walk), i + window + 1)):
            if i != j:
                pair = tuple(sorted([walk[i], walk[j]]))
                cooccurrence[pair] += 1

print(f"\nCo-occurrence counts (window={window}):")
for pair, count in cooccurrence.most_common():
    print(f"  {pair}: {count}")
print("\nHigh co-occurrence → similar embeddings in DeepWalk/node2vec")
```

---

## Summary

### Key Takeaways

- **BFS and DFS** are algorithms you already know, but in ML they become neighborhood sampling strategies. BFS captures local community (homophily); DFS captures structural roles. GraphSAGE, node2vec, and DeepWalk all build on these traversals.

- **Topological sort** is your build system's dependency resolver. In ML, it is how autograd engines (PyTorch, TensorFlow) schedule gradient computation in backpropagation.

- **Shortest paths** (BFS for unweighted, Dijkstra for weighted) produce distance features for link prediction and graph kernels. The same $O(V + E)$ or $O((V+E)\log V)$ complexity you know from algorithms class.

- **Minimum spanning trees** connect network infrastructure optimization to graph-based clustering and feature engineering for geometric deep learning.

- **PageRank** computes global node importance via random walks. The formula $PR(v) = \frac{1-d}{N} + d \sum_{u \in B(v)} \frac{PR(u)}{L(u)}$ is solved by power iteration. Personalized PageRank drives recommendation systems at Pinterest, Twitter, and beyond.

- **Random walks** are the bridge between classical graph theory and modern graph embeddings. DeepWalk and node2vec turn random walks into "sentences" for Word2Vec, producing node embeddings that encode graph structure.

- **GNNs** learn node representations through message passing: $h_v^{(k+1)} = \text{UPDATE}(h_v^{(k)}, \text{AGGREGATE}(\{h_u^{(k)} : u \in \mathcal{N}(v)\}))$. The GCN variant uses symmetric normalization $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ to prevent high-degree nodes from dominating. Stacking $k$ layers captures $k$-hop neighborhoods, but too many layers cause over-smoothing.

- **Spectral clustering** uses the graph Laplacian $L = D - A$ and its eigenvectors to partition graphs. The Fiedler vector (second smallest eigenvector) reveals the graph's natural cluster structure. Think of it as "PCA for graphs."

### The Connection Between Everything

Here is how all the pieces fit together:

```
  Classical Algorithms              Modern Graph ML
  ─────────────────────             ──────────────────────
  BFS traversal          ────→      GNN neighborhood sampling (GraphSAGE)
  DFS / random walks     ────→      Node embeddings (DeepWalk, node2vec)
  PageRank (random walk) ────→      Attention weights (GAT), recommendations
  Shortest paths         ────→      Link prediction features, graph kernels
  MST                    ────→      Graph clustering, point cloud features
  Topological sort       ────→      Autograd execution order (backprop)
  Laplacian eigenvectors ────→      Spectral clustering, GCN foundations
```

Every modern graph ML technique has classical graph algorithms at its core. You did not learn BFS and Dijkstra just for coding interviews. They are the computational primitives of an entire branch of machine learning.

---

> **What's Next**: We have built a powerful toolkit of graph algorithms that traverse, rank, embed, and learn from graph structure. But when you deploy these at scale -- running PageRank on billions of nodes, training GNNs on massive graphs, computing spectral decompositions of huge matrices -- numerical issues start creeping in. Floating-point errors accumulate. Matrix inversions become unstable. Eigenvalue computations fail silently. In **Level 12: Numerical Methods**, we will learn the techniques that ensure our graph algorithms (and all our other mathematical tools) work reliably at scale. That is the difference between an algorithm that works on a whiteboard and one that works in production.
