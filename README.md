# Mathematics for Machine Learning

> You already nailed the hard part -- you think in systems, you debug under pressure, and you ship. This book gives you the math to build what comes next.

---

## Who This Book Is For

You are a **senior software engineer**. You have spent years building distributed systems, designing APIs, reviewing pull requests, and reasoning about performance at scale. You are very good at what you do.

Now you want to move into AI and machine learning -- and you have hit a wall of Greek letters.

Traditional math textbooks were not written for you. They assume you are an eighteen-year-old with no context and infinite patience. They start from definitions and work toward relevance, which is exactly backwards from how you learn. You learn by building something, breaking it, then understanding why it broke.

This book meets you where you are. Every concept starts with a real ML problem, maps to something you already know as an engineer, and earns its place by showing you exactly where it appears in the models you want to build. No busywork. No unmotivated abstraction. Just the math you need, explained the way you think.

**What makes this different from other math books:**

- You already know pure functions, middleware pipelines, and `map`/`filter`/`reduce`. That is function composition. You have been doing math for years -- you just called it "software engineering."
- Every formula gets a plain-English translation paragraph. No "left as an exercise" handwaving.
- Working Python code accompanies every concept. Run it, break it, rebuild it.
- A running example (building a movie recommender) ties the entire book together so you always know *why* a concept matters.

---

## How to Read This Book

### Rendering Math

This book uses LaTeX notation for mathematical formulas (`$x^2$`, `$$\sum_{i=1}^{n} x_i$$`). Here is how to make sure it renders properly in your environment:

| Environment | What to Do |
|-------------|------------|
| **VS Code** | Install the [Markdown+Math](https://marketplace.visualstudio.com/items?itemName=goessner.mdmath) or [Markdown Preview Enhanced](https://marketplace.visualstudio.com/items?itemName=shd101wyy.markdown-preview-enhanced) extension |
| **GitHub** | Math renders natively in `.md` files -- no setup needed |
| **Obsidian** | LaTeX renders out of the box -- just open the vault |
| **PDF export** | Use pandoc: `pandoc chapter.md -o chapter.pdf --pdf-engine=xelatex` |

### Recommended Reading Order

**The designed path**: Follow the levels in order (0 through 14). Each level builds on the ones before it, and chapter transitions tell you exactly what connects where. This path takes roughly 3-4 months at a pace of one chapter per day.

**The impatient path**: If you already have solid algebra and function basics (you can read summation notation, you know what a composite function is, logarithms do not scare you), jump straight to the Phase 1 core:

1. **Level 4** -- Linear Algebra (the data structure every ML model is built on)
2. **Level 6** -- Calculus (the engine behind `loss.backward()`)
3. **Level 7** -- Probability (the framework for reasoning under uncertainty)
4. **Level 9** -- Optimization (the training loop, mathematically)

Then circle back to earlier levels when something feels shaky, push forward to Level 13 for the grand payoff, and dip into Levels 5, 8, 10-12 as your work demands them.

### Code Examples

Every chapter includes working Python code. To run them, you need:

```bash
pip install numpy scipy matplotlib scikit-learn torch
```

Code examples use **NumPy** for linear algebra and numerical work, **scikit-learn** for classical ML, and **PyTorch** for neural network examples. All code is self-contained -- you can copy any block into a script or notebook and run it immediately. The goal is not just to show you the formula but to let you *poke at it* until the intuition clicks.

---

## The Running Example: Building a Movie Recommender

Throughout this book, a single thread ties the levels together: **building a movie recommendation system from scratch**. It starts simple and grows in sophistication as you gain new mathematical tools.

Think of it as a feature spec that keeps getting more ambitious:

- **Level 1**: Users rate movies 1 through 5. Those are numbers. What kind of numbers? What can we do with them?
- **Level 2**: We model ratings as expressions and equations. Algebra gives us the language to write `predicted_rating = bias + weight * feature`.
- **Level 3**: A recommender is a function -- it takes a user and a movie and outputs a predicted rating. Functions let us formalize what "a model" even means.
- **Level 4**: Each user's ratings form a vector. Each movie has a vector too. Similarity is a dot product. This is the backbone of collaborative filtering.
- **Level 5**: Distance metrics tell us which users are "close" in taste-space. Cosine similarity, Euclidean distance -- each tells a different story.
- **Level 6**: We want to minimize prediction error. Calculus tells us which direction to adjust our parameters and by how much.
- **Level 7**: Ratings are noisy. Probability helps us reason about uncertainty in preferences and model confidence.
- **Level 8**: We trained two models. Which one is actually better? Statistics gives us hypothesis tests and confidence intervals so we stop guessing.
- **Level 9**: SGD makes training tractable for Netflix's 200 million users. Regularization stops the model from memorizing the training set.
- **Level 10**: Cross-entropy loss -- the metric your classification model actually optimizes. Information theory explains *why* it works.
- **Level 13**: Putting it all together -- matrix factorization, neural collaborative filtering, and the full mathematical machinery behind real-world recommendations.

You do not need to follow this thread to understand the book, but it gives you a concrete anchor whenever the math feels abstract. Every time you see it, you will know exactly *why* this concept matters.

---

## Table of Contents

### Level 0: Mathematical Foundations -- Learning to read the language of math
*Your first translation layer: from code syntax to mathematical notation. By the end of this level, Greek letters stop being scary and start being just another API you need to learn.* (3 chapters)

| Chapter | Topic |
|---------|-------|
| 0.1 | [Mathematical Language](00-level-0-foundations/01-mathematical-language.md) |
| 0.2 | [Sets & Logic](00-level-0-foundations/02-sets-and-logic.md) |
| 0.3 | [Mathematical Thinking](00-level-0-foundations/03-mathematical-thinking.md) |

### Level 1: Arithmetic & Numbers -- Where floating-point bites back
*You know numbers. But do you know why `0.1 + 0.2 != 0.3` matters for model training? This level connects number representation to the numerical stability issues that silently break ML pipelines.* (3 chapters)

| Chapter | Topic |
|---------|-------|
| 1.1 | [Number Systems](01-level-1-arithmetic/01-number-systems.md) |
| 1.2 | [Arithmetic Operations](01-level-1-arithmetic/02-arithmetic-operations.md) |
| 1.3 | [Ratios & Scales](01-level-1-arithmetic/03-ratios-and-scales.md) |

### Level 2: Algebra -- From code variables to math variables
*You already think in variables, expressions, and transformations every time you write code. Here you formalize that intuition and learn the algebraic toolkit that every ML paper takes for granted.* (5 chapters)

| Chapter | Topic |
|---------|-------|
| 2.1 | [Variables & Expressions](02-level-2-algebra/01-variables-and-expressions.md) |
| 2.2 | [Linear Equations](02-level-2-algebra/02-linear-equations.md) |
| 2.3 | [Polynomials](02-level-2-algebra/03-polynomials.md) |
| 2.4 | [Exponentials & Logarithms](02-level-2-algebra/04-exponentials-and-logarithms.md) |
| 2.5 | [Inequalities](02-level-2-algebra/05-inequalities.md) |

### Level 3: Functions -- Every ML model is a function (yes, really)
*A neural network with 175 billion parameters is still just a function: inputs in, outputs out. This level builds the vocabulary you need to talk about what models do, mathematically.* (3 chapters)

| Chapter | Topic |
|---------|-------|
| 3.1 | [Functions](03-level-3-functions/01-functions.md) |
| 3.2 | [Common Function Types](03-level-3-functions/02-common-function-types.md) |
| 3.3 | [Multivariable Functions](03-level-3-functions/03-multivariable-functions.md) |

### Level 4: Linear Algebra -- The data structure every ML model is built on
*Vectors, matrices, transformations, decompositions -- linear algebra is to machine learning what arrays and hash maps are to software engineering. If you learn one level deeply, make it this one.* (7 chapters)

| Chapter | Topic |
|---------|-------|
| 4.1 | [Vectors](04-level-4-linear-algebra/01-vectors.md) |
| 4.2 | [Geometry of Vectors](04-level-4-linear-algebra/02-geometry-of-vectors.md) |
| 4.3 | [Matrices](04-level-4-linear-algebra/03-matrices.md) |
| 4.4 | [Matrix as Transformation](04-level-4-linear-algebra/04-matrix-as-transformation.md) |
| 4.5 | [Systems of Linear Equations](04-level-4-linear-algebra/05-systems-of-linear-equations.md) |
| 4.6 | [Eigenvalues & Eigenvectors](04-level-4-linear-algebra/06-eigenvalues-and-eigenvectors.md) |
| 4.7 | [Decompositions](04-level-4-linear-algebra/07-decompositions.md) |

### Level 5: Analytic Geometry -- Measuring similarity in high-dimensional space
*When your vectors live in 768 dimensions, Euclidean intuition breaks down. This level gives you the geometric tools to measure how "close" things are -- the foundation of search, clustering, and retrieval.* (2 chapters)

| Chapter | Topic |
|---------|-------|
| 5.1 | [Coordinate Systems](05-level-5-analytic-geometry/01-coordinate-systems.md) |
| 5.2 | [Distance Metrics](05-level-5-analytic-geometry/02-distance-metrics.md) |

### Level 6: Calculus -- How models learn from their mistakes
*Gradient descent is just calculus in a trench coat. This level takes you from limits through derivatives to the gradient -- the mathematical engine behind every `loss.backward()` call in PyTorch.* (5 chapters)

| Chapter | Topic |
|---------|-------|
| 6.1 | [Limits](06-level-6-calculus/01-limits.md) |
| 6.2 | [Derivatives](06-level-6-calculus/02-derivatives.md) |
| 6.3 | [Gradients](06-level-6-calculus/03-gradients.md) |
| 6.4 | [Optimization](06-level-6-calculus/04-optimization.md) |
| 6.5 | [Integral Calculus](06-level-6-calculus/05-integral-calculus.md) |

### Level 7: Probability Theory -- Reasoning under uncertainty
*Your model says "85% cat." What does that mean? Probability theory is the framework for reasoning about uncertainty, noise, and confidence -- the stuff that separates a demo from a production system.* (5 chapters)

| Chapter | Topic |
|---------|-------|
| 7.1 | [Probability Foundations](07-level-7-probability/01-probability-foundations.md) |
| 7.2 | [Conditional Probability](07-level-7-probability/02-conditional-probability.md) |
| 7.3 | [Random Variables](07-level-7-probability/03-random-variables.md) |
| 7.4 | [Expectation & Moments](07-level-7-probability/04-expectation-and-moments.md) |
| 7.5 | [Common Distributions](07-level-7-probability/05-common-distributions.md) |

### Level 8: Statistics -- Proving your model actually works
*You shipped the model. Is it better than the old one? How do you know it is not just overfitting? Statistics gives you the rigorous tools for model evaluation, A/B testing, and the confidence intervals that keep you from fooling yourself.* (4 chapters)

| Chapter | Topic |
|---------|-------|
| 8.1 | [Descriptive Statistics](08-level-8-statistics/01-descriptive-statistics.md) |
| 8.2 | [Sampling Theory](08-level-8-statistics/02-sampling-theory.md) |
| 8.3 | [Estimation](08-level-8-statistics/03-estimation.md) |
| 8.4 | [Hypothesis Testing](08-level-8-statistics/04-hypothesis-testing.md) |

### Level 9: Optimization Theory -- The engine room of model training
*This is where math meets `model.fit()`. Loss functions, gradient descent, Adam, regularization -- you have been calling these APIs for years. Now you understand what happens inside them and *why* things go wrong when they go wrong.* (4 chapters)

| Chapter | Topic |
|---------|-------|
| 9.1 | [Loss Functions](09-level-9-optimization-theory/01-loss-functions.md) |
| 9.2 | [Optimization Algorithms](09-level-9-optimization-theory/02-optimization-algorithms.md) |
| 9.3 | [Convex Optimization](09-level-9-optimization-theory/03-convex-optimization.md) |
| 9.4 | [Regularization](09-level-9-optimization-theory/04-regularization.md) |

### Level 10: Information Theory -- What your loss function is really measuring
*Ever wonder why cross-entropy is the loss function for classification? Information theory explains why -- and gives you the vocabulary to reason about what your model knows, what it does not, and how to measure the gap.* (4 chapters)

| Chapter | Topic |
|---------|-------|
| 10.1 | [Information](10-level-10-information-theory/01-information.md) |
| 10.2 | [Entropy](10-level-10-information-theory/02-entropy.md) |
| 10.3 | [Cross-Entropy](10-level-10-information-theory/03-cross-entropy.md) |
| 10.4 | [KL Divergence](10-level-10-information-theory/04-kl-divergence.md) |

### Level 11: Graph Theory -- Networks, relationships, and GNNs
*Social networks, knowledge graphs, molecular structures -- some data is fundamentally relational. Graph theory gives you the math to model it, and graph neural networks are one of the fastest-growing areas in ML.* (3 chapters)

| Chapter | Topic |
|---------|-------|
| 11.1 | [Graph Basics](11-level-11-graph-theory/01-graph-basics.md) |
| 11.2 | [Graph Properties](11-level-11-graph-theory/02-graph-properties.md) |
| 11.3 | [Graph Algorithms](11-level-11-graph-theory/03-graph-algorithms.md) |

### Level 12: Numerical Methods -- When math meets floating-point reality
*Elegant math on paper, numerical disaster in code. This level covers the gap between theoretical algorithms and their practical implementation -- the stability issues, approximation trade-offs, and computational tricks that make ML actually work at scale.* (3 chapters)

| Chapter | Topic |
|---------|-------|
| 12.1 | [Numerical Stability](12-level-12-numerical-methods/01-numerical-stability.md) |
| 12.2 | [Approximation](12-level-12-numerical-methods/02-approximation.md) |
| 12.3 | [Optimization in Practice](12-level-12-numerical-methods/03-optimization-in-practice.md) |

### Level 13: ML Models Math -- The grand unification
*Everything comes together. You will see linear regression, logistic regression, neural networks, and PCA through a purely mathematical lens -- and realize they are all variations on the same handful of ideas you have been building toward since Level 0.* (4 chapters)

| Chapter | Topic |
|---------|-------|
| 13.1 | [Linear Regression](13-level-13-ml-models-math/01-linear-regression.md) |
| 13.2 | [Logistic Regression](13-level-13-ml-models-math/02-logistic-regression.md) |
| 13.3 | [Neural Networks](13-level-13-ml-models-math/03-neural-networks.md) |
| 13.4 | [Dimensionality Reduction](13-level-13-ml-models-math/04-dimensionality-reduction.md) |

### Level 14: Advanced Topics -- Where to go from here
*Pointers to the deeper waters: measure theory, topology, stochastic processes, and the mathematical frontiers behind today's research. You will not need most of this for day-to-day ML work, but when you do, you will know where to look.* (1 chapter)

| Chapter | Topic |
|---------|-------|
| 14.1 | [Advanced Topics Reference](14-level-14-advanced/01-advanced-topics-reference.md) |

### Appendices -- Quick-reference material you will reach for repeatedly
*Symbols, formulas, distributions, and code patterns from all 14 levels, collected into lookup-friendly tables. Keep them open in a second tab while you read.* (9 appendices)

| Appendix | Topic |
|----------|-------|
| A | [Greek Alphabet & Pronunciation](15-appendices/A-greek-alphabet.md) |
| B | [Notation Quick Reference](15-appendices/B-notation-reference.md) |
| C | [Key Theorems & Identities](15-appendices/C-key-theorems-and-identities.md) |
| D | [Probability Distributions](15-appendices/D-probability-distributions.md) |
| E | [Matrix Cookbook](15-appendices/E-matrix-cookbook.md) |
| F | [Calculus Reference](15-appendices/F-calculus-reference.md) |
| G | [NumPy / SciPy / PyTorch Cheat Sheet](15-appendices/G-numpy-scipy-pytorch-cheatsheet.md) |
| H | [Glossary](15-appendices/H-glossary.md) |
| I | [Recommended Reading](15-appendices/I-recommended-reading.md) |

---

## Learning Path

The levels build on each other, but not always linearly. Here is the dependency structure:

```
Level 0: Foundations
  |
Level 1: Arithmetic
  |
Level 2: Algebra
  |
Level 3: Functions
  |
  +--------------------------------------------------+
  |                                                   |
Level 4: Linear Algebra                         Level 11: Graph Theory
  |                                                   |
  +-----------------+-----------------+               |
  |                 |                 |               (to Level 13)
Level 5:         Level 6:         Level 7:
Analytic         Calculus         Probability
Geometry            |                |
  |                 |           +----+----+
  |                 |           |         |
(to Level 13)       |        Level 8:  Level 10:
                    |        Statistics Info Theory
                    |           |         |
                    +-----+-----+         |
                          |               |
                     Level 9: Optimization
                          |
                     Level 12: Numerical Methods
                          |
                     Level 13: ML Models Math  <-- everything converges here
                          |
                     Level 14: Advanced Topics
```

**Dependency notes:**

- **Levels 0-3** are strictly sequential. Each builds directly on the previous.
- **Level 4** (Linear Algebra) is the gateway. Three independent branches open up from here.
- **Level 5** (Analytic Geometry), **Level 6** (Calculus), and **Level 7** (Probability) can be studied in any order after Level 4.
- **Level 8** (Statistics) requires Level 7 (Probability).
- **Level 10** (Information Theory) requires Level 7 (Probability).
- **Level 9** (Optimization) requires both Level 6 (Calculus) and Level 8 (Statistics).
- **Level 11** (Graph Theory) only requires Level 3 (Functions) -- it is an independent branch.
- **Level 12** (Numerical Methods) requires Level 9 (Optimization).
- **Level 13** (ML Models Math) pulls from nearly everything: Levels 4-10 and 12.

**Core ML path** (shortest route to reading ML papers with confidence): Levels 0-4, then 6, 7, 9, 13.

---

## Chapter Format

Each chapter is designed to feel like a whiteboard session with a senior ML engineer, not a lecture hall. The order of components varies by topic, but every chapter includes:

1. **A real ML problem** to motivate the concept -- you will always know *why* before *what*
2. **"You Already Know This"** bridges to your engineering experience -- deep conceptual mappings, not surface analogies
3. **Intuition and plain English** before any formulas appear
4. **Mathematical formalism** -- the precise definitions, with a translation paragraph after every derivation
5. **Working code** -- Python you can run, poke at, and extend
6. **ML relevance** -- exactly where this concept shows up in practice
7. **Common mistakes** -- the specific traps engineers fall into (like assuming matrix multiplication is commutative or that gradient points downhill)
8. **Exercises** -- problems with solutions, ranging from "verify you understood" to "now apply this"
9. **What comes next** -- a bridge to the next chapter explaining what you can now do and what you still cannot

---

## Prerequisites

- **Programming**: You write code professionally. Python experience helps but is not strictly required -- the code examples are straightforward and well-commented.
- **Python 3.x**: Installed and working. A virtual environment is recommended.
- **High school math**: You need to remember roughly what algebra is. If you survived a CS degree, you have more than enough.
- **Curiosity**: The willingness to sit with discomfort when something does not click immediately. The math will click. Give it time.

You do **not** need prior university-level math. That is the whole point.

---

## Recommended Reading

See [Appendix I: Recommended Reading](15-appendices/I-recommended-reading.md) for annotated bibliography, free online resources, and suggested learning paths through external materials.

---

## License

Educational use.
