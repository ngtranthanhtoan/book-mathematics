# Chapter 1: Probability Foundations

> **Building On** — Calculus gave you the tools for optimization. But real-world data is noisy and uncertain. To reason about uncertainty mathematically, you need probability.

---

Your spam classifier says an email is "95% likely spam." What does that number actually mean? Can you trust it? Probability theory isn't just about coin flips — it's the language your models speak. If you don't understand it, you can't debug them.

Every time you call `model.predict_proba()`, every softmax output, every confidence score — those are all probability values. They follow specific rules, and when something goes wrong (outputs that don't sum to 1, confidences that seem "off," models that are poorly calibrated), the debugging trail leads back here.

Let's build this from the ground up.

---

## 1. Sample Space: The Universe of Possibilities

Before you can talk about how likely something is, you need to define *what can happen at all*. That's the **sample space**, denoted $\Omega$ (capital omega).

> **You Already Know This** — A sample space is the set of all possible states — like an `enum` of all possible outcomes.
>
> ```python
> class DieRoll(Enum):
>     ONE = 1
>     TWO = 2
>     THREE = 3
>     FOUR = 4
>     FIVE = 5
>     SIX = 6
>
> # The sample space Omega is the set of all enum members
> sample_space = set(DieRoll)  # |Omega| = 6
> ```
>
> If your enum doesn't cover every possible outcome, your probability calculations will be wrong. Same principle: if you define $\Omega$ incorrectly, everything downstream breaks.

### Formal Definition

The **sample space** $\Omega$ is the set of all possible outcomes of a random experiment.

**Examples:**
- Coin flip: $\Omega = \{H, T\}$
- Die roll: $\Omega = \{1, 2, 3, 4, 5, 6\}$
- Real number between 0 and 1: $\Omega = [0, 1]$

### Running Example: Movie Recommendation System

Throughout this chapter, we'll use a movie recommendation system as our running example. Imagine you're building a system that predicts user behavior.

- **User rating a movie:** $\Omega = \{1, 2, 3, 4, 5\}$ (the star ratings)
- **User genre preferences for a session:** $\Omega = \{\text{Action}, \text{Comedy}, \text{Drama}, \text{Horror}, \text{SciFi}, ...\}$
- **User watch time for a video:** $\Omega = [0, \infty)$ (continuous — any non-negative real number of minutes)

The question we want to answer: what is $P(\text{user rates a movie 5 stars})$? We'll get there. First, we need to define what an "event" is.

---

## 2. Events: The Questions You Ask About Outcomes

An **event** $A$ is any subset of the sample space: $A \subseteq \Omega$.

> **You Already Know This** — Events are subsets of states — like filtering an enum with a predicate.
>
> ```python
> # Event A: "user gives a high rating"
> high_rating = {r for r in Rating if r.value >= 4}  # {Rating.FOUR, Rating.FIVE}
>
> # Event B: "user watches action"
> watches_action = {g for g in Genre if g == Genre.ACTION}
>
> # The event is just a predicate applied to the sample space
> ```

Events are the *questions* you ask of your sample space. "Did the user rate it 5 stars?" is an event. "Did the user watch an action movie?" is an event. "Did the user watch action AND comedy?" is also an event — formed by set operations.

### Set Operations on Events

You combine events using the same set operations you use on collections in code:

| Math Notation | Meaning | Code Equivalent |
|---|---|---|
| $A \cup B$ | A or B (union) | `set_a \| set_b` |
| $A \cap B$ | A and B (intersection) | `set_a & set_b` |
| $A^c$ | not A (complement) | `sample_space - set_a` |
| $A \setminus B$ | A but not B (difference) | `set_a - set_b` |

### ASCII Diagram: Sample Space, Events, and Overlaps

Here's how to visualize a sample space with two events overlapping:

```
+-------------------------------------------------------+
|                    Sample Space  Omega                 |
|                                                       |
|   +-------------------+                               |
|   |                   |                               |
|   |    A only         |                               |
|   |                +--+--------------+                |
|   |                |  |              |                |
|   |                |A | intersection |   B only       |
|   |                |  |  (A AND B)   |                |
|   +----------------+--+              |                |
|                    |                 |                |
|                    +-----------------+                |
|                                                       |
|              (everything outside A and B              |
|               is neither A nor B)                     |
+-------------------------------------------------------+
```

- **The whole box** = $\Omega$ (all possible outcomes)
- **Circle A** = event A
- **Circle B** = event B
- **Overlap region** = $A \cap B$ (both A and B)
- **Both circles combined** = $A \cup B$ (A or B or both)
- **Outside both circles** = $(A \cup B)^c$ (neither A nor B)

### Running Example

In our movie system:
- $\Omega$ = all possible (user, movie, rating) combinations
- Event $A$ = "user watches an action movie"
- Event $B$ = "user watches a comedy movie"
- $A \cap B$ = "user watches a movie tagged as both action AND comedy" (yes, some movies are tagged with multiple genres)
- $A \cup B$ = "user watches action OR comedy (or both)"

```
+-------------------------------------------------------+
|              All User-Movie Interactions               |
|                                                       |
|   +-------------------+                               |
|   |                   |                               |
|   |  Action only      |                               |
|   |  (e.g., John Wick)+--+--------------+             |
|   |                |  |              |             |
|   |                |Action & Comedy |  Comedy only |
|   |                |(e.g., Kung Fu  |  (e.g., The  |
|   +----------------+  Panda)        |   Hangover)  |
|                    |              |             |
|                    +--------------+             |
|                                                       |
|         Drama, Horror, SciFi, etc.                    |
+-------------------------------------------------------+
```

---

## 3. The Probability Axioms: The Interface Contract

Now for the core question: what *is* a probability? Kolmogorov formalized it in 1933 with three axioms. Think of these as the interface contract that any probability system must satisfy.

> **You Already Know This** — The probability axioms are the "interface contract" that any probability system must satisfy — just like any class implementing a `Comparable` interface must provide a consistent `compareTo`, or any valid HTTP response must have a status code.
>
> ```python
> from abc import ABC, abstractmethod
>
> class ProbabilityMeasure(ABC):
>     """Any valid probability function must satisfy these three axioms."""
>
>     @abstractmethod
>     def P(self, event: set) -> float:
>         """Returns the probability of an event."""
>         pass
>
>     # Axiom 1: Non-negativity
>     # For all events A: self.P(A) >= 0
>
>     # Axiom 2: Normalization
>     # self.P(omega) == 1.0   (where omega is the full sample space)
>
>     # Axiom 3: Additivity
>     # If A ∩ B == empty_set:
>     #     self.P(A | B) == self.P(A) + self.P(B)
> ```
>
> If any implementation violates these rules, it's not a valid probability function — period. Your softmax layer satisfies them. Raw logits don't. That's why you need softmax.

### Axiom 1: Non-negativity

$$P(A) \geq 0 \quad \text{for all events } A$$

Probabilities are never negative. If your model outputs a negative "probability," something is wrong upstream (you're probably looking at logits, not probabilities).

### Axiom 2: Normalization

$$P(\Omega) = 1$$

Something must happen. The probability of the entire sample space is 1. In ML terms: if your multi-class classifier has $k$ classes, the probabilities across all $k$ classes must sum to 1. This is exactly what softmax enforces:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} \quad \Rightarrow \quad \sum_i \text{softmax}(z_i) = 1$$

### Axiom 3: Additivity (Countable Additivity)

$$\text{If } A \cap B = \emptyset, \text{ then } P(A \cup B) = P(A) + P(B)$$

For **mutually exclusive** events (events that can't both happen), probabilities add up directly.

More generally, for any countable collection of pairwise mutually exclusive events $A_1, A_2, A_3, \ldots$:

$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

**Why this matters in ML:** When your classifier outputs probabilities for classes that are mutually exclusive (a single image can only be one digit, 0-9), you're relying on this axiom. The probability of "it's a 3 or a 7" = P(it's a 3) + P(it's a 7), precisely because an image can't be both.

### Running Example

In our movie system, suppose we analyze user ratings for a particular movie:
- $P(\text{rating} = 5) = 0.20$
- $P(\text{rating} = 4) = 0.30$
- $P(\text{rating} = 3) = 0.25$
- $P(\text{rating} = 2) = 0.15$
- $P(\text{rating} = 1) = 0.10$

Let's check the axioms:
1. **Non-negativity:** All values $\geq 0$. Check.
2. **Normalization:** $0.20 + 0.30 + 0.25 + 0.15 + 0.10 = 1.00$. Check.
3. **Additivity:** $P(\text{rating} \geq 4) = P(\text{rating}=4) + P(\text{rating}=5) = 0.30 + 0.20 = 0.50$. These ratings are mutually exclusive (you can't rate the same movie both 4 and 5), so they add. Check.

---

## 4. Key Formulas Derived from the Axioms

Everything else in basic probability is *derived* from those three axioms. Let's walk through the important ones.

### Complement Rule

$$P(A^c) = 1 - P(A)$$

**Derivation:** Since $A$ and $A^c$ are mutually exclusive (an outcome can't be both in $A$ and not in $A$), and $A \cup A^c = \Omega$:

$$P(\Omega) = P(A \cup A^c) = P(A) + P(A^c) = 1$$

Therefore: $P(A^c) = 1 - P(A)$

**In practice:** "What's the probability a user does NOT give 5 stars?" is easier to compute as $1 - P(\text{5 stars}) = 1 - 0.20 = 0.80$ than by adding up the probabilities of ratings 1 through 4.

### Probability Bounds

$$0 \leq P(A) \leq 1 \quad \text{for all events } A$$

This follows directly from non-negativity and the complement rule. If $P(A) > 1$, then $P(A^c) = 1 - P(A) < 0$, violating Axiom 1.

### Inclusion-Exclusion Principle

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

> **Common Mistake:** $P(A \cup B) \neq P(A) + P(B)$ unless $A$ and $B$ are mutually exclusive. This is like saying "users who like action OR comedy" isn't the sum of action-lovers and comedy-lovers if some users like both. You'd be double-counting the overlap.

```
Correct:   P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
                       ^^^^   ^^^^   ^^^^^^^^^^
                       all A + all B - double-counted overlap

Wrong:     P(A ∪ B) = P(A) + P(B)    <-- only if A ∩ B = empty set!
```

### Running Example

In our movie system:
- $P(\text{watches action}) = 0.40$
- $P(\text{watches comedy}) = 0.35$
- $P(\text{watches action AND comedy}) = 0.15$ (movies tagged as both, or users who watch both genres)

Then:

$$P(\text{watches action OR comedy}) = 0.40 + 0.35 - 0.15 = 0.60$$

If you naively added 0.40 + 0.35 = 0.75, you'd overcount by 0.15 — the users/movies in the overlap.

---

## 5. Counting and Combinatorics: How to Count Without Listing

When outcomes are equally likely, probability reduces to counting:

$$P(A) = \frac{|A|}{|\Omega|} = \frac{\text{number of favorable outcomes}}{\text{total number of outcomes}}$$

But counting gets tricky fast. That's where combinatorics comes in.

> **You Already Know This** — Permutations and combinations are just iteration patterns you already use.
>
> - **Permutations** = `itertools.permutations(items, k)` — order matters
> - **Combinations** = `itertools.combinations(items, k)` — order doesn't matter
>
> ```python
> from itertools import permutations, combinations
>
> movies = ['Inception', 'Parasite', 'Dune', 'Arrival', 'Alien']
>
> # "In how many ways can we arrange 3 movies in a ranked list?"
> # Order matters -> permutations
> ranked_lists = list(permutations(movies, 3))
> print(len(ranked_lists))  # 5 * 4 * 3 = 60
>
> # "In how many ways can we pick 3 movies for a watchlist?"
> # Order doesn't matter -> combinations
> watchlists = list(combinations(movies, 3))
> print(len(watchlists))    # 5! / (3! * 2!) = 10
> ```

### Fundamental Counting Principle

If experiment 1 has $m$ outcomes and experiment 2 has $n$ outcomes, the combined experiment has $m \times n$ outcomes. This is just a Cartesian product.

### Permutations (Order Matters)

The number of ways to arrange $k$ items from $n$ distinct items:

$$P(n, k) = \frac{n!}{(n-k)!}$$

**Example:** You want to display a top-3 ranked movie list from 10 candidates. How many distinct rankings exist?

$$P(10, 3) = \frac{10!}{7!} = 10 \times 9 \times 8 = 720$$

### Combinations (Order Doesn't Matter)

The number of ways to *choose* $k$ items from $n$ distinct items (ignoring order):

$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

**Example:** You want to select 3 movies from 10 for an unordered recommendation set. How many distinct sets?

$$\binom{10}{3} = \frac{10!}{3! \cdot 7!} = \frac{720}{6} = 120$$

Notice: 120 combinations vs. 720 permutations. Because each set of 3 movies can be arranged in $3! = 6$ ways, and $720 / 6 = 120$.

### Running Example

In our movie system, there are 1000 movies. A user has watched 50 of them.

- "How many distinct 5-movie recommendation lists can we generate?" (order matters for UI ranking)

$$P(950, 5) = 950 \times 949 \times 948 \times 947 \times 946 \approx 7.7 \times 10^{14}$$

- "How many distinct 5-movie recommendation sets?" (if order is irrelevant)

$$\binom{950}{5} = \frac{950!}{5! \cdot 945!} \approx 6.4 \times 10^{12}$$

That's a vast search space — which is exactly why we need ML instead of brute force.

---

## 6. Independence: When Events Don't Affect Each Other

Two events $A$ and $B$ are **independent** if and only if:

$$P(A \cap B) = P(A) \cdot P(B)$$

This is the formal definition. Intuitively, knowing that $A$ happened tells you nothing new about whether $B$ will happen.

**Be careful:** Independence is a mathematical property, not an intuitive judgment. Events that *seem* unrelated might not be independent in your data, and events that *seem* related might be independent.

### Running Example

In our movie system:
- Let $A$ = "user watches an action movie tonight"
- Let $B$ = "user watches a comedy movie tomorrow"

Are these independent? If $P(A) = 0.40$, $P(B) = 0.35$, and $P(A \cap B) = 0.14$, then:

$$P(A) \cdot P(B) = 0.40 \times 0.35 = 0.14 = P(A \cap B)$$

So yes, in this case they're independent. The user's action movie choice tonight doesn't predict their comedy choice tomorrow.

But what about $P(\text{watches action} \mid \text{already rated comedy highly})$? That's conditional probability — and that's where things get interesting. We'll tackle that in the next chapter.

> **Why independence matters in ML:** The "Naive" in Naive Bayes literally means "we assume all features are independent." That assumption is almost always wrong, but the algorithm works surprisingly well anyway. Understanding independence helps you know when this assumption will hurt you.

---

## 7. Probability as Area: A Visual Intuition

Here's a powerful way to think about probability: **probability is area**.

```
+-------------------------------------------------------+
|                                                       |
|                 Total area = 1.0                      |
|                 (the whole sample space)               |
|                                                       |
|   +-------------------+                               |
|   |                   |                               |
|   |   P(A) = area     |                               |
|   |   of this region  +--+--------------+             |
|   |   = 0.40          |  |              |             |
|   |                   |  | P(A ∩ B)     | P(B only)   |
|   |                   |  | = 0.15       | = 0.20      |
|   +-------------------+--+              |             |
|                       |                 |             |
|                       +-----------------+             |
|                                                       |
|   P(A only) = P(A) - P(A∩B) = 0.40 - 0.15 = 0.25    |
|   P(B) = P(B only) + P(A∩B) = 0.20 + 0.15 = 0.35    |
|   P(A∪B) = 0.25 + 0.15 + 0.20 = 0.60                |
|   P(neither) = 1 - 0.60 = 0.40                       |
+-------------------------------------------------------+
```

This "area" intuition is exactly right for continuous distributions too. When you integrate a probability density function, you're literally computing the area under the curve. The total area under any PDF must equal 1 (Axiom 2).

---

## 8. Code: Putting It All Together

Let's implement everything we've discussed, using our movie recommendation system.

```python
import numpy as np
from itertools import product, combinations, permutations
from collections import Counter

# =============================================================================
# Part 1: Sample Spaces and Events — Movie Rating System
# =============================================================================

def movie_rating_probability():
    """
    Simulate a movie rating system and verify probability axioms.
    Running example: P(user rates a movie 5 stars)
    """
    np.random.seed(42)

    # --- Define the sample space ---
    # Possible ratings a user can give
    sample_space = [1, 2, 3, 4, 5]

    # Simulate 100,000 user ratings (with realistic distribution)
    # Most ratings cluster around 3-4 stars (common in real systems)
    rating_probs = [0.10, 0.15, 0.25, 0.30, 0.20]
    ratings = np.random.choice(sample_space, size=100_000, p=rating_probs)

    # --- Compute empirical probabilities ---
    counts = Counter(ratings)
    total = len(ratings)

    print("Movie Rating Probability Analysis")
    print("=" * 50)
    print(f"Total ratings simulated: {total:,}")
    print()

    for star in sample_space:
        empirical = counts[star] / total
        theoretical = rating_probs[star - 1]
        print(f"  P(rating = {star}) = {empirical:.4f}  (theoretical: {theoretical:.4f})")

    # --- Verify Axiom 2: Normalization ---
    total_prob = sum(counts[s] / total for s in sample_space)
    print(f"\nAxiom 2 check: sum of all P(rating) = {total_prob:.6f}  (should be 1.0)")

    # --- Event: P(user rates 5 stars) ---
    p_five_stars = counts[5] / total
    print(f"\nRunning Example: P(user rates 5 stars) = {p_five_stars:.4f}")

    # --- Complement: P(user does NOT rate 5 stars) ---
    p_not_five = 1 - p_five_stars
    print(f"Complement: P(NOT 5 stars) = 1 - {p_five_stars:.4f} = {p_not_five:.4f}")

    # --- Event: P(high rating >= 4) using additivity ---
    p_four = counts[4] / total
    p_high = p_four + p_five_stars  # Mutually exclusive: can't be both 4 AND 5
    print(f"Additivity: P(rating >= 4) = P(4) + P(5) = {p_four:.4f} + {p_five_stars:.4f} = {p_high:.4f}")

movie_rating_probability()

# =============================================================================
# Part 2: Set Operations — Genre Overlap Analysis
# =============================================================================

def genre_overlap_analysis():
    """
    Demonstrate inclusion-exclusion with genre preferences.
    Running example: P(user watches action AND comedy), P(action OR comedy)
    """
    np.random.seed(42)
    n_users = 100_000

    # Simulate genre preferences
    # Each user independently has some probability of liking each genre
    p_action = 0.40
    p_comedy = 0.35
    p_both = 0.15  # Users who like BOTH action and comedy

    # Generate correlated preferences
    # (In real systems, genre preferences are correlated)
    likes_action = np.zeros(n_users, dtype=bool)
    likes_comedy = np.zeros(n_users, dtype=bool)

    # Users who like both
    both_mask = np.random.random(n_users) < p_both
    likes_action[both_mask] = True
    likes_comedy[both_mask] = True

    # Additional action-only users
    action_only_prob = p_action - p_both  # 0.25
    action_only_mask = (~both_mask) & (np.random.random(n_users) < action_only_prob / (1 - p_both))
    likes_action[action_only_mask] = True

    # Additional comedy-only users
    comedy_only_prob = p_comedy - p_both  # 0.20
    comedy_only_mask = (~both_mask & ~action_only_mask) & \
                       (np.random.random(n_users) < comedy_only_prob / (1 - p_both - action_only_prob))
    likes_comedy[comedy_only_mask] = True

    # Compute empirical probabilities
    emp_action = np.mean(likes_action)
    emp_comedy = np.mean(likes_comedy)
    emp_both = np.mean(likes_action & likes_comedy)
    emp_either = np.mean(likes_action | likes_comedy)

    print("\n\nGenre Overlap Analysis (Inclusion-Exclusion)")
    print("=" * 50)
    print(f"P(action)           = {emp_action:.4f}")
    print(f"P(comedy)           = {emp_comedy:.4f}")
    print(f"P(action AND comedy) = {emp_both:.4f}")
    print(f"P(action OR comedy)  = {emp_either:.4f}")

    # Verify inclusion-exclusion
    ie_result = emp_action + emp_comedy - emp_both
    print(f"\nInclusion-Exclusion verification:")
    print(f"  P(A) + P(B) - P(A ∩ B) = {emp_action:.4f} + {emp_comedy:.4f} - {emp_both:.4f} = {ie_result:.4f}")
    print(f"  P(A ∪ B) directly       = {emp_either:.4f}")
    print(f"  Match: {abs(ie_result - emp_either) < 0.01}")

    # Common mistake demonstration
    naive_sum = emp_action + emp_comedy
    print(f"\n  WRONG: P(A) + P(B)     = {naive_sum:.4f}  (overcounts by ~{naive_sum - emp_either:.4f})")
    print(f"  RIGHT: P(A) + P(B) - P(A ∩ B) = {ie_result:.4f}")

genre_overlap_analysis()

# =============================================================================
# Part 3: Verifying the Kolmogorov Axioms Programmatically
# =============================================================================

def verify_kolmogorov_axioms():
    """
    Explicitly verify all three axioms hold for a probability function.
    This is what you'd do to sanity-check a model's output.
    """
    print("\n\nVerifying Kolmogorov Axioms (Model Output Sanity Check)")
    print("=" * 50)

    # Simulated softmax output from a 5-class classifier
    # (e.g., predicting star rating 1-5)
    model_output = np.array([0.05, 0.10, 0.25, 0.35, 0.25])
    classes = [1, 2, 3, 4, 5]

    print(f"Model output (softmax): {model_output}")

    # Axiom 1: Non-negativity
    axiom_1 = all(p >= 0 for p in model_output)
    print(f"\nAxiom 1 (Non-negativity): all P >= 0? {axiom_1}")

    # Axiom 2: Normalization
    total = sum(model_output)
    axiom_2 = abs(total - 1.0) < 1e-10
    print(f"Axiom 2 (Normalization): sum = {total:.10f}, equals 1? {axiom_2}")

    # Axiom 3: Additivity for mutually exclusive events
    # Event A: rating is 4 or 5 (high rating)
    # Event B: rating is 1 or 2 (low rating)
    # These are mutually exclusive
    p_high = model_output[3] + model_output[4]  # P(4) + P(5)
    p_low = model_output[0] + model_output[1]   # P(1) + P(2)
    p_high_or_low = p_high + p_low               # Should equal P({1,2,4,5})
    p_direct = sum(model_output[i] for i in [0, 1, 3, 4])

    axiom_3 = abs(p_high_or_low - p_direct) < 1e-10
    print(f"Axiom 3 (Additivity): P(high) + P(low) = {p_high_or_low:.4f}, "
          f"P(high ∪ low) = {p_direct:.4f}, match? {axiom_3}")

    # --- Now show what INVALID probabilities look like ---
    print("\n--- Detecting Invalid Probability Outputs ---")

    # Bad output 1: raw logits (not probabilities!)
    logits = np.array([1.2, -0.5, 2.1, 3.0, 0.8])
    print(f"\nRaw logits: {logits}")
    print(f"  Axiom 1 violated? {any(l < 0 for l in logits)} (has negative values)")
    print(f"  Axiom 2 violated? sum = {sum(logits):.2f} (not 1.0)")
    print(f"  --> These are NOT probabilities. Apply softmax first.")

    # Fix with softmax
    softmax = np.exp(logits) / np.sum(np.exp(logits))
    print(f"\nAfter softmax: {softmax.round(4)}")
    print(f"  All >= 0? {all(p >= 0 for p in softmax)}")
    print(f"  Sum = {sum(softmax):.10f}")

verify_kolmogorov_axioms()

# =============================================================================
# Part 4: Counting and Combinatorics for Recommendations
# =============================================================================

def counting_for_recommendations():
    """
    Demonstrate counting principles in the context of recommendation systems.
    """
    print("\n\nCounting & Combinatorics for Recommendations")
    print("=" * 50)

    movies = ['Inception', 'Parasite', 'Dune', 'Arrival', 'Alien',
              'Coco', 'Up', 'Joker', 'Tenet', 'Her']

    # --- Fundamental Counting Principle ---
    n_movies = len(movies)
    n_time_slots = 3  # morning, afternoon, evening
    print(f"\nFundamental Counting Principle:")
    print(f"  {n_movies} movies x {n_time_slots} time slots = "
          f"{n_movies * n_time_slots} possible (movie, time) pairs")

    # --- Permutations: ranked recommendation list ---
    k = 3
    ranked_lists = list(permutations(movies, k))
    theoretical = 1
    for i in range(k):
        theoretical *= (n_movies - i)
    print(f"\nPermutations (order matters):")
    print(f"  Top-{k} ranked lists from {n_movies} movies:")
    print(f"  P({n_movies}, {k}) = {n_movies}! / {n_movies - k}! = {theoretical}")
    print(f"  Verified by enumeration: {len(ranked_lists)}")
    print(f"  Example: {ranked_lists[0]}")

    # --- Combinations: unordered recommendation set ---
    recommendation_sets = list(combinations(movies, k))
    from math import comb
    theoretical_comb = comb(n_movies, k)
    print(f"\nCombinations (order doesn't matter):")
    print(f"  Choose-{k} sets from {n_movies} movies:")
    print(f"  C({n_movies}, {k}) = {n_movies}! / ({k}! * {n_movies - k}!) = {theoretical_comb}")
    print(f"  Verified by enumeration: {len(recommendation_sets)}")
    print(f"  Example: {recommendation_sets[0]}")

    print(f"\n  Ratio: {len(ranked_lists)} / {len(recommendation_sets)} = "
          f"{len(ranked_lists) / len(recommendation_sets):.0f} = {k}! "
          f"(each set has {k}! orderings)")

    # --- Probability from counting ---
    # If recommendations are random, what's P(a specific set of 3 is chosen)?
    p_specific = 1 / theoretical_comb
    print(f"\nProbability (uniform random):")
    print(f"  P(specific 3-movie set) = 1 / {theoretical_comb} = {p_specific:.6f}")

    # What's P(at least one of your 3 favorites is in a random-3 recommendation)?
    favorites = {'Inception', 'Parasite', 'Dune'}
    sets_with_favorite = [s for s in recommendation_sets
                          if len(set(s) & favorites) >= 1]
    p_has_favorite = len(sets_with_favorite) / len(recommendation_sets)
    print(f"\n  P(at least one favorite in random top-3) = "
          f"{len(sets_with_favorite)}/{len(recommendation_sets)} = {p_has_favorite:.4f}")

    # Using complement: P(at least one) = 1 - P(none)
    non_favorites = [m for m in movies if m not in favorites]
    sets_without = len(list(combinations(non_favorites, k)))
    p_none = sets_without / theoretical_comb
    p_at_least_one_complement = 1 - p_none
    print(f"  Complement check: 1 - P(no favorites) = 1 - {p_none:.4f} = "
          f"{p_at_least_one_complement:.4f}")

counting_for_recommendations()

# =============================================================================
# Part 5: Independence Check
# =============================================================================

def check_independence():
    """
    Test whether two events are independent in simulated user data.
    Running example: P(watches action | already rated comedy highly)
    """
    np.random.seed(42)
    n_users = 100_000

    print("\n\nIndependence Check: Genre Preferences")
    print("=" * 50)

    # --- Scenario 1: Independent events ---
    # Action preference and comedy preference are independent
    p_action = 0.40
    p_comedy = 0.35

    action_indep = np.random.random(n_users) < p_action
    comedy_indep = np.random.random(n_users) < p_comedy  # Independent draws

    p_both_indep = np.mean(action_indep & comedy_indep)
    p_product_indep = np.mean(action_indep) * np.mean(comedy_indep)

    print(f"\nScenario 1: Independent preferences")
    print(f"  P(action) = {np.mean(action_indep):.4f}")
    print(f"  P(comedy) = {np.mean(comedy_indep):.4f}")
    print(f"  P(action AND comedy) = {p_both_indep:.4f}")
    print(f"  P(action) * P(comedy) = {p_product_indep:.4f}")
    print(f"  Independent? {abs(p_both_indep - p_product_indep) < 0.01}")

    # --- Scenario 2: Correlated events (NOT independent) ---
    # Users who like comedy are MORE likely to also like action
    comedy_corr = np.random.random(n_users) < p_comedy
    # If user likes comedy, 60% chance they like action; otherwise 30%
    action_given_comedy = np.where(comedy_corr,
                                   np.random.random(n_users) < 0.60,
                                   np.random.random(n_users) < 0.30)

    p_both_corr = np.mean(action_given_comedy & comedy_corr)
    p_product_corr = np.mean(action_given_comedy) * np.mean(comedy_corr)

    print(f"\nScenario 2: Correlated preferences (comedy fans like action more)")
    print(f"  P(action) = {np.mean(action_given_comedy):.4f}")
    print(f"  P(comedy) = {np.mean(comedy_corr):.4f}")
    print(f"  P(action AND comedy) = {p_both_corr:.4f}")
    print(f"  P(action) * P(comedy) = {p_product_corr:.4f}")
    print(f"  Independent? {abs(p_both_corr - p_product_corr) < 0.01}")
    print(f"  --> Knowing comedy preference DOES change P(action). Not independent.")

check_independence()
```

---

## 9. Common Mistakes

These are the probability errors that bite engineers most often in ML work.

### Mistake 1: Adding Probabilities of Non-Mutually-Exclusive Events

$P(A \cup B) \neq P(A) + P(B)$ unless $A \cap B = \emptyset$.

"Users who like action OR comedy" is NOT "users who like action" + "users who like comedy," because some users like both. You're double-counting the overlap.

**Fix:** Use inclusion-exclusion: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.

### Mistake 2: Confusing Logits with Probabilities

Raw model outputs (logits) can be negative and don't sum to 1. They violate Axioms 1 and 2. Always apply softmax (or sigmoid for binary classification) before interpreting outputs as probabilities.

```python
# WRONG: treating logits as probabilities
logits = model(x)  # Can be [-2.1, 0.5, 3.2]
# These are NOT probabilities!

# RIGHT: convert to probabilities first
probs = torch.softmax(logits, dim=-1)  # Now they satisfy all 3 axioms
```

### Mistake 3: Assuming Independence Without Checking

Naive Bayes assumes feature independence. That's often wrong. If your features are highly correlated (e.g., "word count" and "character count" in an email), the model's probability estimates will be poorly calibrated even if classification accuracy is decent.

### Mistake 4: Forgetting the Sample Space

If you're computing "P(user clicks an ad)" but your sample space only includes users who saw the ad, you get a very different answer than if your sample space includes all users (most of whom never saw the ad). Always be explicit about $\Omega$.

---

## 10. ML Connections

Here's where probability foundations show up in the ML systems you work with:

| ML Concept | Probability Foundation |
|---|---|
| **Softmax output** | Satisfies all 3 Kolmogorov axioms (non-negative, sums to 1, additive for exclusive classes) |
| **Cross-entropy loss** | $L = -\sum_i y_i \log(\hat{y}_i)$ — derived directly from probability theory |
| **Dropout** | Each neuron has probability $p$ of being dropped — a Bernoulli random variable |
| **Data augmentation** | Each transform applied with some probability — sampling from a distribution over transforms |
| **Naive Bayes** | Built entirely on probability axioms + independence assumption |
| **Logistic regression** | Outputs $P(\text{class}=1 \mid x)$ — a valid probability via the sigmoid function |
| **Random forests** | "Random" = probabilistic sampling of features and data points at each split |
| **Monte Carlo methods** | Estimate intractable quantities using random sampling from probability distributions |

---

## Exercises

### Exercise 1: Movie Rating Probabilities

You have a movie with the following rating distribution from 10,000 users:
- 5 stars: 2,500 users
- 4 stars: 3,000 users
- 3 stars: 2,000 users
- 2 stars: 1,500 users
- 1 star: 1,000 users

Calculate:
1. $P(\text{rating} = 5)$
2. $P(\text{rating} \geq 4)$
3. $P(\text{rating} < 3)$ using the complement rule
4. Verify that all three Kolmogorov axioms hold.

**Solution:**

```python
from collections import Counter

ratings = {5: 2500, 4: 3000, 3: 2000, 2: 1500, 1: 1000}
total = sum(ratings.values())  # 10000

# 1. P(rating = 5)
p_5 = ratings[5] / total
print(f"P(rating = 5) = {p_5:.4f}")  # 0.25

# 2. P(rating >= 4) — mutually exclusive events, so we can add
p_gte4 = (ratings[4] + ratings[5]) / total
print(f"P(rating >= 4) = {p_gte4:.4f}")  # 0.55

# 3. P(rating < 3) using complement
# P(rating >= 3) = P(3) + P(4) + P(5)
p_gte3 = (ratings[3] + ratings[4] + ratings[5]) / total
p_lt3 = 1 - p_gte3
print(f"P(rating < 3) = 1 - P(rating >= 3) = 1 - {p_gte3:.4f} = {p_lt3:.4f}")  # 0.25

# 4. Verify axioms
probs = {k: v / total for k, v in ratings.items()}
print(f"\nAxiom 1 (non-negativity): all >= 0? {all(p >= 0 for p in probs.values())}")
print(f"Axiom 2 (normalization): sum = {sum(probs.values()):.4f}")
print(f"Axiom 3 (additivity): P(4 or 5) = {probs[4] + probs[5]:.4f} = "
      f"P(4) + P(5) = {probs[4]:.4f} + {probs[5]:.4f}")
```

### Exercise 2: Inclusion-Exclusion in a Recommendation System

Your recommendation system tracks user genre preferences:
- 45% of users watch action movies
- 30% of users watch sci-fi movies
- 12% of users watch both action AND sci-fi

Calculate:
1. $P(\text{action OR sci-fi})$
2. $P(\text{neither action nor sci-fi})$
3. Are "watches action" and "watches sci-fi" independent?

**Solution:**

```python
p_action = 0.45
p_scifi = 0.30
p_both = 0.12

# 1. Inclusion-exclusion
p_either = p_action + p_scifi - p_both
print(f"P(action OR sci-fi) = {p_action} + {p_scifi} - {p_both} = {p_either}")  # 0.63

# 2. Complement
p_neither = 1 - p_either
print(f"P(neither) = 1 - {p_either} = {p_neither}")  # 0.37

# 3. Independence check: P(A ∩ B) == P(A) * P(B)?
p_product = p_action * p_scifi
print(f"P(action) * P(sci-fi) = {p_product:.4f}")  # 0.135
print(f"P(action AND sci-fi)  = {p_both}")           # 0.12
print(f"Independent? {abs(p_product - p_both) < 0.001}")  # False!
# P(A∩B) < P(A)*P(B), meaning the genres are slightly negatively correlated.
# Users who like action are slightly LESS likely to also like sci-fi than
# you'd expect if the preferences were independent.
```

### Exercise 3: Combinatorics for A/B Testing

You're running an A/B test with 5 different recommendation algorithms. You want to select 2 of them for a head-to-head comparison.

1. How many distinct pairs can you form? (Use combinations.)
2. If you also care about which algorithm is "control" vs. "treatment," how many ordered pairs exist? (Use permutations.)
3. If you pick a pair uniformly at random, what is $P(\text{your favorite algorithm is in the pair})$?

**Solution:**

```python
from math import comb, perm

n_algorithms = 5

# 1. Unordered pairs (combinations)
pairs = comb(n_algorithms, 2)
print(f"Distinct pairs: C(5,2) = {pairs}")  # 10

# 2. Ordered pairs (permutations)
ordered_pairs = perm(n_algorithms, 2)
print(f"Ordered pairs: P(5,2) = {ordered_pairs}")  # 20

# 3. P(favorite is in a random pair)
# Pairs containing your favorite: C(4,1) = 4 (pair it with any of the other 4)
pairs_with_favorite = comb(n_algorithms - 1, 1)
p_favorite = pairs_with_favorite / pairs
print(f"P(favorite in pair) = {pairs_with_favorite}/{pairs} = {p_favorite:.2f}")  # 0.40

# Alternative: complement
# Pairs WITHOUT your favorite: C(4,2) = 6
pairs_without = comb(n_algorithms - 1, 2)
p_favorite_complement = 1 - pairs_without / pairs
print(f"Complement check: 1 - {pairs_without}/{pairs} = {p_favorite_complement:.2f}")  # 0.40
```

---

## Summary

Here's what you now have in your toolkit:

| Concept | Definition | Engineer's Mental Model |
|---|---|---|
| **Sample Space** $\Omega$ | Set of all possible outcomes | `Enum` of all possible states |
| **Event** $A$ | Subset $A \subseteq \Omega$ | Predicate filter on the enum |
| **Axiom 1** | $P(A) \geq 0$ | No negative probabilities |
| **Axiom 2** | $P(\Omega) = 1$ | Probabilities sum to 1 (softmax guarantees this) |
| **Axiom 3** | $P(A \cup B) = P(A) + P(B)$ if $A \cap B = \emptyset$ | Additive for non-overlapping events |
| **Complement** | $P(A^c) = 1 - P(A)$ | "P(not A)" is often easier to compute |
| **Inclusion-Exclusion** | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | Don't double-count the overlap |
| **Permutations** | $P(n,k) = n!/(n-k)!$ | `itertools.permutations` — order matters |
| **Combinations** | $\binom{n}{k} = n!/(k!(n-k)!)$ | `itertools.combinations` — order doesn't matter |
| **Independence** | $P(A \cap B) = P(A) \cdot P(B)$ | Knowing A tells you nothing about B |

The Kolmogorov axioms are the foundation — every probability distribution, every model output, every loss function in ML must respect them. When things go wrong, checking these axioms is often the fastest way to find the bug.

---

> **What's Next** — You can compute $P(A)$ and $P(B)$. But what about $P(A \text{ given that } B \text{ happened})$? Conditional probability and Bayes' theorem are where probability gets truly powerful for ML. In our movie system: what is $P(\text{user watches action} \mid \text{already rated comedy highly})$? That's the question recommendation engines actually need to answer, and it's exactly where we're headed next.

**Next**: [Chapter 2: Conditional Probability](02-conditional-probability.md)
