# Sets and Logic

> **Building On** -- You learned the notation. Now let's use it. Sets and logic are the grammar of mathematics -- they let you make precise statements about collections and reason about truth.

Every SQL query you've ever written uses set theory. `UNION`, `INTERSECT`, `EXCEPT` -- those are set operations with a syntax highlighter. And every `if/else` you've written is propositional logic. You've been doing this math for years. Now let's name it properly.

---

## 1. Sets: Collections Without Duplicates

### The Intuition You Already Have

You know what a Python `set` is: an unordered collection of unique elements. That is exactly what a mathematical set is.

```python
features = {"age", "income", "education", "age"}  # Python removes the duplicate
print(features)  # {'age', 'income', 'education'}
```

> **You Already Know This** -- A Python `set`, a SQL table's distinct rows, a Redis set -- these are all implementations of the same mathematical idea. No duplicates, no ordering.

### Formal Definition

A **set** is a well-defined collection of distinct objects. We write them with curly braces:

$$A = \{1, 2, 3\}$$

**Key properties:**

| Property | Math | Code Equivalent |
|----------|------|-----------------|
| No duplicates | $\{1, 1, 2\} = \{1, 2\}$ | `set([1, 1, 2])` gives `{1, 2}` |
| No ordering | $\{1, 2, 3\} = \{3, 1, 2\}$ | `{1, 2, 3} == {3, 1, 2}` is `True` |
| Can be infinite | $\mathbb{N} = \{1, 2, 3, \ldots\}$ | Generators in Python (lazy, but conceptually infinite) |

**Set membership** -- the `in` operator:

$$x \in A \quad \text{means "x is an element of A"}$$

- $2 \in \{1, 2, 3\}$ is true -- like `2 in {1, 2, 3}` returning `True`
- $5 \notin \{1, 2, 3\}$ is true -- like `5 not in {1, 2, 3}` returning `True`

**The empty set** $\emptyset = \{\}$ -- a set with no elements. Think of it as `set()` in Python, or a SQL query that returns zero rows.

### ML Application: Your Data Is a Set

Here is the running example we will use throughout this chapter:

> **Running Example** -- Training set, validation set, test set -- disjoint sets whose union is the full dataset. That is a **partition**.

$$D = D_{\text{train}} \cup D_{\text{val}} \cup D_{\text{test}}$$

where:

$$D_{\text{train}} \cap D_{\text{val}} = \emptyset, \quad D_{\text{train}} \cap D_{\text{test}} = \emptyset, \quad D_{\text{val}} \cap D_{\text{test}} = \emptyset$$

If those intersections are NOT empty, you have **data leakage** -- one of the most common bugs in ML pipelines. The math makes the requirement precise: the sets must be **pairwise disjoint**.

---

## 2. Subsets and Power Sets

### Elements vs. Subsets

> **Common Mistake** -- `A` $\in$ `S` means "A is an element of S". `A` $\subseteq$ `S` means "A is a subset of S". Elements vs. subsets is like values vs. types.

```python
S = {1, 2, 3}

# Element check: is 2 IN the set?
2 in S          # True   --  this is  2 ∈ S

# Subset check: is {1, 2} CONTAINED in the set?
{1, 2} <= S     # True   --  this is  {1, 2} ⊆ S
{1, 2} in S     # False  --  {1, 2} is not an ELEMENT of S
```

**Subset**: $A \subseteq B$ means every element of $A$ is also in $B$.

$$A = \{1, 2\}, \quad B = \{1, 2, 3\} \quad \Rightarrow \quad A \subseteq B$$

**Proper subset**: $A \subset B$ means $A \subseteq B$ and $A \neq B$ (i.e., $B$ has something $A$ does not).

### Power Sets

The **power set** $\mathcal{P}(A)$ is the set of ALL subsets of $A$.

For $A = \{1, 2\}$:

$$\mathcal{P}(A) = \{\emptyset, \{1\}, \{2\}, \{1, 2\}\}$$

A set with $n$ elements has $2^n$ subsets. That exponent should make you nervous.

> **You Already Know This** -- The power set is all possible subsets = all possible feature combinations. If you have 20 features, there are $2^{20} = 1{,}048{,}576$ possible feature subsets. This is why exhaustive feature selection is NP-hard and why you use greedy approaches or regularization instead.

```python
from itertools import combinations

features = ["age", "income", "education"]
n = len(features)

# Generate the power set (all possible feature combinations)
power_set = []
for r in range(n + 1):
    for combo in combinations(features, r):
        power_set.append(set(combo))

print(f"Features: {features}")
print(f"All subsets ({2**n} total):")
for s in power_set:
    print(f"  {s if s else '{}'}")
```

---

## 3. Set Operations

This is where the SQL connection becomes explicit. Here are the four core set operations, side by side with code you have written hundreds of times.

### The Operations at a Glance

| Operation | Math | SQL | Python |
|-----------|------|-----|--------|
| Union | $A \cup B$ | `SELECT * FROM A UNION SELECT * FROM B` | `A \| B` |
| Intersection | $A \cap B$ | `SELECT * FROM A INTERSECT SELECT * FROM B` | `A & B` |
| Difference | $A - B$ | `SELECT * FROM A EXCEPT SELECT * FROM B` | `A - B` |
| Symmetric Difference | $A \triangle B$ | (union minus intersection) | `A ^ B` |
| Complement | $A^c$ | `SELECT * FROM U WHERE id NOT IN (SELECT id FROM A)` | `U - A` |

> **You Already Know This** -- `UNION`, `INTERSECT`, `EXCEPT` in SQL are literally set union, intersection, and difference. You have been doing set theory every time you write a query.

### Formal Definitions

**Union** ($\cup$) -- elements in $A$ OR $B$ (or both):

$$A \cup B = \{x : x \in A \text{ or } x \in B\}$$

**Intersection** ($\cap$) -- elements in $A$ AND $B$:

$$A \cap B = \{x : x \in A \text{ and } x \in B\}$$

**Complement** ($A^c$ or $\overline{A}$) -- elements NOT in $A$, relative to a universal set $U$:

$$A^c = \{x \in U : x \notin A\}$$

**Difference** ($A - B$ or $A \setminus B$) -- elements in $A$ but NOT in $B$:

$$A - B = \{x : x \in A \text{ and } x \notin B\}$$

**Symmetric Difference** ($A \triangle B$) -- elements in $A$ or $B$, but NOT both (XOR for sets):

$$A \triangle B = (A - B) \cup (B - A)$$

### ASCII Venn Diagrams

**Union** $A \cup B$ -- the entire shaded region:

```
         ┌───────────────────────────────────┐
         │            Universal Set U         │
         │                                    │
         │       ┌───────┐   ┌───────┐        │
         │      /  ░░░░░░░\ /░░░░░░░  \      │
         │     │ ░░░░░░░░░░X░░░░░░░░░░ │     │
         │     │ ░░░A░░░░░/░\░░░░B░░░░ │     │
         │     │ ░░░░░░░░/ ░ \░░░░░░░░ │     │
         │      \  ░░░░░/  ░  \░░░░░  /      │
         │       └──────┘   └──────┘          │
         │                                    │
         │     ░ = A ∪ B (everything shaded)  │
         └───────────────────────────────────┘
```

**Intersection** $A \cap B$ -- only the overlap:

```
         ┌───────────────────────────────────┐
         │            Universal Set U         │
         │                                    │
         │       ┌───────┐   ┌───────┐        │
         │      /         \ /         \       │
         │     │     A     X░░░  B     │      │
         │     │          /░░░\        │      │
         │     │         / ░░░ \       │      │
         │      \       /  ░░░  \     /       │
         │       └──────┘   └──────┘          │
         │                                    │
         │     ░ = A ∩ B (only the overlap)   │
         └───────────────────────────────────┘
```

**Difference** $A - B$ -- A without the overlap:

```
         ┌───────────────────────────────────┐
         │            Universal Set U         │
         │                                    │
         │       ┌───────┐   ┌───────┐        │
         │      /  ░░░░░░░\ /         \      │
         │     │ ░░░░░░░░░░X     B     │     │
         │     │ ░░░A░░░░░/ \          │     │
         │     │ ░░░░░░░░/   \         │     │
         │      \  ░░░░░/     \       /      │
         │       └──────┘   └──────┘          │
         │                                    │
         │     ░ = A - B (A without overlap)  │
         └───────────────────────────────────┘
```

**Complement** $A^c$ -- everything outside A:

```
         ┌───────────────────────────────────┐
         │ ░░░░░░░░░░ Universal Set U ░░░░░░ │
         │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
         │ ░░░░░░░┌───────┐░░░░░░░░░░░░░░░░ │
         │ ░░░░░░/         \░░░░░░░░░░░░░░░ │
         │ ░░░░░│     A     │░░░░░░░░░░░░░░ │
         │ ░░░░░│           │░░░░░░░░░░░░░░ │
         │ ░░░░░░\         /░░░░░░░░░░░░░░░ │
         │ ░░░░░░░└───────┘░░░░░░░░░░░░░░░░ │
         │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
         │     ░ = Aᶜ (everything outside A) │
         └───────────────────────────────────┘
```

### ML Application: Data Splits as Set Operations

```python
import numpy as np

# Full dataset: 1000 sample IDs
D = set(range(1000))

# Random split into train / val / test
indices = np.random.permutation(1000)
D_train = set(indices[:700])
D_val   = set(indices[700:850])
D_test  = set(indices[850:])

# Verify it is a valid partition
assert D_train | D_val | D_test == D            # Union = full dataset
assert D_train & D_val == set()                  # Pairwise disjoint
assert D_train & D_test == set()
assert D_val & D_test == set()

print("Valid partition: no data leakage.")
print(f"|D_train| = {len(D_train)}, |D_val| = {len(D_val)}, |D_test| = {len(D_test)}")
```

---

## 4. Propositional Logic

### The Intuition You Already Have

Every `if/else` you have ever written is propositional logic. Every `WHERE` clause in SQL. Every boolean mask in pandas. You reason about truth values constantly -- math just gives it a formal notation.

> **You Already Know This** -- Boolean logic = `if/else` statements, boolean masks in NumPy/pandas, `WHERE` clauses in SQL. The symbols change, the reasoning is identical.

### Logical Operators

| Operation | Math Symbol | Python | NumPy | SQL |
|-----------|-------------|--------|-------|-----|
| AND | $\land$ | `and` | `&` / `np.logical_and` | `AND` |
| OR | $\lor$ | `or` | `\|` / `np.logical_or` | `OR` |
| NOT | $\lnot$ | `not` | `~` / `np.logical_not` | `NOT` |
| XOR | $\oplus$ | `^` | `^` / `np.logical_xor` | (no direct equivalent) |

**AND** ($\land$) -- both must be true:

$$P \land Q \text{ is true only when both } P \text{ and } Q \text{ are true}$$

```python
if user.is_authenticated and user.has_permission("admin"):
    # P ∧ Q -- both conditions must hold
```

**OR** ($\lor$) -- at least one must be true:

$$P \lor Q \text{ is true when } P \text{ is true, } Q \text{ is true, or both}$$

```python
if request.is_GET or request.is_HEAD:
    # P ∨ Q -- either suffices
```

**NOT** ($\lnot$) -- negation:

$$\lnot P \text{ is true when } P \text{ is false}$$

### Implication

**Implication** ($\Rightarrow$): if $P$ then $Q$.

$$P \Rightarrow Q$$

This is the most misunderstood logical operator. $P \Rightarrow Q$ is **only false** when $P$ is true and $Q$ is false. In all other cases, it is true.

Think of it like a contract: "If the build passes, we deploy." If the build fails, the contract says nothing about deploying -- it is not violated either way.

| $P$ | $Q$ | $P \Rightarrow Q$ |
|-----|-----|--------------------|
| T | T | T |
| T | F | **F** |
| F | T | T |
| F | F | T |

The third and fourth rows surprise people. "False implies anything" is true because the premise was never activated. A function that is never called cannot throw an error.

**Biconditional** ($\Leftrightarrow$): $P$ if and only if $Q$.

$$P \Leftrightarrow Q \equiv (P \Rightarrow Q) \land (Q \Rightarrow P)$$

This is logical equality. In code: `P == Q` for booleans.

### Full Truth Table

Here is the complete truth table. If you have ever written unit tests for boolean logic, this is the same exhaustive enumeration of cases.

| $P$ | $Q$ | $\lnot P$ | $P \land Q$ | $P \lor Q$ | $P \oplus Q$ | $P \Rightarrow Q$ | $P \Leftrightarrow Q$ |
|-----|-----|-----------|-------------|------------|--------------|--------------------|-----------------------|
| T | T | F | T | T | F | T | T |
| T | F | F | F | T | T | F | F |
| F | T | T | F | T | T | T | F |
| F | F | T | F | F | F | T | T |

```python
# Generate the same truth table in code
print(f"{'P':<6}{'Q':<6}{'~P':<6}{'P&Q':<6}{'P|Q':<6}{'P^Q':<6}{'P=>Q':<7}{'P<=>Q':<6}")
print("-" * 49)
for P in [True, False]:
    for Q in [True, False]:
        impl = (not P) or Q           # P => Q  is equivalent to  (not P) or Q
        bic  = P == Q                  # P <=> Q is equivalent to  P == Q
        print(f"{P!s:<6}{Q!s:<6}{(not P)!s:<6}{(P and Q)!s:<6}"
              f"{(P or Q)!s:<6}{(P ^ Q)!s:<6}{impl!s:<7}{bic!s:<6}")
```

---

## 5. De Morgan's Laws

### The Intuition You Already Have

> **You Already Know This** -- You use De Morgan's laws every time you refactor a complex conditional: `not(A and B)` = `(not A) or (not B)`. That is De Morgan's first law. You have been applying it intuitively for years.

### The Laws

**In logic:**

$$\lnot(P \land Q) \equiv \lnot P \lor \lnot Q$$
$$\lnot(P \lor Q) \equiv \lnot P \land \lnot Q$$

**In set theory:**

$$(A \cap B)^c = A^c \cup B^c$$
$$(A \cup B)^c = A^c \cap B^c$$

**In English:**

- "NOT (A AND B)" is the same as "(NOT A) OR (NOT B)"
- "NOT (A OR B)" is the same as "(NOT A) AND (NOT B)"

### Code Refactoring Example

Consider this real refactoring scenario:

```python
# Before: hard to read
if not (user.is_active and user.is_verified):
    deny_access()

# After applying De Morgan's: clearer intent
if (not user.is_active) or (not user.is_verified):
    deny_access()

# Even cleaner with positive names
if user.is_inactive or user.is_unverified:
    deny_access()
```

### Proof with NumPy

```python
import numpy as np

data = np.array([1, 5, 3, 8, 2, 9, 4, 7, 6])

A = data > 3    # [F, T, F, T, F, T, T, T, T]
B = data < 7    # [T, T, T, F, T, F, T, F, T]

# De Morgan's first law: NOT(A AND B) = (NOT A) OR (NOT B)
lhs = ~(A & B)
rhs = (~A) | (~B)
print(f"NOT(A AND B):           {lhs}")
print(f"(NOT A) OR (NOT B):     {rhs}")
print(f"Equal? {np.array_equal(lhs, rhs)}")  # True

# De Morgan's second law: NOT(A OR B) = (NOT A) AND (NOT B)
lhs2 = ~(A | B)
rhs2 = (~A) & (~B)
print(f"\nNOT(A OR B):            {lhs2}")
print(f"(NOT A) AND (NOT B):    {rhs2}")
print(f"Equal? {np.array_equal(lhs2, rhs2)}")  # True
```

### ML Application: Simplifying Data Filters

In pandas, you often build complex boolean masks. De Morgan's laws let you simplify them:

```python
import pandas as pd

# Suppose you want rows that are NOT (high_income AND young)
# De Morgan says: that is the same as (NOT high_income) OR (NOT young)

mask_original  = ~((df["income"] > 100000) & (df["age"] < 30))
mask_demorgan  = (df["income"] <= 100000) | (df["age"] >= 30)

assert mask_original.equals(mask_demorgan)  # Always true, by De Morgan
```

---

## 6. Boolean Operations in NumPy: The ML Workhorse

Everything above becomes concrete when you work with arrays. Boolean masking is how you filter data, build attention masks, and construct loss functions.

```python
import numpy as np

data = np.array([1, 5, 3, 8, 2, 9, 4, 7, 6])
print("Data:", data)

# Create boolean masks -- each is a set membership indicator
gt5 = data > 5     # "the set of elements greater than 5"
lt8 = data < 8     # "the set of elements less than 8"

print(f"data > 5: {gt5}")    # [F, F, F, T, F, T, F, T, T]
print(f"data < 8: {lt8}")    # [T, T, T, F, T, F, T, F, T]

# Intersection: elements > 5 AND < 8
mask_and = gt5 & lt8
print(f"\n(data > 5) & (data < 8): {mask_and}")
print(f"Filtered (5 < x < 8):   {data[mask_and]}")   # [7, 6]

# Union: elements > 7 OR < 3
mask_or = (data > 7) | (data < 3)
print(f"\n(data > 7) | (data < 3): {mask_or}")
print(f"Filtered (x > 7 or x < 3): {data[mask_or]}")  # [1, 8, 2, 9]

# Complement: NOT (data > 5)
mask_not = ~gt5
print(f"\n~(data > 5): {mask_not}")
print(f"Filtered (x <= 5): {data[mask_not]}")          # [1, 5, 3, 2, 4]
```

> **Gotcha: Operator Precedence** -- In NumPy, `&` and `|` have HIGHER precedence than `<` and `>`. You MUST use parentheses around comparisons:
>
> ```python
> # WRONG -- evaluates as data > (5 & data) < 10
> data > 5 & data < 10
>
> # RIGHT
> (data > 5) & (data < 10)
> ```
>
> This is the number one source of boolean mask bugs. Parenthesize everything.

---

## 7. Putting It All Together: ML Data Pipeline

Here is a realistic example that uses every concept from this chapter -- sets, logic, De Morgan's, boolean masks, and partitions.

```python
import numpy as np

np.random.seed(42)

# ── Simulate a dataset ──────────────────────────────────────────────
n_samples = 1000
ages     = np.random.randint(18, 80, n_samples)
incomes  = np.random.exponential(50000, n_samples)
has_deg  = np.random.choice([True, False], n_samples, p=[0.3, 0.7])

print("=== Dataset ===")
print(f"Samples: {n_samples}")
print(f"Age range: {ages.min()} - {ages.max()}")
print(f"Income range: ${incomes.min():.0f} - ${incomes.max():.0f}")
print(f"% with degree: {has_deg.mean() * 100:.1f}%\n")

# ── Set operations: build segments via boolean masks ────────────────
young        = (ages >= 25) & (ages <= 35)          # A
high_income  = incomes > 60000                       # B
educated     = has_deg                               # C

# Intersection: A ∩ B ∩ C
target = young & high_income & educated
print(f"Target segment (young & high income & educated): {target.sum()} samples")

# Union: A ∪ B
broad_segment = young | high_income
print(f"Broad segment (young | high income): {broad_segment.sum()} samples")

# Complement: ~A
non_young = ~young
print(f"Non-young: {non_young.sum()} samples")

# Difference: A - B  (young but NOT high income)
young_low_income = young & ~high_income
print(f"Young but low income (A - B): {young_low_income.sum()} samples")

# ── Partition: train / val / test ───────────────────────────────────
indices = np.random.permutation(n_samples)
D_train = set(indices[:700])
D_val   = set(indices[700:850])
D_test  = set(indices[850:])

# Verify the partition property
assert D_train | D_val | D_test == set(range(n_samples)), "Union must be full dataset"
assert len(D_train & D_val) == 0, "Train and val must be disjoint"
assert len(D_train & D_test) == 0, "Train and test must be disjoint"
assert len(D_val & D_test) == 0, "Val and test must be disjoint"

print(f"\nPartition verified: |train|={len(D_train)}, |val|={len(D_val)}, |test|={len(D_test)}")
print("No data leakage: all pairwise intersections are empty.")

# ── De Morgan's in action ──────────────────────────────────────────
# "NOT (young AND high_income)" vs "(NOT young) OR (NOT high_income)"
dm_lhs = ~(young & high_income)
dm_rhs = (~young) | (~high_income)
assert np.array_equal(dm_lhs, dm_rhs), "De Morgan's law holds"
print(f"\nDe Morgan verified: ~(A & B) == (~A) | (~B) for all {n_samples} samples")
```

---

## 8. Common Mistakes

These are the bugs you will encounter -- in proofs and in code.

| Mistake | Why It Is Wrong | Fix |
|---------|----------------|-----|
| Confusing $\in$ and $\subseteq$ | $\{1\} \in \{1, 2\}$ is **false** (the element `1` is in the set, but the set $\{1\}$ is not an element). $\{1\} \subseteq \{1, 2\}$ is **true**. | Elements vs. subsets is like values vs. types. Use `in` for elements, `<=` for subsets in Python. |
| AND/OR mixup | `(age > 25) & (age > 35)` is not the same as `(age > 25) | (age > 35)` | Draw a Venn diagram. AND narrows, OR widens. |
| Missing parentheses in NumPy | `data > 5 & data < 10` silently gives wrong results | Always: `(data > 5) & (data < 10)` |
| Forgetting De Morgan's | Writing `not(A and B)` when `(not A) or (not B)` is clearer | Refactor complex conditionals with De Morgan's |
| Implication direction | "$P$ implies $Q$" is NOT the same as "$Q$ implies $P$" | "If it rains, the ground is wet" does NOT mean "if the ground is wet, it rained" |
| Empty set edge cases | Forgetting that $A \cap \emptyset = \emptyset$ and $A \cup \emptyset = A$ | Test your logic with empty inputs |

---

## 9. Exercises

### Exercise 1: Set Operations (Pencil and Paper)

Given $A = \{1, 2, 3, 4\}$ and $B = \{3, 4, 5, 6\}$, compute by hand, then verify in code:

- $A \cup B$
- $A \cap B$
- $A - B$
- $B - A$
- $A \triangle B$

**Solution:**

```python
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

print(f"A ∪ B = {A | B}")       # {1, 2, 3, 4, 5, 6}
print(f"A ∩ B = {A & B}")       # {3, 4}
print(f"A - B = {A - B}")       # {1, 2}
print(f"B - A = {B - A}")       # {5, 6}
print(f"A △ B = {A ^ B}")       # {1, 2, 5, 6}
```

### Exercise 2: Truth Table Construction

Construct the truth table for: $(P \lor Q) \land \lnot R$

This has 3 variables, so you need $2^3 = 8$ rows. Think of it as testing all branches.

**Solution:**

```python
print(f"{'P':<7}{'Q':<7}{'R':<7}{'P∨Q':<7}{'~R':<7}{'(P∨Q)∧~R':<10}")
print("-" * 45)
for P in [True, False]:
    for Q in [True, False]:
        for R in [True, False]:
            p_or_q = P or Q
            not_r = not R
            result = p_or_q and not_r
            print(f"{P!s:<7}{Q!s:<7}{R!s:<7}{p_or_q!s:<7}{not_r!s:<7}{result!s:<10}")
```

### Exercise 3: Data Filtering with Boolean Masks

Given a dataset with columns `age`, `salary`, and `department`, write boolean expressions to select:

a) Employees over 30 in Engineering
b) Employees under 25 OR with salary over 100k
c) Employees NOT in Sales
d) (Bonus) Rewrite (a) using De Morgan's to express who is NOT in the target segment

**Solution:**

```python
import numpy as np
import pandas as pd

np.random.seed(42)
df = pd.DataFrame({
    "age": np.random.randint(22, 60, 100),
    "salary": np.random.randint(40000, 150000, 100),
    "department": np.random.choice(["Engineering", "Sales", "Marketing"], 100)
})

# a) Over 30 in Engineering: intersection of two conditions
mask_a = (df["age"] > 30) & (df["department"] == "Engineering")
print(f"a) Over 30 in Engineering: {mask_a.sum()} employees")

# b) Under 25 OR salary > 100k: union of two conditions
mask_b = (df["age"] < 25) | (df["salary"] > 100000)
print(f"b) Under 25 OR salary > 100k: {mask_b.sum()} employees")

# c) NOT in Sales: complement
mask_c = df["department"] != "Sales"
print(f"c) Not in Sales: {mask_c.sum()} employees")

# d) De Morgan's: NOT(over 30 AND engineering) = (NOT over 30) OR (NOT engineering)
mask_d_direct  = ~mask_a
mask_d_demorgan = (df["age"] <= 30) | (df["department"] != "Engineering")
assert mask_d_direct.equals(mask_d_demorgan)
print(f"d) NOT target (De Morgan's verified): {mask_d_direct.sum()} employees")
```

### Exercise 4: Partition Verification

Write a function that takes a list of sets and verifies whether they form a valid partition of a universal set. It should check: (1) the union equals the universal set, and (2) all pairs are disjoint.

**Solution:**

```python
def is_partition(universal, *subsets):
    """Check if subsets form a valid partition of the universal set."""
    # Check union
    union = set()
    for s in subsets:
        union |= s
    if union != universal:
        return False, "Union does not equal universal set"

    # Check pairwise disjoint
    for i in range(len(subsets)):
        for j in range(i + 1, len(subsets)):
            if subsets[i] & subsets[j]:
                return False, f"Sets {i} and {j} overlap: {subsets[i] & subsets[j]}"

    return True, "Valid partition"

# Test with a correct partition
D = set(range(10))
D_train, D_val, D_test = {0,1,2,3,4,5,6}, {7,8}, {9}
print(is_partition(D, D_train, D_val, D_test))  # (True, 'Valid partition')

# Test with data leakage
D_train_bad = {0,1,2,3,4,5,6,7}  # overlaps with D_val
print(is_partition(D, D_train_bad, D_val, D_test))  # (False, 'Sets 0 and 1 overlap: {7}')
```

---

## 10. Summary

| Concept | Math Notation | Python / NumPy | Why It Matters for ML |
|---------|--------------|----------------|----------------------|
| Set | $\{1, 2, 3\}$ | `{1, 2, 3}` | Datasets, feature sets, label sets |
| Membership | $x \in A$ | `x in A` | Checking if a sample is in a split |
| Subset | $A \subseteq B$ | `A <= B` | "Is my feature set a subset of all features?" |
| Power set | $\mathcal{P}(A)$, size $2^n$ | `itertools.combinations` | All possible feature subsets ($2^n$ explosion) |
| Union | $A \cup B$ | `A \| B`, `UNION` | Combining datasets |
| Intersection | $A \cap B$ | `A & B`, `INTERSECT` | Finding common samples, detecting leakage |
| Difference | $A - B$ | `A - B`, `EXCEPT` | Removing outliers, exclusion filters |
| Complement | $A^c$ | `~mask` | "Everything NOT matching this condition" |
| Partition | Disjoint union | `assert` checks | Train/val/test splits, k-fold CV |
| AND | $P \land Q$ | `&`, `and` | Combining filter conditions |
| OR | $P \lor Q$ | `\|`, `or` | Broadening filter conditions |
| NOT | $\lnot P$ | `~`, `not` | Inverting masks |
| Implication | $P \Rightarrow Q$ | `(not P) or Q` | Mathematical proofs, type constraints |
| De Morgan's | $\lnot(P \land Q) = \lnot P \lor \lnot Q$ | `~(A & B) == (~A) \| (~B)` | Simplifying complex boolean masks |

---

> **What's Next** -- You can define collections and reason logically. Now: mathematical thinking -- the problem-solving strategies that turn notation into solutions.

---

*Previous: [Mathematical Language](./01-mathematical-language.md) | Next: [Mathematical Thinking](./03-mathematical-thinking.md)*
