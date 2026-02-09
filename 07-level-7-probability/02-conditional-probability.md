# Chapter 2: Conditional Probability and Bayes' Theorem

> **Building On** -- You can compute basic probabilities. But the real power comes from *updating* probabilities when you learn new information. If a user watched 3 action movies in a row, what's the probability they'll click on the next action recommendation? That question isn't about raw frequency -- it's about conditioning on what you've already observed. This chapter gives you the machinery to answer it.

---

## The Puzzle That Breaks Your Intuition

A medical test is 99% accurate. A patient tests positive. What's the probability they actually have the disease? If you said 99%, you just fell for one of the most famous probability traps. The real answer depends on how rare the disease is -- and Bayes' theorem tells you exactly how to compute it.

Let's work through it together. By the end of this chapter, you'll not only understand *why* 99% is wrong, you'll have a general framework for updating any belief with new evidence -- the same framework that powers spam filters, recommendation engines, and Bayesian A/B testing.

Here are the facts:

- The disease affects **1%** of the population: P(disease) = 0.01
- The test has **99% sensitivity** (true positive rate): P(positive | disease) = 0.99
- The test has **95% specificity** (true negative rate): P(negative | no disease) = 0.95, which means the false positive rate is P(positive | no disease) = 0.05

You test positive. What's P(disease | positive)?

Don't answer yet. First, you need a concept.

---

## Conditional Probability: Filtering Your Universe

The notation P(A|B) reads "the probability of A, given B." It answers: *if we already know B happened, how likely is A?*

The definition is clean:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad \text{provided } P(B) > 0$$

> **You Already Know This -- Filtering a Dataset**
>
> P(A|B) is just P(A AND B) / P(B). Think of it like filtering a SQL table or a pandas DataFrame. You have all your data (the sample space). You filter to only the rows where B is true. Then you ask: of those filtered rows, what fraction also have A?
>
> ```python
> # P(churned | premium_user)
> premium_users = df[df['plan'] == 'premium']        # filter to B
> churned_premium = premium_users[premium_users['churned'] == True]  # A AND B
> p_churn_given_premium = len(churned_premium) / len(premium_users)  # P(A|B)
> ```
>
> That's conditional probability. You just narrowed the universe to B, then measured A within it.

Here's the picture. When you condition on B, you're zooming into B's circle and asking how much of it overlaps with A:

```
ASCII Venn Diagram: P(A|B) = P(A AND B) / P(B)

+--------------------------------------------------+
|                  Sample Space                     |
|                                                   |
|       +-------------+                             |
|      /    A          \                            |
|     /                 \                           |
|    |         +--------+----------+                |
|    |         |////////|          |                |
|    |         |//A AND/|          |                |
|     \        |///B////|          |                |
|      \       +--------+    B    |                |
|       +------+        |         |                |
|                       |         |                |
|                       +---------+                |
|                                                   |
+--------------------------------------------------+

P(A|B) = the shaded region (A AND B) relative to ALL of B
        = P(A AND B) / P(B)

When you "condition on B," you shrink your universe to just
the B circle. The shaded overlap becomes your new numerator.
```

The key insight: conditioning *changes the denominator*. You're no longer measuring against the whole sample space. You're measuring against B.

---

## The Chain Rule: Decomposing Joint Probabilities

Rearranging the definition of conditional probability gives you the **chain rule** (also called the product rule):

$$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

This says: the probability of A AND B happening together equals the probability of B, times the probability of A *given* B already happened. Or equivalently, P(A) times P(B given A).

For three or more events, you can keep chaining:

$$P(A \cap B \cap C) = P(A) \cdot P(B|A) \cdot P(C|A \cap B)$$

In general:

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1 \cap A_2) \cdots P(A_n|A_1 \cap \cdots \cap A_{n-1})$$

If that looks familiar, it should. This is exactly how **autoregressive language models** generate text:

$$P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i | w_1, \ldots, w_{i-1})$$

Each token's probability is conditioned on everything before it. GPT doesn't predict words in isolation -- it applies the chain rule of probability at every step.

---

## The Law of Total Probability: Accounting for All Cases

Before we can solve the medical test puzzle, you need one more tool. If $\{B_1, B_2, \ldots, B_n\}$ form a **partition** of the sample space (they're mutually exclusive and together cover everything), then:

$$P(A) = \sum_{i=1}^{n} P(A|B_i) \cdot P(B_i)$$

This is the **law of total probability**. It says: to find the overall probability of A, break it down by every possible "world" (each $B_i$) and add up the weighted contributions.

For our medical test, there are two worlds: the patient has the disease, or they don't.

$$P(\text{positive}) = P(\text{positive}|\text{disease}) \cdot P(\text{disease}) + P(\text{positive}|\text{no disease}) \cdot P(\text{no disease})$$

Let's compute:

$$P(\text{positive}) = 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0099 + 0.0495 = 0.0594$$

About 5.94% of all people will test positive. Notice that most of those positives come from the second term -- healthy people who got false positives -- because healthy people vastly outnumber sick ones.

---

## Bayes' Theorem: The Answer

Now you have everything you need. The chain rule gave us two ways to write P(A AND B):

$$P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

Divide both sides by P(B), and you get **Bayes' theorem**:

$$\boxed{P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}}$$

Each piece has a name:

| Term | Name | What It Means |
|------|------|---------------|
| P(A\|B) | **Posterior** | What you want: your updated belief about A after observing B |
| P(B\|A) | **Likelihood** | How probable is the evidence B if A were true? |
| P(A) | **Prior** | Your belief about A *before* seeing any evidence |
| P(B) | **Evidence** | The total probability of observing B (normalizing constant) |

> **You Already Know This -- Prior/Posterior as Before/After Data**
>
> Think of it like this: you launch a feature and you have an *initial estimate* (prior) of the conversion rate. Users start clicking. Each day of data *updates* your estimate (posterior). Bayes' theorem is the exact formula for that update. It's the engine behind Bayesian A/B testing frameworks -- they don't just ask "is variant B better?" They ask "given the data so far, what's the *probability distribution* over how much better B is?"

### Solving the Medical Test Puzzle

Let's apply Bayes' theorem to finally answer our opening question.

$$P(\text{disease}|\text{positive}) = \frac{P(\text{positive}|\text{disease}) \cdot P(\text{disease})}{P(\text{positive})}$$

$$= \frac{0.99 \times 0.01}{0.0594} = \frac{0.0099}{0.0594} \approx 0.1667$$

**The answer is about 16.7%.** Not 99%. Not even close.

A 99%-accurate test, a positive result, and the patient still has only a 1-in-6 chance of actually being sick. Why? Because the disease is *rare*. The prior probability is so low (1%) that even a highly accurate test can't overcome the base rate. Most positive results come from the 99% of healthy people who occasionally trigger a false positive.

This is the **base rate fallacy** -- ignoring the prior and treating the test accuracy as the answer.

### Watching Bayes Update

What if the patient takes a *second* independent test and it's also positive? Now our prior is no longer 0.01 -- it's the posterior from the first test: 0.1667. Watch what happens:

$$P(\text{disease}|\text{two positives}) = \frac{0.99 \times 0.1667}{P(\text{positive}_2)}$$

Computing the new evidence:

$$P(\text{positive}_2) = 0.99 \times 0.1667 + 0.05 \times 0.8333 = 0.1650 + 0.0417 = 0.2067$$

$$P(\text{disease}|\text{two positives}) = \frac{0.1650}{0.2067} \approx 0.798$$

After two positive tests, the probability jumps to about **80%**. That's Bayes in action: each piece of evidence ratchets the posterior closer to certainty. The prior matters most when evidence is scarce.

---

## Independence: When Conditioning Doesn't Matter

Events A and B are **independent** if knowing one tells you nothing about the other:

$$P(A|B) = P(A) \quad \text{or equivalently} \quad P(A \cap B) = P(A) \cdot P(B)$$

If A and B are independent, the chain rule simplifies: P(A AND B) = P(A) * P(B). No conditioning needed.

> **You Already Know This -- No Shared State**
>
> Independence is like two microservices with no shared state and no communication channel. Knowing that Service A is overloaded tells you nothing about whether Service B is overloaded. They're independent. But if they share a database, knowing A is overloaded *does* tell you something about B (the shared DB might be the bottleneck). That's dependence.

### Conditional Independence

A and B can be dependent in general but become independent once you condition on C:

$$P(A \cap B | C) = P(A|C) \cdot P(B|C)$$

This is **conditional independence**, and it's absolutely critical in ML. Naive Bayes (which we'll build shortly) assumes that all features are conditionally independent given the class label. The features might be correlated in general (e.g., emails containing "free" also tend to contain "click"), but Naive Bayes assumes that *given you know it's spam*, knowing it contains "free" tells you nothing additional about whether it contains "click."

This assumption is almost always violated in practice. And yet Naive Bayes works surprisingly well. That's one of the beautiful puzzles of ML.

---

## Marginalization: Summing Out What You Don't Care About

Sometimes you have a joint distribution P(A, B) but you only care about P(A). You **marginalize** by summing (or integrating) over the variable you want to eliminate:

**Discrete:**
$$P(A) = \sum_b P(A, B=b) = \sum_b P(A|B=b) \cdot P(B=b)$$

**Continuous:**
$$p(x) = \int p(x, y) \, dy = \int p(x|y) \cdot p(y) \, dy$$

This is the same idea as the law of total probability, just written in the language of joint distributions. You'll see marginalization everywhere -- from computing the evidence term in Bayes' theorem to integrating out latent variables in variational autoencoders:

$$p(x) = \int p(x|z) \, p(z) \, dz$$

---

## Running Example: Building a Naive Bayes Spam Classifier

Let's bring everything together. The Naive Bayes spam classifier is Bayes' theorem in production:

$$P(\text{spam} | \text{words}) \propto P(\text{words} | \text{spam}) \times P(\text{spam})$$

**That's literally Bayes' theorem.** The posterior probability that an email is spam, given its words, is proportional to the likelihood of those words in spam emails times the prior probability of spam.

The "naive" part is the conditional independence assumption: given the class (spam or not), each word is independent:

$$P(w_1, w_2, \ldots, w_n | \text{spam}) = \prod_{i=1}^{n} P(w_i | \text{spam})$$

Let's build it.

```python
import numpy as np

# =============================================================================
# The Medical Test Puzzle -- Solved with Bayes' Theorem
# =============================================================================

def medical_diagnosis():
    """
    Solving the chapter's opening puzzle step by step.
    Disease prevalence: 1%, Sensitivity: 99%, Specificity: 95%
    """
    print("THE MEDICAL TEST PUZZLE")
    print("=" * 55)

    # --- Step 1: Define the priors ---
    P_disease    = 0.01   # 1% of the population has the disease
    P_no_disease = 0.99

    # --- Step 2: Define the likelihoods (test characteristics) ---
    P_pos_given_disease    = 0.99   # sensitivity (true positive rate)
    P_neg_given_no_disease = 0.95   # specificity (true negative rate)
    P_pos_given_no_disease = 0.05   # false positive rate = 1 - specificity

    # --- Step 3: Law of total probability for P(positive) ---
    P_positive = (P_pos_given_disease * P_disease
                + P_pos_given_no_disease * P_no_disease)

    # --- Step 4: Bayes' theorem ---
    P_disease_given_pos = (P_pos_given_disease * P_disease) / P_positive

    print(f"Prior P(disease)               = {P_disease:.2%}")
    print(f"Likelihood P(+|disease)        = {P_pos_given_disease:.2%}")
    print(f"False positive P(+|no disease) = {P_pos_given_no_disease:.2%}")
    print(f"Evidence P(+)                  = {P_positive:.4f}")
    print(f"\n>>> Posterior P(disease|+)      = {P_disease_given_pos:.2%}")
    print("\n99% accurate test, positive result... only ~16.7% chance of disease.")
    print("The base rate (1% prevalence) dominates.\n")

    # --- Sequential update: second positive test ---
    print("SEQUENTIAL UPDATE: Second positive test")
    print("-" * 55)
    prior_2 = P_disease_given_pos  # yesterday's posterior is today's prior
    P_positive_2 = (P_pos_given_disease * prior_2
                  + P_pos_given_no_disease * (1 - prior_2))
    posterior_2 = (P_pos_given_disease * prior_2) / P_positive_2
    print(f"New prior (from first test)    = {prior_2:.4f}")
    print(f">>> After 2nd positive test    = {posterior_2:.2%}")
    print("Each piece of evidence ratchets the posterior up.\n")

medical_diagnosis()


# =============================================================================
# Naive Bayes Spam Classifier
# =============================================================================

def naive_bayes_spam_classifier():
    """
    A from-scratch Naive Bayes spam classifier.

    Core equation:
      P(spam|words) proportional to P(words|spam) * P(spam)

    We use log-probabilities for numerical stability --
    multiplying many small probabilities would underflow to 0.
    """
    print("NAIVE BAYES SPAM CLASSIFIER")
    print("=" * 55)

    # Learned word likelihoods (from training data)
    # P(word | spam) and P(word | ham)
    word_given_spam = {
        'free': 0.8, 'winner': 0.7, 'click': 0.6,
        'meeting': 0.1, 'project': 0.05
    }
    word_given_ham = {
        'free': 0.1, 'winner': 0.02, 'click': 0.1,
        'meeting': 0.5, 'project': 0.6
    }

    # Prior: 30% of training emails were spam
    P_spam = 0.3
    P_ham  = 0.7

    def classify(words):
        """
        Classify an email using Naive Bayes.
        Returns (P(spam|words), P(ham|words)).
        """
        # Start with log-priors
        log_spam = np.log(P_spam)
        log_ham  = np.log(P_ham)

        # Multiply in likelihoods (add in log-space)
        # This is the "naive" conditional independence assumption:
        #   P(w1, w2, ..., wn | spam) = prod P(wi | spam)
        for word in words:
            if word in word_given_spam:
                log_spam += np.log(word_given_spam[word])
                log_ham  += np.log(word_given_ham[word])

        # Normalize to get proper probabilities (softmax of log-probs)
        max_log = max(log_spam, log_ham)  # numerical stability trick
        p_spam = np.exp(log_spam - max_log)
        p_ham  = np.exp(log_ham - max_log)
        total  = p_spam + p_ham

        return p_spam / total, p_ham / total

    # Test on three emails
    test_emails = [
        (['free', 'winner', 'click'],    "Suspicious email"),
        (['meeting', 'project'],          "Work email"),
        (['free', 'meeting'],             "Ambiguous email"),
    ]

    for words, description in test_emails:
        p_spam_post, p_ham_post = classify(words)
        label = "SPAM" if p_spam_post > 0.5 else "HAM"
        print(f"\n{description}: {words}")
        print(f"  P(spam|words) = {p_spam_post:.4f}")
        print(f"  P(ham|words)  = {p_ham_post:.4f}")
        print(f"  Verdict: {label}")

naive_bayes_spam_classifier()


# =============================================================================
# Independence Testing -- Empirical Verification
# =============================================================================

def independence_demo():
    """
    Empirically verify independence vs. dependence.
    Independent: P(A AND B) = P(A) * P(B)
    Dependent:   P(A AND B) != P(A) * P(B)
    """
    print("\n\nINDEPENDENCE TESTING")
    print("=" * 55)

    np.random.seed(42)
    n = 100_000

    # Two independent events
    A = np.random.random(n) < 0.3   # P(A) = 0.3
    B = np.random.random(n) < 0.5   # P(B) = 0.5

    P_A = np.mean(A)
    P_B = np.mean(B)
    P_A_and_B = np.mean(A & B)

    print("\nIndependent events A, B:")
    print(f"  P(A)          = {P_A:.4f}")
    print(f"  P(B)          = {P_B:.4f}")
    print(f"  P(A AND B)    = {P_A_and_B:.4f}")
    print(f"  P(A) * P(B)   = {P_A * P_B:.4f}")
    print(f"  Independent?  {np.isclose(P_A_and_B, P_A * P_B, atol=0.01)}")

    # A dependent event: C depends on A
    C = np.where(A,
                 np.random.random(n) < 0.8,   # P(C|A) = 0.8
                 np.random.random(n) < 0.2)   # P(C|not A) = 0.2

    P_C = np.mean(C)
    P_A_and_C = np.mean(A & C)

    print("\nDependent events A, C  (C depends on A):")
    print(f"  P(A)          = {P_A:.4f}")
    print(f"  P(C)          = {P_C:.4f}")
    print(f"  P(A AND C)    = {P_A_and_C:.4f}")
    print(f"  P(A) * P(C)   = {P_A * P_C:.4f}")
    print(f"  Independent?  {np.isclose(P_A_and_C, P_A * P_C, atol=0.01)}")

independence_demo()


# =============================================================================
# Marginalization -- Summing Out Variables
# =============================================================================

def marginalization_demo():
    """
    Given a joint distribution P(Weather, Mood), demonstrate
    marginalizing out one variable to recover the other.
    """
    print("\n\nMARGINALIZATION")
    print("=" * 55)

    # Joint distribution P(Weather, Mood)
    joint = np.array([
        [0.25, 0.10, 0.05],   # Sunny:  Happy, Neutral, Sad
        [0.05, 0.10, 0.15],   # Rainy:  Happy, Neutral, Sad
        [0.10, 0.12, 0.08],   # Cloudy: Happy, Neutral, Sad
    ])

    weathers = ['Sunny', 'Rainy', 'Cloudy']
    moods    = ['Happy', 'Neutral', 'Sad']

    print("\nJoint P(Weather, Mood):")
    print(f"{'':>8}  {'Happy':>6}  {'Neutral':>7}  {'Sad':>5}")
    for i, w in enumerate(weathers):
        print(f"{w:>8}  {joint[i,0]:>6.2f}  {joint[i,1]:>7.2f}  {joint[i,2]:>5.2f}")

    # Marginalize over Mood -> P(Weather)
    P_weather = joint.sum(axis=1)
    print("\nMarginal P(Weather) -- sum each row:")
    for i, w in enumerate(weathers):
        print(f"  P({w}) = {P_weather[i]:.2f}")

    # Marginalize over Weather -> P(Mood)
    P_mood = joint.sum(axis=0)
    print("\nMarginal P(Mood) -- sum each column:")
    for i, m in enumerate(moods):
        print(f"  P({m}) = {P_mood[i]:.2f}")

    # Conditional: P(Mood | Sunny)
    P_mood_given_sunny = joint[0, :] / P_weather[0]
    print("\nConditional P(Mood | Sunny) -- filter to Sunny, then normalize:")
    for i, m in enumerate(moods):
        print(f"  P({m}|Sunny) = {P_mood_given_sunny[i]:.2f}")

marginalization_demo()
```

---

## Common Mistakes

### Confusing P(A|B) with P(B|A) -- The Prosecutor's Fallacy

This is the single most important pitfall in conditional probability.

- P(positive test | disease) = 0.99 -- the probability of testing positive *if you have the disease*
- P(disease | positive test) = 0.167 -- the probability of having the disease *if you test positive*

These are **not the same thing**. Swapping the direction of conditioning is called the **prosecutor's fallacy** because it infamously appears in courtroom arguments: "The probability of this DNA evidence given innocence is 1 in a million" gets confused with "The probability of innocence given this DNA evidence is 1 in a million." Those are very different statements.

Bayes' theorem is the *only* correct way to flip a conditional probability. You cannot just swap the arguments.

### Base Rate Neglect

Ignoring the prior. We just saw this: a 99%-accurate test gives only ~17% posterior when the disease prevalence is 1%. Always ask: "How common is this in the population?"

### Assuming Independence When Features Are Correlated

Naive Bayes assumes conditional independence of features. In a spam classifier, "Nigerian" and "prince" are clearly not independent given spam. But Naive Bayes treats them as if they are. This works better than you'd expect (the classification boundary is often robust), but it means the *probability values* it outputs are often poorly calibrated. Don't trust the magnitude -- trust the ranking.

### Forgetting to Normalize

When you compute the numerator of Bayes' theorem for multiple hypotheses, the results won't sum to 1 unless you divide by the evidence. The evidence P(B) is just the sum of numerator terms across all hypotheses. Skip it, and your "probabilities" are just unnormalized scores.

---

## Where This Appears in ML

### Naive Bayes Classifier

$$P(y|x_1, \ldots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i|y)$$

Our running example. Assumes features are conditionally independent given the class. Used in text classification, spam detection, and as a fast baseline for any classification task.

### Autoregressive Language Models (GPT, etc.)

$$P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i | w_1, \ldots, w_{i-1})$$

The chain rule of probability, implemented by a transformer. Every token is predicted conditioned on all previous tokens.

### Hidden Markov Models

$$P(x_t | x_1, \ldots, x_{t-1}) = P(x_t | x_{t-1})$$

The Markov property: the future depends only on the present, not the past. This conditional independence assumption makes inference tractable.

### Variational Autoencoders (VAEs)

$$p(x) = \int p(x|z) \, p(z) \, dz$$

Marginalization over latent variables z. The model generates data x by first sampling a latent code z from a prior p(z), then generating x conditioned on z.

### Bayesian Neural Networks

$$P(\theta | \mathcal{D}) = \frac{P(\mathcal{D} | \theta) \, P(\theta)}{P(\mathcal{D})}$$

Bayes' theorem applied to network weights. The posterior over weights theta, given training data D, combines the likelihood of the data under those weights with a prior over weights. The evidence term is typically intractable, which is why variational inference or MCMC is used.

### Bayesian Optimization

Uses Bayes' theorem to maintain a posterior distribution over an objective function (typically modeled as a Gaussian process). Each evaluation updates the posterior, guiding the search toward promising regions of hyperparameter space.

---

## Exercises

### Exercise 1: Bayes' Theorem -- Factory Defect

A factory has two machines. Machine A produces 60% of items, Machine B produces 40%. Machine A has a 2% defect rate, Machine B has a 5% defect rate.

An item is found defective. What's the probability it came from Machine A?

**Solution:**

```python
# Priors
P_A = 0.6
P_B = 0.4

# Likelihoods
P_defect_given_A = 0.02
P_defect_given_B = 0.05

# Evidence: law of total probability
P_defect = P_defect_given_A * P_A + P_defect_given_B * P_B
print(f"P(defect) = {P_defect:.4f}")   # 0.032

# Posterior: Bayes' theorem
P_A_given_defect = (P_defect_given_A * P_A) / P_defect
print(f"P(Machine A | defect) = {P_A_given_defect:.4f}")   # 0.375

# Machine A produces more items overall, but defective items
# are more likely to come from Machine B (higher defect rate).
# The prior (60/40 split) isn't enough to overcome the
# likelihood ratio (2% vs 5%).
```

### Exercise 2: Testing Independence

Given P(A) = 0.4, P(B) = 0.5, P(A AND B) = 0.2. Are A and B independent?

**Solution:**

```python
P_A = 0.4
P_B = 0.5
P_A_and_B = 0.2

# Test: does P(A AND B) = P(A) * P(B)?
product = P_A * P_B
print(f"P(A) * P(B) = {product}")     # 0.2
print(f"P(A AND B)  = {P_A_and_B}")   # 0.2
print(f"Independent? {P_A_and_B == product}")   # True

# Also verify: P(A|B) should equal P(A)
P_A_given_B = P_A_and_B / P_B
print(f"P(A|B) = {P_A_given_B}, P(A) = {P_A}")   # Both 0.4
```

### Exercise 3: Chain Rule -- Drawing Cards

Three cards are drawn without replacement from a standard 52-card deck. What's the probability all three are hearts?

**Solution:**

```python
from math import comb

# Chain rule: P(all 3 hearts) =
#   P(1st heart) * P(2nd heart | 1st heart) * P(3rd heart | first two hearts)

P_first  = 13 / 52   # 13 hearts out of 52 cards
P_second = 12 / 51   # 12 hearts left out of 51 cards
P_third  = 11 / 50   # 11 hearts left out of 50 cards

P_all_hearts = P_first * P_second * P_third
print(f"P(all 3 hearts) = {P_all_hearts:.6f}")   # 0.012941

# Verify with combinatorics: C(13,3) / C(52,3)
P_verify = comb(13, 3) / comb(52, 3)
print(f"Combinatorial verification: {P_verify:.6f}")   # Same answer
```

---

## Summary

| Concept | Formula | One-Line Intuition |
|---------|---------|-------------------|
| Conditional Probability | $P(A\|B) = \frac{P(A \cap B)}{P(B)}$ | Filter to B, measure A within it |
| Chain Rule | $P(A \cap B) = P(A\|B) \cdot P(B)$ | Joint = conditional times marginal |
| Bayes' Theorem | $P(A\|B) = \frac{P(B\|A) \, P(A)}{P(B)}$ | Flip the conditional: posterior = likelihood times prior / evidence |
| Law of Total Probability | $P(A) = \sum_i P(A\|B_i) P(B_i)$ | Break into cases, add them up |
| Independence | $P(A \cap B) = P(A) \cdot P(B)$ | Knowing one tells you nothing about the other |
| Conditional Independence | $P(A \cap B\|C) = P(A\|C) \cdot P(B\|C)$ | Independent *once you know C* |
| Marginalization | $P(A) = \sum_b P(A, B=b)$ | Sum out what you don't care about |

The key takeaway: **Bayes' theorem is a belief-updating machine.** You start with a prior, observe evidence, and compute a posterior. That posterior becomes your new prior when the next piece of evidence arrives. This sequential update pattern is the foundation of Bayesian inference, and it shows up everywhere from spam filters to neural network weight posteriors.

The Naive Bayes spam classifier captures the entire idea in one line:

$$P(\text{spam} | \text{words}) \propto P(\text{words} | \text{spam}) \times P(\text{spam})$$

---

> **What's Next** -- We've been talking about events: "the test is positive," "the email is spam," "all three cards are hearts." But what about numerical outcomes -- like "the model predicts 0.73" or "the user's session lasted 4.2 minutes"? Random variables formalize the connection between probability and numbers, giving us the machinery for expectations, variances, and distributions.

**Next**: [Chapter 3: Random Variables](03-random-variables.md)
