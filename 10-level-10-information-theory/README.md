# Level 10: Information Theory

You've been using `nn.CrossEntropyLoss()` for years. But do you know what "cross-entropy" actually measures? Information theory answers that question — and explains why it's the right loss function for classification.

Here's the truth: you already use information theory every day. When you train a classifier, minimize log-loss, or monitor KL divergence in your VAE, you're applying concepts Claude Shannon developed in 1948 to solve communication problems. Those same ideas now power modern machine learning. Let's understand what's happening under the hood.

## You Already Know This

If you've looked at metrics dashboards, you've seen these terms:
- **Log-loss** = Cross-entropy (they're literally the same thing)
- **Binary cross-entropy** = BCE loss for binary classification
- **KL divergence** = That regularization term in your VAE loss
- **Entropy** = A measure of model confidence/uncertainty

This level explains *why* these formulas work, *when* they're appropriate, and *what* they're really measuring.

## The Core Insight: Information = Surprise

Here's the foundational idea: **rare events carry more information than common events**.

Think about your application logs. If you see `INFO: Request processed successfully` for the millionth time, you learn nothing. But if you see `ERROR: Database connection timeout`, you snap to attention. Why? Because the error is rare — low probability — so it carries high information content.

This intuition leads to Shannon's formula for self-information:

```
I(x) = -log P(x)
```

Rare events (low P(x)) have high information. Common events have low information. Everything else in this level builds from this single principle.

## The Information Theory Hierarchy

Each concept builds on the previous one:

```
Self-Information I(x) = -log P(x)
    ↓ (take expectation over distribution)
Entropy H(P) = E[-log P(x)]
    ↓ (use different distribution Q for encoding)
Cross-Entropy H(P, Q) = E[-log Q(x)] where x ~ P
    ↓ (subtract optimal entropy)
KL Divergence D_KL(P || Q) = H(P, Q) - H(P)
```

Master this hierarchy and you'll understand why cross-entropy is the natural loss for classification, why KL divergence measures distribution distance, and what your model is optimizing when you call `.backward()`.

## What You'll Learn

### Chapter 1: Information
**File**: `01-information.md`

Self-information and surprise. Why rare events carry more info than common ones. The bit as the fundamental unit. Think of this as quantifying your reaction to log messages — why `ERROR` is more surprising than `INFO`.

**Key concepts**: Self-information formula, bits vs nats, relationship to probability

### Chapter 2: Entropy
**File**: `02-entropy.md`

Shannon entropy as *average surprise*. If self-information measures the surprise of one event, entropy measures the expected surprise across your entire distribution. This is like computing the average information content of your log messages.

Also covers the maximum entropy principle (when in doubt, assume maximum uncertainty) and surprising connections to thermodynamics.

**Key concepts**: Entropy formula, uniform distributions maximize entropy, entropy as uncertainty

### Chapter 3: Cross-Entropy
**File**: `03-cross-entropy.md`

The big one. Cross-entropy measures how well distribution Q (your model) represents distribution P (reality). When you minimize cross-entropy loss, you're teaching your model to assign high probability to the correct classes — because predicting rare events costs you more loss.

This chapter shows why cross-entropy is the natural classification loss and connects it to maximum likelihood estimation. Spoiler: minimizing cross-entropy = maximizing likelihood.

**Key concepts**: Cross-entropy formula, why it's used for classification, connection to MLE, categorical vs binary cross-entropy

### Chapter 4: KL Divergence
**File**: `04-kl-divergence.md`

KL divergence measures how much information is lost when you approximate distribution P with distribution Q. Unlike cross-entropy, it's normalized (subtracts out the entropy of P). But here's the catch: it's *asymmetric*. D_KL(P || Q) ≠ D_KL(Q || P).

You see this everywhere: VAE loss functions (the KL term), knowledge distillation (matching student to teacher), model drift monitoring (comparing training vs production distributions).

**Key concepts**: KL formula, asymmetry and its implications, forward vs reverse KL, practical applications

## Navigation: Where Each Concept Appears in ML

| Concept | Where You've Seen It | This Level |
|---------|---------------------|------------|
| Cross-entropy loss | PyTorch `nn.CrossEntropyLoss()`, Keras `categorical_crossentropy` | Chapter 3 |
| Log-loss | Scikit-learn metrics, Kaggle leaderboards | Chapter 3 (same as cross-entropy) |
| Binary cross-entropy | `nn.BCELoss()`, logistic regression | Chapter 3 |
| KL divergence | VAE loss function, ELBO objective | Chapter 4 |
| Entropy | Decision tree splitting criteria, model confidence | Chapter 2 |
| Mutual information | Feature selection, ICA | Built from entropy (Ch 2) |
| Perplexity | Language model evaluation (2^entropy) | Chapter 2 |

## Building On Previous Levels

You need these foundations:

**Level 7 (Probability)**: Information theory quantifies properties of probability distributions. You can't understand entropy without understanding expectation. You can't understand cross-entropy without understanding how to sample from P and evaluate under Q.

**Level 9 (Optimization)**: Cross-entropy is the loss function you optimize with gradient descent. When you call `.backward()` on your classification loss, you're computing the gradient of cross-entropy with respect to your model parameters.

If these feel shaky, review them first. Information theory is precise mathematics, not hand-waving.

## What Comes Next

**Level 13 (ML Models Math)**: You'll use cross-entropy loss directly in logistic regression (Chapter 2) and neural networks (Chapter 3). The formulas here become the objective functions there.

**Level 14 (Advanced Topics)**: Information geometry treats probability distributions as points in a geometric space, with KL divergence defining the distance metric. Natural gradient descent uses this geometry for better optimization.

## Key Notation

| Symbol | Meaning |
|--------|---------|
| $P(x)$ | True/data distribution (reality) |
| $Q(x)$ | Model/approximate distribution (your neural network's output) |
| $I(x)$ | Self-information of event x |
| $H(P)$ | Entropy of distribution P |
| $H(P, Q)$ | Cross-entropy between P and Q |
| $D_{KL}(P \|\| Q)$ | KL divergence from Q to P (note the direction!) |
| $\log$ | Natural logarithm (nats) unless specified |
| $\log_2$ | Logarithm base 2 (bits) |

Note: In ML we typically use natural log (nats) because it's cleaner for calculus. But Shannon's original work used base-2 (bits) for communication theory. The concepts are identical, just scaled by ln(2) ≈ 0.693.

## How to Approach This Level

1. **Read in order**. Each chapter builds on the previous. Don't skip to cross-entropy.
2. **Run the code**. The formulas are simple but the intuition comes from seeing concrete examples.
3. **Connect to your work**. Every time you train a classifier or tune a VAE, you're using these concepts. Make the connections explicit.
4. **Remember the hierarchy**. Self-information → entropy → cross-entropy → KL divergence. Each is the expectation or refinement of the previous.

By the end of this level, you'll understand what happens when you write:

```python
loss = nn.CrossEntropyLoss()(predictions, labels)
loss.backward()
```

You're minimizing the expected surprise your model assigns to the true labels, weighted by the information content of each class. That's information theory in action.

---

*"The fundamental problem of communication is that of reproducing at one point either exactly or approximately a message selected at another point."* — Claude Shannon, 1948

(In ML, we're trying to reproduce reality's distribution in our model's parameters. Same problem, different domain.)
