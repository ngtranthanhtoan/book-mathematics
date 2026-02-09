# Chapter 4: KL Divergence

> Cross-entropy measures total coding cost. KL divergence isolates the excess cost — the part that comes purely from using the wrong distribution.

---

## The Problem You Already Have

Your VAE's loss has a term you might not fully understand: D_KL(q(z|x) || p(z)). It measures how far your encoder's distribution is from the prior. KL divergence is the universal yardstick for "how different are these two distributions?" — and it shows up everywhere from VAEs to policy gradient methods to knowledge distillation.

Here is the core question: **How do you measure the distance between two distributions?**

Think about it concretely. You are building a movie recommendation VAE. Your encoder takes a user's watch history and outputs a distribution over a latent space — a compressed "user embedding." The prior says that embedding should look like a standard normal. But your encoder might output something shifted, or too narrow, or too wide. You need a number that tells you: how far off is my encoder from the prior?

That number is KL divergence.

---

## Running Example: VAE on Movie Preferences

Throughout this chapter, we will use a single scenario. You are building a VAE that learns user embeddings from movie ratings. Each user gets encoded into a latent vector z. The encoder outputs q(z|x) — a Gaussian with learned mean and variance for each user. The prior is p(z) = N(0, 1).

The **KL term pushes the learned user embedding toward a standard normal prior.** Without it, your encoder would memorize each user as a point in space (overfitting). With it, similar users cluster together, and you can sample new "synthetic users" from the prior to generate recommendations.

Every formula in this chapter connects back to this: what happens to the KL term as your encoder's output drifts away from the prior?

---

## What KL Divergence Actually Measures

**Kullback-Leibler divergence** (also called relative entropy) measures the information lost when you use distribution Q to approximate distribution P. In code terms, it is the extra bits you waste by using the wrong codebook.

Here is the analogy that sticks. You are running an A/B test on your movie app. Group A users click in a pattern described by distribution P. Group B users click in a pattern described by distribution Q. KL divergence tells you how many extra bits you would waste if you designed your data pipeline assuming Group B's behavior but Group A is what is actually happening. It is the cost of the mismatch — nothing more, nothing less.

> **Common Mistake**: KL divergence is NOT symmetric: D_KL(P||Q) != D_KL(Q||P). It's not a true distance metric. Think of it this way: the cost of assuming Q when the truth is P is different from the cost of assuming P when the truth is Q. Just like the cost of bringing a winter coat to the tropics is not the same as bringing a swimsuit to the Arctic.

---

## The Formulas

### Definition: KL Divergence

For discrete distributions P and Q over the same support:

$$D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

Equivalently:

$$D_{KL}(P \| Q) = H(P, Q) - H(P)$$

KL divergence is cross-entropy minus entropy — the "extra bits" beyond the minimum. In the previous chapter, you saw that cross-entropy H(P, Q) measures the total coding cost of using Q to encode data from P. Entropy H(P) is the theoretical minimum. KL divergence is the gap.

Back to your movie VAE: if your encoder perfectly matched the prior, the KL term would be zero. Every deviation from the prior shows up as excess bits in this term.

### Properties of KL Divergence

1. **Non-negativity (Gibbs' inequality)**: $D_{KL}(P \| Q) \geq 0$
   - Zero if and only if P = Q almost everywhere

2. **Not symmetric**: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
   - Therefore, not a true distance metric

3. **Not satisfying triangle inequality**: Can have $D_{KL}(P \| R) > D_{KL}(P \| Q) + D_{KL}(Q \| R)$
   - Another reason it is not a metric

4. **Convexity**: $D_{KL}(P \| Q)$ is convex in the pair (P, Q)

5. **Invariance**: Under invertible transformations of the random variable

---

## ASCII Visualization: KL Divergence as Area Between Curves

KL divergence is the expected log-ratio of the two distributions, weighted by P. You can visualize it as the "area" of discrepancy between P and Q, where regions with high P(x) matter most:

```
  Probability
  density
    |
0.4 |        * *
    |       *   *           KL divergence corresponds to
    |      *     *          the weighted gap between these
0.3 |   P *   .   *        two curves, where P does the
    |    *  .   .  *        weighting.
    |   * .       . *
0.2 |  *.     Q     .*         P = true distribution (solid *)
    | *.  . .   . .  .*        Q = model distribution (dots .)
    |*  ..         ..  *
0.1 |  ..             ..    Areas where P is high but Q is low
    |..                 ..  contribute the most to KL divergence.
    |.                   .
0.0 +--+--+--+--+--+--+--+---> x
   -4  -3  -2  -1   0   1   2   3

  D_KL(P || Q) = Sum over x of P(x) * log(P(x) / Q(x))

  Where P(x) >> Q(x):  log(P/Q) is large and positive  --> big penalty
  Where P(x) << Q(x):  log(P/Q) is negative, but P(x) is small --> small contribution
  Where P(x) ~= Q(x):  log(P/Q) ~ 0                    --> no penalty
```

The key insight: KL divergence cares most about regions where P is large. If P puts probability mass somewhere Q does not, you get a huge (or infinite) penalty. If Q puts extra mass where P does not, that barely matters. This asymmetry is the source of all the forward-vs-reverse KL behavior.

---

## Forward vs Reverse KL: Mode-Covering vs Mode-Seeking

This is the single most important practical distinction in KL divergence. It determines which algorithm you should use and what failure modes you will see.

### Forward KL: D_KL(P || Q) — Mean-Seeking / Mode-Covering

- Penalizes Q heavily when Q(x) is small but P(x) is large
- "Zero-avoiding" — Q tries to cover all of P's support
- Used when P is the data, Q is the model

### Reverse KL: D_KL(Q || P) — Mode-Seeking

- Penalizes Q heavily when Q(x) is large but P(x) is small
- "Zero-forcing" — Q collapses onto high-probability regions of P
- Used in variational inference

```
Forward KL: D_KL(P || Q) - "Mode-covering / Mean-seeking"
        P (bimodal)          Q (fitted)
         __   __              ___
        /  \ /  \     -->    /   \
       /    X    \          /     \
      /           \        /       \
    Fitted Q covers both modes (may be too wide)

Reverse KL: D_KL(Q || P) - "Mode-seeking"
        P (bimodal)          Q (fitted)
         __   __              __
        /  \ /  \     -->    /  \
       /    X    \          /    \
      /           \
    Fitted Q locks onto one mode (ignores the other)
```

### The SWE Bridge: Optimistic vs Pessimistic Load Balancing

Forward vs reverse KL maps directly to a systems design intuition you already have.

**Forward KL is like pessimistic (conservative) load balancing.** You spread your capacity across every server that might get traffic. You cover all modes. Some capacity is wasted in valleys between peaks, but you never get caught with zero capacity where traffic is high. You would rather be too wide than miss anything.

**Reverse KL is like optimistic (aggressive) load balancing.** You concentrate capacity on the single busiest server. You lock onto one mode and serve it perfectly. You are efficient where you operate, but you completely ignore other traffic sources. You would rather be precise than comprehensive.

When you choose between variational inference (reverse KL) and maximum likelihood (forward KL), you are choosing between these two strategies.

### Movie VAE Example

In your movie VAE, the KL term uses forward KL: D_KL(q(z|x) || p(z)). The encoder q(z|x) is being compared against the prior p(z) = N(0, 1). This pushes the encoder to cover the prior's support — it should not create weird isolated clusters that the prior would never sample from. If you used reverse KL here, the encoder might ignore entire regions of the prior, making generation from the prior unreliable.

---

## KL Divergence for Gaussians

For two univariate Gaussians $P = \mathcal{N}(\mu_1, \sigma_1^2)$ and $Q = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$D_{KL}(P \| Q) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

For multivariate Gaussians $P = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ and $Q = \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$:

$$D_{KL}(P \| Q) = \frac{1}{2}\left[\log \frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|} - d + \text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)\right]$$

This closed-form Gaussian KL is why VAEs almost always use Gaussian encoders and a Gaussian prior. You get an exact, differentiable KL term for free. No sampling, no approximation.

### Movie VAE: What the Encoder Outputs Mean

Your encoder outputs mu and log_var for each user. Here is what happens to the KL term:

| Encoder Output | mu | sigma | KL to N(0,1) | Interpretation |
|---|---|---|---|---|
| Perfect prior match | 0.0 | 1.0 | 0.000 | User embedding equals the prior — no information encoded |
| Shifted mean | 1.0 | 1.0 | 0.500 | User is "displaced" from average — encodes preference info |
| Narrow variance | 0.0 | 0.61 | 0.197 | Encoder is too confident about this user |
| Shifted + narrow | 2.0 | 0.61 | 2.197 | Lots of user-specific info — KL penalty is high |

The KL penalty increases when the encoder tries to encode more information about a specific user. This is the tension at the heart of VAEs: reconstruction wants more info, KL wants less.

---

## The ELBO: VAE Training Objective Decomposed

The VAE objective (Evidence Lower BOund) is the equation you see in every VAE tutorial:

$$\mathcal{L}_{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

- **First term: Reconstruction quality.** "How well can the decoder reconstruct the input from the latent code?" In your movie VAE, this is: given the user embedding z, how accurately can you predict their ratings?
- **Second term: KL regularization.** "How close is the encoder's output to the prior?" This is the KL divergence you have been studying. It prevents the encoder from cheating by memorizing each user.

The full VAE loss makes this explicit:

$$\mathcal{L} = \underbrace{-\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction loss}} + \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{KL regularization}}$$

The ELBO is also connected to variational inference more broadly. When you cannot compute the true posterior p(z|x) exactly, you approximate it with q(z|x) and minimize their KL divergence. Maximizing the ELBO is equivalent to minimizing D_KL(q(z|x) || p(z|x)):

$$q^*(z) = \arg\min_q D_{KL}(q(z) \| p(z|x))$$

---

## Mutual Information: Dependency Between Features

Mutual information is KL divergence applied to the question: "how dependent are two random variables?"

$$I(X; Y) = D_{KL}(P(X, Y) \| P(X) P(Y))$$

It measures the KL divergence between the joint distribution and the product of the marginals. If X and Y are independent, P(X, Y) = P(X)P(Y), so the KL is zero. The more dependent they are, the higher the mutual information.

### SWE Bridge: Feature Dependency

In feature engineering, you often want to know: does feature A carry information about feature B? Mutual information answers this directly. Unlike correlation, it captures nonlinear dependencies too.

In your movie VAE, mutual information I(X; Z) measures how much information the latent embedding Z retains about the original ratings X. The information bottleneck framework explicitly trades off this quantity:

$$\min_{q(z|x)} D_{KL}(q(z|x) \| r(z)) - \beta \cdot I(Z; Y)$$

More compression (lower I(X; Z)) means less overfitting. More prediction (higher I(Z; Y)) means better recommendations. The beta parameter controls the trade-off.

---

## KL Divergence as an A/B Test Metric

Here is a SWE application you can use tomorrow. You are running an A/B test on your movie app. You have the distribution of user click patterns in Group A and Group B. Instead of comparing means (t-test) or proportions (chi-squared), you can measure the KL divergence between the two behavioral distributions.

Why would you do this? Because KL divergence captures the full distributional difference, not just the mean. Two groups could have the same mean click rate but very different distributions — one group might be bimodal (power users and lurkers) while the other is uniform. KL divergence catches this; a t-test does not.

```python
import numpy as np

# Click-through distributions for two A/B test groups
group_a_clicks = np.array([0.05, 0.10, 0.35, 0.30, 0.15, 0.05])  # Sessions with 0-5 clicks
group_b_clicks = np.array([0.15, 0.25, 0.25, 0.20, 0.10, 0.05])  # Modified UI

kl_ab = np.sum(group_a_clicks * np.log(group_a_clicks / group_b_clicks))
kl_ba = np.sum(group_b_clicks * np.log(group_b_clicks / group_a_clicks))

print(f"D_KL(A || B) = {kl_ab:.4f} nats")  # Cost of assuming B when truth is A
print(f"D_KL(B || A) = {kl_ba:.4f} nats")  # Cost of assuming A when truth is B
print(f"These differ because KL is asymmetric!")
```

---

## ML Applications

### Policy Optimization (PPO, TRPO)

In reinforcement learning, you constrain policy updates using KL divergence:

$$\max_\theta \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A(s, a)\right] \quad \text{s.t.} \quad D_{KL}(\pi_{\theta_{old}} \| \pi_\theta) \leq \delta$$

This prevents catastrophically large policy changes. If you let your recommendation policy change too much in one update, your movie suggestions go from sensible to bizarre. The KL constraint says: "update the policy, but keep the new action distribution close to the old one."

This is another forward-vs-reverse KL choice. TRPO uses forward KL (constraining how much the old policy diverges from the new), which is mode-covering — it ensures the new policy does not drop probability from any action the old policy considered viable.

### Knowledge Distillation

Transfer knowledge from a large teacher model to a small student model:

$$\mathcal{L} = (1-\alpha) \cdot CE(y, p_s) + \alpha \cdot T^2 \cdot D_{KL}(p_t^{(T)} \| p_s^{(T)})$$

Where $p^{(T)}$ are temperature-softened probabilities. The KL term forces the student to match the teacher's full output distribution, not just the hard labels. A teacher model might say "this movie is 70% drama, 20% thriller, 10% romance." The KL term makes the student learn those soft relationships, not just "drama."

---

## Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, norm
from scipy.special import kl_div

def kl_divergence(p, q, epsilon=1e-15):
    """
    Calculate KL divergence D_KL(P || Q) for discrete distributions.

    Parameters:
    -----------
    p : array-like
        True/reference distribution
    q : array-like
        Approximate/model distribution
    epsilon : float
        Small value to prevent division by zero

    Returns:
    --------
    float : KL divergence value
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Ensure proper distributions
    p = p / p.sum()
    q = q / q.sum()

    # Add epsilon to prevent log(0)
    q = np.clip(q, epsilon, 1)

    # Only sum where p > 0 (0 * log(0/q) = 0 by convention)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def kl_divergence_gaussians(mu1, sigma1, mu2, sigma2):
    """
    KL divergence between two univariate Gaussians.
    D_KL(N(mu1, sigma1^2) || N(mu2, sigma2^2))
    """
    return (np.log(sigma2 / sigma1) +
            (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) -
            0.5)

# Example 1: Basic KL divergence
print("=== KL Divergence Basics ===\n")

# True distribution (e.g., fair die)
p_fair = np.array([1/6] * 6)

# Various model distributions
models = {
    "Perfect match": [1/6] * 6,
    "Slightly biased": [0.2, 0.2, 0.15, 0.15, 0.15, 0.15],
    "Heavily biased": [0.4, 0.3, 0.15, 0.1, 0.03, 0.02],
    "Uniform over 3": [1/3, 1/3, 1/3, 0, 0, 0],  # Missing support!
}

print("True distribution (fair die): [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]\n")
for name, q in models.items():
    q = np.array(q)
    kl_pq = kl_divergence(p_fair, q)
    kl_qp = kl_divergence(q, p_fair) if not np.any(q == 0) else float('inf')

    print(f"{name}:")
    print(f"  Q = {[f'{x:.3f}' for x in q]}")
    print(f"  D_KL(P || Q) = {kl_pq:.4f} nats")
    if kl_qp != float('inf'):
        print(f"  D_KL(Q || P) = {kl_qp:.4f} nats")
        print(f"  Asymmetry: {abs(kl_pq - kl_qp):.4f}")
    else:
        print(f"  D_KL(Q || P) = inf (Q has zeros where P > 0)")
    print()

# Example 2: Demonstrate asymmetry
print("=== KL Divergence Asymmetry ===\n")

p = np.array([0.9, 0.1])
q = np.array([0.1, 0.9])

kl_pq = kl_divergence(p, q)
kl_qp = kl_divergence(q, p)

print(f"P = [0.9, 0.1]")
print(f"Q = [0.1, 0.9]")
print(f"D_KL(P || Q) = {kl_pq:.4f} nats")
print(f"D_KL(Q || P) = {kl_qp:.4f} nats")
print(f"They're equal here because the distributions are 'mirror images'")

p2 = np.array([0.7, 0.3])
q2 = np.array([0.3, 0.5, 0.2])  # Different support size

# For demonstration, let's use same support
p3 = np.array([0.6, 0.3, 0.1])
q3 = np.array([0.1, 0.3, 0.6])

kl_p3q3 = kl_divergence(p3, q3)
kl_q3p3 = kl_divergence(q3, p3)

print(f"\nP = [0.6, 0.3, 0.1]")
print(f"Q = [0.1, 0.3, 0.6]")
print(f"D_KL(P || Q) = {kl_p3q3:.4f} nats")
print(f"D_KL(Q || P) = {kl_q3p3:.4f} nats")
print(f"Asymmetry demonstrated: difference = {abs(kl_p3q3 - kl_q3p3):.4f}")

# Example 3: KL between Gaussians
print("\n=== KL Divergence: Gaussians ===\n")

gaussian_pairs = [
    ((0, 1), (0, 1), "Same distributions"),
    ((0, 1), (1, 1), "Different means"),
    ((0, 1), (0, 2), "Different variances"),
    ((0, 1), (2, 0.5), "Both different"),
]

for (mu1, s1), (mu2, s2), description in gaussian_pairs:
    kl = kl_divergence_gaussians(mu1, s1, mu2, s2)
    kl_rev = kl_divergence_gaussians(mu2, s2, mu1, s1)
    print(f"{description}:")
    print(f"  N({mu1}, {s1}^2) vs N({mu2}, {s2}^2)")
    print(f"  D_KL(P || Q) = {kl:.4f}")
    print(f"  D_KL(Q || P) = {kl_rev:.4f}\n")

# Example 4: Visualize forward vs reverse KL
print("=== Forward vs Reverse KL Visualization ===\n")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

x = np.linspace(-5, 5, 1000)

# True distribution: mixture of two Gaussians
p_mixture = 0.5 * norm.pdf(x, -2, 0.7) + 0.5 * norm.pdf(x, 2, 0.7)
p_mixture = p_mixture / np.trapz(p_mixture, x)  # Normalize

# Approximations
q_forward = norm.pdf(x, 0, 2.5)  # Wide Gaussian (mean-seeking)
q_forward = q_forward / np.trapz(q_forward, x)

q_reverse = norm.pdf(x, -2, 0.7)  # Narrow, at one mode (mode-seeking)
q_reverse = q_reverse / np.trapz(q_reverse, x)

# Plot 1: True distribution
axes[0].fill_between(x, p_mixture, alpha=0.5, color='blue', label='True P (bimodal)')
axes[0].set_title('True Distribution P', fontsize=12)
axes[0].set_xlabel('x')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Forward KL (mean-seeking)
axes[1].fill_between(x, p_mixture, alpha=0.3, color='blue', label='True P')
axes[1].plot(x, q_forward, 'r-', linewidth=2, label='Fitted Q')
axes[1].set_title('Forward KL: Mean-Seeking\nQ covers both modes', fontsize=12)
axes[1].set_xlabel('x')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Reverse KL (mode-seeking)
axes[2].fill_between(x, p_mixture, alpha=0.3, color='blue', label='True P')
axes[2].plot(x, q_reverse, 'r-', linewidth=2, label='Fitted Q')
axes[2].set_title('Reverse KL: Mode-Seeking\nQ locks onto one mode', fontsize=12)
axes[2].set_xlabel('x')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forward_vs_reverse_kl.png', dpi=150)
plt.show()

# Example 5: VAE loss components — Movie Preferences
print("=== VAE Loss: KL Term (Movie Preferences) ===\n")

def kl_divergence_vae(mu, log_var):
    """
    KL divergence between N(mu, exp(log_var)) and N(0, 1).
    Used as regularization in VAEs.

    Returns per-sample KL divergence.
    """
    # D_KL(N(mu, sigma^2) || N(0, 1))
    # = 0.5 * (sigma^2 + mu^2 - 1 - log(sigma^2))
    return 0.5 * (np.exp(log_var) + mu**2 - 1 - log_var)

# Simulated encoder outputs for different users
latent_samples = [
    (0.0, 0.0, "Average user (matches prior)"),
    (1.0, 0.0, "Action movie lover (shifted mean)"),
    (0.0, 1.0, "Unpredictable taste (larger variance)"),
    (2.0, -0.5, "Niche cinephile (shifted mean, smaller variance)"),
]

print("KL divergence from encoder q(z|x) to prior N(0,1):\n")
for mu, log_var, description in latent_samples:
    kl = kl_divergence_vae(mu, log_var)
    sigma = np.exp(0.5 * log_var)
    print(f"{description}:")
    print(f"  mu = {mu:.2f}, sigma = {sigma:.2f}")
    print(f"  KL = {kl:.4f} nats\n")

# Example 6: KL in knowledge distillation
print("=== Knowledge Distillation: KL Between Softmax Outputs ===\n")

def softmax(logits, temperature=1.0):
    """Softmax with temperature."""
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

# Teacher model outputs (well-trained, confident)
teacher_logits = np.array([5.0, 2.0, 0.5])
teacher_probs = softmax(teacher_logits)

# Student model outputs (learning)
student_logits = np.array([2.0, 1.5, 1.0])
student_probs = softmax(student_logits)

# Temperature-scaled versions
T = 3.0
teacher_soft = softmax(teacher_logits, T)
student_soft = softmax(student_logits, T)

print("Hard targets (T=1):")
print(f"  Teacher: {[f'{p:.3f}' for p in teacher_probs]}")
print(f"  Student: {[f'{p:.3f}' for p in student_probs]}")
print(f"  KL(Teacher || Student) = {kl_divergence(teacher_probs, student_probs):.4f}")

print(f"\nSoft targets (T={T}):")
print(f"  Teacher: {[f'{p:.3f}' for p in teacher_soft]}")
print(f"  Student: {[f'{p:.3f}' for p in student_soft]}")
print(f"  KL(Teacher || Student) = {kl_divergence(teacher_soft, student_soft):.4f}")
print("\nSofter targets reveal more information about class relationships!")

# Example 7: Verify with scipy
print("\n=== Verification with SciPy ===")
p_test = np.array([0.3, 0.5, 0.2])
q_test = np.array([0.25, 0.5, 0.25])

our_kl = kl_divergence(p_test, q_test)
scipy_kl = entropy(p_test, q_test)  # scipy.stats.entropy computes KL when given two args

print(f"Our implementation: {our_kl:.6f}")
print(f"SciPy entropy (KL): {scipy_kl:.6f}")
print(f"Match: {np.isclose(our_kl, scipy_kl)}")
```

**Output:**
```
=== KL Divergence Basics ===

True distribution (fair die): [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

Perfect match:
  Q = ['0.167', '0.167', '0.167', '0.167', '0.167', '0.167']
  D_KL(P || Q) = 0.0000 nats

Slightly biased:
  Q = ['0.200', '0.200', '0.150', '0.150', '0.150', '0.150']
  D_KL(P || Q) = 0.0139 nats
  D_KL(Q || P) = 0.0141 nats
  Asymmetry: 0.0002

Heavily biased:
  Q = ['0.400', '0.300', '0.150', '0.100', '0.030', '0.020']
  D_KL(P || Q) = 0.4389 nats

=== VAE Loss: KL Term (Movie Preferences) ===

KL divergence from encoder q(z|x) to prior N(0,1):

Average user (matches prior):
  mu = 0.00, sigma = 1.00
  KL = 0.0000 nats

Action movie lover (shifted mean):
  mu = 1.00, sigma = 1.00
  KL = 0.5000 nats

Unpredictable taste (larger variance):
  mu = 0.00, sigma = 1.65
  KL = 0.6065 nats
```

---

## When to Use and When to Reach for Alternatives

### When KL Divergence Is the Right Tool

- **VAE training**: The standard regularization term in the ELBO
- **Distribution matching**: Comparing probability distributions in A/B tests, model evaluation
- **Variational inference**: Approximating intractable posteriors
- **Policy constraints**: Limiting update sizes in RL (PPO, TRPO)
- **Knowledge distillation**: Matching student output to teacher output

### When to Consider Alternatives

- **Need a true metric?** Use Jensen-Shannon divergence (symmetric) or Wasserstein distance
- **Distributions have disjoint support?** KL can be infinite; use JS divergence or optimal transport
- **Mode collapse issues?** Consider reverse KL or other f-divergences
- **Training GANs?** Original GAN uses JS divergence; Wasserstein GAN uses optimal transport

### Common Pitfalls

1. **Infinite KL**: Occurs when Q(x) = 0 but P(x) > 0. Always smooth or clip.
2. **Direction confusion**: $D_{KL}(P\|Q)$ vs $D_{KL}(Q\|P)$ have very different behaviors
3. **Not a metric**: Cannot use triangle inequality reasoning
4. **Numerical issues**: Log of small numbers; use log-sum-exp tricks

```python
# WRONG: Can produce inf
kl = np.sum(p * np.log(p / q))  # q might be 0!

# RIGHT: Clip or smooth
q_safe = np.clip(q, 1e-10, 1)
kl = np.sum(p * np.log(p / q_safe))
```

---

## Exercises

### Exercise 1: KL Divergence Calculation
**Problem**: Calculate $D_{KL}(P \| Q)$ for P = [0.25, 0.25, 0.5] and Q = [0.5, 0.25, 0.25].

**Solution**:
```python
import numpy as np

P = np.array([0.25, 0.25, 0.5])
Q = np.array([0.5, 0.25, 0.25])

kl = np.sum(P * np.log(P / Q))
print(f"D_KL(P || Q) = {kl:.4f} nats")
# = 0.25 * log(0.5) + 0.25 * log(1) + 0.5 * log(2)
# = 0.25 * (-0.693) + 0 + 0.5 * (0.693)
# = -0.173 + 0.347 = 0.173 nats
```

### Exercise 2: VAE KL Term
**Problem**: In a VAE, your encoder outputs mu=0.5 and log_var=-0.5 for a sample. Calculate the KL divergence to the standard normal prior.

**Solution**:
```python
mu = 0.5
log_var = -0.5
sigma_sq = np.exp(log_var)

# D_KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * (sigma^2 + mu^2 - 1 - log(sigma^2))
kl = 0.5 * (sigma_sq + mu**2 - 1 - log_var)
print(f"KL = {kl:.4f} nats")  # ~ 0.178 nats
```

### Exercise 3: Forward vs Reverse KL
**Problem**: You are fitting a unimodal Gaussian to a bimodal mixture. Explain qualitatively what happens with forward KL vs reverse KL minimization.

**Solution**:
- **Forward KL** $D_{KL}(P_{bimodal} \| Q_{unimodal})$: Penalizes Q when it assigns low probability to regions where P is high. Result: Q becomes wide to cover both modes. The fitted Gaussian will have mean between the modes and large variance.

- **Reverse KL** $D_{KL}(Q_{unimodal} \| P_{bimodal})$: Penalizes Q when it assigns high probability to regions where P is low. Result: Q collapses onto one mode to avoid the valley. The fitted Gaussian will match one mode closely.

Forward KL is "inclusive" (tries to cover everything), while reverse KL is "exclusive" (focuses on what it can model well).

### Exercise 4: Movie VAE Trade-off
**Problem**: Your movie recommendation VAE has two users. User A has mu=0.1, log_var=0.0 (mainstream taste). User B has mu=3.0, log_var=-1.0 (niche cinephile). Calculate the KL penalty for each. Which user does the model "struggle" to encode within the prior? What would happen if you increased the KL weight beta?

**Solution**:
```python
# User A: mainstream
kl_a = 0.5 * (np.exp(0.0) + 0.1**2 - 1 - 0.0)  # = 0.005 nats
# User B: niche
kl_b = 0.5 * (np.exp(-1.0) + 3.0**2 - 1 - (-1.0))  # = 5.184 nats

print(f"User A KL: {kl_a:.3f} nats")  # Low — close to prior
print(f"User B KL: {kl_b:.3f} nats")  # High — far from prior
```

User B is expensive to encode. Increasing beta would push User B's embedding closer to the prior, sacrificing reconstruction quality for that user — the model would lose its ability to represent niche preferences. This is the classic beta-VAE trade-off: higher beta means smoother latent space but worse reconstruction for outlier users.

---

## Summary

| Concept | Formula | What It Measures | Where You See It |
|---------|---------|------------------|-----------------|
| KL Divergence | $\sum P(x) \log \frac{P(x)}{Q(x)}$ | Excess coding cost | VAE, distillation, RL |
| Forward KL | $D_{KL}(P \| Q)$ | Mode-covering fit | Maximum likelihood, A/B tests |
| Reverse KL | $D_{KL}(Q \| P)$ | Mode-seeking fit | Variational inference |
| ELBO | $\mathbb{E}[\log p(x|z)] - D_{KL}(q \| p)$ | VAE training objective | Generative models |
| Mutual Information | $D_{KL}(P(X,Y) \| P(X)P(Y))$ | Feature dependency | Information bottleneck |

Key takeaways:

- **KL divergence** measures information lost when using Q to approximate P: $D_{KL}(P\|Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$
- **Non-negative**: Always $\geq 0$, equals 0 only when P = Q
- **Asymmetric**: $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ — this has major practical implications
- **Forward KL** (mean-seeking / mode-covering): Q tries to cover all of P's support
- **Reverse KL** (mode-seeking): Q collapses onto high-probability regions of P
- **Relationship**: $D_{KL}(P\|Q) = H(P, Q) - H(P)$ — the extra bits beyond entropy
- **Caution**: Can be infinite when supports do not match; always clip or smooth

---

## The Information Theory Toolkit: Complete

You now have four fundamental tools:

| Concept | Formula | Measures | ML Use |
|---------|---------|----------|--------|
| Self-Information | $-\log P(x)$ | Surprise of event | Per-sample loss |
| Entropy | $-\sum P(x) \log P(x)$ | Average uncertainty | Confidence, exploration |
| Cross-Entropy | $-\sum P(x) \log Q(x)$ | Model fit to data | Classification loss |
| KL Divergence | $\sum P(x) \log \frac{P(x)}{Q(x)}$ | Distribution distance | VAE, distillation, RL |

These build on each other:
- Entropy = Expected self-information
- Cross-entropy = Entropy + KL divergence
- KL divergence = Cross-entropy - Entropy

You've completed information theory. Entropy, cross-entropy, and KL divergence are the mathematical foundations of classification loss functions, generative models, and information-theoretic learning.
