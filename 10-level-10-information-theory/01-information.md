# Chapter 1: Information and Surprise

Probability tells you how likely events are. Information theory asks the reverse: how surprised are you when an event occurs? This inversion is the foundation of cross-entropy loss.

## Intuition

How much information does a message carry? If I tell you "the sun rose today," that's zero information -- you already knew. If I tell you "a meteor just hit Earth," that's maximum information -- completely unexpected. Shannon formalized this intuition: information = surprise = -log(probability).

Think about it from your daily experience. You scroll through movie reviews on some aggregator. You see a 3-star review on a movie with a 3.1 average rating. You shrug -- that's exactly what you expected. Zero surprise, zero information. Now imagine the same aggregator shows a glowing 5-star review on a movie sitting at 1.2 stars. You stop scrolling. *That* is information. Something unexpected happened, and your brain knows it.

**The core insight: rare events carry more information than common events.**

This isn't philosophy. It's the mathematical backbone of every loss function you've ever used in ML. When your model predicts a class with 99% confidence and gets it right, the loss is tiny -- low surprise. When it assigns 1% to the true class, the loss explodes -- high surprise. That loss *is* self-information.

### The Running Example: Movie Reviews

We'll use movie reviews throughout this chapter to build intuition:

- **A rare 5-star review on a generally bad movie: high information content.** Say only 2% of reviews for this movie are 5-star. That review carries a lot of surprise -- it tells you something unexpected is going on. Maybe the movie is a guilty pleasure, maybe the reviewer is the director's mom. Either way, you learned something.
- **A 3-star review on an average movie: low information content.** If 40% of reviews cluster around 3 stars, seeing another one barely moves the needle. You already expected this.

The question Shannon answered was: *exactly how much* surprise does each of these carry?

## Formal Definition

### Self-Information

The **self-information** (also called **surprisal** or **information content**) of an event $x$ with probability $P(x)$ is:

$$I(x) = -\log P(x) = \log \frac{1}{P(x)}$$

That's it. One formula. The negative log of the probability.

### Why the Logarithm?

You might wonder: why not just use $1/P(x)$ as the surprise? A fair coin flip has $P = 0.5$, so $1/P = 2$. A die roll of 6 has $P = 1/6$, so $1/P = 6$. Seems reasonable. But here's the problem: surprise should be *additive* for independent events.

If you flip a coin and roll a die independently, the total surprise should be the sum of the individual surprises. With $1/P$, you'd get $2 \times 6 = 12$ (multiplicative, not additive). The logarithm is the unique function that converts multiplication into addition:

$$-\log P(x,y) = -\log[P(x) \cdot P(y)] = -\log P(x) + (-\log P(y)) = I(x) + I(y)$$

So the logarithm isn't an arbitrary choice. It's the *only* function that satisfies three requirements simultaneously:

- Information is additive for independent events
- Rarer events carry more information
- Certain events carry zero information

### Properties of Self-Information

These follow directly from the definition:

1. **Non-negativity**: $I(x) \geq 0$ for all events
   - Since $0 \leq P(x) \leq 1$, we have $-\log P(x) \geq 0$

2. **Certainty yields zero information**: If $P(x) = 1$, then $I(x) = 0$
   - A certain event tells you nothing new

3. **Impossible events yield infinite information**: As $P(x) \to 0$, $I(x) \to \infty$
   - Observing the impossible would be infinitely surprising

4. **Additivity for independent events**: $I(x, y) = I(x) + I(y)$ when $x$ and $y$ are independent
   - Since $P(x,y) = P(x)P(y)$, we get $-\log P(x)P(y) = -\log P(x) - \log P(y)$

### Example Calculations

**Fair coin flip (heads):**
$$I(\text{heads}) = -\log_2(0.5) = 1 \text{ bit}$$

**Rolling a 6 on a fair die:**
$$I(\text{six}) = -\log_2(1/6) \approx 2.58 \text{ bits}$$

**Drawing the Ace of Spades:**
$$I(\text{ace of spades}) = -\log_2(1/52) \approx 5.7 \text{ bits}$$

**Back to our movie reviews:**
$$I(\text{5-star on bad movie}) = -\log_2(0.02) \approx 5.64 \text{ bits}$$
$$I(\text{3-star on average movie}) = -\log_2(0.40) \approx 1.32 \text{ bits}$$

That 5-star review on the bad movie carries over 4x the information of the expected 3-star review. Your intuition was right, and now you can quantify it.

## Units: Bits, Nats, and Why You Should Care

The log base determines the unit of information:

- **Base 2 (bits)**: $I(x) = -\log_2 P(x)$
- **Base $e$ (nats)**: $I(x) = -\ln P(x)$
- **Base 10 (hartleys)**: $I(x) = -\log_{10} P(x)$ (rarely used)

### Bits: You Already Think in Bits

As a software engineer, bits are second nature. You know 8 bits make a byte. A fair coin flip carries exactly 1 bit of information -- and that's not a coincidence. One bit is the answer to one yes/no question. A byte gives you $2^8 = 256$ possibilities, so identifying one specific byte value out of 256 carries $\log_2(256) = 8$ bits of information.

**SWE bridge -- information content and compression ratio**: Self-information tells you the *minimum* number of bits needed to encode an event. Rare events need more bits; common events need fewer. This is literally how Huffman coding and arithmetic coding work. If a character appears with probability $P$, the optimal code assigns it $-\log_2 P$ bits. That's why compressed files are smaller -- common patterns get short codes.

### Nats: The Natural Unit

Here's where ML practice diverges from classical information theory. Most textbooks introduce information in bits, but **ML frameworks use natural logarithm (ln), not log base 2**. When you call `torch.nn.CrossEntropyLoss` or `tf.keras.losses.CategoricalCrossentropy`, the computation under the hood uses $\ln$, giving you nats.

Why? Because $\ln$ has nicer derivatives. The derivative of $-\ln(x)$ is simply $-1/x$. No extra constants. Cleaner gradients, simpler backpropagation.

The conversion is straightforward:

$$1 \text{ nat} = \frac{1}{\ln 2} \text{ bits} \approx 1.443 \text{ bits}$$

$$\text{bits} = \text{nats} \times \log_2 e \approx \text{nats} \times 1.443$$

> **Common Mistake**: ML uses natural log (nats), not log base 2 (bits). When papers say "entropy," check which base they mean. A cross-entropy loss of 0.693 nats is the same as 1.0 bits -- it's the entropy of a fair coin. If you mix up the bases, your numerical comparisons will be off by a factor of 1.443.

## The Self-Information Curve

Here's what $I(x) = -\log_2 P(x)$ looks like. Stare at this shape -- you'll see it everywhere in ML:

```
  I(x)
  [bits]
    |
 10 |*
    | *
  8 |  *
    |   *
  6 |    *          <-- 5-star review on bad movie (P=0.02)
    |     *              I = 5.64 bits
  4 |      **
    |        **
  2 |          ***
    |    1 bit--> ****
  1 |  (P=0.5)       *****
    |                      ********
  0 +---+---+---+---+---+---+---+---+--> P(x)
    0  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0

  Key points:
  P = 0.01 --> I = 6.64 bits  (rare event: very surprising)
  P = 0.50 --> I = 1.00 bit   (coin flip: moderate surprise)
  P = 0.99 --> I = 0.01 bits  (near-certain: almost no surprise)
  P = 1.00 --> I = 0.00 bits  (certain: zero surprise)
```

The curve drops steeply near $P = 0$ and flattens out as $P \to 1$. This shape has a direct consequence in ML: your loss function penalizes confident-and-wrong predictions *far* more than it rewards confident-and-right ones. That asymmetry is a feature, not a bug.

## The ML Connection

### Self-Information Is the -log(p) in Cross-Entropy Loss

Here's the punchline. The negative log-likelihood loss used throughout ML is *literally* self-information:

$$\mathcal{L} = -\log P(y | x; \theta)$$

When your classifier assigns probability $q$ to the true label, the loss is $-\log q$. That's the self-information of the true label under your model's distribution. You've been computing self-information every time you've trained a neural network.

### The Gradient Connection

When training with log-loss, the gradient with respect to the predicted probability $q$ is:

$$\frac{\partial (-\log q)}{\partial q} = -\frac{1}{q}$$

This means:
- When $q$ is small (model is surprised by the truth), gradient is large -- big update
- When $q$ is close to 1 (model is confident and correct), gradient is small -- small update

This is exactly what you want: learn aggressively from mistakes, coast when you're already right. The $1/q$ relationship means the gradient literally is the reciprocal of confidence.

### Applications Across ML

1. **Classification loss**: Cross-entropy sums self-information over the training set
2. **Language models**: Perplexity is $2^{\text{average surprise in bits}}$ -- a language model's perplexity of 30 means it's as surprised, on average, as if it were choosing uniformly from 30 words
3. **Anomaly detection**: Flag samples with unusually high self-information -- they're the "5-star review on a bad movie" of your dataset
4. **Compression**: Optimal codes assign $-\log_2 P(x)$ bits to event $x$ -- this is Shannon's source coding theorem

### When to Use Self-Information

- **Measuring how surprising a single event is** to your model
- **Debugging predictions**: Sort samples by loss (= self-information) to find what your model struggles with
- **Anomaly detection**: Samples with $I(x)$ far above the mean are worth investigating
- **Understanding loss landscape**: Each training sample contributes its individual surprise to the total loss

### When to Look Beyond

- **Comparing distributions**: You need KL divergence
- **Average-case analysis**: You need entropy (expected self-information)
- **When probability is 0**: Self-information is undefined; use label smoothing or add a small epsilon

## Code Example

```python
import numpy as np
import matplotlib.pyplot as plt

def self_information(probability, base='e'):
    """
    Calculate self-information (surprise) of an event.

    Parameters:
    -----------
    probability : float or array
        Probability of the event(s), must be in (0, 1]
    base : str
        'e' for nats, '2' for bits, '10' for hartleys

    Returns:
    --------
    float or array : Self-information value(s)
    """
    probability = np.asarray(probability)

    if np.any(probability <= 0) or np.any(probability > 1):
        raise ValueError("Probability must be in (0, 1]")

    if base == 'e':
        return -np.log(probability)
    elif base == '2':
        return -np.log2(probability)
    elif base == '10':
        return -np.log10(probability)
    else:
        raise ValueError("Base must be 'e', '2', or '10'")

# Example 1: Compare surprise for different events
print("=== Self-Information Examples ===\n")

events = [
    ("Fair coin (heads)", 0.5),
    ("Fair die (roll 6)", 1/6),
    ("Deck of cards (Ace of Spades)", 1/52),
    ("Rare disease (1 in 10,000)", 0.0001),
    ("Almost certain event", 0.99),
]

for name, prob in events:
    info_bits = self_information(prob, base='2')
    info_nats = self_information(prob, base='e')
    print(f"{name}:")
    print(f"  Probability: {prob:.6f}")
    print(f"  Surprise: {info_bits:.3f} bits = {info_nats:.3f} nats\n")

# Example 2: Visualize the self-information curve
probabilities = np.linspace(0.001, 1.0, 1000)
surprises = self_information(probabilities, base='2')

plt.figure(figsize=(10, 6))
plt.plot(probabilities, surprises, 'b-', linewidth=2)
plt.xlabel('Probability P(x)', fontsize=12)
plt.ylabel('Self-Information I(x) [bits]', fontsize=12)
plt.title('Self-Information: How Surprise Relates to Probability', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 10)

# Mark some reference points
reference_points = [(0.5, "Coin flip"), (1/6, "Die roll"), (0.01, "1% event")]
for prob, label in reference_points:
    info = self_information(prob, base='2')
    plt.plot(prob, info, 'ro', markersize=8)
    plt.annotate(f'{label}\n{info:.2f} bits',
                 xy=(prob, info),
                 xytext=(prob + 0.1, info + 0.5),
                 fontsize=9)

plt.tight_layout()
plt.savefig('self_information_curve.png', dpi=150)
plt.show()

# Example 3: Verify additivity for independent events
print("=== Additivity Property ===\n")

p_coin = 0.5  # P(heads)
p_die = 1/6   # P(roll 6)

# Joint probability (independent events)
p_joint = p_coin * p_die

# Individual surprises
I_coin = self_information(p_coin, base='2')
I_die = self_information(p_die, base='2')
I_joint = self_information(p_joint, base='2')

print(f"I(heads) = {I_coin:.4f} bits")
print(f"I(roll 6) = {I_die:.4f} bits")
print(f"I(heads) + I(roll 6) = {I_coin + I_die:.4f} bits")
print(f"I(heads AND roll 6) = {I_joint:.4f} bits")
print(f"\nAdditivity verified: {np.isclose(I_coin + I_die, I_joint)}")
```

**Output:**
```
=== Self-Information Examples ===

Fair coin (heads):
  Probability: 0.500000
  Surprise: 1.000 bits = 0.693 nats

Fair die (roll 6):
  Probability: 0.166667
  Surprise: 2.585 bits = 1.792 nats

Deck of cards (Ace of Spades):
  Probability: 0.019231
  Surprise: 5.700 bits = 3.951 nats

Rare disease (1 in 10,000):
  Probability: 0.000100
  Surprise: 13.288 bits = 9.210 nats

Almost certain event:
  Probability: 0.990000
  Surprise: 0.014 bits = 0.010 nats

=== Additivity Property ===

I(heads) = 1.0000 bits
I(roll 6) = 2.5850 bits
I(heads) + I(roll 6) = 3.5850 bits
I(heads AND roll 6) = 3.5850 bits

Additivity verified: True
```

## Exercises

### Exercise 1: Computing Surprise
**Problem**: A spam filter assigns P(spam) = 0.001 to an email that turns out to be spam. What is the self-information in bits?

**Solution**:
$$I(\text{spam}) = -\log_2(0.001) = -\log_2(10^{-3}) = 3 \cdot \log_2(10) \approx 9.97 \text{ bits}$$

The filter was very surprised -- nearly 10 bits of information. This suggests the email had unusual characteristics for spam.

### Exercise 2: Comparing Events
**Problem**: Which carries more information: rolling a 1 on a fair 6-sided die, or drawing a heart from a standard deck?

**Solution**:
- Die roll: $I(1) = -\log_2(1/6) \approx 2.58$ bits
- Drawing heart: $I(\text{heart}) = -\log_2(13/52) = -\log_2(1/4) = 2$ bits

Rolling a specific number carries more information because it's less likely (1/6 < 1/4).

### Exercise 3: Additivity Verification
**Problem**: Verify that for two independent fair coin flips, the information in "both heads" equals the sum of individual informations.

**Solution**:
```python
import numpy as np

# Individual flips
p_single = 0.5
I_single = -np.log2(p_single)  # 1 bit

# Both heads (independent)
p_both = 0.5 * 0.5  # = 0.25
I_both = -np.log2(p_both)  # 2 bits

print(f"I(head) = {I_single} bit")
print(f"I(head) + I(head) = {2 * I_single} bits")
print(f"I(both heads) = {I_both} bits")
# Output: All equal 2 bits, confirming additivity
```

## Summary

- **Self-information** quantifies surprise: $I(x) = -\log P(x)$
- **Rare events** carry more information than common events
- **Units** depend on log base: bits (base 2) or nats (base $e$)
- **Key properties**: non-negative, additive for independent events, zero for certain events
- **ML connection**: Log-loss is literally the surprise of observing the true outcome -- the $-\log(p)$ you see in cross-entropy loss is self-information
- **Foundation**: This is the building block for entropy, cross-entropy, and KL divergence

Single-event surprise is useful. But how do you measure the average surprise of an entire distribution? That's entropy.
