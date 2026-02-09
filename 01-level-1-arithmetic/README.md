# Level 1: Arithmetic & Numbers

You're three epochs into training a transformer when the loss suddenly becomes `NaN`. You check the architecture, the data, the optimizer config—nothing obvious. Then you switch from `float16` to `float32` and it works. But why? And how do you know when you can safely use lower precision?

Or: You're debugging a recommendation model that won't converge. Gradient descent zigzags wildly instead of descending smoothly. You profile the gradients and realize your features span six orders of magnitude—gene expression levels around 0.001, annual salaries around 1,000,000. The learning rate that works for one feature destroys the other.

These aren't ML problems. They're arithmetic problems. And you can't solve them by tweaking hyperparameters—you need to understand how computers represent numbers, how arithmetic operations accumulate error, and how scaling transforms your optimization landscape.

---

## What You'll Learn

This level covers the numerical foundations that every ML system depends on. These aren't abstract mathematical concepts—they're the reasons your training breaks, your metrics misbehave, and your models fail to converge.

### [Chapter 1: Number Systems](./01-number-systems.md)

**Why this matters**: Your computer can't represent real numbers exactly. Understanding IEEE 754 floating-point formats explains why `0.1 + 0.2 != 0.3`, why gradients underflow in `float16`, and how mixed-precision training actually works.

You'll learn the progression from natural numbers through integers, rationals, and reals—then see how computers approximate this with `float16`, `bfloat16`, `float32`, and `float64`. You'll understand machine epsilon (the smallest representable difference), overflow and underflow (where numbers become infinity or zero), and why certain operations produce `NaN`.

**ML applications**: Debugging NaN loss, choosing precision for mixed-precision training, understanding gradient underflow, implementing numerically stable softmax, knowing when to use Kahan summation for gradient accumulation.

**Key concepts**: Natural → integer → rational → real numbers, IEEE 754 formats, significand/exponent/sign, machine epsilon, overflow/underflow, subnormal numbers, numerical stability patterns.

### [Chapter 2: Arithmetic Operations](./02-arithmetic-operations.md)

**Why this matters**: Floating-point arithmetic is NOT the same as real arithmetic. Addition isn't associative. Division by zero produces `NaN` instead of crashing. Understanding these differences is essential for numerical stability.

You'll learn how addition, multiplication, and division behave with floating-point numbers, where they break mathematical laws you assumed were universal, and practical patterns like Kahan summation (for accurate accumulation), safe division (avoiding `NaN`), and modulo arithmetic (for cyclic learning rates).

**ML applications**: Gradient accumulation without losing precision, avoiding NaN in division, implementing cyclic learning rates, understanding L1 loss (absolute value) and its non-differentiability at zero, leveraging vectorization for performance.

**Key concepts**: Associativity failure in addition, Kahan summation algorithm, multiplication/division overflow, safe division patterns, modulo for cycles, absolute value properties, vectorization and broadcasting.

### [Chapter 3: Ratios and Scales](./03-ratios-and-scales.md)

**Why this matters**: Features with different scales destroy convergence. A feature ranging from 0 to 1 and another from 0 to 1,000,000 create a loss surface that's steep in one direction and flat in another. Gradient descent can't navigate it efficiently.

You'll learn how ratios express relationships (precision/recall, class balance), how percentages and growth rates measure change (learning rate decay), and how different normalization techniques (min-max, z-score, robust, L2) reshape your optimization landscape.

**ML applications**: Feature normalization before training, choosing between min-max and z-score scaling, implementing learning rate decay schedules, computing precision/recall, handling class imbalance with weights, avoiding data leakage in normalization.

**Key concepts**: Ratios and proportions, percentages vs percentage points, growth rates and CAGR, min-max normalization, z-score normalization, robust scaling, L2 normalization, when to use which, data leakage prevention.

---

## Building On

**Prerequisites**: [Level 0: Foundations](../00-level-0-foundations/README.md) gave you the mathematical language (notation, quantifiers, implications), the logical foundations (sets, propositions, proof techniques), and the thinking patterns (abstraction, decomposition, edge cases).

You don't need calculus or linear algebra yet. You just need comfort with reading mathematical notation and following logical arguments—which Level 0 provided.

---

## What Comes Next

**Next level**: [Level 2: Algebra](../02-level-2-algebra/README.md) moves from concrete numbers to abstract patterns. You'll use variables to represent unknowns, solve equations to find them, work with polynomials and exponentials, and manipulate inequalities. Algebra lets you express relationships symbolically instead of numerically—essential for understanding ML formulas like gradient descent, loss functions, and optimization constraints.

After algebra, you'll be ready for functions (Level 3), which combine multiple inputs into outputs—the foundation of ML models themselves.

---

## Navigation

| Chapter | Topic | ML Applications |
|---------|-------|-----------------|
| [1. Number Systems](./01-number-systems.md) | Natural → real numbers, IEEE 754, precision | Mixed-precision training, NaN debugging, gradient underflow |
| [2. Arithmetic Operations](./02-arithmetic-operations.md) | Addition, multiplication, division, modulo, absolute value | Kahan summation, safe division, cyclic LR, L1 loss |
| [3. Ratios and Scales](./03-ratios-and-scales.md) | Ratios, percentages, growth rates, normalization | Feature scaling, LR decay, precision/recall, class weights |

---

## Estimated Time

- **Reading**: 2-3 hours
- **Exercises**: 1-2 hours
- **Total**: 3-5 hours

---

**Start here**: [Chapter 1: Number Systems](./01-number-systems.md) — Learn why `0.1 + 0.2 != 0.3` and what that means for your training loop.
