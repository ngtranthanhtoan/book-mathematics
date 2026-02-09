# Level 2: Algebra - The Grammar of ML Formulas

## Why You Are Here

You've seen this formula a thousand times:

$$\mathcal{L} = -\sum_{i=1}^{n} y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)$$

Binary cross-entropy loss. You've typed it. You've minimized it. But can you *read* it? Can you manipulate it? Can you derive it yourself?

That's algebra. It's the grammar of machine learning — the language that turns intuition into precise computation. Variables are function parameters. Expressions are calculations. Equations are constraints you solve for. And every ML formula you'll ever implement is built from algebraic pieces.

This level teaches you to read, manipulate, and reason about the formulas that power ML.

## What Algebra Gives You

Algebra is not abstract symbol manipulation. It's the foundation of every ML concept you care about:

- **Linear regression**: Solve $X^T X \, w = X^T y$ for optimal weights (Chapter 2: Linear Equations)
- **Cross-entropy loss**: $-\sum y_i \log(\hat{y}_i)$ turns products into sums using logarithms (Chapter 4: Exponentials and Logarithms)
- **Polynomial features**: Degree controls model complexity — degree 2 vs degree 20 is the difference between fitting and overfitting (Chapter 3: Polynomials)
- **Softmax**: $e^{z_i} / \sum e^{z_j}$ converts logits to probabilities (Chapter 4: Exponentials and Logarithms)
- **SVM constraints**: $y_i(w \cdot x_i + b) \geq 1$ defines the margin (Chapter 5: Inequalities)
- **Gradient clipping**: $\|\nabla\| \leq \text{threshold}$ keeps training stable (Chapter 5: Inequalities)

You already use these formulas. This level teaches you to understand them.

## What You Will Learn

### [Chapter 1: Variables and Expressions](./01-variables-and-expressions.md)
Variables are function parameters. Constants are hardcoded values. Unknowns are what you solve for. Learn to read $y = wx + b$ — the simplest ML model — and understand the difference between weights you train ($w, b$), hyperparameters you set (learning rate $\alpha$), and constants that never change ($\pi, e$). Expressions are computations: $wx + b$ is a function call. Algebra is code.

**ML Connection**: Every model is an expression. Linear regression, neural networks, transformers — all are functions of learnable parameters and input data.

### [Chapter 2: Linear Equations](./02-linear-equations.md)
Solving equations is root-finding. The normal equation $X^T X \, w = X^T y$ is a system of linear equations — solve it, and you get optimal weights for linear regression. Master substitution, elimination, and matrix methods. Understand when a system has one solution (unique weights), no solution (infeasible constraints), or infinite solutions (underspecified model).

**ML Connection**: Closed-form solutions like the normal equation are fast but limited. When you can't solve analytically, you iterate (gradient descent). Knowing when a closed form exists saves you training time.

### [Chapter 3: Polynomials](./03-polynomials.md)
Degree 1 underfits. Degree 2 fits well. Degree 20 overfits. Polynomials make the bias-variance tradeoff concrete. A degree-$d$ polynomial has $d+1$ parameters — more parameters means more capacity to memorize noise. Learn polynomial terms, factoring, and roots (where loss hits zero). Watch overfitting happen as you increase degree.

**ML Connection**: Polynomial regression is a simple model where you can see overfitting directly. High-degree polynomials are like neural networks with too many parameters — they memorize the training set and fail on test data.

### [Chapter 4: Exponentials and Logarithms](./04-exponentials-and-logarithms.md)
Logarithms turn products into sums: $\log(xy) = \log(x) + \log(y)$. That's why cross-entropy uses $\log$ — it makes probabilities additive. Exponentials turn logits into probabilities: $e^x$ is always positive, which makes softmax work. Master the product rule, quotient rule, and understand numerical stability (subtract max before exponentiating). See why learning rate schedules use exponential decay.

**ML Connection**: Every time you compute loss, you use $\log$. Every time you compute class probabilities, you use $\exp$. Understanding these functions explains why cross-entropy works and why softmax needs numerical tricks to avoid overflow.

### [Chapter 5: Inequalities](./05-inequalities.md)
Constraints are everywhere in ML. Learning rate $\alpha > 0$. Probabilities in $[0, 1]$. SVM margin $y_i(w \cdot x_i + b) \geq 1$. Gradient norm $\|\nabla\| \leq \text{threshold}$. Inequalities define the feasible region — the valid parameter space your optimizer searches. Learn to solve them, visualize them, and understand when constraints make problems infeasible.

**ML Connection**: Optimization with constraints is constrained optimization. When you clip gradients, you're enforcing an inequality. When you project probabilities to $[0,1]$, you're enforcing constraints. SVMs are entirely defined by inequality constraints.

## Building On: Level 1 (Arithmetic)

Level 1 gave you number systems and arithmetic operations — the building blocks of computation. You learned:
- How numbers work (integers, fractions, decimals)
- Basic operations (addition, multiplication, exponentiation)
- Ratios and scales (normalization, percentages)

Algebra takes those operations and generalizes them with variables. Instead of "compute $3 \times 5$", you write "compute $wx$ for any $w, x$". Instead of "add 7 to 10", you write "add $b$ to any number". Variables make operations reusable — like function parameters make code reusable.

If you skipped Level 1 but are comfortable with basic arithmetic (fractions, exponents, order of operations), you're ready.

## What Comes Next: Level 3 (Functions)

After algebra, you move to **Level 3: Functions**. Algebra works with expressions — formulas you evaluate. Functions are the next abstraction: they're reusable, composable building blocks.

Level 3 teaches you:
- **Functions as core abstraction** (Chapter 1): $f(x) = wx + b$ is a function. So is ReLU, sigmoid, and softmax.
- **Common function types** (Chapter 2): Linear, polynomial, exponential, piecewise (ReLU). These are the activation functions you use every day.
- **Multivariable functions** (Chapter 3): $f(x, y) = x^2 + y^2$ takes multiple inputs. Neural networks are multivariable functions.

Functions are how you think about models. Linear algebra (Level 4) comes after — it scales functions from single inputs to batches.

## Navigation

| Chapter | Topic | Key ML Connection |
|---------|-------|-------------------|
| [1. Variables and Expressions](./01-variables-and-expressions.md) | Variables as parameters | $y = wx + b$ — every model is an expression |
| [2. Linear Equations](./02-linear-equations.md) | Solving for unknowns | Normal equation $X^T X w = X^T y$ |
| [3. Polynomials](./03-polynomials.md) | Degree and complexity | Polynomial degree controls overfitting |
| [4. Exponentials and Logarithms](./04-exponentials-and-logarithms.md) | Log and exp functions | Cross-entropy uses $\log$, softmax uses $\exp$ |
| [5. Inequalities](./05-inequalities.md) | Constraints | Learning rate $> 0$, SVM margin $\geq 1$ |

## How to Use This Level

Work through chapters in order. Each builds on the last. Every chapter includes:
- **A running ML example**: One concrete problem carried through the entire chapter
- **"You Already Know This" sections**: Connecting math symbols to code patterns you use daily
- **Python implementations**: See the math in action, with NumPy and PyTorch
- **Exercises with solutions**: Practice translating formulas to code

Don't just read. Code. The investment you make here pays off the first time you read a paper and actually understand the derivation instead of skipping to the results.

**Start here**: [Chapter 1: Variables and Expressions](./01-variables-and-expressions.md) →
