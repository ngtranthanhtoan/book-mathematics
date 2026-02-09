# Chapter 1: Variables and Expressions

> **Building On** -- Numbers and operations are your atoms. Algebra introduces variables -- placeholders for unknown values. This is where math starts looking like code.

---

## The Punchline

In code, you write `y = w * x + b`. In math, you write y = wx + b. Same thing. Algebra is just programming with pen and paper -- manipulating symbolic expressions to find unknowns. And in ML, the "unknowns" are your model's weights.

You have been doing algebra since you wrote your first function. Every time you parameterize behavior, substitute a value, or simplify a conditional, you are doing exactly what algebra formalizes. This chapter makes that mapping explicit so that the next time you read a paper full of Greek letters, you see code -- not hieroglyphics.

### Running Example -- The Simplest ML Model

We will use one equation throughout this entire chapter:

$$y = wx + b$$

This is linear regression in one variable -- the "Hello World" of machine learning.

| Symbol | Role | SWE Analogy |
|--------|------|-------------|
| $x$ | **Input** (data) | A function argument you pass in |
| $w$ | **Weight** (unknown) | A value the model learns during training |
| $b$ | **Bias** (unknown) | Another learned value -- an offset |
| $y$ | **Output** (prediction) | The return value of your function |

In Python you would write it exactly like this:

```python
def predict(x: float, w: float, b: float) -> float:
    return w * x + b
```

The only difference between the math and the code is that math drops the `*` between `w` and `x`. Everything else is the same.

---

## Variables -- You Already Know This

> **You Already Know This** -- A variable in algebra is the same concept as a function parameter, a config value, or an environment variable. It is a name that stands for a value you may not know yet.

**Formal Definition**: A *variable* is a symbol (usually a letter) that represents a quantity which can take on different values.

That is literally a function parameter:

```python
# 'x' is a variable -- it takes on a different value every time you call the function
def square(x):
    return x ** 2

square(3)   # x = 3 -> 9
square(10)  # x = 10 -> 100
```

In our running example $y = wx + b$, every symbol is a variable. But not all variables play the same role. Let us break them down.

---

### Variables as Unknowns

An **unknown** is a variable whose value you are trying to find. In the equation:

$$2x + 3 = 7$$

The variable $x$ is an unknown. You solve to find $x = 2$.

**Formal Definition**: An unknown is a variable that satisfies some condition or equation. The goal is to determine which value(s) make the equation true.

In our ML running example, $w$ and $b$ are the unknowns. Training is the process of finding values for $w$ and $b$ that make $y$ match reality as closely as possible. The entire optimization loop -- gradient descent, backpropagation, all of it -- exists to solve for these unknowns.

---

### Variables as Parameters

A **parameter** is a variable that represents a fixed (but potentially adjustable) value within a system.

In the linear model:

$$y = mx + b$$

- $x$ and $y$ are variables (inputs/outputs)
- $m$ (slope) and $b$ (intercept) are **parameters**

Parameters define the model's behavior. Different parameter values create different models:

| $m$ | $b$ | Model |
|-----|-----|-------|
| 2 | 1 | $y = 2x + 1$ |
| -1 | 3 | $y = -x + 3$ |
| 0.5 | 0 | $y = 0.5x$ |

> **You Already Know This** -- Parameters vs unknowns map directly to a distinction you already make:
> - **Hyperparameters** = values *you* choose before training (learning rate, batch size, number of layers). These are like config values you set in a YAML file.
> - **Weights** = values the *model* learns during training. These are the unknowns that optimization solves for.
>
> Before training, weights are unknowns. After training, they become fixed parameters for inference. Same variable, different role depending on the phase.

---

### Constants

> **You Already Know This** -- Constants are hardcoded values. `Math.PI`, `Math.E`, the `3` in RGB channels. They never change.

**Constants** are values that never change. They are fixed by definition:

- $\pi \approx 3.14159...$ (ratio of circumference to diameter)
- $e \approx 2.71828...$ (base of natural logarithm)
- Physical constants like speed of light $c$

In ML contexts, constants might include:
- Feature values after standardization
- Fixed architecture parameters (like the 3 in RGB channels)
- The $2$ in the MSE denominator

---

### Anatomy of an ML Function -- ASCII Diagram

Here is the standard notation you will see in every ML paper:

```
        y  =  f( x  ;  theta )
        |     |  |     |
        |     |  |     +--- Parameters (model learns these)
        |     |  +--------- Inputs (data you feed in)
        |     +------------ The model / function
        +------------------ Output / prediction

  The semicolon separates inputs from parameters.
  It means: "these are different kinds of arguments."

  In Python terms:

        def f(x, *, theta):    # x is positional, theta is keyword-only
            return theta @ x    # (simplified)
```

> **Common Mistake** -- In ML notation, $f(x; \theta)$ separates inputs ($x$) from parameters ($\theta$) with a semicolon. The semicolon means "these are different kinds of arguments." A comma would imply they are the same kind. When you see $f(x; \theta)$, think of $x$ as positional args and $\theta$ as the model's internal state.

Mapping this back to our running example:

```
        y  =  f( x  ;  w, b )

  where f(x; w, b) = w * x + b

  Inputs:     x       (the data point)
  Parameters: w, b    (the weight and bias)
  Output:     y       (the prediction)
```

---

## Algebraic Expressions

> **You Already Know This** -- An expression in algebra is the same as an expression in code: anything that evaluates to a value. `3 * x + 2` is an expression in Python. $3x + 2$ is an expression in math. Neither one is a statement -- neither one asserts anything. They just compute.

**Formal Definition**: An *expression* combines variables, constants, and operations. It evaluates to a value but makes no claim about equality.

$$3x^2 + 2xy - 5z + 7$$

**Components of an expression**:

- **Terms**: Parts separated by $+$ or $-$ signs ($3x^2$, $2xy$, $-5z$, $7$)
- **Coefficients**: Numbers multiplying variables ($3$, $2$, $-5$)
- **Variables**: Symbols representing quantities ($x$, $y$, $z$)
- **Constant term**: A term with no variable ($7$)

In our running example $wx + b$:
- There are two terms: $wx$ and $b$
- The coefficient of $x$ is $w$ (which itself is a variable here -- coefficients do not have to be literal numbers)
- The constant term is $b$ (technically also a variable, but it plays the role of a constant offset)

---

### Expression vs Equation

This distinction matters and people confuse it constantly.

$$\text{Expression: } 3x + 2y - 5$$

$$\text{Equation: } 3x + 2y - 5 = 0$$

An **expression** is a mathematical phrase -- it has no equals sign and does not make a claim. An **equation** makes an assertion that two things are equal, creating something you can solve.

| Concept | Math | Code Analogy |
|---------|------|-------------|
| Expression | $3x + 2$ | `3 * x + 2` (evaluates to a value) |
| Equation | $3x + 2 = 8$ | `assert 3 * x + 2 == 8` (makes a claim) |

When you write a loss function, you are writing an *expression* -- it produces a number. When you set up "find $w$ such that the loss equals zero," you have created an *equation* (or more typically, an optimization problem).

---

### Evaluating Expressions

To evaluate an expression, substitute values for variables. You do this every time you call a function.

$$f(x, y) = 3x^2 + 2y$$

When $x = 2$ and $y = 3$:

$$f(2, 3) = 3(2)^2 + 2(3) = 3(4) + 6 = 12 + 6 = 18$$

This is literally a function call:

```python
def f(x, y):
    return 3 * x**2 + 2 * y

f(2, 3)  # 18
```

No mystery here. Evaluation is function invocation.

---

### Simplifying Expressions

**Like terms** have the same variable parts:
- $3x$ and $5x$ are like terms (both have $x$)
- $3x$ and $3y$ are NOT like terms (different variables)
- $2x^2$ and $5x^2$ are like terms (both have $x^2$)
- $2x^2$ and $2x$ are NOT like terms (different powers)

**Combining like terms**:

$$4x + 3y - 2x + 5y = (4x - 2x) + (3y + 5y) = 2x + 8y$$

This is exactly what a compiler does during constant folding and expression simplification. You are the compiler.

---

## Code Example

```python
import numpy as np
from typing import Callable

# =============================================================
# Running example: y = wx + b (the simplest ML model)
# =============================================================

# --- Variables as unknowns: solving for x in 2x + 3 = 7 ---
# Rearranging: x = (7 - 3) / 2
x_unknown = (7 - 3) / 2
print(f"Unknown x = {x_unknown}")  # x = 2.0


# --- Variables as parameters: defining a linear model ---
def linear_model(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """
    Linear model: y = wx + b

    This is our running example -- the simplest ML model.

    Args:
        x: Input variable (features / data)
        w: Weight parameter (learned during training)
        b: Bias parameter (learned during training)

    Returns:
        y: Output predictions
    """
    return w * x + b


# Same input, different parameters = different models
x_data = np.array([1, 2, 3, 4, 5])

# Parameter set 1: steep positive relationship
y1 = linear_model(x_data, w=2.0, b=1.0)
print(f"w=2, b=1: {y1}")  # [3. 5. 7. 9. 11.]

# Parameter set 2: gentle negative relationship
y2 = linear_model(x_data, w=-0.5, b=10.0)
print(f"w=-0.5, b=10: {y2}")  # [9.5 9. 8.5 8. 7.5]

# Notice: same function, same data, different parameters -> different outputs.
# Training is the process of finding which (w, b) pair makes y match reality.


# --- Evaluating expressions ---
def evaluate_expression(x: float, y: float) -> float:
    """
    Evaluate: 3x^2 + 2xy - 5
    """
    return 3 * x**2 + 2 * x * y - 5

result = evaluate_expression(2, 3)
print(f"3(2)^2 + 2(2)(3) - 5 = {result}")  # 19


# --- Simplifying expressions with SymPy ---
from sympy import symbols, simplify, expand

x, y = symbols('x y')

# Original expression with redundant terms
expr = 4*x + 3*y - 2*x + 5*y + x**2 - x**2

# Simplify (combine like terms, cancel)
simplified = simplify(expr)
print(f"Simplified: {simplified}")  # 2*x + 8*y


# --- Expression vs Equation ---
from sympy import Eq, solve

# Expression (no equals sign -- just evaluates to a value)
expression = 3*x + 2

# Equation (asserts equality -- can be solved)
equation = Eq(3*x + 2, 8)

# Solve the equation for x
solution = solve(equation, x)
print(f"Solution to 3x + 2 = 8: x = {solution}")  # [2]


# --- Practical ML example: Loss as an expression ---
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error -- a loss *expression* (it evaluates to a number).

    MSE = (1/n) * sum((y_true - y_pred)^2)

    The loss tells you how wrong the model is. Training minimizes this expression.
    """
    n = len(y_true)
    return (1/n) * np.sum((y_true - y_pred)**2)

y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.1, 2.2, 2.8, 4.1])

loss = mse_loss(y_true, y_pred)
print(f"MSE Loss: {loss:.4f}")  # 0.0225
```

---

## ML Relevance

Let us map the algebra concepts directly to ML practice.

### Features as Variables

In a dataset, each feature column is a variable. The running example scales up naturally:

```python
# Housing price prediction -- multiple features, same structure
# x1 = square_footage, x2 = num_bedrooms, x3 = age
# Each w_i is a weight (unknown) that training learns
price = w1*x1 + w2*x2 + w3*x3 + b
```

This is still $y = wx + b$, just with vectors instead of scalars. The algebra is identical.

### Weights as Unknowns

The weights ($w_1, w_2, w_3$) and bias ($b$) are unknowns that training determines. This is the core of supervised learning -- finding the parameter values that minimize prediction error.

### Hyperparameters as Parameters

Learning rate, batch size, and regularization strength are parameters you set before training begins:

```python
# These are parameters YOU choose (not learned by the model)
learning_rate = 0.01    # How big a step to take each iteration
batch_size = 32         # How many samples per gradient update
lambda_reg = 0.001      # How much to penalize large weights
```

### Loss Functions as Expressions

Loss functions like MSE, Cross-Entropy, and MAE are expressions you evaluate:

$$\mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

The loss value tells you how wrong your predictions are -- a number with meaning but no equality assertion until you set up the optimization problem ("find $\theta$ that minimizes $\mathcal{L}$").

---

## When to Use This / Common Pitfalls

### Use This Knowledge When:
- **Defining model architectures**: Understanding what is a variable vs a parameter helps you structure code. Learned values go in `nn.Parameter`; fixed values go in config.
- **Debugging**: If outputs are wrong, check if you are treating a constant as a variable or vice versa. A classic bug: accidentally making a hyperparameter trainable, or freezing a layer you meant to train.
- **Reading papers**: Mathematical notation uses variables extensively. Knowing the types (input, parameter, constant, unknown) is like knowing the type system of a language -- it tells you what operations are valid.
- **Feature engineering**: Creating new features is building new expressions from existing variables. $x_3 = x_1 \cdot x_2$ is an expression that creates a new feature from two old ones.

### Common Pitfalls:
1. **Confusing parameters and hyperparameters**: Parameters are learned by the model; hyperparameters are set by you. Mixing them up leads to subtle bugs (e.g., making learning rate trainable).
2. **Treating expressions as equations**: An expression evaluates to a value; an equation can be solved. The loss function is an expression. "Minimize the loss" is the optimization problem.
3. **Forgetting variable scope**: A variable $x$ in one equation may mean something completely different in another. In code you would never reuse a variable name for a different purpose in the same scope. Math papers sometimes do. Watch for this.
4. **Mixing up constants and variables**: Using magic numbers instead of named constants hurts readability in code *and* in math.

---

## Exercises

### Exercise 1: Identify Variable Types

In the neural network layer equation $y = \sigma(Wx + b)$, where $\sigma$ is the activation function:

**Question**: Classify each symbol as unknown, parameter, variable, constant, or function.

**Solution**:
- $y$: Output variable (computed from input)
- $W$: Weight matrix (unknown during training, parameter after training)
- $x$: Input variable (changes with each data point)
- $b$: Bias vector (unknown during training, parameter after training)
- $\sigma$: Activation function (constant -- it does not change during training)

### Exercise 2: Simplify the Expression

Simplify: $5a + 3b - 2a + 4b - a + b$

**Solution**:
$$= (5a - 2a - a) + (3b + 4b + b)$$
$$= 2a + 8b$$

### Exercise 3: Evaluate and Implement

Given the expression $f(x, y, z) = 2x^2 - 3xy + z$

**Question**: Write Python code to evaluate this for $x=3$, $y=2$, $z=5$

**Solution**:
```python
def f(x, y, z):
    return 2*x**2 - 3*x*y + z

result = f(3, 2, 5)
# 2(9) - 3(3)(2) + 5 = 18 - 18 + 5 = 5
print(result)  # 5
```

### Exercise 4: Map the ML Notation

Given: $\hat{y} = f(x; W, b) = Wx + b$

**Question**: You have a trained model with $W = [0.5, -0.3, 0.8]$ and $b = 0.1$. A new data point arrives: $x = [2.0, 1.0, 3.0]$. What is $\hat{y}$?

**Solution**:
```python
import numpy as np

W = np.array([0.5, -0.3, 0.8])
b = 0.1
x = np.array([2.0, 1.0, 3.0])

y_hat = W @ x + b  # dot product + bias
# = (0.5*2.0) + (-0.3*1.0) + (0.8*3.0) + 0.1
# = 1.0 - 0.3 + 2.4 + 0.1
# = 3.2
print(y_hat)  # 3.2
```

---

## Summary

- **Variables** are symbols representing quantities. They come in different roles:
  - **Unknowns**: Values you solve for (model weights during training)
  - **Parameters**: Fixed values that define behavior (hyperparameters you set, trained weights at inference)
  - **Constants**: Values that never change ($\pi$, $e$, architectural constants)

- **Expressions** combine variables, constants, and operations without making equality claims. Loss functions are expressions.

- **Equations** assert that two expressions are equal, creating something you can solve. "Find $w$ such that the loss is minimized" turns an expression into an optimization problem.

- The notation $f(x; \theta)$ separates inputs from parameters with a semicolon. Think: positional args vs internal state.

- In ML, understanding variable types helps you:
  - Structure code: what goes in `nn.Parameter` vs what goes in config
  - Distinguish what is learned vs what is set
  - Read mathematical notation as if it were code
  - Debug numerical issues by checking variable roles

- **Key formula**: $y = wx + b$ -- the simplest ML model. Everything in this chapter maps to it.

---

> **What's Next** -- You can write expressions with variables. But when does $wx + b = y$? Solving that equation is how you find optimal weights.

Next: [Chapter 2: Linear Equations](./02-linear-equations.md) -->
