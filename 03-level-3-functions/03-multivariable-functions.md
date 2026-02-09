# Chapter 3: Multivariable Functions

A neural network takes 784 pixel values and outputs 10 class probabilities. That's f: R^784 -> R^10. Most functions in ML have many inputs and many outputs. Understanding multivariable functions is understanding what your model actually does.

---

## Building On

Single-variable functions map one number to one number. But real ML models have hundreds of inputs and dozens of outputs. Let's extend our understanding to functions of many variables.

---

## You Already Know This

If you've written code with more than one parameter, you've already used multivariable functions. The math just gives you a precise language for what you've been doing all along.

| Math Concept | Software Equivalent |
|---|---|
| Multivariable function $f(x_1, x_2, \ldots, x_n)$ | A function with multiple parameters: `def f(x1, x2, ..., xn)` |
| Parameterized function $f(\mathbf{x}; \theta)$ | A function with config: `f(x, theta)` where `theta` is tuned, not provided by the caller at runtime |
| Contour plots | Heatmaps you've seen in monitoring dashboards (latency by time and endpoint, CPU by node and hour) |
| Level sets | Decision boundaries in classification -- the line where your model switches from "cat" to "dog" |

**Running example for this chapter**: Think of a recommendation model: `f(user_features, movie_features; weights) -> predicted_rating`. Multiple inputs, parameters, one output. We'll build up to understanding exactly what that notation means.

---

## Part 1: From One Variable to Two

You already know $f(x) = x^2$. One input, one output. Now consider:

$$f(x_1, x_2) = x_1^2 + x_2^2$$

Two inputs, one output. In code:

```python
def f(x1: float, x2: float) -> float:
    return x1**2 + x2**2
```

This function defines a surface. For every pair $(x_1, x_2)$, you get a height $z = f(x_1, x_2)$. You can visualize it as a bowl -- and that bowl shape is exactly what a loss landscape looks like during training.

### Contour Plots: The Top-Down View

Instead of looking at the bowl from the side, look straight down. Draw curves connecting all points that share the same function value. That's a contour plot.

```
        Contour Plot of f(x1, x2) = x1^2 + x2^2

        x2
        ^
   2.0  |  .  .  .  . 4  .  .  .  .
        |     .  .  . 3  .  .  .
   1.0  |     .  . 2  .  .  .
        |        . 1  .  .
   0.0  | --4--3--2--1--*--1--2--3--4-->  x1
        |        . 1  .  .
  -1.0  |     .  . 2  .  .  .
        |     .  .  . 3  .  .  .
  -2.0  |  .  .  .  . 4  .  .  .  .
        |
        * = minimum at (0, 0), value = 0
        Numbers = contour levels (f = 1, 2, 3, 4)
        Circles are level sets: {(x1, x2) : f(x1, x2) = c}
```

Each ring is a **level set** -- the set of all points where $f$ equals some constant $c$. In this case they're circles because $x_1^2 + x_2^2 = c$ is the equation of a circle with radius $\sqrt{c}$.

**Why you care**: When you visualize a loss landscape, you're looking at a contour plot. Gradient descent moves perpendicular to these contour lines, heading toward the center (the minimum). Decision boundaries in classification are also level sets -- specifically the set where $f(\mathbf{x}) = 0$ (or 0.5 for probability outputs).

---

## Part 2: Functions of n Variables

Two variables was just the warmup. In practice, your inputs are vectors.

**Definition**: A function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ maps an $n$-dimensional input to an $m$-dimensional output.

**Notation**: You can write all the variables out:

$$f(x_1, x_2, \ldots, x_n)$$

Or use vector notation (bold means "this is a vector"):

$$f(\mathbf{x}) \quad \text{where } \mathbf{x} \in \mathbb{R}^n$$

In code, that's the difference between:

```python
# Spelled out
def f(x1, x2, x3, x4, x5):
    ...

# Vector form (what you actually write)
def f(x: np.ndarray) -> np.ndarray:
    ...
```

You always use the vector form in practice. Nobody writes 784 separate parameters.

### The Three Signatures That Matter in ML

**1. Many inputs, one output** -- $f: \mathbb{R}^n \rightarrow \mathbb{R}$

This is regression. Your recommendation model takes user features and movie features and outputs a single predicted rating.

$$f(x_1, x_2, \ldots, x_n) = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$$

Or in vector notation: $f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$

**2. Many inputs, many outputs** -- $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$

This is classification. MNIST takes 784 pixel values and outputs 10 class probabilities:

$$f: \mathbb{R}^{784} \rightarrow \mathbb{R}^{10}$$

**3. Many inputs, many outputs, composed** -- Deep networks

A deep neural network is a composition of functions:

$$f(\mathbf{x}) = f_L \circ f_{L-1} \circ \cdots \circ f_1(\mathbf{x})$$

Each $f_i$ is itself a multivariable function. You've seen this in code as stacking layers.

---

## Part 3: Parameterized Functions -- The Heart of ML

Here's where it gets interesting for ML. Not all inputs to a function play the same role.

Back to the running example. Your recommendation model:

```
f(user_features, movie_features; weights) -> predicted_rating
```

The user features and movie features are **data** -- they change with every request. The weights are **parameters** -- they were set during training and stay fixed when you serve predictions.

### The Notation

$$\hat{y} = f(\mathbf{x}; \theta)$$

or equivalently:

$$\hat{y} = f_\theta(\mathbf{x})$$

Where:
- $\mathbf{x} \in \mathbb{R}^n$ -- input features (data that changes per request)
- $\theta \in \mathbb{R}^p$ -- model parameters (weights and biases, fixed after training)
- $\hat{y} \in \mathbb{R}^m$ -- predicted output

The semicolon is doing important work. It says: "everything before the semicolon is data; everything after is parameters."

> **Common Mistakes**
>
> "$f(\mathbf{x}; \theta)$ means $\theta$ is a parameter, not an input. The semicolon separates data ($\mathbf{x}$) from parameters ($\theta$)." This is not just notational fussiness. During training, you compute gradients with respect to $\theta$ (to update it) but not with respect to $\mathbf{x}$ (that's just your training data). The semicolon tells you which variables the optimizer is allowed to touch.

### Two Perspectives on the Same Function

This is a subtle but important point. The function $f(\mathbf{x}; \theta)$ can be viewed two ways:

**During inference** -- $\theta$ is fixed (your trained model). You vary $\mathbf{x}$ (new data comes in). The function maps inputs to predictions.

**During training** -- $\mathbf{x}$ is fixed (your training batch). You vary $\theta$ (gradient descent updates weights). The loss becomes a function of the parameters.

```
During inference (fixed theta):        During training (fixed x):
Different x -> Different y_hat         Different theta -> Different loss

    y_hat                                  loss
    |   . . .                              |     . (theta_1)
    |  .     .                             |   . (theta_2)
    | .       .                            | . (theta_3)
    +-----------> x                        +-----------> theta
    "What does my model predict?"          "Which weights minimize the loss?"
```

This dual perspective is why the notation matters. The same mathematical object serves two purposes depending on which variables you treat as fixed.

### Common Parameterized Functions in ML

**Linear Model**:
$$f(\mathbf{x}; \theta) = \mathbf{W}\mathbf{x} + \mathbf{b}, \quad \theta = \{\mathbf{W}, \mathbf{b}\}$$

**Neural Network Layer** (adds a nonlinearity):
$$f(\mathbf{x}; \theta) = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b}), \quad \theta = \{\mathbf{W}, \mathbf{b}\}$$

**Deep Neural Network** (composition of layers):
$$f(\mathbf{x}; \theta) = f_L \circ f_{L-1} \circ \cdots \circ f_1(\mathbf{x})$$
$$\theta = \{\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2, \ldots, \mathbf{W}_L, \mathbf{b}_L\}$$

For the running example, the recommendation model might be:

```python
def recommend(user_features, movie_features, weights):
    """
    f(user_features, movie_features; weights) -> predicted_rating

    In math: f(x; theta) where x = concat(user_features, movie_features)
    """
    x = np.concatenate([user_features, movie_features])
    hidden = relu(weights['W1'] @ x + weights['b1'])
    rating = weights['W2'] @ hidden + weights['b2']
    return rating
```

---

## Part 4: Surfaces, Level Sets, and Why They Matter

For a function $f: \mathbb{R}^2 \rightarrow \mathbb{R}$, the graph $z = f(x_1, x_2)$ is a surface in 3D space.

A **level set** is the collection of all inputs that produce the same output value:

$$\{(x_1, x_2) : f(x_1, x_2) = c\}$$

For a function of two variables, level sets are curves (the contour lines you saw earlier). For a function of three variables, level sets are surfaces. For $n$ variables, level sets are $(n-1)$-dimensional.

### Where Level Sets Show Up in ML

**Loss landscapes**: The contour plot of your loss function over parameter space. Gradient descent follows a path that cuts across these contours toward the minimum.

**Decision boundaries**: In binary classification, the decision boundary is the level set $\{(\mathbf{x}) : f(\mathbf{x}; \theta) = 0.5\}$. Everything on one side is class 0; everything on the other side is class 1.

**Regularization**: L2 regularization adds a penalty $\lambda \|\theta\|^2$. The level sets of $\|\theta\|^2 = c$ are circles (in 2D) or hyperspheres (in higher dimensions). The regularized optimum is where the loss contours are tangent to a regularization contour.

---

## The Universal ML Equation

Every supervised learning problem follows one pattern:

$$\theta^* = \arg\min_\theta \mathcal{L}(f(\mathbf{X}; \theta), \mathbf{y})$$

Where:
- $f(\mathbf{X}; \theta)$ -- model predictions (a parameterized multivariable function)
- $\mathbf{y}$ -- true labels
- $\mathcal{L}$ -- loss function (also a multivariable function)
- $\theta^*$ -- optimal parameters

Everything in this equation is a multivariable function. The model $f$ is. The loss $\mathcal{L}$ is. Even the optimization process (gradient descent) requires understanding how these functions change with respect to each variable -- which is what partial derivatives and gradients give you.

### Parameter Counts in Modern Models

To give you a sense of scale for $\theta$:

| Model | Parameters ($\|\theta\|$) |
|-------|------------|
| Linear Regression (100 features) | 101 |
| Small CNN (MNIST) | ~100K |
| ResNet-50 | ~25M |
| BERT-base | ~110M |
| GPT-3 | 175B |
| GPT-4 | ~1.7T (estimated) |

Every one of these parameters is a dimension in the space that gradient descent navigates. GPT-3's loss landscape is a surface in 175-billion-dimensional space. The math is the same as $f(x_1, x_2) = x_1^2 + x_2^2$. Just... more dimensions.

---

## Code Example

```python
import numpy as np

# =============================================================================
# PART 1: FUNCTIONS OF MANY INPUTS
# =============================================================================

def linear_multivariable(x, weights, bias):
    """
    Linear function with multiple inputs.

    f(x) = w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b = w . x + b

    This is the building block of linear regression and neural network layers.
    """
    return np.dot(weights, x) + bias


def demonstrate_multivariable():
    """
    Running example: recommendation model.
    Here we simplify to a linear model predicting movie ratings.
    """
    # Features: [user_age, user_avg_rating, movie_year, movie_avg_rating, genre_match]
    x = np.array([25.0, 3.8, 2020, 4.2, 0.9])

    # Learned weights (these are parameters theta, fixed after training)
    weights = np.array([0.01, 0.5, -0.001, 0.8, 1.2])
    bias = 1.0

    predicted_rating = linear_multivariable(x, weights, bias)

    print("=" * 60)
    print("MULTIVARIABLE FUNCTION: Movie Rating Prediction")
    print("=" * 60)
    print(f"Input x (5 features): {x}")
    print(f"Weights (theta):      {weights}")
    print(f"Bias (theta):         {bias}")
    print(f"\nPredicted rating: {predicted_rating:.2f}")
    print(f"\nBreakdown:")
    print(f"  Base rating:          {bias:>8.2f}")
    for i, (name, w, xi) in enumerate(zip(
        ["user_age", "user_avg_rating", "movie_year", "movie_avg_rating", "genre_match"],
        weights, x
    )):
        print(f"  + {name:20s} {w:>6.3f} * {xi:<8.1f} = {w * xi:>8.2f}")
    print(f"  = Predicted rating:   {predicted_rating:>8.2f}")


# =============================================================================
# PART 2: PARAMETERIZED FUNCTIONS -- y = f(x; theta)
# =============================================================================

class LinearModel:
    """
    Parameterized linear function: f(x; theta) = Wx + b

    theta = {W, b} are the parameters.
    x is the data input.
    The semicolon in the math means W and b are tuned, not provided per-call.
    """

    def __init__(self, input_dim, output_dim):
        """Initialize parameters theta = {W, b}."""
        self.W = np.random.randn(output_dim, input_dim) * 0.1
        self.b = np.zeros(output_dim)

    @property
    def parameters(self):
        """Return all parameters as a dictionary."""
        return {'W': self.W, 'b': self.b}

    @property
    def num_parameters(self):
        """Count total number of parameters."""
        return self.W.size + self.b.size

    def forward(self, x):
        """Compute f(x; theta) = Wx + b."""
        return np.dot(self.W, x) + self.b

    def __call__(self, x):
        return self.forward(x)


class NeuralNetwork:
    """
    Multi-layer parameterized function (neural network).

    f(x; theta) = f_L(f_{L-1}(...f_1(x)))

    Each layer has its own parameters, and theta is the collection of all.
    """

    def __init__(self, layer_dims):
        """
        Initialize network with given layer dimensions.

        layer_dims: [input_dim, hidden1_dim, hidden2_dim, ..., output_dim]
        """
        self.layers = []
        self.layer_dims = layer_dims

        for i in range(len(layer_dims) - 1):
            layer = {
                'W': np.random.randn(layer_dims[i+1], layer_dims[i]) * 0.1,
                'b': np.zeros(layer_dims[i+1])
            }
            self.layers.append(layer)

    @property
    def parameters(self):
        """Return all parameters."""
        return self.layers

    @property
    def num_parameters(self):
        """Count total parameters."""
        count = 0
        for layer in self.layers:
            count += layer['W'].size + layer['b'].size
        return count

    def forward(self, x):
        """Forward pass through all layers."""
        activation = x

        # All layers except last: linear + ReLU
        for i, layer in enumerate(self.layers[:-1]):
            z = np.dot(layer['W'], activation) + layer['b']
            activation = np.maximum(0, z)  # ReLU

        # Last layer: linear only
        last_layer = self.layers[-1]
        output = np.dot(last_layer['W'], activation) + last_layer['b']

        return output

    def __call__(self, x):
        return self.forward(x)


def demonstrate_parameterized_functions():
    """Show the y = f(x; theta) paradigm."""
    print("\n" + "=" * 60)
    print("PARAMETERIZED FUNCTIONS: y = f(x; theta)")
    print("=" * 60)

    # Simple linear model
    print("\n--- Linear Model (like a single-layer recommendation model) ---")
    model = LinearModel(input_dim=5, output_dim=1)

    x = np.array([25.0, 3.8, 2020.0, 4.2, 0.9])  # user + movie features
    y = model(x)

    print(f"Input x shape: {x.shape}  (5 features: user + movie)")
    print(f"Output y shape: {y.shape}  (1 predicted rating)")
    print(f"Number of parameters: {model.num_parameters}")
    print(f"  W shape: {model.parameters['W'].shape} = {model.parameters['W'].size} params")
    print(f"  b shape: {model.parameters['b'].shape} = {model.parameters['b'].size} params")

    # Neural network
    print("\n--- Deep Recommendation Network ---")
    # Architecture: 5 features -> 64 hidden -> 32 hidden -> 1 rating
    nn = NeuralNetwork([5, 64, 32, 1])

    y_pred = nn(x)

    print(f"Architecture: {nn.layer_dims}")
    print(f"Input x shape: {x.shape}")
    print(f"Output y shape: {y_pred.shape}")
    print(f"Total parameters: {nn.num_parameters:,}")

    print("\nParameter breakdown:")
    for i, layer in enumerate(nn.layers):
        w_params = layer['W'].size
        b_params = layer['b'].size
        print(f"  Layer {i+1}: W{layer['W'].shape} + b{layer['b'].shape}"
              f" = {w_params + b_params:,} params")

    # Now show a larger example
    print("\n--- MNIST Classifier (for scale) ---")
    mnist_nn = NeuralNetwork([784, 256, 128, 10])
    x_image = np.random.randn(784)
    y_logits = mnist_nn(x_image)

    print(f"Architecture: {mnist_nn.layer_dims}")
    print(f"Total parameters: {mnist_nn.num_parameters:,}")
    print(f"Input: 784 pixels -> Output: 10 class scores")


# =============================================================================
# PART 3: TRAINING VS INFERENCE -- TWO VIEWS OF f(x; theta)
# =============================================================================

def demonstrate_training_vs_inference():
    """
    Show how the same function is used differently:
    - Training: vary theta while x is fixed (from training data)
    - Inference: vary x while theta is fixed (from trained model)
    """
    print("\n" + "=" * 60)
    print("TRAINING vs INFERENCE: Two Views of f(x; theta)")
    print("=" * 60)

    np.random.seed(42)
    model = LinearModel(input_dim=3, output_dim=1)

    # Training data (fixed during training)
    X_train = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    y_true = np.array([14.0, 32.0, 50.0])

    print("\n--- Training Phase: fix x, adjust theta ---")
    for iteration in range(3):
        model.W = np.random.randn(1, 3)
        model.b = np.random.randn(1)

        predictions = np.array([model(x)[0] for x in X_train])
        mse = np.mean((predictions - y_true) ** 2)

        print(f"\n  Iteration {iteration + 1}:")
        print(f"    theta (W): {model.W.flatten()}")
        print(f"    theta (b): {model.b}")
        print(f"    Predictions: {predictions}")
        print(f"    MSE Loss: {mse:.2f}")

    # Set "trained" weights
    model.W = np.array([[2.0, 2.0, 2.0]])
    model.b = np.array([0.0])

    print("\n--- Inference Phase: fix theta, vary x ---")
    print(f"  Trained theta: W={model.W.flatten()}, b={model.b}")

    X_test = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 3.0, 4.0]
    ])

    for x in X_test:
        y_pred = model(x)[0]
        print(f"  f({x}; theta) = {y_pred:.2f}")


# =============================================================================
# PART 4: CONTOUR PLOT AND LEVEL SETS
# =============================================================================

def demonstrate_2d_function():
    """
    Visualize a function of two variables and its level sets.
    f(x1, x2) = x1^2 + x2^2 (paraboloid -- like a loss landscape)
    """
    print("\n" + "=" * 60)
    print("2D FUNCTION: SURFACE AND LEVEL SETS")
    print("=" * 60)

    x = np.linspace(-2, 2, 5)
    y = np.linspace(-2, 2, 5)

    def loss(x, y):
        return x**2 + y**2

    print("f(x1, x2) = x1^2 + x2^2  (a loss surface)")
    print("\nFunction values (minimum at origin):")
    print("         x2=-2   x2=-1   x2=0    x2=1    x2=2")

    for xi in x:
        row = [f"{loss(xi, yi):6.2f}" for yi in y]
        print(f"x1={xi:4.1f}  " + "  ".join(row))

    print(f"\nMinimum at (0, 0): f(0, 0) = {loss(0, 0)}")
    print("Level set f = 1:  all (x1, x2) where x1^2 + x2^2 = 1  (unit circle)")
    print("Level set f = 4:  all (x1, x2) where x1^2 + x2^2 = 4  (circle, r=2)")
    print("\nGradient descent follows the steepest path across these contours.")


# =============================================================================
# BATCH PROCESSING (bonus -- real code always uses batches)
# =============================================================================

def batch_linear(X, W, b):
    """
    Process multiple inputs at once: Y = X @ W.T + b

    X: (batch_size, input_dim)
    W: (output_dim, input_dim)
    b: (output_dim,)
    Y: (batch_size, output_dim)
    """
    return np.dot(X, W.T) + b


def demonstrate_batch_processing():
    """Show efficient batch computation."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING: Many Inputs at Once")
    print("=" * 60)

    W = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])  # 2 outputs, 3 inputs
    b = np.array([0.5, -0.5])

    X = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ])  # 4 samples, 3 features each

    Y = batch_linear(X, W, b)

    print(f"Input batch X shape:  {X.shape}  (4 samples, 3 features)")
    print(f"Weight matrix W shape: {W.shape}  (2 outputs, 3 inputs)")
    print(f"Output batch Y shape: {Y.shape}  (4 samples, 2 outputs)")

    print("\nInput -> Output mapping:")
    for i in range(len(X)):
        print(f"  x={X[i]} -> y={Y[i]}")


# Run all demonstrations
if __name__ == "__main__":
    demonstrate_multivariable()
    demonstrate_parameterized_functions()
    demonstrate_training_vs_inference()
    demonstrate_2d_function()
    demonstrate_batch_processing()
```

---

## When to Pay Attention vs. When to Move On

### Pay Attention When:
- **Designing network architectures** -- you need to know that the output dimension of layer $k$ must match the input dimension of layer $k+1$
- **Debugging shape mismatches** -- `RuntimeError: mat1 and mat2 shapes cannot be multiplied` means your dimensions don't agree
- **Understanding what your model is** -- it's a parameterized multivariable function, and that perspective clarifies everything from training to deployment
- **Reading ML papers** -- the notation $f(\mathbf{x}; \theta)$ appears on every page

### Common Mistakes

1. **Confusing $f(\mathbf{x}, \theta)$ with $f(\mathbf{x}; \theta)$**: The comma version treats $\theta$ as just another input. The semicolon version makes it clear that $\theta$ is a parameter to be learned. Many texts are sloppy about this, but the distinction matters when you think about gradients.

2. **Forgetting the bias term**: The "+b" in $\mathbf{W}\mathbf{x} + \mathbf{b}$ is a separate parameter. Without it, your function is forced to pass through the origin, which is rarely what you want.

3. **Confusing data flow and gradient flow**: Data ($\mathbf{x}$) flows forward through the network. Gradients flow backward to update parameters ($\theta$). Understanding which variables are which prevents confusion about what backpropagation actually does.

4. **Ignoring the batch dimension**: In real code, you almost never process a single input. You process batches: $\mathbf{X}$ is $(B, n)$ not $(n,)$. This is a matrix, not a vector, and all the dimension math changes accordingly.

---

## Exercises

### Exercise 1: Parameter Counting

**Problem**: A neural network has architecture [100, 64, 32, 10] (input to output dimensions). How many total parameters does it have? Count both weights and biases.

**Solution**:
- Layer 1: $100 \times 64 + 64 = 6{,}464$ (W + b)
- Layer 2: $64 \times 32 + 32 = 2{,}080$
- Layer 3: $32 \times 10 + 10 = 330$
- **Total: 8,874 parameters**

Think of it this way: each layer's parameter count is `input_dim * output_dim + output_dim`. The `+ output_dim` is the bias.

### Exercise 2: Batch Dimensions

**Problem**: You have a batch of 32 images, each 28x28 pixels. After flattening, what are the dimensions of $\mathbf{X}$? If the first layer has 128 neurons, what are the dimensions of $\mathbf{W}$ and the output?

**Solution**:
- $\mathbf{X}$: $(32, 784)$ -- 32 samples, 784 features each
- $\mathbf{W}$: $(128, 784)$ -- 128 outputs, 784 inputs
- Output $\mathbf{Y} = \mathbf{X}\mathbf{W}^T + \mathbf{b}$: $(32, 128)$

Quick dimension check: $(32, 784) \times (784, 128) = (32, 128)$. The inner dimensions (784) cancel. This is the dimension-matching rule you'll use constantly.

### Exercise 3: The Training Objective

**Problem**: Write out the full optimization objective for training a neural network $f(\mathbf{x}; \theta)$ on a classification task with cross-entropy loss. Identify which parts are multivariable functions.

**Solution**:

$$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{CE}(f(\mathbf{x}_i; \theta), y_i)$$

Where:
$$\mathcal{L}_{CE}(\hat{y}, y) = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

And $\hat{y} = \text{softmax}(f(\mathbf{x}; \theta))$

**Multivariable functions in this equation**:
- $f(\mathbf{x}; \theta)$: maps $\mathbb{R}^n \rightarrow \mathbb{R}^C$ (the model)
- $\mathcal{L}_{CE}$: maps $\mathbb{R}^C \times \mathbb{R}^C \rightarrow \mathbb{R}$ (the loss)
- The total loss as a function of $\theta$: maps $\mathbb{R}^p \rightarrow \mathbb{R}$ (what gradient descent optimizes)

### Exercise 4: Recommendation Model (Running Example)

**Problem**: Your recommendation model takes 10 user features and 15 movie features. It has two hidden layers of size 64 and 32, and outputs a single predicted rating. Write the function signature in mathematical notation and count the parameters.

**Solution**:

$$f: \mathbb{R}^{25} \rightarrow \mathbb{R}^1, \quad f(\mathbf{x}; \theta) \text{ where } \mathbf{x} = [\mathbf{x}_{\text{user}}, \mathbf{x}_{\text{movie}}] \in \mathbb{R}^{25}$$

Parameter count:
- Layer 1: $25 \times 64 + 64 = 1{,}664$
- Layer 2: $64 \times 32 + 32 = 2{,}080$
- Layer 3: $32 \times 1 + 1 = 33$
- **Total: 3,777 parameters**

---

## Summary

- **Multivariable functions** map multiple inputs to outputs: $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$. In code, that's `def f(x: np.ndarray) -> np.ndarray`.
- **Parameterized functions** separate data from learnable weights: $\hat{y} = f(\mathbf{x}; \theta)$. The semicolon tells you which variables the optimizer adjusts.
- **During training**, you fix $\mathbf{x}$ and adjust $\theta$. **During inference**, you fix $\theta$ and vary $\mathbf{x}$.
- **Level sets** are contour lines of constant function value. In ML, they show up as loss landscape contours and decision boundaries.
- **Neural networks** are compositions of parameterized multivariable functions, one per layer.
- **Parameter count** = sum of (input_dim * output_dim + output_dim) across all layers.
- **Batch processing** handles multiple inputs via matrix operations: $\mathbf{Y} = \mathbf{X}\mathbf{W}^T + \mathbf{b}$.

---

## What's Next

You now understand what ML models are mathematically: parameterized multivariable functions. Level 4 introduces the language for working with many variables at once: linear algebra.

**Next Level**: [Level 4 - Linear Algebra](../04-level-4-linear-algebra/README.md)
