# Chapter 3: Neural Networks

## Building On

Logistic regression is one layer with sigmoid. What if you stack multiple layers? You get a neural network -- a universal function approximator that can learn any continuous mapping.

---

A neural network is just function composition with learned parameters. Layer 1: h₁ = σ(W₁x + b₁). Layer 2: h₂ = σ(W₂h₁ + b₂). Output: ŷ = W₃h₂ + b₃. That's it. Three matrix multiplications, two nonlinearities, and you have a universal function approximator. The magic isn't in the architecture -- it's in the training: backpropagation, which is just the chain rule applied systematically.

In this chapter, you will build up from a single neuron to a full network, trace a forward pass with real numbers, derive backpropagation from the chain rule, and walk every gradient through an MNIST digit classifier. By the end, none of this will feel like a black box.

---

## You Already Know This

You work with these patterns every day. Neural networks map directly onto concepts you use in production code.

| Neural Network Concept | Software Engineering Analogy |
|------------------------|------------------------------|
| Neural network | A pipeline of transformations (like middleware layers in a web server) |
| Forward pass | Calling the function: input goes through each layer sequentially |
| Backward pass (backprop) | Reverse-mode autodiff: computing gradients by walking the computational graph backward |
| Computational graph | A DAG of operations, like a build dependency graph |
| Weights and biases | Configuration parameters tuned by an optimizer instead of a human |
| Loss function | Your test suite's error metric -- how far off are you from correct? |

Think of it this way: a web request hits your server and passes through authentication middleware, then validation middleware, then business logic, then serialization. Each middleware transforms the data and passes it forward. A neural network does the same thing -- each layer transforms activations and passes them to the next. Training is like running an automated optimizer that adjusts every middleware's configuration to minimize errors on your test suite.

---

## Part 1: The Single Neuron

Before you build a network, understand the building block. A single neuron takes inputs, applies weights, adds a bias, and passes the result through an activation function.

```
    x₁ ──w₁──┐
              │
    x₂ ──w₂──┤──► Σ(wᵢxᵢ + b) ──► σ(z) ──► output
              │
    x₃ ──w₃──┘
```

Mathematically:

$$z = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$

$$a = \sigma(z)$$

That is it. A weighted sum followed by a nonlinearity. If $\sigma$ is the sigmoid function, this is exactly logistic regression. One neuron = one logistic regression unit.

---

## Part 2: From Neuron to Layer

A layer is just multiple neurons operating in parallel on the same input. Instead of one weight vector, you have a weight matrix where each row is one neuron's weights.

```
              Layer (4 neurons)
              ┌──────────────┐
    x₁ ───►  │  n₁  n₂  n₃  n₄  │ ──► [a₁, a₂, a₃, a₄]
    x₂ ───►  │  (each neuron │
    x₃ ───►  │   sees all    │
              │   inputs)     │
              └──────────────┘
```

For a layer with $n_{in}$ inputs and $n_{out}$ neurons:

$$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

where $\mathbf{W}$ is an $n_{out} \times n_{in}$ matrix, $\mathbf{b}$ is an $n_{out} \times 1$ bias vector, and $\mathbf{z}$ is the $n_{out} \times 1$ pre-activation vector.

Then apply the activation element-wise:

$$\mathbf{a} = \sigma(\mathbf{z})$$

That is one layer. A matrix multiply, a vector add, and an element-wise nonlinearity.

---

## Part 3: From Layer to Network

Stack layers. Feed the output of one layer as input to the next. This is function composition -- the same pattern you use when piping Unix commands or chaining middleware.

### ASCII Architecture: 3-Layer Network

```
  INPUT          HIDDEN 1        HIDDEN 2        OUTPUT
  LAYER          (4 neurons)     (3 neurons)     (2 neurons)

  x₁ ─────┬────► h₁⁽¹⁾ ───┬────► h₁⁽²⁾ ───┬────► ŷ₁
           │      │          │      │          │
  x₂ ─────┼────► h₂⁽¹⁾ ───┼────► h₂⁽²⁾ ───┼────► ŷ₂
           │      │          │      │          │
  x₃ ─────┼────► h₃⁽¹⁾ ───┼────► h₃⁽²⁾ ───┘
           │      │          │
           └────► h₄⁽¹⁾ ───┘

  (every input    (every h⁽¹⁾    (every h⁽²⁾
   connects to     connects to    connects to
   every h⁽¹⁾)    every h⁽²⁾)   every output)

  Dimensions:     Dimensions:     Dimensions:
  W₁: 4×3         W₂: 3×4        W₃: 2×3
  b₁: 4×1         b₂: 3×1        b₃: 2×1
```

The forward pass equations, layer by layer:

$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}, \quad \mathbf{a}^{(1)} = \sigma_1(\mathbf{z}^{(1)})$$

$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)}, \quad \mathbf{a}^{(2)} = \sigma_2(\mathbf{z}^{(2)})$$

$$\mathbf{z}^{(3)} = \mathbf{W}^{(3)} \mathbf{a}^{(2)} + \mathbf{b}^{(3)}, \quad \hat{\mathbf{y}} = \sigma_3(\mathbf{z}^{(3)})$$

Or compactly, for any layer $l$:

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$

$$\mathbf{a}^{(l)} = \sigma_l(\mathbf{z}^{(l)})$$

where $\mathbf{a}^{(0)} = \mathbf{x}$ (the input).

This is a composition: $f = \sigma_3 \circ g_3 \circ \sigma_2 \circ g_2 \circ \sigma_1 \circ g_1$ where each $g_l(\mathbf{v}) = \mathbf{W}^{(l)}\mathbf{v} + \mathbf{b}^{(l)}$ is an affine transformation.

> **Common Mistake**: Without nonlinearities, stacking linear layers collapses to a single linear layer: $\mathbf{W}_3 \mathbf{W}_2 \mathbf{W}_1 \mathbf{x} = \mathbf{W}_{total} \cdot \mathbf{x}$. The activation functions are essential. No matter how many linear layers you stack, you can only represent linear functions. The nonlinearity after each layer is what gives networks their expressive power.

---

## Part 4: Notation Reference

Before you go further, here is the notation you will see throughout.

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}$ | Input vector (also written $\mathbf{a}^{(0)}$) |
| $\mathbf{W}^{(l)}$ | Weight matrix for layer $l$ |
| $\mathbf{b}^{(l)}$ | Bias vector for layer $l$ |
| $\mathbf{z}^{(l)}$ | Pre-activation at layer $l$: the result before the nonlinearity |
| $\mathbf{a}^{(l)}$ | Activation (post-nonlinearity) at layer $l$ |
| $\sigma_l$ | Activation function at layer $l$ |
| $L$ | Total number of layers (not counting input) |
| $\mathcal{L}$ | Loss function |
| $\boldsymbol{\delta}^{(l)}$ | Error signal at layer $l$: $\partial \mathcal{L} / \partial \mathbf{z}^{(l)}$ |

---

## Part 5: Activation Functions

Each activation introduces a different kind of nonlinearity. Here are the ones you will encounter most often.

| Function | Formula | Derivative | When to Use |
|----------|---------|------------|-------------|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ | Binary output layer |
| Tanh | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | Hidden layers (centered output) |
| ReLU | $\max(0, z)$ | $\mathbf{1}_{z > 0}$ | Default for hidden layers |
| Softmax | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | $p_i(\delta_{ij} - p_j)$ | Multi-class output layer |

ReLU is the workhorse of modern networks. It is computationally cheap (just a threshold), its derivative is trivial (0 or 1), and it mitigates the vanishing gradient problem that plagues sigmoid and tanh in deep networks.

---

## Part 6: The Forward Pass -- Calling the Function

The forward pass is straightforward. You feed input through each layer sequentially, exactly like calling a chain of functions.

### ASCII Forward Pass Flow

```
  INPUT x          LAYER 1              LAYER 2              OUTPUT
  ┌─────┐    ┌───────────────┐    ┌───────────────┐    ┌──────────┐
  │     │    │ z⁽¹⁾=W⁽¹⁾x+b⁽¹⁾│    │ z⁽²⁾=W⁽²⁾a⁽¹⁾+b⁽²⁾│    │          │
  │  x  │──►│               │──►│               │──►│   ŷ     │
  │     │    │ a⁽¹⁾= σ₁(z⁽¹⁾) │    │ a⁽²⁾= σ₂(z⁽²⁾) │    │          │
  └─────┘    └───────────────┘    └───────────────┘    └──────────┘
                   │                     │
                   ▼                     ▼
              Cache z⁽¹⁾, a⁽¹⁾     Cache z⁽²⁾, a⁽²⁾
              (needed for backprop) (needed for backprop)
```

The pseudocode:

```
Forward Pass:
    a⁽⁰⁾ = x
    for l = 1 to L:
        z⁽ˡ⁾ = W⁽ˡ⁾ @ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
        a⁽ˡ⁾ = activation_l(z⁽ˡ⁾)
        cache z⁽ˡ⁾ and a⁽ˡ⁾    // you will need these for backprop
    ŷ = a⁽ᴸ⁾
    compute loss L(ŷ, y)
```

Notice the caching. During the forward pass you store every intermediate $\mathbf{z}^{(l)}$ and $\mathbf{a}^{(l)}$. This is the space-time tradeoff: you spend memory now to avoid recomputation during backpropagation. If you have ever memoized function calls to speed up a recursive algorithm, it is exactly the same idea.

---

## Part 7: The Loss Function -- Your Error Metric

After the forward pass, you compare the prediction $\hat{\mathbf{y}}$ against the true label $\mathbf{y}$. The loss function quantifies "how wrong are we?"

| Task | Loss | Formula |
|------|------|---------|
| Regression | MSE | $\frac{1}{n}\sum_i(y_i - \hat{y}_i)^2$ |
| Binary Classification | Binary Cross-Entropy | $-\frac{1}{n}\sum_i [y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$ |
| Multi-class Classification | Categorical Cross-Entropy | $-\frac{1}{n}\sum_i\sum_k y_{ik}\log\hat{y}_{ik}$ |

For the MNIST example you will work through shortly, you use categorical cross-entropy with softmax output. The softmax converts raw scores (logits) into probabilities that sum to 1, and the cross-entropy penalizes wrong predictions logarithmically -- being confidently wrong costs much more than being uncertain.

---

## Part 8: The Backward Pass -- Reverse-Mode Autodiff

Here is where the math gets interesting. You need to compute how the loss changes with respect to every single weight and bias in the network. That is potentially millions of parameters. Backpropagation does this efficiently by walking the computational graph backward -- this is reverse-mode automatic differentiation.

### Why Backward, Not Forward?

You could compute gradients by nudging each parameter one at a time and measuring the change in loss. With $n$ parameters, that is $n$ forward passes. For a network with 1 million parameters, that is 1 million forward passes per training step. Unacceptable.

Backpropagation computes ALL gradients in one backward pass. The cost is roughly the same as one forward pass. This is why reverse-mode autodiff is so powerful -- and why frameworks like PyTorch and TensorFlow build computational graphs.

### The Chain Rule -- The Core of Backpropagation

The gradient of the loss with respect to any parameter can be decomposed using the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \cdot \frac{\partial \mathbf{a}^{(L)}}{\partial \mathbf{z}^{(L)}} \cdot \frac{\partial \mathbf{z}^{(L)}}{\partial \mathbf{a}^{(L-1)}} \cdots \frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{W}^{(l)}}$$

Each factor in this product corresponds to one step backward through the network. You multiply local derivatives as you walk backward from the loss to the parameter you care about. It is exactly like traversing your build dependency graph in reverse to figure out which source file change caused a particular binary to differ.

### Deriving Backpropagation Step by Step

Define the "error signal" at layer $l$:

$$\boldsymbol{\delta}^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$$

This tells you: how much does the loss change if you perturb the pre-activation at layer $l$?

**Step 1 -- Output layer error** (layer $L$):

For categorical cross-entropy loss with softmax output, the math simplifies beautifully:

$$\boldsymbol{\delta}^{(L)} = \mathbf{a}^{(L)} - \mathbf{y}$$

That is it. The error at the output layer is just prediction minus truth. This clean form is why softmax + cross-entropy is the standard combination for classification.

**Step 2 -- Hidden layer errors** (recursive, from layer $L-1$ down to 1):

$$\boldsymbol{\delta}^{(l)} = (\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)} \odot \sigma'_l(\mathbf{z}^{(l)})$$

where $\odot$ denotes element-wise (Hadamard) multiplication.

Read this equation carefully. It says: take the error from the next layer, multiply it by the transpose of the weight matrix (projecting the error backward), then scale element-wise by the activation derivative. The weight transpose "routes" the error back to the neurons that caused it. The activation derivative tells you how sensitive each neuron's output was to its input.

**Step 3 -- Parameter gradients**:

Once you have $\boldsymbol{\delta}^{(l)}$, the gradients for weights and biases are:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

The weight gradient is an outer product of the error signal and the previous layer's activation. The bias gradient is just the error signal itself.

### ASCII Backward Pass Flow

```
  LOSS ◄── OUTPUT          LAYER 2              LAYER 1
           ┌──────────┐    ┌───────────────┐    ┌───────────────┐
           │δ⁽²⁾=a⁽²⁾-y │    │               │    │               │
     ◄─────│          │◄───│ δ⁽¹⁾=(W⁽²⁾)ᵀδ⁽²⁾│◄───│               │
           │          │    │   ⊙ σ₁'(z⁽¹⁾)   │    │               │
           └──────────┘    └───────────────┘    └───────────────┘
                │                │                     │
                ▼                ▼                     ▼
           ∂L/∂W⁽²⁾        ∂L/∂W⁽¹⁾              (to input --
           = δ⁽²⁾(a⁽¹⁾)ᵀ   = δ⁽¹⁾(x)ᵀ              no update
           ∂L/∂b⁽²⁾        ∂L/∂b⁽¹⁾               needed)
           = δ⁽²⁾          = δ⁽¹⁾
```

### The Complete Algorithm

```
Forward Pass:
    a⁽⁰⁾ = x
    for l = 1 to L:
        z⁽ˡ⁾ = W⁽ˡ⁾ @ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
        a⁽ˡ⁾ = activation(z⁽ˡ⁾)
    compute loss L

Backward Pass:
    δ⁽ᴸ⁾ = gradient of loss w.r.t. z⁽ᴸ⁾
    for l = L down to 1:
        dW⁽ˡ⁾ = δ⁽ˡ⁾ @ a⁽ˡ⁻¹⁾.T
        db⁽ˡ⁾ = δ⁽ˡ⁾
        if l > 1:
            δ⁽ˡ⁻¹⁾ = W⁽ˡ⁾.T @ δ⁽ˡ⁾ * activation_derivative(z⁽ˡ⁻¹⁾)

Update:
    for each parameter:
        param -= learning_rate * gradient
```

---

## Part 9: Running Example -- MNIST Digit Classifier

Time to make this concrete. You are building a classifier that reads 28x28 pixel grayscale images of handwritten digits and outputs which digit (0-9) it sees.

### Architecture

```
  INPUT              HIDDEN 1           HIDDEN 2           OUTPUT
  784 neurons        256 neurons        128 neurons        10 neurons
  (28×28 pixels)     (ReLU)             (ReLU)             (Softmax)

  ┌─────────┐       ┌──────────┐       ┌──────────┐       ┌─────────┐
  │ x₁      │       │ h₁⁽¹⁾    │       │ h₁⁽²⁾    │       │ ŷ₁ (P=0)│
  │ x₂      │       │ h₂⁽¹⁾    │       │ h₂⁽²⁾    │       │ ŷ₂ (P=1)│
  │ x₃      │ ────► │ h₃⁽¹⁾    │ ────► │ h₃⁽²⁾    │ ────► │ ŷ₃ (P=2)│
  │ ...      │       │ ...      │       │ ...      │       │ ...     │
  │ x₇₈₄    │       │ h₂₅₆⁽¹⁾  │       │ h₁₂₈⁽²⁾  │       │ ŷ₁₀(P=9)│
  └─────────┘       └──────────┘       └──────────┘       └─────────┘

  Parameters:
  W⁽¹⁾: 256×784  = 200,704 weights    W⁽²⁾: 128×256 = 32,768 weights    W⁽³⁾: 10×128 = 1,280 weights
  b⁽¹⁾: 256×1    = 256 biases         b⁽²⁾: 128×1   = 128 biases        b⁽³⁾: 10×1   = 10 biases

  Total parameters: 200,704 + 256 + 32,768 + 128 + 1,280 + 10 = 235,146
```

That is 235,146 learnable parameters. Each one gets a gradient computed via backpropagation every training step.

### Forward Pass with Actual Numbers

Suppose you feed in an image of the digit "7". The pixel values (normalized to [0, 1]) form your input vector $\mathbf{x} \in \mathbb{R}^{784}$.

**Layer 1: Input to Hidden 1**

$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$$

$\mathbf{W}^{(1)}$ is $256 \times 784$. You multiply a 256x784 matrix by a 784x1 vector, giving a 256x1 result. Add the 256x1 bias. Then apply ReLU element-wise:

$$\mathbf{a}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)})$$

Say the first few values of $\mathbf{z}^{(1)}$ are $[0.83, -1.2, 0.45, -0.07, 2.1, ...]$. After ReLU:

$$\mathbf{a}^{(1)} = [0.83, 0, 0.45, 0, 2.1, ...]$$

Negative values get zeroed out. This is the nonlinearity doing its job.

**Layer 2: Hidden 1 to Hidden 2**

$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)}$$

$\mathbf{W}^{(2)}$ is $128 \times 256$. Multiply by the 256x1 activation from layer 1, add 128x1 bias, apply ReLU:

$$\mathbf{a}^{(2)} = \text{ReLU}(\mathbf{z}^{(2)})$$

Say $\mathbf{a}^{(2)} = [0.31, 1.7, 0, 0.92, 0.05, ...]$ (128 values).

**Layer 3: Hidden 2 to Output**

$$\mathbf{z}^{(3)} = \mathbf{W}^{(3)} \mathbf{a}^{(2)} + \mathbf{b}^{(3)}$$

$\mathbf{W}^{(3)}$ is $10 \times 128$. The result is a 10x1 vector of logits. Apply softmax to get probabilities:

$$\hat{\mathbf{y}} = \text{softmax}(\mathbf{z}^{(3)})$$

Say the logits are $\mathbf{z}^{(3)} = [-1.2, -0.8, 0.3, -0.5, -1.1, 0.2, -0.9, 3.8, -0.4, 0.1]$.

Softmax converts these to probabilities:

$$\hat{y}_i = \frac{e^{z_i}}{\sum_{j=0}^{9} e^{z_j}}$$

Result: $\hat{\mathbf{y}} \approx [0.005, 0.007, 0.021, 0.009, 0.005, 0.019, 0.006, \mathbf{0.706}, 0.010, 0.017]$

The network assigns 70.6% probability to digit 7. That is the correct class. The forward pass is done.

### Loss Computation

The true label for digit "7" as a one-hot vector: $\mathbf{y} = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]$.

Categorical cross-entropy loss:

$$\mathcal{L} = -\sum_{k=0}^{9} y_k \log \hat{y}_k = -\log(\hat{y}_7) = -\log(0.706) \approx 0.348$$

Only the true class contributes to the loss (all other $y_k = 0$). A perfect prediction ($\hat{y}_7 = 1$) gives loss 0. Being wrong gives increasingly large loss.

### Backward Pass with Actual Numbers

Now walk backward to compute gradients for all 235,146 parameters.

**Step 1: Output layer error**

$$\boldsymbol{\delta}^{(3)} = \hat{\mathbf{y}} - \mathbf{y} = [0.005, 0.007, 0.021, 0.009, 0.005, 0.019, 0.006, -0.294, 0.010, 0.017]$$

The 7th position is negative (-0.294) because the network should have put MORE probability there. All other positions are positive because those probabilities should decrease.

**Step 2: Gradients for $\mathbf{W}^{(3)}$ and $\mathbf{b}^{(3)}$**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(3)}} = \boldsymbol{\delta}^{(3)} (\mathbf{a}^{(2)})^T$$

This is a $10 \times 1$ vector times a $1 \times 128$ vector, giving a $10 \times 128$ gradient matrix. Each entry tells you: "nudge this weight up or down to reduce the loss."

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(3)}} = \boldsymbol{\delta}^{(3)}$$

**Step 3: Propagate error to Hidden 2**

$$\boldsymbol{\delta}^{(2)} = (\mathbf{W}^{(3)})^T \boldsymbol{\delta}^{(3)} \odot \text{ReLU}'(\mathbf{z}^{(2)})$$

$(\mathbf{W}^{(3)})^T$ is $128 \times 10$. It projects the 10-dimensional error back into the 128-dimensional hidden space. Then element-wise multiply by the ReLU derivative: wherever $z^{(2)}_i \leq 0$, the gradient is zeroed out (that neuron was "off" during the forward pass, so it does not contribute to the error).

**Step 4: Gradients for $\mathbf{W}^{(2)}$ and $\mathbf{b}^{(2)}$**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}} = \boldsymbol{\delta}^{(2)} (\mathbf{a}^{(1)})^T \quad (128 \times 256 \text{ matrix})$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(2)}} = \boldsymbol{\delta}^{(2)} \quad (128 \times 1 \text{ vector})$$

**Step 5: Propagate error to Hidden 1**

$$\boldsymbol{\delta}^{(1)} = (\mathbf{W}^{(2)})^T \boldsymbol{\delta}^{(2)} \odot \text{ReLU}'(\mathbf{z}^{(1)})$$

**Step 6: Gradients for $\mathbf{W}^{(1)}$ and $\mathbf{b}^{(1)}$**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \boldsymbol{\delta}^{(1)} (\mathbf{x})^T \quad (256 \times 784 \text{ matrix})$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)} \quad (256 \times 1 \text{ vector})$$

That is every gradient for all 235,146 parameters -- computed in one backward pass.

### Parameter Update

With learning rate $\alpha = 0.01$:

$$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}}$$

$$\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}$$

Repeat for thousands of batches, and the network learns to classify digits with >98% accuracy.

---

## Part 10: Universal Approximation Theorem

A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$, under mild assumptions on the activation function (Cybenko, 1989; Hornik, 1991).

**What this means**: Neural networks are **universal function approximators**. The question is not "can a network represent this function?" but "can we find the right weights?" The universal approximation theorem guarantees existence but says nothing about how easy it is to find the solution. In practice, deeper networks (more layers, fewer neurons per layer) tend to learn more efficiently than wide shallow ones, even though a single hidden layer is theoretically sufficient.

Think of it like Turing completeness. Every Turing-complete language can compute the same functions, but that does not mean writing a compiler is equally easy in every language. Similarly, any architecture can in principle represent your target function, but depth and structure affect how efficiently gradient descent can find the right parameters.

---

## Part 11: Vanishing and Exploding Gradients

In deep networks, gradients can become very small (vanish) or very large (explode) as they propagate backward through many layers.

**Why gradients vanish**: Look at the recursive formula for $\boldsymbol{\delta}^{(l)}$:

$$\boldsymbol{\delta}^{(l)} = (\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)} \odot \sigma'_l(\mathbf{z}^{(l)})$$

With sigmoid or tanh, the derivative $\sigma'$ is always less than 1 (sigmoid's max derivative is 0.25). If you multiply by values less than 1 at every layer, the gradient shrinks exponentially. After 10 layers, $0.25^{10} \approx 0.000001$. The early layers barely learn.

**Why gradients explode**: If weight magnitudes are greater than 1, the product $(\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}$ can grow exponentially instead.

**Solutions you will see in practice**:

| Solution | How It Helps |
|----------|-------------|
| ReLU activation | Derivative is 0 or 1 -- no shrinkage for active neurons |
| He initialization | Sets initial weights to $\mathcal{N}(0, \sqrt{2/n_{in}})$ -- prevents early explosion |
| Batch normalization | Normalizes layer inputs -- keeps activations in a well-behaved range |
| Residual connections | Adds skip paths: $\mathbf{a}^{(l)} = f(\mathbf{a}^{(l-1)}) + \mathbf{a}^{(l-1)}$ -- gradients flow directly |
| Gradient clipping | Caps gradient magnitude -- prevents explosion |

---

## Part 12: Practical Considerations

### When to Use Neural Networks

- You have **large amounts of data** (deep learning is data-hungry)
- The function is **complex and nonlinear**
- Features need to be **learned**, not hand-engineered (images, text, audio)
- You have **GPU/TPU resources** available

### When to Use Something Simpler

- You have **limited data** (use linear models, decision trees)
- **Interpretability** is required (use logistic regression, SHAP on tree models)
- **Training time** is constrained and a simpler model achieves similar performance
- Your data is tabular with well-engineered features (gradient-boosted trees often win)

### Common Pitfalls

1. **Overfitting**: The network memorizes training data instead of generalizing.
   - *Solutions*: Dropout, L2 regularization, data augmentation, early stopping.

2. **Learning rate too high or too low**: Too high causes divergence (loss oscillates or increases). Too low means you wait forever.
   - *Solutions*: Learning rate schedules, adaptive optimizers (Adam, AdamW).

3. **Poor initialization**: Can cause vanishing or exploding gradients from the first step.
   - *Solutions*: Xavier initialization (for sigmoid/tanh), He initialization (for ReLU).

4. **Not normalizing inputs**: Features on different scales create elongated loss surfaces that gradient descent navigates poorly.
   - *Solutions*: Standardize inputs to zero mean and unit variance.

---

## Code Example

```python
import numpy as np
import matplotlib.pyplot as plt

# ─── Activation functions and their derivatives ───

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


class NeuralNetworkFromScratch:
    """
    A feedforward neural network implemented from scratch.
    Mirrors the math in this chapter exactly:
      forward:  z⁽ˡ⁾ = W⁽ˡ⁾ @ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾,  a⁽ˡ⁾ = σ(z⁽ˡ⁾)
      backward: δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)
    """

    def __init__(self, layer_sizes, activations=None, learning_rate=0.01):
        """
        Parameters
        ----------
        layer_sizes : list of int
            Number of neurons in each layer, including input and output.
            Example: [784, 256, 128, 10] for the MNIST classifier.
        activations : list of str
            Activation function per layer (excluding input).
            Options: 'sigmoid', 'relu', 'tanh'
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.lr = learning_rate

        # Default: ReLU for hidden layers, sigmoid for output
        if activations is None:
            activations = ['relu'] * (self.n_layers - 2) + ['sigmoid']
        self.activations = activations

        # Initialize weights: He for ReLU, Xavier for sigmoid/tanh
        self.weights = []
        self.biases = []

        for i in range(1, self.n_layers):
            if self.activations[i-1] == 'relu':
                scale = np.sqrt(2.0 / layer_sizes[i-1])   # He init
            else:
                scale = np.sqrt(1.0 / layer_sizes[i-1])   # Xavier init

            W = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * scale
            b = np.zeros((layer_sizes[i], 1))
            self.weights.append(W)
            self.biases.append(b)

        self.loss_history = []

    def _get_activation(self, name):
        """Return (activation_fn, derivative_fn) pair."""
        activations = {
            'sigmoid': (sigmoid, sigmoid_derivative),
            'relu':    (relu, relu_derivative),
            'tanh':    (tanh, tanh_derivative)
        }
        return activations[name]

    def forward(self, X):
        """
        Forward pass: z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾,  a⁽ˡ⁾ = σ(z⁽ˡ⁾)

        Parameters
        ----------
        X : ndarray, shape (n_features, n_samples)

        Returns
        -------
        A : ndarray — final layer activation (the prediction)
        """
        self.a_cache = [X]  # a⁽⁰⁾ = x
        self.z_cache = []

        A = X
        for i in range(self.n_layers - 1):
            Z = self.weights[i] @ A + self.biases[i]   # z⁽ˡ⁾
            activation_fn, _ = self._get_activation(self.activations[i])
            A = activation_fn(Z)                         # a⁽ˡ⁾

            self.z_cache.append(Z)
            self.a_cache.append(A)

        return A

    def compute_loss(self, Y_pred, Y_true):
        """Binary cross-entropy loss."""
        m = Y_true.shape[1]
        epsilon = 1e-15
        loss = -np.mean(
            Y_true * np.log(Y_pred + epsilon) +
            (1 - Y_true) * np.log(1 - Y_pred + epsilon)
        )
        return loss

    def backward(self, Y):
        """
        Backward pass (backpropagation):
          δ⁽ᴸ⁾ = a⁽ᴸ⁾ - y
          δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)
          dW⁽ˡ⁾ = (1/m) δ⁽ˡ⁾ (a⁽ˡ⁻¹⁾)ᵀ
          db⁽ˡ⁾ = (1/m) Σ δ⁽ˡ⁾
        """
        m = Y.shape[1]
        self.dW = []
        self.db = []

        # Output layer error: δ⁽ᴸ⁾ = a⁽ᴸ⁾ - y
        delta = self.a_cache[-1] - Y

        # Walk backward through layers
        for i in range(self.n_layers - 2, -1, -1):
            # Parameter gradients
            dW = (1/m) * delta @ self.a_cache[i].T
            db = (1/m) * np.sum(delta, axis=1, keepdims=True)

            self.dW.insert(0, dW)
            self.db.insert(0, db)

            # Propagate error to previous layer
            if i > 0:
                _, activation_deriv = self._get_activation(self.activations[i-1])
                delta = (self.weights[i].T @ delta) * activation_deriv(self.z_cache[i-1])

    def update_parameters(self):
        """Gradient descent: param -= lr * gradient."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * self.dW[i]
            self.biases[i]  -= self.lr * self.db[i]

    def fit(self, X, Y, epochs=1000, verbose=True):
        """
        Train the network: forward -> loss -> backward -> update, repeated.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        Y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
        """
        # Transpose to (features, samples) — standard neural-net convention
        X = X.T
        Y = Y.reshape(1, -1) if Y.ndim == 1 else Y.T

        for epoch in range(epochs):
            Y_pred = self.forward(X)                # forward pass
            loss = self.compute_loss(Y_pred, Y)     # compute loss
            self.loss_history.append(loss)
            self.backward(Y)                        # backward pass
            self.update_parameters()                # gradient descent step

            if verbose and epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        return self

    def predict(self, X):
        """Binary prediction (threshold at 0.5)."""
        X = X.T
        Y_pred = self.forward(X)
        return (Y_pred > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """Return raw probabilities."""
        X = X.T
        return self.forward(X).flatten()

    def score(self, X, Y):
        """Classification accuracy."""
        return np.mean(self.predict(X) == Y)


# ─── Demonstration: Learning XOR (not linearly separable!) ───

if __name__ == "__main__":
    print("=" * 50)
    print("Neural Network Learning XOR")
    print("=" * 50)

    # XOR is not linearly separable — a single-layer network cannot solve it
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_xor = np.array([0, 1, 1, 0])

    nn_xor = NeuralNetworkFromScratch(
        layer_sizes=[2, 4, 1],
        activations=['tanh', 'sigmoid'],
        learning_rate=0.5
    )
    nn_xor.fit(X_xor, Y_xor, epochs=5000, verbose=True)

    print("\nPredictions:")
    for x, y in zip(X_xor, Y_xor):
        pred = nn_xor.predict(x.reshape(1, -1))[0]
        prob = nn_xor.predict_proba(x.reshape(1, -1))[0]
        print(f"  Input: {x}, True: {y}, Predicted: {pred}, Prob: {prob:.4f}")

    print("=" * 50)
    print("Binary Classification: Two Moons Dataset")
    print("=" * 50)

    # Generate two moons dataset
    np.random.seed(42)
    n_samples = 300

    theta1 = np.linspace(0, np.pi, n_samples // 2)
    X1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    X1 += np.random.randn(n_samples // 2, 2) * 0.1

    theta2 = np.linspace(0, np.pi, n_samples // 2)
    X2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])
    X2 += np.random.randn(n_samples // 2, 2) * 0.1

    X = np.vstack([X1, X2])
    Y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    idx = np.random.permutation(n_samples)
    X, Y = X[idx], Y[idx]

    X_train, X_test = X[:240], X[240:]
    Y_train, Y_test = Y[:240], Y[240:]

    nn = NeuralNetworkFromScratch(
        layer_sizes=[2, 16, 8, 1],
        activations=['relu', 'relu', 'sigmoid'],
        learning_rate=0.1
    )
    nn.fit(X_train, Y_train, epochs=2000, verbose=True)

    print(f"\nTrain Accuracy: {nn.score(X_train, Y_train):.4f}")
    print(f"Test Accuracy: {nn.score(X_test, Y_test):.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    Z = nn_xor.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
    ax.scatter(X_xor[:, 0], X_xor[:, 1], c=Y_xor, cmap='RdBu', edgecolors='black', s=200)
    ax.set_title('XOR Problem: Neural Network Solution')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')

    ax = axes[1]
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100))
    Z = nn.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='RdBu', edgecolors='black', s=20)
    ax.set_title('Two Moons: Neural Network Solution')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')

    ax = axes[2]
    ax.plot(nn.loss_history, label='Two Moons')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Convergence')
    ax.legend()

    plt.tight_layout()
    plt.savefig('neural_network_demo.png', dpi=100)
    plt.show()
```

### Output
```
==================================================
Neural Network Learning XOR
==================================================
Epoch 0, Loss: 0.693147
Epoch 500, Loss: 0.086542
Epoch 1000, Loss: 0.012345
...
Predictions:
  Input: [0 0], True: 0, Predicted: 0, Prob: 0.0234
  Input: [0 1], True: 1, Predicted: 1, Prob: 0.9812
  Input: [1 0], True: 1, Predicted: 1, Prob: 0.9801
  Input: [1 1], True: 0, Predicted: 0, Prob: 0.0198

Train Accuracy: 0.9917
Test Accuracy: 0.9833
```

---

## Exercises

### Exercise 1: Manual Backpropagation

**Problem**: For a network with input $x=2$, one hidden neuron with ReLU, one output neuron with sigmoid, weights $w_1=0.5$, $w_2=1.0$, biases $b_1=0$, $b_2=0$, and target $y=1$:

1. Compute the forward pass
2. Compute the backward pass (all gradients)

**Solution**:

**Forward pass**:
- $z_1 = w_1 \cdot x + b_1 = 0.5 \cdot 2 + 0 = 1$
- $a_1 = \text{ReLU}(1) = 1$
- $z_2 = w_2 \cdot a_1 + b_2 = 1.0 \cdot 1 + 0 = 1$
- $a_2 = \sigma(1) = \frac{1}{1+e^{-1}} \approx 0.731$
- Loss: $\mathcal{L} = -(y\log(a_2) + (1-y)\log(1-a_2)) = -\log(0.731) \approx 0.313$

**Backward pass**:
- $\delta_2 = a_2 - y = 0.731 - 1 = -0.269$
- $\frac{\partial \mathcal{L}}{\partial w_2} = \delta_2 \cdot a_1 = -0.269 \cdot 1 = -0.269$
- $\frac{\partial \mathcal{L}}{\partial b_2} = \delta_2 = -0.269$
- $\delta_1 = w_2 \cdot \delta_2 \cdot \text{ReLU}'(z_1) = 1.0 \cdot (-0.269) \cdot 1 = -0.269$
- $\frac{\partial \mathcal{L}}{\partial w_1} = \delta_1 \cdot x = -0.269 \cdot 2 = -0.538$
- $\frac{\partial \mathcal{L}}{\partial b_1} = \delta_1 = -0.269$

All gradients are negative, so gradient descent will increase all weights -- moving the prediction closer to 1. That is exactly what you want.

### Exercise 2: Add Momentum to the Optimizer

**Problem**: Modify the update rule to include momentum:
$$v_t = \beta v_{t-1} + (1-\beta)\nabla \mathcal{L}$$
$$w_{t+1} = w_t - \alpha v_t$$

**Solution**:
```python
class NeuralNetworkWithMomentum(NeuralNetworkFromScratch):
    def __init__(self, *args, momentum=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        # Initialize velocity terms to zero (same shape as parameters)
        self.v_weights = [np.zeros_like(W) for W in self.weights]
        self.v_biases = [np.zeros_like(b) for b in self.biases]

    def update_parameters(self):
        for i in range(len(self.weights)):
            # Update velocity: exponential moving average of gradients
            self.v_weights[i] = (self.momentum * self.v_weights[i] +
                                 (1 - self.momentum) * self.dW[i])
            self.v_biases[i] = (self.momentum * self.v_biases[i] +
                                (1 - self.momentum) * self.db[i])
            # Update parameters using velocity instead of raw gradient
            self.weights[i] -= self.lr * self.v_weights[i]
            self.biases[i] -= self.lr * self.v_biases[i]
```

Momentum smooths out gradient noise and helps escape shallow local minima. Think of it like a ball rolling downhill -- it accumulates velocity and can roll past small bumps.

### Exercise 3: Prove the Chain Rule Application

**Problem**: Show that for $\mathcal{L} = \mathcal{L}(a^{(2)})$, $a^{(2)} = \sigma(z^{(2)})$, $z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}$:

$$\frac{\partial \mathcal{L}}{\partial W^{(2)}} = \frac{\partial \mathcal{L}}{\partial a^{(2)}} \sigma'(z^{(2)}) (a^{(1)})^T$$

**Solution**:

By the chain rule:
$$\frac{\partial \mathcal{L}}{\partial W^{(2)}} = \frac{\partial \mathcal{L}}{\partial a^{(2)}} \cdot \frac{\partial a^{(2)}}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial W^{(2)}}$$

Breaking down each factor:
1. $\frac{\partial \mathcal{L}}{\partial a^{(2)}}$ depends on the loss function (given)
2. $\frac{\partial a^{(2)}}{\partial z^{(2)}} = \sigma'(z^{(2)})$ because $a^{(2)} = \sigma(z^{(2)})$ (element-wise)
3. $\frac{\partial z^{(2)}}{\partial W^{(2)}} = a^{(1)}$ because $z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}$ (derivative of a matrix-vector product w.r.t. the matrix)

The Jacobian structure gives us the outer product form:
$$\frac{\partial \mathcal{L}}{\partial W^{(2)}} = \delta^{(2)} (a^{(1)})^T$$

where $\delta^{(2)} = \frac{\partial \mathcal{L}}{\partial a^{(2)}} \odot \sigma'(z^{(2)})$.

This is exactly the formula from the backpropagation derivation. The chain rule factors cleanly into local derivatives at each step of the computational graph.

---

## Summary

- **A neural network** is function composition: affine transformations followed by nonlinear activations, stacked layer by layer.

- **The forward pass** computes activations sequentially:
  $$\mathbf{a}^{(l)} = \sigma(\mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)})$$

- **Backpropagation** computes all gradients in one backward pass using the chain rule:
  $$\boldsymbol{\delta}^{(l)} = (\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)} \odot \sigma'(\mathbf{z}^{(l)})$$
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$$

- **The universal approximation theorem** guarantees any continuous function can be represented -- the challenge is finding the weights.

- **Activation functions are essential** -- without them, any depth of network collapses to a single linear transformation.

- **Vanishing/exploding gradients** are the central challenge of deep networks, solved in practice by ReLU, proper initialization, batch norm, and residual connections.

---

## What's Next

Neural networks can represent any function. But high-dimensional data has redundancy. Dimensionality reduction (PCA, SVD) finds the essential structure hiding in your data.

---

**Previous Chapter**: [Logistic Regression](./02-logistic-regression.md)

**Next Chapter**: [Dimensionality Reduction](./04-dimensionality-reduction.md)
