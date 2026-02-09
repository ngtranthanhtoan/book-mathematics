# Chapter 3: Gradients — The Compass for Navigating Loss Landscapes

## Building On

Derivatives measure change in one variable. But your loss function depends on millions of parameters. The gradient packages all those derivatives into a single vector that says "move THIS way to reduce loss fastest."

---

Your neural network has 10 million parameters. To train it, you need to know how the loss changes with respect to EACH of those 10 million numbers — simultaneously. That's 10 million partial derivatives, organized into one object: the gradient.

Let that sink in. Every time PyTorch calls `loss.backward()`, it's computing *millions* of partial derivatives, one for every single parameter in your model, and packaging them into gradient vectors. The gradient is the answer to the question: "Which direction should I move ALL parameters at once to reduce loss the fastest?"

If derivatives are the speedometer on a straight road, the gradient is a compass on a mountain, pointing you downhill in a space with millions of dimensions. And yes, we're going to make that precise.

---

## Running Example: Movie Recommendation Loss

Throughout this chapter, we'll use a concrete scenario. You're building a movie recommendation system. Each user has a preference vector (say, 50 numbers representing how much they like action, comedy, drama, etc.), and each movie has a feature vector (50 numbers too). The predicted rating is the dot product of these vectors.

Your loss function $\mathcal{L}$ measures how far off your predictions are from actual ratings, across all users and movies. The parameters? Every number in every user preference vector and every movie feature vector. Thousands of users times 50 features, plus thousands of movies times 50 features. That's a LOT of parameters.

The gradient $\nabla \mathcal{L}$ tells you how to adjust **every single user preference and movie feature vector simultaneously** to make your predictions better. One gradient computation, one update, all parameters move together in the direction that reduces prediction error fastest.

---

## From One Derivative to Many: The Gradient Vector

### You Already Know This

Think of a topographical map. At any point on the map, the gradient is a compass arrow that:
- **Points directly uphill** (direction of steepest ascent)
- **Has a length** proportional to how steep the hill is

Want to go downhill? Walk in the **opposite** direction of the arrow. That's gradient descent in two dimensions. Now imagine the same thing, but the "hill" lives in 10-million-dimensional space. Your brain can't visualize it, but the math works exactly the same way.

### Code First: Computing a Gradient Numerically

Before the formal definition, let's just compute one. The idea is dead simple: wiggle each parameter a tiny bit, see how the loss changes, and that ratio is the partial derivative.

```python
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    """
    Compute the gradient of f at point x using central differences.

    This is the 'brute force' approach: for each parameter,
    nudge it up by h, nudge it down by h, measure the difference.

    Time complexity: O(n) function evaluations, where n = len(x).
    For 10M parameters, this would take... forever. That's why we need autograd.
    """
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# A simple loss surface: f(x, y) = x^2 + 3xy + y^2
# Think of x and y as two parameters in a tiny model
def f(point):
    x, y = point
    return x**2 + 3*x*y + y**2

point = np.array([1.0, 2.0])
grad = numerical_gradient(f, point)
print(f"Point:    {point}")
print(f"Gradient: {grad}")
# Output: Gradient: [ 8.  7.]
# Translation: "increasing x by 1 would increase f by ~8,
#               increasing y by 1 would increase f by ~7"
```

Notice what just happened: we fed in a point with 2 coordinates, and got back a vector with 2 components. Each component says "here's how much the function changes if you tweak this one parameter." That vector IS the gradient.

### Now the Math: Formal Definition

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$ (takes $n$ inputs, produces one number — like a loss function), the gradient is the vector of all partial derivatives:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Translation**: Stack all the partial derivatives from Chapter 2 into a column vector. That's it. The gradient is just the collection of "how does the output change when I wiggle each input independently?"

**Worked example**: For $f(x, y) = x^2 + 3xy + y^2$:

$$\frac{\partial f}{\partial x} = 2x + 3y, \qquad \frac{\partial f}{\partial y} = 3x + 2y$$

$$\nabla f = \begin{bmatrix} 2x + 3y \\ 3x + 2y \end{bmatrix}$$

At point $(1, 2)$: $\nabla f = \begin{bmatrix} 2(1) + 3(2) \\ 3(1) + 2(2) \end{bmatrix} = \begin{bmatrix} 8 \\ 7 \end{bmatrix}$

Which matches our numerical computation above. Good — the math agrees with the code.

### Autograd: How Real Systems Do It

Numerical gradients are great for checking your work, but painfully slow for real models. Here's how PyTorch does it — using automatic differentiation (autograd), which is both exact AND fast:

```python
import torch

# Same function, now with PyTorch tensors
x = torch.tensor([1.0, 2.0], requires_grad=True)
loss = x[0]**2 + 3*x[0]*x[1] + x[1]**2

loss.backward()  # Compute ALL gradients in one pass

print(f"Gradient: {x.grad}")
# Output: Gradient: tensor([8., 7.])
# Same answer, but computed via the chain rule, not finite differences.
# For 10M parameters, this takes ~2-3x the cost of a single forward pass.
# Numerical gradients would take 20M forward passes. Autograd wins.
```

> **You Already Know This**: `loss.backward()` is calling the chain rule recursively through the computation graph. Every operation PyTorch records during the forward pass gets a corresponding backward rule. It's like a compiler that automatically generates derivative code from your forward code.

---

## Properties of the Gradient: Why It's Special

The gradient isn't just *any* vector of derivatives. It has three remarkable geometric properties:

**1. Direction of steepest ascent.** Among all directions you could move from a point, the gradient direction increases $f$ the fastest.

**2. Magnitude = steepness.** $|\nabla f|$ tells you how steep the surface is in the steepest direction. Large gradient magnitude = the loss is changing rapidly. Small magnitude = you're near a flat region (possibly a minimum!).

**3. Perpendicular to contour lines.** If you draw the "level curves" of the function (points where $f$ has the same value — like elevation lines on a topographical map), the gradient is always perpendicular to these curves.

```
    Contour Plot with Gradient Arrows

    y ^
      |         ___________________
      |        /                   \
      |       /    _____________    \
      |      /    /             \    \
      |     /    /    _______    \    \
      |    |    |    /       \    |    |
      |    |    |   |  min *  |   |    |     * = minimum
      |    |    |    \_______/    |    |
      |     \    \      ^       /    /
      |      \    \     |      /    /
      |       \    \____|_____/    /       Each ---> is a gradient
      |        \        |        /         arrow, perpendicular to
      |         \_______|_______/          the contour it crosses
      |                 |
      |     -grad       |  <--- gradient points UPHILL
      |     (descent)   |       -gradient points DOWNHILL
      +-------------------------------------------> x

    Key insight: gradient arrows always cross contour
    lines at right angles. They point from low to high.
```

### The Directional Derivative

What if you want to know the rate of change in some *specific* direction, not just the steepest one? The directional derivative gives you that:

$$D_\mathbf{u} f = \nabla f \cdot \mathbf{u} = |\nabla f| \cos \theta$$

where $\mathbf{u}$ is a unit vector in your chosen direction, and $\theta$ is the angle between $\nabla f$ and $\mathbf{u}$.

**Translation**: The dot product between the gradient and your direction tells you how fast $f$ changes in that direction. This is maximized when $\mathbf{u}$ is aligned with $\nabla f$ ($\theta = 0$, $\cos 0 = 1$), minimized when opposite ($\theta = \pi$, $\cos \pi = -1$), and zero when perpendicular ($\theta = \pi/2$, moving along a contour).

| Direction | $\cos\theta$ | Rate of change | Interpretation |
|-----------|:---:|:---:|---|
| With gradient ($\theta = 0$) | $+1$ | $+|\nabla f|$ | Steepest ascent (maximum increase) |
| Against gradient ($\theta = \pi$) | $-1$ | $-|\nabla f|$ | Steepest descent (maximum decrease) |
| Perpendicular ($\theta = \pi/2$) | $0$ | $0$ | Moving along a contour (no change) |

---

## Common Mistakes

> **The gradient points UPHILL (steepest ascent). For descent, you go in the NEGATIVE gradient direction.**
>
> This trips people up constantly. The gradient $\nabla f$ points toward *increasing* $f$. If $f$ is your loss and you want to *decrease* it, you move in the $-\nabla f$ direction. Every gradient descent update has that minus sign for a reason:
>
> `params = params - learning_rate * gradient`
>
> Forget the minus sign and your loss goes UP. Ask me how I know.

---

## The Jacobian Matrix: Gradients for Vector-Valued Functions

### The Problem

The gradient handles scalar-valued functions: many inputs, one output. But what about a function that takes a vector in and produces a vector out? A neural network layer, for example: it takes a 512-dimensional input and produces a 256-dimensional output. How do you describe all the partial derivatives of 256 outputs with respect to 512 inputs?

You need a *matrix* of partial derivatives. That's the Jacobian.

### You Already Know This

> Think of the Jacobian as the **schema** of a vector-valued function. Just like a database schema tells you how every column in the output table relates to every column in the input table, the Jacobian tells you how every output component changes with respect to every input component. Each row is one output's derivatives, each column is one input.

```
    Jacobian as a Grid

              Inputs
              x_1     x_2     x_3   ...   x_n
           +-------+-------+-------+---+-------+
    f_1    | df1/  | df1/  | df1/  |   | df1/  |   <- how output 1 depends
           | dx1   | dx2   | dx3   |   | dxn   |      on each input
           +-------+-------+-------+---+-------+
    f_2    | df2/  | df2/  | df2/  |   | df2/  |   <- how output 2 depends
           | dx1   | dx2   | dx3   |   | dxn   |      on each input
           +-------+-------+-------+---+-------+
    ...    |  ...  |  ...  |  ...  |   |  ...  |
           +-------+-------+-------+---+-------+
    f_m    | dfm/  | dfm/  | dfm/  |   | dfm/  |   <- how output m depends
           | dx1   | dx2   | dx3   |   | dxn   |      on each input
           +-------+-------+-------+---+-------+

    Shape: m x n  (outputs x inputs)
    Each ROW = gradient of one output w.r.t. all inputs
    Each COLUMN = how one input affects all outputs
```

### Formal Definition

For a vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is:

$$\mathbf{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\[6pt]
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\[6pt]
\vdots & \vdots & \ddots & \vdots \\[6pt]
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

**Shape**: $m \times n$ (number of outputs $\times$ number of inputs).

**Translation**: Row $i$ of the Jacobian is the gradient of the $i$-th output. The Jacobian is literally "all the gradients, stacked into a matrix." If the function has one output ($m = 1$), the Jacobian is a single row — the gradient transposed.

### Worked Example

For $\mathbf{f}(x, y) = \begin{bmatrix} x^2 + y \\ xy \end{bmatrix}$:

- Output 1: $f_1 = x^2 + y \implies \frac{\partial f_1}{\partial x} = 2x, \quad \frac{\partial f_1}{\partial y} = 1$
- Output 2: $f_2 = xy \implies \frac{\partial f_2}{\partial x} = y, \quad \frac{\partial f_2}{\partial y} = x$

$$\mathbf{J} = \begin{bmatrix}
2x & 1 \\
y & x
\end{bmatrix}$$

At point $(2, 3)$: $\mathbf{J} = \begin{bmatrix} 4 & 1 \\ 3 & 2 \end{bmatrix}$

**ML Connection**: In backpropagation, the chain rule through a layer uses the Jacobian:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \mathbf{J}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$$

This is the "vector-Jacobian product" (VJP) — and it's exactly what `loss.backward()` computes at each layer. You rarely build the full Jacobian matrix; instead, PyTorch computes $\mathbf{J}^\top \mathbf{v}$ directly, which is much cheaper.

### Code: Computing a Jacobian

```python
import numpy as np

def numerical_jacobian(f, x, h=1e-5):
    """Compute the Jacobian matrix numerically."""
    n = len(x)
    f_x = f(x)
    m = len(f_x)
    J = np.zeros((m, n))

    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        J[:, j] = (f(x_plus) - f(x_minus)) / (2 * h)

    return J

def vector_function(point):
    """f(x,y) = [x^2 + y, xy]"""
    x, y = point
    return np.array([x**2 + y, x * y])

def analytical_jacobian(point):
    """Jacobian of [x^2 + y, xy] = [[2x, 1], [y, x]]"""
    x, y = point
    return np.array([
        [2*x, 1],
        [y, x]
    ])

point = np.array([2.0, 3.0])
print(f"Analytical Jacobian:\n{analytical_jacobian(point)}")
print(f"\nNumerical Jacobian:\n{numerical_jacobian(vector_function, point)}")
# Both output:
# [[4. 1.]
#  [3. 2.]]
```

And with PyTorch:

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)

# torch.autograd.functional.jacobian computes the full Jacobian
def f_torch(x):
    return torch.stack([x[0]**2 + x[1], x[0] * x[1]])

J = torch.autograd.functional.jacobian(f_torch, x)
print(f"Jacobian via autograd:\n{J}")
# tensor([[4., 1.],
#         [3., 2.]])
```

---

## The Hessian Matrix: Second-Order Information

### You Already Know This

> If the gradient is velocity (which direction the loss is moving and how fast), the Hessian is **acceleration** (how the gradient itself is changing). It tells you about **curvature** — is the loss surface shaped like a bowl? A saddle? A flat plateau?
>
> This is the difference between knowing "the road slopes down to the left" (gradient) and knowing "the road slopes down to the left AND is getting steeper" (Hessian). Both are useful when deciding how big a step to take.

### Formal Definition

For $f: \mathbb{R}^n \to \mathbb{R}$, the Hessian is the matrix of all second partial derivatives:

$$\mathbf{H} = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\[6pt]
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\[6pt]
\vdots & \vdots & \ddots & \vdots \\[6pt]
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}$$

**Translation**: The Hessian is the Jacobian of the gradient. You took all first derivatives (gradient), and then took the derivative of each of those, with respect to all variables. It's "derivatives of derivatives."

**Key properties**:
- **Symmetric**: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$ (under standard smoothness conditions)
- **Eigenvalues reveal curvature**:
  - All positive eigenvalues → local minimum (bowl pointing up). Loss function is *convex* locally.
  - All negative eigenvalues → local maximum (bowl pointing down)
  - Mixed positive and negative → saddle point. This is extremely common in high-dimensional spaces.

### Worked Example

For $f(x, y) = x^2 + 3xy + y^2$:

- $\frac{\partial^2 f}{\partial x^2} = 2$, $\quad \frac{\partial^2 f}{\partial x \partial y} = 3$
- $\frac{\partial^2 f}{\partial y \partial x} = 3$, $\quad \frac{\partial^2 f}{\partial y^2} = 2$

$$\mathbf{H} = \begin{bmatrix} 2 & 3 \\ 3 & 2 \end{bmatrix}$$

The eigenvalues are $5$ and $-1$ (mixed signs) — this is a **saddle point**. The function curves up in one direction and down in another.

### Code: Computing the Hessian

```python
import numpy as np

def hessian_numerical(f, x, h=1e-5):
    """Compute Hessian matrix numerically using finite differences."""
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
            x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
            x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
            x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h

            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h * h)

    return H

def f(point):
    x, y = point
    return x**2 + 3*x*y + y**2

point = np.array([1.0, 2.0])
H = hessian_numerical(f, point)
print(f"Hessian:\n{H}")
# [[2. 3.]
#  [3. 2.]]

eigenvalues = np.linalg.eigvals(H)
print(f"Eigenvalues: {eigenvalues}")
# [ 5. -1.]  -> mixed signs -> saddle point
```

**ML Relevance**: Second-order optimization methods (Newton's method, L-BFGS, natural gradient) use Hessian information to take smarter steps — instead of just knowing *which direction* to go, they also know *how far* to go based on curvature. The catch? For a model with $n$ parameters, the Hessian is $n \times n$. With 10 million parameters, that's a $10^7 \times 10^7$ matrix — 100 trillion entries. That's why most deep learning uses first-order methods (SGD, Adam) and only approximates second-order information.

---

## Gradient Descent: The Core Algorithm

### You Already Know This

```
// Gradient descent in pseudocode — you've seen this pattern a million times
while (loss > threshold) {
    gradient = compute_gradient(loss_function, params);
    params -= learning_rate * gradient;
}
```

That's it. That's gradient descent. Compute the gradient, take a step in the negative gradient direction, repeat. Every training loop you've ever written is a dressed-up version of this.

### The Math

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t)$$

where:
- $\theta_t$ = parameter vector at step $t$ (all 10 million of them)
- $\eta$ = learning rate (how big a step)
- $\nabla_\theta \mathcal{L}$ = gradient of loss with respect to parameters

**Translation**: New parameters = old parameters minus (learning rate times gradient). The gradient vector has one entry per parameter, each telling you "if you increase this parameter, the loss goes up by this much." Multiply by the learning rate, subtract, and every parameter gets nudged in the right direction.

### Visualizing the Steps

```
    Gradient Descent on a Contour Plot
    (minimizing f(x,y) = x^2 + 4y^2)

    y ^
      |     _________________________
      |    /        _____________    \
      |   /        /     _____   \    \
      |  /        /     /     \   \    \
      | /   S    /     /       \   \    \     S = start (3, 2)
      ||   *----*     /    *    \   \    |    * = min (0, 0)
      | \        *   /   /       \   \    \
      |  \        *-*   /         \   \    \
      |   \        \*  /           \   \    \
      |    \        +*/             \   \    \
      |     \      / * <-- end      \   \    \
      |      \____/   \___________/   \   /
      |       \       \_____________/  /
      |        \_________________________/
      +--------------------------------------------> x

    Each * is one gradient descent step.
    The path zigzags because the contours are elliptical
    (different curvature in x vs y direction).
    The learning rate and curvature together determine step size.
```

### Movie Recommendation Example

Back to our running example. Your movie recommendation loss:

$$\mathcal{L} = \frac{1}{|R|} \sum_{(u, m) \in R} \left( \mathbf{p}_u \cdot \mathbf{q}_m - r_{um} \right)^2$$

where $\mathbf{p}_u$ is user $u$'s preference vector, $\mathbf{q}_m$ is movie $m$'s feature vector, $r_{um}$ is the actual rating, and $R$ is the set of known ratings.

The gradient with respect to user $u$'s preference vector:

$$\nabla_{\mathbf{p}_u} \mathcal{L} = \frac{2}{|R|} \sum_{m: (u,m) \in R} \left( \mathbf{p}_u \cdot \mathbf{q}_m - r_{um} \right) \mathbf{q}_m$$

**Translation**: For each movie that user $u$ has rated, compute the prediction error, multiply it by the movie's feature vector, and sum them up. That tells you how to adjust $u$'s preferences. The gradient with respect to movie vectors is symmetric.

One call to `loss.backward()` computes the gradient for ALL user vectors and ALL movie vectors simultaneously. Then one optimizer step adjusts them all at once.

### Full Code: Gradient Descent from Scratch and with PyTorch

```python
import numpy as np

# ============================================================
# Part 1: Gradient descent from scratch
# ============================================================

def numerical_gradient(f, x, h=1e-5):
    """Compute gradient using central differences."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Minimize f(x,y) = x^2 + 4y^2
# Minimum at (0, 0), loss = 0
def f_bowl(point):
    return point[0]**2 + 4*point[1]**2

def grad_bowl(point):
    return np.array([2*point[0], 8*point[1]])

# Run gradient descent
x = np.array([3.0, 2.0])  # Start far from minimum
learning_rate = 0.1
history = [x.copy()]

for step in range(30):
    grad = grad_bowl(x)
    x = x - learning_rate * grad
    history.append(x.copy())
    if step % 5 == 0:
        loss = f_bowl(x)
        print(f"Step {step:2d}: x={x}, loss={loss:.6f}, |grad|={np.linalg.norm(grad):.4f}")

# Verify we're close to the minimum
print(f"\nFinal: x={x}, loss={f_bowl(x):.8f}")
```

```python
import torch

# ============================================================
# Part 2: Same thing with PyTorch autograd
# ============================================================

x = torch.tensor([3.0, 2.0], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

for step in range(30):
    optimizer.zero_grad()
    loss = x[0]**2 + 4*x[1]**2
    loss.backward()                  # Computes x.grad = [2*x0, 8*x1]
    optimizer.step()                 # x = x - lr * x.grad
    if step % 5 == 0:
        print(f"Step {step:2d}: x={x.data.numpy()}, loss={loss.item():.6f}")

print(f"\nFinal: x={x.data.numpy()}, loss={(x[0]**2 + 4*x[1]**2).item():.8f}")
```

```python
import numpy as np

# ============================================================
# Part 3: Gradient descent on a simple neural network
# ============================================================

def simple_nn(params, x):
    """
    2-layer network: input(2) -> hidden(2) -> output(1)
    params layout: W1(4) + b1(2) + W2(2) + b2(1) = 9 parameters
    """
    W1 = params[:4].reshape(2, 2)
    b1 = params[4:6]
    W2 = params[6:8]
    b2 = params[8]

    h = np.tanh(W1 @ x + b1)   # Hidden layer with tanh activation
    y = W2 @ h + b2             # Linear output
    return y

def loss_fn(params, x, target):
    """Mean squared error."""
    pred = simple_nn(params, x)
    return (pred - target)**2

def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Initialize
np.random.seed(42)
params = np.random.randn(9) * 0.1
x_input = np.array([1.0, 2.0])
target = 0.5

print("Training a tiny neural network with gradient descent:\n")
for step in range(100):
    loss = loss_fn(params, x_input, target)
    grad = numerical_gradient(lambda p: loss_fn(p, x_input, target), params)
    params = params - 0.1 * grad

    if step % 20 == 0:
        pred = simple_nn(params, x_input)
        print(f"Step {step:3d}: loss={loss:.6f}, pred={pred:.4f}, |grad|={np.linalg.norm(grad):.6f}")

print(f"\nFinal prediction: {simple_nn(params, x_input):.4f} (target: {target})")
```

---

## ML Applications: Beyond Vanilla Gradient Descent

### Stochastic Gradient Descent (SGD)

Computing the gradient over the entire dataset is expensive. Instead, estimate it from a random mini-batch:

$$\nabla_\theta \mathcal{L} \approx \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \ell_i$$

**Translation**: Instead of averaging gradients over all 100 million training examples, average over a random sample of 64. The estimate is noisy but unbiased, and you update parameters much more frequently. This is why your training loop has a `DataLoader` with `batch_size`.

### Gradient Clipping

Sometimes gradients explode (especially in RNNs). Gradient clipping rescales them:

$$\mathbf{g}_{\text{clipped}} = \min\left(1, \frac{c}{\|\nabla \mathcal{L}\|}\right) \nabla \mathcal{L}$$

**Translation**: If the gradient vector's length exceeds threshold $c$, shrink it to length $c$ while keeping its direction. In PyTorch: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

### Second-Order Methods

Newton's method uses the Hessian to take optimal-sized steps:

$$\theta_{t+1} = \theta_t - \mathbf{H}^{-1} \nabla_\theta \mathcal{L}$$

**Translation**: Instead of a fixed learning rate, the Hessian inverse automatically scales the step size based on curvature. Steep curvature = small step. Gentle curvature = big step. Converges much faster, but computing $\mathbf{H}^{-1}$ for 10M parameters is impractical. L-BFGS and Adam approximate this cheaply.

---

## When to Care About This (and When to Let PyTorch Handle It)

### You NEED to understand gradients when:
- **Implementing custom loss functions** — if your loss isn't differentiable everywhere, you need to know what happens to the gradient
- **Debugging training** — "why is my loss NaN?" often means exploding gradients. "Why did training stall?" often means vanishing gradients
- **Understanding optimizer behavior** — Adam, SGD with momentum, and AdaGrad all manipulate gradients differently. Knowing what the raw gradient is helps you understand what the optimizer is doing to it
- **Research** — novel architectures, new training techniques, or custom backward passes

### You can safely abstract away gradients when:
- Using standard PyTorch/TensorFlow layers and losses — autograd handles everything
- Fine-tuning pre-trained models — the gradient computation is automatic
- Most day-to-day ML engineering — focus on data, architecture, hyperparameters

### Common Pitfalls in Practice
1. **Vanishing gradients**: Deep networks with sigmoid/tanh activations squash gradients toward zero. Fix: use ReLU, residual connections, or proper initialization.
2. **Exploding gradients**: Gradients grow exponentially through layers (common in RNNs). Fix: gradient clipping, LSTM/GRU architectures.
3. **Noisy gradients**: Small batch sizes = high variance in gradient estimates. Fix: larger batches, learning rate warmup, gradient accumulation.
4. **Saddle points**: In high-dimensional spaces, most critical points are saddle points, not local minima. Modern optimizers handle these reasonably well, but it's worth knowing they exist.

---

## Exercises

### Exercise 1: Compute the Gradient (by hand)

**Problem**: Find $\nabla f$ for $f(x, y, z) = x^2y + e^{yz}$

**Solution**:

$$\frac{\partial f}{\partial x} = 2xy, \qquad \frac{\partial f}{\partial y} = x^2 + ze^{yz}, \qquad \frac{\partial f}{\partial z} = ye^{yz}$$

$$\nabla f = \begin{bmatrix} 2xy \\ x^2 + ze^{yz} \\ ye^{yz} \end{bmatrix}$$

Verify numerically:

```python
import numpy as np

def f(p):
    x, y, z = p
    return x**2 * y + np.exp(y * z)

point = np.array([1.0, 2.0, 0.5])
grad = numerical_gradient(f, point)
print(f"Numerical gradient at (1,2,0.5): {grad}")
# Should be approximately [4.0, 1.0 + 0.5*e^1, 2.0*e^1] = [4.0, 2.359, 5.437]
```

### Exercise 2: Jacobian Matrix

**Problem**: Compute the Jacobian of $\mathbf{f}(x, y) = \begin{bmatrix} \sin(xy) \\ x + y^2 \end{bmatrix}$

**Solution**:

Row 1 (gradient of $\sin(xy)$): $\frac{\partial}{\partial x} \sin(xy) = y\cos(xy)$, $\quad \frac{\partial}{\partial y} \sin(xy) = x\cos(xy)$

Row 2 (gradient of $x + y^2$): $\frac{\partial}{\partial x}(x + y^2) = 1$, $\quad \frac{\partial}{\partial y}(x + y^2) = 2y$

$$\mathbf{J} = \begin{bmatrix}
y\cos(xy) & x\cos(xy) \\
1 & 2y
\end{bmatrix}$$

### Exercise 3: Gradient Descent Step (by hand)

**Problem**: Given $f(x, y) = x^2 + 4y^2$ and starting point $(2, 1)$ with learning rate $\eta = 0.1$, compute one gradient descent step.

**Solution**:
- Gradient: $\nabla f = \begin{bmatrix} 2x \\ 8y \end{bmatrix} = \begin{bmatrix} 4 \\ 8 \end{bmatrix}$ at $(2, 1)$
- Update: $(x', y') = (2, 1) - 0.1 \cdot (4, 8) = (1.6, 0.2)$
- Old loss: $f(2, 1) = 4 + 4 = 8$
- New loss: $f(1.6, 0.2) = 2.56 + 0.16 = 2.72$
- Loss decreased from 8 to 2.72 — gradient descent is working!

### Exercise 4 (Coding Challenge): Gradient Descent on the Movie Loss

**Problem**: Implement gradient descent for a simple matrix factorization recommender. Create random user vectors and movie vectors, generate synthetic ratings, and train with gradient descent. Track the loss over time.

Hint: Use `torch.nn.Embedding` for the vectors and `torch.optim.SGD` for the optimizer.

---

## Summary

| Concept | What It Is | Shape | ML Role |
|---------|-----------|-------|---------|
| **Gradient** $\nabla f$ | Vector of all partial derivatives of a scalar function | $n \times 1$ | Tells optimizer which direction to move parameters |
| **Jacobian** $\mathbf{J}$ | Matrix of all partial derivatives of a vector function | $m \times n$ | Chain rule through layers (backprop) |
| **Hessian** $\mathbf{H}$ | Matrix of all second partial derivatives | $n \times n$ | Curvature info for second-order optimization |
| **Gradient Descent** | $\theta \leftarrow \theta - \eta \nabla \mathcal{L}$ | — | THE training algorithm for neural networks |

Key takeaways:
- The gradient of a loss function is a vector with **one entry per parameter**, each saying "this is how much the loss changes if you tweak this parameter"
- The gradient points **uphill** (steepest ascent). Negate it for descent.
- The gradient is **perpendicular to contour lines** of the loss surface
- The Jacobian generalizes gradients to vector-valued functions. Backprop = chain of Jacobian-vector products.
- The Hessian captures curvature. Useful in theory, too expensive to compute in practice for large networks.
- Autograd (`loss.backward()`) computes exact gradients efficiently using the chain rule — no finite differences needed.

---

## What's Next

The gradient tells you which direction to move. But how FAR should you move? And how do you know when you've found the minimum? What if the loss surface has many minima — how do you find the best one? That's optimization theory.

*Next: Chapter 4 — Optimization, where we explore learning rates, momentum, Adam, and the art of navigating loss landscapes without getting stuck.*
