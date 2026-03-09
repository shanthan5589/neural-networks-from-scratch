# Deep Neural Network from Scratch

An L-layer neural network built using only NumPy — no frameworks. The number of layers and nodes per layer are fully configurable. Applied to the Cat vs Non-Cat dataset.

---

## Motivation

The shallow neural network added one hidden layer and gained the ability to learn non-linear decision boundaries. But one hidden layer is still limited in what it can represent.

A deeper network stacks multiple hidden layers. Each layer learns to recognize increasingly abstract patterns — early layers detect simple combinations of features, later layers combine those into more complex representations. This is the same principle behind how the visual cortex in the brain processes information.

The question is: can we generalize the forward and backward pass equations to work for any number of layers, without writing separate code for each layer?

The answer is yes — using for loops.

---

## Architecture

For a network with L layers:

```
X → [Z[1]=W[1]·X+b[1] → A[1]=tanh(Z[1])] → ... → [Z[L]=W[L]·A[L-1]+b[L] → A[L]=sigmoid(Z[L])] → Loss
```

- **Layers 1 to L-1 (hidden):** tanh activation
- **Layer L (output):** sigmoid activation
- **A[0] = X** — the input is treated as the activation of layer 0

---

## Parameter Initialization

For each layer $l$ from 1 to L:

$$W^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}, \quad b^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$$

Where $n^{[l]}$ is the number of nodes in layer $l$ and $n^{[l-1]}$ is the number of nodes in the previous layer (or number of features for $l=1$).

Weights are initialized with small random values:

```python
W = np.random.randn(n[l], n[l-1]) * 0.01
b = np.zeros((n[l], 1))
```

**Why `* 0.01`?** If weights are large, $Z = WA + b$ will be large, pushing tanh and sigmoid into their flat saturated regions where the derivative is near zero. Training becomes extremely slow or stops entirely. Small weights keep Z near zero at initialization, where gradients are largest.

**Why not initialize to zero?** Zero initialization makes every neuron in a layer identical — they compute the same output, receive the same gradients, and update identically forever. The neurons never differentiate. Random initialization breaks this symmetry.

---

## Forward Propagation

The same two equations repeat for every layer:

**Step 1 — Linear combination:**

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$

**Step 2 — Activation:**

$$A^{[l]} = \begin{cases} \tanh(Z^{[l]}) & \text{if } l < L \quad \text{(hidden layers)} \\ \sigma(Z^{[l]}) & \text{if } l = L \quad \text{(output layer)} \end{cases}$$

Starting from $A^{[0]} = X$, this loop runs from $l = 1$ to $l = L$.

**Loss (binary cross-entropy):**

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(A_L^{(i)}) + (1 - y^{(i)}) \log(1 - A_L^{(i)}) \right]$$

---

## Backward Propagation

The goal is to compute $dW^{[l]}$ and $db^{[l]}$ for every layer so we can update the weights. This requires computing $dZ^{[l]}$ for each layer, starting from the output and working backward.

### Output Layer (l = L)

We want $dZ^{[L]} = \frac{\partial \mathcal{L}}{\partial Z^{[L]}}$. Applying the chain rule:

$$dZ^{[L]} = \frac{\partial \mathcal{L}}{\partial Z^{[L]}} = \frac{\partial \mathcal{L}}{\partial A^{[L]}} \cdot \frac{\partial A^{[L]}}{\partial Z^{[L]}}$$

**Part 1 — $\frac{\partial \mathcal{L}}{\partial A^{[L]}}$:**

$A^{[L]}$ appears directly in the loss, so we differentiate:

$$\frac{\partial \mathcal{L}}{\partial A^{[L]}} = -\frac{Y}{A^{[L]}} + \frac{1 - Y}{1 - A^{[L]}}$$

**Part 2 — $\frac{\partial A^{[L]}}{\partial Z^{[L]}}$:**

The output layer uses sigmoid:

$$\frac{\partial A^{[L]}}{\partial Z^{[L]}} = \sigma'(Z^{[L]}) = A^{[L]}(1 - A^{[L]})$$

**Combining both parts:**

$$dZ^{[L]} = \left(-\frac{Y}{A^{[L]}} + \frac{1-Y}{1-A^{[L]}}\right) \cdot A^{[L]}(1 - A^{[L]})$$

Expanding and simplifying:

$$dZ^{[L]} = A^{[L]} - Y$$

This simplification is specific to sigmoid + binary cross-entropy. It is a derived result, not a shortcut we invented.

### Hidden Layers (l = L-1 down to 1)

For hidden layers, $dZ^{[l]}$ cannot be computed from the loss directly. We go through the chain:

$$dZ^{[l]} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}} = \frac{\partial \mathcal{L}}{\partial A^{[l]}} \cdot \frac{\partial A^{[l]}}{\partial Z^{[l]}}$$

**Part 1 — $\frac{\partial \mathcal{L}}{\partial A^{[l]}}$:**

$A^{[l]}$ is not in the loss directly. It fed into $Z^{[l+1]}$ through $W^{[l+1]}$. So we trace the path: $A^{[l]} \rightarrow Z^{[l+1]} \rightarrow \mathcal{L}$:

$$\frac{\partial \mathcal{L}}{\partial A^{[l]}} = \frac{\partial \mathcal{L}}{\partial Z^{[l+1]}} \cdot \frac{\partial Z^{[l+1]}}{\partial A^{[l]}}$$

From the forward pass $Z^{[l+1]} = W^{[l+1]} A^{[l]} + b^{[l+1]}$, so $\frac{\partial Z^{[l+1]}}{\partial A^{[l]}} = W^{[l+1]}$.

And $\frac{\partial \mathcal{L}}{\partial Z^{[l+1]}} = dZ^{[l+1]}$ (already computed in the previous backward step).

Putting it together (transposed for correct dimensions):

$$\frac{\partial \mathcal{L}}{\partial A^{[l]}} = W^{[l+1]T} \cdot dZ^{[l+1]}$$

**Part 2 — $\frac{\partial A^{[l]}}{\partial Z^{[l]}}$:**

Hidden layers use tanh:

$$\frac{\partial A^{[l]}}{\partial Z^{[l]}} = \tanh'(Z^{[l]}) = 1 - \tanh^2(Z^{[l]}) = 1 - (A^{[l]})^2$$

**Combining both parts:**

$$dZ^{[l]} = W^{[l+1]T} \cdot dZ^{[l+1]} \times (1 - (A^{[l]})^2)$$

This is the general backward step for any hidden layer — the error signal from the layer ahead, scaled by the local activation derivative.

### Gradients for Weights and Biases (all layers)

Once $dZ^{[l]}$ is known, $dW^{[l]}$ and $db^{[l]}$ follow the same formula for every layer:

$$dW^{[l]} = \frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}} \cdot \frac{\partial Z^{[l]}}{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} \cdot A^{[l-1]T}$$

$$db^{[l]} = \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}} \cdot \frac{\partial Z^{[l]}}{\partial b^{[l]}} = \frac{1}{m} \sum dZ^{[l]}$$

From $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$: $\frac{\partial Z^{[l]}}{\partial W^{[l]}} = A^{[l-1]}$ and $\frac{\partial Z^{[l]}}{\partial b^{[l]}} = 1$.

We divide by $m$ to average the gradient across all training examples.

---

## The Vanishing Gradient Problem

During backprop, gradients flow backward through every layer by being multiplied by the activation derivative at each step.

For tanh, the derivative is $1 - A^2$. When $|A|$ is close to 1 (saturated region), this derivative approaches 0. In a deep network, the gradient at layer $l$ is proportional to the product of all activation derivatives from layer $L$ down to $l+1$:

$$\frac{\partial \mathcal{L}}{\partial Z^{[l]}} \propto \prod_{l+1}^{L} (1 - (A^{[l]})^2)$$

If each term is small (say 0.1), and there are 5 layers, the gradient reaching layer 1 is $0.1^4 = 0.0001$. The first layers receive almost no learning signal and stop updating.

This is why the deep network in this repo performs poorly with a small learning rate — the gradient vanishes before reaching the early layers.

---

## Code Structure

```
sigmoid(z)           → sigmoid activation
initialize_params()  → random init with * 0.01 for each layer
propagate()          → full forward pass + full backward pass for L layers
optimize()           → gradient descent loop
predict()            → threshold A[L] > 0.5
model()              → initializes all layers, runs optimization
evaluate()           → accuracy, precision, recall, F1, confusion matrix
```

---

## Dataset

**Cat vs Non-Cat** — same dataset used in Andrew Ng's Deep Learning Specialization.

- 209 training examples, 50 test examples
- 64×64 RGB images → flattened to 12,288 features per image
- Binary labels: cat (1) vs non-cat (0)

---

## Results

Trained with `learning_rate=0.005, epochs=2000`, architecture `{1:8, 2:4, 3:1}`:

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Train | 65.55% | 0.00 | 0.00 | 0.00 |
| Test | 34.00% | 0.00 | 0.00 | 0.00 |

The network predicts everything as 0 — it never learns. The vanishing gradient problem prevents the learning signal from reaching the early layers with this learning rate.

---

## Limitations

### Hyperparameter Sensitivity

Deeper networks are harder to train. The same learning rate that works for logistic regression causes the deep network to either not learn at all (too small) or oscillate (too large).

### No Spatial Awareness

This limitation carries over from logistic regression and the shallow network. Fully connected layers treat every pixel as an independent feature — spatial structure is destroyed at the flattening step and cannot be recovered by any number of FC layers.

The solution is Convolutional Neural Networks.

---

**Previous:** [Shallow Neural Network ←](../02_shallow_neural_network/)