# Shallow Neural Network from Scratch

A two-layer neural network built using only NumPy — no frameworks. Applied to the Cat vs Non-Cat dataset.

---

## Motivation

Logistic regression draws a single linear decision boundary. For problems where classes are not linearly separable, this is a fundamental limitation — no amount of training will fix it.

Adding a hidden layer with a non-linear activation function allows the network to learn curved, complex decision boundaries. This is the key step from logistic regression to a neural network.

---

## Architecture

```
X → Z1 = W1·X + b1 → A1 = tanh(Z1) → Z2 = W2·A1 + b2 → A2 = sigmoid(Z2) → Loss
```

- **Layer 1 (hidden):** tanh activation — learns non-linear representations of the input
- **Layer 2 (output):** sigmoid activation — outputs a probability between 0 and 1

---

## Forward Propagation

**Hidden layer:**

$$Z_1 = W_1 X + b_1$$

$$A_1 = \tanh(Z_1) = \frac{e^{Z_1} - e^{-Z_1}}{e^{Z_1} + e^{-Z_1}}$$

**Output layer:**

$$Z_2 = W_2 A_1 + b_2$$

$$A_2 = \sigma(Z_2) = \frac{1}{1 + e^{-Z_2}}$$

**Loss (binary cross-entropy):**

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(a_2^{(i)}) + (1 - y^{(i)}) \log(1 - a_2^{(i)}) \right]$$

---

## Backward Propagation

Gradients are computed layer by layer using the chain rule, starting from the output and flowing backward.

### Output Layer

We want $dZ_2 = \frac{\partial \mathcal{L}}{\partial Z_2}$. $Z_2$ is not directly in the loss formula, so we apply the chain rule:

$$dZ_2 = \frac{\partial \mathcal{L}}{\partial Z_2} = \frac{\partial \mathcal{L}}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2}$$

**Part 1 — $\frac{\partial \mathcal{L}}{\partial A_2}$ (derivative of loss w.r.t $A_2$):**

$A_2$ appears directly in the loss formula, so we differentiate:

$$\frac{\partial \mathcal{L}}{\partial A_2} = -\frac{Y}{A_2} + \frac{1 - Y}{1 - A_2}$$

**Part 2 — $\frac{\partial A_2}{\partial Z_2}$ (derivative of sigmoid w.r.t $Z_2$):**

$$\frac{\partial A_2}{\partial Z_2} = \sigma'(Z_2) = A_2(1 - A_2)$$

**Combining both parts:**

$$dZ_2 = \left(-\frac{Y}{A_2} + \frac{1-Y}{1-A_2}\right) \cdot A_2(1 - A_2)$$

Expanding and simplifying:

$$dZ_2 = A_2 - Y$$

This simplification only works for sigmoid + binary cross-entropy together. It is a derived result, not a definition.

$$dW_2 = \frac{\partial \mathcal{L}}{\partial W_2} = \frac{1}{m} dZ_2 \cdot A_1^T$$

$$db_2 = \frac{\partial \mathcal{L}}{\partial b_2} = \frac{1}{m} \sum dZ_2$$

### Hidden Layer

For the hidden layer, $dZ_1$ cannot be computed directly from the loss. We apply the chain rule:

$$dZ_1 = \frac{\partial \mathcal{L}}{\partial Z_1} = \frac{\partial \mathcal{L}}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1}$$

**Part 1 — $\frac{\partial \mathcal{L}}{\partial A_1}$ (how much does loss change when $A_1$ changes):**

$A_1$ is not directly in the loss formula. To find how the loss changes with $A_1$, we go through the chain — $A_1$ affects $Z_2$, and $Z_2$ affects the loss:

$$\frac{\partial \mathcal{L}}{\partial A_1} = \frac{\partial \mathcal{L}}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial A_1}$$

From the forward pass: $Z_2 = W_2 A_1 + b_2$, so:

$$\frac{\partial Z_2}{\partial A_1} = W_2$$

And $\frac{\partial \mathcal{L}}{\partial Z_2} = dZ_2$ (already computed in the output layer step).

Putting it together (transposed for correct dimensions):

$$\frac{\partial \mathcal{L}}{\partial A_1} = W_2^T \cdot dZ_2$$

This is the general pattern: the error signal at any hidden layer comes from the weights and the gradient of the layer directly ahead of it.

**Part 2 — $\frac{\partial A_1}{\partial Z_1}$ (derivative of the activation function used in this layer):**

This layer used tanh, so:

$$\frac{\partial A_1}{\partial Z_1} = \tanh'(Z_1) = 1 - \tanh^2(Z_1) = 1 - A_1^2$$

**Combining both parts:**

$$dZ_1 = \frac{\partial \mathcal{L}}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} = W_2^T \cdot dZ_2 \times (1 - A_1^2)$$

This is the general pattern for any hidden layer — the error signal from the next layer multiplied by the derivative of the activation function used in the current layer. The activation derivative changes depending on what activation was used (tanh, ReLU, sigmoid), but the structure is always the same.

$$dW_1 = \frac{\partial \mathcal{L}}{\partial W_1} = \frac{1}{m} dZ_1 \cdot X^T$$

$$db_1 = \frac{\partial \mathcal{L}}{\partial b_1} = \frac{1}{m} \sum dZ_1$$

---

## Why tanh for the Hidden Layer?

Both sigmoid and tanh are valid activation functions, but tanh is preferred for hidden layers for one key reason: **zero-centering**.

| Property | Sigmoid | tanh |
|---|---|---|
| Output range | (0, 1) | (-1, 1) |
| Zero-centered | No | Yes |
| Saturation | Both ends | Both ends |

Sigmoid always outputs positive values. When all activations are positive, the gradients during backprop are always the same sign — causing zig-zag updates in weight space, slowing convergence.

tanh outputs values between -1 and 1, centered around zero. This means gradients can be both positive and negative, allowing more direct paths to the minimum.

---

## Why Weights Are Initialized with `* 0.01`

Weights are initialized as small random values:

```python
W1 = np.random.randn(n_hidden, n_features) * 0.01
```

If weights are large at initialization, $Z = WX + b$ will be large, pushing tanh and sigmoid into their **saturated regions** (flat parts of the curve). In these regions, the derivative is near zero — gradients vanish and learning slows dramatically or stops.

Small weights keep $Z$ near zero at the start, where both tanh and sigmoid have the steepest gradients and learn fastest.

---

## Code Structure

```
sigmoid(z)               → sigmoid activation
initialize_parameters()  → random init with * 0.01
propagate()              → forward pass + backward pass
optimize()               → gradient descent loop
predict()                → threshold A2 > 0.5
model()                  → ties everything together
evaluate()               → accuracy, precision, recall, F1, confusion matrix
```

---

## Dataset

**Cat vs Non-Cat** — same dataset used in Andrew Ng's Deep Learning Specialization.

- 209 training examples, 50 test examples
- 64×64 RGB images → flattened to 12,288 features per image
- Binary labels: cat (1) vs non-cat (0)

---

## Results

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Train | 96.17% | 0.9444 | 0.9444 | 0.9444 |
| Test | 68.00% | 0.8148 | 0.6667 | 0.7333 |

**Confusion Matrix (Test):**
```
                 Predicted
                 Non-Cat  Cat
Actual Non-Cat  [  12        5  ]
Actual Cat      [  11       22  ]
```

Test accuracy dropped slightly from logistic regression (70% → 68%). This can happen with a small dataset — the hidden layer adds parameters that need more data to generalize well.

---

## Limitations

### Fully Connected — No Spatial Awareness

The hidden layer learns non-linear combinations of pixel values, but it still treats every pixel as an independent feature. There is no concept of proximity — pixels that are spatially close in the image are not treated any differently from pixels that are far apart.

The same cat appearing in different positions of the image produces entirely different activation patterns. The network has no mechanism to recognize positional invariance.

Adding more hidden layers increases the depth and representational power of the network, but does not solve this problem. Spatial structure, once destroyed by flattening, cannot be recovered by any number of fully connected layers.

The correct solution is Convolutional Neural Networks — which preserve and exploit spatial structure through learned filters.

---

**Previous:** [Logistic Regression ←](../01_logistic_regression/)  
**Next:** [Deep Neural Network →](../03_deep_neural_network/)
