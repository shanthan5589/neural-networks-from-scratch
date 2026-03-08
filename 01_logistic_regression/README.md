# Logistic Regression from Scratch

A binary classifier built using only NumPy — no frameworks. Applied to the Cat vs Non-Cat dataset.

---

## What is Logistic Regression?

Logistic regression is the simplest neural network possible — a single neuron. It takes an input, computes a weighted sum, passes it through a sigmoid activation, and outputs a probability between 0 and 1.

It answers one question: **given this input, what is the probability it belongs to class 1?**

---

## Architecture

```
X (features)  →  Z = W·X + b  →  A = sigmoid(Z)  →  Loss
```

- **X** — input features, shape `(n_features, m)` where m = number of training examples
- **W** — weights, shape `(n_features, 1)`
- **b** — bias, scalar
- **A** — predicted probability, shape `(1, m)`
- **Y** — true labels, shape `(1, m)`

---

## Forward Propagation

**Step 1 — Linear combination:**

$$Z = W^T X + b$$

Each feature is multiplied by its weight, summed up, and a bias is added. This gives a raw score.

**Step 2 — Sigmoid activation:**

$$A = \sigma(Z) = \frac{1}{1 + e^{-Z}}$$

Sigmoid squashes the raw score into a probability between 0 and 1.

- If Z is very large → A ≈ 1 (confident it's class 1)
- If Z is very small → A ≈ 0 (confident it's class 0)
- If Z = 0 → A = 0.5 (completely uncertain)

---

## Loss Function

We use **binary cross-entropy loss:**

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)}) \right]$$

**Why not mean squared error?**

MSE with sigmoid creates a non-convex loss surface with many local minima — gradient descent gets stuck. Cross-entropy with sigmoid creates a convex surface — gradient descent always finds the global minimum.

**Intuition behind the formula:**
- When y=1: loss = -log(A). If A→1 (correct), loss→0. If A→0 (wrong), loss→∞
- When y=0: loss = -log(1-A). If A→0 (correct), loss→0. If A→1 (wrong), loss→∞

The network is heavily penalized for confident wrong predictions.

---

## Backward Propagation

The goal: find how much each weight contributes to the loss, so we can reduce it.

We use the **chain rule** to compute how the loss changes with respect to each parameter.

**dZ — derivative of loss w.r.t Z:**

$$\frac{\partial \mathcal{L}}{\partial Z} = \frac{\partial \mathcal{L}}{\partial A} \cdot \frac{\partial A}{\partial Z}$$

Breaking it down:

$$\frac{\partial \mathcal{L}}{\partial A} = -\frac{Y}{A} + \frac{1-Y}{1-A}$$

$$\frac{\partial A}{\partial Z} = A(1-A) \quad \text{(sigmoid derivative)}$$

Multiplying these together simplifies cleanly to:

$$dZ = A - Y$$

This is the key shortcut — sigmoid + cross-entropy always gives `A - Y` for the output layer.

**dW — derivative of loss w.r.t W:**

$$dW = \frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial Z} \cdot \frac{\partial Z}{\partial W}$$

From $Z = W^T X + b$, we get $\frac{\partial Z}{\partial W} = X$. So:

$$dW = \frac{\partial \mathcal{L}}{\partial W} = \frac{1}{m} X \cdot dZ^T$$

**db — derivative of loss w.r.t b:**

$$db = \frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial Z} \cdot \frac{\partial Z}{\partial b}$$

From $Z = W^T X + b$, we get $\frac{\partial Z}{\partial b} = 1$. So:

$$db = \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m} \sum dZ$$

We divide by m to average the gradient across all training examples — without this, the gradient magnitude would scale with dataset size.

---

## Gradient Descent

Once we have the gradients, we update the weights:

$$W := W - \alpha \cdot dW$$
$$b := b - \alpha \cdot db$$

Where **α** (alpha) is the learning rate — how big a step we take toward the minimum.

- Too large → overshoots the minimum, loss diverges
- Too small → takes forever to converge
- Just right → loss decreases smoothly to minimum

---

## Code Structure

```
sigmoid(z)              → applies sigmoid activation
initialize_with_zeros() → initializes W=0, b=0
propagate()             → forward pass + compute gradients
optimize()              → gradient descent loop
predict()               → threshold A > 0.5 to get 0/1 predictions
model()                 → ties everything together
evaluate()              → prints accuracy, precision, recall, F1, confusion matrix
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
| Train | 99.04% | 0.9861 | 0.9861 | 0.9861 |
| Test | 70.00% | 0.8214 | 0.6970 | 0.7541 |

**Classic overfitting** — 99% train vs 70% test. The model memorizes the 209 training examples perfectly but doesn't generalize.

**Confusion Matrix (Test):**
```
                 Predicted
                 Non-Cat  Cat
Actual Non-Cat  [  12        5  ]
Actual Cat      [  10       23  ]
```

---

## Limitations

### 1. Linear Decision Boundary

Logistic regression can only separate classes with a single hyperplane in feature space. Any problem where the decision boundary is curved, circular, or otherwise non-linear cannot be solved by logistic regression regardless of how long it trains.

The solution is hidden layers — a neural network with non-linear activations can approximate arbitrarily complex decision boundaries.

### 2. No Spatial Awareness

Flattening a 64×64 image into a 12,288-length vector destroys all spatial structure. Pixel (0,0) and pixel (63,63) are treated as completely independent features with no notion of proximity. The same object appearing in different positions of the image produces entirely different input vectors — the model has no mechanism to recognize them as the same pattern.

This problem is **not solved by adding more fully connected layers**. A deep FC network still processes each pixel independently — more layers add representational power but cannot recover spatial structure that was lost during flattening.

The correct solution is Convolutional Neural Networks (CNNs), which apply learned filters across the entire image — detecting patterns regardless of their position.

---

**Next:** [Shallow Neural Network →](../02_shallow_neural_network/)

---

## Key Takeaways

1. Logistic regression = single neuron with sigmoid activation
2. `dZ = A - Y` is a shortcut from the chain rule for sigmoid + cross-entropy
3. The loss surface is convex — gradient descent always converges
4. Dividing gradients by m ensures learning rate is independent of dataset size
5. Linear models cannot learn the complex patterns in image data
