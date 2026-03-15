# L2 Regularization

---

## The Problem: Why Overfitting Happens

A neural network learns by adjusting its weights to minimize the training loss. With enough capacity and enough iterations, it can reduce training loss to near zero — not by learning the true pattern, but by memorizing every training example individually.

The tell-tale sign of memorization is **large weights**. When a weight is very large, a tiny change in one input feature causes a huge swing in the output. The network becomes hypersensitive — it fits the exact quirks of the training set but falls apart on anything new.

The fix is to make large weights expensive. If the cost function punishes large weights, the optimizer is forced to find a solution that is both accurate *and* small — which tends to be a smoother, more generalizable function.

---

## Step 1 — The Original Loss

Start with the standard binary cross-entropy loss for $m$ training examples:

$$J = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

Where:
- $y^{(i)} \in \{0, 1\}$ is the true label for example $i$
- $\hat{y}^{(i)} = A^{[L](i)}$ is the network's predicted probability
- The sum averages the loss over all examples so that $J$ does not grow with dataset size

This is the only thing the optimizer currently cares about. It will make weights as large as needed to drive this number down — and that is the problem.

---

## Step 2 — Building a Penalty for Large Weights

We want a term that is small when weights are small, and large when weights are large.

**Start with the simplest idea:** sum the weights.

$$\text{penalty} = W_1 + W_2 + \dots + W_n$$

**Problem:** negative weights cancel positive weights. A weight of $-1000$ and a weight of $+1000$ would sum to zero, fooling the penalty into thinking the model is well-behaved.

**Fix:** square each weight before summing.

$$\text{penalty} = W_1^2 + W_2^2 + \dots + W_n^2$$

Squaring does two things: it removes the sign (large negative weights are just as costly as large positive ones), and it penalizes large weights disproportionately — a weight of 10 costs 100 times more than a weight of 1.

This is the **L2 norm squared**, also called the **squared Euclidean norm**:

$$\text{penalty} = \|\mathbf{w}\|_2^2 = \sum_{k=1}^{n} W_k^2$$

---

## Step 3 — Scaling the Penalty

If we add the raw penalty to the loss, it will have a completely different magnitude and mess up the balance between the two terms. We need a scaling factor.

The chosen form is:

$$\text{scaled penalty} = \frac{\lambda}{2m} \sum_{k=1}^{n} W_k^2$$

Each piece of this scaling factor has a specific reason:

**Why $\lambda$?**
$\lambda$ is a hyperparameter that controls how much we care about regularization vs. fitting the data. A larger $\lambda$ means more weight shrinkage, less overfitting, but potentially underfitting if too large. It sits in the numerator because we want a larger $\lambda$ to mean a larger penalty.

**Why divide by $m$?**
The loss averages over all examples so that it does not grow with dataset size. If we don't divide the penalty by $m$ as well, then the penalty gets larger on a large dataset. That's why we divide by $m$ — so the penalty is independent of dataset size. Dividing the penalty by $m$ keeps both terms on the same scale, so $\lambda$ can be tuned independently of dataset size.

**Why divide by 2?**
When we differentiate $W^2$, we get $2W$. The $\frac{1}{2}$ preemptively cancels this factor:

$$\frac{\partial}{\partial W} \left( \frac{1}{2} W^2 \right) = \frac{1}{2} \cdot 2W = W$$

This keeps the gradient clean — no stray factor of 2 propagated through every layer during backprop. It has no effect on the optimization itself.

---

## Step 4 — Extending the Penalty to a Neural Network

**Goal:** generalize the flat sum $\sum_k W_k^2$ from Step 2 to the full weight structure of an $L$-layer network.

In a neural network, weights are not a flat list. Layer $l$ has a weight matrix $W^{[l]}$ of shape $(n^{[l]} \times n^{[l-1]})$, where $n^{[l]}$ is the number of neurons in layer $l$.

**Sub-step 4.1 — Penalty for a single element of one layer:**

Pick any single weight $W_{jk}^{[l]}$ — row $j$, column $k$, layer $l$. Its contribution to the penalty is:

$$\left(W_{jk}^{[l]}\right)^2$$

**Sub-step 4.2 — Sum over all elements of one layer:**

To cover every element in $W^{[l]}$, run two nested sums — one over rows $j$, one over columns $k$:

$$\sum_{j=1}^{n^{[l]}} \sum_{k=1}^{n^{[l-1]}} \left(W_{jk}^{[l]}\right)^2$$

This quantity has a standard name: the **Frobenius norm squared**, written $\|W^{[l]}\|_F^2$.

$$\|W^{[l]}\|_F^2 \;=\; \sum_{j=1}^{n^{[l]}} \sum_{k=1}^{n^{[l-1]}} \left(W_{jk}^{[l]}\right)^2$$

**Sub-step 4.3 — Sum over all $L$ layers:**

Repeat Sub-step 4.2 for every layer and add the results:

$$\sum_{l=1}^{L} \|W^{[l]}\|_F^2 = \sum_{l=1}^{L} \sum_{j=1}^{n^{[l]}} \sum_{k=1}^{n^{[l-1]}} \left(W_{jk}^{[l]}\right)^2$$

Apply the $\frac{\lambda}{2m}$ scaling from Step 3:

$$\boxed{\text{penalty} = \frac{\lambda}{2m} \sum_{l=1}^{L} \|W^{[l]}\|_F^2}$$

---

## Step 5 — The Regularized Cost Function

**Goal:** combine the original loss (Step 1) and the penalty (Step 4) into a single objective.

Take the cross-entropy loss $J$ from Step 1 and add the penalty from Step 4 directly:

$$J_{\text{original}} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

$$+ \quad \frac{\lambda}{2m} \sum_{l=1}^{L} \|W^{[l]}\|_F^2$$

$$\boxed{J_{\text{L2}} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2m} \sum_{l=1}^{L} \|W^{[l]}\|_F^2}$$

The optimizer now minimizes both terms together. A solution with large weights pays a higher cost even if the prediction error is low — it must find a balance between accuracy and small weights. $\lambda$ controls how strictly that balance is enforced.

---

## Step 6 — Backpropagation with L2 Regularization

**Goal:** derive how gradients change when the regularized cost $J_{\text{L2}}$ is used instead of $J_{\text{original}}$.

### 6.1 — Split the derivative using the sum rule

We need $\frac{\partial J_{\text{L2}}}{\partial W^{[l]}}$. Since $J_{\text{L2}}$ is a sum of two terms, the derivative splits:

$$\frac{\partial J_{\text{L2}}}{\partial W^{[l]}} = \underbrace{\frac{\partial J_{\text{original}}}{\partial W^{[l]}}}_{\text{Term A}} + \underbrace{\frac{\partial}{\partial W^{[l]}} \left[ \frac{\lambda}{2m} \sum_{r=1}^{L} \|W^{[r]}\|_F^2 \right]}_{\text{Term B}}$$

We derive Term A and Term B separately, then add them.

---

### 6.2 — Derive Term A (gradient from the original loss)

The forward pass for layer $l$ is:

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$

Applying the chain rule:

$$\frac{\partial J_{\text{original}}}{\partial W^{[l]}} = \frac{\partial J_{\text{original}}}{\partial Z^{[l]}} \cdot \frac{\partial Z^{[l]}}{\partial W^{[l]}}$$

The first factor is the upstream gradient $dZ^{[l]}$, already computed during backprop. The second factor is $A^{[l-1]}$. Averaged over $m$ examples:

$$\text{Term A} = \frac{1}{m}\, dZ^{[l]} \cdot {A^{[l-1]}}^T \quad \text{(transposed for correct dimensions)}$$

---

### 6.3 — Derive Term B (gradient from the penalty)

Start simple. Imagine the penalty is just a flat list of weights:

$$\frac{\lambda}{2m}\left(W_1^2 + W_2^2 + W_3^2 + \dots + W_n^2\right)$$

Now take the derivative with respect to $W_1$:

$$\frac{\partial}{\partial W_1}\left[\frac{\lambda}{2m}\left(W_1^2 + W_2^2 + W_3^2 + \dots + W_n^2\right)\right]$$

$W_2^2,\ W_3^2,\ \dots,\ W_n^2$ do not contain $W_1$ — they are constants. Constants vanish under differentiation. Only $W_1^2$ survives:

$$= \frac{\lambda}{2m} \cdot \frac{\partial}{\partial W_1}\, W_1^2 = \frac{\lambda}{2m} \cdot 2W_1 = \frac{\lambda}{m} W_1$$

The same logic holds for any $W_k$ — differentiate w.r.t. $W_k$, every other term is a constant and drops out, leaving:

$$\frac{\partial}{\partial W_k}\left[\frac{\lambda}{2m}\sum_{j=1}^n W_j^2\right] = \frac{\lambda}{m} W_k$$

In a neural network the weights are a matrix, not a flat list — but the idea is identical. Each element $W_{jk}^{[l]}$ appears in the sum exactly once. Differentiating w.r.t. it kills every other term, and we get:

$$\text{Term B} = \frac{\lambda}{m} W^{[l]}$$

---

### 6.4 — Combine: the new $dW^{[l]}$

Add Term A and Term B:

$$\boxed{dW^{[l]}_{\text{L2}} = \underbrace{\frac{1}{m}\, dZ^{[l]} \cdot {A^{[l-1]}}^T}_{\text{Term A: original gradient}} + \underbrace{\frac{\lambda}{m} W^{[l]}}_{\text{Term B: penalty gradient}}}$$

Everything else in backprop is unchanged. Only $dW^{[l]}$ gains an extra term.

---

### 6.5 — Why $db^{[l]}$ is unaffected

The penalty $\frac{\lambda}{2m} \sum_r \|W^{[r]}\|_F^2$ contains no bias variables at all. Differentiating it with respect to $b^{[l]}$ gives zero, so Term B for biases is zero. Therefore:

$$db^{[l]}_{\text{L2}} = db^{[l]}_{\text{original}} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l](i)}$$

Biases shift the output but do not scale inputs — they do not make the network hypersensitive in the same way large weights do. There is no reason to penalize them.

---

## Step 7 — The Update Rule (Weight Decay)

**Goal:** substitute the new $dW^{[l]}_{\text{L2}}$ into gradient descent and simplify to its final form.

**7.1 — Start from the standard gradient descent update:**

$$W^{[l]} \leftarrow W^{[l]} - \alpha \cdot dW^{[l]}_{\text{L2}}$$

**7.2 — Substitute $dW^{[l]}_{\text{L2}}$ from Step 6.4:**

$$W^{[l]} \leftarrow W^{[l]} - \alpha \left( \frac{1}{m}\, dZ^{[l]} \cdot {A^{[l-1]}}^T + \frac{\lambda}{m} W^{[l]} \right)$$

**7.3 — Distribute $\alpha$ across both terms:**

$$W^{[l]} \leftarrow W^{[l]} - \frac{\alpha}{m}\, dZ^{[l]} \cdot {A^{[l-1]}}^T - \frac{\alpha\lambda}{m} W^{[l]}$$

**7.4 — Group the $W^{[l]}$ terms on the right:**

$$W^{[l]} \leftarrow W^{[l]} - \frac{\alpha\lambda}{m} W^{[l]} - \frac{\alpha}{m}\, dZ^{[l]} \cdot {A^{[l-1]}}^T$$

$$W^{[l]} \leftarrow W^{[l]} \!\left(1 - \frac{\alpha\lambda}{m}\right) - \frac{\alpha}{m}\, dZ^{[l]} \cdot {A^{[l-1]}}^T$$

**7.5 — Final form:**

$$\boxed{W^{[l]} \leftarrow \underbrace{\left(1 - \frac{\alpha\lambda}{m}\right)}_{\text{shrink factor}} W^{[l]} - \frac{\alpha}{m}\, dZ^{[l]} \cdot {A^{[l-1]}}^T}$$

**Reading the result:**

Every update now has two sequential effects on $W^{[l]}$:

1. **Shrink** — multiply $W^{[l]}$ by $\left(1 - \frac{\alpha\lambda}{m}\right)$, a number slightly less than 1. This pulls every weight toward zero unconditionally, on every single iteration.
2. **Gradient step** — subtract the gradient term, exactly as in unregularized training.

This is called **weight decay**. The engineers who discovered this trick wrote it as:

```python
W *= 0.99               # shrink
W -= learning_rate * dW  # gradient step
```

The statisticians who derived L2 regularization from the loss function arrived at the same formula through calculus. They are identical.

---

## Summary

| Step | What was done | Result |
|---|---|---|
| 1 | Define the original loss | $J = -\frac{1}{m}\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| 2 | Build a penalty for large weights | $\text{penalty} = \sum_k W_k^2$ |
| 3 | Scale the penalty | $\frac{\lambda}{2m}\sum_k W_k^2$ |
| 4 | Extend to matrices (Frobenius norm) | $\frac{\lambda}{2m}\sum_l \|W^{[l]}\|_F^2$ |
| 5 | Form the regularized cost | $J_{\text{L2}} = J + \frac{\lambda}{2m}\sum_l \|W^{[l]}\|_F^2$ |
| 6 | Differentiate: new $dW^{[l]}$ | $dW^{[l]}_{\text{L2}} = \frac{1}{m}dZ^{[l]}{A^{[l-1]}}^T + \frac{\lambda}{m}W^{[l]}$ |
| 7 | Substitute into update rule | $W^{[l]} \leftarrow (1 - \frac{\alpha\lambda}{m})W^{[l]} - \frac{\alpha}{m}dZ^{[l]}{A^{[l-1]}}^T$ |
