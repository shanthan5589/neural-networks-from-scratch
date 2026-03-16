# Dropout

## Core Idea

During training, randomly zero out neurons with probability $p$ so the network cannot rely on any specific neuron always being present. This forces distributed, redundant representations — each neuron must be independently useful. Only robust solutions survive training; fragile co-adapted solutions don't.

At test time, no neurons are dropped.

---

## Forward Pass (Inverted Dropout)

For layer $l$, generate a binary mask:

$$R^{[l]} \sim \text{Bernoulli}(1 - p)$$

Apply and scale:

$$\tilde{A}^{[l]} = \frac{R^{[l]} \odot A^{[l]}}{1 - p}$$

The $\frac{1}{1-p}$ scaling keeps the expected activation the same as test time — so no adjustment is needed at inference.

```python
# Training
mask = (np.random.rand(*A.shape) > p)
A_dropped = (A * mask) / (1 - p)

# Test time — use A directly, no changes
```

---

## Backward Pass

Apply the same mask from the forward pass — gradients flow only through neurons that were kept.

---

## Key Notes

- Apply to hidden layers only, not input or output
- Standard keep probability: 0.5 for hidden layers
- Training loss will be noisy — expected. Watch test loss, not training loss
- Dropout only during training, not at test/inference time
